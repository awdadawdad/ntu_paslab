from dataclasses import dataclass
from pathlib import Path
from statistics import mean
import argparse
import json
import os
import time
import termcolor

from transformers import AutoTokenizer
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
import torch

# Environment variables set by torch.distributed.launch
LOCAL_WORLD_SIZE = int(os.environ["LOCAL_WORLD_SIZE"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
GROUP_RANK = int(os.environ["GROUP_RANK"])
WORLD_RANK = int(os.environ["RANK"])

DEFAULT_SEED = 7


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = x.shape
    x = x[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return x.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def get_json(file_path: Path) -> dict:
    with open(file_path, "r") as f:
        return json.load(f)


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    # assert 0 <= p <= 1

    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    return torch.gather(probs_idx, -1, next_token)


@dataclass
class ModelArgs:
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    norm_eps: float
    vocab_size: int
    rope_theta: float
    attn_tp: bool = False
    device: torch.device = None

    @classmethod
    def from_hf_config(cls, params: dict):
        return cls(
            dim=params["hidden_size"],
            n_layers=params["num_hidden_layers"],
            head_dim=params["hidden_size"] // params["num_attention_heads"],
            hidden_dim=params["intermediate_size"],
            n_heads=params["num_attention_heads"],
            n_kv_heads=params["num_key_value_heads"],
            norm_eps=params["rms_norm_eps"],
            vocab_size=params["vocab_size"],
            rope_theta=params["rope_theta"],
        )

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=8192, base=1000000.0, device=None):
        super().__init__()
        self.dim = dim
        self.max_seqlen = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.float, device=device)
                / self.dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, device, dtype):
        t = torch.arange(self.max_seqlen, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, dtype):
        return (
            self.cos_cached[: self.max_seqlen].to(dtype=dtype),
            self.sin_cached[: self.max_seqlen].to(dtype=dtype),
        )

class Attention(nn.Module):
    def __init__(self, args: ModelArgs, li: int):
        super().__init__()
        self.args = args
        self.li = li
        self.cache: torch.Tensor
        self.mask: torch.Tensor
        self.prefill_storage_idx: torch.Tensor
        self.decode_storage_idx: torch.Tensor

        self.n_heads: int = args.n_heads
        self.head_dim: int = args.head_dim
        self.sqrt_head_dim = self.head_dim**0.5
        self.n_kv_heads: int = args.n_kv_heads
        self.repeats = self.n_heads // self.n_kv_heads

        self.q_proj = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=True)
        self.k_proj = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=True)
        self.v_proj = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=True)
        self.o_proj = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)
        self.rotary_emb = RotaryEmbedding(
            args.head_dim, base=args.rope_theta, device=args.device
        )

    def set_batch_level_args(
        self,
        cache: torch.Tensor,
        mask: torch.Tensor,
        prefill_storage_idx: torch.Tensor,
        decode_storage_idx: torch.Tensor,
    ):
        self.cache = cache
        self.mask = mask
        self.prefill_storage_idx = prefill_storage_idx
        self.decode_storage_idx = decode_storage_idx

    def forward(
        self,
        x: torch.Tensor,
        storage_idx: torch.Tensor,
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary_emb(xv.dtype)
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin, storage_idx.unsqueeze(0))

        # assumes bsz matches that of cache
        self.cache[0, self.li].index_copy_(dim=-2, index=storage_idx, source=xk)
        self.cache[1, self.li].index_copy_(dim=-2, index=storage_idx, source=xv)
        keys = self.cache[0, self.li]
        values = self.cache[1, self.li]

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.repeats)  # (bs, max_seq_len, n_heads, head_dim)
        values = repeat_kv(values, self.repeats)  # (bs, max_seq_len, n_heads, head_dim)

        output = F.scaled_dot_product_attention(
            xq,
            keys,
            values,
            attn_mask=self.mask[storage_idx],
            dropout_p=0.0,
            is_causal=False,
        )
        output = output.transpose(1, 2).contiguous().reshape(bsz, seqlen, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.up_proj = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.down_proj = nn.Linear(args.hidden_dim, args.dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, li: int, local_group):
        super().__init__()
        self.local_group = local_group
        self.input_layernorm = RMSNorm(args.dim, eps=args.norm_eps)
        self.self_attn = Attention(args, li)
        self.post_attention_layernorm = RMSNorm(args.dim, eps=args.norm_eps)
        self.mlp = MLP(args=args)

    # NOTATION for code below
    # r: residual connection
    # h: hidden states

    def prefill_attn(self, x: torch.Tensor):
        return self.self_attn(
            self.input_layernorm(x), self.self_attn.prefill_storage_idx
        )

    def decode_attn(self, x: torch.Tensor):
        return self.self_attn(
            self.input_layernorm(x), self.self_attn.decode_storage_idx
        )

    def ffn(self, r: torch.Tensor, h: torch.Tensor):
        # also includes attn communication and residual connection
        dist.all_reduce(h, op=dist.ReduceOp.SUM, group=self.local_group)
        r = r + h
        h = self.mlp(self.post_attention_layernorm(r))
        dist.all_reduce(h, op=dist.ReduceOp.SUM)
        return r + h

    def prefill_forward(self, x: torch.Tensor):
        h = self.prefill_attn(x)
        return self.ffn(x, h)

    def decode_forward(self, x: torch.Tensor):
        h = self.decode_attn(x)
        return self.ffn(x, h)


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs, local_group):
        super().__init__()
        self.args: ModelArgs = args
        self.local_group = local_group
        self.embed_tokens = nn.Embedding(
            args.vocab_size,
            args.dim // LOCAL_WORLD_SIZE if args.attn_tp else args.dim,
        )
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.lm_head = nn.Linear(
            args.dim,
            args.vocab_size // LOCAL_WORLD_SIZE if args.attn_tp else args.vocab_size,
            bias=False,
        )
        self.layers = nn.ModuleDict(
            {
                str(li): TransformerBlock(args, li, local_group)
                for li in range(args.n_layers)
            }
        )
        self.prefill_forward_graph = None
        self.decode_forward_graph = None

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def set_batch_level_args(
        self,
        cache: torch.Tensor,
        mask: torch.Tensor,
        prefill_storage_idx: torch.Tensor,
        decode_storage_idx: torch.Tensor,
    ):
        for li in range(self.args.n_layers):
            self.layers[str(li)].self_attn.set_batch_level_args(
                cache, mask, prefill_storage_idx, decode_storage_idx
            )

    def draw_graphs(self, batch_size: int, prefill_len: int):
        prefill_x = torch.ones(
            (batch_size, prefill_len),
            dtype=torch.int64,
            device=self.device,
        )
        decode_x = torch.ones(
            (batch_size, 1),
            dtype=torch.int64,
            device=self.device,
        )

        with torch.cuda.device(device=self.device):
            callables = torch.cuda.make_graphed_callables(
                (self.prefill_forward, self.decode_forward),
                ((prefill_x,), (decode_x,)),
                num_warmup_iters=16,
            )
            self.prefill_forward_graph, self.decode_forward_graph = callables

    def clear_graph(self):
        self.prefill_forward_graph = None
        self.decode_forward_graph = None

    def all_gather(self, h: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, dim = h.shape
        res = torch.zeros(
            (dim * LOCAL_WORLD_SIZE, seqlen, bsz),
            dtype=h.dtype,
            device=h.device,
        )
        h = h.transpose(0, 2).contiguous()
        dist.all_gather_into_tensor(res, h, group=self.local_group)
        return res.transpose(0, 2)

    def prefill_forward(self, tokens: torch.Tensor):
        h = self.embed_tokens(tokens)
        h = self.all_gather(h)  # .shape = (bsz, seqlen, dim)
        for li in range(self.args.n_layers):
            h = self.layers[str(li)].prefill_forward(h)
        h = self.lm_head(self.norm(h))
        return self.all_gather(h).float()  # .shape = (bsz, seqlen, vocab_size)

    def decode_forward(self, tokens: torch.Tensor):
        h = self.embed_tokens(tokens)
        h = self.all_gather(h)  # .shape = (bsz, seqlen, dim)
        for li in range(self.args.n_layers):
            h = self.layers[str(li)].decode_forward(h)
        h = self.lm_head(self.norm(h))
        return self.all_gather(h).float()  # .shape = (bsz, seqlen, vocab_size)


class QwQ:

    @staticmethod
    def build(model_path: str, node_id: int, device: torch.device) -> "QwQ":
        model_path = Path(model_path)
        non_ffn_filename = "non-ffn.pt"
        if not (model_path / non_ffn_filename).is_file():
            non_ffn_filename = f"non-ffn-{node_id}-{LOCAL_RANK}.pt"
        ffn_filename = f"ffn-{WORLD_RANK}.pt"
        if not (model_path / ffn_filename).is_file():
            ffn_filename = f"ffn-{node_id}-{LOCAL_RANK}.pt"

        model_args = ModelArgs.from_hf_config(get_json(model_path / "config.json"))
        model_args.device = device
        ws = torch.load(
            model_path / non_ffn_filename,
            map_location=device,
            weights_only=True,
            mmap=True,
        )
        ws.update(
            torch.load(
                model_path / ffn_filename,
                map_location=device,
                weights_only=True,
                mmap=True,
            )
        )

        intra_node_parallel = False
        # adjust for tensor parallel attention
        # WARNING: assumes that attention is intra-node parallel
        # TODO: adjust for pipeline parallelism
        if (
            ws[f"layers.0.self_attn.q_proj.weight"].shape[0]
            < model_args.n_heads * model_args.head_dim
        ):
            assert model_args.n_heads % LOCAL_WORLD_SIZE == 0
            assert model_args.n_kv_heads % LOCAL_WORLD_SIZE == 0
            model_args.n_heads //= LOCAL_WORLD_SIZE
            model_args.n_kv_heads //= LOCAL_WORLD_SIZE
            model_args.attn_tp = True
            intra_node_parallel = True
        if ws[f"layers.0.mlp.gate_proj.weight"].shape[0] < model_args.hidden_dim:
            model_args.hidden_dim //= WORLD_SIZE

        local_group = None
        if intra_node_parallel:
            global_map = torch.zeros((WORLD_SIZE, 2), dtype=torch.int64, device=device)
            local_map = torch.tensor(
                [node_id, WORLD_RANK], dtype=torch.int64, device=device
            )
            dist.all_gather_into_tensor(global_map, local_map)
            first_node = torch.min(global_map[:, 0]).item()
            last_node = torch.max(global_map[:, 0]).item()

            for ni in range(first_node, last_node + 1):
                ranks_on_node = global_map[global_map[:, 0] == ni][:, 1].tolist()
                node_group = dist.new_group(ranks_on_node, backend="nccl")
                if node_id == ni:
                    local_group = node_group

        with torch.device("meta"):
            model = Transformer(model_args, local_group)
        model.load_state_dict(ws, assign=True, strict=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        return QwQ(model, tokenizer)

    def __init__(
        self,
        model: Transformer,
        tokenizer,
    ):
        self.model: Transformer = model
        self.tokenizer = tokenizer

    def encode_prompts(
        self, prompts: list[str], device: torch.device
    ) -> list[torch.Tensor]:
        tokens = []
        for p in prompts:
            msgs = [{"role": "user", "content": p}]
            txt = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            # tensor.dtype = torch.int64
            ids = self.tokenizer([txt], return_tensors="pt").input_ids
            tokens.append(ids.view(-1).to(device))
        return tokens

    def get_cache(
        self, max_batch_size: int, max_seq_len: int, device: torch.device
    ) -> list[torch.Tensor]:
        return torch.empty(
            (
                2,  # key and value
                self.model.args.n_layers,
                max_batch_size,
                self.model.args.n_kv_heads,
                max_seq_len,
                self.model.args.head_dim,
            ),
            dtype=torch.bfloat16,
            device=device,
        )

    def clear_cache(self, cache: torch.Tensor):
        cache.zero_()

    def get_mask(self, max_seq_len: int, dtype: torch.dtype, device: torch.device):
        mask = torch.full(
            (max_seq_len, max_seq_len), float("-inf"), dtype=dtype, device=device
        )
        mask = torch.triu(mask, diagonal=1)
        return mask

    @torch.inference_mode()
    def generate(
        self,
        prompts: list[str],
        *,
        max_gen_len: int,
        temperature: float,
        device: torch.device,
        draw_new_graph: bool = True,
        profile: bool = False,
    ) -> tuple[list[str], int, float, int, float]:
        encoded_prompts: list[torch.Tensor] = self.encode_prompts(prompts, device)
        min_p_len = min(p.size(dim=0) for p in encoded_prompts)
        max_p_len = max(p.size(dim=0) for p in encoded_prompts)
        max_seq_len = max_p_len + max_gen_len
        bsz = len(encoded_prompts)

        model = self.model.eval()
        cache = self.get_cache(bsz, max_seq_len, device)
        mask = self.get_mask(max_seq_len, model.dtype, device)
        p_store_idx = torch.arange(min_p_len, dtype=torch.long, device=device)
        d_store_idx = torch.arange(1, dtype=torch.long, device=device)
        model.set_batch_level_args(
            cache,
            mask,
            p_store_idx,
            d_store_idx,
        )
        if draw_new_graph:
            model.draw_graphs(bsz, min_p_len)
        dist.barrier()

        # warmup
        model.prefill_forward_graph(
            torch.ones((bsz, min_p_len), dtype=torch.int64, device=device)
        )
        model.decode_forward_graph(
            torch.ones((bsz, 1), dtype=torch.int64, device=device)
        )
        self.clear_cache(cache)

        dist.barrier()
        tic = time.time()
        prefill_time: float  # in sec
        decode_time: float  # in sec
        if profile:
            torch.cuda.cudart().cudaProfilerStart()

        tokens = torch.full(
            (bsz, max_seq_len),
            self.tokenizer.pad_token_id,
            dtype=torch.int64,
            device=device,
        )
        for k, t in enumerate(encoded_prompts):
            tokens[k, : t.size(dim=0)] = t

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device=device)
        input_text_mask = tokens != self.tokenizer.pad_token_id

        # notice:
        # 1. it seems that prompts with length < max will generate
        # max_seq_len - len(prompt) tokens
        # 2. when batch size > 1, only the first bsz * min_prompt_len tokens
        # will be processed in parallel. Longer prompts' remaining tokens are
        # evaluated one-by-one with the min prompt's token generation
        for cur_pos in range(min_p_len, max_seq_len):
            dist.barrier()
            if prev_pos > 0:
                idx = torch.arange(prev_pos, cur_pos, dtype=torch.long, device=device)
                d_store_idx.copy_(idx)
                logits = model.decode_forward_graph(tokens[:, prev_pos:cur_pos])
            else:
                logits = model.prefill_forward_graph(tokens[:, prev_pos:cur_pos])
                torch.cuda.synchronize()
                prefill_time = time.time() - tic
                tic = time.time()

            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, 0.8)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            eos_reached |= ~input_text_mask[:, cur_pos] & (next_token == self.tokenizer.eos_token_id)

            prev_pos = cur_pos
            if all(eos_reached):
                break

        if min_p_len != max_p_len:
            warning = termcolor.colored(
                "-" * 25
                + "\nprompts have non-unifrom length, performance analysis might be inaccurate\n"
                + "-" * 25,
                "red",
            )
            print(warning)

        # this part is from here:
        # https://github.com/meta-llama/llama3/blob/main/llama/generation.py
        responses = []
        for bi, tkns in enumerate(tokens.tolist()):
            # cut to max_gen_len
            p_len = encoded_prompts[bi].size(dim=0)
            tkns: list = tkns[p_len : p_len + max_gen_len]
            responses.append(tkns)
        responses = self.tokenizer.batch_decode(responses, skip_special_tokens=True)

        n_p_tkns = min_p_len * bsz
        n_gen_tkns = (cur_pos - min_p_len) * bsz

        decode_time = time.time() - tic
        if profile:
            torch.cuda.cudart().cudaProfilerStop()
        self.model.clear_graph()
        torch.cuda.empty_cache()

        return responses, n_p_tkns, n_gen_tkns, prefill_time, decode_time


def main(
    model_path: str,
    node_id: int,
    prompt: str,
    prompt_path: str,
    n_prompts: int = 1,
    batch_size: int = 1,
    max_gen_len: int = 128,
    hide_resp: bool = False,
):
    # assert prompt or (prompt_path and n_prompts and n_prompts > 0)
    # assert n_prompts % batch_size == 0
    prompts: list[str] = None
    if prompt:
        prompts = [prompt]
    else:
        dataset: list[str] = get_json(Path(prompt_path))["prompts"]
        n_repeats = -(n_prompts // -len(dataset))  # ceil division
        prompts = (dataset * n_repeats)[:n_prompts]

    gpu = torch.device(f"cuda:{LOCAL_RANK}")
    dist.init_process_group(
        "nccl", rank=WORLD_RANK, world_size=WORLD_SIZE, device_id=gpu
    )
    model = QwQ.build(model_path, node_id, gpu)

    prefill_tps = []
    decode_tps = []
    start = 0
    for end in range(batch_size, n_prompts + 1, batch_size):
        prompt_batch = prompts[start:end]
        bsz = len(prompt_batch)
        responses, n_p_tkns, n_gen_tkns, prefill_time, decode_time = model.generate(
            prompt_batch,
            max_gen_len=max_gen_len,
            temperature=0.0,
            device=gpu,
            draw_new_graph=True,
            profile=end == n_prompts,
        )

        if WORLD_RANK == 0:
            prefill_tp = n_p_tkns / prefill_time
            decode_tp = n_gen_tkns / decode_time
            if n_gen_tkns / bsz > max_gen_len * 0.9:
                prefill_tps.append(prefill_tp)
                decode_tps.append(decode_tp)

            print("=" * 20)
            print("PERFORMANCE BREAKDOWN\n")
            print("PROMPT EVALUATION:")
            print(f"token count: {n_p_tkns}")
            print(f"total time in sec(s): {prefill_time:.2f}")
            print(f"throughput: {prefill_tp:.2f} t/s")
            print("TOKEN GENERATION:")
            print(f"token count: {n_gen_tkns}")
            print(f"total time in sec(s): {decode_time:.2f}")
            if n_gen_tkns > 0:
                print(f"throughput: {decode_tp:.2f} t/s")
            else:
                responses = ["" for _ in prompt_batch]
            if not hide_resp:
                print("=" * 20)
                print("INS-N-OUTS")
                print(f"AVG seqlen: {(n_p_tkns / bsz):.2f}")
                for p, resp in zip(prompt_batch, responses):
                    print(f"PROMPT:\n{p}")
                    print(f"RESPONSE:\n{resp}\n")

        start = end
        time.sleep(3)

    if WORLD_RANK == 0:
        print("=" * 20)
        print("RUN STATISTICS")
        print(f"avg prefill throughput: {mean(prefill_tps):.2f} t/s")
        print(f"avg decode throughput: {mean(decode_tps):.2f} t/s")

    dist.barrier()
    # dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--node-id", type=int)
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--prompt-path", type=str)
    parser.add_argument("--n-prompts", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--hide-resp", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(DEFAULT_SEED)
    main(
        args.model_path,
        args.node_id or GROUP_RANK,
        args.prompt,
        args.prompt_path,
        args.n_prompts,
        args.batch_size,
        args.max_tokens,
        args.hide_resp,
    )

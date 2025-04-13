from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name = "Qwen/QwQ-32B"

model = AutoModelForCausalLM.from_pretrained(
    "/mnt/disk2/llm_team/silicon_mind/QwQ-32B",
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=False,
)
tokenizer = AutoTokenizer.from_pretrained("/mnt/disk2/llm_team/silicon_mind/QwQ-32B")

prompt = "How many r's are in the word \"strawberry\""
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
)
    

#model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
generation_args = {
    "max_new_tokens": 2048,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}

output = pipe(text, batch_size= 1,**generation_args)

print(output[0]["generated_text"])


# generated_ids = model.generate(
#     **model_inputs,
#     max_new_tokens=32768
# ) 
# generated_ids = [
#     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
# ]

# response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print(response)

from operator import itemgetter
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import subprocess
import time
import os

"""terminal color"""
TC = SimpleNamespace(
    **{
        "YELLOW": "\033[33m",
        "GREEN": "\033[92m",
        "RED": "\033[91m",
        "BLUE": "\033[34m",
        "RESET": "\033[0m",
    }
)


class Cmd:
    def __new__(
        self, cmd: str, cwd="./", timeout_duration=None, suppress=True
    ) -> tuple[int, str, str]:
        self.cmd = cmd
        self.cwd = cwd
        self.returncode = 0
        self.has_err = True

        if not suppress:
            print(f"{self.cmd}", end="", flush=True)
        cwd_not_cur = f" in {self.cwd}" if self.cwd != "./" else ""

        """ process setup """
        process = subprocess.Popen(
            self.cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            executable="bash",
            cwd=self.cwd,
        )

        """ timeout """
        # https://stackoverflow.com/a/13821695
        import signal

        class TimeoutError(Exception):
            pass

        def handler(signum, frame):
            raise TimeoutError()

        # set the timeout handler
        if timeout_duration is not None:
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(timeout_duration)

        """ execution """
        out = bytearray()
        err = bytearray()
        timeStarted = time.time()
        try:
            _out, _err = process.communicate()
            out = _out if _out is not None else out
            err = _err if _err is not None else err
            self.returncode = process.returncode
            if process.returncode != 0:
                raise RuntimeError(
                    f"returncode is not 0 but {process.returncode}. "
                    + str(out + err, encoding="utf8")
                )
        except RuntimeError as e:
            if not suppress:
                print(f"{cwd_not_cur} {TC.RED}[failed]{TC.RESET}")
            return self.returncode, str(out, encoding="utf8"), str(err, encoding="utf8")
        except TimeoutError as e:
            if not suppress:
                print(f"{cwd_not_cur} {TC.RED}[failed]{TC.RESET}")
            return self.returncode, str(out, encoding="utf8"), str(err, encoding="utf8")
        except:
            if not suppress:
                print(f"{cwd_not_cur} {TC.RED}[failed]{TC.RESET}")
            return self.returncode, str(out, encoding="utf8"), str(err, encoding="utf8")
        finally:  # reset timeout handler
            signal.alarm(0)

        timeDelta = time.time() - timeStarted
        if not suppress:
            print(f"{cwd_not_cur} {TC.GREEN}[passed]{TC.RESET} ({timeDelta:.3f}s)")
        self.has_err = False
        return self.returncode, str(out, encoding="utf8"), str(err, encoding="utf8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch-config", required=True, type=str)
    parser.add_argument("--terminate", action="store_true")
    args = parser.parse_args()
    try:
        with open(Path(args.launch_config), "r") as f:
            config: dict = json.load(f)
    except FileNotFoundError:
        raise

    world_size, master_addr, master_port, nodes, username = itemgetter(
        "world_size", "master_addr", "master_port", "nodes", "username"
    )(config)

    if not args.terminate:
        shared_exec_args = ""
        tmp = config["shared_exec_args"]
        if "prompt" in tmp:
            shared_exec_args += f'--prompt="{tmp["prompt"]}" '
        else:
            shared_exec_args += (
                f'--prompt-path={os.path.expanduser(tmp["prompt_path"])} '
            )
        for k in ["n_prompts", "batch_size", "max_tokens"]:
            if k in tmp:
                shared_exec_args += f'--{k.replace("_", "-")}={tmp[k]} '
        if tmp.get("hide_resp", False):
            shared_exec_args += f"--hide-resp "

    url: str
    node_info: dict
    for url, node_info in nodes.items():
        ssh_port, node_rank, ngpus, script, model_path, node_id = itemgetter(
            "ssh_port", "node_rank", "ngpus", "script", "model_path", "node_id"
        )(node_info)
        print(f"node {url}")
        base_cmd = (
            f"ssh -i ~/.ssh/id_merlin {username}@{url} -p {ssh_port} "
            + "'"
            + f'export PATH="$PATH:/home/{username}/miniconda3/condabin/" && '
            + f"cd /mnt/disk3/{username}/ntu_paslab && "
            + f"git pull origin main && "
            + f"conda activate phi && "
            + f"python /mnt/disk3/gaven/ntu_paslab/merlin/qwq-32B/launch/run_node.py "
        )

        if args.terminate:
            rc, out, err = Cmd(base_cmd + " --terminate'")
            if rc != 0:
                print(err.strip())
            else:
                print("[terminated successfully]")
            continue

        if node_info.get("profile", False):
            base_cmd += "--profile "
            if "profiling_output" in node_info:
                base_cmd += f'--profiling-output={node_info["profiling_output"]} '

        rc, out, err = Cmd(
            base_cmd
            + f"--nnodes={world_size} "
            + f"--node-rank={node_rank} "
            + f"--nproc-per-node={ngpus} "
            + f"--master-addr={master_addr} "
            + f"--master-port={master_port} "
            + f"--script={os.path.expanduser(script)} "
            + f"--model-path={model_path} "
            + f"--node-id={node_id} "
            + shared_exec_args
            + "'"
        )
        if rc != 0:
            print(err.strip())
        else:
            print("[launched successfully]")


if __name__ == "__main__":
    main()

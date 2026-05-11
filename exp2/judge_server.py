"""Judge LLM server using vLLM via Apptainer.

Apptainer invocation pattern adapted from Abhishek Kumar's working Prajna
setup: --cleanenv + --contain + --no-home + bind /dev/shm + bind $HOME.
This isolation is essential — without it, host CUDA/torch state bleeds into
the container and triggers a "driver too old" error during EngineCore init.

Designed to run as a separate SLURM job from the trainer. The trainer
discovers this server via the connection file written here.
"""

import os
import json
import logging
import socket
import subprocess
import time
import urllib.request
import urllib.error
from pathlib import Path

import torch
import tqdm


class JudgeServer:
    """Manages the vLLM judge server lifecycle via Apptainer."""

    PARTITION_TO_SCENARIO = {
        "dgx": "a100", "a100": "a100", "a40": "a40", "l40": "l40",
    }
    SCENARIO_DEFAULTS = {
        "a100": {"dtype": "bfloat16", "gpu_memory_utilization": 0.90},
        "a40":  {"dtype": "bfloat16", "gpu_memory_utilization": 0.90},
        "l40":  {"dtype": "bfloat16", "gpu_memory_utilization": 0.90},
    }

    def __init__(self, config: dict):
        self.model_path = Path(config["model_path"]).resolve()
        self.sif_path = Path(config["sif_path"]).resolve()
        self.host = config.get("host", "0.0.0.0")
        self.port = int(config.get("port", 8765))
        self.max_model_len = config.get("max_model_len", 4096)
        self.log_dir = Path(config["log_dir"])
        self.connection_file = Path(config["connection_file"])
        self.health_check_timeout = config.get("health_check_timeout", 1200)
        self.hf_home = Path(
            config.get("hf_home", str(Path.home() / ".cache" / "huggingface"))
        ).resolve()
        self.extra_serve_args = config.get("extra_serve_args", [])

        partition = os.environ.get(
            "SLURM_JOB_PARTITION", config.get("partition", "l40")
        )
        num_gpus = self._detect_num_gpus(config.get("tensor_parallel_size", 2))
        scenario = self.PARTITION_TO_SCENARIO.get(partition, "l40")
        defaults = self.SCENARIO_DEFAULTS[scenario]

        self.partition = partition
        self.scenario = scenario
        self.tensor_parallel_size = num_gpus
        self.dtype = config.get("dtype", defaults["dtype"])
        self.gpu_memory_utilization = config.get(
            "gpu_memory_utilization", defaults["gpu_memory_utilization"]
        )

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.connection_file.parent.mkdir(parents=True, exist_ok=True)
        self.hf_home.mkdir(parents=True, exist_ok=True)

        self._setup_logging()
        self.logger.info(
            f"Auto-detected: partition={partition}, num_gpus={num_gpus} "
            f"-> scenario={scenario}"
        )

    def _detect_num_gpus(self, fallback: int) -> int:
        for var in ("SLURM_GPUS_ON_NODE", "SLURM_GPUS", "SLURM_JOB_GPUS"):
            val = os.environ.get(var)
            if val:
                try:
                    if "," in val:
                        return len(val.split(","))
                    return int(val)
                except ValueError:
                    continue
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible:
            return len([x for x in cuda_visible.split(",") if x.strip()])
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        return fallback

    def _setup_logging(self):
        log_file = self.log_dir / "judge_server.log"
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            fh = logging.FileHandler(log_file, mode="w")
            fh.setFormatter(fmt)
            sh = logging.StreamHandler()
            sh.setFormatter(fmt)
            self.logger.addHandler(fh)
            self.logger.addHandler(sh)
        self.logger.info(f"Logging to {log_file}")

    def _get_hostname(self) -> str:
        return socket.gethostname()

    def _get_node_ip(self) -> str:
        """Get a routable IP for this node — same as Abhishek's `hostname -i`."""
        try:
            result = subprocess.run(
                ["hostname", "-i"], capture_output=True, text=True, check=True,
            )
            return result.stdout.strip().split()[0]
        except (subprocess.CalledProcessError, IndexError):
            return socket.gethostname()

    def _check_port_free(self) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", self.port))
                return True
            except OSError:
                return False

    def _build_command(self) -> list[str]:
        """Build the apptainer command using Abhishek's isolation pattern.

        Critical flags:
          --cleanenv:  drop host environment (prevents CUDA/torch state leakage)
          --contain:   minimal in-container filesystem
          --no-home:   don't auto-mount user $HOME
          --home /tmp: container's $HOME is /tmp (writable scratch)
          --workdir /tmp: cwd inside container
          -B /dev/shm: vLLM multiprocessing needs shared memory
          -B $HF_HOME:$HF_HOME: HF cache at same path inside
          -B $HOME:$HOME: model files live under $HOME, bind for read access
        """
        home = str(Path.home())
        hf_home = str(self.hf_home)

        cmd = [
            "apptainer", "exec", "--cleanenv", "--nv",
            "--contain", "--no-home",
            "--home", "/tmp", "--workdir", "/tmp",
            "-B", "/dev/shm",
            "-B", f"{hf_home}:{hf_home}",
            "-B", f"{home}:{home}",
            # Container-side env (--cleanenv strips the rest)
            "--env", f"HF_HOME={hf_home}",
            "--env", "HF_HUB_OFFLINE=1",
            "--env", "TRANSFORMERS_OFFLINE=1",
            "--env", "USER=user",
            "--env", "LOGNAME=user",
            str(self.sif_path),
            "vllm", "serve", str(self.model_path),
            "--host", self.host,
            "--port", str(self.port),
            "--dtype", self.dtype,
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--max-model-len", str(self.max_model_len),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--trust-remote-code",
        ]
        cmd.extend(self.extra_serve_args)
        return cmd

    def _write_connection_file(self, ready: bool, pid: int = None):
        node_ip = self._get_node_ip()
        info = {
            "host": self._get_hostname(),
            "ip": node_ip,
            "port": self.port,
            "url": f"http://{node_ip}:{self.port}",
            "model_path": str(self.model_path),
            "scenario": self.scenario,
            "partition": self.partition,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pid": pid or os.getpid(),
            "ready": ready,
            "timestamp": time.time(),
        }
        with self.connection_file.open("w") as f:
            json.dump(info, f, indent=2)
        self.logger.info(
            f"Connection file: {self.connection_file} (ready={ready}, ip={node_ip})"
        )

    def _poll_health(self, process, vllm_log: Path) -> bool:
        start = time.time()
        elapsed_so_far = 0
        with tqdm.tqdm(
            total=self.health_check_timeout, desc="Waiting for vLLM", unit="s",
        ) as pbar:
            while time.time() - start < self.health_check_timeout:
                ret = process.poll()
                if ret is not None:
                    self.logger.error(
                        f"Apptainer exited prematurely with code {ret}. "
                        f"See {vllm_log}"
                    )
                    return False
                try:
                    url = f"http://localhost:{self.port}/health"
                    with urllib.request.urlopen(url, timeout=5) as resp:
                        if resp.status == 200:
                            elapsed = time.time() - start
                            self.logger.info(f"Server ready (took {elapsed:.1f}s)")
                            pbar.update(self.health_check_timeout - pbar.n)
                            return True
                except (urllib.error.URLError, ConnectionRefusedError,
                        TimeoutError, OSError):
                    pass
                time.sleep(5)
                new_elapsed = int(time.time() - start)
                pbar.update(new_elapsed - elapsed_so_far)
                elapsed_so_far = new_elapsed
        return False

    def run(self):
        if not self.sif_path.exists():
            self.logger.error(f".sif not found: {self.sif_path}")
            return
        if not self.model_path.exists():
            self.logger.error(f"Model dir not found: {self.model_path}")
            return
        if not self._check_port_free():
            self.logger.error(f"Port {self.port} already in use.")
            return

        cmd = self._build_command()
        self.logger.info("=" * 80)
        self.logger.info("Starting Judge Server")
        self.logger.info(f"  .sif:           {self.sif_path}")
        self.logger.info(f"  Model:          {self.model_path}")
        self.logger.info(f"  Hostname:       {self._get_hostname()}")
        self.logger.info(f"  Node IP:        {self._get_node_ip()}")
        self.logger.info(f"  Endpoint:       http://{self._get_node_ip()}:{self.port}")
        self.logger.info(f"  Partition:      {self.partition}")
        self.logger.info(f"  TP size:        {self.tensor_parallel_size}")
        self.logger.info(f"  Dtype:          {self.dtype}")
        self.logger.info(f"  Max model len:  {self.max_model_len}")
        self.logger.info(f"  GPU mem util:   {self.gpu_memory_utilization}")
        self.logger.info(f"  HF_HOME:        {self.hf_home}")
        self.logger.info(f"  Extra args:     {self.extra_serve_args}")
        self.logger.info(f"  Command:        {' '.join(cmd)}")
        self.logger.info("=" * 80)

        self._write_connection_file(ready=False)

        vllm_log = self.log_dir / "judge_vllm.log"
        with vllm_log.open("w") as log_f:
            process = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
        self.logger.info(f"Apptainer started (PID={process.pid})")
        self.logger.info(f"vLLM stdout/stderr -> {vllm_log}")

        ready = self._poll_health(process, vllm_log)
        if not ready:
            self.logger.error(
                f"Server not ready in {self.health_check_timeout}s. Terminating."
            )
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
            return

        self._write_connection_file(ready=True, pid=process.pid)
        self.logger.info("Server running. Block until exit (scancel to stop).")

        try:
            return_code = process.wait()
            self.logger.info(f"Server exited with code {return_code}")
        except KeyboardInterrupt:
            self.logger.info("Interrupt received. Shutting down.")
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    )
    main_logger = logging.getLogger("Main")

    if not torch.cuda.is_available():
        main_logger.error("No GPU detected. Exiting.")
        return
    main_logger.info(f"Detected {torch.cuda.device_count()} GPU(s).")

    config = {
        "sif_path":   Path("./models/vllm_gemma4.sif"),
        "model_path": Path("./models/gemma-4-31b-it"),
        "hf_home":    Path.home() / ".cache" / "huggingface",

        "host": "0.0.0.0",
        "port": 8765,
        "max_model_len": 4096,

        "partition": "l40",
        "tensor_parallel_size": 2,

        # Headroom for the multimodal-budget check; language-only skips
        # the vision tower entirely.
        "extra_serve_args": [
            "--language-model-only",
            "--max-num-batched-tokens", "16384",
            "--generation-config", "vllm",
        ],

        "connection_file": Path("./exp2/outputs/judge_connection.json"),
        "log_dir":         Path("./exp2/logs"),
        "health_check_timeout": 1200,
    }

    server = JudgeServer(config)
    server.run()


if __name__ == "__main__":
    main()
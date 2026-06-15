"""CUDA / driver / vLLM diagnostic script.

Run this as a SLURM job on the same partition you intend to run the judge on
(l40), so we capture the *compute node's* driver — not the login node's.

What this script does:
  1. Prints host-side info (driver, runtime, torch CUDA, nvcc).
  2. Inspects the .sif: prints the container's PyTorch CUDA build version,
     vLLM version, and what nvidia-smi *inside* the container sees.
  3. Prints all SLURM_* and CUDA_* env vars so we can sanity-check what the
     scheduler is handing the job.
  4. Reports the apptainer / singularity binary version.
  5. Tries a tiny `torch.cuda.is_available()` + small kernel run inside the
     container to confirm the driver is actually usable, not just visible.

The output is the entire diagnosis. Paste it back to me.

Usage:
  Submit via your sbgpu.sh wrapper. The SBATCH directives below match
  your judge job:

    #SBATCH --partition=l40
    #SBATCH --qos=l40
    #SBATCH --gres=gpu:2
    #SBATCH --cpus-per-task=4
    #SBATCH --mem=16G

  Path to .sif and conda env are configured at the top of main().
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


# -------------------- helpers --------------------

def banner(title: str):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80, flush=True)


def run(cmd, timeout=60, check=False, env=None):
    """Run a command, return (rc, stdout, stderr). Never raises unless check=True."""
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            check=check, env=env,
        )
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"TIMEOUT after {timeout}s"
    except FileNotFoundError as e:
        return -2, "", f"NOT FOUND: {e}"
    except Exception as e:
        return -3, "", f"ERROR: {type(e).__name__}: {e}"


def show(cmd, timeout=60):
    """Run + pretty-print."""
    print(f"\n$ {' '.join(cmd) if isinstance(cmd, list) else cmd}", flush=True)
    rc, out, err = run(cmd, timeout=timeout)
    if out:
        print(out.rstrip())
    if err:
        print(f"[stderr] {err.rstrip()}")
    print(f"[exit {rc}]", flush=True)
    return rc, out, err


# -------------------- host checks --------------------

def host_info():
    banner("HOST: identity")
    show(["hostname"])
    show(["hostname", "-i"])
    show(["whoami"])
    show(["uname", "-a"])

    banner("HOST: nvidia-smi (driver + GPUs)")
    show(["nvidia-smi"])
    show(["nvidia-smi", "--query-gpu=name,driver_version,compute_cap,memory.total",
          "--format=csv"])

    banner("HOST: nvcc (CUDA toolkit, may be absent on compute nodes)")
    if shutil.which("nvcc"):
        show(["nvcc", "--version"])
    else:
        print("nvcc not on PATH (this is fine — driver matters, toolkit doesn't)")

    banner("HOST: CUDA libraries on disk")
    for p in ("/usr/local/cuda", "/usr/lib/x86_64-linux-gnu/libcuda.so",
              "/usr/lib/x86_64-linux-gnu/libcuda.so.1"):
        if os.path.exists(p):
            print(f"  {p} -> exists")
            if os.path.islink(p):
                print(f"     symlink to: {os.readlink(p)}")
        else:
            print(f"  {p} -> missing")
    show("ls -la /usr/local/ | grep -i cuda || true")


def host_torch():
    banner("HOST: Python + torch (in conda env that submitted this job)")
    print(f"sys.executable = {sys.executable}")
    print(f"sys.version    = {sys.version}")
    try:
        import torch
        print(f"torch.__version__       = {torch.__version__}")
        print(f"torch.version.cuda      = {torch.version.cuda}")
        print(f"torch.cuda.is_available = {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"torch.cuda.device_count = {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {p.name}  cc={p.major}.{p.minor}  "
                      f"mem={p.total_memory / 1024**3:.1f} GiB")
            try:
                _ = (torch.ones(2, device="cuda") + 1).cpu()
                print("tiny kernel run    = OK")
            except Exception as e:
                print(f"tiny kernel run    = FAILED: {type(e).__name__}: {e}")
    except Exception as e:
        print(f"torch import failed: {type(e).__name__}: {e}")


def host_env():
    banner("HOST: relevant env vars")
    keys = sorted(k for k in os.environ
                  if k.startswith(("SLURM_", "CUDA", "NVIDIA", "HF_", "HOME"))
                  or k in ("USER", "LOGNAME", "PATH", "LD_LIBRARY_PATH"))
    for k in keys:
        v = os.environ[k]
        if k == "PATH" or k == "LD_LIBRARY_PATH":
            v = v.replace(":", ":\n     ")
        print(f"  {k} = {v}")


# -------------------- apptainer / sif checks --------------------

def apptainer_binary():
    banner("APPTAINER: binary + version")
    if not shutil.which("apptainer"):
        print("apptainer not on PATH")
        if shutil.which("singularity"):
            print("(but singularity is — that's the older name)")
            show(["singularity", "--version"])
        return False
    show(["apptainer", "--version"])
    show(["which", "apptainer"])
    return True


def sif_inspect(sif_path: Path):
    banner(f"SIF: inspect {sif_path}")
    if not sif_path.exists():
        print(f"NOT FOUND: {sif_path}")
        print("Skipping all container checks.")
        return False
    print(f"size: {sif_path.stat().st_size / 1024**3:.2f} GiB")
    show(["apptainer", "inspect", str(sif_path)])
    show(["apptainer", "inspect", "--list-apps", str(sif_path)])
    return True


def container_versions(sif_path: Path):
    """Run nvidia-smi + a Python probe INSIDE the container."""
    banner("CONTAINER: nvidia-smi (with --nv passthrough)")
    show(["apptainer", "exec", "--nv", str(sif_path), "nvidia-smi"], timeout=120)

    banner("CONTAINER: torch + vLLM versions")
    probe = (
        "import sys, json, os; "
        "info = {'python': sys.version}; "
        "exec(\"\"\"\n"
        "try:\n"
        "    import torch\n"
        "    info['torch'] = torch.__version__\n"
        "    info['torch_cuda_build'] = torch.version.cuda\n"
        "    info['cudnn'] = getattr(torch.backends.cudnn, 'version', lambda: None)()\n"
        "    info['cuda_available'] = torch.cuda.is_available()\n"
        "    if torch.cuda.is_available():\n"
        "        info['device_count'] = torch.cuda.device_count()\n"
        "        info['device_0'] = torch.cuda.get_device_name(0)\n"
        "        try:\n"
        "            x = torch.ones(2, device='cuda') + 1\n"
        "            info['tiny_kernel'] = 'OK'\n"
        "        except Exception as e:\n"
        "            info['tiny_kernel'] = f'{type(e).__name__}: {e}'\n"
        "except Exception as e:\n"
        "    info['torch_error'] = f'{type(e).__name__}: {e}'\n"
        "try:\n"
        "    import vllm\n"
        "    info['vllm'] = vllm.__version__\n"
        "except Exception as e:\n"
        "    info['vllm_error'] = f'{type(e).__name__}: {e}'\n"
        "try:\n"
        "    import transformers\n"
        "    info['transformers'] = transformers.__version__\n"
        "except Exception as e:\n"
        "    info['transformers_error'] = f'{type(e).__name__}: {e}'\n"
        "\"\"\"); "
        "print(json.dumps(info, indent=2))"
    )
    cmd = [
        "apptainer", "exec", "--cleanenv", "--nv",
        "--contain", "--no-home",
        "--home", "/tmp", "--workdir", "/tmp",
        "-B", "/dev/shm",
        str(sif_path),
        "python3", "-c", probe,
    ]
    show(cmd, timeout=180)

    banner("CONTAINER: which vllm + vllm --version")
    show(["apptainer", "exec", "--cleanenv", "--nv", "--contain", "--no-home",
          "--home", "/tmp", str(sif_path), "which", "vllm"])
    show(["apptainer", "exec", "--cleanenv", "--nv", "--contain", "--no-home",
          "--home", "/tmp", str(sif_path), "vllm", "--version"], timeout=120)


def container_cuda_lib(sif_path: Path):
    banner("CONTAINER: CUDA libs that --nv tried to inject")
    # When --nv is used, host's libcuda.so etc. get bind-mounted in.
    # If it failed, you'll see container-baked stubs instead.
    cmd_inside = (
        "ls -la /usr/lib/x86_64-linux-gnu/ 2>/dev/null | grep -E 'libcuda|libnvidia' | head -40; "
        "echo '---'; "
        "find / -name 'libcuda.so*' 2>/dev/null | head -10"
    )
    show(["apptainer", "exec", "--nv", str(sif_path), "bash", "-c", cmd_inside],
         timeout=60)


# -------------------- driver vs torch CUDA decode --------------------

def decode_driver_compatibility():
    banner("ANALYSIS: driver vs container's torch CUDA build")
    print(
        "Rule of thumb:\n"
        "  • Driver version determines the *highest* CUDA runtime it supports.\n"
        "  • A torch built against CUDA X.Y needs driver >= the minimum for X.Y.\n"
        "  • CUDA 12.4 driver supports torch built for CUDA <= 12.4.\n"
        "  • CUDA 12.6 torch needs driver >= 525.85 (CUDA 12.0 forward-compat region)\n"
        "    BUT in practice on modern wheels, you typically need driver matching\n"
        "    or newer than the build's CUDA minor.\n"
        "  • The 'driver too old (found 12040)' error means: torch detected the host\n"
        "    driver reports CUDA runtime API 12.4, but the wheel inside the .sif was\n"
        "    compiled against a newer CUDA minor and refused to run.\n\n"
        "What to compare in the output above:\n"
        "  HOST  nvidia-smi 'CUDA Version: ??.?'   (this is the *driver's max* CUDA)\n"
        "  HOST  torch.version.cuda                (your conda env's torch CUDA)\n"
        "  CONT  torch.version.cuda                (the .sif's torch CUDA — KEY)\n"
        "  CONT  vllm.__version__\n\n"
        "If host driver max < container torch CUDA -> driver is too old.\n"
        "Either:\n"
        "  (a) Pull a vLLM image built against an older CUDA, OR\n"
        "  (b) Ask cluster admins to update the GPU driver, OR\n"
        "  (c) Use the conda-env vLLM directly (no container) matched to the driver.\n"
    )


def main():
    # ---- configure these to your paths ----
    SIF_PATH = Path("./gemma4.sif").resolve()
    # If your sif is elsewhere, change above. Same for the alt path Abhishek uses:
    if not SIF_PATH.exists():
        alt = Path("./models/vllm_gemma4.sif").resolve()
        if alt.exists():
            SIF_PATH = alt

    print(f"sif being inspected: {SIF_PATH}")
    print(f"cwd: {os.getcwd()}")

    host_info()
    host_torch()
    host_env()

    if apptainer_binary():
        if sif_inspect(SIF_PATH):
            container_versions(SIF_PATH)
            container_cuda_lib(SIF_PATH)

    decode_driver_compatibility()

    banner("DONE")


if __name__ == "__main__":
    main()
#!/bin/bash
# sbgpu.sh — Submit a Python GPU job to SLURM
#
# Usage:
#   bash sbgpu.sh <script.py> [partition] [num_gpus] [time] [job_name] [mem] [node]
#
# Positional Arguments:
#   1. script.py    Path to the Python script to run (required)
#   2. partition    SLURM partition: dgx (default), a40, l40
#   3. num_gpus     Number of GPUs (default: 1)
#   4. time         Wall time limit (default: 12:00:00)
#   5. job_name     SLURM job name (default: script basename)
#   6. mem          Memory per node (default: 64G)
#   7. node         Node name to pin to, e.g. cn14-dgx (default: SLURM decides)
#
# Environment Variables:
#   SBGPU_CONDA_ENV   Conda environment name (default: mirl)
#   SBGPU_ACCOUNT     SLURM account (default: 23m2157)
#   SBGPU_NODE        Node name override, same as positional arg 7
#
# Partition Limits:
#   dgx   max 6 days  |  a40   max 4 days  |  l40   max 2 days
#
# Examples:
#   bash sbgpu.sh exp2/train.py
#       → dgx, 1 GPU, 12h, 64G, SLURM picks node
#
#   bash sbgpu.sh exp2/train.py dgx 1 12:00:00 grpo_train 128G
#       → dgx, 1 GPU, 12h, 128G, SLURM picks node
#
#   bash sbgpu.sh exp2/train.py dgx 1 12:00:00 grpo_train 128G cn14-dgx
#       → dgx, 1 GPU, 12h, 128G, pinned to cn14-dgx
#
#   bash sbgpu.sh exp2/train.py l40 2 48:00:00 grpo_l40 64G
#       → l40, 2 GPUs, 48h, 64G, SLURM picks node
#
#   SBGPU_NODE=cn14-dgx bash sbgpu.sh exp2/train.py
#       → dgx, 1 GPU, 12h, 64G, pinned to cn14-dgx via env var


PYSCRIPT="$1"
PARTITION="${2:-dgx}"
GPUS="${3:-1}"
TIME="${4:-12:00:00}"
JOBNAME="${5:-$(basename "${PYSCRIPT%.py}")}"
MEM="${6:-64G}"
NODELIST="${7:-${SBGPU_NODE:-}}"   # optional: e.g. "cn14-dgx"
CONDA_ENV="${SBGPU_CONDA_ENV:-mirl}"
ACCOUNT="${SBGPU_ACCOUNT:-23m2157}"
QOS="$PARTITION"

# If a specific node is requested, skip exclude; otherwise exclude cn11-dgx on dgx partition
EXCLUDE_NODES=""
if [ -z "$NODELIST" ] && [ "$PARTITION" = "dgx" ]; then
    EXCLUDE_NODES="cn11-dgx"
fi

if [ -z "$PYSCRIPT" ]; then
    echo "Usage: bash sbgpu.sh <script.py> [partition] [num_gpus] [time] [job_name] [mem] [node]"
    echo "Example: bash sbgpu.sh exp2/train.py dgx 1 12:00:00 grpo_train 128G cn14-dgx"
    echo "Partitions: dgx (6d), a40 (4d), l40 (2d)"
    echo "Node names: cn14-dgx, cn15-dgx, ... (omit to let SLURM decide)"
    exit 1
fi

if [ ! -f "$PYSCRIPT" ]; then
    echo "Error: '$PYSCRIPT' not found"
    exit 1
fi

PYSCRIPT_ABS=$(realpath "$PYSCRIPT")
WORKDIR=$(pwd)
LOGDIR="$HOME/logs/sbatch"
mkdir -p "$LOGDIR"

TMPSCRIPT=$(mktemp "$HOME/.sbgpu_job_XXXXXX.sh")
cat > "$TMPSCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=$JOBNAME
#SBATCH --partition=$PARTITION
#SBATCH --qos=$QOS
#SBATCH --account=$ACCOUNT
#SBATCH --gpus=$GPUS
#SBATCH --time=$TIME
#SBATCH --cpus-per-task=4
#SBATCH --mem=$MEM
${EXCLUDE_NODES:+#SBATCH --exclude=$EXCLUDE_NODES}
${NODELIST:+#SBATCH --nodelist=$NODELIST}
#SBATCH --output=$LOGDIR/${JOBNAME}_%j.log
#SBATCH --chdir=$WORKDIR

echo "=============================================="
echo "  SBGPU Job Script"
echo "=============================================="
echo "  Job ID:    \$SLURM_JOB_ID"
echo "  Node:      \$(hostname -s)"
echo "  Started:   \$(date)"
echo "  Python:    $PYSCRIPT_ABS"
echo "  Partition: $PARTITION"
echo "  GPUs:      $GPUS"
echo "  Memory:    $MEM"
echo "  Time:      $TIME"
echo "  Conda:     $CONDA_ENV"
echo "=============================================="
echo ""
echo "---------- Batch Script Contents ----------"
cat "\$0"
echo "---------- End of Script -------------------"
echo ""

source ~/miniconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV
python -u $PYSCRIPT_ABS

echo ""
echo "Job finished at \$(date)"
EOF
chmod +x "$TMPSCRIPT"

NODE_DISPLAY="${NODELIST:-<slurm picks>}"
echo "Submitting job:"
echo "  python:    $PYSCRIPT_ABS"
echo "  workdir:   $WORKDIR"
echo "  partition: $PARTITION | qos: $QOS | account: $ACCOUNT${EXCLUDE_NODES:+ | exclude: $EXCLUDE_NODES}"
echo "  GPUs:      $GPUS | node: $NODE_DISPLAY | mem: $MEM | time: $TIME | conda: $CONDA_ENV"
echo "  job name:  $JOBNAME"
echo "  log:       $LOGDIR/${JOBNAME}_<jobid>.log"
echo "----------------------------------------------------------------"

JOB_ID=$(sbatch --parsable "$TMPSCRIPT")

if [ -z "$JOB_ID" ]; then
    echo "Error: sbatch submission failed"
    rm -f "$TMPSCRIPT"
    exit 1
fi

echo "Submitted job $JOB_ID"
echo "  Tail log:     tail -f $LOGDIR/${JOBNAME}_${JOB_ID}.log"
echo "  Check status: squeue -j $JOB_ID"
echo "  Cancel:       scancel $JOB_ID"
echo "  Temp script:  $TMPSCRIPT (auto-deleted after job starts)"

(
    while true; do
        STATE=$(squeue -j "$JOB_ID" -h -o "%T" 2>/dev/null)
        if [ -z "$STATE" ]; then
            break
        elif [ "$STATE" = "RUNNING" ]; then
            sleep 10
            break
        fi
        sleep 5
    done
    rm -f "$TMPSCRIPT"
) &
disown
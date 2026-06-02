#!/bin/bash
# sbgpu.sh — Submit a Python GPU job to SLURM
#
# Usage: bash sbgpu.sh <script.py> [partition] [num_gpus] [time] [job_name] [mem]
#
# Examples:
#   bash sbgpu.sh exp2/train.py                                    # dgx, 1 GPU, 12h, 64G
#   bash sbgpu.sh exp2/train.py dgx 1 12:00:00 grpo_train 128G
#   bash sbgpu.sh exp2/train.py l40 2 48:00:00 grpo_l40 64G
#   bash sbgpu.sh exp2/train.py a40 1 04:00:00 reward_test
#
# Env vars:
#   SBGPU_CONDA_ENV  (default: mirl)
#   SBGPU_ACCOUNT    (default: 23m2157)

PYSCRIPT="$1"
PARTITION="${2:-dgx}"
GPUS="${3:-1}"
TIME="${4:-12:00:00}"
JOBNAME="${5:-$(basename "${PYSCRIPT%.py}")}"
MEM="${6:-64G}"
CONDA_ENV="${SBGPU_CONDA_ENV:-mirl}"
ACCOUNT="${SBGPU_ACCOUNT:-23m2157}"
QOS="$PARTITION"
EXCLUDE_NODES=""
if [ "$PARTITION" = "dgx" ]; then
    EXCLUDE_NODES="cn11-dgx"
fi

if [ -z "$PYSCRIPT" ]; then
    echo "Usage: bash sbgpu.sh <script.py> [partition] [num_gpus] [time] [job_name] [mem]"
    echo "Example: bash sbgpu.sh exp2/train.py dgx 1 12:00:00 grpo_train 128G"
    echo "Partitions: dgx (6d), a40 (4d), l40 (2d)"
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

# Create a proper bash batch script (avoids --wrap and /bin/sh issues on DGX)
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

echo "Submitting job:"
echo "  python:    $PYSCRIPT_ABS"
echo "  workdir:   $WORKDIR"
echo "  partition: $PARTITION | qos: $QOS | account: $ACCOUNT${EXCLUDE_NODES:+ | exclude: $EXCLUDE_NODES}"
echo "  GPUs:      $GPUS | mem: $MEM | time: $TIME | conda: $CONDA_ENV"
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

# Background process: wait for job to start running, then clean up
(
    while true; do
        STATE=$(squeue -j "$JOB_ID" -h -o "%T" 2>/dev/null)
        if [ -z "$STATE" ]; then
            # Job gone from queue (finished/cancelled/failed)
            break
        elif [ "$STATE" = "RUNNING" ]; then
            # Job running — script contents already printed via cat $0
            sleep 10
            break
        fi
        sleep 5
    done
    rm -f "$TMPSCRIPT"
) &
disown
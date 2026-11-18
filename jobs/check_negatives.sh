#!/bin/bash
#SBATCH --partition=uri-cpu,cpu-preempt
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --job-name=climx_check_negatives
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# Submit from the jobs/ directory:  cd jobs && sbatch check_negatives.sh
# Forcings only by default. To also check all 7 daily targets, pass --targets. That reads
# ~35 daily arrays of ~7 GB each (uncompressed), so raise the time limit:
#   sbatch --time=02:00:00 check_negatives.sh --targets
#
# Resources are based on the full-data run with --targets (job 1875511, 8 CPUs, 16G):
#   27.5 min elapsed, 15 min total CPU time (~0.55 cores busy on average: reading is
#   limited by HDF5's single-reader lock, so extra CPUs do not help), peak memory ~16 GB
#   (at that limit). Hence 2 CPUs and 32G here.
set -euo pipefail

hostname
date

# Make conda available in this non-interactive shell, then activate the env
module load conda/latest
set +u                                                # conda's activation scripts use unset variables
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate clima_emu_new
set -u
module list
which python
python --version

DATA="$(ws_find climx)/data/full"
echo "Data: $DATA"

python "$SLURM_SUBMIT_DIR/check_negatives.py" "$DATA" "$@"

date

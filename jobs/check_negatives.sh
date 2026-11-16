#!/bin/bash
#SBATCH --partition=uri-cpu,cpu-preempt
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=climx_check_negatives
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# Submit from the jobs/ directory:  cd jobs && sbatch check_negatives.sh
# Forcings only by default. To also check pr/huss/sfcWind, pass --targets
# (reads the large daily arrays: raise --time, e.g. sbatch --time=04:00:00 check_negatives.sh --targets).
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

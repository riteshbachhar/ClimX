#!/bin/bash
#SBATCH --partition=uri-cpu,cpu-preempt
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --job-name=climx_download_full
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail

hostname
date

# Make conda available in this non-interactive shell, then activate the env
module load conda/latest
set +u                                                # conda's activation scripts use unset variables
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate clima_emu_new
set -u

WS=$(ws_find climx)
DEST="$WS/data"
mkdir -p "$DEST"
echo "Downloading to $DEST"

hf download isp-uv-es/ClimX --repo-type dataset \
    --include "full/*" --exclude "*forcings_all*" \
    --local-dir "$DEST" --max-workers 2

ls -l "$DEST/full"
date

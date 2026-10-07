#!/usr/bin/env bash
#SBATCH --job-name=protein-jepa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=protein-jepa-%j.log
set -euo pipefail
# Select your site's account/partition at sbatch time; none is assumed here.
: "${MANIFEST:?Set MANIFEST to the audited train/val/test JSONL manifest}"
: "${OUTDIR:?Set OUTDIR to a new output directory}"
export OMP_NUM_THREADS=4
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:$PYTHONPATH}"
NPROC="${NPROC:-4}"
srun torchrun --standalone --nproc_per_node="$NPROC" -m protein_jepa.cli train \
  --config configs/cueq_gpu.yaml --manifest "$MANIFEST" --output "$OUTDIR" "$@"

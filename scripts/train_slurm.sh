#!/usr/bin/env bash
#SBATCH --job-name=protein-jepa
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=protein-jepa-%j.log
set -euo pipefail
# Select your site's account/partition at sbatch time; none is assumed here.
# Multi-node: sbatch --nodes=N ...; one torchrun per node, ranks meet at the first node.
: "${MANIFEST:?Set MANIFEST to the audited train/val/test JSONL manifest}"
: "${OUTDIR:?Set OUTDIR to a new output directory}"
CONFIG="${CONFIG:-configs/cueq_gpu.yaml}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:$PYTHONPATH}"
NPROC="${NPROC:-4}"
NNODES="${SLURM_NNODES:-1}"
HEAD="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
srun torchrun --nnodes="$NNODES" --nproc_per_node="$NPROC" \
  --rdzv_id="$SLURM_JOB_ID" --rdzv_backend=c10d --rdzv_endpoint="$HEAD:${RDZV_PORT:-29500}" \
  -m protein_jepa.cli train --config "$CONFIG" --manifest "$MANIFEST" --output "$OUTDIR" "$@"

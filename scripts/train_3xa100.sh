#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-configs/a100_3gpu_plasma_d704.yaml}"
shift || true

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

NPROC_PER_NODE="${NPROC_PER_NODE:-3}"
NNODES="${NNODES:-1}"

exec torchrun \
  --standalone \
  --nnodes "$NNODES" \
  --nproc_per_node "$NPROC_PER_NODE" \
  -m phi_plasma.train \
  --config "$CONFIG" \
  "$@"

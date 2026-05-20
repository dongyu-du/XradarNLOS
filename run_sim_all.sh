#!/bin/bash
# Run two-stage inference on ALL simulation data (train + test) → sim_all/
# Run from: /nfs/horai.dgpsrv/ondemand28/dongyu/Xbandradar/public/
set -e

PYTHON="/scratch/ondemand28/dongyu/anaconda3/envs/radar/bin/python"

PUBLIC_DIR="$(cd "$(dirname "$0")" && pwd)"
SIM_DATA="/nfs/horai.dgpsrv/ondemand28/dongyu/Xbandradar/public/data/simulated_data"
SPLITS_DIR="$PUBLIC_DIR/data/data_splits"
OUT_BASE="/scratch/ondemand28/dongyu/Xbandradar/public/results/sim_all"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

cd "$PUBLIC_DIR"

# ---------------------------------------------------------------------------
# Stage 1: SwinUNet — train
# ---------------------------------------------------------------------------
log "=== Stage 1 / train (1512 files) ==="
python infer_swin.py \
    --mode       sim \
    --input_dir  "$SIM_DATA/train/RA" \
    --gt_dir     "$SIM_DATA/train/points" \
    --file_list  "$SPLITS_DIR/train_files.txt" \
    --output_dir "$OUT_BASE/stage1/train" \
    --save_figs

# ---------------------------------------------------------------------------
# Stage 1: SwinUNet — test
# ---------------------------------------------------------------------------
log "=== Stage 1 / test (324 files) ==="
python infer_swin.py \
    --mode       sim \
    --input_dir  "$SIM_DATA/test/RA" \
    --gt_dir     "$SIM_DATA/test/points" \
    --file_list  "$SPLITS_DIR/test_files.txt" \
    --output_dir "$OUT_BASE/stage1/test" \
    --save_figs

# ---------------------------------------------------------------------------
# Stage 2: NLOS reconstruction — train
# ---------------------------------------------------------------------------
log "=== Stage 2 / train ==="
python infer_nlos.py \
    --input_dir  "$OUT_BASE/stage1/train" \
    --output_dir "$OUT_BASE/stage2/train" \
    --checkpoint "checkpoints/reflectnet_stage2_simdata.pth" \
    --save_figs

# ---------------------------------------------------------------------------
# Stage 2: NLOS reconstruction — test
# ---------------------------------------------------------------------------
log "=== Stage 2 / test ==="
python infer_nlos.py \
    --input_dir  "$OUT_BASE/stage1/test" \
    --output_dir "$OUT_BASE/stage2/test" \
    --checkpoint "checkpoints/reflectnet_stage2_simdata.pth" \
    --save_figs

log "=== All done. Results in $OUT_BASE ==="

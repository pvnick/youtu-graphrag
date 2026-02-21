#!/usr/bin/env bash
# rebuild_full_pipeline.sh
# Full NTSB rebuild with improved schema:
#   1. Graph extraction + community detection (~25h)
#   2. Post-processing: fix_chunks, canonicalize, add_aggregates
#   3. FAISS cache rebuild (offline, before service restart)
#   4. Restart service

set -euo pipefail
LOGDIR=/tmp
WORKDIR=/root/youtu-graphrag
PYTHON=/root/youtu-graphrag/venv/bin/python
cd "$WORKDIR"

log() { echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*" | tee -a "$LOGDIR/ntsb-full-rebuild-pipeline.log"; }

log "=========================================="
log "NTSB Full Rebuild Pipeline Starting"
log "=========================================="

# ── Step 1: Graph extraction + community detection ──────────────────────────
log "STEP 1: Graph extraction + community detection"
export OPENBLAS_NUM_THREADS=1

$PYTHON main.py --datasets ntsb_full 2>&1 | tee -a "$LOGDIR/ntsb-full-rebuild-pipeline.log"
EXIT_CODE=${PIPESTATUS[0]}

if [ "$EXIT_CODE" -ne 0 ]; then
  log "ERROR: main.py exited with code $EXIT_CODE — aborting pipeline"
  exit 1
fi

log "Graph extraction complete. Verifying output..."
if [ ! -f "output/graphs/ntsb_full_new.json" ]; then
  log "ERROR: ntsb_full_new.json not found — aborting"
  exit 1
fi

GRAPH_SIZE=$(du -sh output/graphs/ntsb_full_new.json | cut -f1)
log "Graph file: $GRAPH_SIZE"

# ── Step 2: fix_chunks ───────────────────────────────────────────────────────
log "STEP 2: Reconstructing chunk file..."
$PYTHON scripts/fix_chunks.py --dataset ntsb_full 2>&1 | tee -a "$LOGDIR/ntsb-full-rebuild-pipeline.log"
log "fix_chunks done"

# ── Step 3: canonicalize ─────────────────────────────────────────────────────
log "STEP 3: Canonicalizing near-duplicate entities..."
$PYTHON scripts/canonicalize.py --dataset ntsb_full 2>&1 | tee -a "$LOGDIR/ntsb-full-rebuild-pipeline.log"
log "canonicalize done"

# ── Step 4: add_aggregates ───────────────────────────────────────────────────
log "STEP 4: Adding aggregate nodes..."
$PYTHON scripts/add_aggregates.py --dataset ntsb_full 2>&1 | tee -a "$LOGDIR/ntsb-full-rebuild-pipeline.log"
log "add_aggregates done"

# ── Step 5: Clear old FAISS cache ────────────────────────────────────────────
log "STEP 5: Clearing old FAISS cache..."
rm -rf retriever/faiss_cache_new/ntsb_full/
mkdir -p retriever/faiss_cache_new/ntsb_full/
log "FAISS cache cleared"

# ── Step 6: Rebuild FAISS indices offline ────────────────────────────────────
log "STEP 6: Rebuilding FAISS indices (this takes ~4-6 hours)..."
$PYTHON - <<'PYEOF' 2>&1 | tee -a "$LOGDIR/ntsb-full-rebuild-pipeline.log"
import sys, os
sys.path.insert(0, '/root/youtu-graphrag')
os.chdir('/root/youtu-graphrag')

import json
import time
from config import get_config
from models.retriever.enhanced_kt_retriever import EnhancedKTRetriever

print("Loading config...", flush=True)
config = get_config('config/base_config.yaml')

print("Loading graph from output/graphs/ntsb_full_new.json ...", flush=True)
t0 = time.time()
with open('output/graphs/ntsb_full_new.json', 'r') as f:
    graph_data = json.load(f)
print(f"Graph loaded in {time.time()-t0:.1f}s", flush=True)

print("Initializing retriever...", flush=True)
retriever = EnhancedKTRetriever('ntsb_full', graph_data, config=config)

print("Building FAISS indices...", flush=True)
t1 = time.time()
retriever.build_indices()
print(f"FAISS indices built in {time.time()-t1:.1f}s", flush=True)

print("Building fast lookup indices...", flush=True)
retriever._build_fast_lookup_indices()
print("Done!", flush=True)
PYEOF

log "FAISS indices built"

# ── Step 7: Restart service ──────────────────────────────────────────────────
log "STEP 7: Restarting youtu-graphrag service..."
systemctl restart youtu-graphrag
sleep 10
STATUS=$(systemctl is-active youtu-graphrag)
log "Service status: $STATUS"

log "=========================================="
log "Pipeline COMPLETE — ntsb_full rebuilt with improved schema"
log "=========================================="

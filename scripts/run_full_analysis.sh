#!/usr/bin/env bash
set -euo pipefail
PYTHON=/root/youtu-graphrag/venv/bin/python
LOG=/tmp/ntsb_full_analysis.log
cd /root/youtu-graphrag

log() { echo "[$(date -u '+%H:%M:%S UTC')] $*" | tee -a "$LOG"; }

log "====== NTSB Forensic Analysis Pipeline ======"

log "Phase 1: Forensic causal chain + interaction + hidden factor queries (15 queries)"
$PYTHON scripts/ntsb_forensic_queries.py 2>&1 | tee -a "$LOG"

log "Phase 2: Community cluster profile queries (10 queries)"
$PYTHON scripts/ntsb_community_profiles.py 2>&1 | tee -a "$LOG"

log "Phase 3: Compiling final report"
$PYTHON scripts/compile_forensic_report.py 2>&1 | tee -a "$LOG"

log "====== Pipeline complete ======"
log "Report: /root/.openclaw/workspace/reports/ntsb_root_cause_analysis.md"

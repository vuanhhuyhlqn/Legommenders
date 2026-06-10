#!/usr/bin/env bash
# =============================================================================
# setup.sh — bootstrap a fresh / remote machine to run Legommenders.
#
# Idempotent: safe to re-run. By default it will
#   1. (optionally) create & use a local .venv,
#   2. install Python dependencies from requirements.txt,
#   3. ensure a `.data` file exists (copied from .data.example; the `mindimp`
#      entry uses `hf://huyva/mind-small` to auto-download the raw data),
#   4. wire up the HuggingFace token (from an existing .env, --token, or the
#      HUGGINGFACE_TOKEN / HF_TOKEN env vars),
#   5. process the dataset (auto-downloads the raw MIND data on first run).
#
# Typical remote flow:
#   scp .env  user@remote:/path/to/Legommenders/.env   # move your token over
#   ssh user@remote
#   cd /path/to/Legommenders
#   bash setup.sh --venv                 # install + build the mindimp dataset
#
# Useful flags:
#   --venv              create/use a local .venv (recommended on a fresh box)
#   --data NAME         dataset to process (default: mindimp)
#   --token hf_xxx      HuggingFace token (else read from .env / env vars)
#   --skip-install      do not (re)install Python dependencies
#   --skip-process      do not run data processing (just install + config)
#   --workers N         DataLoader/metric workers cap (sets LEGO_NUM_WORKERS &
#                       LEGO_GROUP_WORKER; use a small N / 1 on low-RAM boxes)
#   -h | --help         show this header
#
# NOTE: data processing for full MIND-small is RAM-heavy (needs well over the
# 13 GB that crashed the original dev box). Run it on a machine with ample RAM.
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")"
REPO_ROOT="$(pwd)"

# ---- defaults ------------------------------------------------------------- #
DATA="mindimp"
USE_VENV=0
SKIP_INSTALL=0
SKIP_PROCESS=0
WORKERS=""
TOKEN="${HUGGINGFACE_TOKEN:-${HF_TOKEN:-}}"
PY="${PYTHON:-python3}"

# ---- arg parsing ---------------------------------------------------------- #
while [ $# -gt 0 ]; do
  case "$1" in
    --venv)         USE_VENV=1; shift ;;
    --data)         DATA="${2:?--data needs a value}"; shift 2 ;;
    --token)        TOKEN="${2:?--token needs a value}"; shift 2 ;;
    --workers)      WORKERS="${2:?--workers needs a value}"; shift 2 ;;
    --skip-install) SKIP_INSTALL=1; shift ;;
    --skip-process) SKIP_PROCESS=1; shift ;;
    -h|--help)      sed -n '2,38p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

log()  { printf '\n\033[1;34m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\n\033[1;33m[setup:warn]\033[0m %s\n' "$*"; }

# ---- 1. (optional) virtual environment ------------------------------------ #
if [ "$USE_VENV" -eq 1 ]; then
  if [ ! -d .venv ]; then
    log "Creating virtual environment (.venv) ..."
    "$PY" -m venv .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  PY="python"
  log "Using venv: $(command -v python)"
fi

log "Python: $("$PY" --version 2>&1)"

# ---- 2. dependencies ------------------------------------------------------ #
if [ "$SKIP_INSTALL" -eq 0 ]; then
  log "Installing Python dependencies from requirements.txt ..."
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install -r requirements.txt
else
  log "Skipping dependency install (--skip-install)."
fi

# ---- 3. .data config ------------------------------------------------------ #
if [ ! -f .data ]; then
  log "No .data found — creating it from .data.example"
  cp .data.example .data
  warn "Edit .data if you need different dataset paths. The 'mindimp' entry"
  warn "defaults to 'hf://huyva/mind-small' (auto-download from the HF Hub)."
else
  log ".data already present — leaving it unchanged."
fi

# ---- 4. HuggingFace token ------------------------------------------------- #
if [ -f .env ] && grep -qE '^(HUGGINGFACE_TOKEN|HF_TOKEN|HUGGINGFACEHUB_API_TOKEN)=' .env; then
  log ".env present with a HuggingFace token — it will be picked up automatically."
elif [ -n "$TOKEN" ]; then
  log "Writing HuggingFace token to .env"
  printf 'HUGGINGFACE_TOKEN=%s\n' "$TOKEN" >> .env
else
  warn "No HuggingFace token found (.env / --token / HUGGINGFACE_TOKEN env)."
  warn "If 'huyva/mind-small' is private, the download will fail — provide a token:"
  warn "  scp your .env here, or run: bash setup.sh --token hf_xxx"
fi

# ---- 5. process the dataset (auto-downloads raw data) --------------------- #
if [ -n "$WORKERS" ]; then
  export LEGO_NUM_WORKERS="$WORKERS"
  export LEGO_GROUP_WORKER="$WORKERS"
  log "Set LEGO_NUM_WORKERS=LEGO_GROUP_WORKER=$WORKERS"
fi

if [ "$SKIP_PROCESS" -eq 0 ]; then
  log "Processing dataset '$DATA' (raw data auto-downloads on first run) ..."
  "$PY" process.py --data "$DATA"
  log "Dataset ready at data/$DATA/"
else
  log "Skipping data processing (--skip-process)."
fi

# ---- 6. next steps -------------------------------------------------------- #
log "Setup complete. Repo: $REPO_ROOT"
cat <<EOF

Next steps:
  # train (adjust batch size to your GPU; BERT embeddings ship in data/embeddings)
  ${PY} trainer.py \\
    --data config/data/${DATA}.yaml \\
    --model config/model/fastformer.yaml \\
    --embed config/embed/bertbase.yaml \\
    --hidden_size 256 --lr 0.001 --batch_size 64

  # then popularity-stratified (popular vs unpopular) impression evaluation
  ${PY} tester_popularity.py \\
    --data config/data/${DATA}.yaml \\
    --model config/model/fastformer.yaml \\
    --embed config/embed/bertbase.yaml \\
    --load_sign <signature-printed-during-training> \\
    --hidden_size 256 --pop_mass 0.8

  # on a low-RAM box, throttle parallelism, e.g.:
  #   LEGO_NUM_WORKERS=2 LEGO_GROUP_WORKER=1 ${PY} tester_popularity.py ...
EOF

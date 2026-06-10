#!/usr/bin/env bash
# =============================================================================
# setup.sh — bootstrap a fresh / remote machine to run Legommenders, using uv.
#
# Idempotent: safe to re-run. By default it will
#   1. install `uv` if missing,
#   2. create a local .venv (uv venv) with the requested Python,
#   3. install dependencies (uv pip install -r requirements.txt),
#   4. ensure a `.data` file exists (copied from .data.example; the `mindimp`
#      entry uses `hf://huyva/mind-small` to auto-download the raw data),
#   5. wire up the HuggingFace token (from an existing .env, --token, or the
#      HUGGINGFACE_TOKEN / HF_TOKEN env vars),
#   6. process the dataset (auto-downloads the raw MIND data on first run).
#
# Typical remote flow:
#   scp .env  user@remote:/workspace/Legommenders/.env   # move your token over
#   ssh user@remote
#   cd /workspace/Legommenders
#   bash setup.sh                        # install + build the mindimp dataset
#
# Useful flags:
#   --data NAME         dataset to process (default: mindimp)
#   --python VER        Python version for the venv (default: 3.12)
#   --token hf_xxx      HuggingFace token (else read from .env / env vars)
#   --skip-install      do not (re)create the venv / install dependencies
#   --skip-process      do not run data processing (just install + config)
#   --workers N         DataLoader/metric workers cap (sets LEGO_NUM_WORKERS &
#                       LEGO_GROUP_WORKER; use a small N / 1 on low-RAM boxes)
#   -h | --help         show this header
#
# After setup, activate the env with:  source .venv/bin/activate
#
# NOTE: data processing for full MIND-small is RAM-heavy (needs well over the
# 13 GB that crashed the original dev box). Run it on a machine with ample RAM.
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")"
REPO_ROOT="$(pwd)"

# ---- defaults ------------------------------------------------------------- #
DATA="mindimp"
PYVER="${PYVER:-3.12}"
SKIP_INSTALL=0
SKIP_PROCESS=0
WORKERS=""
TOKEN="${HUGGINGFACE_TOKEN:-${HF_TOKEN:-}}"

# ---- arg parsing ---------------------------------------------------------- #
while [ $# -gt 0 ]; do
  case "$1" in
    --data)         DATA="${2:?--data needs a value}"; shift 2 ;;
    --python)       PYVER="${2:?--python needs a value}"; shift 2 ;;
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

# ---- 1. uv ---------------------------------------------------------------- #
if ! command -v uv >/dev/null 2>&1; then
  log "Installing uv ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # make uv available in this shell session
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || { warn "uv not on PATH; open a new shell or add ~/.local/bin to PATH"; exit 1; }
log "uv: $(uv --version)"

# ---- 2 & 3. venv + dependencies ------------------------------------------ #
if [ "$SKIP_INSTALL" -eq 0 ]; then
  log "Creating virtual environment .venv (Python $PYVER) ..."
  uv venv --python "$PYVER" .venv
  log "Installing dependencies from requirements.txt ..."
  # Install into the .venv without needing to activate it first.
  VIRTUAL_ENV="$REPO_ROOT/.venv" uv pip install -r requirements.txt
else
  log "Skipping venv creation / dependency install (--skip-install)."
  [ -d .venv ] || { warn "No .venv found; re-run without --skip-install."; exit 1; }
fi

# Use the venv interpreter for everything below.
PY="$REPO_ROOT/.venv/bin/python"

# ---- 4. .data config ------------------------------------------------------ #
if [ ! -f .data ]; then
  log "No .data found — creating it from .data.example"
  cp .data.example .data
  warn "Edit .data if you need different paths. The 'mindimp' entry defaults to"
  warn "'hf://huyva/mind-small' (auto-download from the HuggingFace Hub)."
else
  log ".data already present — leaving it unchanged."
fi

# ---- 5. HuggingFace token ------------------------------------------------- #
if [ -f .env ] && grep -qE '^(HUGGINGFACE_TOKEN|HF_TOKEN|HUGGINGFACEHUB_API_TOKEN)=' .env; then
  log ".env present with a HuggingFace token — it will be picked up automatically."
elif [ -n "$TOKEN" ]; then
  log "Writing HuggingFace token to .env"
  printf 'HUGGINGFACE_TOKEN=%s\n' "$TOKEN" >> .env
else
  warn "No HuggingFace token found (.env / --token / HUGGINGFACE_TOKEN env)."
  warn "If 'huyva/mind-small' is private the download fails — scp your .env here"
  warn "or run: bash setup.sh --token hf_xxx"
fi

# ---- 6. process the dataset (auto-downloads raw data) --------------------- #
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

# ---- 7. next steps -------------------------------------------------------- #
log "Setup complete. Repo: $REPO_ROOT"
cat <<EOF

Activate the environment first:
  source .venv/bin/activate

Then train (adjust batch size to your GPU; BERT embeddings ship in data/embeddings):
  python trainer.py \\
    --data config/data/${DATA}.yaml \\
    --model config/model/fastformer.yaml \\
    --embed config/embed/bertbase.yaml \\
    --hidden_size 256 --lr 0.001 --batch_size 64

Then run popularity-stratified (popular vs unpopular) impression evaluation:
  python tester_popularity.py \\
    --data config/data/${DATA}.yaml \\
    --model config/model/fastformer.yaml \\
    --embed config/embed/bertbase.yaml \\
    --load_sign <signature-printed-during-training> \\
    --hidden_size 256 --pop_mass 0.8

On a low-RAM box, throttle parallelism, e.g.:
  LEGO_NUM_WORKERS=2 LEGO_GROUP_WORKER=1 python tester_popularity.py ...
EOF

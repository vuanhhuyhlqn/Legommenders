"""
hf_data.py

Auto-download **raw** datasets from the HuggingFace Hub so the same config runs
on a fresh machine (e.g. an SSH remote) without manually copying data.

A ``.data`` entry may point either to a local directory *or* to a HF *dataset*
repo using the ``hf://<repo_id>`` scheme, e.g.::

    mindimp = hf://huyva/mind-small

When the ``hf://`` scheme is used and the data is not already present locally,
the raw split folders (``train/`` and ``dev/`` with ``news.tsv`` /
``behaviors.tsv``) are downloaded once into ``download_dir`` (default
``data/raw/<name>``) and reused on subsequent runs.

Token resolution (for private repos), in order of preference:
    1. an explicit token argument / ``--hf_token`` CLI flag,
    2. the ``HF_TOKEN`` / ``HUGGINGFACE_TOKEN`` / ``HUGGINGFACEHUB_API_TOKEN``
       environment variables,
    3. a ``.env`` file in the current directory,
    4. ``../rec/.env`` (the sibling project's env file).

Secrets are never printed — only their presence is implied by behaviour.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

HF_SCHEME = "hf://"

_TOKEN_KEYS = ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACEHUB_API_TOKEN")
_DEFAULT_ENV_FILES = (".env", os.path.join("..", "rec", ".env"))


# --------------------------------------------------------------------------- #
# Token resolution                                                            #
# --------------------------------------------------------------------------- #
def _parse_env_file(path: str) -> dict:
    out: dict = {}
    if not os.path.isfile(path):
        return out
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def resolve_hf_token(
    explicit: Optional[str] = None,
    env_files: Sequence[str] = _DEFAULT_ENV_FILES,
) -> Optional[str]:
    """Return a HuggingFace token from the first available source (or None)."""
    if explicit:
        return explicit
    for key in _TOKEN_KEYS:
        if os.environ.get(key):
            return os.environ[key]
    for env_file in env_files:
        data = _parse_env_file(env_file)
        for key in _TOKEN_KEYS:
            if data.get(key):
                return data[key]
    return None


# --------------------------------------------------------------------------- #
# Raw-data resolution / download                                              #
# --------------------------------------------------------------------------- #
def _has_mind_splits(path: Optional[str], subfolders: Sequence[str]) -> bool:
    if not path:
        return False
    return all(
        os.path.exists(os.path.join(path, sub, "news.tsv")) for sub in subfolders
    )


def ensure_raw_data(
    name: str,
    local_dir: Optional[str],
    token: Optional[str] = None,
    download_dir: Optional[str] = None,
    subfolders: Sequence[str] = ("train", "dev"),
    logger=print,
) -> str:
    """Resolve a usable raw-data directory for dataset ``name``.

    * If ``local_dir`` is a real directory that already holds the splits, it is
      returned unchanged.
    * If ``local_dir`` uses the ``hf://<repo_id>`` scheme, the splits are
      downloaded (once) from that HF *dataset* repo into ``download_dir`` and
      that directory is returned.
    * Otherwise a clear :class:`FileNotFoundError` is raised.
    """
    # 1) Plain local path that already contains the data.
    if local_dir and not local_dir.startswith(HF_SCHEME) and os.path.isdir(local_dir):
        return local_dir

    # 2) Not an hf:// entry and not present locally -> cannot proceed.
    if not (local_dir and local_dir.startswith(HF_SCHEME)):
        raise FileNotFoundError(
            f"Raw data for '{name}' not found at '{local_dir}'. Point the .data "
            f"entry to a local directory containing {list(subfolders)} sub-folders, "
            f"or use 'hf://<repo_id>' to auto-download from the HuggingFace Hub."
        )

    # 3) hf:// scheme -> download into a project-local folder (reuse if present).
    repo_id = local_dir[len(HF_SCHEME):]
    download_dir = download_dir or os.path.join("data", "raw", name)

    if _has_mind_splits(download_dir, subfolders):
        return download_dir

    token = resolve_hf_token(token)
    from huggingface_hub import snapshot_download

    os.makedirs(download_dir, exist_ok=True)
    logger(
        f"[hf_data] downloading raw '{name}' from dataset repo '{repo_id}' "
        f"into '{download_dir}' (token: {'yes' if token else 'no'}) ..."
    )
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=[f"{sub}/*" for sub in subfolders],
        token=token,
        local_dir=download_dir,
    )
    if not _has_mind_splits(download_dir, subfolders):
        raise FileNotFoundError(
            f"Downloaded '{repo_id}' into '{download_dir}', but expected "
            f"{list(subfolders)} sub-folders with news.tsv were not found."
        )
    logger(f"[hf_data] raw '{name}' ready at '{download_dir}'")
    return download_dir

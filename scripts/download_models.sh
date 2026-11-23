#!/usr/bin/env bash
# Download the open-source models used to run longdoc locally (no API key needed).
#   Phi-3.5-mini-instruct (MIT)  - answers questions      (~2.4 GB)
#   bge-small-en-v1.5     (MIT)  - embeddings for search  (~36 MB)
# Usage: scripts/download_models.sh [target-dir]   (default: ./models)
set -euo pipefail
dir="${1:-models}"
mkdir -p "$dir"

fetch() {  # fetch <repo> <file>
  local url="https://huggingface.co/$1/resolve/main/$2"
  if [[ -s "$dir/$2" ]]; then echo "already present: $dir/$2"; return; fi
  echo "downloading $2 ..."
  curl -fL --retry 3 -o "$dir/$2.part" "$url"
  mv "$dir/$2.part" "$dir/$2"
}

fetch bartowski/Phi-3.5-mini-instruct-GGUF Phi-3.5-mini-instruct-Q4_K_M.gguf
fetch CompendiumLabs/bge-small-en-v1.5-gguf bge-small-en-v1.5-q8_0.gguf
ls -lh "$dir"

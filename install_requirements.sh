#!/usr/bin/env bash
set -euo pipefail

# Resolve script directory to allow invocation from anywhere
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_FILE="${SCRIPT_DIR}/requirements.txt"

if [[ ! -f "$REQ_FILE" ]]; then
  echo "requirements.txt not found at $REQ_FILE" >&2
  exit 1
fi

echo "[+] Using Python: $(which python)"
python -m pip install --upgrade pip wheel setuptools
echo "[+] Installing dependencies from $REQ_FILE"
python -m pip install -r "$REQ_FILE"

echo "[+] Verifying imports"
python - <<'EOF'
mods=['torch','transformers','datasets','numpy','tqdm']
import importlib, json
info={}
for m in mods:
    try:
        mod=importlib.import_module(m)
        info[m]=getattr(mod,'__version__','?')
    except Exception as e:
        info[m]=f"ERROR: {e}"
print(json.dumps(info, indent=2))
import torch
print('CUDA available:', torch.cuda.is_available())
EOF
echo "[✓] Installation complete"

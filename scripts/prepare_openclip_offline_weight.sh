#!/usr/bin/env bash
set -euo pipefail

KEY=${1:?usage: prepare_openclip_offline_weight.sh vit_l14|convnext_b}
ROOT=/data0/BadMerging
CACHE="$ROOT/open_clip"
mkdir -p "$CACHE/manual_downloads"

case "$KEY" in
  vit_l14)
    URL="https://huggingface.co/timm/vit_large_patch14_clip_224.openai/resolve/main/open_clip_model.safetensors"
    SIZE=1710517724
    REPO="$CACHE/models--timm--vit_large_patch14_clip_224.openai"
    COMMIT="18d0535469bb561bf468d76c1d73aa35156c922b"
    BLOB="9ce2e8a8ebfff3793d7d375ad6d3c35cb9aebf3de7ace0fc7308accab7cd207e"
    FILENAME="open_clip_model.safetensors"
    ;;
  convnext_b)
    URL="https://huggingface.co/laion/CLIP-convnext_base-laion400M-s13B-b51K/resolve/main/open_clip_pytorch_model.bin"
    SIZE=606154023
    REPO="$CACHE/models--laion--CLIP-convnext_base-laion400M-s13B-b51K"
    COMMIT="4c478c07a1511dafb7331443cd20bf7867a0f1b1"
    BLOB="f4a46714436bfb10d950778127f038db639b38d5e224167750c810a68662384c"
    FILENAME="open_clip_pytorch_model.bin"
    ;;
  *) echo "unknown key: $KEY" >&2; exit 2 ;;
esac

mkdir -p "$REPO/blobs" "$REPO/snapshots/$COMMIT"
TMP="$CACHE/manual_downloads/${KEY}.${FILENAME}.download"
echo "[$(date -Is)] downloading $KEY expected_bytes=$SIZE"
curl -L --fail --retry 100 --retry-delay 30 --retry-all-errors \
  --continue-at - --output "$TMP" "$URL"

ACTUAL=$(stat -c %s "$TMP")
if [[ "$ACTUAL" != "$SIZE" ]]; then
  echo "size mismatch for $KEY: $ACTUAL != $SIZE" >&2
  exit 1
fi

DEST="$REPO/blobs/$BLOB"
if [[ -e "$DEST" ]]; then
  mv "$DEST" "$DEST.corrupt.$(date +%Y%m%d_%H%M%S)"
fi
if [[ -e "$DEST.incomplete" ]]; then
  mv "$DEST.incomplete" "$DEST.incomplete.corrupt.$(date +%Y%m%d_%H%M%S)"
fi
mv "$TMP" "$DEST"
ln -sfn "../../blobs/$BLOB" "$REPO/snapshots/$COMMIT/$FILENAME"

if [[ "$KEY" == "vit_l14" ]]; then
  python - "$DEST" <<'PY'
import sys
from safetensors import safe_open
with safe_open(sys.argv[1], framework="pt") as f:
    keys = list(f.keys())
assert keys, "empty safetensors checkpoint"
print("validated safetensors tensors=", len(keys))
PY
else
  python - "$DEST" <<'PY'
import sys, torch
x = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert isinstance(x, dict) and x, "invalid PyTorch checkpoint"
print("validated torch checkpoint keys=", len(x))
PY
fi
echo "[$(date -Is)] offline cache ready: $DEST"

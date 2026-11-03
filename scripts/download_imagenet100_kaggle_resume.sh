#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATASET="${DATASET:-ambityga/imagenet100}"
ROOT="${ROOT:-./data/ImageNet100}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-${ROOT}/imagenet100_kaggle}"
EXTRACT_DIR="${EXTRACT_DIR:-${ROOT}/_extracting}"
BAD_DIR="${BAD_DIR:-${DOWNLOAD_DIR}/.bad}"
SLEEP_SECONDS="${SLEEP_SECONDS:-60}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-0}"
USE_PROXY="${USE_PROXY:-auto}"
PROXY_URL="${PROXY_URL:-http://127.0.0.1:10808}"
LOG_DIR="${LOG_DIR:-./logs}"

mkdir -p "${DOWNLOAD_DIR}" "${BAD_DIR}" "${LOG_DIR}"
LOG="${LOG_DIR}/download_imagenet100_kaggle_resume_$(date +%Y%m%d_%H%M%S).log"
ZIP_PATH="${DOWNLOAD_DIR}/imagenet100.zip"

log() {
  echo "[ImageNet100-download] $*" | tee -a "${LOG}"
}

proxy_available() {
  timeout 2 bash -c '</dev/tcp/127.0.0.1/10808' >/dev/null 2>&1
}

maybe_enable_proxy() {
  if [ "${USE_PROXY}" = "1" ]; then
    export http_proxy="${PROXY_URL}"
    export https_proxy="${PROXY_URL}"
    log "proxy enabled: ${PROXY_URL}"
    return
  fi
  if [ "${USE_PROXY}" = "auto" ] && proxy_available; then
    export http_proxy="${PROXY_URL}"
    export https_proxy="${PROXY_URL}"
    log "proxy auto-detected: ${PROXY_URL}"
  else
    log "proxy disabled"
  fi
}

zip_ok() {
  python3 - "$ZIP_PATH" <<'PY'
import sys, zipfile
path = sys.argv[1]
try:
    with zipfile.ZipFile(path) as zf:
        bad = zf.testzip()
        if bad is not None:
            print(f"bad member: {bad}")
            raise SystemExit(1)
        print(f"zip ok: {len(zf.infolist())} entries")
except Exception as exc:
    print(f"zip invalid: {type(exc).__name__}: {exc}")
    raise SystemExit(1)
PY
}

quarantine_bad_zip() {
  if [ -f "${ZIP_PATH}" ]; then
    local stamp
    stamp="$(date +%Y%m%d_%H%M%S)"
    log "moving invalid zip to ${BAD_DIR}/imagenet100.${stamp}.bad.zip"
    mv "${ZIP_PATH}" "${BAD_DIR}/imagenet100.${stamp}.bad.zip"
  fi
  if [ -f "${ZIP_PATH}.kaggle-partial" ]; then
    local stamp
    stamp="$(date +%Y%m%d_%H%M%S)"
    log "moving stale partial to ${BAD_DIR}/imagenet100.${stamp}.kaggle-partial"
    mv "${ZIP_PATH}.kaggle-partial" "${BAD_DIR}/imagenet100.${stamp}.kaggle-partial"
  fi
}

has_final_layout() {
  [ -d "${ROOT}/train" ] && [ -d "${ROOT}/val" ]
}

normalize_layout() {
  python3 - "$ROOT" "$EXTRACT_DIR" <<'PY'
import os
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1])
extract = Path(sys.argv[2])

def count_class_dirs(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for p in path.iterdir() if p.is_dir())

def find_named(name: str):
    candidates = []
    for p in [extract, *extract.rglob("*")]:
        if p.is_dir() and p.name.lower() == name and count_class_dirs(p) >= 10:
            candidates.append(p)
    return sorted(candidates, key=lambda p: (len(p.parts), str(p)))

def find_split(alias):
    names = [alias]
    if alias == "val":
        names += ["validation"]
    for name in names:
        hits = find_named(name)
        if hits:
            return hits[0]
    return None

train = find_split("train")
val = find_split("val")

# ambityga/imagenet100 stores the training classes across train.X1--X4
# and the validation classes under val.X. Treat the training shards as one
# ImageFolder split while retaining the extracted files in place.
train_sources = [train] if train is not None else sorted(
    p for p in extract.glob("train.X*") if count_class_dirs(p) > 0
)
if val is None:
    val_x = extract / "val.X"
    if count_class_dirs(val_x) > 0:
        val = val_x
val_sources = [val] if val is not None else []

if not train_sources or not val_sources:
    dirs = [p for p in extract.rglob("*") if p.is_dir()]
    dirs = sorted(dirs, key=lambda p: count_class_dirs(p), reverse=True)[:20]
    print("could not find train/val layout; top dirs:")
    for p in dirs:
        print(count_class_dirs(p), p)
    raise SystemExit(1)

def materialize_split(name: str, sources):
    dst = root / name
    if dst.exists():
        print(f"{dst} already exists; keep it")
        return
    tmp = root / f".{name}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    classes = {}
    for src in sources:
        for class_dir in sorted(p for p in src.iterdir() if p.is_dir()):
            if class_dir.name in classes:
                raise RuntimeError(
                    f"duplicate class {class_dir.name} in {src} and {classes[class_dir.name]}"
                )
            classes[class_dir.name] = class_dir
    for class_name, class_dir in sorted(classes.items()):
        target = os.path.relpath(class_dir, start=tmp)
        (tmp / class_name).symlink_to(target, target_is_directory=True)
    print(f"linked {len(classes)} classes from {len(sources)} source(s) -> {tmp}")
    tmp.rename(dst)

materialize_split("train", train_sources)
materialize_split("val", val_sources)

train_classes = {p.name for p in (root / "train").iterdir() if p.is_dir()}
val_classes = {p.name for p in (root / "val").iterdir() if p.is_dir()}
if train_classes != val_classes:
    raise RuntimeError(
        f"train/val class mismatch: train={len(train_classes)} val={len(val_classes)}"
    )

print("final layout:")
print("train classes", count_class_dirs(root / "train"))
print("val classes", count_class_dirs(root / "val"))
PY
}

download_once() {
  log "kaggle download start: ${DATASET}"
  kaggle datasets download -d "${DATASET}" -p "${DOWNLOAD_DIR}"
}

main() {
  maybe_enable_proxy
  log "root=${ROOT}"
  log "download_dir=${DOWNLOAD_DIR}"

  if has_final_layout; then
    log "final layout already exists: ${ROOT}/{train,val}"
    exit 0
  fi

  if [ -d "${EXTRACT_DIR}/val.X" ] && compgen -G "${EXTRACT_DIR}/train.X*" >/dev/null; then
    log "found extracted Kaggle shard layout; normalizing without re-extraction"
    normalize_layout 2>&1 | tee -a "${LOG}"
    if has_final_layout; then
      log "done: ${ROOT}/{train,val}"
      exit 0
    fi
  fi

  local attempt=0
  while true; do
    attempt=$((attempt + 1))
    if [ "${MAX_ATTEMPTS}" != "0" ] && [ "${attempt}" -gt "${MAX_ATTEMPTS}" ]; then
      log "exceeded MAX_ATTEMPTS=${MAX_ATTEMPTS}"
      exit 2
    fi

    log "attempt=${attempt}"
    if [ -f "${ZIP_PATH}" ]; then
      log "found zip: ${ZIP_PATH}"
      if zip_ok 2>&1 | tee -a "${LOG}"; then
        break
      fi
      quarantine_bad_zip
    fi

    if download_once 2>&1 | tee -a "${LOG}"; then
      if [ -f "${ZIP_PATH}" ] && zip_ok 2>&1 | tee -a "${LOG}"; then
        break
      fi
      quarantine_bad_zip
    else
      log "kaggle download failed"
    fi

    log "sleep ${SLEEP_SECONDS}s before retry"
    sleep "${SLEEP_SECONDS}"
  done

  log "extracting ${ZIP_PATH} -> ${EXTRACT_DIR}"
  mkdir -p "${EXTRACT_DIR}"
  python3 - "$ZIP_PATH" "$EXTRACT_DIR" <<'PY'
import sys, zipfile
zip_path, out_dir = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(zip_path) as zf:
    zf.extractall(out_dir)
print("extract done")
PY
  normalize_layout 2>&1 | tee -a "${LOG}"

  if has_final_layout; then
    log "done: ${ROOT}/{train,val}"
  else
    log "failed to create final layout"
    exit 3
  fi
}

main "$@"

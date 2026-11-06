#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 5 ]]; then
  echo "usage: $0 STATE_DIR NAME GPU COMMAND..." >&2
  exit 2
fi

STATE_DIR="$1"
NAME="$2"
GPU="$3"
shift 3

mkdir -p "${STATE_DIR}"
LOG="${STATE_DIR}/${NAME}.log"
printf '%s\n' "$*" > "${STATE_DIR}/${NAME}.command.txt"
printf 'gpu=%s\nstarted=%s\n' "${GPU}" "$(date --iso-8601=seconds)" \
  > "${STATE_DIR}/${NAME}.running"

set +e
"$@" > >(tee -a "${LOG}") 2>&1
RC=$?
set -e

rm -f "${STATE_DIR}/${NAME}.running"
if (( RC == 0 )); then
  printf 'gpu=%s\nfinished=%s\n' "${GPU}" "$(date --iso-8601=seconds)" \
    > "${STATE_DIR}/${NAME}.success"
else
  printf 'gpu=%s\nfailed=%s\nexit_code=%s\n' \
    "${GPU}" "$(date --iso-8601=seconds)" "${RC}" \
    > "${STATE_DIR}/${NAME}.failed"
fi
exit "${RC}"

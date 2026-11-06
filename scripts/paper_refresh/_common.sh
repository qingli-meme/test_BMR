#!/usr/bin/env bash
# shellcheck shell=bash

paper_refresh_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd
}

paper_refresh_archive_dir() {
  local root="$1"
  echo "${root}/paper_refresh_archive_010c97c"
}

archive_once() {
  local path="$1"
  local archive_root="$2"
  if [[ ! -e "${path}" ]]; then
    return 0
  fi
  mkdir -p "${archive_root}"
  local base
  base="$(basename "${path}")"
  local stamp
  stamp="$(date +%Y%m%d_%H%M%S)"
  mv "${path}" "${archive_root}/${base}__${stamp}"
}

require_file() {
  local p="$1"
  if [[ ! -f "${p}" ]]; then
    echo "[paper-refresh] missing prerequisite: ${p}" >&2
    exit 4
  fi
}

# Resolve a physical nvidia-smi index to its stable UUID. CUDA numeric
# ordinals can be reordered after a reboot.
paper_refresh_cuda_device() {
  local selector="$1"
  if [[ "${selector}" =~ ^[0-9]+$ ]]; then
    nvidia-smi -i "${selector}" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]'
  else
    printf '%s\n' "${selector}"
  fi
}

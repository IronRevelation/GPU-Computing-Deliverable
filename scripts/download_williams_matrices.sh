#this script was generated using AI for convenience

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${REPO_ROOT}/data"
BASE_URL="https://www.cise.ufl.edu/research/sparse/MM"
FORCE=0

MATRICES=(
  pdb1HYS
  consph
  cant
  pwtk
  mac_econ_fwd500
  mc2depi
  cop20k_A
  scircuit
  webbase-1M
  rail4284
)

matrix_group() {
  case "$1" in
    pwtk)
      echo "Boeing"
      ;;
    scircuit)
      echo "Hamm"
      ;;
    rail4284)
      echo "Mittelmann"
      ;;
    *)
      echo "Williams"
      ;;
  esac
}

usage() {
  cat <<EOF
Usage: $(basename "$0") [--force]

Download the Williams SuiteSparse Matrix Collection matrices used by this
project into:

  ${DATA_DIR}

Options:
  --force    Re-download and overwrite existing .mtx files.
  -h, --help Show this help message.
EOF
}

while (($#)); do
  case "$1" in
    --force)
      FORCE=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

for cmd in curl tar find mktemp; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command: $cmd" >&2
    exit 1
  fi
done

mkdir -p "$DATA_DIR"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

for matrix in "${MATRICES[@]}"; do
  target="${DATA_DIR}/${matrix}.mtx"

  if [[ -f "$target" && "$FORCE" -eq 0 ]]; then
    echo "Skipping ${matrix}: ${target} already exists"
    continue
  fi

  archive="${TMP_DIR}/${matrix}.tar.gz"
  extract_dir="${TMP_DIR}/${matrix}"
  mkdir -p "$extract_dir"

  group="$(matrix_group "$matrix")"

  echo "Downloading ${group}/${matrix}..."
  curl --fail --location --retry 3 --retry-delay 2 \
    --output "$archive" \
    "${BASE_URL}/${group}/${matrix}.tar.gz"

  tar -xzf "$archive" -C "$extract_dir"

  extracted_mtx="$(find "$extract_dir" -type f -name "${matrix}.mtx" -print -quit)"
  if [[ -z "$extracted_mtx" ]]; then
    echo "Could not find ${matrix}.mtx in downloaded archive" >&2
    exit 1
  fi

  cp "$extracted_mtx" "${target}.partial"
  mv "${target}.partial" "$target"
  echo "Saved ${target}"
done

echo "Done."

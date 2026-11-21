#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
make
repeats="${1:-3}"
for ((i=1; i<=repeats; i++)); do
  echo "== Repeat $i / $repeats =="
  for transfer in explicit hostptr aligned; do
    for kernel in scalar vec4; do
      echo "--- transfer=$transfer kernel=$kernel ---"
      ./build/super-boring-task --matvec-only \
        --matvec-transfer "$transfer" --matvec-kernel "$kernel"
    done
  done
done

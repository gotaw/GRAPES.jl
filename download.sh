#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"

KYOSHIN_USER="gotaw"
KYOSHIN_PASSWORD="wxG0xTKzLAtx"

export KYOSHIN_USER KYOSHIN_PASSWORD

mkdir -p "$DATA_DIR/kik" "$DATA_DIR/knet"

common_opts=(
  --user="$KYOSHIN_USER"
  --password="$KYOSHIN_PASSWORD"
  -m
  -np
  -nH
  --cut-dirs=4
)

wget "${common_opts[@]}" \
  -A ".kik.tar.gz" \
  -P "$DATA_DIR/kik" \
  "https://www.kyoshin.bosai.go.jp/kyoshin/download/kik/alldata/" &

wget "${common_opts[@]}" \
  -A ".knt.tar.gz" \
  -P "$DATA_DIR/knet" \
  "https://www.kyoshin.bosai.go.jp/kyoshin/download/knet/alldata/" &

wait
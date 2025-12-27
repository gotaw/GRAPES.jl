#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="${ENV_FILE:-$REPO_DIR/.env}"
DATA_DIR="$SCRIPT_DIR/data"

if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi

if [ -z "${KYOSHIN_USER:-}" ] || [ -z "${KYOSHIN_PASSWORD:-}" ]; then
  echo "Missing KYOSHIN_USER/KYOSHIN_PASSWORD. Set them in $ENV_FILE or export them before running." >&2
  exit 1
fi

export KYOSHIN_USER KYOSHIN_PASSWORD

mkdir -p "$DATA_DIR/kik" "$DATA_DIR/knet"

common_opts=(
  --user="$KYOSHIN_USER"
  --password="$KYOSHIN_PASSWORD"
  -m
  -np
  -nH
  --cut-dirs=4
  --accept-regex='^.*/kyoshin/download/(kik|knet)/alldata/(1997(/(1[0-2](/.*)?)|/?)|199[89](/.*)?|20(0[0-9]|1[0-8])(/.*)?)$'
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

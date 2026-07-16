#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="${TLDW_DOCS_SOURCE_DIR:-$ROOT_DIR/Docs}"
DEST_DIR="${TLDW_DOCS_PUBLISHED_DIR:-$ROOT_DIR/Docs/Published}"
STAGE_DIR="${DEST_DIR}.stage"
BACKUP_DIR="${DEST_DIR}.backup"

if [[ "$SRC_DIR" != /* || "$DEST_DIR" != /* || "$DEST_DIR" == "/" || "$SRC_DIR" == "$DEST_DIR" ]]; then
  echo "Docs source and destination must be distinct, safe absolute paths" >&2
  exit 1
fi

cleanup() {
  local status=$?
  trap - EXIT
  rm -rf -- "$STAGE_DIR"
  if [[ $status -ne 0 && -e "$BACKUP_DIR" ]]; then
    rm -rf -- "$DEST_DIR"
    mv "$BACKUP_DIR" "$DEST_DIR"
  fi
  exit "$status"
}
trap cleanup EXIT

require_file() {
  [[ -f "$1" ]] || { echo "Missing required docs file: $1" >&2; exit 1; }
}

require_dir() {
  [[ -d "$1" ]] || { echo "Missing required docs directory: $1" >&2; exit 1; }
}

copy_tree() {
  require_dir "$1"
  mkdir -p "$2"
  cp -R "$1"/. "$2"/
}

copy_file() {
  require_file "$1"
  mkdir -p "$(dirname "$2")"
  cp "$1" "$2"
}

rm -rf -- "$STAGE_DIR" "$BACKUP_DIR"
mkdir -p "$STAGE_DIR"

copy_file "$SRC_DIR/Site/index.md" "$STAGE_DIR/index.md"
copy_file "$SRC_DIR/Site/RELEASE_NOTES.md" "$STAGE_DIR/RELEASE_NOTES.md"
copy_tree "$SRC_DIR/Wiki" "$STAGE_DIR/Wiki"
copy_tree "$SRC_DIR/API-related" "$STAGE_DIR/API-related"
copy_tree "$SRC_DIR/ADR" "$STAGE_DIR/ADR"
copy_tree "$SRC_DIR/Code_Documentation" "$STAGE_DIR/Code_Documentation"

require_dir "$SRC_DIR/Deployment"
mkdir -p "$STAGE_DIR/Deployment"
shopt -s dotglob nullglob
for item in "$SRC_DIR/Deployment"/*; do
  [[ "${item##*/}" == "Monitoring" ]] && continue
  cp -R "$item" "$STAGE_DIR/Deployment/"
done
shopt -u dotglob nullglob

copy_tree "$SRC_DIR/Deployment/Monitoring" "$STAGE_DIR/Monitoring"
if [[ -d "$SRC_DIR/Evals" ]]; then
  copy_tree "$SRC_DIR/Evals" "$STAGE_DIR/Evaluations"
elif [[ -d "$SRC_DIR/Evaluations" ]]; then
  copy_tree "$SRC_DIR/Evaluations" "$STAGE_DIR/Evaluations"
else
  echo "Missing required docs directory: $SRC_DIR/Evals" >&2
  exit 1
fi
copy_tree "$SRC_DIR/Getting_Started" "$STAGE_DIR/Getting_Started"
copy_tree "$SRC_DIR/User_Guides" "$STAGE_DIR/User_Guides"
copy_file "$SRC_DIR/Architecture.md" "$STAGE_DIR/Architecture.md"
copy_file "$SRC_DIR/Operations/Env_Vars.md" "$STAGE_DIR/Env_Vars.md"
copy_file "$SRC_DIR/Overview/Feature_Status.md" "$STAGE_DIR/Overview/Feature_Status.md"
copy_file "$SRC_DIR/Logo.png" "$STAGE_DIR/assets/logo.png"
copy_file "$SRC_DIR/Logo.png" "$STAGE_DIR/assets/favicon.png"

while IFS= read -r -d '' readme; do
  [[ -f "${readme%/*}/index.md" ]] && rm -- "$readme"
done < <(find "$STAGE_DIR" -type f \( -name README -o -name README.md \) -print0)

mkdir -p "$(dirname "$DEST_DIR")"
if [[ -e "$DEST_DIR" ]]; then
  mv "$DEST_DIR" "$BACKUP_DIR"
fi
if [[ -n "${TLDW_DOCS_TEST_FAIL_AFTER_BACKUP:-}" ]]; then
  echo "Injected docs refresh failure after backup" >&2
  exit 1
fi
mv "$STAGE_DIR" "$DEST_DIR"
rm -rf -- "$BACKUP_DIR"

echo "Refreshed curated docs in: $DEST_DIR"

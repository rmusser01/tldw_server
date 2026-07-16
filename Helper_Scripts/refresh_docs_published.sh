#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${TLDW_DOCS_TEST_MODE:-}" != "1" ]]; then
  for seam in \
    TLDW_DOCS_SOURCE_DIR \
    TLDW_DOCS_PUBLISHED_DIR \
    TLDW_DOCS_TEST_FAIL_AFTER_BACKUP; do
    if [[ -n "${!seam+x}" ]]; then
      echo "$seam requires TLDW_DOCS_TEST_MODE=1" >&2
      exit 1
    fi
  done
fi

SRC_DIR="${TLDW_DOCS_SOURCE_DIR:-$ROOT_DIR/Docs}"
DEST_DIR="${TLDW_DOCS_PUBLISHED_DIR:-$ROOT_DIR/Docs/Published}"

if [[ "$SRC_DIR" != /* || "$DEST_DIR" != /* || "$DEST_DIR" == "/" ]]; then
  echo "Docs source and destination must be safe absolute paths" >&2
  exit 1
fi
if [[ ! -d "$SRC_DIR" ]]; then
  echo "Missing required docs directory: $SRC_DIR" >&2
  exit 1
fi
if [[ -L "$DEST_DIR" || ( -e "$DEST_DIR" && ! -d "$DEST_DIR" ) ]]; then
  echo "Docs destination must be a real directory path" >&2
  exit 1
fi

SRC_DIR="$(cd "$SRC_DIR" && pwd -P)"
if [[ -d "$DEST_DIR" ]]; then
  DEST_DIR="$(cd "$DEST_DIR" && pwd -P)"
else
  DEST_NAME="${DEST_DIR##*/}"
  DEST_PARENT="${DEST_DIR%/*}"
  [[ -n "$DEST_PARENT" && -d "$DEST_PARENT" ]] || {
    echo "Missing docs destination parent: $DEST_PARENT" >&2
    exit 1
  }
  DEST_PARENT="$(cd "$DEST_PARENT" && pwd -P)"
  DEST_DIR="${DEST_PARENT%/}/$DEST_NAME"
fi

if [[ "$DEST_DIR" == "/" || "$SRC_DIR" == "$DEST_DIR" || "$SRC_DIR" == "$DEST_DIR"/* ]]; then
  echo "Docs destination must not equal or contain the source path" >&2
  exit 1
fi

LOCK_DIR="${DEST_DIR}.lock"
BACKUP_DIR="${DEST_DIR}.backup"
STAGE_DIR="${DEST_DIR}.stage.$$"

if ! mkdir "$LOCK_DIR"; then
  echo "Docs refresh already locked: $LOCK_DIR" >&2
  exit 1
fi

LOCK_OWNED=1
STAGE_OWNED=0
BACKUP_OWNED=0

cleanup() {
  local status=$?
  trap - EXIT INT TERM HUP

  if [[ $STAGE_OWNED -eq 1 ]]; then
    if ! rm -rf -- "$STAGE_DIR"; then
      echo "Failed to clean docs stage: $STAGE_DIR" >&2
      status=1
    fi
  fi
  if [[ $status -ne 0 && $BACKUP_OWNED -eq 1 && ( -e "$BACKUP_DIR" || -L "$BACKUP_DIR" ) ]]; then
    if ! rm -rf -- "$DEST_DIR"; then
      echo "Failed to remove incomplete docs destination: $DEST_DIR" >&2
      status=1
    elif ! mv "$BACKUP_DIR" "$DEST_DIR"; then
      echo "Failed to restore docs backup: $BACKUP_DIR" >&2
      status=1
    else
      BACKUP_OWNED=0
    fi
  fi
  if [[ $LOCK_OWNED -eq 1 ]]; then
    if ! rmdir "$LOCK_DIR"; then
      echo "Failed to remove docs refresh lock: $LOCK_DIR" >&2
      status=1
    fi
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

if [[ -e "$BACKUP_DIR" || -L "$BACKUP_DIR" ]]; then
  if [[ -e "$DEST_DIR" || -L "$DEST_DIR" ]]; then
    echo "Docs destination and backup both exist; preserving both for inspection" >&2
  else
    mv "$BACKUP_DIR" "$DEST_DIR"
    echo "Restored interrupted docs backup; rerun the refresh" >&2
  fi
  exit 1
fi

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

if ! mkdir "$STAGE_DIR"; then
  echo "Docs stage already exists: $STAGE_DIR" >&2
  exit 1
fi
STAGE_OWNED=1

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
if [[ -d "$SRC_DIR/Evaluations" ]]; then
  copy_tree "$SRC_DIR/Evaluations" "$STAGE_DIR/Evaluations"
elif [[ -d "$SRC_DIR/Evals" ]]; then
  copy_tree "$SRC_DIR/Evals" "$STAGE_DIR/Evaluations"
else
  echo "Missing required docs directory: $SRC_DIR/Evaluations or $SRC_DIR/Evals" >&2
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
  BACKUP_OWNED=1
  mv "$DEST_DIR" "$BACKUP_DIR"
fi
if [[ -n "${TLDW_DOCS_TEST_FAIL_AFTER_BACKUP:-}" ]]; then
  echo "Injected docs refresh failure after backup" >&2
  exit 1
fi
mv "$STAGE_DIR" "$DEST_DIR"
STAGE_OWNED=0
if [[ $BACKUP_OWNED -eq 1 ]]; then
  rm -rf -- "$BACKUP_DIR"
  BACKUP_OWNED=0
fi

echo "Refreshed curated docs in: $DEST_DIR"

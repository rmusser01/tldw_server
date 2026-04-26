#!/usr/bin/env bash
set -euo pipefail

# Refresh curated docs in Docs/Published from source folders.
# Keeps only approved sections and preserves each section's index.md.
# Hosted/commercial docs stay out of Docs/Published and belong in the private repo.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$ROOT_DIR/Docs"
DEST_DIR="$SRC_DIR/Published"

echo "Refreshing curated docs in: $DEST_DIR"
mkdir -p "$DEST_DIR"

preserve_and_copy() {
  local src="$1"
  local dest="$2"
  mkdir -p "$dest"
  # Remove everything except index.md to preserve section landing pages
  if [ -d "$dest" ]; then
    find "$dest" -mindepth 1 -not -name 'index.md' -exec rm -rf {} +
  fi
  if [ -d "$src" ]; then
    # Copy contents of src into dest (without clobbering dest/index.md)
    shopt -s dotglob nullglob
    for item in "$src"/*; do
      # Skip Monitoring when syncing Deployment; handled separately
      if [ "$(basename "$src")" = "Deployment" ] && [ "$(basename "$item")" = "Monitoring" ]; then
        continue
      fi
      # Avoid README vs index conflicts in MkDocs: skip README files when an index.md is present
      if [ -f "$dest/index.md" ] && [[ "$(basename "$item")" =~ ^README(\.md)?$ ]]; then
        continue
      fi
      cp -R "$item" "$dest" 2>/dev/null || true
    done
    shopt -u dotglob nullglob
  fi
}

# API-related
preserve_and_copy "$SRC_DIR/API-related" "$DEST_DIR/API-related"

# Code_Documentation
preserve_and_copy "$SRC_DIR/Code_Documentation" "$DEST_DIR/Code_Documentation"

# Deployment (excluding embedded Monitoring dir)
preserve_and_copy "$SRC_DIR/Deployment" "$DEST_DIR/Deployment"

# Monitoring (promoted to top-level from Deployment/Monitoring)
preserve_and_copy "$SRC_DIR/Deployment/Monitoring" "$DEST_DIR/Monitoring"

# Evaluations
preserve_and_copy "$SRC_DIR/Evaluations" "$DEST_DIR/Evaluations"

# Getting Started (canonical onboarding profiles)
preserve_and_copy "$SRC_DIR/Getting_Started" "$DEST_DIR/Getting_Started"

# User Guides
preserve_and_copy "$SRC_DIR/User_Guides" "$DEST_DIR/User_Guides"

# Copy logo into docs assets (used as logo and favicon)
mkdir -p "$DEST_DIR/assets"
if [ -f "$SRC_DIR/Logo.png" ]; then
  cp "$SRC_DIR/Logo.png" "$DEST_DIR/assets/logo.png"
  cp "$SRC_DIR/Logo.png" "$DEST_DIR/assets/favicon.png"
fi

echo "Done."

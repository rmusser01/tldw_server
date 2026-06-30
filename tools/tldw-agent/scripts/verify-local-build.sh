#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tldw-agent-build.XXXXXX")"
GOCACHE_DIR="${TMP_DIR}/go-build"
trap 'rm -rf "${TMP_DIR}"' EXIT

cd "${ROOT_DIR}"

mkdir -p "${GOCACHE_DIR}"
export GOCACHE="${GOCACHE_DIR}"

go build -o "${TMP_DIR}/tldw-agent-host" ./cmd/tldw-agent-host
go build -o "${TMP_DIR}/tldw-agent-acp" ./cmd/tldw-agent-acp
go test ./...

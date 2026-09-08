#!/usr/bin/env python3
"""Render candidate-only frontend Dockerfiles from canonical recipes."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

EXPECTED_RUNTIME_FROM = (
    "FROM node:24.20.0-bookworm-slim@sha256:"
    "ba849c60be29959425b8734d57b8b4b7d56f98edd9504c9af091d5281095a71e AS runtime\n"
)
EXPECTED_BUILDER_FROM = (
    "FROM node:24.20.0-bookworm-slim@sha256:"
    "ba849c60be29959425b8734d57b8b4b7d56f98edd9504c9af091d5281095a71e AS builder\n"
)
CANDIDATE_RUNTIME_BLOCK = r"""# Candidate only: not admitted for production use.
FROM ubuntu:24.04@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517 AS runtime

ARG ZLIB1G_VERSION=1:1.3.dfsg-3.1ubuntu2.2
ARG LIBC6_VERSION=2.39-0ubuntu8.8
ENV NODE_VERSION=24.20.0

# APT verifies the distribution's signed InRelease metadata and the package
# hashes it contains. Retain those inputs and the acquired package hash.
RUN set -eux; \
    evidence=/usr/local/share/tldw-candidate-evidence; \
    mkdir -p "$evidence/apt-inrelease"; \
    apt-get update; \
    cp /var/lib/apt/lists/*_InRelease "$evidence/apt-inrelease/"; \
    sha256sum "$evidence"/apt-inrelease/* > "$evidence/apt-inrelease.sha256"; \
    apt-cache policy zlib1g > "$evidence/zlib1g-apt-policy.txt"; \
    apt-get install --download-only -y --no-install-recommends "zlib1g=${ZLIB1G_VERSION}"; \
    zlib_deb="$(find /var/cache/apt/archives -maxdepth 1 -type f -name 'zlib1g_*_amd64.deb' -print -quit)"; \
    test -n "$zlib_deb"; \
    sha256sum "$zlib_deb" > "$evidence/zlib1g-download.sha256"; \
    apt-get install -y --no-install-recommends "zlib1g=${ZLIB1G_VERSION}"; \
    test "$(dpkg-query -W -f='${Version}' zlib1g)" = "$ZLIB1G_VERSION"; \
    test "$(dpkg-query -W -f='${Version}' libc6)" = "$LIBC6_VERSION"; \
    dpkg-query -W -f='${binary:Package}\t${Version}\n' zlib1g libc6 libstdc++6 \
        > "$evidence/installed-versions.tsv"; \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*.deb

# The exact pinned Node image contains these artifacts but no system CA bundle.
# Node therefore retains the same built-in root store as the canonical runtime.
COPY --from=builder /usr/local/bin/node /usr/local/bin/node
COPY --from=builder /usr/local/bin/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
COPY --from=builder /usr/local/LICENSE /usr/local/LICENSE

ENTRYPOINT ["docker-entrypoint.sh"]
"""

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CANONICAL_DOCKERFILES = {
    "webui": REPOSITORY_ROOT / "Dockerfiles/Dockerfile.webui",
    "admin-ui": REPOSITORY_ROOT / "Dockerfiles/Dockerfile.admin-ui",
}


def render_candidate(source: str) -> str:
    """Replace the one canonical runtime marker or reject source drift."""
    if source.count(EXPECTED_RUNTIME_FROM) != 1:
        raise ValueError("expected exactly one canonical runtime stage")
    prefix, remainder = source.split(EXPECTED_RUNTIME_FROM)
    if prefix.count(EXPECTED_BUILDER_FROM) != 1:
        raise ValueError("expected canonical Node builder stage")
    return prefix + CANDIDATE_RUNTIME_BLOCK + remainder


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--application", required=True, choices=tuple(CANONICAL_DOCKERFILES))
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Render one canonical application recipe to the requested candidate path."""
    parser = _parser()
    args = parser.parse_args(argv)
    source = CANONICAL_DOCKERFILES[args.application].read_text(encoding="utf-8")
    try:
        candidate = render_candidate(source)
    except ValueError as exc:
        parser.error(str(exc))
    args.output.write_text(candidate, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

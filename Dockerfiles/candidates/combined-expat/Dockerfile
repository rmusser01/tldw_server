# TASK-13013.7.9: native qualification only; named inputs are verified by CI.
FROM ffmpeg AS runtime-deps
SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
# Inherit the FFmpeg input's signed, date-pinned repositories unchanged.
RUN apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 update \
    && apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 install -y --no-install-recommends \
        libmagic1 libportaudio2 ca-certificates fontconfig fonts-dejavu-core \
    && mkdir -p /opt/combined-evidence/apt-runtime \
    && cp /etc/apt/sources.list.d/debian.sources /opt/combined-evidence/apt-runtime/ \
    && cp /var/lib/apt/lists/*InRelease /opt/combined-evidence/apt-runtime/ \
    && rm -rf /var/lib/apt/lists/*

FROM runtime-deps AS patched-python
COPY --from=qualified /python-install.tar.gz /tmp/qualified-python.tar.gz
COPY --from=qualified /installed-binaries.sha256 /opt/combined-evidence/installed-binaries.sha256
RUN tar -xzf /tmp/qualified-python.tar.gz --no-same-owner -C / \
    && ldconfig \
    && cd / && sha256sum --check --strict /opt/combined-evidence/installed-binaries.sha256 \
    && rm /tmp/qualified-python.tar.gz

FROM patched-python AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12.7@sha256:95f2aa1fe59274951cfe9b0cbc7972e879ff1004bc8945d130a32eb0dbd85945 /uv /uvx /bin/
# Install the same build dependencies as production before replacing system Expat;
# Debian's old development-package exact dependency must not downgrade the fix.
RUN apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 update \
    && apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 install -y --no-install-recommends \
        build-essential python3-dev portaudio19-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ENV UV_PROJECT_ENVIRONMENT=/opt/tldw-venv UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_NO_CACHE=1 UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE /app/
COPY LICENSES /app/LICENSES
COPY tldw_Server_API /app/tldw_Server_API
COPY apps/mcp-unified/src /app/apps/mcp-unified/src
COPY packages/tldw_profile_core /app/packages/tldw_profile_core
RUN uv sync --locked --no-dev --no-editable --python /usr/local/bin/python3.12
COPY Dockerfiles/candidates/combined-expat/test-tools.py /opt/test-tools.py
RUN uv --version | tee /opt/combined-evidence/uv-version.txt \
    && uv --version | grep -Eq '^uv 0\.12\.7( |$)' \
    && python /opt/test-tools.py /app/uv.lock > /opt/combined-evidence/test-tools.requirements.txt \
    && find /opt/tldw-venv -type f -exec sha256sum '{}' + | LC_ALL=C sort > /opt/combined-evidence/runtime-before.sha256 \
    && find /opt/tldw-venv -type l -printf '%p -> %l\n' | LC_ALL=C sort > /opt/combined-evidence/runtime-before.links \
    && uv pip install --python /opt/tldw-venv/bin/python --target /opt/expat-test-tools \
        --no-deps --require-hashes --only-binary :all: \
        --requirement /opt/combined-evidence/test-tools.requirements.txt \
    && find /opt/tldw-venv -type f -exec sha256sum '{}' + | LC_ALL=C sort > /opt/combined-evidence/runtime-after.sha256 \
    && find /opt/tldw-venv -type l -printf '%p -> %l\n' | LC_ALL=C sort > /opt/combined-evidence/runtime-after.links \
    && cmp /opt/combined-evidence/runtime-before.sha256 /opt/combined-evidence/runtime-after.sha256 \
    && cmp /opt/combined-evidence/runtime-before.links /opt/combined-evidence/runtime-after.links \
    && find /opt/expat-test-tools -type f -exec sha256sum '{}' + | LC_ALL=C sort > /opt/combined-evidence/test-tools.sha256

FROM patched-python AS candidate
COPY --from=qualified /libexpat1_2.8.4-1~deb13u1+tldw1_amd64.deb /tmp/qualified-libexpat1.deb
# Keep this package complete despite the slim base's documentation exclusions.
RUN dpkg --path-include='/usr/share/doc/libexpat1/*' --install /tmp/qualified-libexpat1.deb \
    && ldconfig && apt-get check \
    && dpkg --audit > /opt/combined-evidence/dpkg-audit.txt \
    && test ! -s /opt/combined-evidence/dpkg-audit.txt \
    && test "$(dpkg-query -W -f='${Version}' libexpat1)" = '2.8.4-1~deb13u1+tldw1' \
    && dpkg-query -W -f='${binary:Package}=${Version}\n' | LC_ALL=C sort > /opt/combined-evidence/runtime-packages.txt \
    && rm /tmp/qualified-libexpat1.deb
WORKDIR /app
COPY --from=builder /opt/tldw-venv /opt/tldw-venv
COPY --from=builder /opt/expat-test-tools /opt/expat-test-tools
COPY --from=builder /opt/combined-evidence /opt/combined-evidence
COPY --from=qualified /inputs.json /opt/combined-evidence/inputs.json
COPY pyproject.toml uv.lock /app/
COPY tldw_Server_API /app/tldw_Server_API
COPY Helper_Scripts /app/Helper_Scripts
COPY Docs /app/Docs
COPY LICENSE THIRD_PARTY_NOTICES.txt /app/
COPY LICENSES /app/LICENSES
# Separate tracked-test context; production .dockerignore remains unchanged.
COPY --from=tests / /app/
COPY Dockerfiles/candidates/combined-expat /opt/combined
COPY Dockerfiles/candidates/expat/python-controls.py /opt/combined/python-controls.py
RUN useradd -m -u 10001 appuser \
    && mkdir -p /app/Databases && chown appuser:appuser /app/Databases
ENV PATH=/opt/tldw-venv/bin:/opt/tldw-ffmpeg9/bin:$PATH
USER 10001:10001
CMD ["/bin/bash", "/opt/combined/check-runtime.sh"]

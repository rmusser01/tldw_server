FROM python:3.12.14-slim-trixie@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea

SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

RUN rm -f /etc/apt/sources.list /etc/apt/sources.list.d/*
COPY Dockerfiles/candidates/ffmpeg/debian.sources /etc/apt/sources.list.d/debian.sources

RUN mkdir -p /opt/tldw-util-linux/apt-build/lists \
    && apt-get -o Debug::Acquire::gpgv=true update \
        > /opt/tldw-util-linux/apt-build/apt-update-authentication.log 2>&1 \
    && apt-get build-dep -y --no-install-recommends util-linux=2.41.5-0+deb13u1 \
    && apt-get install -y --no-install-recommends \
        binutils \
        ca-certificates \
        git \
        liblastlog2-2 \
        patch \
        quilt \
        xz-utils \
    && cp /etc/apt/sources.list.d/debian.sources /opt/tldw-util-linux/apt-build/ \
    && cp /usr/share/keyrings/debian-archive-keyring.gpg /opt/tldw-util-linux/apt-build/ \
    && find /var/lib/apt/lists -maxdepth 1 -type f \
        \( -name '*InRelease' -o -name '*_source_Sources*' \) \
        -exec cp '{}' /opt/tldw-util-linux/apt-build/lists/ ';' \
    && test "$(find /opt/tldw-util-linux/apt-build/lists -name '*InRelease' | wc -l)" -eq 3 \
    && test "$(find /opt/tldw-util-linux/apt-build/lists -name '*_source_Sources*' | wc -l)" -eq 3 \
    && sha256sum /opt/tldw-util-linux/apt-build/debian-archive-keyring.gpg \
        > /opt/tldw-util-linux/apt-build/debian-archive-keyring.sha256 \
    && groupadd --gid 1000 builder \
    && useradd --uid 1000 --gid 1000 --create-home builder \
    && printf '#!/bin/sh\nexit 101\n' > /usr/sbin/policy-rc.d \
    && chmod 0755 /usr/sbin/policy-rc.d \
    && mkdir -p /candidate /work \
    && chown 1000:1000 /work

COPY --chmod=0755 Dockerfiles/candidates/util-linux/qualify.sh /opt/tldw-util-linux/qualify.sh

ENTRYPOINT ["/opt/tldw-util-linux/qualify.sh"]

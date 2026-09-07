FROM python:3.12.14-slim-trixie@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea AS system

SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
RUN rm -f /etc/apt/sources.list /etc/apt/sources.list.d/*
COPY Dockerfiles/candidates/ffmpeg/debian.sources /etc/apt/sources.list.d/debian.sources
RUN mkdir -p /opt/expat/apt /work /candidate \
    && apt-get -o Debug::Acquire::gpgv=true update > /opt/expat/apt/authentication.log 2>&1 \
    && apt-get build-dep -y --no-install-recommends expat=2.8.3-1~deb13u1 \
    && apt-get install -y --no-install-recommends \
       build-essential autoconf automake libtool pkg-config cmake \
       curl ca-certificates gnupg git fakeroot binutils libexpat1-dev \
    && cp /etc/apt/sources.list.d/debian.sources /opt/expat/apt/ \
    && cp /usr/share/keyrings/debian-archive-keyring.gpg /opt/expat/apt/ \
    && find /var/lib/apt/lists -maxdepth 1 -name '*InRelease' -exec cp '{}' /opt/expat/apt/ ';' \
    && groupadd --gid 1000 builder \
    && useradd --uid 1000 --gid 1000 --create-home builder \
    && chown 1000:1000 /work
COPY Helper_Scripts/Supply_Chain/expat_candidate.py /opt/expat/expat_candidate.py
COPY Dockerfiles/candidates/expat/wide-controls.c /opt/expat/wide-controls.c
COPY Dockerfiles/candidates/expat/attribute-scaling.py /opt/expat/attribute-scaling.py
COPY --chmod=0755 Dockerfiles/candidates/expat/qualify.sh /opt/expat/qualify.sh
USER 1000:1000
RUN /opt/expat/qualify.sh fetch
ENTRYPOINT ["/opt/expat/qualify.sh"]

FROM system AS python
USER 0:0
RUN apt-get install -y --no-install-recommends \
    libbluetooth-dev libbz2-dev libdb-dev libffi-dev libgdbm-dev \
    liblzma-dev libncursesw5-dev libreadline-dev libsqlite3-dev libssl-dev \
    tk-dev uuid-dev xz-utils zlib1g-dev \
    && dpkg-query -W > /opt/expat/apt/python-build-packages.txt
COPY Dockerfiles/candidates/expat/python-source.py Dockerfiles/candidates/expat/python-controls.py Dockerfiles/candidates/expat/python-suite.py /opt/expat/
COPY --chmod=0755 Dockerfiles/candidates/expat/offline-bin/curl /opt/expat/offline-bin/curl
COPY --chmod=0755 Dockerfiles/candidates/expat/python-qualify.sh /opt/expat/python-qualify.sh
USER 1000:1000
ENTRYPOINT ["/opt/expat/python-qualify.sh"]

# Candidate-only FFmpeg 9 build for TASK-13013.7.6. This is not a production image.
FROM python:3.12.14-slim-trixie@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea AS snapshot-base

SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
RUN rm -f /etc/apt/sources.list /etc/apt/sources.list.d/*
COPY Dockerfiles/candidates/ffmpeg/debian.sources /etc/apt/sources.list.d/debian.sources

FROM snapshot-base AS build-deps

ARG BUILD_JOBS=4

RUN apt-get update \
    && apt-get build-dep -y --no-install-recommends ffmpeg \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        file \
        glslang-tools \
        gnupg \
        libfast-float-dev \
        libxxhash-dev \
        meson \
        ninja-build \
        python3-glad \
        python3-jinja2 \
    && evidence_dir=/opt/tldw-ffmpeg9/share/candidate/apt-build \
    && mkdir -p /opt/tldw-ffmpeg9/share/candidate/apt-build \
    && cp /etc/apt/sources.list.d/debian.sources "$evidence_dir/" \
    && for suite in trixie trixie-updates trixie-security; do \
        metadata=(/var/lib/apt/lists/*_dists_"${suite}"_InRelease); \
        test "${#metadata[@]}" -eq 1; \
        test -f "${metadata[0]}"; \
        cp -- "${metadata[0]}" "$evidence_dir/"; \
    done \
    && rm -rf /var/lib/apt/lists/*

FROM build-deps AS build

WORKDIR /tmp/sources
RUN curl --fail --location --proto '=https' --tlsv1.2 --output ffmpeg.tar.xz \
        https://ffmpeg.org/releases/ffmpeg-9.0.1.tar.xz \
    && curl --fail --location --proto '=https' --tlsv1.2 --output ffmpeg.tar.xz.asc \
        https://ffmpeg.org/releases/ffmpeg-9.0.1.tar.xz.asc \
    && curl --fail --location --proto '=https' --tlsv1.2 --output ffmpeg-devel.asc \
        https://ffmpeg.org/ffmpeg-devel.asc \
    && printf '%s  %s\n' \
        b613a00005232a1245ace7080088781ac23a916119d3e5b0d6c042368eee0177 ffmpeg.tar.xz.asc \
        397b3becedcd5a98769967ff1ff8501ddc89f8368b8f766e4701377d7dbaabe5 ffmpeg-devel.asc \
        | sha256sum --check --strict \
    && export GNUPGHOME="$(mktemp -d)" \
    && trap 'rm -rf "$GNUPGHOME"' EXIT \
    && gpg --batch --import ffmpeg-devel.asc \
    && gpg --batch --with-colons --fingerprint FCF986EA15E6E293A5644F10B4322F04D67658D8 \
        > ffmpeg-key-fingerprint.txt \
    && grep -Fqx 'fpr:::::::::FCF986EA15E6E293A5644F10B4322F04D67658D8:' ffmpeg-key-fingerprint.txt \
    && gpg --batch --status-fd 1 --verify ffmpeg.tar.xz.asc ffmpeg.tar.xz \
        > ffmpeg-signature-status.txt \
    && grep -Fq '[GNUPG:] VALIDSIG FCF986EA15E6E293A5644F10B4322F04D67658D8 ' ffmpeg-signature-status.txt \
    && printf '%s  %s\n' \
        cf38e0e28c7e5605942c4a77755349b0145804a397af37eb1fb4c77cb237f635 ffmpeg.tar.xz \
        | sha256sum --check --strict \
    && curl --fail --location --proto '=https' --tlsv1.2 --output libplacebo.tar.gz \
        https://codeload.github.com/haasn/libplacebo/tar.gz/cee9b076f2c63104ccfd497fa79c39a867293ec4 \
    && curl --fail --location --proto '=https' --tlsv1.2 --output vulkan-headers.tar.gz \
        https://codeload.github.com/KhronosGroup/Vulkan-Headers/tar.gz/450bd2232225d6c7728a4108055ac2e37cef6475 \
    && printf '%s  %s\n' \
        6f8fa218cbafd8e5f50b8a82d918e1d8bbb92f9f980820bc0b34d92e9b79e484 libplacebo.tar.gz \
        26df9841c30806a994e2fdf42f7c87bcb1ced9db9a06033469123939fb3fa075 vulkan-headers.tar.gz \
        | sha256sum --check --strict

RUN mkdir /tmp/ffmpeg /tmp/libplacebo \
    && tar --extract --file ffmpeg.tar.xz --directory /tmp/ffmpeg --strip-components=1 \
    && tar --extract --file libplacebo.tar.gz --directory /tmp/libplacebo --strip-components=1 \
    && mkdir -p /tmp/libplacebo/3rdparty/Vulkan-Headers \
    && tar --extract --file vulkan-headers.tar.gz \
        --directory /tmp/libplacebo/3rdparty/Vulkan-Headers --strip-components=1

RUN PATH=/usr/bin:/bin /usr/bin/meson setup /tmp/libplacebo-build /tmp/libplacebo \
        --buildtype=release \
        --default-library=shared \
        --prefix=/opt/tldw-ffmpeg9 \
        --libdir=lib \
        --wrap-mode=nodownload \
        -Ddemos=false \
        -Dglslang=enabled \
        -Dlcms=enabled \
        -Dopengl=enabled \
        -Dvulkan=enabled \
    && PATH=/usr/bin:/bin /usr/bin/meson compile -C /tmp/libplacebo-build -j "$BUILD_JOBS" \
    && PATH=/usr/bin:/bin /usr/bin/meson install -C /tmp/libplacebo-build

WORKDIR /tmp/ffmpeg
RUN PKG_CONFIG_PATH=/opt/tldw-ffmpeg9/lib/pkgconfig \
    LD_LIBRARY_PATH=/opt/tldw-ffmpeg9/lib \
    ./configure \
        --prefix=/opt/tldw-ffmpeg9 \
        --libdir=/opt/tldw-ffmpeg9/lib \
        --incdir=/opt/tldw-ffmpeg9/include \
        --enable-rpath \
        --enable-gpl \
        --disable-stripping \
        --disable-libmfx \
        --enable-gnutls \
        --enable-libaom \
        --enable-libass \
        --enable-libbs2b \
        --enable-libcdio \
        --enable-libcodec2 \
        --enable-libdav1d \
        --enable-libflite \
        --enable-libfontconfig \
        --enable-libfreetype \
        --enable-libfribidi \
        --enable-libgme \
        --enable-libgsm \
        --enable-libharfbuzz \
        --enable-libmp3lame \
        --enable-libmysofa \
        --enable-libopenjpeg \
        --enable-libopenmpt \
        --enable-libopus \
        --enable-librubberband \
        --enable-libshine \
        --enable-libsnappy \
        --enable-libsoxr \
        --enable-libspeex \
        --enable-libtheora \
        --enable-libtwolame \
        --enable-libvidstab \
        --enable-libvorbis \
        --enable-libvpx \
        --enable-libwebp \
        --enable-libx265 \
        --enable-libxml2 \
        --enable-libxvid \
        --enable-libzimg \
        --enable-openal \
        --enable-opencl \
        --enable-opengl \
        --disable-sndio \
        --enable-libvpl \
        --enable-libdc1394 \
        --enable-libdrm \
        --enable-libiec61883 \
        --enable-chromaprint \
        --enable-frei0r \
        --enable-ladspa \
        --enable-libbluray \
        --enable-libcaca \
        --enable-libdvdnav \
        --enable-libdvdread \
        --enable-libjack \
        --enable-libpulse \
        --enable-librabbitmq \
        --enable-librist \
        --enable-libsrt \
        --enable-libssh \
        --enable-libsvtav1 \
        --enable-libx264 \
        --enable-libzmq \
        --enable-libzvbi \
        --enable-lv2 \
        --enable-sdl2 \
        --enable-libplacebo \
        --enable-librav1e \
        --enable-pocketsphinx \
        --enable-librsvg \
        --enable-libjxl \
        --enable-shared \
    && make -j"$BUILD_JOBS" \
    && make install \
    && test -x /opt/tldw-ffmpeg9/bin/ffmpeg \
    && test -x /opt/tldw-ffmpeg9/bin/ffprobe \
    && test -x /opt/tldw-ffmpeg9/bin/ffplay

RUN mkdir -p /opt/tldw-ffmpeg9/share/candidate \
    && cp /tmp/sources/ffmpeg-signature-status.txt /opt/tldw-ffmpeg9/share/candidate/ \
    && /opt/tldw-ffmpeg9/bin/ffmpeg -hide_banner -buildconf \
        > /opt/tldw-ffmpeg9/share/candidate/buildconf.txt \
    && dpkg-query -W -f='${binary:Package}=${Version}\n' \
        | sort -u > /opt/tldw-ffmpeg9/share/candidate/build-packages.txt \
    && printf '%s\n' \
        '{' \
        '  "ffmpeg": {' \
        '    "version": "9.0.1",' \
        '    "archive_url": "https://ffmpeg.org/releases/ffmpeg-9.0.1.tar.xz",' \
        '    "archive_sha256": "cf38e0e28c7e5605942c4a77755349b0145804a397af37eb1fb4c77cb237f635",' \
        '    "signature_sha256": "b613a00005232a1245ace7080088781ac23a916119d3e5b0d6c042368eee0177",' \
        '    "key_sha256": "397b3becedcd5a98769967ff1ff8501ddc89f8368b8f766e4701377d7dbaabe5",' \
        '    "signing_fingerprint": "FCF986EA15E6E293A5644F10B4322F04D67658D8",' \
        '    "verification": "local detached RSA signature verification in a fresh GnuPG home"' \
        '  },' \
        '  "libplacebo": {' \
        '    "version": "7.360.1",' \
        '    "signed_tag_object": "719cc95244a1f1d648dd72459822e026e6530f22",' \
        '    "commit": "cee9b076f2c63104ccfd497fa79c39a867293ec4",' \
        '    "archive_sha256": "6f8fa218cbafd8e5f50b8a82d918e1d8bbb92f9f980820bc0b34d92e9b79e484",' \
        '    "verification": "GitHub API reported the upstream maintainer tag signature valid; the commit archive is locally hash-verified"' \
        '  },' \
        '  "vulkan_headers": {' \
        '    "commit": "450bd2232225d6c7728a4108055ac2e37cef6475",' \
        '    "archive_sha256": "26df9841c30806a994e2fdf42f7c87bcb1ced9db9a06033469123939fb3fa075",' \
        '    "purpose": "exact libplacebo gitlink build input; not a runtime library"' \
        '  }' \
        '}' > /opt/tldw-ffmpeg9/share/candidate/source-identities.json

RUN mapfile -t candidate_elfs < <( \
        find /opt/tldw-ffmpeg9/bin -type f -perm /111 -print; \
        find /opt/tldw-ffmpeg9/lib -type f -name '*.so*' -print \
    ) \
    && test "${#candidate_elfs[@]}" -gt 0 \
    && : > /tmp/candidate-ldd.txt \
    && for candidate_elf in "${candidate_elfs[@]}"; do \
        if file "$candidate_elf" | grep -Fq 'ELF'; then \
            ldd "$candidate_elf" >> /tmp/candidate-ldd.txt; \
        fi; \
    done \
    && ! grep -Fq 'not found' /tmp/candidate-ldd.txt \
    && ! grep -Eq 'lib(avcodec\.so\.61|avdevice\.so\.61|avfilter\.so\.10|avformat\.so\.61|avutil\.so\.59|postproc\.so\.58|swresample\.so\.5|swscale\.so\.8)' /tmp/candidate-ldd.txt \
    && awk '/ => \// {print $(NF-1)} /^\// {print $1}' /tmp/candidate-ldd.txt \
        | sort -u > /tmp/runtime-library-paths.txt \
    && awk '$0 ~ /^\/opt\/tldw-ffmpeg9\//' /tmp/runtime-library-paths.txt \
        > /opt/tldw-ffmpeg9/share/candidate/source-built-libraries.txt \
    && : > /tmp/runtime-packages.txt \
    && while IFS= read -r library; do \
        canonical="$(readlink -f "$library")"; \
        if [[ "$canonical" == /opt/tldw-ffmpeg9/* ]]; then continue; fi; \
        owner="$(dpkg-query -S "$canonical" | head -n 1)"; \
        package="${owner%: *}"; \
        version="$(dpkg-query -W -f='${Version}' "$package")"; \
        printf '%s=%s\n' "$package" "$version" >> /tmp/runtime-packages.txt; \
    done < /tmp/runtime-library-paths.txt \
    && sort -u -o /tmp/runtime-packages.txt /tmp/runtime-packages.txt \
    && test -s /tmp/runtime-packages.txt \
    && ! grep -Eq '^(ffmpeg|libavcodec61|libavdevice61|libavfilter10|libavformat61|libavutil59|libpostproc58|libswresample5|libswscale8)(:.*)?=' /tmp/runtime-packages.txt \
    && cp /tmp/candidate-ldd.txt /opt/tldw-ffmpeg9/share/candidate/build-ldd.txt \
    && cp /tmp/runtime-packages.txt /opt/tldw-ffmpeg9/share/candidate/required-runtime-packages.txt

FROM snapshot-base AS candidate

ENV PATH=/opt/tldw-ffmpeg9/bin:$PATH \
    LD_LIBRARY_PATH=/opt/tldw-ffmpeg9/lib

COPY --from=build /opt/tldw-ffmpeg9 /opt/tldw-ffmpeg9

RUN apt-get update \
    && xargs -r apt-get install -y --no-install-recommends \
        < /opt/tldw-ffmpeg9/share/candidate/required-runtime-packages.txt \
    && evidence_dir=/opt/tldw-ffmpeg9/share/candidate/apt-runtime \
    && mkdir -p /opt/tldw-ffmpeg9/share/candidate/apt-runtime \
    && cp /etc/apt/sources.list.d/debian.sources "$evidence_dir/" \
    && for suite in trixie trixie-updates trixie-security; do \
        metadata=(/var/lib/apt/lists/*_dists_"${suite}"_InRelease); \
        test "${#metadata[@]}" -eq 1; \
        test -f "${metadata[0]}"; \
        cp -- "${metadata[0]}" "$evidence_dir/"; \
    done \
    && rm -rf /var/lib/apt/lists/* \
    && test "$(dpkg-query -W -f='${db:Status-Status}' ffmpeg 2>/dev/null || true)" != installed \
    && mapfile -t candidate_elfs < <( \
        find /opt/tldw-ffmpeg9/bin -type f -perm /111 -print; \
        find /opt/tldw-ffmpeg9/lib -type f -name '*.so*' -print \
    ) \
    && : > /opt/tldw-ffmpeg9/share/candidate/runtime-ldd.txt \
    && for candidate_elf in "${candidate_elfs[@]}"; do \
        ldd "$candidate_elf" >> /opt/tldw-ffmpeg9/share/candidate/runtime-ldd.txt; \
    done \
    && ! grep -Fq 'not found' /opt/tldw-ffmpeg9/share/candidate/runtime-ldd.txt \
    && ! grep -Eq 'lib(avcodec\.so\.61|avdevice\.so\.61|avfilter\.so\.10|avformat\.so\.61|avutil\.so\.59|postproc\.so\.58|swresample\.so\.5|swscale\.so\.8)' /opt/tldw-ffmpeg9/share/candidate/runtime-ldd.txt \
    && while IFS='=' read -r package expected; do \
        actual="$(dpkg-query -W -f='${Version}' "$package")"; \
        test "$actual" = "$expected"; \
        printf '%s=%s\n' "$package" "$actual"; \
    done < /opt/tldw-ffmpeg9/share/candidate/required-runtime-packages.txt \
        > /opt/tldw-ffmpeg9/share/candidate/installed-runtime-packages.txt \
    && ffmpeg -version \
    && ffprobe -version \
    && ffplay -version

CMD ["ffmpeg", "-version"]

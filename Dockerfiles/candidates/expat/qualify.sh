#!/bin/bash
set -Eeuo pipefail

readonly VERSION='2.8.4-1~deb13u1+tldw1'
readonly SOURCE=/work/expat-2.8.4

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

verify_evidence() {
    local evidence="$1" phase="$2" file
    test -f "$evidence/phase.exit" && test "$(< "$evidence/phase.exit")" = 0 || die "missing/failed phase status: $phase"
    test -f "$evidence/complete.txt" && test "$(< "$evidence/complete.txt")" = "$phase" || die "incomplete phase: $phase"
    case "$phase" in
        prepare) test -s "$evidence/authentication.json" ;;
        build)
            for file in parser-verbose.log wide-controls.log scaling-comparison.log abi.txt artifacts/SHA256SUMS; do
                test -s "$evidence/$file" || die "missing build evidence: $file"
            done
            grep -Fxq 'PASS: test_default_attr_index_after_dtd_copy' "$evidence/parser-verbose.log"
            grep -Fxq 'PASS: wide XML controls' "$evidence/wide-controls.log"
            grep -Fxq 'PASS: bounded whole/incremental attribute scaling' "$evidence/scaling-comparison.log"
            test "$(< "$evidence/abi.txt")" = compatible
            ;;
        sanitize) grep -Fq '100% tests passed, 0 tests failed' "$evidence/sanitizer-tests.log" ;;
        install)
            for file in versions.txt apt-check.log; do test -s "$evidence/$file"; done
            test -f "$evidence/dpkg-audit.txt" && ! test -s "$evidence/dpkg-audit.txt"
            ;;
        *) die "unknown phase: $phase" ;;
    esac
}

controller() {
    local evidence="$1" phase container status image arch prepared base
    mkdir "$evidence"
    mkdir "$evidence/identity"
    printf 'commit=%s\nkernel=%s\narch=%s\n' "$(git rev-parse HEAD)" "$(uname -s)" "$(uname -m)" > "$evidence/identity/runner.txt"
    [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die 'requires native Linux x86_64'
    arch="$(docker info --format '{{.Architecture}}')"
    printf 'daemon_arch=%s\n' "$arch" >> "$evidence/identity/runner.txt"
    [[ "$arch" == x86_64 || "$arch" == amd64 ]] || die 'requires native amd64 Docker daemon'
    if [[ -n "${GITHUB_SHA:-}" ]]; then
        [[ "$(git rev-parse HEAD)" == "$GITHUB_SHA" ]] || die 'checkout differs from requested commit'
    fi
    docker build --platform linux/amd64 --file Dockerfiles/candidates/expat/Dockerfile \
        --tag "tldw-expat-system-base:${GITHUB_RUN_ID}" . 2>&1 | tee "$evidence/image-build.log"
    base="$(docker image inspect --format '{{.Id}}' "tldw-expat-system-base:${GITHUB_RUN_ID}")"
    arch="$(docker image inspect --format '{{.Architecture}}' "$base")"
    printf 'base=%s\nbase_arch=%s\n' "$base" "$arch" >> "$evidence/identity/runner.txt"
    [[ "$arch" == amd64 ]] || die 'image is not native amd64'
    prepared="$base"
    for phase in prepare build sanitize install; do
        image="$prepared"
        local user=1000:1000
        local security=(--security-opt no-new-privileges:true --cap-drop ALL)
        if [[ "$phase" == install ]]; then
            image="$base"; user=0:0
            # Debian installation needs its ordinary container capabilities,
            # never privileged mode, host mounts or a host Docker socket.
            security=(--security-opt no-new-privileges:true)
        fi
        container="$(docker create --name "tldw-expat-${phase}-${GITHUB_RUN_ID}" \
            --network none "${security[@]}" \
            --user "$user" --cpus 4 --memory 5g --pids-limit 512 "$image" "$phase")"
        if [[ "$phase" == install ]]; then
            docker cp "$evidence/build/artifacts/." "$container:/candidate/"
        fi
        set +e
        docker start --attach "$container"
        status=$?
        set -e
        mkdir -p "$evidence/$phase"
        docker cp "$container:/work/evidence/$phase/." "$evidence/$phase/"
        printf '%s\n' "$status" > "$evidence/$phase/container.exit"
        if (( status != 0 )); then exit "$status"; fi
        verify_evidence "$evidence/$phase" "$phase"
        if [[ "$phase" == prepare ]]; then
            docker commit "$container" "tldw-expat-prepared:${GITHUB_RUN_ID}" > /dev/null
            prepared="$(docker image inspect --format '{{.Id}}' "tldw-expat-prepared:${GITHUB_RUN_ID}")"
            [[ "$(docker image inspect --format '{{.Architecture}}' "$prepared")" == amd64 ]] || die 'prepared image is not native amd64'
            printf 'prepared=%s\n' "$prepared" >> "$evidence/identity/runner.txt"
        fi
        docker rm "$container" > /dev/null
    done
    printf 'System-only qualification; Python bundled parser remains unqualified.\n' > "$evidence/system-qualified.txt"
}

fetch() {
    mkdir /work/downloads
    local url filename fingerprint key
    for filename in Python-3.12.14.tar.xz Python-3.12.14.tar.xz.asc; do
        curl --fail --location --retry 3 --max-time 180 "https://www.python.org/ftp/python/3.12.14/$filename" -o "/work/downloads/$filename"
    done
    for filename in expat-2.8.4.tar.gz expat-2.8.4.tar.gz.asc; do
        curl --fail --location --retry 3 --max-time 180 "https://github.com/libexpat/libexpat/releases/download/R_2_8_4/$filename" -o "/work/downloads/$filename"
    done
    for filename in expat_2.8.4-1.dsc expat_2.8.4.orig.tar.gz expat_2.8.4-1.debian.tar.xz; do
        curl --fail --location --retry 3 --max-time 180 "https://deb.debian.org/debian/pool/main/e/expat/$filename" -o "/work/downloads/$filename"
    done
    for key in \
        expat-key.asc:3176EF7DB2367F1FCA4F306B1F9B0E909AF37285 \
        python-key.asc:7169605F62C751356D054A26A821E680E5FA6305 \
        debian-maintainer-full-key.asc:A0DF7E0D3851E0EE45C00BC8ACE1F33CB933BBBB; do
        filename="${key%%:*}"; fingerprint="${key#*:}"
        url="https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x${fingerprint}"
        curl --fail --location --retry 3 --max-time 180 "$url" -o "/work/downloads/$filename"
    done
    python /opt/expat/expat_candidate.py verify-sources /work/downloads
    chmod a-w /work/downloads/* /work/downloads
}

run_step() {
    local name="$1"; shift
    local statuses
    set +e
    (set -e; "$@") 2>&1 | tee "$EVIDENCE/$name.log"
    statuses=("${PIPESTATUS[@]}")
    set -e
    printf '%s\n' "${statuses[0]}" > "$EVIDENCE/$name.exit"
    (( statuses[0] == 0 )) || return "${statuses[0]}"
    (( statuses[1] == 0 )) || return "${statuses[1]}"
}

record_abi() {
    local library="$1" destination="$2"
    mkdir -p "$destination"
    readelf -d "$library" | sed -n 's/.*Library soname: \[\(.*\)\]/\1/p' > "$destination/soname"
    nm -D --defined-only --with-symbol-versions "$library" | awk 'NF >= 3 {print $2, $3}' | LC_ALL=C sort -u > "$destination/symbols"
    test -s "$destination/soname" && test -s "$destination/symbols"
    sha256sum "$library" > "$destination/sha256"
}

prepare() {
    python /opt/expat/expat_candidate.py authenticate-sources /work/downloads \
        --public-keys /work/downloads --evidence "$EVIDENCE/authentication"
    cp "$EVIDENCE/authentication/authentication.json" "$EVIDENCE/"
    cp -a /opt/expat/apt "$EVIDENCE/"
    # dpkg-source copies its authenticated input archives into /work itself.
    # Pre-copying a read-only download makes that destination unwritable.
    run_step source-extract dpkg-source -x /work/downloads/expat_2.8.4-1.dsc "$SOURCE"
    run_step upstream-extract tar -xzf /work/downloads/expat-2.8.4.tar.gz -C /work --one-top-level=upstream
    {
        printf 'expat (%s) UNRELEASED; urgency=medium\n\n' "$VERSION"
        printf '  * Candidate-only complete Expat 2.8.4 rebuild on signed Trixie snapshot.\n\n'
        printf ' -- tldw_server CI <noreply@tldw.local>  Mon, 07 Sep 2026 00:00:00 +0000\n\n'
        cat "$SOURCE/debian/changelog"
    } > /work/changelog.candidate
    mv /work/changelog.candidate "$SOURCE/debian/changelog"
    local stem
    for stem in libexpat libexpatw; do
        record_abi "/usr/lib/x86_64-linux-gnu/$stem.so.1" "/work/baseline/$stem"
    done
    cp -a /work/baseline "$EVIDENCE/abi-baseline"
    dpkg-query -W > "$EVIDENCE/packages.txt"
    python -c 'import pyexpat; print(pyexpat.EXPAT_VERSION)' > "$EVIDENCE/python-baseline.txt"
}

build() {
    cd "$SOURCE"
    run_step source-package env DEB_BUILD_OPTIONS=parallel=4 dpkg-buildpackage -S -us -uc
    run_step binary-package env DEB_BUILD_OPTIONS=parallel=4 dpkg-buildpackage -b -us -uc
    mkdir "$EVIDENCE/artifacts"
    find /work -maxdepth 1 -type f \( -name '*.deb' -o -name '*.udeb' -o -name '*.dsc' -o -name '*.tar.*' -o -name '*.buildinfo' -o -name '*.changes' \) \
        -exec cp '{}' "$EVIDENCE/artifacts/" ';'
    (cd "$EVIDENCE/artifacts"; sha256sum ./* > SHA256SUMS)
    run_step parser-tests make -C build check
    run_step parser-verbose build/tests/runtests -v
    grep -Fxq 'PASS: test_default_attr_index_after_dtd_copy' "$EVIDENCE/parser-verbose.log"
    mkdir /work/installed-package-tree
    local package stem library
    for package in "$EVIDENCE/artifacts"/*.deb; do
        test "$(dpkg-deb -f "$package" Version)" = "$VERSION"
        dpkg-deb -x "$package" /work/installed-package-tree
    done
    local libdir=/work/installed-package-tree/usr/lib/x86_64-linux-gnu
    for stem in libexpat libexpatw; do
        library="$libdir/$stem.so.1"
        record_abi "$library" "$EVIDENCE/abi/$stem"
        cmp "/work/baseline/$stem/soname" "$EVIDENCE/abi/$stem/soname"
        LC_ALL=C comm -23 "/work/baseline/$stem/symbols" "$EVIDENCE/abi/$stem/symbols" > "$EVIDENCE/abi/$stem/missing"
        test ! -s "$EVIDENCE/abi/$stem/missing"
        ldd "$library" > "$EVIDENCE/abi/$stem/ldd"
        ! grep -Fq 'not found' "$EVIDENCE/abi/$stem/ldd"
    done
    printf 'compatible\n' > "$EVIDENCE/abi.txt"
    run_step wide-compile cc -std=c99 -Wall -Wextra -Werror -DXML_UNICODE \
        -I"$SOURCE/src/lib" /opt/expat/wide-controls.c -L"$libdir" -lexpatw -o /work/wide-controls
    run_step wide-controls env LD_LIBRARY_PATH="$libdir" /work/wide-controls expat_2.8.4
    run_step scaling-baseline python /opt/expat/attribute-scaling.py measure /usr/lib/x86_64-linux-gnu/libexpat.so.1
    run_step scaling-candidate python /opt/expat/attribute-scaling.py measure "$libdir/libexpat.so.1"
    run_step scaling-comparison python /opt/expat/attribute-scaling.py compare \
        "$EVIDENCE/scaling-baseline.log" "$EVIDENCE/scaling-candidate.log"
}

sanitize() {
    run_step sanitizer-configure cmake -S /work/upstream/expat-2.8.4 -B /work/sanitized \
        -DEXPAT_BUILD_DOCS=OFF -DEXPAT_BUILD_EXAMPLES=OFF -DEXPAT_SHARED_LIBS=OFF \
        -DEXPAT_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Debug \
        '-DCMAKE_C_FLAGS=-fsanitize=address,undefined -fno-omit-frame-pointer' \
        '-DCMAKE_CXX_FLAGS=-fsanitize=address,undefined -fno-omit-frame-pointer' \
        '-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address,undefined'
    run_step sanitizer-build cmake --build /work/sanitized --parallel 4
    run_step sanitizer-tests env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
        ctest --test-dir /work/sanitized --output-on-failure --timeout 300
}

install_candidate() {
    cd /candidate
    sha256sum -c SHA256SUMS
    local package
    for package in ./*.deb; do test "$(dpkg-deb -f "$package" Version)" = "$VERSION"; done
    run_step apt-install apt-get install -y --no-download --no-install-recommends ./*.deb
    run_step apt-check apt-get check
    dpkg --audit > "$EVIDENCE/dpkg-audit.txt"
    test ! -s "$EVIDENCE/dpkg-audit.txt"
    dpkg-query -W libexpat1 libexpat1-dev expat > "$EVIDENCE/packages.txt"
    python -c 'import ctypes
for name in ("libexpat.so.1", "libexpatw.so.1"):
    lib = ctypes.CDLL(name)
    lib.XML_ExpatVersion.restype = ctypes.c_char_p
    version = lib.XML_ExpatVersion().decode()
    if version != "expat_2.8.4": raise SystemExit("wrong installed Expat version")
    print(name, version)' > "$EVIDENCE/versions.txt"
    run_step wide-compile cc -std=c99 -Wall -Wextra -Werror -DXML_UNICODE /opt/expat/wide-controls.c -lexpatw -o /work/wide-installed
    run_step wide-controls /work/wide-installed expat_2.8.4
    python -c 'import pyexpat; print("Unmodified bundled Python parser:", pyexpat.EXPAT_VERSION)' > "$EVIDENCE/python-not-qualified.txt"
}

case "${1:-}" in
    controller) controller "$2" ;;
    verify-evidence) verify_evidence "$2" "$3" ;;
    fetch) fetch ;;
    prepare|build|sanitize|install)
        phase="$1"
        [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die 'requires native Linux x86_64'
        if [[ "$phase" == install ]]; then test "$(id -u)" = 0; else test "$(id -u)" = 1000; fi
        export EVIDENCE="/work/evidence/$phase"
        mkdir -p /work/evidence
        mkdir "$EVIDENCE"
        trap 'status=$?; printf "%s\n" "$status" > "$EVIDENCE/phase.exit"' EXIT
        if [[ "$phase" == install ]]; then install_candidate; else "$phase"; fi
        printf '%s\n' "$phase" > "$EVIDENCE/complete.txt"
        ;;
    *) die 'expected controller, verify-evidence, fetch, prepare, build, sanitize or install' ;;
esac

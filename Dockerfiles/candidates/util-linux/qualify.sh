#!/bin/bash
set -Eeuo pipefail

readonly DEBIAN_SOURCE_VERSION="2.41.5-0+deb13u1"
readonly CANDIDATE_VERSION="2.41.5-0+deb13u1+tldw1"
readonly UPSTREAM_COMMIT="1d14676ea70003e9f5b2a6a76af0cadb1190411a"
readonly UPSTREAM_PATCH_SHA256="678cf342348f559e039db18d7e022a71f66d326590ba3e3c558651f30a967096"
readonly EVIDENCE="/work/evidence"
readonly SOURCE_TREE="/work/util-linux-2.41.5"
readonly PATCH_NAME="libmount-skip-post-mount-hooks-after-failed-mount-helper.patch"
readonly LIBRARIES=(libmount libblkid libuuid libsmartcols liblastlog2)

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

verify_sources() {
    local source_dir="$1"
    (
        cd "$source_dir"
        printf '%s  %s\n' \
            43e9b2cbebd10fdc598c4ad10217c8202c28de53af6eafc892c8e9b5cbf3a3a5 "util-linux_${DEBIAN_SOURCE_VERSION}.dsc" \
            f586e35d320ff537aab3ffeca37e9ecd482ccbe013590db4429a414d8aa6a728 util-linux_2.41.5.orig.tar.xz \
            5b327ccd22f0f4ed28a389870aa51d04ecedb8693e52a1d122850f2b3188cbf6 "util-linux_${DEBIAN_SOURCE_VERSION}.debian.tar.xz" \
            | sha256sum -c -
    )
}

record_library() {
    local package="$1"
    local stem="$2"
    local output="$3"
    local library resolved
    library="$(dpkg-query -L "$package" | awk -v stem="$stem" '$0 ~ "/" stem "\\.so\\.[0-9.]+$" { print; exit }')"
    test -n "$library"
    resolved="$(readlink -f "$library")"
    mkdir -p "$output"
    printf '%s\n' "$package|$library|$resolved" > "$output/path.txt"
    sha256sum "$resolved" > "$output/library.sha256"
    readelf -d "$resolved" | sed -n 's/.*Library soname: \[\(.*\)\]/\1/p' > "$output/soname.txt"
    test -s "$output/soname.txt"
    nm -D --defined-only --with-symbol-versions "$resolved" \
        | awk 'NF >= 3 && $3 ~ /@/ { print $2, $3 }' \
        | LC_ALL=C sort -u > "$output/exported-symbols.txt"
    test -s "$output/exported-symbols.txt"
}

record_candidate_library() {
    local root="$1"
    local stem="$2"
    local output="$3"
    local library
    library="$(find "$root" -type f -name "${stem}.so.*" -print | head -n 1)"
    test -n "$library"
    mkdir -p "$output"
    printf '%s\n' "$library" > "$output/path.txt"
    sha256sum "$library" > "$output/library.sha256"
    readelf -d "$library" | sed -n 's/.*Library soname: \[\(.*\)\]/\1/p' > "$output/soname.txt"
    test -s "$output/soname.txt"
    nm -D --defined-only --with-symbol-versions "$library" \
        | awk 'NF >= 3 && $3 ~ /@/ { print $2, $3 }' \
        | LC_ALL=C sort -u > "$output/exported-symbols.txt"
    test -s "$output/exported-symbols.txt"
}

compare_abi() {
    local baseline="$1"
    local candidate="$2"
    local output="$3"
    local library status=0
    mkdir -p "$output"
    for library in "${LIBRARIES[@]}"; do
        for required in soname.txt exported-symbols.txt; do
            test -s "$baseline/$library/$required" || die "missing baseline ABI file: $library/$required"
            test -s "$candidate/$library/$required" || die "missing candidate ABI file: $library/$required"
        done
        if ! cmp -s "$baseline/$library/soname.txt" "$candidate/$library/soname.txt"; then
            diff -u "$baseline/$library/soname.txt" "$candidate/$library/soname.txt" \
                > "$output/$library-soname.diff" || true
            status=1
        fi
        LC_ALL=C comm -23 \
            <(LC_ALL=C sort -u "$baseline/$library/exported-symbols.txt") \
            <(LC_ALL=C sort -u "$candidate/$library/exported-symbols.txt") \
            > "$output/$library-missing-symbols.txt"
        if test -s "$output/$library-missing-symbols.txt"; then
            status=1
        fi
    done
    if (( status != 0 )); then
        printf 'incompatible: changed SONAME or missing baseline public export\n' > "$output/comparison.txt"
        return "$status"
    fi
    printf 'compatible: candidate retains all baseline SONAMEs and public versioned exports\n' \
        > "$output/comparison.txt"
}

copy_outputs() {
    local source_output="$EVIDENCE/artifacts/source"
    local binary_output="$EVIDENCE/artifacts/binary"
    mkdir -p "$source_output" "$binary_output"
    find /work -maxdepth 1 -type f \
        \( -name '*.dsc' -o -name '*.tar.*' -o -name '*_source.buildinfo' -o -name '*_source.changes' \) \
        -exec cp -f '{}' "$source_output/" ';'
    find /work -maxdepth 1 -type f \
        \( -name '*.deb' -o -name '*.udeb' -o -name '*_amd64.buildinfo' -o -name '*_amd64.changes' \) \
        -exec cp -f '{}' "$binary_output/" ';'
}

write_sums() {
    local directory="$1"
    (
        cd "$directory"
        local temporary
        temporary="$(mktemp)"
        trap 'rm -f "$temporary"' EXIT
        find . -maxdepth 1 -type f ! -name SHA256SUMS -print0 \
            | LC_ALL=C sort -z \
            | xargs -0 -r sha256sum > "$temporary"
        test -s "$temporary"
        mv "$temporary" SHA256SUMS
        trap - EXIT
    )
}

prepare() {
    test "$(id -u)" -eq 0 || die "prepare mode requires container root"
    test "$(uname -s)" = Linux || die "prepare mode requires Linux"
    test "$(uname -m)" = x86_64 || die "prepare mode requires native x86_64"

    local provenance="$EVIDENCE/provenance"
    local downloads="/work/debian-source"
    local upstream="/work/upstream"
    local patch_file="$upstream/$PATCH_NAME"
    mkdir -p "$EVIDENCE/logs" "$EVIDENCE/status" "$provenance" "$downloads" "$upstream"
    exec > >(tee -a "$EVIDENCE/logs/prepare.log") 2>&1

    cp -a /opt/tldw-util-linux/apt-build "$provenance/"
    sha256sum /usr/share/keyrings/debian-archive-keyring.gpg \
        > "$provenance/debian-archive-keyring-runtime.sha256"
    apt-cache showsrc util-linux > "$provenance/apt-cache-showsrc-util-linux.txt"
    # APT, not Bash, expands these literal $(...) field templates.
    # shellcheck disable=SC2016
    apt-get indextargets --no-release-info \
        --format '$(IDENTIFIER)|$(SITE)|$(RELEASE)|$(FILENAME)' \
        > "$provenance/apt-indextargets.txt"
    (
        cd "$downloads"
        apt-get source --download-only "util-linux=$DEBIAN_SOURCE_VERSION"
    ) > "$EVIDENCE/logs/apt-get-source.log" 2>&1
    verify_sources "$downloads" | tee "$provenance/debian-source-sha256-check.log"
    sha256sum "$downloads"/* > "$provenance/debian-source-downloads.sha256"
    dpkg-source -x "$downloads/util-linux_${DEBIAN_SOURCE_VERSION}.dsc" "$SOURCE_TREE" \
        > "$EVIDENCE/logs/dpkg-source-extract.log" 2>&1
    cp "$downloads/util-linux_2.41.5.orig.tar.xz" /work/

    git init "$upstream/repository" > "$EVIDENCE/logs/git-init.log" 2>&1
    git -C "$upstream/repository" remote add origin https://github.com/util-linux/util-linux.git
    git -C "$upstream/repository" ls-remote origin > "$provenance/git-ls-remote.txt"
    git -C "$upstream/repository" fetch --depth=2 origin "$UPSTREAM_COMMIT" \
        > "$EVIDENCE/logs/git-fetch.log" 2>&1
    test "$(git -C "$upstream/repository" rev-parse FETCH_HEAD)" = "$UPSTREAM_COMMIT"
    test "$(git -C "$upstream/repository" cat-file -t FETCH_HEAD)" = commit
    git -C "$upstream/repository" cat-file commit FETCH_HEAD > "$provenance/upstream-commit-object.txt"
    git -C "$upstream/repository" show --format=fuller --stat --summary FETCH_HEAD \
        > "$provenance/upstream-commit-metadata.txt"
    git -C "$upstream/repository" diff-tree --no-commit-id --name-only -r FETCH_HEAD \
        | LC_ALL=C sort > "$provenance/upstream-touched-files.txt"
    diff -u <(printf '%s\n' libmount/src/context_mount.c libmount/src/hook_loopdev.c) \
        "$provenance/upstream-touched-files.txt"
    git -C "$upstream/repository" format-patch --stdout -1 FETCH_HEAD > "$patch_file"
    printf '%s  %s\n' "$UPSTREAM_PATCH_SHA256" "$patch_file" \
        | sha256sum -c - \
        | tee "$provenance/upstream-patch-sha256-check.log"
    if git -C "$upstream/repository" verify-commit FETCH_HEAD \
        > "$provenance/upstream-commit-signature-check.log" 2>&1; then
        printf 'The exact commit has a locally verifiable Git signature.\n' \
            >> "$provenance/upstream-commit-signature-check.log"
    else
        printf 'No direct maintainer signature claim: the exact Git object was fetched over HTTPS.\n' \
            >> "$provenance/upstream-commit-signature-check.log"
    fi

    patch --directory "$SOURCE_TREE" --dry-run --fuzz=0 -p1 < "$patch_file" \
        > "$provenance/patch-dry-run-fuzz0.log" 2>&1
    mkdir -p "$SOURCE_TREE/debian/patches/upstream"
    cp "$patch_file" "$SOURCE_TREE/debian/patches/upstream/$PATCH_NAME"
    cmp "$patch_file" "$SOURCE_TREE/debian/patches/upstream/$PATCH_NAME"
    grep -Fqx "upstream/$PATCH_NAME" "$SOURCE_TREE/debian/patches/series" \
        && die "candidate patch already exists in Debian series"
    printf '%s\n' "upstream/$PATCH_NAME" >> "$SOURCE_TREE/debian/patches/series"
    (
        cd "$SOURCE_TREE"
        QUILT_PATCHES=debian/patches quilt push --fuzz=0
    ) > "$provenance/quilt-push-fuzz0.log" 2>&1
    cmp "$patch_file" "$SOURCE_TREE/debian/patches/upstream/$PATCH_NAME"

    {
        printf 'util-linux (%s) UNRELEASED; urgency=medium\n\n' "$CANDIDATE_VERSION"
        printf '  * Candidate-only local backport of upstream commit %s\n' "$UPSTREAM_COMMIT"
        printf '    for CVE-2026-76642 qualification; not for production adoption.\n\n'
        printf ' -- tldw_server CI <noreply@tldw.local>  Sun, 06 Sep 2026 00:00:00 +0000\n\n'
        cat "$SOURCE_TREE/debian/changelog"
    } > /work/changelog.candidate
    mv /work/changelog.candidate "$SOURCE_TREE/debian/changelog"
    test "$(dpkg-parsechangelog -l"$SOURCE_TREE/debian/changelog" -S Version)" = "$CANDIDATE_VERSION"

    mkdir -p "$EVIDENCE/abi/baseline"
    record_library libmount1 libmount "$EVIDENCE/abi/baseline/libmount"
    record_library libblkid1 libblkid "$EVIDENCE/abi/baseline/libblkid"
    record_library libuuid1 libuuid "$EVIDENCE/abi/baseline/libuuid"
    record_library libsmartcols1 libsmartcols "$EVIDENCE/abi/baseline/libsmartcols"
    record_library liblastlog2-2 liblastlog2 "$EVIDENCE/abi/baseline/liblastlog2"
    dpkg-query -W -f='${binary:Package}\t${Version}\t${source:Package}\t${source:Version}\n' \
        bsdutils bsdextrautils libblkid1 liblastlog2-2 libmount1 libsmartcols1 \
        libuuid1 login mount util-linux uuid-dev \
        > "$EVIDENCE/abi/baseline-installed-package-versions.txt"
    dpkg-query -W -f='${binary:Package}=${Version}\n' | LC_ALL=C sort \
        > "$provenance/preparation-packages.txt"
    printf '0\n' > "$EVIDENCE/status/prepare.exit"
    chown -R 1000:1000 /work
}

verify_package_versions() {
    local package_dir="$1"
    local output="$2"
    local deb package version source expected
    : > "$output"
    for deb in "$package_dir"/*.deb; do
        package="$(dpkg-deb -f "$deb" Package)"
        version="$(dpkg-deb -f "$deb" Version)"
        source="$(dpkg-deb -f "$deb" Source 2>/dev/null || true)"
        if [[ -z "$source" && "$package" == util-linux ]]; then
            source=util-linux
        fi
        case "$package" in
            bsdutils|bsdutils-dbgsym) expected="1:$CANDIDATE_VERSION" ;;
            login|login-dbgsym) expected="1:4.16.0-2+really$CANDIDATE_VERSION" ;;
            *) expected="$CANDIDATE_VERSION" ;;
        esac
        test "$version" = "$expected" || die "$package has version $version, expected $expected"
        case "$source" in
            util-linux|"util-linux ($CANDIDATE_VERSION)") ;;
            *) die "$package does not identify the candidate util-linux source: $source" ;;
        esac
        printf '%s\t%s\t%s\n' "$package" "$version" "$source" >> "$output"
    done
    LC_ALL=C sort -o "$output" "$output"
}

build_candidate() {
    test "$(id -u)" -eq 1000 || die "build mode requires UID 1000"
    test "$(uname -s)" = Linux || die "build mode requires Linux"
    test "$(uname -m)" = x86_64 || die "build mode requires native x86_64"
    mkdir -p "$EVIDENCE/logs" "$EVIDENCE/status" "$EVIDENCE/artifacts/source" "$EVIDENCE/artifacts/binary"
    id > "$EVIDENCE/logs/build-user-identity.log"

    set +e
    (
        cd "$SOURCE_TREE"
        DEB_BUILD_OPTIONS=parallel=4 dpkg-buildpackage -S -us -uc
    ) 2>&1 | tee "$EVIDENCE/logs/source-package-build.log"
    local source_status="${PIPESTATUS[0]}"
    set -e
    printf '%s\n' "$source_status" > "$EVIDENCE/status/source-build.exit"
    copy_outputs
    if (( source_status != 0 )); then
        printf 'not-run\n' > "$EVIDENCE/status/binary-build.exit"
        exit "$source_status"
    fi

    set +e
    (
        cd "$SOURCE_TREE"
        DEB_BUILD_OPTIONS=parallel=4 dpkg-buildpackage -b -us -uc -j4
    ) 2>&1 | tee "$EVIDENCE/logs/binary-package-build-and-tests.log"
    local binary_status="${PIPESTATUS[0]}"
    set -e
    printf '%s\n' "$binary_status" > "$EVIDENCE/status/binary-build.exit"
    copy_outputs
    if (( binary_status != 0 )); then
        exit "$binary_status"
    fi

    find "$EVIDENCE/artifacts/binary" -maxdepth 1 -type f \( -name '*.deb' -o -name '*.udeb' \) \
        -exec dpkg-deb -f '{}' Package Version Architecture Depends Pre-Depends ';' \
        > "$EVIDENCE/artifacts/binary/package-control-summary.txt"
    verify_package_versions \
        "$EVIDENCE/artifacts/binary" \
        "$EVIDENCE/artifacts/binary/source-family-versions.txt"
    dpkg-parsechangelog -l"$SOURCE_TREE/debian/changelog" \
        > "$EVIDENCE/artifacts/source/candidate-changelog-fields.txt"

    local candidate_dsc
    candidate_dsc="$(find "$EVIDENCE/artifacts/source" -maxdepth 1 -name "util-linux_${CANDIDATE_VERSION}.dsc" -print)"
    test -n "$candidate_dsc"
    dpkg-source -x "$candidate_dsc" /work/verified-source \
        > "$EVIDENCE/logs/candidate-source-reextract.log" 2>&1
    cmp "/work/upstream/$PATCH_NAME" \
        "/work/verified-source/debian/patches/upstream/$PATCH_NAME"

    mkdir -p /work/candidate-root "$EVIDENCE/abi/candidate"
    for deb in "$EVIDENCE"/artifacts/binary/*.deb; do
        dpkg-deb -x "$deb" /work/candidate-root
    done
    for library in "${LIBRARIES[@]}"; do
        record_candidate_library /work/candidate-root "$library" "$EVIDENCE/abi/candidate/$library"
    done
    set +e
    compare_abi "$EVIDENCE/abi/baseline" "$EVIDENCE/abi/candidate" "$EVIDENCE/abi"
    local abi_status=$?
    set -e
    printf '%s\n' "$abi_status" > "$EVIDENCE/status/abi.exit"
    write_sums "$EVIDENCE/artifacts/source"
    write_sums "$EVIDENCE/artifacts/binary"
    exit "$abi_status"
}

install_packages() {
    local package_dir="$1"
    local evidence="$2"
    local package_file
    local packages=()
    mkdir -p "$evidence/install" "$evidence/status"
    while IFS= read -r package_file; do
        packages+=("$package_file")
    done < <(find "$package_dir" -maxdepth 1 -type f -name '*.deb' -print | LC_ALL=C sort)
    test "${#packages[@]}" -gt 0 || die "no candidate packages were supplied"
    apt-get install -y --no-download --no-remove --no-install-recommends "${packages[@]}"
    apt-get check > "$evidence/install/apt-get-check.log" 2>&1
    dpkg --audit > "$evidence/install/dpkg-audit.txt"
    test ! -s "$evidence/install/dpkg-audit.txt"
    dpkg-query -W -f='${binary:Package}\t${Version}\t${source:Package}\t${source:Version}\n' \
        > "$evidence/install/package-versions.txt"
    local package version
    for package_file in "${packages[@]}"; do
        package="$(dpkg-deb -f "$package_file" Package)"
        version="$(dpkg-deb -f "$package_file" Version)"
        test "$(dpkg-query -W -f='${Version}' "$package")" = "$version"
    done
    {
        mount --version
        findmnt --version
        lsblk --version
        blkid -V
        uuidgen --version
        lastlog2 --version
    } > "$evidence/install/smoke-tests.log" 2>&1
    printf '0\n' > "$evidence/status/install.exit"
}

install_candidate() {
    test "$(id -u)" -eq 0 || die "install mode requires container root"
    test "$(uname -s)" = Linux || die "install mode requires Linux"
    test "$(uname -m)" = x86_64 || die "install mode requires native x86_64"
    mkdir -p "$EVIDENCE/install"
    exec > >(tee -a "$EVIDENCE/install/install.log") 2>&1
    install_packages /candidate "$EVIDENCE"
}

verify_evidence() {
    local evidence="$1"
    local build_status="$2"
    local required=(
        identity/runner.txt
        logs/binary-package-build-and-tests.log
        status/source-build.exit
        status/binary-build.exit
    )
    local path
    for path in "${required[@]}"; do
        test -f "$evidence/$path" || die "required qualification evidence is missing: $path"
    done
    if (( build_status == 0 )); then
        for path in artifacts/source/SHA256SUMS artifacts/binary/SHA256SUMS abi/comparison.txt; do
            test -s "$evidence/$path" || die "required qualification evidence is missing: $path"
        done
        find "$evidence/artifacts/binary" -maxdepth 1 -type f -name '*.deb' -print -quit \
            | grep -q . || die "required qualification evidence is missing: candidate .deb files"
    fi
}

verify_install_evidence() {
    local evidence="$1"
    local path
    for path in install/package-versions.txt install/apt-get-check.log install/dpkg-audit.txt install/smoke-tests.log status/install.exit; do
        test -f "$evidence/$path" || die "required qualification evidence is missing: $path"
    done
    test "$(cat "$evidence/status/install.exit")" = 0
    test ! -s "$evidence/install/dpkg-audit.txt"
}

case "${1:-}" in
    prepare) prepare ;;
    build) build_candidate ;;
    install) install_candidate ;;
    verify-sources)
        test "$#" -eq 2 || die "usage: $0 verify-sources SOURCE_DIR"
        verify_sources "$2"
        ;;
    compare-abi)
        test "$#" -eq 4 || die "usage: $0 compare-abi BASELINE CANDIDATE OUTPUT"
        compare_abi "$2" "$3" "$4"
        ;;
    verify-evidence)
        test "$#" -eq 3 || die "usage: $0 verify-evidence EVIDENCE BUILD_STATUS"
        verify_evidence "$2" "$3"
        ;;
    verify-install-evidence)
        test "$#" -eq 2 || die "usage: $0 verify-install-evidence EVIDENCE"
        verify_install_evidence "$2"
        ;;
    write-sums)
        test "$#" -eq 2 || die "usage: $0 write-sums ARTIFACT_DIRECTORY"
        write_sums "$2"
        ;;
    verify-package-versions)
        test "$#" -eq 3 || die "usage: $0 verify-package-versions PACKAGE_DIRECTORY OUTPUT"
        verify_package_versions "$2" "$3"
        ;;
    install-packages)
        test "$#" -eq 3 || die "usage: $0 install-packages PACKAGE_DIRECTORY EVIDENCE"
        install_packages "$2" "$3"
        ;;
    *) die "usage: $0 {prepare|build|install|verify-sources|compare-abi|verify-evidence|verify-install-evidence|write-sums|verify-package-versions|install-packages}" ;;
esac

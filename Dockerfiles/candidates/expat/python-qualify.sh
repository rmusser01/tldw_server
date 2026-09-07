#!/bin/bash
set -Eeuo pipefail
readonly TOOL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$TOOL_DIR/qualify.sh"
readonly PY_SOURCE=/work/Python-3.12.14
readonly XML_TESTS=(test_pyexpat test_xml_etree test_xml_etree_c test_minidom test_sax)

verify_evidence() {
    local evidence="$1" phase="$2" file
    test -f "$evidence/phase.exit" && test "$(< "$evidence/phase.exit")" = 0 || die "missing/failed phase status: $phase"
    test -f "$evidence/complete.txt" && test "$(< "$evidence/complete.txt")" = "$phase" || die "incomplete phase: $phase"
    case "$phase" in
        prepare)
            for file in authentication.json source-verification.log python-source.tar.xz baseline.xml; do
                test -s "$evidence/$file" || die "missing Python preparation evidence: $file"
            done ;;
        build)
            for file in xml-tests.log xml-results.xml suite-comparison.log python-controls.log scaling-comparison.log abi.txt artifacts/SHA256SUMS; do
                test -s "$evidence/$file" || die "missing Python build evidence: $file"
            done
            grep -Fxq 'PASS: bounded whole/incremental attribute scaling' "$evidence/scaling-comparison.log"
            test "$(< "$evidence/abi.txt")" = compatible ;;
        install)
            for file in python-controls.log binary-checksums.log apt-check.log; do
                test -s "$evidence/$file" || die "missing Python install evidence: $file"
            done
            test -f "$evidence/dpkg-audit.txt" && ! test -s "$evidence/dpkg-audit.txt" ;;
        *) die 'unknown Python phase' ;;
    esac
}

prepare() {
    python "$TOOL_DIR/expat_candidate.py" authenticate-sources /work/downloads \
        --public-keys /work/downloads --evidence "$EVIDENCE/authentication"
    cp "$EVIDENCE/authentication/authentication.json" "$EVIDENCE/"
    cp -a "$TOOL_DIR/apt" "$EVIDENCE/"
    run_step source-extract tar -xJf /work/downloads/Python-3.12.14.tar.xz -C /work
    mkdir "$EVIDENCE/baseline"
    cp "$PY_SOURCE/Modules/expat/expat_config.h" "$PY_SOURCE/Modules/expat/pyexpatns.h" \
        "$PY_SOURCE/Modules/expat/refresh.sh" "$PY_SOURCE/Misc/sbom.spdx.json" \
        "$PY_SOURCE/Misc/externals.spdx.json" "$EVIDENCE/baseline/"
    run_step metadata python "$TOOL_DIR/expat_candidate.py" update-python-metadata "$PY_SOURCE"
    run_step refresh env PATH="$TOOL_DIR/offline-bin:$PATH" bash "$PY_SOURCE/Modules/expat/refresh.sh"
    run_step source-git git -C "$PY_SOURCE" init
    run_step sbom env -u CI python "$PY_SOURCE/Tools/build/generate_sbom.py"
    run_step source-verification python "$TOOL_DIR/python-source.py" "$PY_SOURCE" "$EVIDENCE/baseline"
    run_step source-archive tar --exclude=.git -cJf "$EVIDENCE/python-source.tar.xz" -C /work Python-3.12.14
    run_step baseline-tests env PYTHONPATH="$PY_SOURCE/Lib" /usr/local/bin/python3.12 -m test \
        -v --timeout 300 --junit-xml "$EVIDENCE/baseline.xml" "${XML_TESTS[@]}"
    run_step baseline-validation python "$TOOL_DIR/python-suite.py" "$EVIDENCE/baseline.xml" "$EVIDENCE/baseline.xml"
    run_step scaling-baseline /usr/local/bin/python3.12 "$TOOL_DIR/python-controls.py" measure
}

check_elf() {
    local report="$1" prefix="$2" path label
    mkdir "$EVIDENCE/elf"
    python -c 'import json,sys; d=json.load(open(sys.argv[1]))["identity"]; print("\n".join(d[k] for k in ("executable","libpython","pyexpat","elementtree")))' "$report" > "$EVIDENCE/elf/paths.txt"
    while IFS= read -r path; do
        [[ "$path" == "$prefix/"* ]] || die 'ELF path escaped candidate'
        label="$(basename "$path")"
        sha256sum "$path" > "$EVIDENCE/elf/$label.sha256"
        readelf --wide --dynamic "$path" > "$EVIDENCE/elf/$label.dynamic"
        readelf --wide --dyn-syms "$path" > "$EVIDENCE/elf/$label.symbols"
        ldd "$path" > "$EVIDENCE/elf/$label.ldd"
        if grep -Fq 'not found' "$EVIDENCE/elf/$label.ldd"; then die "unresolved dependency: $label"; fi
        if grep -Eq 'NEEDED.*libexpat' "$EVIDENCE/elf/$label.dynamic"; then die "dynamic system Expat linkage: $label"; fi
        if awk '$8 ~ /^(XML_|Xml)/ {found=1} END {exit !found}' "$EVIDENCE/elf/$label.symbols"; then die "unnamespaced dynamic Expat symbol: $label"; fi
    done < "$EVIDENCE/elf/paths.txt"
}

build() {
    cd "$PY_SOURCE"
    run_step configure ./configure --build="$(dpkg-architecture --query DEB_BUILD_GNU_TYPE)" \
        --enable-loadable-sqlite-extensions --enable-optimizations --enable-option-checking=fatal \
        --enable-shared --with-lto --with-ensurepip
    cp config.log Makefile "$EVIDENCE/"
    grep -Eq '^LIBEXPAT_A[[:space:]]*=[[:space:]]*Modules/expat/libexpat.a$' Makefile
    local cflags ldflags
    cflags="$(dpkg-buildflags --get CFLAGS) -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer"
    ldflags="$(dpkg-buildflags --get LDFLAGS) -Wl,--strip-all"
    run_step python-build make -j4 "EXTRA_CFLAGS=$cflags" "LDFLAGS=$ldflags"
    # Official image relink: make the interpreter find its own installed libpython.
    test -f python
    mv python /work/python-before-rpath
    run_step python-relink make -j4 "EXTRA_CFLAGS=$cflags" "LDFLAGS=$ldflags -Wl,-rpath='\$\$ORIGIN/../lib'" python
    export LD_LIBRARY_PATH="$PY_SOURCE"
    run_step python-controls ./python "$TOOL_DIR/python-controls.py" controls --prefix "$PY_SOURCE"
    run_step xml-tests ./python -m test -v --timeout 300 --junit-xml "$EVIDENCE/xml-results.xml" "${XML_TESTS[@]}"
    run_step suite-comparison python "$TOOL_DIR/python-suite.py" /work/evidence/prepare/baseline.xml "$EVIDENCE/xml-results.xml"
    run_step scaling-candidate ./python "$TOOL_DIR/python-controls.py" measure
    run_step scaling-comparison python "$TOOL_DIR/attribute-scaling.py" compare \
        /work/evidence/prepare/scaling-baseline.log "$EVIDENCE/scaling-candidate.log"
    nm Modules/expat/libexpat.a > "$EVIDENCE/static-expat-symbols.txt"
    grep -Eq '[[:space:]]PyExpat_XML_Parse$' "$EVIDENCE/static-expat-symbols.txt"
    run_step elf-check check_elf "$EVIDENCE/python-controls.log" "$PY_SOURCE"
    printf 'compatible\n' > "$EVIDENCE/abi.txt"
    run_step staged-install make -j4 install DESTDIR=/work/python-install
    mkdir "$EVIDENCE/artifacts"
    cp /work/evidence/prepare/python-source.tar.xz "$EVIDENCE/artifacts/"
    run_step install-archive tar -czf "$EVIDENCE/artifacts/python-install.tar.gz" -C /work/python-install usr/local
    (cd /work/python-install; find usr/local -type f \( -name 'python3.12' -o -name 'libpython3.12.so.1.0' -o -name 'pyexpat*.so' -o -name '_elementtree*.so' \) -exec sha256sum '{}' ';') > "$EVIDENCE/artifacts/installed-binaries.sha256"
    test "$(wc -l < "$EVIDENCE/artifacts/installed-binaries.sha256")" -eq 4
    (cd "$EVIDENCE/artifacts"; sha256sum ./* > SHA256SUMS)
}

install_candidate() {
    cd /candidate
    run_step artifact-checksums sha256sum -c SHA256SUMS
    run_step python-install tar -xzf python-install.tar.gz --no-same-owner -C /
    run_step linker-cache ldconfig
    cd /
    run_step binary-checksums sha256sum -c /candidate/installed-binaries.sha256
    # No build-tree LD_LIBRARY_PATH: the actual installed loader must work.
    run_step python-controls env -u LD_LIBRARY_PATH -u PYTHONPATH /usr/local/bin/python3.12 \
        "$TOOL_DIR/python-controls.py" controls --prefix /usr/local
    run_step elf-check check_elf "$EVIDENCE/python-controls.log" /usr/local
    run_step apt-check apt-get check
    dpkg --audit > "$EVIDENCE/dpkg-audit.txt"
    test ! -s "$EVIDENCE/dpkg-audit.txt"
}

case "${1:-}" in
    controller) controller "$2" python ;;
    verify-evidence) verify_evidence "$2" "$3" ;;
    prepare|build|install)
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
    *) die 'expected controller, verify-evidence, prepare, build or install' ;;
esac

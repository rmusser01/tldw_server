"""Prepare pinned Expat candidate metadata; never install or qualify a binary.

Call source verification before extracting archives into a task-owned directory.
The metadata update prepares CPython's own refresh/SBOM tools; it does not replace
their execution, archive signature verification, or native parser tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json

# Fixed, non-shell GnuPG invocation for source authentication.
import subprocess  # nosec B404
import tempfile
from pathlib import Path

SOURCE_SHA256 = {
    "Python-3.12.14.tar.xz": "5c8462af5790baf43a321a1559dbe0db06d1be4300fb85fb53c40060668e548a",
    "expat-2.8.4.tar.gz": "b8ece2437692dad44d851c4532723390a5a330990007706be9c8d2b90d294f36",
    "expat_2.8.4-1.dsc": "b3dc30ff68a32b95746899d2c8e03cfbb5350b982916d649fab179e56ef5ed3e",
    "expat_2.8.4.orig.tar.gz": "a8a9c5cbba9110000b13cc9943f50fcd7e552a5cbad49cb191142c500a0a11b7",
    "expat_2.8.4-1.debian.tar.xz": "a90e0731e6ccdee5f4368a69ab12cf8cc9f5f29e7e61959e2e839d4ca00361fc",
}
BASELINE_SHA256 = "22920a86c83f32300b11463635b71f11137a917975af297725e55525027d4e50"
RELEASE_SHA256 = SOURCE_SHA256["expat-2.8.4.tar.gz"]
BASELINE_URL = "https://github.com/libexpat/libexpat/releases/download/R_2_8_3/expat-2.8.3.tar.gz"
RELEASE_URL = "https://github.com/libexpat/libexpat/releases/download/R_2_8_4/expat-2.8.4.tar.gz"
BASELINE_CPE = "cpe:2.3:a:libexpat_project:libexpat:2.8.3:*:*:*:*:*:*:*"
SOURCE_SIGNERS = {
    "expat-2.8.4.tar.gz": (
        "CB8DE70A90CFBF6C3BF5CC5696262ACFFBD3AEC6",
        "3176EF7DB2367F1FCA4F306B1F9B0E909AF37285",
        "00",
    ),
    "Python-3.12.14.tar.xz": (
        "7169605F62C751356D054A26A821E680E5FA6305",
        "7169605F62C751356D054A26A821E680E5FA6305",
        "00",
    ),
    "expat_2.8.4-1.dsc": (
        "7D887DC8BA7BBBA7B835E3BADCE310E7864CC8BF",
        "A0DF7E0D3851E0EE45C00BC8ACE1F33CB933BBBB",
        "01",
    ),
}
PUBLIC_KEY_FILES = ("expat-key.asc", "python-key.asc", "debian-maintainer-full-key.asc")


def verify_sources(directory: Path) -> dict[str, str]:
    """Require every pinned source file; reject absent, aliased or altered inputs."""
    verified = {}
    for filename, expected in SOURCE_SHA256.items():
        path = directory / filename
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"source input must be a regular file: {filename}")
        with path.open("rb") as stream:
            observed = hashlib.file_digest(stream, "sha256").hexdigest()
        if observed != expected:
            raise ValueError(f"source digest mismatch: {filename}")
        verified[filename] = observed
    return verified


def validate_signature_status(filename: str, status: str, returncode: int) -> dict[str, str]:
    """Accept one fresh GnuPG verification, bound to both approved fingerprints.

    Only consume the dedicated status channel, never human diagnostics. A valid
    cryptographic signature can coexist with an expired/revoked-key status, so
    neither VALIDSIG nor process success alone is sufficient. Owner trust is not
    required: the independently approved full fingerprints are the trust anchor.
    """
    rejected = {
        "BADSIG",
        "ERRSIG",
        "NO_PUBKEY",
        "EXPSIG",
        "EXPKEYSIG",
        "REVKEYSIG",
        "KEYEXPIRED",
        "SIGEXPIRED",
        "KEYREVOKED",
        "NODATA",
        "FAILURE",
        "ERROR",
        "TRUST_NEVER",
    }
    records = []
    for line in status.splitlines():
        if not line.startswith("[GNUPG:] ") or not line[9:].split():
            raise ValueError(f"malformed signature status: {filename}")
        records.append(line[9:].split())
    if returncode != 0 or any(record[0] in rejected for record in records):
        raise ValueError(f"signature verification failed: {filename}")
    valid = [record[1:] for record in records if record[0] == "VALIDSIG"]
    good = [record[1:] for record in records if record[0] == "GOODSIG"]
    if (
        sum(record[0] == "NEWSIG" for record in records) != 1
        or len(valid) != 1
        or len(valid[0]) < 10
        or len(good) != 1
        or not good[0]
    ):
        raise ValueError(f"expected exactly one complete signature: {filename}")
    signer, primary, signature_class = SOURCE_SIGNERS[filename]
    fields = valid[0]
    if (
        fields[0] != signer
        or fields[9] != primary
        or fields[8] != signature_class
        or fields[7] not in {"8", "9", "10", "11"}  # SHA-256/384/512/224, not SHA-1/MD5
        or good[0][0] not in {signer, signer[-16:]}
    ):
        raise ValueError(f"unexpected signature identity or algorithm: {filename}")
    return {"signer": signer, "primary": primary}


def _run_gpg(home: Path, evidence: Path, label: str, arguments: list[str]) -> subprocess.CompletedProcess[str]:
    """Run offline GnuPG with private state; retain diagnostics before checking."""
    command = [
        "/usr/bin/gpg",
        "--no-options",
        "--batch",
        "--no-tty",
        "--no-autostart",
        "--no-auto-key-retrieve",
        "--no-auto-key-import",
        "--homedir",
        str(home),
        "--status-fd",
        "1",
        *arguments,
    ]
    (evidence / f"{label}.command.json").write_text(json.dumps(command) + "\n", encoding="utf-8")
    # Fixed executable/options; only explicit local input paths are variable.
    try:
        result = subprocess.run(  # nosec B603
            command,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as error:
        timed_out = isinstance(error, subprocess.TimeoutExpired)
        stdout = error.stdout or b"" if timed_out else b""
        stderr = error.stderr or b"" if timed_out else str(error).encode()
        # TimeoutExpired retains bytes even when run() requests text output.
        (evidence / f"{label}.status").write_bytes(stdout)
        (evidence / f"{label}.stderr").write_bytes(stderr)
        outcome = "timeout" if timed_out else "os-error"
        (evidence / f"{label}.exit").write_text(outcome + "\n", encoding="utf-8")
        raise ValueError(f"GnuPG process failed: {label} ({outcome})") from error
    (evidence / f"{label}.status").write_text(result.stdout, encoding="utf-8")
    (evidence / f"{label}.stderr").write_text(result.stderr, encoding="utf-8")
    (evidence / f"{label}.exit").write_text(f"{result.returncode}\n", encoding="utf-8")
    return result


def authenticate_sources(directory: Path, public_keys: Path, evidence: Path) -> dict:
    """Authenticate all pinned inputs without extracting or executing sources.

    The caller must supply immutable inputs (e.g. read-only container mounts),
    an offline execution environment and a new evidence directory. Offline key
    material cannot establish the absence of a newer upstream key revocation.
    No existing status log or success record is accepted as authentication.
    """
    sources = verify_sources(directory)
    inputs = [directory / (name + ".asc") for name in SOURCE_SIGNERS if not name.endswith(".dsc")]
    key_paths = [public_keys / name for name in PUBLIC_KEY_FILES]
    inputs.extend(key_paths)
    identities = {}
    for path in inputs:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"authentication input must be a regular file: {path.name}")
        identities[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    evidence.mkdir(mode=0o700, parents=True, exist_ok=False)
    report = {"sources": sources, "authentication_inputs": identities, "signatures": {}}
    with tempfile.TemporaryDirectory(prefix="gnupg-", dir=evidence) as temporary_home:
        home = Path(temporary_home)
        version = _run_gpg(home, evidence, "gpg-version", ["--version"])
        if version.returncode != 0:
            raise ValueError("GnuPG version check failed")
        imported = _run_gpg(home, evidence, "key-import", ["--import", *map(str, key_paths)])
        if imported.returncode != 0:
            raise ValueError("public key import failed")
        for filename in SOURCE_SIGNERS:
            arguments = ["--verify"]
            if not filename.endswith(".dsc"):
                arguments.append(str(directory / (filename + ".asc")))
            arguments.append(str(directory / filename))
            result = _run_gpg(home, evidence, filename, arguments)
            report["signatures"][filename] = validate_signature_status(filename, result.stdout, result.returncode)
    # Recheck archive identity before emitting success. Future build controllers
    # must keep the same verified bytes immutable through extraction/execution.
    verify_sources(directory)
    (evidence / "authentication.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def update_python_metadata(root: Path) -> None:
    """Prepare the authenticated 3.12.14 tree for its official Expat refresh.

    Validate all expected baseline metadata before writing either file. This is
    intentionally not idempotent: a reused or unexpected tree needs inspection.
    The caller owns a fresh extracted tree, never a live Python installation.
    """
    refresh_path = root / "Modules/expat/refresh.sh"
    sbom_path = root / "Misc/sbom.spdx.json"
    for path in (refresh_path, sbom_path):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"baseline metadata must be a regular file: {path}")
    refresh = refresh_path.read_text(encoding="utf-8")
    replacements = {
        "expected_libexpat_tag": ("R_2_8_3", "R_2_8_4"),
        "expected_libexpat_version": ("2.8.3", "2.8.4"),
        "expected_libexpat_sha256": (BASELINE_SHA256, RELEASE_SHA256),
    }
    lines = refresh.splitlines(keepends=True)
    for name, (old, new) in replacements.items():
        matches = [index for index, line in enumerate(lines) if line.startswith(name + "=")]
        if len(matches) != 1 or lines[matches[0]] != f'{name}="{old}"\n':
            raise ValueError(f"unexpected refresh baseline assignment: {name}")
        lines[matches[0]] = f'{name}="{new}"\n'

    sbom = json.loads(sbom_path.read_text(encoding="utf-8"))
    packages = [item for item in sbom["packages"] if item.get("name") == "expat"]
    if len(packages) != 1:
        raise ValueError("baseline SBOM must contain exactly one Expat package")
    package = packages[0]
    if (
        package.get("SPDXID") != "SPDXRef-PACKAGE-expat"
        or package.get("versionInfo") != "2.8.3"
        or package.get("downloadLocation") != BASELINE_URL
        or package.get("checksums") != [{"algorithm": "SHA256", "checksumValue": BASELINE_SHA256}]
    ):
        raise ValueError("unexpected Expat baseline SBOM metadata")
    cpes = [ref for ref in package.get("externalRefs", []) if ref.get("referenceType") == "cpe23Type"]
    if len(cpes) != 1 or cpes[0].get("referenceLocator") != BASELINE_CPE:
        raise ValueError("unexpected Expat baseline CPE")
    package.update(
        versionInfo="2.8.4",
        downloadLocation=RELEASE_URL,
        checksums=[{"algorithm": "SHA256", "checksumValue": RELEASE_SHA256}],
    )
    cpes[0]["referenceLocator"] = BASELINE_CPE.replace(":2.8.3:", ":2.8.4:")
    # Do not pretend to regenerate file hashes here: CPython's source SBOM tool
    # must run after refresh.sh has copied and namespaced the actual new sources.
    refresh_path.write_text("".join(lines), encoding="utf-8")
    sbom_path.write_text(json.dumps(sbom, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    """Expose explicit, separate source-verification and metadata-update steps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify-sources", "authenticate-sources", "update-python-metadata"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--public-keys", type=Path)
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args()
    if args.command == "verify-sources":
        print(json.dumps(verify_sources(args.directory), indent=2, sort_keys=True))
    elif args.command == "authenticate-sources":
        if args.public_keys is None or args.evidence is None:
            parser.error("authenticate-sources requires --public-keys and --evidence")
        print(
            json.dumps(authenticate_sources(args.directory, args.public_keys, args.evidence), indent=2, sort_keys=True)
        )
    else:
        update_python_metadata(args.directory)
        print("Metadata prepared; source refresh, SBOM regeneration and native qualification are still required.")


if __name__ == "__main__":
    main()

"""Check exact-candidate identity and embedded runtime assumptions without waivers."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import re
import sys
import tempfile
import tomllib
from pathlib import Path


def _read_blob(layout: Path, digest: str) -> dict:
    """Read only a SHA-256-addressed, integrity-checked OCI JSON object."""
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise ValueError("invalid OCI digest")
    path = layout / "blobs" / "sha256" / digest[7:]
    raw = path.read_bytes() if path.exists() else (layout / "index.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest[7:]:
        raise ValueError("OCI digest mismatch")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("OCI object must be a mapping")
    return value


def candidate_config(layout: Path, subject: str) -> str:
    """Resolve the single linux/amd64 candidate to its integrity-checked config."""
    manifest = _read_blob(layout, subject)
    if "manifests" in manifest:
        matches = [
            item
            for item in manifest["manifests"]
            if item.get("platform", {}).get("os") == "linux" and item.get("platform", {}).get("architecture") == "amd64"
        ]
        if len(matches) != 1:
            raise ValueError("expected one linux/amd64 manifest")
        manifest = _read_blob(layout, matches[0]["digest"])
    config_digest = manifest["config"]["digest"]
    config = _read_blob(layout, config_digest)
    if (config.get("os"), config.get("architecture")) != ("linux", "amd64"):
        raise ValueError("candidate config platform mismatch")
    return config_digest


def require_no_listeners(proc_net: Path = Path("/proc/net")) -> None:
    """Fail closed if the isolated Linux probe namespace has any TCP listener."""
    for name in ("tcp", "tcp6"):
        lines = (proc_net / name).read_text().splitlines()
        if not lines:
            raise ValueError("missing socket table header")
        for line in lines[1:]:
            fields = line.split()
            if len(fields) < 4 or not re.fullmatch(r"[0-9A-F]{2}", fields[3]):
                raise ValueError("malformed socket table")
            if fields[3] == "0A":
                raise ValueError("unexpected listening socket")


def require_locked_versions(lock: Path) -> dict[str, str]:
    """Require the installed probe dependencies to match the candidate lock."""
    packages = tomllib.loads(lock.read_text(encoding="utf-8"))["package"]
    versions = {}
    for name in ("chromadb", "python-jose", "cryptography"):
        expected = {item["version"] for item in packages if item["name"] == name}
        actual = importlib.metadata.version(name)
        if expected != {actual}:
            raise ValueError(f"{name} does not match its locked version")
        versions[name] = actual
    return versions


def probe_crypto() -> str:
    """Exercise an ephemeral ES256 signature through python-jose's EC backend."""
    from cryptography.hazmat.primitives.asymmetric import ec
    from jose import backends, jwt

    backend = backends.ECKey.__module__
    if backend != "jose.backends.cryptography_backend":
        raise ValueError("JWT EC signing must use the cryptography backend")
    key = ec.generate_private_key(ec.SECP256R1())
    claims = {"sub": "runtime-probe", "aud": "runtime-probe"}
    token = jwt.encode(claims, key, algorithm="ES256")
    if jwt.decode(token, key.public_key(), algorithms=["ES256"], audience="runtime-probe") != claims:
        raise ValueError("JWT signature round trip failed")
    return backend


def probe_chroma(*, proc_net: Path | None = None) -> str:
    """Exercise real application-managed storage in two distinct user directories."""
    from tldw_Server_API.app.core.Embeddings.ChromaDB_Library import ChromaDBManager

    with tempfile.TemporaryDirectory(prefix="tldw-runtime-probe-") as scratch:
        managers = []
        try:
            for user in ("130137301", "130137302"):
                manager = ChromaDBManager(
                    user_id=user,
                    user_embedding_config={
                        "USER_DB_BASE_DIR": scratch,
                        "chroma_client_settings": {"backend": "persistent", "anonymized_telemetry": False},
                    },
                )
                managers.append(manager)
                backend = type(getattr(manager.client, "_server", None)).__module__
                if backend != "chromadb.api.rust":
                    raise ValueError("probe requires the real embedded Chroma backend")
            if proc_net is not None:
                require_no_listeners(proc_net)
            if managers[0].user_chroma_path == managers[1].user_chroma_path:
                raise ValueError("Chroma user directories are not isolated")
            collections = [manager.get_or_create_collection("runtime-probe") for manager in managers]
            for index, collection in enumerate(collections):
                collection.add(ids=[f"user-{index}"], documents=[f"private-{index}"], embeddings=[[1.0, 0.0]])
            for index, collection in enumerate(collections):
                if collection.get()["ids"] != [f"user-{index}"]:
                    raise ValueError("Chroma collection isolation failed")
                result = collection.query(query_embeddings=[[1.0, 0.0]], n_results=1)
                if result["ids"] != [[f"user-{index}"]]:
                    raise ValueError("Chroma query isolation failed")
            if proc_net is not None:
                require_no_listeners(proc_net)
            return backend
        finally:
            for manager in reversed(managers):
                manager.close()


def main() -> None:
    """Resolve an OCI identity on the host, or produce in-container evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    identity = subcommands.add_parser("config")
    identity.add_argument("--layout", type=Path, required=True)
    identity.add_argument("--subject", required=True)
    probe = subcommands.add_parser("probe")
    probe.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "config":
        print(candidate_config(args.layout, args.subject))
        return
    # Application imports may print diagnostics; reserve stdout for JSON evidence.
    with contextlib.redirect_stdout(sys.stderr):
        versions = require_locked_versions(args.lock)
        require_no_listeners()
        chroma_backend = probe_chroma(proc_net=Path("/proc/net"))
        require_no_listeners()
        jwt_backend = probe_crypto()
    print(
        json.dumps(
            {
                "versions": versions,
                "lock_sha256": hashlib.sha256(args.lock.read_bytes()).hexdigest(),
                "chroma_backend": chroma_backend,
                "jwt_ec_backend": jwt_backend,
                "checks": {"per_user_collection_and_query_isolation": True, "no_tcp_listeners": True},
                "scope": "isolated embedded-client probe; not application startup or a vulnerability waiver",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

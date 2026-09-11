"""Bind the loaded native execution config to its canonical OCI build archive.

This checks metadata and hashes the complete retained archive. Layer validation
is performed by the trusted build/export/load path, not a claim made here.
"""

import argparse
import hashlib
import json
import re
import tarfile
from pathlib import Path


def verify(path: Path, loaded_config: str, commit: str) -> dict:
    """Reject an unrelated loaded image or ambiguous/tampered OCI metadata."""
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", loaded_config) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("invalid loaded config or checkout identity")
    if path.is_symlink() or not path.is_file():
        raise ValueError("missing or linked OCI archive")
    with tarfile.open(path, "r:") as archive:
        members = archive.getmembers()
        if len({item.name for item in members}) != len(members):
            raise ValueError("duplicate OCI archive member")

        def read(name: str) -> bytes:
            member = archive.getmember(name)
            if not member.isfile() or not 0 < member.size <= 16 * 1024 * 1024:
                raise ValueError("invalid OCI metadata member")
            with archive.extractfile(member) as stream:
                return stream.read()

        def blob(descriptor: dict) -> dict:
            digest = descriptor["digest"]
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
                raise ValueError("invalid OCI descriptor digest")
            raw = read("blobs/sha256/" + digest[7:])
            if len(raw) != descriptor["size"] or hashlib.sha256(raw).hexdigest() != digest[7:]:
                raise ValueError("OCI metadata integrity mismatch")
            return json.loads(raw)

        index_raw = read("index.json")
        index = json.loads(index_raw)
        if len(index["manifests"]) != 1:
            raise ValueError("expected one exported OCI subject")
        subject = index["manifests"][0]
        descriptor = subject
        for _ in range(4):
            manifest = blob(descriptor)
            if "manifests" not in manifest:
                break
            native = [
                item
                for item in manifest["manifests"]
                if item.get("platform", {}).get("os") == "linux"
                and item.get("platform", {}).get("architecture") == "amd64"
            ]
            if len(native) != 1:
                raise ValueError("expected one native amd64 manifest")
            descriptor = native[0]
        else:
            raise ValueError("excessive OCI index nesting")
        if manifest["config"]["digest"] != loaded_config:
            raise ValueError("execution image differs from retained OCI config")
        config = blob(manifest["config"])
        if (config.get("os"), config.get("architecture"), config.get("config", {}).get("User")) != (
            "linux",
            "amd64",
            "10001:10001",
        ):
            raise ValueError("unexpected execution platform or user")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "scope": "combined-candidate-not-admitted",
        "commit": commit,
        "subject_digest": subject["digest"],
        "index_digest": "sha256:" + hashlib.sha256(index_raw).hexdigest(),
        "config_digest": loaded_config,
        "archive_sha256": digest,
    }


def main() -> None:
    """Emit evidence only after all execution/retention binding checks pass."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--config", required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.archive, args.config, args.commit), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

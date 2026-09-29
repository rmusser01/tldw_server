"""Check fixed successful producer metadata and bytes before native assembly.

Metadata must be fetched by the trusted workflow directly from GitHub's API.
These checks do not turn caller-supplied JSON into an attestation.
"""

import argparse
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

REPOSITORY = {"id": 794199679, "full_name": "rmusser01/tldw_server"}
BRANCH = "codex/task-13013-7-supply-chain-design"
PARSER_COMMIT = "aa93e0058978248e5be8311fb8e4c0db2d05aacb"
FFMPEG_SUBJECT = "sha256:de2b316796fd39f58cc76617e66ce400933e9a63ab3f7a261cac66fa461f2cb4"
PRODUCERS = {
    "ffmpeg": {
        "run": 34172954787,
        "commit": "d19bdf2734350f3b8c4f839b6b9150ee8b3a1afe",
        "workflow": ".github/workflows/ffmpeg-candidate.yml",
        "artifact": 10036487817,
        "name": "ffmpeg-native-candidate-34172954787",
    },
    "system": {
        "run": 34170600664,
        "commit": PARSER_COMMIT,
        "workflow": ".github/workflows/expat-candidate.yml",
        "artifact": 10035680097,
        "name": "expat-system-candidate-34170600664",
    },
    "python": {
        "run": 34170600664,
        "commit": PARSER_COMMIT,
        "workflow": ".github/workflows/expat-candidate.yml",
        "artifact": 10035842132,
        "name": "expat-python-candidate-34170600664",
    },
}
PAYLOADS = {
    "ffmpeg/ffmpeg.oci.tar": "78d303ff89cb3ec08f52b327d95c6529998bc29c7fba516bf116553e7f8ffc43",
    "ffmpeg/oci-subject.json": "1a1ec2d67d26cc308f9e195597702aaf0bef2f5389c12532b907cec9220e0900",
    "system/build/artifacts/libexpat1_2.8.4-1~deb13u1+tldw1_amd64.deb": "0f06b4b09147e9baf790ce6118a6b705b8c44edb780138af179c3f5600b580dd",
    "python/build/artifacts/python-install.tar.gz": "d1c97c21e5f940509acccbbf971ea918ed8e1ce6a74916c7c20e758c283f9077",
}


def verify_metadata(run: dict, artifact: dict, producer: dict) -> None:
    """Reject unqualified runs, forks, wrong workflows or detached artifacts."""
    expected_run = {
        "id": producer["run"],
        "head_sha": producer["commit"],
        "head_branch": BRANCH,
        "path": producer["workflow"],
        "status": "completed",
        "conclusion": "success",
    }
    if any(run.get(key) != value for key, value in expected_run.items()) or run.get("event") not in {
        "push",
        "workflow_dispatch",
    }:
        raise ValueError("producer run identity or outcome mismatch")
    for key in ("repository", "head_repository"):
        if not isinstance(run.get(key), dict) or any(run[key].get(k) != v for k, v in REPOSITORY.items()):
            raise ValueError("producer repository mismatch")
    if (
        artifact.get("id") != producer["artifact"]
        or artifact.get("name") != producer["name"]
        or artifact.get("expired") is not False
    ):
        raise ValueError("artifact identity or retention mismatch")
    binding = artifact.get("workflow_run")
    expected_binding = {
        "id": producer["run"],
        "head_sha": producer["commit"],
        "head_branch": BRANCH,
        "repository_id": REPOSITORY["id"],
        "head_repository_id": REPOSITORY["id"],
    }
    if not isinstance(binding, dict) or any(binding.get(k) != v for k, v in expected_binding.items()):
        raise ValueError("artifact is not bound to the expected producer")


def verify_payload(path: Path, expected: str) -> None:
    """Check reviewed immutable input bytes without following a payload symlink."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or linked payload: {path.name}")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != expected:
        raise ValueError(f"payload hash mismatch: {path.name}")


def main() -> None:
    """Reverify all producer gates, then stage only the two runtime replacements."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("downloads", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.downloads.resolve()
    for label, producer in PRODUCERS.items():
        run = json.loads((root / f"{label}-run.json").read_text())
        artifact = json.loads((root / f"{label}-artifact.json").read_text())
        verify_metadata(run, artifact, producer)
    for name, digest in PAYLOADS.items():
        verify_payload(root / name, digest)
    helper = Path(__file__).resolve().parents[1] / "expat/combined-inputs.py"
    spec = importlib.util.spec_from_file_location("qualified_parser_inputs", helper)
    qualified = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qualified)
    parser_inputs = qualified.verify(root / "system", root / "python", PARSER_COMMIT)
    args.output.mkdir(parents=True, exist_ok=False)
    for label, name in (
        ("system", "libexpat1_2.8.4-1~deb13u1+tldw1_amd64.deb"),
        ("python", "python-install.tar.gz"),
        ("python", "installed-binaries.sha256"),
    ):
        shutil.copyfile(root / label / "build/artifacts" / name, args.output / name)
    record = {
        "scope": "assembly-inputs-only",
        "producers": PRODUCERS,
        "payloads": PAYLOADS,
        "ffmpeg_subject": FFMPEG_SUBJECT,
        "parser_inputs": parser_inputs,
    }
    (args.output / "inputs.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

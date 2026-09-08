"""Bind native assembly inputs to the reviewed producer runs and exact bytes."""

import copy
import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "Dockerfiles/candidates/combined-expat/assembly-inputs.py"


def module():
    assert SCRIPT.is_file(), "assembly input binding is not implemented"
    spec = importlib.util.spec_from_file_location("assembly_inputs", SCRIPT)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def producer():
    return {
        "run": 41,
        "commit": "a" * 40,
        "workflow": ".github/workflows/candidate.yml",
        "artifact": 42,
        "name": "candidate-41",
    }


def metadata():
    run = {
        "id": 41,
        "head_sha": "a" * 40,
        "head_branch": "codex/task-13013-7-supply-chain-design",
        "path": ".github/workflows/candidate.yml",
        "event": "push",
        "status": "completed",
        "conclusion": "success",
        "repository": {"id": 794199679, "full_name": "rmusser01/tldw_server"},
        "head_repository": {"id": 794199679, "full_name": "rmusser01/tldw_server"},
    }
    artifact = {
        "id": 42,
        "name": "candidate-41",
        "expired": False,
        "workflow_run": {
            "id": 41,
            "head_sha": "a" * 40,
            "head_branch": "codex/task-13013-7-supply-chain-design",
            "repository_id": 794199679,
            "head_repository_id": 794199679,
        },
    }
    return run, artifact


def test_accepts_only_successful_exact_producer_metadata_without_mutation():
    run, artifact = metadata()
    before = copy.deepcopy((run, artifact))
    module().verify_metadata(run, artifact, producer())
    assert (run, artifact) == before


@pytest.mark.parametrize(
    "target,key,value",
    [
        ("run", "id", 99),
        ("run", "head_sha", "b" * 40),
        ("run", "head_branch", "dev"),
        ("run", "path", ".github/workflows/other.yml"),
        ("run", "event", "pull_request"),
        ("run", "status", "in_progress"),
        ("run", "conclusion", "failure"),
        ("run", "repository", {"id": 794199679, "full_name": "other/repo"}),
        ("run", "head_repository", {"id": 1, "full_name": "other/repo"}),
        ("artifact", "id", 99),
        ("artifact", "name", "other"),
        ("artifact", "expired", True),
        ("binding", "id", 99),
        ("binding", "head_sha", "b" * 40),
        ("binding", "head_branch", "dev"),
        ("binding", "repository_id", 1),
        ("binding", "head_repository_id", 1),
    ],
)
def test_rejects_metadata_from_wrong_run_workflow_checkout_or_repository(target, key, value):
    run, artifact = metadata()
    {"run": run, "artifact": artifact, "binding": artifact["workflow_run"]}[target][key] = value
    with pytest.raises(ValueError):
        module().verify_metadata(run, artifact, producer())


@pytest.mark.parametrize("kind", ["valid", "changed", "symlink", "missing"])
def test_payload_hash_checks_real_bytes_without_following_links(tmp_path, kind):
    payload = tmp_path / "payload"
    if kind != "missing":
        payload.write_bytes(b"abc" if kind != "changed" else b"abcd")
    if kind == "symlink":
        link = tmp_path / "link"
        link.symlink_to(payload)
        payload = link
    digest = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    if kind == "valid":
        module().verify_payload(payload, digest)
    else:
        with pytest.raises(ValueError):
            module().verify_payload(payload, digest)

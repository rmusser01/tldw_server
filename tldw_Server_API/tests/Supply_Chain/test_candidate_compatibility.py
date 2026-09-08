"""Reject substituted retained inputs and wrong containerd execution subjects."""

import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "Dockerfiles/candidates/compatibility/retained-inputs.py"
SUBJECT = "sha256:08f4a090041b1d87d779e1436073910c0b6c4afc2ffcb9a6d957a94c307b45bb"


def module():
    assert SCRIPT.is_file(), "retained compatibility binding is not implemented"
    spec = importlib.util.spec_from_file_location("retained_inputs", SCRIPT)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.mark.parametrize("mutation", [None, "subject", "config-as-id", "platform", "root", "empty", "multiple"])
def test_loaded_containerd_identity_rejects_wrong_subject_platform_or_user(mutation):
    inspection = [{"Id": SUBJECT, "Os": "linux", "Architecture": "amd64", "Config": {"User": "10001:10001"}}]
    if mutation == "subject":
        inspection[0]["Id"] = "sha256:" + "a" * 64
    elif mutation == "config-as-id":
        inspection[0]["Id"] = "sha256:601019026e81c779df97656211f6a972a71a7a8b55a57e5a3588be8880dd9aee"
    elif mutation == "platform":
        inspection[0]["Architecture"] = "arm64"
    elif mutation == "root":
        inspection[0]["Config"]["User"] = "0"
    elif mutation == "empty":
        inspection = []
    elif mutation == "multiple":
        inspection *= 2
    helper = module()
    if mutation is None:
        assert helper.verify_loaded(inspection) == SUBJECT
    else:
        with pytest.raises(ValueError):
            helper.verify_loaded(inspection)


@pytest.mark.parametrize(
    "mutation", [None, "run", "fork", "artifact", "archive", "baseline", "source", "evaluator", "symlink"]
)
def test_input_gate_rejects_substitution_before_emitting_evidence(tmp_path, monkeypatch, mutation):
    helper = module()
    repository = {"id": 794199679, "full_name": "rmusser01/tldw_server"}
    run = {
        "id": 34177651966,
        "head_sha": "20fbadd2dc23ff3c186153f581a9b1dbd2731b24",
        "head_branch": "codex/task-13013-7-supply-chain-design",
        "path": ".github/workflows/combined-expat-candidate.yml",
        "event": "push",
        "status": "completed",
        "conclusion": "success",
        "repository": repository,
        "head_repository": repository,
    }
    artifact = {
        "id": 10038017399,
        "name": "combined-expat-candidate-34177651966",
        "expired": False,
        "workflow_run": {
            "id": 34177651966,
            "head_sha": "20fbadd2dc23ff3c186153f581a9b1dbd2731b24",
            "head_branch": "codex/task-13013-7-supply-chain-design",
            "repository_id": 794199679,
            "head_repository_id": 794199679,
        },
    }
    if mutation == "run":
        run["conclusion"] = "failure"
    elif mutation == "fork":
        run["head_repository"] = {"id": 1, "full_name": "other/repo"}
    elif mutation == "artifact":
        artifact["workflow_run"]["head_sha"] = "a" * 40
    (tmp_path / "run.json").write_text(json.dumps(run))
    (tmp_path / "artifact.json").write_text(json.dumps(artifact))
    # Small real payloads exercise hashing; the native workflow uses the fixed production pins.
    payloads = {name: tmp_path / name for name in ("archive", "baseline", "source", "evaluator")}
    for name, path in payloads.items():
        path.write_bytes(b"abcd" if mutation == name else b"abc")
    if mutation == "symlink":
        link = tmp_path / "linked"
        link.symlink_to(payloads["baseline"])
        payloads["baseline"] = link
    monkeypatch.setattr(
        helper,
        "PAYLOAD_HASHES",
        dict.fromkeys(payloads, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
    )
    if mutation is None:
        assert helper.verify_inputs(tmp_path, payloads)["scope"] == "retained-candidate-inputs-not-admitted"
    else:
        with pytest.raises(ValueError):
            helper.verify_inputs(tmp_path, payloads)

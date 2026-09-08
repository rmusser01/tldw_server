"""Use real OCI tar fixtures to bind an execution config to retained evidence."""

import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "Dockerfiles/candidates/combined-expat/image-identity.py"


def module():
    assert SCRIPT.is_file(), "execution image binding is not implemented"
    spec = importlib.util.spec_from_file_location("image_identity", SCRIPT)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "config-bytes",
        "descriptor-size",
        "wrong-config",
        "wrong-platform",
        "root-user",
        "extra-subject",
        "duplicate-member",
        "linked-member",
        "bad-commit",
        "ambiguous-native",
    ],
)
def test_image_binding_rejects_tampering_or_ineligible_execution_identity(tmp_path, mutation):
    helper = module()
    files = {}

    def blob(value):
        raw = json.dumps(value).encode()
        digest = "sha256:" + hashlib.sha256(raw).hexdigest()
        files["blobs/sha256/" + digest[7:]] = raw
        return {"digest": digest, "size": len(raw)}

    config = blob(
        {
            "os": "linux",
            "architecture": "arm64" if mutation == "wrong-platform" else "amd64",
            "config": {"User": "0" if mutation == "root-user" else "10001:10001"},
        }
    )
    if mutation == "descriptor-size":
        config["size"] += 1
    manifest = blob({"config": config, "layers": []})
    manifest["platform"] = {"os": "linux", "architecture": "amd64"}
    subject = blob({"manifests": [manifest, manifest] if mutation == "ambiguous-native" else [manifest]})
    files["index.json"] = json.dumps(
        {"manifests": [subject, subject] if mutation == "extra-subject" else [subject]}
    ).encode()
    if mutation == "config-bytes":
        files["blobs/sha256/" + config["digest"][7:]] += b" "
    path = tmp_path / "candidate.oci.tar"
    with tarfile.open(path, "w") as bundle:
        for name, raw in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            if mutation == "linked-member" and name == "index.json":
                member.type = tarfile.SYMTYPE
                member.linkname = "/etc/passwd"
                member.size = 0
                bundle.addfile(member)
            else:
                bundle.addfile(member, io.BytesIO(raw))
            if mutation == "duplicate-member" and name == "index.json":
                bundle.addfile(member, io.BytesIO(raw))
    args = (
        path,
        "sha256:" + "0" * 64 if mutation == "wrong-config" else config["digest"],
        "invalid" if mutation == "bad-commit" else "a" * 40,
    )
    if mutation is not None:
        with pytest.raises(ValueError):
            helper.verify(*args)
    else:
        result = helper.verify(*args)
        assert result["config_digest"] == config["digest"]
        assert result["subject_digest"] == subject["digest"]
        assert result["archive_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert result["scope"] == "combined-candidate-not-admitted"

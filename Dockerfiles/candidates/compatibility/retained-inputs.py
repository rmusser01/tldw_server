"""Bind compatibility probes to one retained candidate; never admit or rebuild it."""

import argparse
import importlib.util
import json
from pathlib import Path

PRODUCER = {
    "run": 34177651966,
    "commit": "20fbadd2dc23ff3c186153f581a9b1dbd2731b24",
    "workflow": ".github/workflows/combined-expat-candidate.yml",
    "artifact": 10038017399,
    "name": "combined-expat-candidate-34177651966",
}
SUBJECT = "sha256:08f4a090041b1d87d779e1436073910c0b6c4afc2ffcb9a6d957a94c307b45bb"
CONFIG = "sha256:601019026e81c779df97656211f6a972a71a7a8b55a57e5a3588be8880dd9aee"
PAYLOAD_HASHES = {
    "archive": "864b2e19887641bbe2f6505d0b28f44ccec3cb1cc3d1de4f65063f5211c1c478",
    "baseline": "ec383730c4906414d7dda7d92ddeb7a9ffb1934f74c3c741de79906d37026b32",
    "source": "cf38e0e28c7e5605942c4a77755349b0145804a397af37eb1fb4c77cb237f635",
    "evaluator": "bb52d9989c1d621af2dfe1fd71e80b5989d801d2943b74ada92a4952df960084",
}


def verify_inputs(metadata: Path, payloads: dict[str, Path]) -> dict:
    """Require trusted API metadata and reviewed bytes before Docker loads anything."""
    script = Path(__file__).resolve().parents[1] / "combined-expat/assembly-inputs.py"
    spec = importlib.util.spec_from_file_location("assembly_input_checks", script)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.verify_metadata(
        json.loads((metadata / "run.json").read_text()),
        json.loads((metadata / "artifact.json").read_text()),
        PRODUCER,
    )
    if payloads.keys() != PAYLOAD_HASHES.keys():
        raise ValueError("incomplete compatibility input set")
    for name, digest in PAYLOAD_HASHES.items():
        helper.verify_payload(payloads[name], digest)
    return {
        "scope": "retained-candidate-inputs-not-admitted",
        "producer": PRODUCER,
        "payload_sha256": PAYLOAD_HASHES,
        "subject_digest": SUBJECT,
        "config_digest": CONFIG,
    }


def verify_loaded(inspection: list[dict]) -> str:
    """Check containerd's loaded subject, distinct from the OCI config digest."""
    if not isinstance(inspection, list) or len(inspection) != 1:
        raise ValueError("expected exactly one loaded candidate")
    image = inspection[0]
    if (
        image.get("Id"),
        image.get("Os"),
        image.get("Architecture"),
        image.get("Config", {}).get("User"),
    ) != (SUBJECT, "linux", "amd64", "10001:10001"):
        raise ValueError("loaded subject, platform or user mismatch")
    return SUBJECT


def main() -> None:
    """Emit an input or loaded-identity record only after successful verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inputs = commands.add_parser("inputs")
    inputs.add_argument("metadata", type=Path)
    for name in PAYLOAD_HASHES:
        inputs.add_argument(f"--{name}", type=Path, required=True)
    loaded = commands.add_parser("loaded")
    loaded.add_argument("inspection", type=Path)
    args = parser.parse_args()
    if args.command == "inputs":
        result = verify_inputs(args.metadata, {name: getattr(args, name) for name in PAYLOAD_HASHES})
    else:
        result = {"loaded_subject": verify_loaded(json.loads(args.inspection.read_text()))}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

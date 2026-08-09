"""Lightweight spawned-process entrypoint for bounded schema validation."""

from __future__ import annotations

import json
import struct
import sys
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Literal, TypeAlias

SchemaWorkerVerdict: TypeAlias = tuple[Literal["ok", "invalid", "internal"], str]

_MAX_VERDICT_BYTES = 4_096
_MAX_PAYLOAD_BYTES = 20_971_528
_PAYLOAD_HEADER = struct.Struct(">II")
_NO_INSTANCE = 0xFFFFFFFF
_DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"
_DRAFT_7 = "http://json-schema.org/draft-07/schema#"


def _deny_schema_retrieval(uri: str) -> Any:
    """Fail every registry retrieval without consulting network-capable defaults."""

    from referencing.exceptions import NoSuchResource

    raise NoSuchResource(ref=uri)


def _encode_verdict(verdict: SchemaWorkerVerdict) -> bytes:
    """Encode one small verdict without exposing validator diagnostics."""

    payload = json.dumps(verdict, separators=(",", ":")).encode("utf-8")
    if len(payload) > _MAX_VERDICT_BYTES:
        return b'["internal","schema_validation_worker_failed"]'
    return payload


def _send_verdict(connection: Connection, verdict: SchemaWorkerVerdict) -> None:
    """Send one small JSON verdict and never expose validator diagnostics."""

    try:
        connection.send_bytes(_encode_verdict(verdict))
    except (BrokenPipeError, EOFError, OSError):
        pass


def _validate(
    schema_json: bytes,
    instance_json: bytes | None,
    dialect: str,
) -> SchemaWorkerVerdict:
    """Compile a schema and optionally validate one value."""

    try:
        from jsonschema import Draft7Validator, Draft202012Validator
        from jsonschema.exceptions import SchemaError, ValidationError
        from referencing import Registry

        validators = {
            _DRAFT_2020_12: Draft202012Validator,
            _DRAFT_7: Draft7Validator,
        }
        try:
            schema = json.loads(schema_json)
            validator_class = validators[dialect]
            registry: Registry[Any] = Registry(retrieve=_deny_schema_retrieval)
            validator = validator_class(schema, registry=registry)
            validator_class.check_schema(schema)
            if instance_json is not None:
                validator.validate(json.loads(instance_json))
            return ("ok", "")
        except KeyError:
            return ("internal", "schema_validation_worker_failed")
        except SchemaError:
            return ("invalid", "invalid_schema")
        except ValidationError:
            return ("invalid", "schema_validation_failed")
        except Exception:  # noqa: BLE001 - reduce validator failures to a safe verdict
            return ("internal", "schema_validation_worker_failed")
    except Exception:  # noqa: BLE001 - reduce import failures to a safe verdict
        return ("internal", "schema_validation_worker_failed")


def _read_exec_payload(path: Path) -> tuple[bytes, bytes | None]:
    """Read one bounded, exact-length payload created by the parent process."""

    with path.open("rb") as payload_file:
        payload = payload_file.read(_MAX_PAYLOAD_BYTES + 1)
    if len(payload) > _MAX_PAYLOAD_BYTES or len(payload) < _PAYLOAD_HEADER.size:
        raise ValueError("invalid schema worker payload")
    schema_length, instance_length = _PAYLOAD_HEADER.unpack_from(payload)
    expected = _PAYLOAD_HEADER.size + schema_length
    if instance_length != _NO_INSTANCE:
        expected += instance_length
    if expected != len(payload):
        raise ValueError("invalid schema worker payload")
    schema_start = _PAYLOAD_HEADER.size
    schema_end = schema_start + schema_length
    instance = None if instance_length == _NO_INSTANCE else payload[schema_end:]
    return payload[schema_start:schema_end], instance


def schema_validation_worker(
    connection: Connection,
    schema_json: bytes,
    instance_json: bytes | None,
    dialect: str,
) -> None:
    """Compile a schema and optionally validate one value in a disposable child."""

    _send_verdict(connection, _validate(schema_json, instance_json, dialect))
    connection.close()


def _main(argv: list[str]) -> int:
    """Run the fixed-argument subprocess backend used on Windows."""

    verdict: SchemaWorkerVerdict = ("internal", "schema_validation_worker_failed")
    if len(argv) == 3:
        try:
            schema_json, instance_json = _read_exec_payload(Path(argv[1]))
            verdict = _validate(schema_json, instance_json, argv[2])
        except Exception:  # noqa: BLE001 - CLI must emit only a stable safe verdict
            verdict = ("internal", "schema_validation_worker_failed")
    try:
        sys.stdout.buffer.write(_encode_verdict(verdict))
        sys.stdout.buffer.flush()
    except (BrokenPipeError, OSError):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))


__all__ = ["schema_validation_worker"]

"""Lightweight spawned-process entrypoint for bounded schema validation."""

from __future__ import annotations

import json
from multiprocessing.connection import Connection
from typing import Any, Literal, TypeAlias

from jsonschema import Draft7Validator, Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError
from referencing import Registry
from referencing.exceptions import NoSuchResource

SchemaWorkerVerdict: TypeAlias = tuple[Literal["ok", "invalid", "internal"], str]

_MAX_VERDICT_BYTES = 4_096
_DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"
_DRAFT_7 = "http://json-schema.org/draft-07/schema#"
_VALIDATORS = {
    _DRAFT_2020_12: Draft202012Validator,
    _DRAFT_7: Draft7Validator,
}


def _deny_schema_retrieval(uri: str) -> Any:
    """Fail every registry retrieval without consulting network-capable defaults."""

    raise NoSuchResource(ref=uri)


def _send_verdict(connection: Connection, verdict: SchemaWorkerVerdict) -> None:
    """Send one small JSON verdict and never expose validator diagnostics."""

    payload = json.dumps(verdict, separators=(",", ":")).encode("utf-8")
    if len(payload) > _MAX_VERDICT_BYTES:
        payload = b'["internal","schema_validation_worker_failed"]'
    try:
        connection.send_bytes(payload)
    except (BrokenPipeError, EOFError, OSError):
        pass


def schema_validation_worker(
    connection: Connection,
    schema_json: bytes,
    instance_json: bytes | None,
    dialect: str,
) -> None:
    """Compile a schema and optionally validate one value in a disposable child."""

    verdict: SchemaWorkerVerdict
    try:
        schema = json.loads(schema_json)
        validator_class = _VALIDATORS[dialect]
        registry: Registry[Any] = Registry(retrieve=_deny_schema_retrieval)
        validator = validator_class(schema, registry=registry)
        validator_class.check_schema(schema)
        if instance_json is not None:
            validator.validate(json.loads(instance_json))
        verdict = ("ok", "")
    except KeyError:
        verdict = ("internal", "schema_validation_worker_failed")
    except SchemaError:
        verdict = ("invalid", "invalid_schema")
    except ValidationError:
        verdict = ("invalid", "schema_validation_failed")
    except Exception:  # noqa: BLE001 - child must reduce every validator failure to a safe verdict
        verdict = ("internal", "schema_validation_worker_failed")
    _send_verdict(connection, verdict)
    connection.close()


__all__ = ["schema_validation_worker"]

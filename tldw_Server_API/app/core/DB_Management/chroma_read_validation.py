"""Aggregate work ceilings and non-executing Rust Chroma metadata validation."""

from __future__ import annotations

# Tokenization only: never load a pickle or invoke globals/reducers.
import pickletools  # nosec B403
import sqlite3
from contextlib import closing, contextmanager
from dataclasses import dataclass, fields
from pathlib import Path

from tldw_Server_API.app.core.Embeddings.chroma_storage import ChromaStorageError


def invalid() -> ChromaStorageError:
    return ChromaStorageError("vector_corruption", retryable=False)


def resource_limit() -> ChromaStorageError:
    return ChromaStorageError("resource_limit", retryable=False)


@dataclass(frozen=True)
class ChromaPreflightLimits:
    """Aggregate per-held-session ceilings; defaults are also hard maxima."""

    collections: int = 256
    segments: int = 512
    elements: int = 100_000
    metadata_bytes: int = 33_554_432
    opcodes: int = 1_000_000
    sql_steps: int = 2_000_000
    catalog_bytes: int = 1_048_576
    queue_records: int = 100_000
    queue_components: int = 2_000_000
    vector_width: int = 4096
    returned_components: int = 2_000_000

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) is not int or not 1 <= value <= field.default:
                raise ValueError(f"Invalid Chroma preflight limit: {field.name}")


class ReadBudget:
    """One aggregate counter shared by catalogs, segments, parser and log reads."""

    def __init__(self, limits: ChromaPreflightLimits) -> None:
        self.limits = limits
        self.used: dict[str, int] = {}
        self.sql_exhausted = False

    def charge(self, name: str, amount: int = 1) -> None:
        value = self.used.get(name, 0) + amount
        if value > getattr(self.limits, name):
            raise resource_limit()
        self.used[name] = value

    def progress(self) -> int:
        value = self.used.get("sql_steps", 0) + 1
        self.used["sql_steps"] = value
        self.sql_exhausted = value > self.limits.sql_steps
        return int(self.sql_exhausted)

    def rows(self, db, sql: str, cap: int, params: tuple = ()) -> list[tuple]:
        """Bound returned rows AND UTF-8/blob materialization; VM work is separate."""
        result = []
        for row in db.execute(sql + " LIMIT ?", (*params, cap + 1)):
            if len(result) >= cap:
                raise resource_limit()
            for value in row:
                if isinstance(value, str):
                    self.charge("catalog_bytes", len(value.encode("utf-8")))
                elif isinstance(value, bytes):
                    self.charge("catalog_bytes", len(value))
            result.append(row)
        return result


@contextmanager
def existing_sqlite(path: Path, budget: ReadBudget):
    """Read without recovery/sidecar creation, with aggregate SQLite VM bounds."""
    try:
        with closing(
            sqlite3.connect((path / "chroma.sqlite3").as_uri() + "?mode=ro&immutable=1", uri=True, timeout=1)
        ) as db:
            db.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, budget.limits.catalog_bytes)
            db.set_progress_handler(budget.progress, 1)
            db.execute("PRAGMA query_only=ON")
            db.execute("PRAGMA temp_store=MEMORY")
            yield db
    except sqlite3.Error as exc:
        if budget.sql_exhausted or getattr(exc, "sqlite_errorcode", None) == sqlite3.SQLITE_TOOBIG:
            raise resource_limit() from exc
        if isinstance(exc, sqlite3.OperationalError) and ("locked" in str(exc) or "unable to open" in str(exc)):
            raise ChromaStorageError("vector_access", retryable=True) from exc
        raise invalid() from exc


class _MetadataStream:
    """Prevent declared pickle argument lengths exceeding the remaining file."""

    def __init__(self, file, size: int) -> None:
        self.file = file
        self.size = size
        self.remaining = size

    def tell(self) -> int:
        return self.size - self.remaining

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self.remaining
        if size > self.remaining:
            raise invalid()
        data = self.file.read(size)
        self.remaining -= len(data)
        return data

    def readline(self) -> bytes:
        data = self.file.readline(self.remaining)
        self.remaining -= len(data)
        return data


def read_metadata(path: Path, *, size: int, count: int, dimension: int, budget: ReadBudget) -> dict:
    """Interpret only data-dictionary opcodes, with checked stack/container types.

    Rust 1.5.5 emits protocol 3 without memoization. No GLOBAL, REDUCE, object
    construction, executable deserialization, recursive alias or arbitrary types.
    """
    budget.charge("metadata_bytes", size)
    stack = []
    marks = []
    marker = object()
    stopped = False
    with path.open("rb") as file:
        stream = _MetadataStream(file, size)
        for opcode, argument, position in pickletools.genops(stream):
            budget.charge("opcodes")
            name = opcode.name
            if position == 0 and name != "PROTO":
                raise invalid()
            if name == "PROTO":
                if position != 0 or argument != 3:
                    raise invalid()
            elif name == "EMPTY_DICT":
                stack.append({})
            elif name == "MARK":
                marks.append(len(stack))
                stack.append(marker)
            elif name in ("BINUNICODE", "BININT", "BININT1", "BININT2", "LONG1"):
                stack.append(argument)
            elif name == "NONE":
                stack.append(None)
            elif name in ("SETITEM", "SETITEMS"):
                if name == "SETITEMS":
                    if not marks:
                        raise invalid()
                    start = marks.pop()
                    if start < 1 or (len(stack) - start - 1) % 2:
                        raise invalid()
                    pairs = stack[start + 1 :]
                    del stack[start:]
                else:
                    if len(stack) < 3 or (marks and marks[-1] >= len(stack) - 2):
                        raise invalid()
                    pairs = stack[-2:]
                    del stack[-2:]
                if not stack or not isinstance(stack[-1], dict):
                    raise invalid()
                target = stack[-1]
                for i in range(0, len(pairs), 2):
                    key, value = pairs[i : i + 2]
                    if type(key) not in (str, int) or key in target:
                        raise invalid()
                    target[key] = value
            elif name == "STOP":
                if marks or len(stack) != 1 or not isinstance(stack[0], dict) or stream.remaining:
                    raise invalid()
                stopped = True
            else:
                raise invalid()
            if len(stack) > 2 * budget.limits.elements + 64:
                raise resource_limit()
    if not stopped:
        raise invalid()
    result = stack[0]
    keys = {"dimensionality", "total_elements_added", "max_seq_id", "id_to_label", "label_to_id", "id_to_seq_id"}
    if (
        set(result) != keys
        or type(result["total_elements_added"]) is not int
        or result["total_elements_added"] != count
    ):
        raise invalid()
    if result["dimensionality"] is not None and (
        type(result["dimensionality"]) is not int or result["dimensionality"] != dimension
    ):
        raise invalid()
    if result["max_seq_id"] is not None and (type(result["max_seq_id"]) is not int or result["max_seq_id"] < 0):
        raise invalid()
    ids, labels, sequences = (result[key] for key in ("id_to_label", "label_to_id", "id_to_seq_id"))
    if any(type(value) is not dict or len(value) > count for value in (ids, labels, sequences)):
        raise invalid()
    if any(type(key) is not str or type(value) is not int or not 0 < value <= count for key, value in ids.items()):
        raise invalid()
    if any(type(key) is not int or type(value) is not str for key, value in labels.items()):
        raise invalid()
    if len(labels) != len(ids) or any(labels.get(label) != key for key, label in ids.items()):
        raise invalid()
    if any(
        type(key) is not str or key not in ids or type(value) is not int or value < 0
        for key, value in sequences.items()
    ):
        raise invalid()
    return result

"""Process-level de-duplication for idempotent schema setup.

``ensure_schema()`` implementations are idempotent -- they are written with
``CREATE TABLE IF NOT EXISTS`` and guarded ``ALTER TABLE`` -- but they are not
free. Running one issues tens to hundreds of DDL statements.

Several repositories are constructed per request and ensure their schema on
construction, so that cost landed on every request:

    GET /api/v1/scheduled-tasks    229 SQL statements, 175 of them DDL
    GET /api/v1/watchlists/items    73 SQL statements,  61 of them DDL
    GET /api/v1/rpg/rules/adapters  19 SQL statements,  10 of them DDL

This does not change *whether* the schema is ensured, only how often. The first
call for a given database still runs the full setup; later ones in the same
process replace it with a single catalogue lookup.

Only wrap the schema setup itself. Anything that reconciles ongoing state --
re-seeding rows from files that can change on disk, for instance -- must keep
running on its own schedule, because "the tables exist" says nothing about
whether that state is current.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from collections.abc import Callable
from urllib.parse import unquote, urlsplit

from loguru import logger

# Databases are identified by (scope, path, device, inode).
_Identity = tuple[str, str, int, int]

# Per-database bookkeeping. ``done`` guards the memo; ``lock`` serializes
# concurrent first-touches of the same database so they cannot each run the
# full DDL.
class _Entry:
    __slots__ = ("done", "lock")

    def __init__(self) -> None:
        self.done = False
        self.lock = threading.Lock()


# Bounded so a long-lived multi-user process that opens a database per user does
# not accumulate entries forever. Eviction is least-recently-used; an evicted
# database simply pays for its setup again on next use.
_MAX_TRACKED_DATABASES = 4096

_ENTRIES: OrderedDict[_Identity, _Entry] = OrderedDict()
_ENTRIES_LOCK = threading.Lock()


def _resolve_path(text: str) -> str | None:
    """Reduce a SQLite path or URI to something ``os.stat`` accepts.

    Returns ``None`` for anything that is not a real file, which disables
    memoization for that caller -- the safe direction, since an in-memory
    database is distinct per connection despite sharing a name.
    """
    if not text or text == ":memory:":
        return None
    if not text.startswith("file:"):
        return text

    # file:/srv/app.db?mode=rwc -> /srv/app.db, and file::memory:... -> None
    split = urlsplit(text)
    path = unquote(split.path)
    if not path or ":memory:" in text[: text.find("?") if "?" in text else len(text)]:
        return None
    return path


def _identity(scope: str, path: str | os.PathLike[str] | None) -> _Identity | None:
    """Build a stable identity for a database file, or ``None`` if unsafe to cache."""
    if not path:
        return None
    resolved = _resolve_path(str(path))
    if resolved is None:
        return None
    try:
        stat = os.stat(resolved)
    except OSError:
        return None
    return (scope, resolved, stat.st_dev, stat.st_ino)


def _entry_for(identity: _Identity) -> _Entry:
    """Return this database's bookkeeping, creating and aging it as needed."""
    with _ENTRIES_LOCK:
        entry = _ENTRIES.get(identity)
        if entry is None:
            entry = _Entry()
            _ENTRIES[identity] = entry
            while len(_ENTRIES) > _MAX_TRACKED_DATABASES:
                _ENTRIES.popitem(last=False)
        else:
            _ENTRIES.move_to_end(identity)
        return entry


def _schema_still_present(verify: Callable[[], bool], identity: _Identity) -> bool:
    """Run the caller's check, treating a failed check as "rebuild"."""
    try:
        return bool(verify())
    except Exception as exc:  # noqa: BLE001 - any failure means we cannot trust the memo
        logger.warning(
            "schema_once: verification failed for {} ({}); re-running schema setup",
            identity[1],
            exc,
        )
        return False


def ensure_once(
    scope: str,
    path: str | os.PathLike[str] | None,
    ensure: Callable[[], None],
    *,
    verify: Callable[[], bool],
) -> None:
    """Run ``ensure`` the first time this database is seen in this process.

    Args:
        scope: Distinguishes callers that share a database file but own
            different tables, so one caller's setup cannot satisfy another's.
        path: Filesystem path or SQLite URI of the database. When it cannot be
            resolved to a real file the call falls through to ``ensure`` every
            time, which is the safe direction.
        ensure: The idempotent schema routine to run. It must set up schema
            only; see the module docstring.
        verify: Cheap check that the schema really is present, run on every
            memo hit. Required, because path/device/inode is not sufficient on
            its own: a filesystem may hand back the same inode for a file
            recreated at the same path, and the memo would then skip setup for
            a database that no longer has the tables. One catalogue query is
            still far cheaper than replaying the whole schema. A check that
            raises is treated as a failed check.

    Returns:
        None.
    """
    identity = _identity(scope, path)
    if identity is None:
        ensure()
        return

    entry = _entry_for(identity)
    if entry.done and _schema_still_present(verify, identity):
        return

    # Serialize first-touches of this database so concurrent callers do not each
    # replay the DDL. (An entry evicted while its lock is held is replaced by a
    # fresh one; the worst case is a duplicate run of an idempotent routine.)
    with entry.lock:
        if entry.done and _schema_still_present(verify, identity):
            return
        entry.done = False
        ensure()
        # Only remember setups that completed; a failure must be retried.
        entry.done = True


def reset(scope: str | None = None) -> None:
    """Forget completed setups, so the next call runs the full schema routine.

    For tests that delete and recreate databases in place.

    Args:
        scope: Forget only this scope's entries. When ``None``, forget every
            scope.

    Returns:
        None.
    """
    with _ENTRIES_LOCK:
        if scope is None:
            _ENTRIES.clear()
            return
        for identity in [key for key in _ENTRIES if key[0] == scope]:
            del _ENTRIES[identity]

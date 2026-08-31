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
construction for a given database still runs the full setup; later ones in the
same process skip it.

The key includes the file's device and inode, so a database that is deleted and
recreated at the same path -- test teardown, a restored backup -- is set up
again rather than assumed. Databases with no resolvable path are never
memoized, which keeps in-memory databases (distinct per connection despite
sharing a name) correct by default.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable

_COMPLETED: set[tuple[str, str, int, int]] = set()
_LOCK = threading.Lock()


def _identity(scope: str, path: str | os.PathLike[str] | None) -> tuple[str, str, int, int] | None:
    """Build a stable identity for a database file, or ``None`` if unsafe to cache."""
    if not path:
        return None
    text = str(path)
    if not text or text == ":memory:" or text.startswith("file::memory:"):
        return None
    try:
        stat = os.stat(text)
    except OSError:
        return None
    return (scope, text, stat.st_dev, stat.st_ino)


def ensure_once(
    scope: str,
    path: str | os.PathLike[str] | None,
    ensure: Callable[[], None],
    *,
    verify: Callable[[], bool] | None = None,
) -> None:
    """Run ``ensure`` the first time this database is seen in this process.

    Args:
        scope: Distinguishes callers that share a database file but own
            different tables, so one caller's setup cannot satisfy another's.
        path: Filesystem path of the database. When it cannot be resolved the
            call falls through to ``ensure`` every time, which is the safe
            direction.
        ensure: The idempotent schema routine to run.
        verify: Optional cheap check that the schema really is present. Supply
            this whenever a database may be deleted and recreated underneath a
            live process: path/device/inode is not sufficient on its own,
            because a filesystem may hand back the same inode for a file
            recreated at the same path. When the memo hits and ``verify``
            returns False, the full setup runs again. One existence query is
            still far cheaper than replaying the whole schema.
    """
    identity = _identity(scope, path)
    if identity is not None:
        with _LOCK:
            remembered = identity in _COMPLETED
        if remembered:
            if verify is None:
                return
            try:
                if verify():
                    return
            except Exception:
                pass  # fall through and rebuild
            with _LOCK:
                _COMPLETED.discard(identity)

    ensure()

    # Only remember setups that completed; a failure must be retried.
    if identity is not None:
        with _LOCK:
            _COMPLETED.add(identity)


def reset(scope: str | None = None) -> None:
    """Forget completed setups, for tests that recreate databases in place."""
    with _LOCK:
        if scope is None:
            _COMPLETED.clear()
            return
        for entry in [e for e in _COMPLETED if e[0] == scope]:
            _COMPLETED.discard(entry)

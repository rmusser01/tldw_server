"""Test apps over PostgreSQL ChaCha install the middleware main.py installs (TASK-13861 AC4).

Without ChaChaOperationMiddleware every request falls back to a thread-local
PostgreSQL checkout; TASK-13841's 5-connection pool ran out that way.

It checks files that mention postgres (any case), pg_restricted_backend or
pg_database_config, and CharactersRAGDB or get_chacha_db_for_user. In each, every
``<name> = FastAPI(...)`` needs a matching ``<name>.add_middleware(ChaChaOperationMiddleware)``.
An app built another way (a helper, a factory fixture) is not checked.
"""

import re
from collections import Counter
from pathlib import Path

_TESTS = Path(__file__).resolve().parents[1]
_POSTGRES = re.compile(r"postgres|pg_restricted_backend|pg_database_config", re.IGNORECASE)
_BUILDER = re.compile(r"^[ \t]*(\w+) = FastAPI\(", re.MULTILINE)
_INSTALLED = re.compile(r"^[ \t]*(\w+)\.add_middleware\(ChaChaOperationMiddleware\)[ \t]*$", re.MULTILINE)


def test_every_postgres_chacha_test_app_installs_the_operation_middleware():
    missing = []
    for path in sorted(_TESTS.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        builders = Counter(_BUILDER.findall(text))
        if not builders or not _POSTGRES.search(text):
            continue
        if not re.search(r"CharactersRAGDB|get_chacha_db_for_user", text):
            continue
        installed = Counter(_INSTALLED.findall(text))
        if any(installed[name] < count for name, count in builders.items()):
            missing.append(str(path.relative_to(_TESTS)))

    assert missing == [], f"Add <app>.add_middleware(ChaChaOperationMiddleware) after <app> = FastAPI() in: {missing}"

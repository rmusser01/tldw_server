"""Test apps over PostgreSQL ChaCha install the middleware main.py installs (TASK-13861 AC4).

Without ChaChaOperationMiddleware every request falls back to a thread-local
PostgreSQL checkout; TASK-13841's 5-connection pool ran out that way.

It recognises the literal ``app = FastAPI()`` in files that mention postgres and
CharactersRAGDB or get_chacha_db_for_user, which is how every such test app is built
today. An app built another way is not checked.
"""

import re
from pathlib import Path

_TESTS = Path(__file__).resolve().parents[1]
_BUILDER = re.compile(r"^[ \t]*app = FastAPI\(\)[ \t]*$", re.MULTILINE)
_INSTALLED = re.compile(r"^[ \t]*app\.add_middleware\(ChaChaOperationMiddleware\)[ \t]*$", re.MULTILINE)


def test_every_postgres_chacha_test_app_installs_the_operation_middleware():
    missing = []
    for path in sorted(_TESTS.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        builders = len(_BUILDER.findall(text))
        if not builders or "postgres" not in text:
            continue
        if not re.search(r"CharactersRAGDB|get_chacha_db_for_user", text):
            continue
        if len(_INSTALLED.findall(text)) < builders:
            missing.append(str(path.relative_to(_TESTS)))

    assert missing == [], f"Add app.add_middleware(ChaChaOperationMiddleware) after app = FastAPI() in: {missing}"

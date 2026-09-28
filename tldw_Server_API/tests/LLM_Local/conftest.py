"""Shared stand-ins for the Llama.cpp endpoint suites.

`/llamacpp/inference` authenticates like every other route on its router, so
each test app has to supply a caller. Defining that stand-in once keeps the
inference and runtime suites from drifting apart.
"""

from __future__ import annotations

from typing import Any

import pytest

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import llamacpp as lp
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User


def llamacpp_test_user() -> User:
    """Return the caller the Llama.cpp endpoint suites authenticate as."""
    return User(id=1, username="llamacpp-test-user", email=None, is_active=True)


def override_llamacpp_request_user(app: Any) -> None:
    """Point `get_request_user` at a stand-in caller for *app*.

    Overrides both the `auth_deps` symbol and the one bound into the endpoint
    module, because FastAPI resolves dependencies by identity.
    """

    async def _fake_get_request_user() -> User:
        return llamacpp_test_user()

    app.dependency_overrides[auth_deps.get_request_user] = _fake_get_request_user
    app.dependency_overrides[lp.get_request_user] = _fake_get_request_user


@pytest.fixture()
def llamacpp_user() -> User:
    """The stand-in caller, for tests that need to assert on it."""
    return llamacpp_test_user()

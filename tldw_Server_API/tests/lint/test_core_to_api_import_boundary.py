"""Ratchet: core/ must not grow new dependencies on the api/ layer.

Docs/Architecture.md states the direction:

    Clients -> FastAPI endpoints -> Core domain services -> Databases / ...

and "keep endpoints thin, push logic into core modules". A core module importing
from app/api/ inverts that. The 2026-09-21 core-module review found two distinct
shapes, which are not equally bad:

* Schema-only imports from ``api/v1/schemas/*`` -- the common case. The honest reading
  is that those schemas are in the wrong package, not that core is wrong to need them.
* True inversions: core importing ``api/v1/endpoints/*`` or ``api/v1/API_Deps/*``.
  ``Ingestion_Media_Processing/persistence.py`` has one: the background embedding task
  that ``schedule_media_add_embeddings`` schedules (``generate_embeddings_task``) lazily
  imports ``generate_embeddings_for_media`` and ``get_media_content`` from
  ``api.v1.endpoints.media_embeddings``, so core calls into an endpoint module.

Pair-based since TASK-13950.9. This is the core -> api view of the layering contract
(test_layering_contract.py), which counts module- and function-level import statements
per (core package, api package) pair in layering_baseline.txt. A new pair or a higher
count fails and prints the importing modules; a count that fell is stale until its line
is lowered or deleted, so the number only goes down. A new pair into api.v1.endpoints.*
or api.v1.API_Deps is a new true inversion.

Companion to tests/lint/test_endpoint_auth_deps_import_boundary.py, which guards the
OPPOSITE direction. Note those two currently describe a real cycle: that test bans
endpoints from importing core.AuthNZ.User_DB_Handling, while
core/AuthNZ/User_DB_Handling.py imports oauth2_scheme from api/v1/API_Deps.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TypeVar

import pytest

from tldw_Server_API.tests.lint.test_layering_contract import BASELINE, Pair, load, measure, violations

pytestmark = pytest.mark.unit

_V = TypeVar("_V")


def _core_to_api(entries: Mapping[Pair, _V]) -> dict[Pair, _V]:
    return {
        pair: value
        for pair, value in entries.items()
        if (pair[0].split(".")[0], pair[1].split(".")[0]) == ("core", "api")
    }


def test_a_seeded_core_to_api_import_fails(tmp_path: Path) -> None:
    module = tmp_path / "app" / "core" / "Chat" / "x.py"
    module.parent.mkdir(parents=True)
    module.write_text("def f():\n    from tldw_Server_API.app.api.v1.endpoints.media import y\n", encoding="utf-8")

    (problem,) = violations(_core_to_api(measure(tmp_path / "app")), {})

    assert problem.startswith("core.Chat -> api.v1.endpoints.media: 1 statement, not in the baseline"), problem


def test_core_to_api_imports_only_fall() -> None:
    problems = violations(_core_to_api(measure()), _core_to_api(load(BASELINE)))
    assert not problems, (
        "core -> api imports changed; core/ must not depend on app/api/ (Docs/Architecture.md). "
        "A schema: move it into core. api.v1.endpoints.* or api.v1.API_Deps (core reaching into the "
        "transport layer): inject the dependency instead.\n" + "\n".join(problems)
    )

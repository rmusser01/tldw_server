"""Record what a caller hands to ``RagRunConfig.from_kwargs`` (TASK-13950.24 PR M).

``rcfg.as_kwargs()`` fills every field with its default, so it cannot show that a caller passed a
value explicitly. Assert "the caller passes X" against this spy instead.
"""

from __future__ import annotations

from typing import Any

import pytest

from tldw_Server_API.app.core.RAG.rag_service.pipeline.run_config import RagRunConfig


def spy_from_kwargs(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Wrap the real ``RagRunConfig.from_kwargs``; the returned dict accumulates the kwargs it receives."""
    passed: dict[str, Any] = {}
    real = RagRunConfig.from_kwargs

    def spy(**kwargs: Any) -> RagRunConfig:
        passed.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(RagRunConfig, "from_kwargs", spy)
    return passed

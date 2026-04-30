"""
Payload exemplars sampler for failed adaptive checks.

Writes small, redacted samples of query and contexts to a JSONL sink for
operators to debug retrieval/generation issues without leaking sensitive data.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import time
from pathlib import Path
from typing import Any

from loguru import logger

BASE_DIR = Path("Databases/observability")
SINK = BASE_DIR / "rag_payload_exemplars.jsonl"


def _sampling_random() -> float:
    return secrets.SystemRandom().random()


def _enforce_base_dir(candidate: Path) -> Path:
    base_dir = BASE_DIR.resolve()
    try:
        resolved = candidate.expanduser().resolve()
    except Exception:
        return SINK
    if not resolved.is_relative_to(base_dir):
        logger.warning("RAG payload exemplar path outside base dir; using default path.")
        return SINK
    return resolved


def _safe_sink(user_id: str | None = None, namespace: str | None = None) -> Path:
    try:
        # Prefer explicit override when provided
        p = os.getenv("RAG_PAYLOAD_EXEMPLAR_PATH")
        if p:
            sink = _enforce_base_dir(Path(p))
        else:
            # In multi-tenant setups, segregate exemplars per user/namespace
            if namespace:
                safe_namespace = "".join(c for c in str(namespace) if c.isalnum() or c in ('-', '_', '.'))
                sink = BASE_DIR / "tenants" / safe_namespace / "rag_payload_exemplars.jsonl" if safe_namespace else SINK
            elif user_id:
                safe_user_id = "".join(c for c in str(user_id) if c.isalnum() or c in ('-', '_', '.'))
                sink = BASE_DIR / "users" / safe_user_id / "rag_payload_exemplars.jsonl" if safe_user_id else SINK
            else:
                sink = SINK
        sink.parent.mkdir(parents=True, exist_ok=True)
        return sink
    except Exception:
        return SINK


def _redact(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text
    # Redact emails/URLs/numbers-long
    t = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "[EMAIL]", t)
    t = re.sub(r"https?://[^\s]+", "[URL]", t)
    t = re.sub(r"\b\d{4,}\b", "[NUM]", t)
    # Collapse whitespace and truncate
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > 400:
        t = t[:400] + "…"
    return t


def maybe_record_exemplar(
    *,
    query: str,
    documents: list[Any],
    answer: str,
    reason: str,
    user_id: str | None = None,
    namespace: str | None = None,
) -> None:
    """Sample and persist a redacted exemplar of the payload.

    Sampling rate is controlled by RAG_PAYLOAD_EXEMPLAR_SAMPLING (0..1, default 0.05).
    """
    try:
        rate = float(os.getenv("RAG_PAYLOAD_EXEMPLAR_SAMPLING", "0.05"))
    except Exception:
        rate = 0.05
    try:
        if _sampling_random() > max(0.0, min(1.0, rate)):
            return
        sink = _safe_sink(user_id=user_id, namespace=namespace)
        sample = {
            "ts": time.time(),
            "reason": reason,
            "user": user_id or "",
            "namespace": namespace or "",
            "query": _redact(query or ""),
            "answer": _redact(answer or ""),
            "docs": [
                {
                    "id": getattr(d, "id", None) or (d.get("id") if isinstance(d, dict) else None),
                    "score": float(getattr(d, "score", 0.0) if hasattr(d, "score") else (d.get("score", 0.0) if isinstance(d, dict) else 0.0)),
                    "content": _redact(getattr(d, "content", "") if hasattr(d, "content") else (d.get("content", "") if isinstance(d, dict) else "")),
                }
                for d in (documents[:5] if documents else [])
            ],
        }
        with sink.open("a", encoding="utf-8") as f:
            f.write(json.dumps(sample) + "\n")
    except Exception:
        logger.debug("Failed to write exemplar")

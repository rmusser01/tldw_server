"""
Per-user lightweight personalization store for RAG.

- Stores doc-level priors and implicit interactions (click/expand/copy)
  under <USER_DB_BASE_DIR>/<user_id>/rag_personalization.json (defaults to repo-root Databases/user_databases)
- Provides a simple boosting function to adjust document scores.

This avoids cross-tenant leakage by isolating data per user.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths


@dataclass
class Prior:
    score: float
    last_updated: float
    corpus: str | None = None


_MAX_EVENT_LOG = 200
_MAX_LIST_ITEMS = 50


def _empty_store() -> dict[str, Any]:
    return {"priors": {}, "events": {}, "pairs": {}, "event_log": []}


class UserPersonalizationStore:
    def __init__(self, user_id: str | None) -> None:
        self.path = DatabasePaths.get_user_rag_personalization_path(user_id)
        # Use the sanitized directory name as the canonical user id for storage/logging.
        self.user_id = self.path.parent.name
        self._data: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        try:
            if self.path.exists():
                with self.path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    self._data = data
                else:
                    self._data = _empty_store()
            else:
                self._data = _empty_store()
            self._data.setdefault("priors", {})
            self._data.setdefault("events", {})
            self._data.setdefault("pairs", {})
            self._data.setdefault("event_log", [])
        except Exception:
            logger.warning("Failed loading personalization data")
            self._data = _empty_store()

    def _save(self) -> None:
        try:
            with self.path.open("w", encoding="utf-8") as f:
                json.dump(self._data, f)
        except Exception:
            logger.debug("Failed saving personalization")

    def record_event(
        self,
        *,
        event_type: str,
        doc_id: str | None,
        corpus: str | None = None,
        impression: list[str] | None = None,
        chunk_ids: list[str] | None = None,
        rank: int | None = None,
        session_id: str | None = None,
        conversation_id: str | None = None,
        message_id: str | None = None,
        dwell_ms: int | None = None,
        query: str | None = None,
    ) -> None:
        now = time.time()
        # Count events
        evt = self._data.setdefault("events", {})
        evt[event_type] = int(evt.get(event_type, 0)) + 1
        # Update priors modestly
        if doc_id:
            priors: dict[str, dict[str, Any]] = self._data.setdefault("priors", {})
            cur: dict[str, Any] = priors.get(doc_id) or {"score": 0.0, "last_updated": 0.0, "corpus": corpus}
            cur["score"] = min(5.0, float(cur.get("score", 0.0)) + (0.1 if event_type == "expand" else 0.2))
            cur["last_updated"] = now
            if corpus:
                cur["corpus"] = corpus
            priors[doc_id] = cur
        # Pairwise updates: clicked doc wins over docs ranked above but not clicked
        if event_type in {"click", "copy"} and doc_id and impression:
            pairs: dict[str, int] = self._data.setdefault("pairs", {})
            for other in impression:
                if other == doc_id:
                    break  # only compare with items above in the list
                key = f"{doc_id}|{other}"
                pairs[key] = int(pairs.get(key, 0)) + 1

        impression_list = list(impression or [])
        if len(impression_list) > _MAX_LIST_ITEMS:
            impression_list = impression_list[:_MAX_LIST_ITEMS]
        chunk_list = list(chunk_ids or [])
        if len(chunk_list) > _MAX_LIST_ITEMS:
            chunk_list = chunk_list[:_MAX_LIST_ITEMS]

        event_log = self._data.setdefault("event_log", [])
        event_log.append(
            {
                "ts": now,
                "event_type": event_type,
                "doc_id": doc_id,
                "chunk_ids": chunk_list,
                "rank": rank,
                "dwell_ms": dwell_ms,
                "session_id": session_id,
                "conversation_id": conversation_id,
                "message_id": message_id,
                "corpus": corpus,
                "query": query,
                "impression_list": impression_list,
            }
        )
        if len(event_log) > _MAX_EVENT_LOG:
            self._data["event_log"] = event_log[-_MAX_EVENT_LOG:]
        self._save()

    def get_prior(self, doc_id: str) -> float:
        try:
            p = self._data.get("priors", {}).get(doc_id)
            if not p:
                return 0.0
            # Simple recency decay (half-life ~ 7 days)
            half_life_days = float(os.getenv("RAG_PERSONALIZATION_HALF_LIFE_DAYS", "7"))
            dt = max(0.0, time.time() - float(p.get("last_updated", 0.0)))
            decay = pow(0.5, dt / (half_life_days * 86400.0)) if half_life_days > 0 else 1.0
            return float(p.get("score", 0.0)) * float(decay)
        except Exception:
            return 0.0

    def boost_documents(self, documents: list[Any], *, corpus: str | None = None) -> list[Any]:
        # Adjust scores by a bounded additive boost based on priors
        try:
            weight = float(os.getenv("RAG_PERSONALIZATION_WEIGHT", "0.1"))
        except Exception:
            weight = 0.1
        boosted = []
        for d in documents:
            try:
                if hasattr(d, "id"):
                    did = d.id
                elif isinstance(d, dict):
                    did = d.get("id")
                else:
                    did = None

                if hasattr(d, "score"):
                    base = float(d.score)
                elif isinstance(d, dict):
                    base = float(d.get("score", 0.0))
                else:
                    base = 0.0
                prior_data = self._data.get("priors", {}).get(str(did)) if did is not None else None
                if corpus and prior_data and prior_data.get("corpus") != corpus:
                    prior = 0.0
                else:
                    prior = self.get_prior(str(did)) if did is not None else 0.0
                new_score = base + (weight * prior)
                if hasattr(d, "score"):
                    d.score = new_score
                elif isinstance(d, dict):
                    d["score"] = new_score
                boosted.append(d)
            except Exception:
                boosted.append(d)
        # Keep ordering by updated score
        with contextlib.suppress(Exception):
            boosted.sort(key=lambda x: getattr(x, "score", x.get("score", 0.0) if isinstance(x, dict) else 0.0), reverse=True)
        return boosted

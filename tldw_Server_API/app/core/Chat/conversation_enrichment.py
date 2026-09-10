from __future__ import annotations

import hashlib
import os
import re
import threading
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone

from loguru import logger

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    ConflictError,
    InputError,
)
from tldw_Server_API.app.core.Notes.organization_capture import (
    active_coordinator,
    replace_keywords,
)
from tldw_Server_API.app.core.testing import is_test_mode

AUTO_TAG_MIN_NEW_MESSAGES = 3
AUTO_TAG_MAX_KEYWORDS = 6
AUTO_TAG_MESSAGE_WINDOW = 200

CLUSTER_ID_OPT_OUT = "opt-out"
CLUSTER_ID_UNCLUSTERED = "unclustered"

_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "your",
    "you",
    "are",
    "was",
    "were",
    "have",
    "has",
    "had",
    "not",
    "but",
    "can",
    "could",
    "should",
    "would",
    "about",
    "into",
    "when",
    "then",
    "than",
    "them",
    "they",
    "their",
    "our",
    "out",
    "who",
    "what",
    "where",
    "why",
    "how",
    "use",
    "using",
    "used",
    "any",
    "all",
    "via",
    "per",
    "just",
    "more",
    "most",
    "some",
    "such",
    "each",
    "will",
    "also",
    "able",
    "like",
}


@dataclass(frozen=True)
class AutoTagResult:
    conversation_id: str
    updated: bool
    reason: str
    topic_label: str | None
    keywords: list[str]


@dataclass(frozen=True)
class ClusterResult:
    updated_conversations: int
    skipped_opt_out: int
    clusters_written: int


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]{3,}")


def _should_run_inline() -> bool:
    return is_test_mode() or os.getenv("PYTEST_CURRENT_TEST") is not None


def _normalize_topic_label(label: str) -> str:
    return re.sub(r"\s+", " ", label.strip().lower())


def _cluster_id_for_label(label: str) -> str:
    normalized = _normalize_topic_label(label)
    slug = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")
    slug = slug[:32] or "topic"
    digest = hashlib.sha1(normalized.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]
    return f"topic-{slug}-{digest}"


def _extract_keywords(texts: list[str], max_keywords: int) -> list[str]:
    counts: Counter[str] = Counter()
    for text in texts:
        for token in _TOKEN_PATTERN.findall(text.lower()):
            if token in _STOPWORDS:
                continue
            counts[token] += 1

    if not counts:
        return []
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [token for token, _ in ranked[:max_keywords]]


def _replace_conversation_keywords(
    db: CharactersRAGDB,
    conversation_id: str,
    keywords: list[str],
    *,
    owner_user_id: int | str,
    coordinator=None,
    preflighted: bool = False,
) -> None:
    if not preflighted:
        coordinator = active_coordinator(db, user_id=owner_user_id)
    if coordinator is not None:
        replace_keywords(
            coordinator,
            subject_type="conversation",
            subject_id=conversation_id,
            keywords=keywords,
            source="conversation-enrichment",
        )
        return
    existing = db.get_keywords_for_conversation(conversation_id)
    existing_map = {
        str(k.get("keyword") or "").strip().lower(): int(k.get("id"))
        for k in existing
        if k.get("id")
    }
    target = {str(k).strip() for k in keywords if k is not None and str(k).strip()}
    target_map = {t.lower(): t for t in target}

    for key, kw_id in existing_map.items():
        if key not in target_map:
            try:
                db.unlink_conversation_from_keyword(conversation_id, kw_id)
            except Exception as exc:
                logger.warning("Failed to unlink keyword {} from {}: {}", kw_id, conversation_id, exc)

    for key, original in target_map.items():
        if key in existing_map:
            continue
        try:
            kw = db.get_keyword_by_text(original)
            if not kw:
                kw_id = db.add_keyword(original)
                kw = db.get_keyword_by_id(kw_id) if kw_id is not None else None
            if kw and kw.get("id") is not None:
                db.link_conversation_to_keyword(conversation_id, int(kw["id"]))
        except Exception as exc:
            logger.warning("Failed to link keyword {} to {}: {}", original, conversation_id, exc)


def _load_recent_message_texts(
    db: CharactersRAGDB,
    conversation_id: str,
    message_window: int,
) -> list[str]:
    total_messages = db.count_messages_for_conversation(conversation_id)
    offset = max(total_messages - message_window, 0)
    messages = db.get_messages_for_conversation(
        conversation_id,
        limit=message_window,
        offset=offset,
        order_by_timestamp="ASC",
    )
    texts: list[str] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            texts.append(content.strip())
    return texts


def _update_conversation_with_retry(
    db: CharactersRAGDB,
    conversation_id: str,
    update_data: dict[str, str | None],
    max_attempts: int = 3,
) -> bool:
    for attempt in range(1, max_attempts + 1):
        conversation = db.get_conversation_by_id(conversation_id)
        if not conversation:
            return False
        expected_version = conversation.get("version")
        if not isinstance(expected_version, int):
            return False
        try:
            return bool(db.update_conversation(conversation_id, update_data, expected_version))
        except ConflictError:
            if attempt >= max_attempts:
                return False
        except (InputError, Exception):
            return False
    return False


def auto_tag_conversation(
    db: CharactersRAGDB,
    conversation_id: str,
    *,
    owner_user_id: int | str,
    force: bool = False,
    min_new_messages: int = AUTO_TAG_MIN_NEW_MESSAGES,
    trigger_clustering: bool = True,
) -> AutoTagResult:
    conversation = db.get_conversation_by_id(conversation_id)
    if not conversation:
        return AutoTagResult(conversation_id, False, "missing_conversation", None, [])

    new_messages = db.count_messages_since(
        conversation_id,
        conversation.get("topic_last_tagged_message_id"),
    )
    if not force and new_messages < min_new_messages:
        return AutoTagResult(conversation_id, False, "insufficient_new_messages", None, [])

    latest_message = db.get_latest_message_for_conversation(conversation_id)
    if not latest_message:
        return AutoTagResult(conversation_id, False, "no_messages", None, [])

    topic_source = (conversation.get("topic_label_source") or "").strip().lower()
    manual_override = topic_source == "manual"

    now = datetime.now(timezone.utc).isoformat()
    last_message_id = latest_message.get("id")
    last_message_id = str(last_message_id) if last_message_id else None

    if manual_override and not force:
        _update_conversation_with_retry(
            db,
            conversation_id,
            {
                "topic_last_tagged_at": now,
                "topic_last_tagged_message_id": last_message_id,
            },
        )
        return AutoTagResult(conversation_id, False, "manual_override", None, [])

    texts = _load_recent_message_texts(db, conversation_id, AUTO_TAG_MESSAGE_WINDOW)
    keywords = _extract_keywords(texts, AUTO_TAG_MAX_KEYWORDS)

    title = conversation.get("title")
    topic_label = title.strip() if isinstance(title, str) and title.strip() else None
    if not topic_label and keywords:
        topic_label = keywords[0].replace("-", " ").title()

    keyword_coordinator = active_coordinator(db, user_id=owner_user_id)

    update_ok = _update_conversation_with_retry(
        db,
        conversation_id,
        {
            "topic_label": topic_label,
            "topic_label_source": "auto",
            "topic_last_tagged_at": now,
            "topic_last_tagged_message_id": last_message_id,
        },
    )

    if update_ok:
        _replace_conversation_keywords(
            db,
            conversation_id,
            keywords,
            owner_user_id=owner_user_id,
            coordinator=keyword_coordinator,
            preflighted=True,
        )

    if trigger_clustering and update_ok:
        schedule_conversation_clustering(db)

    return AutoTagResult(conversation_id, update_ok, "updated" if update_ok else "update_failed", topic_label, keywords)


def cluster_conversations_for_user(
    db: CharactersRAGDB,
    *,
    client_id: str | None = None,
    allow_opt_out: bool = True,
) -> ClusterResult:
    conversations = db.search_conversations(None, client_id=client_id)
    clusters: dict[str, dict[str, object]] = {}
    updated = 0
    skipped_opt_out = 0

    for conversation in conversations:
        conv_id = conversation.get("id")
        if not conv_id:
            continue
        current_cluster_id = conversation.get("cluster_id")
        if allow_opt_out and current_cluster_id == CLUSTER_ID_OPT_OUT:
            skipped_opt_out += 1
            continue

        label = conversation.get("topic_label")
        if isinstance(label, str) and label.strip():
            normalized_label = label.strip()
            cluster_id = _cluster_id_for_label(normalized_label)
            cluster_title = normalized_label
        else:
            cluster_id = CLUSTER_ID_UNCLUSTERED
            cluster_title = "Unclustered"

        bucket = clusters.setdefault(cluster_id, {"title": cluster_title, "members": []})
        bucket["members"].append(conv_id)

        if current_cluster_id != cluster_id:
            if _update_conversation_with_retry(db, conv_id, {"cluster_id": cluster_id}):
                updated += 1

    for cluster_id, data in clusters.items():
        title = data.get("title") if isinstance(data.get("title"), str) else None
        members = data.get("members") if isinstance(data.get("members"), list) else []
        db.upsert_conversation_cluster(cluster_id, title=title, centroid=None, size=len(members))

    return ClusterResult(updated, skipped_opt_out, len(clusters))


def schedule_auto_tagging(
    db: CharactersRAGDB,
    conversation_id: str,
    *,
    owner_user_id: int | str,
    force: bool = False,
    min_new_messages: int = AUTO_TAG_MIN_NEW_MESSAGES,
) -> None:
    if _should_run_inline():
        auto_tag_conversation(
            db,
            conversation_id,
            owner_user_id=owner_user_id,
            force=force,
            min_new_messages=min_new_messages,
        )
        return

    def _runner() -> None:
        try:
            auto_tag_conversation(
                db,
                conversation_id,
                owner_user_id=owner_user_id,
                force=force,
                min_new_messages=min_new_messages,
            )
        except Exception as exc:
            logger.warning("Auto-tagging job failed for {}: {}", conversation_id, exc)

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()


def schedule_conversation_clustering(
    db: CharactersRAGDB,
    *,
    client_id: str | None = None,
) -> None:
    if _should_run_inline():
        cluster_conversations_for_user(db, client_id=client_id)
        return

    def _runner() -> None:
        try:
            cluster_conversations_for_user(db, client_id=client_id)
        except Exception as exc:
            logger.warning("Conversation clustering job failed: {}", exc)

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()

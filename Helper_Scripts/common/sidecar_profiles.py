from __future__ import annotations

PROFILE_MAP: dict[str, set[str] | None] = {
    "full": None,
    "llm-only": {"embeddings", "media_ingest"},
    "tts-only": {"audio", "audio_jobs", "tts"},
    "ingest-only": {"media_ingest"},
}


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_workers(
    profile: str | None,
    explicit_workers_csv: str | None,
    default_workers_csv: str,
) -> list[str]:
    explicit = _split_csv(explicit_workers_csv)
    if explicit:
        return explicit

    defaults = _split_csv(default_workers_csv)
    selected = PROFILE_MAP.get((profile or "full").strip().lower())
    if selected is None:
        return defaults
    return [worker for worker in defaults if worker in selected]

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse


@dataclass(frozen=True)
class PlaylistUrlClassification:
    source_url: str
    source_kind: str
    is_playlist: bool
    playlist_id: str | None = None
    video_id: str | None = None
    normalized_source_id: str | None = None


@dataclass(frozen=True)
class PlaylistPreflightItemData:
    ordinal: int
    source_url: str
    normalized_source_id: str | None
    source_kind: str
    title: str | None
    speaker: str | None
    duration_seconds: int | None
    published_at: str | None
    thumbnail_url: str | None
    duplicate_status: str
    duplicate_of_ordinal: int | None
    selected: bool


@dataclass(frozen=True)
class PlaylistPreflightData:
    source_url: str
    source_kind: str
    playlist_id: str | None
    playlist_title: str | None
    video_id: str | None
    item_count: int
    selected_count: int
    duplicate_count: int
    warnings: list[str]
    items: list[PlaylistPreflightItemData]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["items"] = [asdict(item) for item in self.items]
        return payload


def _hostname_matches(hostname: str, allowed_host: str) -> bool:
    return hostname == allowed_host or hostname.endswith(f".{allowed_host}")


def _is_youtube_host(hostname: str) -> bool:
    return _hostname_matches(hostname, "youtube.com") or _hostname_matches(
        hostname, "youtu.be"
    )


def _first_query_value(query: dict[str, list[str]], key: str) -> str | None:
    values = query.get(key) or []
    for value in values:
        trimmed = str(value or "").strip()
        if trimmed:
            return trimmed
    return None


def _youtube_video_id(parsed) -> str | None:
    hostname = parsed.hostname.lower() if parsed.hostname else ""
    path_parts = [part for part in parsed.path.split("/") if part]
    query = parse_qs(parsed.query)
    if _hostname_matches(hostname, "youtu.be") and path_parts:
        return path_parts[0]
    if path_parts and path_parts[0] in {"shorts", "embed", "live"} and len(path_parts) > 1:
        return path_parts[1]
    if parsed.path == "/watch":
        return _first_query_value(query, "v")
    return None


def _canonical_url_without_fragment(url: str) -> str:
    parsed = urlparse(url.strip())
    hostname = parsed.hostname.lower() if parsed.hostname else parsed.netloc.lower()
    netloc = hostname
    if parsed.port:
        netloc = f"{hostname}:{parsed.port}"
    query = urlencode(sorted(parse_qs(parsed.query, keep_blank_values=True).items()), doseq=True)
    return urlunparse(
        (
            parsed.scheme.lower(),
            netloc,
            parsed.path or "/",
            "",
            query,
            "",
        )
    )


def canonical_youtube_video_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def classify_playlist_url(url: str) -> PlaylistUrlClassification:
    trimmed = str(url or "").strip()
    parsed = urlparse(trimmed)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("unsupported_url")

    hostname = parsed.hostname.lower() if parsed.hostname else ""
    query = parse_qs(parsed.query)
    playlist_id = _first_query_value(query, "list")
    video_id = _youtube_video_id(parsed) if _is_youtube_host(hostname) else None

    if _is_youtube_host(hostname):
        if parsed.path == "/playlist" and playlist_id:
            return PlaylistUrlClassification(
                source_url=trimmed,
                source_kind="youtube_playlist",
                is_playlist=True,
                playlist_id=playlist_id,
                video_id=None,
                normalized_source_id=f"youtube:playlist:{playlist_id}",
            )
        if playlist_id and video_id:
            return PlaylistUrlClassification(
                source_url=trimmed,
                source_kind="youtube_watch_playlist",
                is_playlist=True,
                playlist_id=playlist_id,
                video_id=video_id,
                normalized_source_id=f"youtube:playlist:{playlist_id}",
            )
        if playlist_id:
            return PlaylistUrlClassification(
                source_url=trimmed,
                source_kind="youtube_playlist",
                is_playlist=True,
                playlist_id=playlist_id,
                video_id=None,
                normalized_source_id=f"youtube:playlist:{playlist_id}",
            )
        if video_id:
            return PlaylistUrlClassification(
                source_url=trimmed,
                source_kind="youtube_video",
                is_playlist=False,
                playlist_id=None,
                video_id=video_id,
                normalized_source_id=f"youtube:video:{video_id}",
            )

    return PlaylistUrlClassification(
        source_url=trimmed,
        source_kind="generic_url",
        is_playlist=False,
        playlist_id=playlist_id,
        video_id=video_id,
        normalized_source_id=f"url:{_canonical_url_without_fragment(trimmed)}",
    )


def _coerce_duration(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _source_url_from_entry(
    entry: dict[str, Any],
    *,
    assume_youtube: bool = False,
) -> str | None:
    for key in ("source_url", "webpage_url", "original_url"):
        value = _string_or_none(entry.get(key))
        if value and value.startswith(("http://", "https://")):
            return value

    candidate = _string_or_none(entry.get("url"))
    if candidate and candidate.startswith(("http://", "https://")):
        return candidate

    extractor = _string_or_none(entry.get("extractor_key") or entry.get("ie_key"))
    entry_id = _string_or_none(entry.get("id") or candidate)
    if entry_id and (
        assume_youtube or (extractor and extractor.lower().startswith("youtube"))
    ):
        return canonical_youtube_video_url(entry_id)
    return candidate


def normalize_preflight_items(raw_items: list[dict[str, Any]]) -> list[PlaylistPreflightItemData]:
    seen: dict[str, int] = {}
    normalized: list[PlaylistPreflightItemData] = []

    for index, raw in enumerate(raw_items):
        ordinal = int(raw.get("ordinal") or index + 1)
        source_url = _source_url_from_entry(raw) or ""
        source_kind = str(raw.get("source_kind") or "generic_url")
        normalized_source_id = _string_or_none(raw.get("normalized_source_id"))

        if source_url:
            try:
                classified = classify_playlist_url(source_url)
                source_kind = str(raw.get("source_kind") or classified.source_kind)
                normalized_source_id = normalized_source_id or classified.normalized_source_id
                source_hostname = urlparse(source_url).hostname or ""
                if classified.video_id and _is_youtube_host(source_hostname.lower()):
                    source_url = canonical_youtube_video_url(classified.video_id)
            except ValueError:
                normalized_source_id = normalized_source_id or f"url:{source_url}"

        dedupe_key = normalized_source_id or source_url
        duplicate_of = seen.get(dedupe_key)
        duplicate_status = "duplicate_in_batch" if duplicate_of is not None else "new"
        if duplicate_of is None and dedupe_key:
            seen[dedupe_key] = ordinal

        normalized.append(
            PlaylistPreflightItemData(
                ordinal=ordinal,
                source_url=source_url,
                normalized_source_id=normalized_source_id,
                source_kind=source_kind,
                title=_string_or_none(raw.get("title")),
                speaker=_string_or_none(
                    raw.get("speaker") or raw.get("channel") or raw.get("uploader")
                ),
                duration_seconds=_coerce_duration(raw.get("duration") or raw.get("duration_seconds")),
                published_at=_string_or_none(raw.get("published_at") or raw.get("upload_date")),
                thumbnail_url=_string_or_none(raw.get("thumbnail") or raw.get("thumbnail_url")),
                duplicate_status=duplicate_status,
                duplicate_of_ordinal=duplicate_of,
                selected=duplicate_status == "new",
            )
        )

    return normalized


def _youtube_dl_class():
    import yt_dlp  # type: ignore

    return yt_dlp.YoutubeDL


def extract_playlist_preflight(
    url: str,
    *,
    max_items: int = 100,
    youtube_dl_cls: type | None = None,
) -> PlaylistPreflightData:
    classified = classify_playlist_url(url)
    if not classified.is_playlist:
        raise ValueError("not_playlist_url")

    ydl_cls = youtube_dl_cls or _youtube_dl_class()
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": True,
        "noplaylist": False,
    }

    with ydl_cls(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    if not isinstance(info, dict):
        raise ValueError("playlist_metadata_unavailable")

    entries = [entry for entry in (info.get("entries") or []) if isinstance(entry, dict)]
    warnings: list[str] = []
    if len(entries) > max_items:
        warnings.append(f"Playlist truncated to {max_items} items.")
    limited_entries = entries[:max_items]

    raw_items: list[dict[str, Any]] = []
    assume_youtube_entries = classified.source_kind.startswith("youtube")
    for index, entry in enumerate(limited_entries, start=1):
        raw_item = dict(entry)
        raw_item["ordinal"] = index
        raw_item["source_url"] = _source_url_from_entry(
            entry,
            assume_youtube=assume_youtube_entries,
        )
        raw_items.append(raw_item)

    items = normalize_preflight_items(raw_items)
    duplicate_count = sum(1 for item in items if item.duplicate_status != "new")
    selected_count = sum(1 for item in items if item.selected)

    return PlaylistPreflightData(
        source_url=url,
        source_kind=classified.source_kind,
        playlist_id=_string_or_none(info.get("id")) or classified.playlist_id,
        playlist_title=_string_or_none(info.get("title")),
        video_id=classified.video_id,
        item_count=len(items),
        selected_count=selected_count,
        duplicate_count=duplicate_count,
        warnings=warnings,
        items=items,
    )


preflight_playlist_url = extract_playlist_preflight

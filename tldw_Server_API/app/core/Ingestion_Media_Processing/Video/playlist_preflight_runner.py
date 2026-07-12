"""Bounded spawned-process runner for playlist metadata extraction."""

from __future__ import annotations

import asyncio
import contextlib
import json
import multiprocessing
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

from tldw_Server_API.app.core.Ingestion_Media_Processing.Video import playlist_preflight

PlaylistPreflightData = playlist_preflight.PlaylistPreflightData
PlaylistPreflightItemData = playlist_preflight.PlaylistPreflightItemData
PlaylistPreflightProcessError = playlist_preflight.PlaylistPreflightProcessError

_SAFE_CHILD_ERROR_CODES = frozenset(
    {
        "not_playlist_url",
        "playlist_metadata_unavailable",
        "playlist_preflight_cancelled",
        "playlist_preflight_capacity_unavailable",
        "playlist_preflight_failed",
        "playlist_preflight_invalid_request",
        "playlist_preflight_invalid_result",
        "playlist_preflight_result_too_large",
        "playlist_preflight_timeout",
        "playlist_too_large",
    }
)
_MAX_PAYLOAD_BYTES = 4 * 1024 * 1024
_MAX_ITEM_COUNT = 500
_MAX_IDENTITY_URL_LENGTH = 8192
_MAX_IDENTITY_TEXT_LENGTH = 8192
_MAX_SHORT_TEXT_LENGTH = 512
_MAX_DISPLAY_TEXT_LENGTH = 2000
_MAX_WARNING_COUNT = 100
_MAX_WARNING_LENGTH = 1000


def _child_error_code(exc: Exception) -> str:
    code = exc.code if isinstance(exc, PlaylistPreflightProcessError) else str(exc)
    return code if code in _SAFE_CHILD_ERROR_CODES else "playlist_preflight_failed"


def _playlist_preflight_child(send_connection: Any, url: str, max_items: int) -> None:
    """Extract one playlist and send exactly one bounded JSON byte message."""
    try:
        result = playlist_preflight.extract_playlist_preflight(url, max_items=max_items)
        payload: dict[str, Any] = {"status": "ok", "result": result.to_dict()}
    except Exception as exc:  # noqa: BLE001 - child failures must cross the pipe as safe codes
        payload = {"status": "error", "code": _child_error_code(exc)}
    try:
        try:
            encoded = json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError, UnicodeError):
            encoded = b'{"status":"error","code":"playlist_preflight_failed"}'
        if len(encoded) > _MAX_PAYLOAD_BYTES:
            encoded = b'{"status":"error","code":"playlist_preflight_result_too_large"}'
        send_connection.send_bytes(encoded)
    finally:
        with contextlib.suppress(Exception):
            send_connection.close()


def _raise_too_large() -> None:
    raise PlaylistPreflightProcessError("playlist_preflight_result_too_large")


def _clip_display_text(value: str | None, *, max_length: int = _MAX_DISPLAY_TEXT_LENGTH) -> str | None:
    return value[:max_length] if value is not None else None


def _validated_preflight_result(payload: Any, *, max_items: int) -> PlaylistPreflightData:
    if not isinstance(payload, Mapping) or set(payload) != {"status", "result"}:
        raise PlaylistPreflightProcessError("playlist_preflight_invalid_result")
    if payload.get("status") != "ok" or not isinstance(payload.get("result"), Mapping):
        raise PlaylistPreflightProcessError("playlist_preflight_invalid_result")

    result = dict(payload["result"])
    expected_result_keys = {
        "source_url",
        "source_kind",
        "playlist_id",
        "playlist_title",
        "video_id",
        "item_count",
        "selected_count",
        "duplicate_count",
        "warnings",
        "items",
    }
    if set(result) != expected_result_keys:
        raise PlaylistPreflightProcessError("playlist_preflight_invalid_result")
    items_raw = result.get("items")
    warnings = result.get("warnings")
    if (
        type(result.get("source_url")) is not str
        or type(result.get("source_kind")) is not str
        or not isinstance(items_raw, list)
        or len(items_raw) > max_items
        or not isinstance(warnings, list)
        or any(type(warning) is not str for warning in warnings)
    ):
        raise PlaylistPreflightProcessError("playlist_preflight_invalid_result")
    if len(items_raw) > _MAX_ITEM_COUNT:
        _raise_too_large()
    if len(result["source_url"]) > _MAX_IDENTITY_URL_LENGTH:
        _raise_too_large()
    if len(result["source_kind"]) > _MAX_SHORT_TEXT_LENGTH:
        _raise_too_large()
    if len(warnings) > _MAX_WARNING_COUNT:
        warnings = warnings[:_MAX_WARNING_COUNT]
    warnings = [warning[:_MAX_WARNING_LENGTH] for warning in warnings]
    nullable_strings = ("playlist_id", "playlist_title", "video_id")
    if any(result.get(key) is not None and type(result.get(key)) is not str for key in nullable_strings):
        raise PlaylistPreflightProcessError("playlist_preflight_invalid_result")
    for key in ("playlist_id", "video_id"):
        value = result.get(key)
        if value is not None and len(value) > _MAX_SHORT_TEXT_LENGTH:
            _raise_too_large()
    result["playlist_title"] = _clip_display_text(result.get("playlist_title"))
    count_keys = ("item_count", "selected_count", "duplicate_count")
    if any(type(result.get(key)) is not int or int(result[key]) < 0 for key in count_keys):
        raise PlaylistPreflightProcessError("playlist_preflight_invalid_result")
    if result["item_count"] != len(items_raw):
        raise PlaylistPreflightProcessError("playlist_preflight_invalid_result")

    item_fields = set(PlaylistPreflightItemData.__dataclass_fields__)
    items: list[PlaylistPreflightItemData] = []
    try:
        for expected_ordinal, item in enumerate(items_raw, start=1):
            if not isinstance(item, Mapping) or set(item) != item_fields:
                raise ValueError
            normalized_item = PlaylistPreflightItemData(**dict(item))
            if type(normalized_item.ordinal) is not int or normalized_item.ordinal != expected_ordinal:
                raise ValueError
            if type(normalized_item.source_url) is not str or type(normalized_item.source_kind) is not str:
                raise ValueError
            if len(normalized_item.source_url) > _MAX_IDENTITY_URL_LENGTH:
                _raise_too_large()
            if len(normalized_item.source_kind) > _MAX_SHORT_TEXT_LENGTH:
                _raise_too_large()
            if type(normalized_item.availability) is not str or not normalized_item.availability:
                raise ValueError
            optional_strings = (
                normalized_item.normalized_source_id,
                normalized_item.title,
                normalized_item.speaker,
                normalized_item.published_at,
                normalized_item.thumbnail_url,
            )
            if any(value is not None and type(value) is not str for value in optional_strings):
                raise ValueError
            if (
                normalized_item.normalized_source_id is not None
                and len(normalized_item.normalized_source_id) > _MAX_IDENTITY_TEXT_LENGTH
            ):
                _raise_too_large()
            if (
                normalized_item.thumbnail_url is not None
                and len(normalized_item.thumbnail_url) > _MAX_IDENTITY_URL_LENGTH
            ):
                _raise_too_large()
            if normalized_item.duration_seconds is not None and type(normalized_item.duration_seconds) is not int:
                raise ValueError
            if normalized_item.duplicate_of_ordinal is not None and (
                type(normalized_item.duplicate_of_ordinal) is not int
                or not 1 <= normalized_item.duplicate_of_ordinal < normalized_item.ordinal
            ):
                raise ValueError
            if normalized_item.duplicate_status not in {
                "new",
                "duplicate_in_batch",
                "duplicate_existing",
                "unknown",
            }:
                raise ValueError
            if type(normalized_item.selected) is not bool:
                raise ValueError
            normalized_item = replace(
                normalized_item,
                title=_clip_display_text(normalized_item.title),
                speaker=_clip_display_text(normalized_item.speaker),
                published_at=_clip_display_text(normalized_item.published_at),
            )
            items.append(normalized_item)
    except (TypeError, ValueError) as exc:
        raise PlaylistPreflightProcessError("playlist_preflight_invalid_result") from exc
    if result["selected_count"] != sum(item.selected for item in items):
        raise PlaylistPreflightProcessError("playlist_preflight_invalid_result")
    if result["duplicate_count"] != sum(item.duplicate_status != "new" for item in items):
        raise PlaylistPreflightProcessError("playlist_preflight_invalid_result")

    return PlaylistPreflightData(
        source_url=result["source_url"],
        source_kind=result["source_kind"],
        playlist_id=result["playlist_id"],
        playlist_title=result["playlist_title"],
        video_id=result["video_id"],
        item_count=result["item_count"],
        selected_count=result["selected_count"],
        duplicate_count=result["duplicate_count"],
        warnings=list(warnings),
        items=items,
    )


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _decode_child_message(encoded: bytes) -> Any:
    if type(encoded) is not bytes:
        raise PlaylistPreflightProcessError("playlist_preflight_invalid_result")
    try:
        return json.loads(encoded.decode("utf-8"), object_pairs_hook=_strict_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise PlaylistPreflightProcessError("playlist_preflight_invalid_result") from exc


def _terminate_child(process: Any, *, join_timeout_seconds: float) -> None:
    if not process.is_alive():
        process.join(timeout=join_timeout_seconds)
        return
    process.terminate()
    process.join(timeout=join_timeout_seconds)
    if process.is_alive():
        process.kill()
        process.join(timeout=join_timeout_seconds)


async def run_playlist_preflight_process(
    url: str,
    *,
    max_items: int,
    timeout_seconds: float,
    cancel_check: Callable[[], bool] | None = None,
    poll_interval_seconds: float = 0.05,
    join_timeout_seconds: float = 1.0,
    mp_context: Any | None = None,
) -> PlaylistPreflightData:
    """Run playlist metadata extraction in a spawned, bounded child process."""
    try:
        timeout = float(timeout_seconds)
    except (TypeError, ValueError) as exc:
        raise PlaylistPreflightProcessError("playlist_preflight_invalid_request") from exc
    if type(max_items) is not int or max_items < 1 or timeout <= 0:
        raise PlaylistPreflightProcessError("playlist_preflight_invalid_request")
    context = mp_context or multiprocessing.get_context("spawn")
    recv_connection = None
    send_connection = None
    process = None
    started = False
    try:
        try:
            recv_connection, send_connection = context.Pipe(duplex=False)
            process = context.Process(
                target=_playlist_preflight_child,
                args=(send_connection, str(url), max_items),
            )
            process.start()
            started = True
        except Exception as exc:
            raise PlaylistPreflightProcessError("playlist_preflight_capacity_unavailable") from exc
        finally:
            with contextlib.suppress(Exception):
                send_connection.close()

        deadline = time.monotonic() + timeout
        message: Any | None = None
        while message is None:
            if cancel_check is not None:
                try:
                    should_cancel = bool(cancel_check())
                except Exception as exc:
                    raise PlaylistPreflightProcessError("playlist_preflight_cancelled") from exc
                if should_cancel:
                    raise PlaylistPreflightProcessError("playlist_preflight_cancelled")
            if recv_connection.poll(0):
                try:
                    encoded = recv_connection.recv_bytes(_MAX_PAYLOAD_BYTES)
                except EOFError as exc:
                    raise PlaylistPreflightProcessError("playlist_preflight_invalid_result") from exc
                except OSError as exc:
                    raise PlaylistPreflightProcessError("playlist_preflight_result_too_large") from exc
                message = _decode_child_message(encoded)
                break
            if not process.is_alive():
                raise PlaylistPreflightProcessError("playlist_preflight_invalid_result")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PlaylistPreflightProcessError("playlist_preflight_timeout")
            await asyncio.sleep(min(max(0.001, poll_interval_seconds), remaining))

        process.join(timeout=join_timeout_seconds)
        if process.is_alive():
            raise PlaylistPreflightProcessError("playlist_preflight_invalid_result")
        if recv_connection.poll(0):
            try:
                recv_connection.recv_bytes(_MAX_PAYLOAD_BYTES)
            except EOFError:
                pass
            except OSError as exc:
                raise PlaylistPreflightProcessError("playlist_preflight_invalid_result") from exc
            else:
                raise PlaylistPreflightProcessError("playlist_preflight_invalid_result")
        if not isinstance(message, Mapping):
            raise PlaylistPreflightProcessError("playlist_preflight_invalid_result")
        if set(message) == {"status", "code"} and message.get("status") == "error":
            code = message.get("code")
            if type(code) is not str or code not in _SAFE_CHILD_ERROR_CODES:
                raise PlaylistPreflightProcessError("playlist_preflight_invalid_result")
            raise PlaylistPreflightProcessError(code)
        return _validated_preflight_result(message, max_items=max_items)
    finally:
        if started and process is not None:
            with contextlib.suppress(Exception):
                _terminate_child(process, join_timeout_seconds=join_timeout_seconds)
        if recv_connection is not None:
            with contextlib.suppress(Exception):
                recv_connection.close()
        if send_connection is not None:
            with contextlib.suppress(Exception):
                send_connection.close()
        if process is not None:
            with contextlib.suppress(Exception):
                process.close()

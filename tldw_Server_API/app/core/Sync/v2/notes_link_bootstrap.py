from __future__ import annotations

"""Resumable source-verified bootstrap for durable explicit Notes links."""

import hashlib
import json
from collections.abc import Callable, Iterable, Iterator, Mapping

from tldw_Server_API.app.core.DB_Management.chacha.note_link_store import (
    NotesLink,
    NotesLinkStore,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

from .errors import SyncStoreError
from .models import SyncDataset, SyncEnvelope
from .server_origin_batch import (
    ServerOriginMutationStep,
    SyncServerOriginBatchMaterializationError,
    capture_server_origin_mutation_batch,
)
from .service import SyncV2Service

_SAFE_SOURCE_ERROR = "notes_link_bootstrap_source_invalid"
_SAFE_CAPTURE_ERROR = "notes_link_bootstrap_capture_failed"
_SOURCE_PAGE_SIZE = 200


class NotesLinkBootstrapInterrupted(RuntimeError):
    """Test/worker interruption that intentionally preserves initializing state."""


class _SourceInvalidError(SyncStoreError):
    pass


class NotesLinkBootstrapper:
    """Capture existing live and tombstoned explicit links without product replay."""

    def __init__(
        self,
        note_db: CharactersRAGDB,
        *,
        batch_size: int = 200,
        after_group: Callable[[int], None] | None = None,
    ) -> None:
        if batch_size < 1 or batch_size > 1000:
            raise ValueError("Notes link bootstrap batch_size must be 1..1000")
        self._links = NotesLinkStore(note_db)
        self._batch_size = batch_size
        self._after_group = after_group

    def bootstrap(
        self,
        *,
        service: SyncV2Service,
        user_id: str,
        dataset: SyncDataset,
    ) -> SyncDataset:
        """Resume the current bootstrap ID to ready or persist a safe failed state."""

        if dataset.owner_user_id != user_id:
            raise SyncStoreError("Sync dataset was not found or is not accessible")
        metadata = dataset.metadata.get("notes_link_v1")
        if not isinstance(metadata, Mapping):
            raise SyncStoreError("notes_link_sync_not_ready")
        state = metadata.get("state")
        bootstrap_id = metadata.get("bootstrap_id")
        if state == "ready":
            return dataset
        if state != "initializing" or not isinstance(bootstrap_id, str) or not bootstrap_id:
            raise SyncStoreError("notes_link_sync_not_ready")

        captured_count = _non_negative_int(metadata.get("captured_count"))
        expected_count = _non_negative_int(metadata.get("expected_count"))
        source_hash = metadata.get("source_hash")
        if source_hash is not None and not isinstance(source_hash, str):
            source_hash = None
        try:
            fresh_count, fresh_hash = _source_summary(self._links)
            if source_hash is not None and source_hash != fresh_hash:
                raise _SourceInvalidError("Notes link source changed during bootstrap")
            source_hash = fresh_hash
            expected_count = fresh_count
            captured_count = min(captured_count, expected_count)
            dataset = service.store.transition_notes_link_bootstrap(
                dataset.dataset_id,
                bootstrap_id=bootstrap_id,
                expected_state="initializing",
                state="initializing",
                captured_count=captured_count,
                expected_count=expected_count,
                source_hash=source_hash,
            )
            batch: list[ServerOriginMutationStep] = []
            completed_count = 0
            group_index = 0

            def capture_batch() -> None:
                nonlocal batch, captured_count, completed_count, dataset, group_index
                capture_server_origin_mutation_batch(
                    service=service,
                    user_id=user_id,
                    steps=batch,
                    source="notes-link-bootstrap",
                    idempotency_key=f"{bootstrap_id}:{group_index}",
                    trusted_notes_link_bootstrap_id=bootstrap_id,
                    bootstrap_step_verifier=self._step_matches_source,
                )
                completed_count += len(batch)
                captured_count = max(captured_count, min(expected_count, completed_count))
                dataset = service.store.transition_notes_link_bootstrap(
                    dataset.dataset_id,
                    bootstrap_id=bootstrap_id,
                    expected_state="initializing",
                    state="initializing",
                    captured_count=captured_count,
                    expected_count=expected_count,
                    source_hash=source_hash,
                )
                if self._after_group is not None:
                    self._after_group(group_index + 1)
                group_index += 1
                batch = []

            for link in _iter_source_links(self._links):
                batch.append(_step(link, bootstrap_id=bootstrap_id))
                if len(batch) == self._batch_size:
                    capture_batch()
            if batch:
                capture_batch()

            verified_count, verified_hash = _source_summary(self._links)
            if verified_count != expected_count or verified_hash != source_hash:
                raise _SourceInvalidError("Notes link source changed during bootstrap")
            return service.store.transition_notes_link_bootstrap(
                dataset.dataset_id,
                bootstrap_id=bootstrap_id,
                expected_state="initializing",
                state="ready",
                captured_count=expected_count,
                expected_count=expected_count,
                source_hash=source_hash,
                ready_verifier=lambda: _heads_match_source(
                    service,
                    dataset.dataset_id,
                    expected_count=expected_count,
                    source_hash=source_hash,
                ),
            )
        except NotesLinkBootstrapInterrupted:
            raise
        except _SourceInvalidError:
            return self._fail(
                service,
                dataset,
                bootstrap_id=bootstrap_id,
                captured_count=captured_count,
                expected_count=expected_count,
                source_hash=source_hash,
                error_code=_SAFE_SOURCE_ERROR,
            )
        except SyncServerOriginBatchMaterializationError as exc:
            if exc.retryable:
                return dataset
            return self._fail(
                service,
                dataset,
                bootstrap_id=bootstrap_id,
                captured_count=captured_count,
                expected_count=expected_count,
                source_hash=source_hash,
                error_code=_SAFE_CAPTURE_ERROR,
            )
        except Exception:  # noqa: BLE001 - every failure becomes safe durable state.
            return self._fail(
                service,
                dataset,
                bootstrap_id=bootstrap_id,
                captured_count=captured_count,
                expected_count=expected_count,
                source_hash=source_hash,
                error_code=_SAFE_CAPTURE_ERROR,
            )

    def _step_matches_source(self, envelope: SyncEnvelope) -> bool:
        link = self._links.get(envelope.object_id)
        if link is None:
            return False
        expected = _step(link, bootstrap_id=str(envelope.routing_metadata.get("bootstrap_id") or ""))
        return envelope.operation == expected.operation and dict(envelope.payload) == dict(expected.payload)

    @staticmethod
    def _fail(
        service: SyncV2Service,
        dataset: SyncDataset,
        *,
        bootstrap_id: str,
        captured_count: int,
        expected_count: int,
        source_hash: str | None,
        error_code: str,
    ) -> SyncDataset:
        return service.store.transition_notes_link_bootstrap(
            dataset.dataset_id,
            bootstrap_id=bootstrap_id,
            expected_state="initializing",
            state="failed",
            captured_count=min(captured_count, expected_count),
            expected_count=expected_count,
            source_hash=source_hash,
            error_code=error_code,
        )


def _step(link: NotesLink, *, bootstrap_id: str) -> ServerOriginMutationStep:
    payload: dict[str, object] = {
        "source_note_id": link.source_note_id,
        "target_note_id": link.target_note_id,
        "type": link.type,
        "directed": link.directed,
        "weight": link.weight,
        "label": link.label,
        "properties": dict(link.properties),
        "created_at": link.created_at,
        "last_modified": link.last_modified,
        "created_by": link.created_by,
    }
    operation = "tombstone" if link.deleted else "upsert"
    if link.deleted:
        if link.deleted_at is None:
            raise _SourceInvalidError("Stored notes.link tombstone has no deleted_at")
        payload.update({"deleted_at": link.deleted_at, "reason": None})
    return ServerOriginMutationStep(
        domain="notes.link",
        operation=operation,
        object_id=link.edge_id,
        payload=payload,
        routing_metadata={
            "bootstrap_capture": True,
            "bootstrap_id": bootstrap_id,
        },
        stable_key=f"notes.link:{link.edge_id}",
    )


def _iter_source_links(store: NotesLinkStore) -> Iterator[NotesLink]:
    after_edge_id: str | None = None
    while True:
        page = store.list_page(
            after_edge_id=after_edge_id,
            limit=_SOURCE_PAGE_SIZE,
            include_deleted_links=True,
            include_deleted_endpoints=True,
        )
        if not page:
            return
        yield from page
        after_edge_id = page[-1].edge_id
        if len(page) < _SOURCE_PAGE_SIZE:
            return


def _source_summary(store: NotesLinkStore) -> tuple[int, str]:
    return _manifest_digest(_link_manifest_item(link) for link in _iter_source_links(store))


def _link_manifest_item(link: NotesLink) -> Mapping[str, object]:
    return {
        "edge_id": link.edge_id,
        "operation": "tombstone" if link.deleted else "upsert",
        "payload": dict(_step(link, bootstrap_id="snapshot").payload),
    }


def _manifest_digest(items: Iterable[Mapping[str, object]]) -> tuple[int, str]:
    digest = hashlib.sha256()
    digest.update(b"[")
    count = 0
    for item in items:
        if count:
            digest.update(b",")
        digest.update(
            json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        )
        count += 1
    digest.update(b"]")
    return count, digest.hexdigest()


def _heads_match_source(
    service: SyncV2Service,
    dataset_id: str,
    *,
    expected_count: int,
    source_hash: str,
) -> bool:
    all_applied = True

    def head_items() -> Iterator[Mapping[str, object]]:
        nonlocal all_applied
        offset = 0
        while True:
            heads = service.store.list_current_heads(
                dataset_id,
                "notes.link",
                limit=_SOURCE_PAGE_SIZE,
                offset=offset,
            )
            if not heads:
                return
            for head in heads:
                if head.apply_status != "applied":
                    all_applied = False
                yield {
                    "edge_id": head.object_id,
                    "operation": head.operation,
                    "payload": dict(head.payload),
                }
            offset += len(heads)
            if len(heads) < _SOURCE_PAGE_SIZE:
                return

    actual_count, actual_hash = _manifest_digest(head_items())
    return all_applied and actual_count == expected_count and actual_hash == source_hash


def _non_negative_int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


__all__ = ["NotesLinkBootstrapInterrupted", "NotesLinkBootstrapper"]

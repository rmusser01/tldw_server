"""Supervisor-owned, single-dispatch manual slot snapshot operations (ADR-043)."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import httpx

from tldw_Server_API.app.core import http_client
from tldw_Server_API.app.core.Security.egress import ConfiguredEndpointScope

from .llamacpp_runtime_models import LlamaCppProfile, LlamaCppRuntimeState
from .llamacpp_snapshot_compatibility import build_fingerprint, canonical_sha256, compare_fingerprints, hash_file_stable
from .llamacpp_snapshot_models import Fingerprint, OperationReceipt, SnapshotMetadata, SnapshotRequest
from .llamacpp_snapshot_store import SnapshotNotFoundError, SnapshotStore

# Add a hash only together with recorded live save/restart/restore/cache-reuse evidence.
TESTED_TEXT_BUILD_SHA256: frozenset[str] = frozenset()
TOKEN_RETENTION_SECONDS = 30 * 86400
_TERMINAL = {"complete", "failed", "outcome_unknown"}


class SnapshotOperationError(RuntimeError):
    """Safe machine-readable error at the admin operation boundary."""

    def __init__(self, code: str, status_code: int = 409):
        super().__init__(code)
        self.code = code
        self.status_code = status_code


async def disk_call(function, *args, **kwargs):
    """Keep filesystem work owned until its thread finishes, including cancellation."""
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # A thread cannot be cancelled: do not release the reservation or root fence.
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
        task.result()
        raise


async def checked_transport(**kwargs):
    """Use bounded central checked egress without retries, redirects or body logging."""
    origin = kwargs.pop("origin")
    remaining = kwargs.pop("remaining")
    response = await http_client.afetch(
        **kwargs,
        configured_endpoint=ConfiguredEndpointScope.from_url(origin),
        sensitive_observability=True,
        allow_redirects=False,
        retry=http_client.RetryPolicy(attempts=1),
        transport="httpx",
        max_response_bytes=256 * 1024,
        timeout=httpx.Timeout(remaining, connect=5, write=min(30, remaining), pool=5),
    )
    if response.status_code != 200:
        raise SnapshotOperationError("runtime_response_invalid", 503)
    return response.json()


class SnapshotOperations:
    """Own durable receipts, profile reservations and bounded execution tasks."""

    def __init__(
        self,
        store: SnapshotStore,
        *,
        transport=checked_transport,
        supported_builds=None,
        concurrency: int = 1,
        clock=time.time,
    ):
        self.store = store
        self.transport = transport
        self.supported_builds = TESTED_TEXT_BUILD_SHA256 if supported_builds is None else supported_builds
        self.clock = clock
        self.key = store.token_key()
        self.tasks: dict[str, asyncio.Task] = {}
        self.active: dict[str, str] = {}
        self.quarantined: dict[str, str] = {}
        self.receipts: dict[str, OperationReceipt] = {}
        self.locks: dict[str, asyncio.Lock] = {}
        self._admission_lock = asyncio.Lock()
        self.semaphore = asyncio.Semaphore(max(1, concurrency))
        self.accepting = True
        self.deadline = 600.0
        self._recovered: set[str] = set()
        self._sending: set[str] = set()
        self._interrupting: set[str] = set()

    def issue_token(self, profile_id: str) -> str:
        """Issue a bounded profile-bound token without a filesystem write."""
        payload = json.dumps([profile_id, int(self.clock()), uuid4().hex], separators=(",", ":")).encode()
        body = base64.urlsafe_b64encode(payload).decode().rstrip("=")
        signature = hmac.new(self.key, body.encode(), hashlib.sha256).hexdigest()
        return f"{body}.{signature}"

    def validate_token(self, profile_id: str, token: str, *, now=None) -> str:
        """Validate signature/profile/age before lookup, with constant-time MAC comparison."""
        try:
            if not 1 <= len(token) <= 512:
                raise ValueError
            body, signature = token.split(".")
            expected = hmac.new(self.key, body.encode(), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(signature, expected):
                raise ValueError
            owner, issued, nonce = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
            age = (self.clock() if now is None else now) - issued
            if owner != profile_id or type(issued) is not int or not 0 <= age <= TOKEN_RETENTION_SECONDS:
                raise ValueError
            if not isinstance(nonce, str) or len(nonce) != 32:
                raise ValueError
        except (ValueError, TypeError, UnicodeError) as exc:
            raise SnapshotOperationError("invalid_request_token", 422) from exc
        return hashlib.sha256(token.encode()).hexdigest()

    async def recover(self, profile_id: str) -> list[OperationReceipt]:
        """Never replay persisted work; dispatch markers remain uncertainty evidence."""
        receipts = await disk_call(self.store.list_receipts, profile_id)
        if profile_id not in self._recovered:
            for receipt in receipts:
                if receipt.state not in _TERMINAL:
                    receipt = receipt.model_copy(
                        update={
                            "state": "outcome_unknown" if receipt.dispatched else "failed",
                            "error_code": "server_interrupted",
                        }
                    )
                    await disk_call(self.store.write_receipt, receipt)
                self.receipts[receipt.operation_id] = receipt
            self._recovered.add(profile_id)
        return [self.receipts.get(item.operation_id, item) for item in receipts]

    def guard_lifecycle(self, profile_id: str, *, stop: bool = False) -> None:
        """Conflict immediately while reserved; explicit Stop recovers terminal uncertainty."""
        if profile_id in self.active:
            raise SnapshotOperationError("snapshot_operation_busy")
        if profile_id in self.quarantined and not stop:
            raise SnapshotOperationError("launch_quarantined")

    def _check_owner(self, runner, generation: str):
        if runner is None or getattr(runner, "snapshot_process", None) is None:
            raise SnapshotOperationError("runtime_owner_unavailable", 503)
        if runner.snapshot_generation != generation or runner.status().launch_generation != generation:
            raise SnapshotOperationError("stale_launch_generation")
        if runner.snapshot_process.returncode is not None or runner.status().state != LlamaCppRuntimeState.RUNNING:
            raise SnapshotOperationError("runtime_stopped", 503)

    async def fingerprint(self, profile: LlamaCppProfile, runner: Any) -> Fingerprint:
        if profile.mode.value != "chat" or profile.mmproj_model_id:
            raise SnapshotOperationError("unsupported_configuration", 422)
        # Initial coverage is a bounded text configuration, without adapters/router/draft state.
        allowed = {
            "threads",
            "t",
            "ctx_size",
            "c",
            "n_ctx",
            "n_gpu_layers",
            "ngl",
            "gpu_layers",
            "parallel",
            "n_parallel",
            "batch_size",
            "ubatch_size",
            "cache_type_k",
            "cache_type_v",
            "swa_full",
        }
        if set(profile.server_args) - allowed:
            raise SnapshotOperationError("unsupported_configuration", 422)
        fingerprint = await disk_call(
            build_fingerprint,
            model=Path(runner.status().model_path),
            executable=runner.snapshot_executable,
            effective_options=runner.snapshot_options,
            adapters=[],
        )
        if fingerprint.executable_sha256 not in self.supported_builds:
            raise SnapshotOperationError("unsupported_build", 422)
        launched = getattr(runner, "snapshot_fingerprint", None)
        if launched is None or compare_fingerprints(launched, fingerprint):
            raise SnapshotOperationError("runtime_identity_changed", 422)
        return fingerprint

    async def slots(self, profile: LlamaCppProfile, runner: Any) -> dict[str, object]:
        receipts = await self.recover(profile.profile_id)
        result = {
            "capability": "stopped",
            "reason": "runtime_stopped",
            "launch_generation": None,
            "request_id": self.issue_token(profile.profile_id),
            "slots": [],
            "latest_operation_id": receipts[-1].operation_id if receipts else None,
        }
        if runner is not None:
            result["launch_generation"] = runner.status().launch_generation
        if not profile.snapshots_enabled:
            return result | {"capability": "disabled", "reason": "snapshots_disabled"}
        if runner is None or runner.status().state != LlamaCppRuntimeState.RUNNING:
            return result
        if not getattr(runner, "snapshot_working", None):
            return result | {"capability": "restart_required", "reason": "restart_required"}
        try:
            self.guard_lifecycle(profile.profile_id)
            self._check_owner(runner, runner.snapshot_generation)
            await self.fingerprint(profile, runner)
            slots = await self._inspect(runner, runner.snapshot_generation)
            return result | {"capability": "ready", "reason": None, "slots": slots}
        except SnapshotOperationError as exc:
            capability = "busy" if exc.code == "snapshot_operation_busy" else "unsupported"
            return result | {"capability": capability, "reason": exc.code}
        except Exception:  # noqa: BLE001 - capability failures expose only a safe diagnostic.
            return result | {"capability": "unavailable", "reason": "capability_check_failed"}

    async def catalog(self, profile: LlamaCppProfile, runner: Any, offset: int, limit: int) -> dict[str, object]:
        if offset < 0 or not 1 <= limit <= 100:
            raise SnapshotOperationError("invalid_pagination", 422)
        entries = await disk_call(self.store.list, profile.profile_id)
        current = None
        if runner is not None and runner.status().state == LlamaCppRuntimeState.RUNNING:
            try:
                current = await self.fingerprint(profile, runner)
            except Exception:  # noqa: BLE001 - uncertain identity never permits restore.
                current = None
        items = []
        for entry in entries[offset : offset + limit]:
            reasons = compare_fingerprints(entry.fingerprint, current)
            items.append(
                {
                    "snapshot_id": entry.snapshot_id,
                    "source_slot": entry.source_slot,
                    "created_at": entry.created_at.isoformat(),
                    "commit_sequence": entry.commit_sequence,
                    "byte_count": entry.byte_count,
                    "token_count": entry.token_count,
                    "compatibility": "unknown" if current is None else "incompatible" if reasons else "compatible",
                    "reasons": reasons,
                }
            )
        return {
            "snapshots": items,
            "total": len(entries),
            "total_bytes": sum(e.byte_count for e in entries),
            "offset": offset,
            "limit": limit,
            "retention": profile.snapshot_retention,
        }

    async def admit(
        self,
        profile: LlamaCppProfile,
        runner: Any,
        request: SnapshotRequest,
        actor_id: str,
        kind: Literal["save", "restore"],
        snapshot_id: str | None = None,
    ) -> OperationReceipt:
        profile_id = profile.profile_id
        async with self._admission_lock, self.locks.setdefault(profile_id, asyncio.Lock()):
            if not self.accepting:
                raise SnapshotOperationError("server_shutting_down", 503)
            operation_id = self.validate_token(profile_id, request.request_id)
            digest = canonical_sha256({"request": request.model_dump(), "kind": kind, "snapshot_id": snapshot_id})
            await self.recover(profile_id)
            try:
                existing = await disk_call(self.store.read_receipt, profile_id, operation_id)
            except SnapshotNotFoundError:
                existing = None
            if existing is not None:
                if not hmac.compare_digest(existing.request_digest, digest):
                    raise SnapshotOperationError("request_token_conflict")
                return existing
            if not self.accepting:
                raise SnapshotOperationError("server_shutting_down", 503)
            self.guard_lifecycle(profile_id)
            if not profile.snapshots_enabled:
                raise SnapshotOperationError("snapshots_disabled", 422)
            self._check_owner(runner, request.expected_launch_generation)
            if not runner.snapshot_working:
                raise SnapshotOperationError("restart_required")
            if kind == "restore" and not request.replace_confirmed:
                raise SnapshotOperationError("replace_confirmation_required", 422)
            fingerprint = await self.fingerprint(profile, runner)
            source = None
            if kind == "restore":
                entries = await disk_call(self.store.list, profile_id)
                source = next((e for e in entries if e.snapshot_id == snapshot_id), None)
                if source is None:
                    raise SnapshotNotFoundError("snapshot not found")
                if compare_fingerprints(source.fingerprint, fingerprint):
                    raise SnapshotOperationError("snapshot_incompatible", 422)
            receipt = OperationReceipt(
                profile_id=profile_id,
                operation_id=operation_id,
                launch_generation=request.expected_launch_generation,
                request_digest=digest,
                kind=kind,
                state="validating",
                snapshot_id=snapshot_id,
                actor_id=actor_id,
            )
            try:
                await disk_call(self.store.write_receipt, receipt)
            except asyncio.CancelledError:
                await disk_call(
                    self.store.write_receipt,
                    receipt.model_copy(update={"state": "failed", "error_code": "admission_interrupted"}),
                )
                raise
            self.receipts[operation_id] = receipt
            self.active[profile_id] = operation_id
            task = asyncio.create_task(self._execute(profile, runner, request, receipt, actor_id, fingerprint, source))
            self.tasks[operation_id] = task
            return receipt

    async def _inspect(self, runner, generation: str, slot_id=None):
        self._check_owner(runner, generation)
        origin = runner.status().endpoint
        async with asyncio.timeout(5):
            payload = await self.transport(method="GET", url=origin + "/slots", origin=origin, remaining=5)
        if not isinstance(payload, list) or len(payload) > 1024:
            raise SnapshotOperationError("invalid_slot_response", 503)
        result = []
        for item in payload:
            if not isinstance(item, dict):
                raise SnapshotOperationError("invalid_slot_response", 503)
            slot = item.get("id")
            busy = item.get("is_processing")
            # Source contract: llama.cpp 4d9176092d00586775af140581bb0b558ddc4389,
            # server-context.cpp server_slot::to_json. No task history means no count.
            tokens = item.get("n_prompt_tokens")
            if "n_prompt_tokens" not in item and "id_task" not in item and busy is False:
                tokens = 0
            if type(slot) is not int or slot < 0 or type(busy) is not bool or type(tokens) is not int or tokens < 0:
                raise SnapshotOperationError("invalid_slot_response", 503)
            result.append({"slot_id": slot, "busy": busy, "token_count": tokens})
        self._check_owner(runner, generation)
        if slot_id is None:
            return result
        selected = next((s for s in result if s["slot_id"] == slot_id), None)
        if selected is None:
            raise SnapshotOperationError("slot_not_found", 422)
        if selected["busy"]:
            raise SnapshotOperationError("slot_busy")
        return selected

    async def _persist(self, receipt, **updates):
        receipt = receipt.model_copy(update=updates)
        await disk_call(self.store.write_receipt, receipt)
        self.receipts[receipt.operation_id] = receipt
        return receipt

    async def _execute(self, profile, runner, request, receipt, actor_id, fingerprint, source):
        dispatched = False
        process = runner.snapshot_process
        origin = runner.status().endpoint
        generation = receipt.launch_generation
        try:
            async with asyncio.timeout(self.deadline), self.semaphore:
                started = asyncio.get_running_loop().time()
                if source is not None:
                    free = await disk_call(shutil.disk_usage, runner.snapshot_working)
                    if free.free < source.byte_count:
                        raise SnapshotOperationError("insufficient_disk_space", 503)
                    staged = await disk_call(
                        self.store.stage_restore, profile.profile_id, source.snapshot_id, runner.snapshot_working
                    )
                else:
                    staged = runner.snapshot_working / f"save-{uuid4().hex}.bin"
                slot = await self._inspect(runner, generation, request.slot_id)
                if source is None and slot["token_count"] == 0:
                    raise SnapshotOperationError("slot_empty", 422)
                if process is not runner.snapshot_process or origin != runner.status().endpoint:
                    raise SnapshotOperationError("runtime_owner_changed")
                if compare_fingerprints(fingerprint, await self.fingerprint(profile, runner)):
                    raise SnapshotOperationError("runtime_identity_changed")
                slot = await self._inspect(runner, generation, request.slot_id)
                self._check_owner(runner, generation)
                # Cancellation while publishing this marker must never replace it with dispatched=False.
                dispatched = True
                receipt = receipt.model_copy(
                    update={"state": "saving" if source is None else "restoring", "dispatched": True}
                )
                receipt = await self._persist(receipt)
                # No await between final owner check and starting the one upstream mutation.
                self._check_owner(runner, generation)
                if process is not runner.snapshot_process or origin != runner.status().endpoint:
                    raise SnapshotOperationError("runtime_owner_changed")
                self._sending.add(receipt.operation_id)
                try:
                    response = await self.transport(
                        method="POST",
                        url=origin + f"/slots/{request.slot_id}",
                        origin=origin,
                        remaining=max(0.001, self.deadline - (asyncio.get_running_loop().time() - started)),
                        params={"action": receipt.kind},
                        json={"filename": staged.name},
                    )
                finally:
                    self._sending.discard(receipt.operation_id)
                if receipt.operation_id in self._interrupting:
                    raise SnapshotOperationError("server_interrupted")
                tokens, size = self._validate_ack(response, request.slot_id, staged.name, receipt.kind)
                if source is None and tokens != slot["token_count"]:
                    raise SnapshotOperationError("invalid_acknowledgement")
                if source is not None and (tokens != source.token_count or size != source.byte_count):
                    raise SnapshotOperationError("invalid_acknowledgement")
                self._check_owner(runner, generation)
                if process is not runner.snapshot_process or origin != runner.status().endpoint:
                    raise SnapshotOperationError("runtime_owner_changed")
                if compare_fingerprints(fingerprint, await self.fingerprint(profile, runner)):
                    raise SnapshotOperationError("runtime_identity_changed")
                receipt = await self._persist(receipt, state="verifying")
                warning = None
                if source is None:
                    digest = await disk_call(hash_file_stable, staged)
                    sequence = await disk_call(self.store.allocate_sequence, profile.profile_id)
                    source = SnapshotMetadata(
                        profile_id=profile.profile_id,
                        snapshot_id=uuid4().hex,
                        source_slot=request.slot_id,
                        created_at=datetime.now(UTC),
                        commit_sequence=sequence,
                        byte_count=size,
                        token_count=tokens,
                        sha256=digest,
                        fingerprint=fingerprint,
                        actor_id=actor_id,
                    )
                    self._check_owner(runner, generation)
                    await disk_call(self.store.commit, profile.profile_id, staged, source)
                    try:
                        failed = await disk_call(self.store.prune, profile.profile_id, profile.snapshot_retention)
                        warning = "retention_partial_failure" if failed else None
                    except Exception:  # noqa: BLE001 - pruning cannot invalidate a committed save.
                        warning = "retention_partial_failure"
                self._check_owner(runner, generation)
                receipt = await self._persist(
                    receipt, state="complete", snapshot_id=source.snapshot_id, token_count=tokens, warning_code=warning
                )
                await disk_call(self.store.remove_working_file, profile.profile_id, generation, staged.name)
        except BaseException as exc:  # noqa: BLE001 - cancellation and all faults must retain dispatch evidence.
            completed = receipt.state == "complete"
            if dispatched and not completed:
                self.quarantined[profile.profile_id] = generation
            state = "complete" if completed else "outcome_unknown" if dispatched else "failed"
            code = (
                exc.code
                if isinstance(exc, SnapshotOperationError)
                else "operation_interrupted"
                if isinstance(exc, asyncio.CancelledError)
                else "operation_failed"
            )
            terminal = receipt.model_copy(update={"state": state, "error_code": code, "dispatched": dispatched})
            if completed:
                # Cleanup is after durable completion. Its failure cannot turn a
                # verified save/restore into an uncertain upstream mutation.
                terminal = receipt.model_copy(update={"warning_code": "working_cleanup_failed"})
            self.receipts[receipt.operation_id] = terminal
            try:
                await disk_call(self.store.write_receipt, terminal)
            except Exception:  # noqa: BLE001 - preserve the prior durable marker on storage failure.
                # Existing durable dispatch marker remains the crash-recovery authority.
                if not completed:
                    self.quarantined[profile.profile_id] = generation
        finally:
            self._interrupting.discard(receipt.operation_id)
            self.active.pop(profile.profile_id, None)
            self.tasks.pop(receipt.operation_id, None)

    @staticmethod
    def _validate_ack(payload, slot_id, filename, kind):
        count_key, size_key = ("n_saved", "n_written") if kind == "save" else ("n_restored", "n_read")
        if not isinstance(payload, dict):
            raise SnapshotOperationError("invalid_acknowledgement")
        tokens, size = payload.get(count_key), payload.get(size_key)
        if (
            type(payload.get("id_slot")) is not int
            or payload["id_slot"] != slot_id
            or payload.get("filename") != filename
            or type(tokens) is not int
            or tokens <= 0
            or type(size) is not int
            or size <= 0
        ):
            raise SnapshotOperationError("invalid_acknowledgement")
        return tokens, size

    async def operation(self, profile_id: str, operation_id: str) -> OperationReceipt:
        await self.recover(profile_id)
        receipt = self.receipts.get(operation_id)
        if receipt is not None and receipt.profile_id == profile_id:
            return receipt
        return await disk_call(self.store.read_receipt, profile_id, operation_id)

    async def delete(self, profile_id: str, snapshot_id: str) -> None:
        self.guard_lifecycle(profile_id, stop=True)
        await disk_call(self.store.delete, profile_id, snapshot_id)

    async def drain(self, timeout: float = 10.0) -> None:
        """Stop admission, drain boundedly, then cancel owned tasks without abandoning I/O."""
        self.accepting = False
        # Admission may be in an offloaded fingerprint/receipt write. Wait for its
        # final registration before taking the set that must precede child shutdown.
        async with self._admission_lock:
            tasks = list(self.tasks.values())
        if not tasks:
            return
        _, pending = await asyncio.wait(tasks, timeout=timeout)
        for operation_id, task in list(self.tasks.items()):
            if task in pending and operation_id in self._sending:
                self._interrupting.add(operation_id)
                receipt = self.receipts[operation_id]
                self.quarantined[receipt.profile_id] = receipt.launch_generation
                try:
                    await self._persist(receipt, state="outcome_unknown", error_code="server_interrupted")
                except Exception:  # noqa: BLE001 - shutdown must still cancel a hung transport.
                    # The already durable dispatch marker still prevents replay after a crash.
                    self.receipts[operation_id] = receipt.model_copy(
                        update={"state": "outcome_unknown", "error_code": "receipt_storage_failed"}
                    )
        for task in pending:
            # Execution catches cancellation and persists uncertainty before releasing ownership.
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

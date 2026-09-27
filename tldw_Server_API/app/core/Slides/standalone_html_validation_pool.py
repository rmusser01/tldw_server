"""Bounded killable subprocess admission for standalone HTML validation."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import multiprocessing
import sys
import unicodedata
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from multiprocessing.connection import Connection
from typing import Any, Literal

from .standalone_html_contracts import (
    StandaloneHtmlValidationError,
    StandaloneHtmlValidationResult,
)
from .standalone_html_validator import (
    _BIDI_FORMATTING,
    MAX_DOCUMENT_BYTES,
    DeliveryStyle,
    _collapse_html_whitespace,
    _is_forbidden_control,
    _preflight_document_input,
    validate_standalone_html,
)

_IPC_VERSION = 1
_INTERACTIVE_CAPACITY = 24
_GENERATION_CAPACITY = 8
_INTERACTIVE_WEIGHT = 3
_MAX_WORKERS = 4
_RETRY_AFTER_SECONDS = 1
_READY_TIMEOUT_SECONDS = 10.0
_IPC_JOIN_TIMEOUT_SECONDS = 1.0
_HEAVY_SLIDES_MODULES = frozenset(
    {
        "tldw_Server_API.app.core.Slides.slides_db",
        "tldw_Server_API.app.core.Slides.slides_export",
        "tldw_Server_API.app.core.Slides.slides_generator",
    }
)
_SAFE_CODES = frozenset(
    {
        "standalone_html_invalid_document",
        "standalone_html_validation_budget_exceeded",
        "validator_unavailable",
    }
)
_SAFE_REASONS = frozenset(
    {
        "css_bytes",
        "css_declarations",
        "css_depth",
        "css_errors",
        "css_parse_error",
        "css_resource",
        "css_stylesheets",
        "css_token_bytes",
        "css_tokens",
        "css_unbalanced_block",
        "css_unterminated_comment",
        "css_unterminated_string",
        "delivery_style",
        "document_bytes",
        "document_controls",
        "document_encoding",
        "document_type",
        "html_active_attribute",
        "html_active_element",
        "html_attributes",
        "html_attribute_name",
        "html_comment_token",
        "html_declaration",
        "html_depth",
        "html_doctype",
        "html_doctype_token",
        "html_document_boundary",
        "html_document_order",
        "html_document_structure",
        "html_elements",
        "html_end_tag",
        "html_parse_error",
        "html_raw_text_token",
        "html_refresh",
        "html_resource_attribute",
        "html_root",
        "html_tag_name",
        "html_tag_token",
        "html_text_token",
        "html_tokens",
        "html_unterminated_attribute",
        "html_unterminated_comment",
        "html_unterminated_raw_text",
        "html_unterminated_tag",
        "notes_delivery_style",
        "notes_structure",
        "script_policy",
        "script_position",
        "script_structure",
        "slide_count",
        "slide_structure",
        "title_blank",
        "title_characters",
        "title_length",
    }
)
_BUDGET_REASONS = frozenset(
    {
        "css_bytes",
        "css_declarations",
        "css_depth",
        "css_errors",
        "css_stylesheets",
        "css_token_bytes",
        "css_tokens",
        "document_bytes",
        "html_attributes",
        "html_comment_token",
        "html_depth",
        "html_doctype_token",
        "html_elements",
        "html_raw_text_token",
        "html_tag_token",
        "html_text_token",
        "html_tokens",
    }
)
_INVALID_REASONS = _SAFE_REASONS - _BUDGET_REASONS
_LOCATION_REASONS = frozenset({"css_parse_error", "html_parse_error"})
STANDALONE_HTML_VALIDATION_POOL_METADATA_KEY = "_server_standalone_html_validation_pool"
VALIDATION_POOL_ATTR = "standalone_html_validation_pool"
VALIDATION_POOL_LOCK_ATTR = "standalone_html_validation_pool_lock"
VALIDATION_POOL_WORKER_OWNED_ATTR = "standalone_html_validation_pool_worker_owned"


def _validator_worker_main(
    connection: Connection,
    validator: Callable[..., StandaloneHtmlValidationResult],
    require_isolated_imports: bool = False,
) -> None:
    """Run closed validation RPC on a private pipe without logging source."""
    try:
        imports_are_isolated = not bool(_HEAVY_SLIDES_MODULES & sys.modules.keys())
        connection.send((_IPC_VERSION, "ready", imports_are_isolated))
        if require_isolated_imports and not imports_are_isolated:
            return
        while True:
            try:
                message = connection.recv()
            except EOFError:
                return
            if message == (_IPC_VERSION, "close"):
                return
            if not isinstance(message, tuple) or len(message) != 6:
                return
            version, operation, epoch, request_id, document, delivery_style = message
            if version != _IPC_VERSION or operation != "validate":
                return
            safe_error = (
                _IPC_VERSION,
                "error",
                epoch,
                request_id,
                "validator_unavailable",
                503,
                None,
                None,
                None,
                None,
            )
            try:
                try:
                    result = validator(document, delivery_style=delivery_style)
                except StandaloneHtmlValidationError as exc:
                    code = exc.code if exc.code in _SAFE_CODES else "validator_unavailable"
                    status = exc.status_code if exc.status_code in {422, 503} else 503
                    retry_after = (
                        exc.retry_after if isinstance(exc.retry_after, int) and 1 <= exc.retry_after <= 5 else None
                    )
                    reason = exc.reason if exc.reason in _SAFE_REASONS else None
                    line = exc.line if isinstance(exc.line, int) and 1 <= exc.line <= 1_000_000 else None
                    column = exc.column if isinstance(exc.column, int) and 1 <= exc.column <= 1_000_000 else None
                    response = (
                        _IPC_VERSION,
                        "error",
                        epoch,
                        request_id,
                        code,
                        status,
                        retry_after,
                        reason,
                        line,
                        column,
                    )
                except BaseException:  # noqa: BLE001 - isolate validator termination
                    response = safe_error
                else:
                    response = (
                        _IPC_VERSION,
                        "result",
                        epoch,
                        request_id,
                        result.title,
                        result.slide_count,
                        result.html_bytes,
                        result.html_sha256,
                        result.indexable_text,
                    )
            except BaseException:  # noqa: BLE001 - projection must not escape child stderr
                response = safe_error
            try:
                connection.send(response)
            except BaseException:  # noqa: BLE001 - serialization may carry attacker-controlled exceptions
                if response is safe_error:
                    return
                try:
                    connection.send(safe_error)
                except BaseException:  # noqa: BLE001 - corrupt IPC exits silently
                    return
    except BaseException:  # noqa: BLE001 - worker must never print source-bearing tracebacks
        return
    finally:
        with contextlib.suppress(BaseException):  # noqa: BLE001 - closed child exits silently
            connection.close()


@dataclass(slots=True)
class _WorkerSlot:
    index: int
    epoch: int
    process: multiprocessing.Process
    connection: Connection
    ready: bool = False


@dataclass(slots=True)
class _ValidationJob:
    request_id: int
    document: str | bytes = field(repr=False)
    delivery_style: DeliveryStyle | None
    queue_kind: Literal["interactive", "generation"]
    future: asyncio.Future[StandaloneHtmlValidationResult] = field(repr=False)
    reservation: GenerationValidationReservation | None = field(default=None, repr=False)
    worker_index: int | None = None
    worker_epoch: int | None = None
    cancelled: bool = False
    slot_released: bool = False


class GenerationValidationReservation:
    """One pre-provider generation slot transferable exactly once."""

    __slots__ = ("_pool", "_state")

    def __init__(self, pool: StandaloneHtmlValidationPool) -> None:
        self._pool = pool
        self._state: Literal["reserved", "queued", "done", "released"] = "reserved"

    @property
    def consumed(self) -> bool:
        """Whether a returned provider document consumed this reservation."""
        return self._state in {"queued", "done"}

    async def validate(
        self,
        document: str | bytes,
        *,
        delivery_style: DeliveryStyle | None = None,
    ) -> StandaloneHtmlValidationResult:
        """Atomically transfer this reservation into generation validation."""
        return await self._pool._validate_reserved(self, document, delivery_style)

    async def release(self) -> None:
        """Release an unused reservation after provider failure/cancellation."""
        await self._pool._release_reservation(self)


class StandaloneHtmlValidationPool:
    """Supervise two bounded queues across at most four killable processes."""

    def __init__(
        self,
        *,
        max_workers: int = _MAX_WORKERS,
        watchdog_seconds: float = 60.0,
        validator: Callable[..., StandaloneHtmlValidationResult] = validate_standalone_html,
        mp_start_method: str = "spawn",
    ) -> None:
        if not 1 <= max_workers <= _MAX_WORKERS:
            raise ValueError("one through four validator subprocesses are allowed")
        if watchdog_seconds <= 0:
            raise ValueError("watchdog_seconds must be positive")
        self._max_workers = max_workers
        self._watchdog_seconds = float(watchdog_seconds)
        self._validator = validator
        self._context = multiprocessing.get_context(mp_start_method)
        self._require_isolated_imports = mp_start_method == "spawn"
        self._condition = asyncio.Condition()
        self._lifecycle_lock = asyncio.Lock()
        self._interactive: deque[_ValidationJob] = deque()
        self._generation: deque[_ValidationJob] = deque()
        self._active: dict[int, _ValidationJob] = {}
        self._canceling_workers: set[int] = set()
        self._slots: list[_WorkerSlot | None] = []
        self._worker_epochs = [0 for _ in range(max_workers)]
        self._worker_locks = [asyncio.Lock() for _ in range(max_workers)]
        self._tasks: list[asyncio.Task[None]] = []
        self._maintenance_tasks: set[asyncio.Task[Any]] = set()
        self._close_task: asyncio.Task[None] | None = None
        self._reservations: set[GenerationValidationReservation] = set()
        self._request_counter = 0
        self._generation_slots_in_use = 0
        self._interactive_streak = 0
        self._started = False
        self._closing = False
        self._closed = False

    @property
    def worker_pids(self) -> tuple[int, ...]:
        """Return live source-free worker identifiers for health inspection."""
        pids: list[int] = []
        for slot in self._slots:
            if slot is None or not self._slot_is_available(slot):
                continue
            with contextlib.suppress(ValueError):
                pid = slot.process.pid
                if pid is not None:
                    pids.append(pid)
        return tuple(pids)

    @property
    def worker_names(self) -> tuple[str, ...]:
        """Return fixed application-owned worker names."""
        names: list[str] = []
        for slot in self._slots:
            if slot is not None:
                with contextlib.suppress(ValueError):
                    names.append(slot.process.name)
        return tuple(names)

    @staticmethod
    def _slot_is_alive(slot: _WorkerSlot) -> bool:
        with contextlib.suppress(ValueError):
            return slot.process.is_alive()
        return False

    @classmethod
    def _slot_is_available(cls, slot: _WorkerSlot) -> bool:
        return slot.ready and cls._slot_is_alive(slot)

    @property
    def interactive_waiting(self) -> int:
        return len(self._interactive)

    @property
    def generation_slots_in_use(self) -> int:
        return self._generation_slots_in_use

    @property
    def active_count(self) -> int:
        return len(self._active)

    def _busy_error(self) -> StandaloneHtmlValidationError:
        return StandaloneHtmlValidationError(
            "standalone_html_validator_busy",
            status_code=503,
            retry_after=_RETRY_AFTER_SECONDS,
        )

    @staticmethod
    def _unavailable_error() -> StandaloneHtmlValidationError:
        return StandaloneHtmlValidationError("validator_unavailable", status_code=503)

    @staticmethod
    def _timeout_error() -> StandaloneHtmlValidationError:
        return StandaloneHtmlValidationError(
            "standalone_html_validator_timeout",
            status_code=503,
        )

    def _spawn_slot(self, index: int, epoch: int) -> _WorkerSlot:
        parent_connection, child_connection = self._context.Pipe(duplex=True)
        process = self._context.Process(
            name=f"standalone-html-validator-{index + 1}",
            target=_validator_worker_main,
            args=(
                child_connection,
                self._validator,
                self._require_isolated_imports,
            ),
            daemon=True,
        )
        try:
            process.start()
        except BaseException:
            parent_connection.close()
            child_connection.close()
            raise
        child_connection.close()
        return _WorkerSlot(
            index=index,
            epoch=epoch,
            process=process,
            connection=parent_connection,
        )

    @staticmethod
    def _await_ready_sync(
        slot: _WorkerSlot,
        require_isolated_imports: bool,
    ) -> bool:
        try:
            if not slot.connection.poll(_READY_TIMEOUT_SECONDS):
                return False
            response = slot.connection.recv()
            return (
                isinstance(response, tuple)
                and len(response) == 3
                and response[:2] == (_IPC_VERSION, "ready")
                and isinstance(response[2], bool)
                and (response[2] or not require_isolated_imports)
            )
        except (BrokenPipeError, EOFError, OSError, ValueError):
            return False

    @staticmethod
    def _terminate_slot_sync(slot: _WorkerSlot) -> bool:
        """Close IPC, stop the child, confirm reap, then close its handle."""
        process = slot.process
        with contextlib.suppress(OSError):
            slot.connection.close()
        if process.is_alive():
            with contextlib.suppress(ProcessLookupError):
                process.terminate()
            process.join(0.25)
        if process.is_alive():
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            process.join(1.0)
        if process.is_alive():
            return False
        if process.exitcode is not None:
            process.join()
        with contextlib.suppress(ValueError):
            process.close()
        return True

    async def start(self) -> None:
        """Lazily spawn the fixed validator process set."""
        async with self._lifecycle_lock:
            if self._started:
                return
            if self._closed or self._closing:
                raise self._unavailable_error()
            slots: list[_WorkerSlot] = []
            try:
                for index in range(self._max_workers):
                    slot = self._spawn_slot(index, 1)
                    slots.append(slot)
                    if not await asyncio.to_thread(
                        self._await_ready_sync,
                        slot,
                        self._require_isolated_imports,
                    ):
                        raise RuntimeError("validator worker failed readiness")
                    slot.ready = True
            except asyncio.CancelledError:
                for slot in slots:
                    self._terminate_slot_sync(slot)
                raise
            except Exception:  # noqa: BLE001 - any boot failure rejects the pool
                for slot in slots:
                    self._terminate_slot_sync(slot)
                raise self._unavailable_error() from None
            self._slots = list(slots)
            for slot in slots:
                self._worker_epochs[slot.index] = slot.epoch
            self._started = True
            self._tasks = [
                asyncio.create_task(
                    self._worker_loop(index),
                    name=f"standalone-html-validator-supervisor-{index + 1}",
                )
                for index in range(self._max_workers)
            ]

    async def _stop_worker(self, worker_index: int) -> bool:
        """Serialize final retirement with watchdog and cancellation repair."""
        async with self._worker_locks[worker_index]:
            if worker_index >= len(self._slots):
                return True
            slot = self._slots[worker_index]
            if slot is None:
                return True
            retired = await asyncio.to_thread(self._terminate_slot_sync, slot)
            if retired:
                self._slots[worker_index] = None
            return retired

    async def _close_impl(self) -> None:
        async with self._condition:
            queued = [*self._interactive, *self._generation]
            self._interactive.clear()
            self._generation.clear()
            for job in queued:
                self._release_job_slot_locked(job)
                if not job.future.done():
                    job.future.set_exception(self._unavailable_error())
            for job in self._active.values():
                self._release_job_slot_locked(job)
                if not job.future.done():
                    job.future.set_exception(self._unavailable_error())
            for reservation in tuple(self._reservations):
                if reservation._state == "reserved":
                    reservation._state = "released"
                    self._generation_slots_in_use -= 1
                    self._reservations.discard(reservation)
            self._condition.notify_all()

        retired = await asyncio.gather(
            *(self._stop_worker(index) for index in range(len(self._slots))),
            return_exceptions=False,
        )
        if not all(retired):
            raise self._unavailable_error()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        if self._maintenance_tasks:
            await asyncio.gather(*tuple(self._maintenance_tasks), return_exceptions=True)
        self._tasks.clear()
        self._slots.clear()
        self._active.clear()
        self._generation_slots_in_use = 0
        self._started = False
        self._closed = True
        self._closing = False

    async def close(self) -> None:
        """Run terminal cleanup to completion even when the caller is cancelled."""
        async with self._lifecycle_lock:
            if self._closed:
                return
            if self._close_task is None:
                self._closing = True
                self._close_task = asyncio.create_task(
                    self._close_impl(),
                    name="standalone-html-validator-close",
                )
            close_task = self._close_task

        cancelled = False
        while not close_task.done():
            try:
                await asyncio.wait({close_task})
            except asyncio.CancelledError:
                cancelled = True

        try:
            close_task.result()
        except Exception:
            async with self._lifecycle_lock:
                if self._close_task is close_task:
                    self._close_task = None
            raise
        if cancelled:
            raise asyncio.CancelledError

    def _new_job(
        self,
        document: str | bytes,
        delivery_style: DeliveryStyle | None,
        queue_kind: Literal["interactive", "generation"],
        reservation: GenerationValidationReservation | None = None,
    ) -> _ValidationJob:
        self._request_counter += 1
        return _ValidationJob(
            request_id=self._request_counter,
            document=document,
            delivery_style=delivery_style,
            queue_kind=queue_kind,
            future=asyncio.get_running_loop().create_future(),
            reservation=reservation,
        )

    async def validate(
        self,
        document: str | bytes,
        *,
        delivery_style: DeliveryStyle | None = None,
    ) -> StandaloneHtmlValidationResult:
        """Submit authenticated save/restore/export work to the priority queue."""
        _preflight_document_input(document)
        await self.start()
        async with self._condition:
            if self._closing or self._closed:
                raise self._unavailable_error()
            if len(self._interactive) >= _INTERACTIVE_CAPACITY:
                raise self._busy_error()
            job = self._new_job(document, delivery_style, "interactive")
            self._interactive.append(job)
            self._condition.notify()
        return await self._await_job(job)

    async def acquire_generation_reservation(self) -> GenerationValidationReservation:
        """Reserve one of eight slots before the caller dispatches a provider."""
        await self.start()
        if not await self._ensure_live_worker():
            raise self._unavailable_error()
        async with self._condition:
            if self._closing or self._closed:
                raise self._unavailable_error()
            if not self.worker_pids:
                raise self._unavailable_error()
            if self._generation_slots_in_use >= _GENERATION_CAPACITY:
                raise self._busy_error()
            self._generation_slots_in_use += 1
            reservation = GenerationValidationReservation(self)
            self._reservations.add(reservation)
            return reservation

    async def _validate_reserved(
        self,
        reservation: GenerationValidationReservation,
        document: str | bytes,
        delivery_style: DeliveryStyle | None,
    ) -> StandaloneHtmlValidationResult:
        _preflight_document_input(document)
        await self.start()
        async with self._condition:
            if reservation._pool is not self or reservation._state != "reserved":
                raise RuntimeError("generation validation reservation was already consumed")
            if self._closing or self._closed:
                reservation._state = "released"
                self._generation_slots_in_use -= 1
                self._reservations.discard(reservation)
                raise self._unavailable_error()
            reservation._state = "queued"
            job = self._new_job(
                document,
                delivery_style,
                "generation",
                reservation,
            )
            self._generation.append(job)
            self._condition.notify()
        return await self._await_job(job)

    async def _release_reservation(
        self,
        reservation: GenerationValidationReservation,
    ) -> None:
        async with self._condition:
            if reservation._pool is not self:
                raise RuntimeError("reservation belongs to another validator pool")
            if reservation._state == "reserved":
                reservation._state = "released"
                self._generation_slots_in_use -= 1
                self._reservations.discard(reservation)

    def _release_job_slot_locked(self, job: _ValidationJob) -> None:
        if job.slot_released or job.reservation is None:
            return
        job.slot_released = True
        if job.reservation._state == "queued":
            job.reservation._state = "done"
            self._generation_slots_in_use -= 1
            self._reservations.discard(job.reservation)

    async def _await_job(self, job: _ValidationJob) -> StandaloneHtmlValidationResult:
        try:
            return await asyncio.shield(job.future)
        except asyncio.CancelledError:
            cleanup = asyncio.create_task(self._cancel_job(job))
            self._maintenance_tasks.add(cleanup)
            cleanup.add_done_callback(self._maintenance_tasks.discard)
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    continue
            with contextlib.suppress(asyncio.CancelledError):
                cleanup.result()
            raise

    async def _cancel_job(self, job: _ValidationJob) -> None:
        worker_index: int | None = None
        worker_epoch: int | None = None
        async with self._condition:
            job.cancelled = True
            with contextlib.suppress(ValueError):
                self._interactive.remove(job)
            with contextlib.suppress(ValueError):
                self._generation.remove(job)
            candidate = job.worker_index
            if candidate is not None and self._active.get(candidate) is job:
                worker_index = candidate
                worker_epoch = job.worker_epoch
                self._canceling_workers.add(candidate)
            self._release_job_slot_locked(job)
            if not job.future.done():
                job.future.cancel()
            self._condition.notify_all()
        if worker_index is not None:
            try:
                await self._replace_worker(worker_index, worker_epoch)
            finally:
                async with self._condition:
                    self._canceling_workers.discard(worker_index)
                    self._condition.notify_all()

    async def _take_next_job(self, worker_index: int) -> _ValidationJob | None:
        async with self._condition:
            while not self._closing and (
                worker_index in self._canceling_workers or (not self._interactive and not self._generation)
            ):
                await self._condition.wait()
            if self._closing:
                return None
            if self._interactive and self._generation:
                if self._interactive_streak >= _INTERACTIVE_WEIGHT:
                    job = self._generation.popleft()
                    self._interactive_streak = 0
                else:
                    job = self._interactive.popleft()
                    self._interactive_streak += 1
            elif self._interactive:
                job = self._interactive.popleft()
                self._interactive_streak = min(
                    _INTERACTIVE_WEIGHT,
                    self._interactive_streak + 1,
                )
            else:
                job = self._generation.popleft()
                self._interactive_streak = 0
            job.worker_index = worker_index
            self._active[worker_index] = job
            return job

    @staticmethod
    def _rpc_sync(
        slot: _WorkerSlot,
        job: _ValidationJob,
        watchdog_seconds: float,
    ) -> tuple[Any, ...] | None:
        del watchdog_seconds
        try:
            slot.connection.send(
                (
                    _IPC_VERSION,
                    "validate",
                    slot.epoch,
                    job.request_id,
                    job.document,
                    job.delivery_style,
                )
            )
            response = slot.connection.recv()
            return response if isinstance(response, tuple) else ()
        except (BrokenPipeError, EOFError, OSError, ValueError):
            return ()

    async def _rpc_with_watchdog(
        self,
        worker_index: int,
        slot: _WorkerSlot,
        job: _ValidationJob,
    ) -> tuple[Any, ...] | None:
        rpc_task = asyncio.create_task(
            asyncio.to_thread(
                self._rpc_sync,
                slot,
                job,
                self._watchdog_seconds,
            ),
            name=f"standalone-html-validator-ipc-{worker_index + 1}",
        )
        try:
            return await asyncio.wait_for(
                asyncio.shield(rpc_task),
                self._watchdog_seconds,
            )
        except asyncio.TimeoutError:
            await self._replace_worker(worker_index, slot.epoch)
            try:
                await asyncio.wait_for(
                    asyncio.shield(rpc_task),
                    _IPC_JOIN_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError as exc:
                raise RuntimeError("validator IPC did not terminate") from exc
            except (BrokenPipeError, EOFError, OSError, RuntimeError, TypeError, ValueError):
                pass
            return None
        except asyncio.CancelledError:
            if not rpc_task.done():
                await self._replace_worker(worker_index, slot.epoch)
                try:
                    await asyncio.wait_for(
                        asyncio.shield(rpc_task),
                        _IPC_JOIN_TIMEOUT_SECONDS,
                    )
                except asyncio.TimeoutError as exc:
                    raise RuntimeError("validator IPC did not terminate") from exc
                except (BrokenPipeError, EOFError, OSError, RuntimeError, TypeError, ValueError):
                    pass
            raise

    async def _replace_worker(
        self,
        worker_index: int,
        expected_epoch: int | None = None,
    ) -> bool:
        if worker_index >= len(self._slots):
            return False
        async with self._worker_locks[worker_index]:
            current = self._slots[worker_index]
            if current is not None and expected_epoch is not None and current.epoch != expected_epoch:
                return self._slot_is_available(current)
            if current is not None:
                retired = await asyncio.to_thread(self._terminate_slot_sync, current)
                if not retired:
                    return False
                self._slots[worker_index] = None
            if self._closing or self._closed:
                return False
            try:
                replacement = self._spawn_slot(
                    worker_index,
                    self._worker_epochs[worker_index] + 1,
                )
            except Exception:  # noqa: BLE001 - process startup must fail closed
                return False

            readiness = asyncio.create_task(
                asyncio.to_thread(
                    self._await_ready_sync,
                    replacement,
                    self._require_isolated_imports,
                )
            )
            cancelled = False
            ready = False
            while not readiness.done():
                try:
                    ready = await asyncio.shield(readiness)
                except asyncio.CancelledError:
                    cancelled = True
                except (OSError, RuntimeError, ValueError):
                    break
            if readiness.done() and not readiness.cancelled():
                with contextlib.suppress(Exception):
                    ready = readiness.result()

            if cancelled or not ready or self._closing or self._closed:
                retirement = asyncio.create_task(asyncio.to_thread(self._terminate_slot_sync, replacement))
                retired = False
                while not retirement.done():
                    try:
                        retired = await asyncio.shield(retirement)
                    except asyncio.CancelledError:
                        cancelled = True
                    except (OSError, RuntimeError, ValueError):
                        break
                if retirement.done() and not retirement.cancelled():
                    with contextlib.suppress(Exception):
                        retired = retirement.result()
                if not retired:
                    self._slots[worker_index] = replacement
                if cancelled:
                    raise asyncio.CancelledError
                return False

            replacement.ready = True
            self._slots[worker_index] = replacement
            self._worker_epochs[worker_index] = replacement.epoch
            return True

    async def _ensure_live_worker(self) -> bool:
        """Repair dead slots before granting pre-provider capacity."""
        for worker_index, slot in enumerate(tuple(self._slots)):
            if slot is None or not self._slot_is_available(slot):
                expected_epoch = slot.epoch if slot is not None else None
                await self._replace_worker(worker_index, expected_epoch)
        return any(slot is not None and self._slot_is_available(slot) for slot in self._slots)

    @staticmethod
    def _decode_response(
        response: tuple[Any, ...],
        *,
        epoch: int,
        request_id: int,
        document: str | bytes,
    ) -> StandaloneHtmlValidationResult | StandaloneHtmlValidationError | None:
        try:
            if len(response) < 4:
                return None
            if response[:4] == (_IPC_VERSION, "result", epoch, request_id):
                if len(response) != 9:
                    return None
                (
                    _,
                    _,
                    _,
                    _,
                    title,
                    slide_count,
                    html_bytes,
                    html_sha256,
                    indexable_text,
                ) = response
                document_bytes = document if isinstance(document, bytes) else document.encode("utf-8", "strict")
                if len(document_bytes) > MAX_DOCUMENT_BYTES:
                    return None
                if not (
                    isinstance(title, str)
                    and title
                    and unicodedata.normalize("NFC", title) == title
                    and _collapse_html_whitespace(title) == title
                    and not any(
                        _is_forbidden_control(character) or character in _BIDI_FORMATTING for character in title
                    )
                    and len(title) <= 200
                    and len(title.encode("utf-8", "strict")) <= 512
                    and type(slide_count) is int
                    and 1 <= slide_count <= 30
                    and type(html_bytes) is int
                    and html_bytes == len(document_bytes)
                    and html_bytes <= MAX_DOCUMENT_BYTES
                    and isinstance(html_sha256, str)
                    and html_sha256 == hashlib.sha256(document_bytes).hexdigest()
                    and isinstance(indexable_text, str)
                    and len(indexable_text) <= 250_000
                    and len(indexable_text.encode("utf-8", "strict")) <= 1_000_000
                ):
                    return None
                return StandaloneHtmlValidationResult(
                    title=title,
                    slide_count=slide_count,
                    html_bytes=html_bytes,
                    html_sha256=html_sha256,
                    indexable_text=indexable_text,
                )
            if response[:4] == (_IPC_VERSION, "error", epoch, request_id):
                if len(response) != 10:
                    return None
                (
                    _,
                    _,
                    _,
                    _,
                    code,
                    status_code,
                    retry_after,
                    reason,
                    line,
                    column,
                ) = response
                if retry_after is not None:
                    return None
                if code == "standalone_html_invalid_document":
                    valid_policy = status_code == 422 and reason in _INVALID_REASONS
                elif code == "standalone_html_validation_budget_exceeded":
                    valid_policy = status_code == 422 and reason in _BUDGET_REASONS and line is None and column is None
                elif code == "validator_unavailable":
                    valid_policy = status_code == 503 and reason is None and line is None and column is None
                else:
                    valid_policy = False
                if not valid_policy:
                    return None
                if reason in _LOCATION_REASONS:
                    if not (
                        type(line) is int
                        and 1 <= line <= 1_000_000
                        and type(column) is int
                        and 1 <= column <= 1_000_000
                    ):
                        return None
                elif line is not None or column is not None:
                    return None
                return StandaloneHtmlValidationError(
                    code,
                    status_code=status_code,
                    retry_after=retry_after,
                    reason=reason,
                    line=line,
                    column=column,
                )
        except (TypeError, UnicodeError, ValueError):
            return None
        return None

    async def _finish_job(
        self,
        worker_index: int,
        job: _ValidationJob,
        outcome: StandaloneHtmlValidationResult | StandaloneHtmlValidationError,
    ) -> None:
        async with self._condition:
            if self._active.get(worker_index) is job:
                self._active.pop(worker_index, None)
            self._release_job_slot_locked(job)
            if not job.future.done() and not job.cancelled:
                if isinstance(outcome, StandaloneHtmlValidationError):
                    job.future.set_exception(outcome)
                else:
                    job.future.set_result(outcome)
            self._condition.notify_all()

    async def _worker_loop(self, worker_index: int) -> None:
        while not self._closing:
            job = await self._take_next_job(worker_index)
            if job is None:
                return
            outcome: StandaloneHtmlValidationResult | StandaloneHtmlValidationError = self._unavailable_error()
            cancelled = False
            try:
                slot = self._slots[worker_index]
                if slot is None:
                    await self._replace_worker(worker_index)
                else:
                    epoch = slot.epoch
                    job.worker_epoch = epoch
                    response = await self._rpc_with_watchdog(
                        worker_index,
                        slot,
                        job,
                    )
                    if response is None:
                        outcome = self._timeout_error()
                    elif not response:
                        await self._replace_worker(worker_index, epoch)
                    else:
                        decoded = self._decode_response(
                            response,
                            epoch=epoch,
                            request_id=job.request_id,
                            document=job.document,
                        )
                        if decoded is None:
                            await self._replace_worker(worker_index, epoch)
                        else:
                            outcome = decoded
            except asyncio.CancelledError:
                cancelled = True
            except Exception:  # noqa: BLE001 - supervisor capacity must be restored
                if not self._closing:
                    with contextlib.suppress(Exception):
                        await self._replace_worker(worker_index, job.worker_epoch)
            finally:
                completion = asyncio.create_task(self._finish_job(worker_index, job, outcome))
                try:
                    await asyncio.shield(completion)
                except asyncio.CancelledError:
                    cancelled = True
                    await completion
            if cancelled:
                raise asyncio.CancelledError


async def get_app_standalone_html_validation_pool(app: Any) -> StandaloneHtmlValidationPool:
    """Return the lifecycle-owned validation pool stored on an ASGI app."""

    state = app.state
    pool = getattr(state, VALIDATION_POOL_ATTR, None)
    if pool is not None:
        return pool
    lock = getattr(state, VALIDATION_POOL_LOCK_ATTR, None)
    if lock is None:
        lock = asyncio.Lock()
        setattr(state, VALIDATION_POOL_LOCK_ATTR, lock)
    async with lock:
        pool = getattr(state, VALIDATION_POOL_ATTR, None)
        if pool is None:
            pool = StandaloneHtmlValidationPool()
            setattr(state, VALIDATION_POOL_ATTR, pool)
        return pool


__all__ = [
    "GenerationValidationReservation",
    "STANDALONE_HTML_VALIDATION_POOL_METADATA_KEY",
    "StandaloneHtmlValidationPool",
    "VALIDATION_POOL_ATTR",
    "VALIDATION_POOL_LOCK_ATTR",
    "VALIDATION_POOL_WORKER_OWNED_ATTR",
    "get_app_standalone_html_validation_pool",
]

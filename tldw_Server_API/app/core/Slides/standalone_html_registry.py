"""Standalone-HTML digest keyring and source-free shared registry policy."""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import hmac
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from types import MappingProxyType
from typing import Any, Protocol

_KEY_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,31}\Z")
_CONFIG_EPOCH_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}\Z")
_BASE64URL_RE = re.compile(r"[A-Za-z0-9_-]+\Z")
_LOWER_HEX_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_HMAC_FRAME_PREFIX = b"tldw\x00slides\x00standalone-html\x00hmac\x00v1\x00"
_MAX_ACTIVE_KEYS = 4
_RETIREMENT_FLOOR = timedelta(days=32)
MAX_HMAC_KEYS_JSON_BYTES = 4_096


class DigestKeyConfigError(ValueError):
    """Raised when the secret keyring configuration is invalid."""


class DigestKeyRegistryError(RuntimeError):
    """Raised when source-free shared registry metadata is invalid."""


class DigestKeyRotationError(DigestKeyRegistryError):
    """Raised when an explicit current-key compare-and-swap cannot complete."""


class DigestKeyRetirementError(DigestKeyRegistryError):
    """Raised when a retiring key is not yet safe to remove."""


class DigestKeyUnavailableError(DigestKeyRegistryError):
    """Global generation gate used when any required digest secret is absent."""

    error_code = "generation_digest_key_unavailable"


class DigestKeyState(str, Enum):
    """Closed set of states retained in the shared registry."""

    CURRENT = "current"
    RETIRING = "retiring"


class HmacDomain(str, Enum):
    """Closed domains for all persisted standalone-generation correlations."""

    CLIENT_IDEMPOTENCY_KEY = "client-idempotency-key"
    CLIENT_REQUEST = "client-request"
    SOURCE_SNAPSHOT = "source-snapshot"
    EXECUTION_MANIFEST = "execution-manifest"
    JOBS_IDEMPOTENCY_KEY = "jobs-idempotency-key"


def _validate_key_id(key_id: object, *, error_type: type[Exception]) -> str:
    if not isinstance(key_id, str) or _KEY_ID_RE.fullmatch(key_id) is None:
        raise error_type("digest key ID must match [A-Za-z0-9][A-Za-z0-9._-]{0,31}")
    return key_id


def _validate_utc_timestamp(
    value: object,
    *,
    field_name: str,
    error_type: type[Exception] = DigestKeyRegistryError,
) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise error_type(f"{field_name} must be an aware UTC timestamp")
    return value


def _validate_config_epoch(
    value: object,
    *,
    allow_none: bool = False,
    error_type: type[Exception] = DigestKeyRegistryError,
) -> str | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or _CONFIG_EPOCH_RE.fullmatch(value) is None:
        raise error_type("config epoch must be a safe visible revision token")
    return value


@dataclass(frozen=True, slots=True)
class DigestKeyMetadata:
    """Persistable key metadata containing no secret or digest material."""

    key_id: str
    state: DigestKeyState
    activated_at: datetime
    retired_at: datetime | None

    def __post_init__(self) -> None:
        _validate_key_id(self.key_id, error_type=DigestKeyRegistryError)
        if not isinstance(self.state, DigestKeyState):
            raise DigestKeyRegistryError("digest key state is invalid")
        _validate_utc_timestamp(self.activated_at, field_name="activated_at")
        if self.state is DigestKeyState.CURRENT:
            if self.retired_at is not None:
                raise DigestKeyRegistryError("current digest key cannot have retired_at")
            return
        if self.retired_at is None:
            raise DigestKeyRegistryError("retiring digest key requires retired_at")
        _validate_utc_timestamp(self.retired_at, field_name="retired_at")
        if self.retired_at < self.activated_at:
            raise DigestKeyRegistryError("retired_at cannot precede activated_at")


@dataclass(frozen=True, slots=True)
class DigestKeyRegistryState:
    """Authoritative source-free state loaded from the shared Jobs store."""

    records: tuple[DigestKeyMetadata, ...]
    config_epoch: str | None

    def __post_init__(self) -> None:
        try:
            records = tuple(self.records)
        except TypeError as exc:
            raise DigestKeyRegistryError("digest key records must be a sequence") from exc
        if not all(isinstance(record, DigestKeyMetadata) for record in records):
            raise DigestKeyRegistryError("digest key registry returned an invalid record")
        object.__setattr__(self, "records", records)
        _validate_config_epoch(self.config_epoch, allow_none=True)


@dataclass(frozen=True, slots=True)
class CurrentKeyCasResult:
    """Source-free outcome distinguishing a local CAS from an identical winner."""

    state: DigestKeyRegistryState
    applied_here: bool

    def __post_init__(self) -> None:
        if not isinstance(self.state, DigestKeyRegistryState) or not isinstance(self.applied_here, bool):
            raise DigestKeyRegistryError("current-key CAS result is invalid")


@dataclass(frozen=True, slots=True)
class DormantSweepProof:
    """Fenced proof that one complete dormant-database sweep found no references."""

    key_id: str
    config_epoch: str
    fencing_token: int
    sweep_started_at: datetime
    sweep_completed_at: datetime
    complete: bool
    unexpired_reference_count: int

    def __post_init__(self) -> None:
        _validate_key_id(self.key_id, error_type=DigestKeyRegistryError)
        _validate_config_epoch(self.config_epoch)
        if isinstance(self.fencing_token, bool) or not isinstance(self.fencing_token, int) or self.fencing_token <= 0:
            raise DigestKeyRegistryError("sweep fencing token must be a positive integer")
        _validate_utc_timestamp(self.sweep_started_at, field_name="sweep_started_at")
        _validate_utc_timestamp(self.sweep_completed_at, field_name="sweep_completed_at")
        if self.sweep_completed_at < self.sweep_started_at:
            raise DigestKeyRegistryError("sweep completion cannot precede its start")
        if not isinstance(self.complete, bool):
            raise DigestKeyRegistryError("sweep complete marker must be boolean")
        if (
            isinstance(self.unexpired_reference_count, bool)
            or not isinstance(self.unexpired_reference_count, int)
            or self.unexpired_reference_count < 0
        ):
            raise DigestKeyRegistryError("sweep unexpired reference count must be a nonnegative integer")


@dataclass(frozen=True, slots=True)
class DigestKeyAvailability:
    """Local secret availability for every shared current/retiring key ID."""

    missing_key_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        missing = tuple(self.missing_key_ids)
        if not all(isinstance(key_id, str) and _KEY_ID_RE.fullmatch(key_id) is not None for key_id in missing):
            raise DigestKeyRegistryError("missing digest key IDs are invalid")
        if len(set(missing)) != len(missing):
            raise DigestKeyRegistryError("missing digest key IDs contain duplicates")
        object.__setattr__(self, "missing_key_ids", tuple(sorted(missing)))

    @property
    def available(self) -> bool:
        """Return whether this node has every shared nonretired secret."""
        return not self.missing_key_ids


@dataclass(frozen=True, slots=True)
class DigestKeySnapshot:
    """One immutable join of shared metadata and local secret availability."""

    records: tuple[DigestKeyMetadata, ...]
    config_epoch: str | None
    configured_current_key_id: str
    availability: DigestKeyAvailability

    def __post_init__(self) -> None:
        records = tuple(self.records)
        if len(records) > _MAX_ACTIVE_KEYS or not all(isinstance(record, DigestKeyMetadata) for record in records):
            raise DigestKeyRegistryError("digest key snapshot records are invalid")
        key_ids = tuple(record.key_id for record in records)
        if len(set(key_ids)) != len(key_ids):
            raise DigestKeyRegistryError("digest key snapshot contains duplicate IDs")
        current_count = sum(record.state is DigestKeyState.CURRENT for record in records)
        if records and current_count != 1:
            raise DigestKeyRegistryError("digest key snapshot must contain exactly one current key")
        if records and self.config_epoch is None:
            raise DigestKeyRegistryError("nonempty digest key snapshot requires a config epoch")
        _validate_config_epoch(self.config_epoch, allow_none=True)
        _validate_key_id(
            self.configured_current_key_id,
            error_type=DigestKeyRegistryError,
        )
        if not isinstance(self.availability, DigestKeyAvailability):
            raise DigestKeyRegistryError("digest key snapshot availability is invalid")
        if not set(self.availability.missing_key_ids).issubset(key_ids):
            raise DigestKeyRegistryError("missing digest key IDs must exist in the shared registry")
        object.__setattr__(self, "records", records)

    @property
    def current_key_id(self) -> str | None:
        """Return the sole shared current key ID, if the registry is empty."""
        return next(
            (record.key_id for record in self.records if record.state is DigestKeyState.CURRENT),
            None,
        )

    @property
    def generation_ready(self) -> bool:
        """Return whether generation can use the configured shared current key."""
        return (
            self.availability.available
            and self.config_epoch is not None
            and self.current_key_id is not None
            and self.current_key_id == self.configured_current_key_id
        )

    def require_secrets_available(self) -> None:
        """Fail the global admission/worker gate when any shared secret is absent."""
        if not self.availability.available:
            missing = ", ".join(self.availability.missing_key_ids)
            raise DigestKeyUnavailableError(f"digest secret unavailable for shared key IDs: {missing}")

    def require_generation_ready(self) -> None:
        """Require all secrets and an explicitly activated configured current key."""
        self.require_secrets_available()
        if self.current_key_id != self.configured_current_key_id:
            raise DigestKeyUnavailableError("configured digest key is not the shared current key")


@dataclass(frozen=True, slots=True)
class KeyedHmac:
    """A lowercase SHA-256 HMAC paired with its nonsecret key ID."""

    key_id: str
    digest_hex: str = field(repr=False)

    def __post_init__(self) -> None:
        _validate_key_id(self.key_id, error_type=DigestKeyRegistryError)
        if _LOWER_HEX_SHA256_RE.fullmatch(self.digest_hex) is None:
            raise DigestKeyRegistryError("digest must be 64 lowercase hexadecimal characters")


class DigestKeyRegistryStore(Protocol):
    """Shared Jobs-store operations required by the digest registry domain."""

    async def load_digest_key_registry(self) -> DigestKeyRegistryState:
        """Load all current/retiring key metadata and its config epoch."""
        ...

    async def compare_and_swap_current_key(
        self,
        *,
        expected_current_key_id: str | None,
        expected_config_epoch: str | None,
        new_current_key_id: str,
        new_config_epoch: str,
        changed_at: datetime,
    ) -> CurrentKeyCasResult | None:
        """Apply the CAS, return an identical racing winner, or report conflict."""
        ...

    async def load_dormant_sweep_proof(self, *, key_id: str) -> DormantSweepProof | None:
        """Load the latest complete-sweep proof recorded for a retiring key."""
        ...

    async def compare_and_swap_remove_key(
        self,
        *,
        key_id: str,
        expected_retired_at: datetime,
        expected_config_epoch: str | None,
    ) -> DigestKeyRegistryState | None:
        """Remove one unchanged retiring key or report a compare-and-swap conflict."""
        ...


class JobManagerDigestKeyRegistryStore:
    """Async source-free registry adapter over one injected synchronous JobManager."""

    __slots__ = ("_job_manager",)

    def __init__(self, job_manager: Any) -> None:
        self._job_manager = job_manager

    @staticmethod
    def _state(raw: object) -> DigestKeyRegistryState:
        if not isinstance(raw, Mapping):
            raise DigestKeyRegistryError("Jobs key registry returned invalid state")
        raw_records = raw.get("records", ())
        if not isinstance(raw_records, (list, tuple)):
            raise DigestKeyRegistryError("Jobs key registry returned invalid records")
        records: list[DigestKeyMetadata] = []
        for raw_record in raw_records:
            if not isinstance(raw_record, Mapping):
                raise DigestKeyRegistryError("Jobs key registry returned an invalid record")
            try:
                records.append(
                    DigestKeyMetadata(
                        key_id=raw_record["key_id"],
                        state=DigestKeyState(raw_record["state"]),
                        activated_at=raw_record["activated_at"],
                        retired_at=raw_record.get("retired_at"),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise DigestKeyRegistryError("Jobs key registry returned an invalid record") from exc
        return DigestKeyRegistryState(
            records=tuple(records),
            config_epoch=raw.get("config_revision"),
        )

    async def load_digest_key_registry(self) -> DigestKeyRegistryState:
        raw = await asyncio.to_thread(self._job_manager.load_slides_digest_key_registry)
        return self._state(raw)

    async def compare_and_swap_current_key(
        self,
        *,
        expected_current_key_id: str | None,
        expected_config_epoch: str | None,
        new_current_key_id: str,
        new_config_epoch: str,
        changed_at: datetime,
    ) -> CurrentKeyCasResult | None:
        raw = await asyncio.to_thread(
            self._job_manager.compare_and_swap_slides_current_key,
            expected_current_key_id=expected_current_key_id,
            expected_config_revision=expected_config_epoch,
            new_current_key_id=new_current_key_id,
            new_config_revision=new_config_epoch,
            changed_at=changed_at,
        )
        if raw is None:
            return None
        if not isinstance(raw, Mapping) or not isinstance(raw.get("applied_here"), bool):
            raise DigestKeyRegistryError("Jobs current-key CAS returned invalid state")
        return CurrentKeyCasResult(
            state=self._state(raw.get("state")),
            applied_here=raw["applied_here"],
        )

    async def load_dormant_sweep_proof(self, *, key_id: str) -> DormantSweepProof | None:
        raw = await asyncio.to_thread(
            self._job_manager.load_slides_dormant_sweep_proof,
            key_id=key_id,
        )
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise DigestKeyRegistryError("Jobs dormant sweep returned invalid proof")
        try:
            return DormantSweepProof(
                key_id=raw["key_id"],
                config_epoch=raw["config_revision"],
                fencing_token=raw["fencing_token"],
                sweep_started_at=raw["sweep_started_at"],
                sweep_completed_at=raw["sweep_completed_at"],
                complete=raw["complete"],
                unexpired_reference_count=raw["unexpired_reference_count"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DigestKeyRegistryError("Jobs dormant sweep returned invalid proof") from exc

    async def compare_and_swap_remove_key(
        self,
        *,
        key_id: str,
        expected_retired_at: datetime,
        expected_config_epoch: str | None,
    ) -> DigestKeyRegistryState | None:
        raw = await asyncio.to_thread(
            self._job_manager.compare_and_swap_remove_slides_key,
            key_id=key_id,
            expected_retired_at=expected_retired_at,
            expected_config_revision=expected_config_epoch,
        )
        return None if raw is None else self._state(raw)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DigestKeyConfigError("duplicate digest key ID")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> Any:
    del value
    raise DigestKeyConfigError("nonfinite values are forbidden in digest key JSON")


def _decode_secret(key_id: str, encoded: object) -> bytes:
    if (
        not isinstance(encoded, str)
        or not encoded.isascii()
        or _BASE64URL_RE.fullmatch(encoded) is None
        or "=" in encoded
    ):
        raise DigestKeyConfigError(f"digest secret for key {key_id!r} is invalid")
    raw = encoded.encode("ascii")
    padded = raw + b"=" * ((4 - len(raw) % 4) % 4)
    try:
        decoded = base64.b64decode(padded, altchars=b"-_", validate=True)
    except (binascii.Error, ValueError):
        raise DigestKeyConfigError(f"digest secret for key {key_id!r} is invalid") from None
    canonical = base64.urlsafe_b64encode(decoded).rstrip(b"=")
    if len(decoded) != 32 or canonical != raw:
        raise DigestKeyConfigError(f"digest secret for key {key_id!r} is invalid")
    return decoded


class StandaloneHtmlHmacKeyring:
    """Validated in-memory secrets; its representation never includes key material."""

    __slots__ = ("_configured_current_key_id", "_secrets")

    def __init__(self, *, secrets: dict[str, bytes], current_key_id: str) -> None:
        if not isinstance(secrets, dict) or not 1 <= len(secrets) <= _MAX_ACTIVE_KEYS:
            raise DigestKeyConfigError("one to four digest keys must be configured")
        validated: dict[str, bytes] = {}
        for key_id, secret in secrets.items():
            validated_id = _validate_key_id(key_id, error_type=DigestKeyConfigError)
            if not isinstance(secret, bytes) or len(secret) != 32:
                raise DigestKeyConfigError(f"digest secret for key {validated_id!r} is invalid")
            validated[validated_id] = secret
        validated_current = _validate_key_id(
            current_key_id,
            error_type=DigestKeyConfigError,
        )
        if validated_current not in validated:
            raise DigestKeyConfigError("configured current digest key ID is unknown")
        self._secrets = MappingProxyType(validated)
        self._configured_current_key_id = validated_current

    @classmethod
    def from_json(
        cls,
        *,
        keys_json: str,
        current_key_id: str,
    ) -> StandaloneHtmlHmacKeyring:
        """Parse strict JSON containing at most four canonical 32-byte secrets."""
        if not isinstance(keys_json, str):
            raise DigestKeyConfigError("digest key JSON must be a string")
        if len(keys_json) > MAX_HMAC_KEYS_JSON_BYTES:
            raise DigestKeyConfigError(f"digest key JSON exceeds {MAX_HMAC_KEYS_JSON_BYTES} UTF-8 bytes")
        try:
            encoded_size = len(keys_json.encode("utf-8"))
        except UnicodeEncodeError:
            raise DigestKeyConfigError("digest key JSON is invalid") from None
        if encoded_size > MAX_HMAC_KEYS_JSON_BYTES:
            raise DigestKeyConfigError(f"digest key JSON exceeds {MAX_HMAC_KEYS_JSON_BYTES} UTF-8 bytes")
        try:
            parsed = json.loads(
                keys_json,
                object_pairs_hook=_strict_object,
                parse_constant=_reject_nonfinite,
            )
        except DigestKeyConfigError:
            raise
        except (json.JSONDecodeError, RecursionError, TypeError, ValueError):
            raise DigestKeyConfigError("digest key JSON is invalid") from None
        if not isinstance(parsed, dict):
            raise DigestKeyConfigError("digest key JSON must be an object")
        if not parsed:
            raise DigestKeyConfigError("at least one digest key is required")
        if len(parsed) > _MAX_ACTIVE_KEYS:
            raise DigestKeyConfigError("at most four digest keys may be configured")

        secrets: dict[str, bytes] = {}
        for key_id, encoded in parsed.items():
            validated_id = _validate_key_id(key_id, error_type=DigestKeyConfigError)
            secrets[validated_id] = _decode_secret(validated_id, encoded)
        validated_current = _validate_key_id(
            current_key_id,
            error_type=DigestKeyConfigError,
        )
        if validated_current not in secrets:
            raise DigestKeyConfigError("configured current digest key ID is unknown")
        return cls(secrets=secrets, current_key_id=validated_current)

    @classmethod
    def from_env(
        cls,
        *,
        environ: Mapping[str, object] | None = None,
    ) -> StandaloneHtmlHmacKeyring:
        """Load the two required keyring values without coercion or fallback."""
        source: Mapping[str, object] = os.environ if environ is None else environ
        if not isinstance(source, Mapping):
            raise DigestKeyConfigError("digest key environment must be a mapping")
        keys_json = source.get("SLIDES_STANDALONE_HMAC_KEYS_JSON")
        current_key_id = source.get("SLIDES_STANDALONE_HMAC_CURRENT_KEY_ID")
        if not isinstance(keys_json, str) or not keys_json:
            raise DigestKeyConfigError("SLIDES_STANDALONE_HMAC_KEYS_JSON must be a nonblank string")
        if not isinstance(current_key_id, str) or not current_key_id:
            raise DigestKeyConfigError("SLIDES_STANDALONE_HMAC_CURRENT_KEY_ID must be a nonblank string")
        return cls.from_json(
            keys_json=keys_json,
            current_key_id=current_key_id,
        )

    @property
    def configured_key_ids(self) -> tuple[str, ...]:
        """Return configured nonsecret key IDs in deterministic order."""
        return tuple(sorted(self._secrets))

    @property
    def configured_current_key_id(self) -> str:
        """Return the sole locally configured current key ID."""
        return self._configured_current_key_id

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(current_key_id={self.configured_current_key_id!r}, "
            f"key_ids={self.configured_key_ids!r})"
        )

    def _framed_digest(self, *, key_id: str, domain: HmacDomain, payload: bytes) -> str:
        if not isinstance(domain, HmacDomain):
            raise TypeError("domain must be a HmacDomain")
        if not isinstance(payload, bytes):
            raise TypeError("HMAC payload must be bytes")
        if len(payload) >= 1 << 64:
            raise ValueError("HMAC payload is too large")
        secret = self._secrets.get(key_id)
        if secret is None:
            raise DigestKeyUnavailableError(f"digest secret unavailable for shared key ID: {key_id}")
        frame = (
            _HMAC_FRAME_PREFIX
            + domain.value.encode("ascii")
            + b"\x00"
            + len(payload).to_bytes(8, byteorder="big", signed=False)
            + payload
        )
        return hmac.new(secret, frame, hashlib.sha256).hexdigest()

    @staticmethod
    def _active_key_ids(snapshot: DigestKeySnapshot) -> frozenset[str]:
        if not isinstance(snapshot, DigestKeySnapshot):
            raise TypeError("snapshot must be a DigestKeySnapshot")
        return frozenset(record.key_id for record in snapshot.records)

    def _require_snapshot_ready(self, snapshot: DigestKeySnapshot) -> None:
        if not isinstance(snapshot, DigestKeySnapshot):
            raise TypeError("snapshot must be a DigestKeySnapshot")
        snapshot.require_generation_ready()
        if snapshot.configured_current_key_id != self.configured_current_key_id:
            raise DigestKeyUnavailableError("digest key snapshot belongs to a different local configuration")
        missing = self._active_key_ids(snapshot).difference(self._secrets)
        if missing:
            raise DigestKeyUnavailableError(
                "digest secret unavailable for shared key IDs: " + ", ".join(sorted(missing))
            )

    def digest_current(
        self,
        *,
        snapshot: DigestKeySnapshot,
        domain: HmacDomain,
        payload: bytes,
    ) -> KeyedHmac:
        """Compute a framed digest with the explicitly activated current key."""
        self._require_snapshot_ready(snapshot)
        key_id = snapshot.current_key_id
        if key_id is None:  # Kept explicit for type narrowing and fail-closed behavior.
            raise DigestKeyUnavailableError("shared current digest key is unavailable")
        return KeyedHmac(
            key_id=key_id,
            digest_hex=self._framed_digest(key_id=key_id, domain=domain, payload=payload),
        )

    def digest_for_key(
        self,
        *,
        snapshot: DigestKeySnapshot,
        key_id: str,
        domain: HmacDomain,
        payload: bytes,
    ) -> KeyedHmac:
        """Compute a framed digest for one stored current/retiring key ID."""
        self._require_snapshot_ready(snapshot)
        validated_id = _validate_key_id(key_id, error_type=DigestKeyRegistryError)
        if validated_id not in self._active_key_ids(snapshot):
            raise DigestKeyRegistryError("digest key ID is not current or retiring")
        return KeyedHmac(
            key_id=validated_id,
            digest_hex=self._framed_digest(
                key_id=validated_id,
                domain=domain,
                payload=payload,
            ),
        )

    def digest_candidates(
        self,
        *,
        snapshot: DigestKeySnapshot,
        domain: HmacDomain,
        payload: bytes,
    ) -> tuple[KeyedHmac, ...]:
        """Compute the bounded current-first candidates used for replay lookup."""
        self._require_snapshot_ready(snapshot)
        ordered_ids = tuple(
            record.key_id
            for record in sorted(
                snapshot.records,
                key=lambda record: (
                    record.state is not DigestKeyState.CURRENT,
                    record.key_id,
                ),
            )
        )
        return tuple(
            self.digest_for_key(
                snapshot=snapshot,
                key_id=key_id,
                domain=domain,
                payload=payload,
            )
            for key_id in ordered_ids
        )

    def verify(
        self,
        *,
        snapshot: DigestKeySnapshot,
        key_id: str,
        domain: HmacDomain,
        payload: bytes,
        expected_digest_hex: object,
    ) -> bool:
        """Verify a strict stored digest using its stored key ID in constant time."""
        self._require_snapshot_ready(snapshot)
        if not isinstance(key_id, str) or key_id not in self._active_key_ids(snapshot):
            return False
        if not isinstance(expected_digest_hex, str) or _LOWER_HEX_SHA256_RE.fullmatch(expected_digest_hex) is None:
            return False
        actual = self._framed_digest(key_id=key_id, domain=domain, payload=payload)
        return hmac.compare_digest(actual, expected_digest_hex)


class StandaloneHtmlKeyRegistry:
    """Digest-key policy over an injected source-free shared store."""

    def __init__(
        self,
        *,
        store: DigestKeyRegistryStore,
        keyring: StandaloneHtmlHmacKeyring,
    ) -> None:
        self._store = store
        self._keyring = keyring

    def _snapshot_from_state(self, state: DigestKeyRegistryState) -> DigestKeySnapshot:
        if not isinstance(state, DigestKeyRegistryState):
            raise DigestKeyRegistryError("digest key store returned invalid state")
        if len(state.records) > _MAX_ACTIVE_KEYS:
            raise DigestKeyRegistryError("shared digest key registry exceeds four keys")
        if state.records and state.config_epoch is None:
            raise DigestKeyRegistryError("nonempty digest key registry requires a config epoch")
        key_ids = tuple(record.key_id for record in state.records)
        if len(set(key_ids)) != len(key_ids):
            raise DigestKeyRegistryError("shared digest key registry contains duplicate IDs")
        current_count = sum(record.state is DigestKeyState.CURRENT for record in state.records)
        if state.records and current_count != 1:
            raise DigestKeyRegistryError("shared digest key registry must contain exactly one current key")
        records = tuple(
            sorted(
                state.records,
                key=lambda record: (
                    record.state is not DigestKeyState.CURRENT,
                    record.key_id,
                ),
            )
        )
        missing = tuple(sorted(set(key_ids).difference(self._keyring.configured_key_ids)))
        return DigestKeySnapshot(
            records=records,
            config_epoch=state.config_epoch,
            configured_current_key_id=self._keyring.configured_current_key_id,
            availability=DigestKeyAvailability(missing_key_ids=missing),
        )

    @staticmethod
    def _validate_activation_result(
        *,
        before: DigestKeySnapshot,
        after: DigestKeySnapshot,
        desired_key_id: str,
        new_config_epoch: str,
        changed_at: datetime | None,
    ) -> None:
        if changed_at is None:
            after_by_id = {record.key_id: record for record in after.records}
            prior_current = next(
                (record for record in before.records if record.state is DigestKeyState.CURRENT),
                None,
            )
            if prior_current is not None and prior_current.key_id != desired_key_id:
                winner_prior = after_by_id.get(prior_current.key_id)
                if winner_prior is None or winner_prior.retired_at is None:
                    raise DigestKeyRotationError("digest key activation returned conflicting state")
                changed_at = winner_prior.retired_at
            else:
                winner_desired = after_by_id.get(desired_key_id)
                if winner_desired is None:
                    raise DigestKeyRotationError("digest key activation returned conflicting state")
                changed_at = winner_desired.activated_at
        expected: list[DigestKeyMetadata] = []
        desired_found = False
        for prior in before.records:
            if prior.key_id == desired_key_id:
                desired_found = True
                if prior.state is DigestKeyState.CURRENT:
                    expected.append(prior)
                else:
                    expected.append(
                        DigestKeyMetadata(
                            key_id=desired_key_id,
                            state=DigestKeyState.CURRENT,
                            activated_at=changed_at,
                            retired_at=None,
                        )
                    )
            elif prior.state is DigestKeyState.CURRENT:
                expected.append(
                    DigestKeyMetadata(
                        key_id=prior.key_id,
                        state=DigestKeyState.RETIRING,
                        activated_at=prior.activated_at,
                        retired_at=changed_at,
                    )
                )
            else:
                expected.append(prior)
        if not desired_found:
            expected.append(
                DigestKeyMetadata(
                    key_id=desired_key_id,
                    state=DigestKeyState.CURRENT,
                    activated_at=changed_at,
                    retired_at=None,
                )
            )
        expected_records = tuple(
            sorted(
                expected,
                key=lambda record: (
                    record.state is not DigestKeyState.CURRENT,
                    record.key_id,
                ),
            )
        )
        if after.config_epoch != new_config_epoch or after.records != expected_records:
            raise DigestKeyRotationError("digest key activation returned conflicting state")

    async def snapshot(self) -> DigestKeySnapshot:
        """Read shared state without activating or rotating any key."""
        return self._snapshot_from_state(await self._store.load_digest_key_registry())

    async def activate_configured_current(
        self,
        *,
        expected_current_key_id: str | None,
        expected_config_epoch: str | None,
        new_config_epoch: str,
        now: datetime,
    ) -> DigestKeySnapshot:
        """Explicitly activate the configured key with expected-state/epoch fencing."""
        if expected_current_key_id is not None:
            _validate_key_id(
                expected_current_key_id,
                error_type=DigestKeyRotationError,
            )
        _validate_config_epoch(
            expected_config_epoch,
            allow_none=True,
            error_type=DigestKeyRotationError,
        )
        _validate_config_epoch(new_config_epoch, error_type=DigestKeyRotationError)
        _validate_utc_timestamp(
            now,
            field_name="now",
            error_type=DigestKeyRotationError,
        )

        state = await self._store.load_digest_key_registry()
        before = self._snapshot_from_state(state)
        before.require_secrets_available()
        desired = self._keyring.configured_current_key_id
        if before.current_key_id == desired and state.config_epoch == new_config_epoch:
            before.require_generation_ready()
            return before
        prior_current = next(
            (record for record in before.records if record.state is DigestKeyState.CURRENT),
            None,
        )
        if prior_current is not None and now < prior_current.activated_at:
            raise DigestKeyRotationError("digest key rotation time precedes current-key activation")
        desired_prior = next(
            (record for record in before.records if record.key_id == desired),
            None,
        )
        if (
            desired_prior is not None
            and desired_prior.state is DigestKeyState.RETIRING
            and desired_prior.retired_at is not None
            and now < desired_prior.retired_at
        ):
            raise DigestKeyRotationError("digest key rotation time precedes desired-key retirement")
        if before.current_key_id != expected_current_key_id or state.config_epoch != expected_config_epoch:
            raise DigestKeyRotationError("digest key activation compare-and-swap conflict")
        if before.current_key_id not in (None, desired) and new_config_epoch == expected_config_epoch:
            raise DigestKeyRotationError("digest key rotation requires a new config epoch")
        if desired not in {record.key_id for record in before.records} and len(before.records) >= _MAX_ACTIVE_KEYS:
            raise DigestKeyRotationError("digest key rotation requires retiring an eligible key first")

        result = await self._store.compare_and_swap_current_key(
            expected_current_key_id=expected_current_key_id,
            expected_config_epoch=expected_config_epoch,
            new_current_key_id=desired,
            new_config_epoch=new_config_epoch,
            changed_at=now,
        )
        if result is None:
            raise DigestKeyRotationError("digest key activation compare-and-swap conflict")
        if not isinstance(result, CurrentKeyCasResult):
            raise DigestKeyRotationError("digest key activation returned invalid state")
        after = self._snapshot_from_state(result.state)
        after.require_secrets_available()
        self._validate_activation_result(
            before=before,
            after=after,
            desired_key_id=desired,
            new_config_epoch=new_config_epoch,
            changed_at=now if result.applied_here else None,
        )
        after.require_generation_ready()
        return after

    async def remove_retired_key(
        self,
        *,
        key_id: str,
        now: datetime,
    ) -> DigestKeySnapshot:
        """Remove a retiring key only after the floor and a complete fenced sweep."""
        validated_id = _validate_key_id(
            key_id,
            error_type=DigestKeyRetirementError,
        )
        _validate_utc_timestamp(
            now,
            field_name="now",
            error_type=DigestKeyRetirementError,
        )
        state = await self._store.load_digest_key_registry()
        before = self._snapshot_from_state(state)
        record = next(
            (item for item in before.records if item.key_id == validated_id),
            None,
        )
        if record is None:
            return before
        if record.state is not DigestKeyState.RETIRING or record.retired_at is None:
            raise DigestKeyRetirementError("current digest key cannot be removed")
        eligible_at = record.retired_at + _RETIREMENT_FLOOR
        if now < eligible_at:
            raise DigestKeyRetirementError("digest key cannot be removed before the 32-day retirement floor")

        proof = await self._store.load_dormant_sweep_proof(key_id=validated_id)
        if (
            not isinstance(proof, DormantSweepProof)
            or proof.key_id != validated_id
            or proof.config_epoch != state.config_epoch
            or not proof.complete
            or proof.unexpired_reference_count != 0
            or proof.sweep_started_at < eligible_at
            or proof.sweep_completed_at > now
        ):
            raise DigestKeyRetirementError("complete fenced dormant-database sweep proof is required")
        result = await self._store.compare_and_swap_remove_key(
            key_id=validated_id,
            expected_retired_at=record.retired_at,
            expected_config_epoch=state.config_epoch,
        )
        if result is None:
            raise DigestKeyRetirementError("digest key removal compare-and-swap conflict")
        after = self._snapshot_from_state(result)
        expected_records = tuple(item for item in before.records if item.key_id != validated_id)
        if after.config_epoch != before.config_epoch or after.records != expected_records:
            raise DigestKeyRetirementError("digest key removal returned conflicting state")
        return after

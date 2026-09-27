from __future__ import annotations

import base64
import dataclasses
import json
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from tldw_Server_API.app.core.Slides.standalone_html_registry import (
    MAX_HMAC_KEYS_JSON_BYTES,
    CurrentKeyCasResult,
    DigestKeyConfigError,
    DigestKeyMetadata,
    DigestKeyRegistryError,
    DigestKeyRegistryState,
    DigestKeyRetirementError,
    DigestKeyRotationError,
    DigestKeyState,
    DigestKeyUnavailableError,
    DormantSweepProof,
    HmacDomain,
    StandaloneHtmlHmacKeyring,
    StandaloneHtmlKeyRegistry,
)

UTC = timezone.utc
NOW = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)


def _encoded_secret(fill: int) -> str:
    return base64.urlsafe_b64encode(bytes([fill]) * 32).decode("ascii").rstrip("=")


def _keys_json(*key_ids: str) -> str:
    pairs = ",".join(f'"{key_id}":"{_encoded_secret(index + 1)}"' for index, key_id in enumerate(key_ids))
    return "{" + pairs + "}"


def _keyring(*key_ids: str, current: str | None = None) -> StandaloneHtmlHmacKeyring:
    return StandaloneHtmlHmacKeyring.from_json(
        keys_json=_keys_json(*key_ids),
        current_key_id=current or key_ids[0],
    )


def _coordination_epoch(generation: int, fill: str) -> str:
    digest = fill * 64
    return f"sha256:{digest}" if generation == 0 else f"v1:g{generation}:sha256:{digest}"


def _current(key_id: str, *, activated_at: datetime = NOW) -> DigestKeyMetadata:
    return DigestKeyMetadata(
        key_id=key_id,
        state=DigestKeyState.CURRENT,
        activated_at=activated_at,
        retired_at=None,
    )


def _retiring(
    key_id: str,
    *,
    retired_at: datetime,
    activated_at: datetime | None = None,
) -> DigestKeyMetadata:
    return DigestKeyMetadata(
        key_id=key_id,
        state=DigestKeyState.RETIRING,
        activated_at=activated_at or retired_at - timedelta(days=30),
        retired_at=retired_at,
    )


class FakeRegistryStore:
    """Source-free fake for the Task 7 Jobs-store adapter."""

    def __init__(
        self,
        *,
        records: tuple[DigestKeyMetadata, ...] = (),
        config_epoch: str | None = None,
    ) -> None:
        self.state = DigestKeyRegistryState(records=records, config_epoch=config_epoch)
        self.proofs: dict[str, DormantSweepProof] = {}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def load_digest_key_registry(self) -> DigestKeyRegistryState:
        self.calls.append(("load", {}))
        return self.state

    async def compare_and_swap_current_key(
        self,
        *,
        expected_current_key_id: str | None,
        expected_config_epoch: str | None,
        new_current_key_id: str,
        new_config_epoch: str,
        changed_at: datetime,
    ) -> CurrentKeyCasResult | None:
        arguments = {
            "expected_current_key_id": expected_current_key_id,
            "expected_config_epoch": expected_config_epoch,
            "new_current_key_id": new_current_key_id,
            "new_config_epoch": new_config_epoch,
            "changed_at": changed_at,
        }
        self.calls.append(("current_cas", arguments))
        current = next(
            (record for record in self.state.records if record.state is DigestKeyState.CURRENT),
            None,
        )
        current_id = current.key_id if current is not None else None
        if current_id == new_current_key_id and self.state.config_epoch == new_config_epoch:
            return CurrentKeyCasResult(state=self.state, applied_here=False)
        if current_id != expected_current_key_id or self.state.config_epoch != expected_config_epoch:
            return None
        if current_id == new_current_key_id:
            self.state = DigestKeyRegistryState(
                records=self.state.records,
                config_epoch=new_config_epoch,
            )
            return CurrentKeyCasResult(state=self.state, applied_here=True)

        replacement: list[DigestKeyMetadata] = []
        found_new = False
        for record in self.state.records:
            if record.key_id == new_current_key_id:
                replacement.append(_current(record.key_id, activated_at=changed_at))
                found_new = True
            elif record.state is DigestKeyState.CURRENT:
                replacement.append(
                    DigestKeyMetadata(
                        key_id=record.key_id,
                        state=DigestKeyState.RETIRING,
                        activated_at=record.activated_at,
                        retired_at=changed_at,
                    )
                )
            else:
                replacement.append(record)
        if not found_new:
            replacement.append(_current(new_current_key_id, activated_at=changed_at))
        self.state = DigestKeyRegistryState(
            records=tuple(replacement),
            config_epoch=new_config_epoch,
        )
        return CurrentKeyCasResult(state=self.state, applied_here=True)

    async def load_dormant_sweep_proof(self, *, key_id: str) -> DormantSweepProof | None:
        self.calls.append(("load_proof", {"key_id": key_id}))
        return self.proofs.get(key_id)

    async def compare_and_swap_remove_key(
        self,
        *,
        key_id: str,
        expected_retired_at: datetime,
        expected_config_epoch: str | None,
    ) -> DigestKeyRegistryState | None:
        arguments = {
            "key_id": key_id,
            "expected_retired_at": expected_retired_at,
            "expected_config_epoch": expected_config_epoch,
        }
        self.calls.append(("remove_cas", arguments))
        record = next((item for item in self.state.records if item.key_id == key_id), None)
        if record is None:
            return self.state
        if (
            record.state is not DigestKeyState.RETIRING
            or record.retired_at != expected_retired_at
            or self.state.config_epoch != expected_config_epoch
        ):
            return None
        self.state = DigestKeyRegistryState(
            records=tuple(item for item in self.state.records if item.key_id != key_id),
            config_epoch=self.state.config_epoch,
        )
        return self.state


def _proof(
    key_id: str,
    *,
    config_epoch: str = "epoch-2",
    sweep_started_at: datetime,
    sweep_completed_at: datetime | None = None,
    complete: bool = True,
    unexpired_reference_count: int = 0,
) -> DormantSweepProof:
    return DormantSweepProof(
        key_id=key_id,
        config_epoch=config_epoch,
        fencing_token=7,
        sweep_started_at=sweep_started_at,
        sweep_completed_at=sweep_completed_at or sweep_started_at + timedelta(minutes=1),
        complete=complete,
        unexpired_reference_count=unexpired_reference_count,
    )


@pytest.mark.parametrize(
    "keys_json,current_key_id",
    [
        (f'{{"a":"{_encoded_secret(1)}","a":"{_encoded_secret(2)}"}}', "a"),
        ('{"a":NaN}', "a"),
        ('{"a":Infinity}', "a"),
        ("[]", "a"),
        ('{"a":1}', "a"),
    ],
)
def test_keyring_rejects_nonstrict_json_without_echoing_secret(
    keys_json: str,
    current_key_id: str,
) -> None:
    with pytest.raises(DigestKeyConfigError) as exc_info:
        StandaloneHtmlHmacKeyring.from_json(
            keys_json=keys_json,
            current_key_id=current_key_id,
        )

    assert _encoded_secret(1) not in str(exc_info.value)


@pytest.mark.parametrize(
    "key_id",
    ["", "x" * 33, "caf\u00e9", " space", "line\nfeed", "a/b", "a:b", ".hidden", "-dash"],
)
def test_keyring_rejects_ids_outside_one_to_thirty_two_ascii_characters(key_id: str) -> None:
    with pytest.raises(DigestKeyConfigError):
        StandaloneHtmlHmacKeyring.from_json(
            keys_json=json.dumps({key_id: _encoded_secret(1)}),
            current_key_id=key_id,
        )


def test_keyring_accepts_id_boundaries_and_requires_one_known_current_id() -> None:
    keyring = StandaloneHtmlHmacKeyring.from_json(
        keys_json=_keys_json("x", "y" * 32),
        current_key_id="y" * 32,
    )

    assert keyring.configured_key_ids == ("x", "y" * 32)
    assert keyring.configured_current_key_id == "y" * 32

    for current_key_id in ("", "missing"):
        with pytest.raises(DigestKeyConfigError):
            StandaloneHtmlHmacKeyring.from_json(
                keys_json=_keys_json("x"),
                current_key_id=current_key_id,
            )


@pytest.mark.parametrize(
    "encoded",
    [
        _encoded_secret(1) + "=",
        base64.b64encode(bytes([255]) * 32).decode("ascii").rstrip("="),
        _encoded_secret(0)[:-1] + "B",
        base64.urlsafe_b64encode(b"short").decode("ascii").rstrip("="),
        "!" * 43,
    ],
)
def test_keyring_requires_canonical_unpadded_base64url_for_exactly_32_bytes(encoded: str) -> None:
    with pytest.raises(DigestKeyConfigError) as exc_info:
        StandaloneHtmlHmacKeyring.from_json(
            keys_json=f'{{"a":"{encoded}"}}',
            current_key_id="a",
        )

    assert encoded not in str(exc_info.value)


def test_keyring_rejects_more_than_four_configured_keys() -> None:
    with pytest.raises(DigestKeyConfigError):
        _keyring("a", "b", "c", "d", "e")


def test_keyring_bounds_strict_json_by_utf8_bytes_before_decoding() -> None:
    valid = _keys_json("a")
    at_limit = valid + " " * (MAX_HMAC_KEYS_JSON_BYTES - len(valid.encode("utf-8")))

    assert (
        StandaloneHtmlHmacKeyring.from_json(
            keys_json=at_limit,
            current_key_id="a",
        ).configured_current_key_id
        == "a"
    )
    with pytest.raises(DigestKeyConfigError):
        StandaloneHtmlHmacKeyring.from_json(
            keys_json=at_limit + " ",
            current_key_id="a",
        )


def test_keyring_rejects_oversized_text_before_allocating_utf8_copy() -> None:
    class EncodeMustNotRun(str):
        def encode(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("oversized input must be rejected before encode")

    oversized = EncodeMustNotRun("x" * (MAX_HMAC_KEYS_JSON_BYTES + 1))

    with pytest.raises(DigestKeyConfigError):
        StandaloneHtmlHmacKeyring.from_json(
            keys_json=oversized,
            current_key_id="a",
        )


def test_keyring_reduces_deep_json_recursion_to_closed_config_error() -> None:
    deeply_nested = "[" * 2_000 + "0" + "]" * 2_000
    assert len(deeply_nested.encode("utf-8")) <= MAX_HMAC_KEYS_JSON_BYTES

    with pytest.raises(DigestKeyConfigError):
        StandaloneHtmlHmacKeyring.from_json(
            keys_json=deeply_nested,
            current_key_id="a",
        )


def test_malformed_key_json_suppresses_secret_bearing_decoder_exception() -> None:
    malformed = f'{{"a":"{_encoded_secret(7)}"'

    with pytest.raises(DigestKeyConfigError) as exc_info:
        StandaloneHtmlHmacKeyring.from_json(
            keys_json=malformed,
            current_key_id="a",
        )

    assert exc_info.value.__cause__ is None
    assert exc_info.value.__suppress_context__
    assert _encoded_secret(7) not in str(exc_info.value)


@pytest.mark.parametrize(
    "environ",
    [
        {},
        {"SLIDES_STANDALONE_HMAC_KEYS_JSON": _keys_json("a")},
        {"SLIDES_STANDALONE_HMAC_CURRENT_KEY_ID": "a"},
        {
            "SLIDES_STANDALONE_HMAC_KEYS_JSON": " ",
            "SLIDES_STANDALONE_HMAC_CURRENT_KEY_ID": "a",
        },
        {
            "SLIDES_STANDALONE_HMAC_KEYS_JSON": _keys_json("a"),
            "SLIDES_STANDALONE_HMAC_CURRENT_KEY_ID": " ",
        },
        {
            "SLIDES_STANDALONE_HMAC_KEYS_JSON": 1,
            "SLIDES_STANDALONE_HMAC_CURRENT_KEY_ID": "a",
        },
        {
            "SLIDES_STANDALONE_HMAC_KEYS_JSON": _keys_json("a"),
            "SLIDES_STANDALONE_HMAC_CURRENT_KEY_ID": None,
        },
    ],
)
def test_keyring_env_loader_rejects_missing_blank_or_nonstring_values(
    environ: dict[str, object],
) -> None:
    with pytest.raises(DigestKeyConfigError):
        StandaloneHtmlHmacKeyring.from_env(environ=environ)


def test_keyring_env_loader_uses_exact_injected_values() -> None:
    class StripMustNotRun(str):
        def strip(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("environment values must not be normalized")

    keyring = StandaloneHtmlHmacKeyring.from_env(
        environ={
            "SLIDES_STANDALONE_HMAC_KEYS_JSON": StripMustNotRun(_keys_json("old", "current")),
            "SLIDES_STANDALONE_HMAC_CURRENT_KEY_ID": StripMustNotRun("current"),
        }
    )

    assert keyring.configured_key_ids == ("current", "old")
    assert keyring.configured_current_key_id == "current"


def test_keyring_repr_never_contains_secret_material() -> None:
    encoded = _encoded_secret(9)
    keyring = StandaloneHtmlHmacKeyring.from_json(
        keys_json=f'{{"safe-id":"{encoded}"}}',
        current_key_id="safe-id",
    )

    rendered = repr(keyring)

    assert encoded not in rendered
    assert (bytes([9]) * 32).hex() not in rendered
    assert "safe-id" in rendered


@pytest.mark.asyncio
async def test_hmacs_use_current_key_closed_domains_and_unambiguous_framing() -> None:
    keyring = _keyring("old", "current", current="current")
    store = FakeRegistryStore(
        records=(
            _current("current"),
            _retiring("old", retired_at=NOW - timedelta(days=1)),
        ),
        config_epoch="epoch-2",
    )
    snapshot = await StandaloneHtmlKeyRegistry(store=store, keyring=keyring).snapshot()

    all_domain_digests = {
        keyring.digest_current(
            snapshot=snapshot,
            domain=domain,
            payload=b"a\x00b",
        ).digest_hex
        for domain in HmacDomain
    }
    request_digest = keyring.digest_current(
        snapshot=snapshot,
        domain=HmacDomain.CLIENT_REQUEST,
        payload=b"a\x00b",
    )
    framed_differently = keyring.digest_current(
        snapshot=snapshot,
        domain=HmacDomain.CLIENT_REQUEST,
        payload=b"a\x00b\x00",
    )

    assert request_digest.key_id == "current"
    assert len(request_digest.digest_hex) == 64
    assert request_digest.digest_hex == request_digest.digest_hex.lower()
    assert len(HmacDomain) == 5
    assert len(all_domain_digests) == len(HmacDomain)
    assert framed_differently.digest_hex not in all_domain_digests
    with pytest.raises(TypeError):
        keyring.digest_current(  # type: ignore[arg-type]
            snapshot=snapshot,
            domain="client-request",
            payload=b"payload",
        )


@pytest.mark.asyncio
async def test_hmac_v1_framing_and_closed_domain_values_are_compatibility_pinned() -> None:
    assert [(domain.name, domain.value) for domain in HmacDomain] == [
        ("CLIENT_IDEMPOTENCY_KEY", "client-idempotency-key"),
        ("CLIENT_REQUEST", "client-request"),
        ("SOURCE_SNAPSHOT", "source-snapshot"),
        ("EXECUTION_MANIFEST", "execution-manifest"),
        ("JOBS_IDEMPOTENCY_KEY", "jobs-idempotency-key"),
    ]
    keyring = _keyring("current")
    snapshot = await StandaloneHtmlKeyRegistry(
        store=FakeRegistryStore(
            records=(_current("current"),),
            config_epoch="epoch-1",
        ),
        keyring=keyring,
    ).snapshot()

    digest = keyring.digest_current(
        snapshot=snapshot,
        domain=HmacDomain.CLIENT_REQUEST,
        payload=b"known\x00payload",
    )

    assert digest.key_id == "current"
    assert digest.digest_hex == "b3482bdb1cff22b026ec95fa15cba3e5dc5d12df921bc88e2fbb18feca7826e1"
    assert digest.digest_hex not in repr(digest)


@pytest.mark.asyncio
async def test_verification_uses_stored_key_id_strict_shape_and_compare_digest(monkeypatch) -> None:
    keyring = _keyring("old", "current", current="current")
    store = FakeRegistryStore(
        records=(
            _current("current"),
            _retiring("old", retired_at=NOW - timedelta(days=1)),
        ),
        config_epoch="epoch-2",
    )
    snapshot = await StandaloneHtmlKeyRegistry(store=store, keyring=keyring).snapshot()
    digest = keyring.digest_for_key(
        snapshot=snapshot,
        key_id="old",
        domain=HmacDomain.EXECUTION_MANIFEST,
        payload=b"manifest",
    )
    calls: list[tuple[str, str]] = []

    def _compare(left: str, right: str) -> bool:
        calls.append((left, right))
        return left == right

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Slides.standalone_html_registry.hmac.compare_digest",
        _compare,
    )

    assert keyring.verify(
        snapshot=snapshot,
        key_id="old",
        domain=HmacDomain.EXECUTION_MANIFEST,
        payload=b"manifest",
        expected_digest_hex=digest.digest_hex,
    )
    assert not keyring.verify(
        snapshot=snapshot,
        key_id="old",
        domain=HmacDomain.EXECUTION_MANIFEST,
        payload=b"manifest",
        expected_digest_hex=digest.digest_hex.upper(),
    )
    assert not keyring.verify(
        snapshot=snapshot,
        key_id="old",
        domain=HmacDomain.EXECUTION_MANIFEST,
        payload=b"manifest",
        expected_digest_hex="0" * 63,
    )
    assert not keyring.verify(
        snapshot=snapshot,
        key_id="old",
        domain=HmacDomain.EXECUTION_MANIFEST,
        payload=b"manifest",
        expected_digest_hex="g" * 64,
    )
    assert len(calls) == 1
    assert all(len(left) == len(right) == 64 for left, right in calls)


def test_source_free_metadata_requires_coherent_state_and_utc_timestamps() -> None:
    with pytest.raises(DigestKeyRegistryError):
        DigestKeyMetadata(
            key_id="current",
            state=DigestKeyState.CURRENT,
            activated_at=NOW,
            retired_at=NOW,
        )
    with pytest.raises(DigestKeyRegistryError):
        DigestKeyMetadata(
            key_id="retiring",
            state=DigestKeyState.RETIRING,
            activated_at=NOW,
            retired_at=None,
        )
    with pytest.raises(DigestKeyRegistryError):
        _retiring("old", activated_at=NOW, retired_at=NOW - timedelta(seconds=1))
    with pytest.raises(DigestKeyRegistryError):
        _current("current", activated_at=NOW.replace(tzinfo=None))
    with pytest.raises(DigestKeyRegistryError):
        _current(
            "current",
            activated_at=NOW.astimezone(timezone(timedelta(hours=1))),
        )


@pytest.mark.parametrize(
    "updates",
    [
        {"fencing_token": 0},
        {"fencing_token": True},
        {"sweep_started_at": NOW.replace(tzinfo=None)},
        {"sweep_completed_at": NOW - timedelta(seconds=1)},
        {"unexpired_reference_count": -1},
    ],
)
def test_sweep_proof_requires_positive_fence_ordered_utc_interval_and_count(
    updates: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "key_id": "old",
        "config_epoch": "epoch-2",
        "fencing_token": 1,
        "sweep_started_at": NOW,
        "sweep_completed_at": NOW + timedelta(seconds=1),
        "complete": True,
        "unexpired_reference_count": 0,
    }
    values.update(updates)

    with pytest.raises(DigestKeyRegistryError):
        DormantSweepProof(**values)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_snapshot_reports_one_global_missing_secret_state_for_admission_and_worker() -> None:
    keyring = _keyring("current")
    store = FakeRegistryStore(
        records=(
            _current("current"),
            _retiring("lost", retired_at=NOW - timedelta(days=1)),
        ),
        config_epoch="epoch-1",
    )
    snapshot = await StandaloneHtmlKeyRegistry(store=store, keyring=keyring).snapshot()

    assert not snapshot.availability.available
    assert snapshot.availability.missing_key_ids == ("lost",)
    assert not snapshot.generation_ready
    for operation in (
        lambda: keyring.digest_current(
            snapshot=snapshot,
            domain=HmacDomain.CLIENT_REQUEST,
            payload=b"new admission",
        ),
        lambda: keyring.verify(
            snapshot=snapshot,
            key_id="current",
            domain=HmacDomain.EXECUTION_MANIFEST,
            payload=b"worker",
            expected_digest_hex="0" * 64,
        ),
    ):
        with pytest.raises(DigestKeyUnavailableError) as exc_info:
            operation()
        assert exc_info.value.error_code == "generation_digest_key_unavailable"


@pytest.mark.asyncio
async def test_snapshot_is_read_only_and_does_not_rotate_from_local_configuration() -> None:
    keyring = _keyring("old", "new", current="new")
    store = FakeRegistryStore(
        records=(_current("old"),),
        config_epoch="epoch-1",
    )
    registry = StandaloneHtmlKeyRegistry(store=store, keyring=keyring)

    snapshot = await registry.snapshot()

    assert snapshot.current_key_id == "old"
    assert snapshot.availability.available
    assert not snapshot.generation_ready
    assert [name for name, _ in store.calls] == ["load"]


@pytest.mark.asyncio
async def test_nonempty_registry_requires_safe_nonnull_config_epoch() -> None:
    store = FakeRegistryStore(records=(_current("current"),), config_epoch=None)

    with pytest.raises(DigestKeyRegistryError):
        await StandaloneHtmlKeyRegistry(
            store=store,
            keyring=_keyring("current"),
        ).snapshot()


@pytest.mark.parametrize("config_epoch", [" epoch", "line\nfeed", "a/b", "caf\u00e9"])
def test_registry_epoch_is_a_safe_visible_revision_token(config_epoch: str) -> None:
    with pytest.raises(DigestKeyRegistryError):
        DigestKeyRegistryState(records=(), config_epoch=config_epoch)

    accepted = DigestKeyRegistryState(
        records=(),
        config_epoch="sha256:0123_ab-CD.ef",
    )
    assert accepted.config_epoch == "sha256:0123_ab-CD.ef"


@pytest.mark.asyncio
async def test_explicit_current_key_cas_bootstraps_and_rotates_with_source_free_values() -> None:
    keyring = _keyring("old", "new", current="new")
    store = FakeRegistryStore(records=(), config_epoch=None)
    registry = StandaloneHtmlKeyRegistry(store=store, keyring=keyring)

    bootstrapped = await registry.activate_configured_current(
        expected_current_key_id=None,
        expected_config_epoch=None,
        new_config_epoch="epoch-1",
        now=NOW,
    )
    assert bootstrapped.current_key_id == "new"
    assert bootstrapped.generation_ready

    rotating_keyring = _keyring("new", "next", current="next")
    rotating_registry = StandaloneHtmlKeyRegistry(store=store, keyring=rotating_keyring)
    rotated = await rotating_registry.activate_configured_current(
        expected_current_key_id="new",
        expected_config_epoch="epoch-1",
        new_config_epoch="epoch-2",
        now=NOW + timedelta(hours=1),
    )

    assert rotated.current_key_id == "next"
    prior = next(record for record in rotated.records if record.key_id == "new")
    assert prior.state is DigestKeyState.RETIRING
    assert prior.retired_at == NOW + timedelta(hours=1)
    assert rotated.generation_ready


@pytest.mark.asyncio
async def test_identical_rotation_race_is_idempotent_but_conflicting_epoch_fails_closed() -> None:
    store = FakeRegistryStore(records=(_current("old"),), config_epoch="epoch-1")
    registry = StandaloneHtmlKeyRegistry(
        store=store,
        keyring=_keyring("old", "new", current="new"),
    )
    first = await registry.activate_configured_current(
        expected_current_key_id="old",
        expected_config_epoch="epoch-1",
        new_config_epoch="epoch-2",
        now=NOW,
    )
    second = await registry.activate_configured_current(
        expected_current_key_id="old",
        expected_config_epoch="epoch-1",
        new_config_epoch="epoch-2",
        now=NOW,
    )

    assert first == second

    conflicting = StandaloneHtmlKeyRegistry(
        store=store,
        keyring=_keyring("old", "other", "new", current="other"),
    )
    with pytest.raises(DigestKeyRotationError):
        await conflicting.activate_configured_current(
            expected_current_key_id="old",
            expected_config_epoch="epoch-1",
            new_config_epoch="epoch-3",
            now=NOW,
        )


@pytest.mark.asyncio
async def test_identical_rotation_race_accepts_winners_different_timestamp() -> None:
    initial_state = DigestKeyRegistryState(
        records=(_current("old", activated_at=NOW - timedelta(days=1)),),
        config_epoch="epoch-1",
    )

    class StaleReadStore(FakeRegistryStore):
        def __init__(self) -> None:
            super().__init__(
                records=initial_state.records,
                config_epoch=initial_state.config_epoch,
            )
            self.loads = 0

        async def load_digest_key_registry(self) -> DigestKeyRegistryState:
            self.calls.append(("load", {}))
            self.loads += 1
            if self.loads <= 2:
                return initial_state
            return self.state

    registry = StandaloneHtmlKeyRegistry(
        store=StaleReadStore(),
        keyring=_keyring("old", "new", current="new"),
    )
    winner = await registry.activate_configured_current(
        expected_current_key_id="old",
        expected_config_epoch="epoch-1",
        new_config_epoch="epoch-2",
        now=NOW,
    )

    loser = await registry.activate_configured_current(
        expected_current_key_id="old",
        expected_config_epoch="epoch-1",
        new_config_epoch="epoch-2",
        now=NOW + timedelta(microseconds=1),
    )

    assert loser == winner


@pytest.mark.asyncio
async def test_real_rotation_requires_a_new_config_epoch() -> None:
    store = FakeRegistryStore(records=(_current("old"),), config_epoch="epoch-1")
    registry = StandaloneHtmlKeyRegistry(
        store=store,
        keyring=_keyring("old", "new", current="new"),
    )

    with pytest.raises(DigestKeyRotationError):
        await registry.activate_configured_current(
            expected_current_key_id="old",
            expected_config_epoch="epoch-1",
            new_config_epoch="epoch-1",
            now=NOW,
        )
    assert [name for name, _ in store.calls] == ["load"]


@pytest.mark.asyncio
async def test_activation_normalizes_invalid_epoch_to_rotation_error() -> None:
    registry = StandaloneHtmlKeyRegistry(
        store=FakeRegistryStore(records=(_current("old"),), config_epoch="epoch-1"),
        keyring=_keyring("old", "new", current="new"),
    )

    with pytest.raises(DigestKeyRotationError):
        await registry.activate_configured_current(
            expected_current_key_id="old",
            expected_config_epoch="epoch-1",
            new_config_epoch="bad epoch",
            now=NOW,
        )


@pytest.mark.asyncio
async def test_same_current_key_can_advance_epoch_without_changing_key_metadata() -> None:
    activated_at = NOW - timedelta(days=10)
    store = FakeRegistryStore(
        records=(_current("current", activated_at=activated_at),),
        config_epoch="epoch-1",
    )

    snapshot = await StandaloneHtmlKeyRegistry(
        store=store,
        keyring=_keyring("current"),
    ).activate_configured_current(
        expected_current_key_id="current",
        expected_config_epoch="epoch-1",
        new_config_epoch="epoch-2",
        now=NOW,
    )

    assert snapshot.config_epoch == "epoch-2"
    assert snapshot.records == (_current("current", activated_at=activated_at),)


@pytest.mark.asyncio
@pytest.mark.parametrize("same_key", [False, True])
async def test_production_epoch_rejects_lower_coordination_generation_before_cas(
    same_key: bool,
) -> None:
    current_key = "current"
    desired_key = current_key if same_key else "prior"
    records = [_current(current_key, activated_at=NOW - timedelta(days=2))]
    if not same_key:
        records.append(
            _retiring(
                desired_key,
                activated_at=NOW - timedelta(days=10),
                retired_at=NOW - timedelta(days=1),
            )
        )
    store = FakeRegistryStore(
        records=tuple(records),
        config_epoch=_coordination_epoch(2, "b"),
    )
    registry = StandaloneHtmlKeyRegistry(
        store=store,
        keyring=_keyring(
            *((current_key,) if same_key else (current_key, desired_key)),
            current=desired_key,
        ),
    )

    with pytest.raises(DigestKeyRotationError, match="coordination generation"):
        await registry.activate_configured_current(
            expected_current_key_id=current_key,
            expected_config_epoch=_coordination_epoch(2, "b"),
            new_config_epoch=_coordination_epoch(1, "a"),
            now=NOW,
        )

    assert [name for name, _ in store.calls] == ["load"]


@pytest.mark.asyncio
async def test_production_epoch_rejects_equal_generation_different_manifest_before_cas() -> None:
    store = FakeRegistryStore(
        records=(_current("current", activated_at=NOW - timedelta(days=1)),),
        config_epoch=_coordination_epoch(4, "a"),
    )
    registry = StandaloneHtmlKeyRegistry(
        store=store,
        keyring=_keyring("current"),
    )

    with pytest.raises(DigestKeyRotationError, match="coordination generation"):
        await registry.activate_configured_current(
            expected_current_key_id="current",
            expected_config_epoch=_coordination_epoch(4, "a"),
            new_config_epoch=_coordination_epoch(4, "b"),
            now=NOW,
        )

    assert [name for name, _ in store.calls] == ["load"]


@pytest.mark.asyncio
async def test_higher_production_generation_succeeds_and_exact_winner_is_idempotent() -> None:
    store = FakeRegistryStore(
        records=(_current("old", activated_at=NOW - timedelta(days=1)),),
        config_epoch=_coordination_epoch(1, "a"),
    )
    registry = StandaloneHtmlKeyRegistry(
        store=store,
        keyring=_keyring("old", "new", current="new"),
    )
    kwargs = {
        "expected_current_key_id": "old",
        "expected_config_epoch": _coordination_epoch(1, "a"),
        "new_config_epoch": _coordination_epoch(2, "b"),
        "now": NOW,
    }

    first = await registry.activate_configured_current(**kwargs)
    second = await registry.activate_configured_current(**kwargs)

    assert first == second
    assert first.current_key_id == "new"
    assert first.config_epoch == _coordination_epoch(2, "b")
    assert [name for name, _ in store.calls].count("current_cas") == 1


@pytest.mark.asyncio
async def test_forward_then_rollback_generation_prevents_stale_aba_oscillation() -> None:
    store = FakeRegistryStore(
        records=(_current("a", activated_at=NOW - timedelta(days=2)),),
        config_epoch=_coordination_epoch(1, "a"),
    )
    forward = StandaloneHtmlKeyRegistry(
        store=store,
        keyring=_keyring("a", "b", current="b"),
    )
    await forward.activate_configured_current(
        expected_current_key_id="a",
        expected_config_epoch=_coordination_epoch(1, "a"),
        new_config_epoch=_coordination_epoch(2, "b"),
        now=NOW,
    )
    rollback = StandaloneHtmlKeyRegistry(
        store=store,
        keyring=_keyring("a", "b", current="a"),
    )
    await rollback.activate_configured_current(
        expected_current_key_id="b",
        expected_config_epoch=_coordination_epoch(2, "b"),
        new_config_epoch=_coordination_epoch(3, "a"),
        now=NOW + timedelta(minutes=1),
    )
    cas_count = [name for name, _ in store.calls].count("current_cas")

    for stale_registry, stale_epoch in (
        (rollback, _coordination_epoch(1, "a")),
        (forward, _coordination_epoch(2, "b")),
    ):
        with pytest.raises(DigestKeyRotationError, match="coordination generation"):
            await stale_registry.activate_configured_current(
                expected_current_key_id="a",
                expected_config_epoch=_coordination_epoch(3, "a"),
                new_config_epoch=stale_epoch,
                now=NOW + timedelta(minutes=2),
            )

    assert [name for name, _ in store.calls].count("current_cas") == cas_count
    assert store.state.config_epoch == _coordination_epoch(3, "a")


@pytest.mark.asyncio
async def test_legacy_production_epoch_is_generation_zero() -> None:
    store = FakeRegistryStore(
        records=(_current("current", activated_at=NOW - timedelta(days=1)),),
        config_epoch=_coordination_epoch(0, "a"),
    )
    registry = StandaloneHtmlKeyRegistry(
        store=store,
        keyring=_keyring("current"),
    )

    with pytest.raises(DigestKeyRotationError, match="coordination generation"):
        await registry.activate_configured_current(
            expected_current_key_id="current",
            expected_config_epoch=_coordination_epoch(0, "a"),
            new_config_epoch=_coordination_epoch(0, "b"),
            now=NOW,
        )
    advanced = await registry.activate_configured_current(
        expected_current_key_id="current",
        expected_config_epoch=_coordination_epoch(0, "a"),
        new_config_epoch=_coordination_epoch(1, "b"),
        now=NOW,
    )

    assert advanced.config_epoch == _coordination_epoch(1, "b")


@pytest.mark.asyncio
async def test_production_epoch_cannot_downgrade_to_unordered_token() -> None:
    store = FakeRegistryStore(
        records=(_current("current", activated_at=NOW - timedelta(days=1)),),
        config_epoch=_coordination_epoch(1, "a"),
    )
    registry = StandaloneHtmlKeyRegistry(
        store=store,
        keyring=_keyring("current"),
    )

    with pytest.raises(DigestKeyRotationError, match="coordination generation"):
        await registry.activate_configured_current(
            expected_current_key_id="current",
            expected_config_epoch=_coordination_epoch(1, "a"),
            new_config_epoch="unordered-epoch",
            now=NOW,
        )

    assert [name for name, _ in store.calls] == ["load"]


@pytest.mark.asyncio
@pytest.mark.parametrize("future_record", ["current", "desired"])
async def test_rotation_rejects_now_before_existing_key_timeline(
    future_record: str,
) -> None:
    current = _current(
        "old",
        activated_at=NOW + timedelta(days=1) if future_record == "current" else NOW - timedelta(days=10),
    )
    records = [current]
    if future_record == "desired":
        records.append(
            _retiring(
                "new",
                activated_at=NOW - timedelta(days=10),
                retired_at=NOW + timedelta(days=1),
            )
        )
    store = FakeRegistryStore(records=tuple(records), config_epoch="epoch-1")

    with pytest.raises(DigestKeyRotationError):
        await StandaloneHtmlKeyRegistry(
            store=store,
            keyring=_keyring("old", "new", current="new"),
        ).activate_configured_current(
            expected_current_key_id="old",
            expected_config_epoch="epoch-1",
            new_config_epoch="epoch-2",
            now=NOW,
        )
    assert [name for name, _ in store.calls] == ["load"]


@pytest.mark.asyncio
async def test_rotation_rejects_store_result_that_drops_the_prior_current_key() -> None:
    class CorruptingStore(FakeRegistryStore):
        async def compare_and_swap_current_key(self, **kwargs) -> CurrentKeyCasResult | None:
            result = await super().compare_and_swap_current_key(**kwargs)
            assert result is not None
            return CurrentKeyCasResult(
                state=DigestKeyRegistryState(
                    records=(_current("new", activated_at=NOW),),
                    config_epoch="epoch-2",
                ),
                applied_here=result.applied_here,
            )

    registry = StandaloneHtmlKeyRegistry(
        store=CorruptingStore(records=(_current("old"),), config_epoch="epoch-1"),
        keyring=_keyring("old", "new", current="new"),
    )

    with pytest.raises(DigestKeyRotationError):
        await registry.activate_configured_current(
            expected_current_key_id="old",
            expected_config_epoch="epoch-1",
            new_config_epoch="epoch-2",
            now=NOW,
        )


@pytest.mark.asyncio
async def test_rotation_rejects_store_result_with_nonexact_transition_timestamp() -> None:
    activated_at = NOW - timedelta(days=10)

    class CorruptingStore(FakeRegistryStore):
        async def compare_and_swap_current_key(self, **kwargs) -> CurrentKeyCasResult | None:
            result = await super().compare_and_swap_current_key(**kwargs)
            assert result is not None
            return CurrentKeyCasResult(
                state=DigestKeyRegistryState(
                    records=(
                        _current("new", activated_at=NOW),
                        _retiring(
                            "old",
                            activated_at=activated_at,
                            retired_at=NOW - timedelta(seconds=1),
                        ),
                    ),
                    config_epoch="epoch-2",
                ),
                applied_here=result.applied_here,
            )

    with pytest.raises(DigestKeyRotationError):
        await StandaloneHtmlKeyRegistry(
            store=CorruptingStore(
                records=(_current("old", activated_at=activated_at),),
                config_epoch="epoch-1",
            ),
            keyring=_keyring("old", "new", current="new"),
        ).activate_configured_current(
            expected_current_key_id="old",
            expected_config_epoch="epoch-1",
            new_config_epoch="epoch-2",
            now=NOW,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("desired_preexists", [False, True])
async def test_rotation_rejects_nonexact_new_or_reactivated_key_timestamp(
    desired_preexists: bool,
) -> None:
    old_activated_at = NOW - timedelta(days=10)
    desired_prior = _retiring(
        "new",
        activated_at=NOW - timedelta(days=30),
        retired_at=NOW - timedelta(days=1),
    )

    class CorruptingStore(FakeRegistryStore):
        async def compare_and_swap_current_key(self, **kwargs) -> CurrentKeyCasResult | None:
            result = await super().compare_and_swap_current_key(**kwargs)
            assert result is not None
            records = tuple(
                _current("new", activated_at=NOW - timedelta(seconds=1)) if record.key_id == "new" else record
                for record in result.state.records
            )
            return CurrentKeyCasResult(
                state=DigestKeyRegistryState(
                    records=records,
                    config_epoch=result.state.config_epoch,
                ),
                applied_here=result.applied_here,
            )

    records = [_current("old", activated_at=old_activated_at)]
    if desired_preexists:
        records.append(desired_prior)

    with pytest.raises(DigestKeyRotationError):
        await StandaloneHtmlKeyRegistry(
            store=CorruptingStore(records=tuple(records), config_epoch="epoch-1"),
            keyring=_keyring("old", "new", current="new"),
        ).activate_configured_current(
            expected_current_key_id="old",
            expected_config_epoch="epoch-1",
            new_config_epoch="epoch-2",
            now=NOW,
        )


@pytest.mark.asyncio
async def test_rotation_rejects_store_result_that_mutates_unrelated_retiring_key() -> None:
    ancient_retired_at = NOW - timedelta(days=2)

    class CorruptingStore(FakeRegistryStore):
        async def compare_and_swap_current_key(self, **kwargs) -> CurrentKeyCasResult | None:
            result = await super().compare_and_swap_current_key(**kwargs)
            assert result is not None
            records = tuple(
                (
                    _retiring(
                        "ancient",
                        retired_at=ancient_retired_at - timedelta(days=1),
                    )
                    if record.key_id == "ancient"
                    else record
                )
                for record in result.state.records
            )
            return CurrentKeyCasResult(
                state=DigestKeyRegistryState(
                    records=records,
                    config_epoch=result.state.config_epoch,
                ),
                applied_here=result.applied_here,
            )

    with pytest.raises(DigestKeyRotationError):
        await StandaloneHtmlKeyRegistry(
            store=CorruptingStore(
                records=(
                    _current("old"),
                    _retiring("ancient", retired_at=ancient_retired_at),
                ),
                config_epoch="epoch-1",
            ),
            keyring=_keyring("old", "new", "ancient", current="new"),
        ).activate_configured_current(
            expected_current_key_id="old",
            expected_config_epoch="epoch-1",
            new_config_epoch="epoch-2",
            now=NOW,
        )


@pytest.mark.asyncio
async def test_rotation_requires_every_existing_secret_and_respects_four_key_cap() -> None:
    missing_store = FakeRegistryStore(
        records=(
            _current("old"),
            _retiring("lost", retired_at=NOW - timedelta(days=1)),
        ),
        config_epoch="epoch-1",
    )
    missing_registry = StandaloneHtmlKeyRegistry(
        store=missing_store,
        keyring=_keyring("old", "new", current="new"),
    )
    with pytest.raises(DigestKeyUnavailableError):
        await missing_registry.activate_configured_current(
            expected_current_key_id="old",
            expected_config_epoch="epoch-1",
            new_config_epoch="epoch-2",
            now=NOW,
        )
    assert [name for name, _ in missing_store.calls] == ["load"]

    full_store = FakeRegistryStore(
        records=(
            _current("a"),
            _retiring("b", retired_at=NOW - timedelta(days=1)),
            _retiring("c", retired_at=NOW - timedelta(days=2)),
            _retiring("d", retired_at=NOW - timedelta(days=3)),
        ),
        config_epoch="epoch-1",
    )
    full_registry = StandaloneHtmlKeyRegistry(
        store=full_store,
        keyring=_keyring("a", "b", "c", "d", current="a"),
    )
    snapshot = await full_registry.snapshot()
    assert len(snapshot.records) == 4
    with pytest.raises(DigestKeyConfigError):
        _keyring("a", "b", "c", "d", "e", current="e")


@pytest.mark.asyncio
async def test_registry_rejects_corrupt_shared_current_state() -> None:
    store = FakeRegistryStore(
        records=(_current("a"), _current("b")),
        config_epoch="epoch-1",
    )

    with pytest.raises(DigestKeyRegistryError):
        await StandaloneHtmlKeyRegistry(
            store=store,
            keyring=_keyring("a", "b", current="a"),
        ).snapshot()


@pytest.mark.asyncio
async def test_retirement_rejects_current_key_floor_and_missing_or_incomplete_sweep() -> None:
    retired_at = NOW - timedelta(days=32)
    store = FakeRegistryStore(
        records=(_current("current"), _retiring("old", retired_at=retired_at)),
        config_epoch="epoch-2",
    )
    registry = StandaloneHtmlKeyRegistry(
        store=store,
        keyring=_keyring("current", "old"),
    )

    with pytest.raises(DigestKeyRetirementError):
        await registry.remove_retired_key(key_id="current", now=NOW)
    with pytest.raises(DigestKeyRetirementError):
        await registry.remove_retired_key(key_id="old", now=NOW - timedelta(microseconds=1))
    with pytest.raises(DigestKeyRetirementError):
        await registry.remove_retired_key(key_id="old", now=NOW)

    store.proofs["old"] = _proof(
        "old",
        sweep_started_at=NOW,
        complete=False,
    )
    with pytest.raises(DigestKeyRetirementError):
        await registry.remove_retired_key(key_id="old", now=NOW + timedelta(minutes=1))

    store.proofs["old"] = _proof(
        "old",
        config_epoch="stale-epoch",
        sweep_started_at=NOW,
    )
    with pytest.raises(DigestKeyRetirementError):
        await registry.remove_retired_key(key_id="old", now=NOW + timedelta(minutes=1))

    store.proofs["old"] = _proof(
        "old",
        sweep_started_at=NOW,
        sweep_completed_at=NOW + timedelta(minutes=2),
    )
    with pytest.raises(DigestKeyRetirementError):
        await registry.remove_retired_key(key_id="old", now=NOW + timedelta(minutes=1))

    store.proofs["old"] = _proof(
        "old",
        sweep_started_at=NOW,
        unexpired_reference_count=1,
    )
    with pytest.raises(DigestKeyRetirementError):
        await registry.remove_retired_key(key_id="old", now=NOW + timedelta(minutes=1))


@pytest.mark.asyncio
async def test_retirement_requires_sweep_to_start_at_floor_under_same_epoch() -> None:
    retired_at = NOW - timedelta(days=32)
    store = FakeRegistryStore(
        records=(_current("current"), _retiring("old", retired_at=retired_at)),
        config_epoch="epoch-2",
    )
    registry = StandaloneHtmlKeyRegistry(
        store=store,
        keyring=_keyring("current", "old"),
    )
    store.proofs["old"] = _proof(
        "old",
        sweep_started_at=NOW - timedelta(microseconds=1),
    )
    with pytest.raises(DigestKeyRetirementError):
        await registry.remove_retired_key(key_id="old", now=NOW + timedelta(minutes=1))


@pytest.mark.asyncio
async def test_retirement_rejects_wrong_type_sweep_proof_as_domain_error() -> None:
    retired_at = NOW - timedelta(days=32)
    store = FakeRegistryStore(
        records=(_current("current"), _retiring("old", retired_at=retired_at)),
        config_epoch="epoch-2",
    )
    store.proofs["old"] = object()  # type: ignore[assignment]

    with pytest.raises(DigestKeyRetirementError):
        await StandaloneHtmlKeyRegistry(
            store=store,
            keyring=_keyring("current", "old"),
        ).remove_retired_key(key_id="old", now=NOW)


@pytest.mark.asyncio
async def test_retirement_accepts_exact_32_day_boundary_and_complete_fenced_sweep() -> None:
    retired_at = NOW - timedelta(days=32)
    store = FakeRegistryStore(
        records=(_current("current"), _retiring("old", retired_at=retired_at)),
        config_epoch="epoch-2",
    )
    store.proofs["old"] = _proof(
        "old",
        sweep_started_at=NOW,
        sweep_completed_at=NOW,
    )
    registry = StandaloneHtmlKeyRegistry(
        store=store,
        keyring=_keyring("current"),
    )

    snapshot = await registry.remove_retired_key(
        key_id="old",
        now=NOW,
    )

    assert snapshot.current_key_id == "current"
    assert snapshot.records == (_current("current"),)
    assert snapshot.availability.available


@pytest.mark.asyncio
async def test_retirement_rejects_store_result_that_drops_unrelated_current_key() -> None:
    class CorruptingStore(FakeRegistryStore):
        async def compare_and_swap_remove_key(self, **kwargs) -> DigestKeyRegistryState | None:
            await super().compare_and_swap_remove_key(**kwargs)
            return DigestKeyRegistryState(records=(), config_epoch="epoch-2")

    retired_at = NOW - timedelta(days=32)
    store = CorruptingStore(
        records=(_current("current"), _retiring("old", retired_at=retired_at)),
        config_epoch="epoch-2",
    )
    store.proofs["old"] = _proof(
        "old",
        sweep_started_at=NOW,
        sweep_completed_at=NOW,
    )

    with pytest.raises(DigestKeyRetirementError):
        await StandaloneHtmlKeyRegistry(
            store=store,
            keyring=_keyring("current"),
        ).remove_retired_key(key_id="old", now=NOW)


def _walk_values(value: Any):
    if dataclasses.is_dataclass(value):
        yield from _walk_values(dataclasses.asdict(value))
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _walk_values(key)
            yield from _walk_values(item)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            yield from _walk_values(item)
    else:
        yield value


@pytest.mark.asyncio
async def test_repository_calls_never_receive_secret_base64_or_hmac_values() -> None:
    old_secret = _encoded_secret(1)
    new_secret = _encoded_secret(2)
    keyring = StandaloneHtmlHmacKeyring.from_json(
        keys_json=f'{{"old":"{old_secret}","new":"{new_secret}"}}',
        current_key_id="new",
    )
    store = FakeRegistryStore(records=(_current("old"),), config_epoch="epoch-1")
    registry = StandaloneHtmlKeyRegistry(store=store, keyring=keyring)

    snapshot = await registry.activate_configured_current(
        expected_current_key_id="old",
        expected_config_epoch="epoch-1",
        new_config_epoch="epoch-2",
        now=NOW,
    )
    keyed_hmac = keyring.digest_current(
        snapshot=snapshot,
        domain=HmacDomain.CLIENT_REQUEST,
        payload=b"private request",
    )
    values = tuple(_walk_values(store.calls))
    rendered = repr(store.calls)

    assert not any(isinstance(value, bytes) for value in values)
    assert old_secret not in rendered
    assert new_secret not in rendered
    assert (bytes([1]) * 32).hex() not in rendered
    assert (bytes([2]) * 32).hex() not in rendered
    assert keyed_hmac.digest_hex not in rendered

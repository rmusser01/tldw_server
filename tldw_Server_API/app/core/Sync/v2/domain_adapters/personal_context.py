from __future__ import annotations

"""Strict Sync v2 adapters for canonical Personal Context whole objects."""

import base64
import binascii
import hashlib
import hmac
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError
from tldw_profile_core import (
    ProfileManifest,
    ProfileProposal,
    ProfileRecord,
    ProfileScope,
    ProposalState,
    RecordState,
    SyncMode,
)
from tldw_profile_core.canonical import canonical_json_bytes

from tldw_Server_API.app.core.exceptions import (
    PersonalContextSyncDeviceOnlyRecord,
    PersonalContextSyncIdentityConflict,
)
from tldw_Server_API.app.core.Personalization.personal_context_crypto import (
    EncryptedEnvelope,
    EnvelopeAuthenticationError,
    EnvelopeCipher,
)

from ..adapters import (
    AdapterAccepted,
    AdapterConflict,
    AdapterRejected,
    SyncAdapterContext,
    SyncAdapterOutcome,
)
from ..models import SyncDataset, SyncDomain, SyncEnvelope, SyncEnvelopeCreate

PERSONAL_CONTEXT_MAX_OBJECT_BYTES = 16_384
_PERSONAL_CONTEXT_MODELS = {
    "personal_context.manifest": ProfileManifest,
    "personal_context.scope": ProfileScope,
    "personal_context.record": ProfileRecord,
    "personal_context.proposal": ProfileProposal,
}
_OBJECT_TYPE = {
    "personal_context.manifest": "manifest",
    "personal_context.scope": "scope",
    "personal_context.record": "record",
    "personal_context.proposal": "proposal",
    "personal_context.purge": "purge",
}

IntegrityKeyResolver = Callable[[SyncDataset, str], bytes]
EncryptionKeyResolver = Callable[[SyncDataset], tuple[bytes, int]]


@dataclass(slots=True)
class PersonalContextDomainAdapter:
    """Validate one exact Personal Context domain before Sync persistence."""

    domain: SyncDomain
    integrity_key_resolver: IntegrityKeyResolver
    encryption_key_resolver: EncryptionKeyResolver | None = None
    supported_adapter_versions: set[int] = field(default_factory=lambda: {1})

    def __post_init__(self) -> None:
        if self.domain not in {*_PERSONAL_CONTEXT_MODELS, "personal_context.purge"}:
            raise ValueError("Unsupported Personal Context Sync domain")

    def evaluate_envelope(
        self,
        envelope: SyncEnvelopeCreate,
        *,
        dataset: SyncDataset,
        context: SyncAdapterContext | None = None,
    ) -> SyncAdapterOutcome:
        """Return a stable, content-free validation or lineage outcome."""

        if envelope.domain != self.domain:
            return self._reject(envelope, "personal_context_domain_mismatch")
        if (
            self.domain == "personal_context.manifest"
            and envelope.routing_metadata.get("personal_context_authority", {}).get("role") == "client_ingress"
        ):
            return self._reject(envelope, "personal_context_manifest_client_forbidden")
        if envelope.adapter_version != 1 or envelope.schema_version != 1:
            return self._reject(envelope, "personal_context_schema_unsupported")
        if dataset.encryption_policy != "server_trusted_v1":
            return self._reject(envelope, "personal_context_encryption_policy_invalid")
        payload = envelope.payload or envelope.payload_clear
        try:
            canonical = canonical_json_bytes(payload)
        except (TypeError, ValueError):
            return self._reject(envelope, "personal_context_payload_invalid")
        if len(canonical) > PERSONAL_CONTEXT_MAX_OBJECT_BYTES:
            return self._reject(envelope, "personal_context_payload_too_large")

        try:
            dataset_state = _dataset_state(dataset)
        except (TypeError, ValueError):
            return self._reject(envelope, "personal_context_dataset_invalid")
        key_id = envelope.routing_metadata.get("integrity_key_id")
        if (
            not isinstance(key_id, str)
            or not hmac.compare_digest(key_id, dataset_state["integrity_key_id"])
        ):
            return self._reject(envelope, "personal_context_integrity_key_invalid")
        try:
            key = self.integrity_key_resolver(dataset, key_id)
        except (KeyError, RuntimeError, TypeError, ValueError):
            return self._reject(envelope, "personal_context_integrity_unavailable")
        if len(key) != 32:
            return self._reject(envelope, "personal_context_integrity_unavailable")
        expected_tag = "hmac-sha256-v1:" + hmac.new(
            key, canonical, hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(envelope.payload_hash, expected_tag):
            return self._reject(envelope, "personal_context_integrity_invalid")

        try:
            value = _parse_payload(self.domain, payload)
            if (
                self.domain == "personal_context.purge"
                and value["purge_generation"] != dataset_state["purge_generation"] + 1
            ):
                return self._reject(envelope, "personal_context_purge_generation_invalid")
            _validate_envelope_identity(
                envelope,
                value,
                profile_id=dataset_state["profile_id"],
                purge_generation=dataset_state["purge_generation"],
            )
        except PersonalContextSyncIdentityConflict:
            return AdapterConflict(
                client_envelope_id=envelope.client_envelope_id,
                domain=self.domain,
                entity_id=envelope.object_id,
                conflict_type="personal_context_identity_conflict",
            )
        except PersonalContextSyncDeviceOnlyRecord:
            return self._reject(envelope, "personal_context_device_only_forbidden")
        except (ValidationError, ValueError, TypeError):
            return self._reject(envelope, "personal_context_payload_invalid")

        head = _current_object_head(envelope, context)
        if head is not None and not _references_head(envelope, head, value):
            if self.domain == "personal_context.purge":
                # Push verifies the full immutable fingerprint before evaluation
                # and insertion; an exact stored ingress may retry its failed apply.
                if (
                    envelope.routing_metadata.get("personal_context_authority", {}).get("role") == "client_ingress"
                    and envelope.client_envelope_id == head.client_envelope_id
                    and envelope.device_id == head.device_id
                    and envelope.payload_hash == head.payload_hash
                ):
                    return AdapterAccepted(client_envelope_id=envelope.client_envelope_id)
                return AdapterRejected(
                    client_envelope_id=envelope.client_envelope_id,
                    error_code="personal_context_purge_reconfirmation_required",
                    message="Refresh Personal Context and explicitly reconfirm delete-everywhere.",
                    retryable=False,
                )
            return AdapterConflict(
                client_envelope_id=envelope.client_envelope_id,
                domain=self.domain,
                entity_id=envelope.object_id,
                conflict_type="personal_context_base_conflict",
            )
        return AdapterAccepted(client_envelope_id=envelope.client_envelope_id)

    @property
    def storage_encryption_ready(self) -> bool:
        """Return whether clear profile payloads can be protected at rest."""

        return self.encryption_key_resolver is not None

    def key_custody_ready(self, dataset: SyncDataset) -> bool:
        """Return whether this enrolled dataset's canonical keys are usable."""

        if self.encryption_key_resolver is None:
            return False
        try:
            state = _dataset_state(dataset)
            integrity_key = self.integrity_key_resolver(
                dataset,
                state["integrity_key_id"],
            )
            encryption_key, key_version = self.encryption_key_resolver(dataset)
        except (KeyError, RuntimeError, TypeError, ValueError):
            return False
        return (
            len(integrity_key) == 32
            and len(encryption_key) == 32
            and type(key_version) is int
            and key_version >= 1
        )

    def protect_for_storage(
        self,
        envelope: SyncEnvelopeCreate,
        *,
        dataset: SyncDataset,
    ) -> SyncEnvelopeCreate:
        """Encrypt clear Personal Context content before Sync DB persistence."""

        if envelope.domain != self.domain or self.encryption_key_resolver is None:
            raise RuntimeError("Personal Context storage encryption is unavailable")
        key, key_version = self.encryption_key_resolver(dataset)
        if len(key) != 32 or type(key_version) is not int or key_version < 1:
            raise RuntimeError("Personal Context storage encryption is unavailable")
        payload = envelope.payload or envelope.payload_clear
        protected = EnvelopeCipher(key, key_version=key_version).encrypt(
            canonical_json_bytes(payload),
            _storage_aad(envelope),
        )
        return _replace_envelope(
            envelope,
            payload={},
            payload_clear={},
            payload_ciphertext=_b64(protected.ciphertext),
            encryption_metadata={
                **envelope.encryption_metadata,
                "personal_context_at_rest": {
                    "version": 1,
                    "algorithm": protected.algorithm,
                    "nonce": _b64(protected.nonce),
                    "wrapped_dek": _b64(protected.wrapped_dek),
                    "wrapped_dek_nonce": _b64(protected.wrapped_dek_nonce),
                    "key_version": protected.key_version,
                },
            },
        )

    def restore_from_storage(
        self,
        envelope: SyncEnvelope | SyncEnvelopeCreate,
        *,
        dataset: SyncDataset,
    ) -> SyncEnvelope | SyncEnvelopeCreate:
        """Authenticate and restore one Personal Context Sync DB envelope."""

        if envelope.domain != self.domain or self.encryption_key_resolver is None:
            raise RuntimeError("Personal Context storage encryption is unavailable")
        metadata = envelope.encryption_metadata.get("personal_context_at_rest")
        if not isinstance(metadata, Mapping) or not envelope.payload_ciphertext:
            raise RuntimeError("Personal Context stored envelope is invalid")
        key, key_version = self.encryption_key_resolver(dataset)
        try:
            protected = EncryptedEnvelope(
                algorithm=str(metadata["algorithm"]),
                nonce=_unb64(metadata["nonce"]),
                wrapped_dek=_unb64(metadata["wrapped_dek"]),
                wrapped_dek_nonce=_unb64(metadata["wrapped_dek_nonce"]),
                ciphertext=_unb64(envelope.payload_ciphertext),
                key_version=int(metadata["key_version"]),
            )
            if protected.key_version != key_version:
                raise EnvelopeAuthenticationError("envelope authentication failed")
            plaintext = EnvelopeCipher(key, key_version=key_version).decrypt(
                protected,
                _storage_aad(envelope),
            )
            payload = json.loads(plaintext.decode("utf-8"))
        except (
            binascii.Error,
            EnvelopeAuthenticationError,
            KeyError,
            TypeError,
            UnicodeDecodeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            raise RuntimeError("Personal Context stored envelope is invalid") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Personal Context stored envelope is invalid")
        clear_metadata = dict(envelope.encryption_metadata)
        clear_metadata.pop("personal_context_at_rest", None)
        return _replace_envelope(
            envelope,
            payload=payload,
            payload_clear=payload,
            payload_ciphertext=None,
            encryption_metadata=clear_metadata,
        )

    @staticmethod
    def _reject(
        envelope: SyncEnvelopeCreate,
        error_code: str,
    ) -> AdapterRejected:
        return AdapterRejected(
            client_envelope_id=envelope.client_envelope_id,
            error_code=error_code,
            message="Personal Context envelope validation failed",
        )


def _dataset_state(dataset: SyncDataset) -> dict[str, Any]:
    state = dataset.metadata.get("personal_context")
    if not isinstance(state, Mapping):
        raise ValueError("Personal Context dataset metadata is unavailable")
    profile_id = state.get("profile_id")
    key_id = state.get("integrity_key_id")
    purge_generation = state.get("purge_generation", 0)
    if (
        not isinstance(profile_id, str)
        or not profile_id
        or not isinstance(key_id, str)
        or not key_id
        or type(purge_generation) is not int
        or purge_generation < 0
    ):
        raise ValueError("Personal Context dataset metadata is invalid")
    return {
        "profile_id": profile_id,
        "integrity_key_id": key_id,
        "purge_generation": purge_generation,
    }


def _parse_payload(domain: str, payload: Mapping[str, Any]) -> Any:
    """Parse one canonical Personal Context wire payload for its Sync domain."""

    model = _PERSONAL_CONTEXT_MODELS.get(domain)
    if model is not None:
        return model.model_validate(payload)
    if set(payload) != {"schema_version", "profile_id", "purge_generation"}:
        raise ValueError("Personal Context purge barrier is invalid")
    if payload.get("schema_version") != 1:
        raise ValueError("Personal Context purge schema is unsupported")
    profile_id = payload.get("profile_id")
    generation = payload.get("purge_generation")
    if not isinstance(profile_id, str) or not profile_id:
        raise ValueError("Personal Context purge profile is invalid")
    if type(generation) is not int or generation < 1:
        raise ValueError("Personal Context purge generation is invalid")
    return dict(payload)


def _validate_envelope_identity(
    envelope: SyncEnvelopeCreate,
    value: Any,
    *,
    profile_id: str,
    purge_generation: int,
) -> None:
    """Require envelope routing and version identity to match its payload."""

    if envelope.domain == "personal_context.manifest":
        if envelope.operation != "upsert":
            raise ValueError("Manifest operation is invalid")
        object_id, parent_id, value_profile_id = value.profile_id, None, value.profile_id
        entity_version = value.current_version_id
        if value.purge_generation != purge_generation:
            raise PersonalContextSyncIdentityConflict
    elif envelope.domain == "personal_context.scope":
        if envelope.operation != "upsert":
            raise ValueError("Scope operation is invalid")
        object_id, parent_id, value_profile_id = (
            value.scope_id,
            value.profile_id,
            value.profile_id,
        )
        entity_version = value.version_id
    elif envelope.domain == "personal_context.record":
        if value.controls.sync_mode is SyncMode.DEVICE_ONLY:
            raise PersonalContextSyncDeviceOnlyRecord
        if envelope.operation == "tombstone":
            if value.state is not RecordState.DELETED or value.payload is not None:
                raise ValueError("Record tombstone must be content-free")
        elif envelope.operation != "upsert" or value.state is RecordState.DELETED:
            raise ValueError("Record operation is invalid")
        object_id, parent_id, value_profile_id = (
            value.record_id,
            value.scope_id,
            value.profile_id,
        )
        entity_version = value.version_id
    elif envelope.domain == "personal_context.proposal":
        if envelope.operation != "upsert":
            raise ValueError("Proposal operation is invalid")
        if (
            value.state is ProposalState.PENDING
            and value.proposed_record is not None
            and value.proposed_record.controls.sync_mode is SyncMode.DEVICE_ONLY
        ):
            raise PersonalContextSyncDeviceOnlyRecord
        object_id, parent_id, value_profile_id = (
            value.proposal_id,
            value.scope_id,
            value.profile_id,
        )
        entity_version = "sync-proposal-sha256:" + hashlib.sha256(
            canonical_json_bytes(value.model_dump(mode="json"))
        ).hexdigest()
    else:
        if envelope.operation != "tombstone":
            raise ValueError("Purge operation is invalid")
        object_id, parent_id, value_profile_id = (
            value["profile_id"],
            None,
            value["profile_id"],
        )
        entity_version = value["purge_generation"]
    if (
        object_id != envelope.object_id
        or parent_id != envelope.parent_id
        or value_profile_id != profile_id
        or not _same_wire_version(envelope.entity_version, entity_version)
    ):
        raise PersonalContextSyncIdentityConflict


def _same_wire_version(left: Any, right: Any) -> bool:
    """Return whether two version values have identical wire type and value."""

    return type(left) is type(right) and left == right


def _current_object_head(
    envelope: SyncEnvelopeCreate,
    context: SyncAdapterContext | None,
) -> SyncEnvelope | None:
    """Return the newest prior envelope for the same canonical object."""

    if context is None:
        return None
    matching = [
        item
        for item in context.prior_envelopes
        if item.domain == envelope.domain and item.object_id == envelope.object_id
    ]
    return matching[-1] if matching else None


def _references_head(envelope: SyncEnvelopeCreate, head: Any, value: Any) -> bool:
    """Return whether an incoming value extends the current canonical head."""

    if (
        envelope.base_object_hash is not None
        and envelope.base_object_hash != head.payload_hash
    ):
        return False
    head_payload = head.payload or head.payload_clear
    try:
        if envelope.domain == "personal_context.record":
            current = ProfileRecord.model_validate(head_payload)
            return value.parent_version_id == current.version_id
        if envelope.domain == "personal_context.manifest":
            current = ProfileManifest.model_validate(head_payload)
            return (
                value.revision == current.revision + 1
                and value.purge_generation == current.purge_generation
                and value.created_at == current.created_at
            )
        if envelope.domain == "personal_context.scope":
            current = ProfileScope.model_validate(head_payload)
            return envelope.base_version == current.version_id
        if envelope.domain == "personal_context.proposal":
            current = ProfileProposal.model_validate(head_payload)
            if (
                current.state is not ProposalState.PENDING
                or value.state is ProposalState.PENDING
            ):
                return False
            expected = ProfileProposal.model_validate(
                {
                    **current.model_dump(mode="python"),
                    "state": value.state,
                    "proposed_record": None,
                    "confidence": None,
                }
            )
            return expected == value
        current_generation = head_payload.get("purge_generation")
        return (
            type(current_generation) is int
            and value["purge_generation"] == current_generation + 1
        )
    except (AttributeError, KeyError, TypeError, ValidationError, ValueError):
        return False


def _storage_aad(envelope: SyncEnvelope | SyncEnvelopeCreate) -> bytes:
    """Build stable associated data for encrypted transport persistence."""

    return canonical_json_bytes(
        {
            "adapter_version": envelope.adapter_version,
            "client_envelope_id": envelope.client_envelope_id,
            "dataset_id": envelope.dataset_id,
            "domain": envelope.domain,
            "object_id": envelope.object_id,
            "schema_version": envelope.schema_version,
        }
    )


def _replace_envelope(envelope: Any, **changes: Any) -> Any:
    """Return a dataclass envelope with the requested immutable replacements."""

    from dataclasses import replace

    return replace(envelope, **changes)


def _b64(value: bytes) -> str:
    """Encode bytes as canonical ASCII base64 text."""

    return base64.b64encode(value).decode("ascii")


def _unb64(value: Any) -> bytes:
    """Decode strictly validated ASCII base64 text."""

    if not isinstance(value, str):
        raise TypeError("encoded value must be text")
    return base64.b64decode(value.encode("ascii"), validate=True)


__all__ = ["PERSONAL_CONTEXT_MAX_OBJECT_BYTES", "PersonalContextDomainAdapter"]

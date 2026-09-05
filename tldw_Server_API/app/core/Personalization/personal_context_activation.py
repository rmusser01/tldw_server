"""Replayable activation orchestration across independently committed stores."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import nullcontext
from typing import Any

from tldw_Server_API.app.core.exceptions import (
    PersonalContextActivationInputError,
    PersonalContextActivationStaleError,
)
from tldw_Server_API.app.core.Personalization.personal_context_publication import (
    PersonalContextPublicationRelayStore,
    PublicationRelayLease,
)
from tldw_Server_API.app.core.Personalization.personal_context_repository import (
    PersonalContextRepository,
)
from tldw_Server_API.app.core.Personalization.personal_context_repository_models import (
    PreparedPersonalContextActivation,
)
from tldw_Server_API.app.core.Sync.v2.personal_context_ongoing_contract import (
    PersonalContextExchangeProof,
)


class PersonalContextActivationService:
    """Keep the shared lease and verified source coverage in one canonical owner."""

    def __init__(self, repository: PersonalContextRepository) -> None:
        """Use the authenticated user's existing canonical repository."""
        self.repository = repository
        self.publications = PersonalContextPublicationRelayStore(repository.database)

    def prepare(
        self,
        profile_id: str,
        *,
        device_id: str,
        fresh: bool = False,
    ) -> PreparedPersonalContextActivation:
        """Prepare or replay the exact device baseline under the shared relay lease."""
        with self.publications.profile_lease(profile_id) as lease:
            return self.repository.prepare_activation(
                profile_id,
                device_id=device_id,
                fresh=fresh,
                lease=lease,
            )

    def install(
        self,
        activation_id: str,
        baseline_digest: str,
        *,
        install: Callable[[PreparedPersonalContextActivation], Mapping[str, Any]],
        verify: Callable[[PreparedPersonalContextActivation, Mapping[str, Any]], bool],
        lease: PublicationRelayLease | None = None,
    ) -> PreparedPersonalContextActivation:
        """Commit coverage only after the caller commits and verifies the exact Sync receipt.

        The Sync owner acquires the profile lease before its dataset guard and
        supplies that lease here, preserving lease -> Sync -> canonical ordering.
        Its install callback commits Sync independently while the canonical
        generation remains fenced. A later failure leaves preparation replayable.

        Args:
            activation_id: Identifier of the prepared activation.
            baseline_digest: Expected digest of its exact baseline.
            install: Callback that commits the baseline and returns its Sync receipt.
            verify: Callback that verifies the receipt against the preparation.
            lease: Already owned profile lease, or None to acquire one here.

        Returns:
            The activation with verified installation coverage recorded.

        Raises:
            PersonalContextActivationInputError: The receipt is malformed.
            PersonalContextActivationStaleError: Verification or fencing fails.
        """
        prepared = self.repository.load_activation(activation_id)
        with nullcontext(lease) if lease is not None else self.publications.profile_lease(prepared.profile_id) as lease:
            with self.repository.activation_install_guard(
                activation_id,
                baseline_digest,
                lease=lease,
            ) as current:
                receipt = install(current)
                if not verify(current, receipt):
                    raise PersonalContextActivationStaleError("personal_context_activation_required")
                receipt_id = receipt.get("receipt_id")
                home_server_cursor = receipt.get("home_server_cursor")
                if not isinstance(receipt_id, str) or type(home_server_cursor) is not int:
                    raise PersonalContextActivationInputError("personal_context_activation_required")
            return self.repository.complete_activation_install(
                activation_id,
                baseline_digest,
                receipt_id,
                home_server_cursor=home_server_cursor,
                lease=lease,
            )

    def acknowledge(
        self,
        activation_id: str,
        baseline_digest: str,
        device_id: str,
        sync_ack_receipt_id: str,
        *,
        local_receipt_id: str,
        dataset_id: str,
    ) -> PreparedPersonalContextActivation:
        """Record a device acknowledgement after its exact Sync receipt is verified."""
        return self.repository.confirm_activation_device(
            activation_id,
            baseline_digest,
            device_id,
            sync_ack_receipt_id,
            local_receipt_id=local_receipt_id,
            dataset_id=dataset_id,
        )

    def validate_exchange(
        self,
        *,
        profile_id: str,
        device_id: str,
        dataset_id: str,
        activation_epoch: str,
        continuity_token: str,
    ) -> PersonalContextExchangeProof:
        """Validate current generation and durable device acknowledgement."""
        return self.repository.validate_activation_exchange(
            profile_id=profile_id,
            device_id=device_id,
            dataset_id=dataset_id,
            activation_epoch=activation_epoch,
            continuity_token=continuity_token,
        )

"""Strict study-pack generation and persistence orchestration."""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from loguru import logger

from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import DEFAULT_LLM_PROVIDER
from tldw_Server_API.app.api.v1.schemas.study_packs import StudyPackCreateJobRequest
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError, ConflictError
from tldw_Server_API.app.core.Flashcards.scheduler_sm2 import get_default_scheduler_settings
from tldw_Server_API.app.core.LLM_Calls.adapter_utils import (
    ensure_app_config,
    normalize_provider,
    resolve_provider_model,
)
from tldw_Server_API.app.core.StudyPacks.provenance import (
    FlashcardProvenanceStore,
    normalize_citations_for_persistence,
)
from tldw_Server_API.app.core.StudyPacks.source_resolver import StudySourceResolver
from tldw_Server_API.app.core.StudyPacks.types import (
    StudyCitationDraft,
    StudyPackCardDraft,
    StudyPackCreationResult,
    StudyPackGenerationResult,
    StudySourceBundle,
    StudySourceBundleItem,
)
from tldw_Server_API.app.core.Workflows.adapters._common import extract_openai_content

try:
    from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call_async
except ImportError:  # pragma: no cover - covered by existing adapter tests elsewhere
    async def perform_chat_api_call_async(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise ImportError("chat_service_unavailable")

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.media_db.media_database_impl import MediaDatabase


_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)
_MAX_REPAIR_ATTEMPTS = 1
_MAX_DECK_NAME_ATTEMPTS = 1000


class StudyPackGenerationError(ValueError):
    """Base error for study-pack generation failures."""


class StudyPackMalformedResponseError(StudyPackGenerationError):
    """Raised when the model output cannot be parsed into the required JSON contract."""


class StudyPackValidationError(StudyPackGenerationError):
    """Raised when parsed cards fail the strict study-pack validation rules."""


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _bundle_item_key(source_type: Any, source_id: Any) -> tuple[str, str]:
    return _clean_text(source_type).lower(), _clean_text(source_id)


def _serialize_bundle(bundle: StudySourceBundle) -> dict[str, Any]:
    return {
        "items": [
            {
                "source_type": item.source_type,
                "source_id": item.source_id,
                "label": item.label,
                "evidence_text": item.evidence_text,
                "locator": dict(item.locator),
            }
            for item in bundle.items
        ]
    }


def _serialize_citations(citations: Sequence[StudyCitationDraft]) -> list[dict[str, Any]]:
    return [
        {
            "source_type": citation.source_type,
            "source_id": citation.source_id,
            "citation_text": citation.citation_text,
            "locator": dict(citation.locator),
        }
        for citation in citations
    ]


def _coerce_locator_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {
            str(key): item
            for key, item in value.items()
            if item not in (None, "", [], {})
        }
    locator_text = _clean_text(value)
    if not locator_text:
        return {}
    try:
        parsed = json.loads(locator_text)
    except json.JSONDecodeError:
        return {"locator": locator_text}
    if isinstance(parsed, Mapping):
        return {
            str(key): item
            for key, item in parsed.items()
            if item not in (None, "", [], {})
        }
    return {"locator": locator_text}


def _config_default_llm_provider(app_config: Mapping[str, Any] | None) -> str | None:
    if not isinstance(app_config, Mapping):
        return None

    for section in ("llm_api_settings", "API"):
        section_data = app_config.get(section)
        if not isinstance(section_data, Mapping):
            continue
        default_api = section_data.get("default_api")
        if isinstance(default_api, str) and default_api.strip():
            return default_api.strip()
    return None


def _resolve_generation_provider_and_model(
    provider: str | None,
    model: str | None,
) -> tuple[str, str | None, dict[str, Any]]:
    app_config = ensure_app_config()

    resolved_provider = normalize_provider(provider)
    if not resolved_provider:
        resolved_provider = normalize_provider(_config_default_llm_provider(app_config))
    if not resolved_provider:
        resolved_provider = normalize_provider(os.getenv("DEFAULT_LLM_PROVIDER"))
    if not resolved_provider:
        resolved_provider = normalize_provider(DEFAULT_LLM_PROVIDER or "openai") or "openai"

    resolved_model = _clean_text(model) or None
    if resolved_model is None:
        env_model_key = f"DEFAULT_MODEL_{resolved_provider.replace('.', '_').replace('-', '_').upper()}"
        resolved_model = _clean_text(os.getenv(env_model_key)) or None
    if resolved_model is None:
        resolved_model = resolve_provider_model(resolved_provider, app_config)

    return resolved_provider, resolved_model, app_config


class StudyPackGenerationService:
    """Generates validated flashcards from a resolved study source bundle and persists them atomically."""

    def __init__(
        self,
        *,
        note_db: CharactersRAGDB,
        media_db: MediaDatabase | Any,
        provider: str | None,
        model: str | None,
    ) -> None:
        self.note_db = note_db
        self.media_db = media_db
        self.provider, self.model, self.app_config = _resolve_generation_provider_and_model(provider, model)
        self.source_resolver = StudySourceResolver(db=note_db, media_db=media_db)
        self.provenance_store = FlashcardProvenanceStore(note_db)

    async def create_from_request(
        self,
        request: StudyPackCreateJobRequest,
        *,
        regenerate_from_pack_id: int | None = None,
        expected_regenerate_version: int | None = None,
    ) -> StudyPackCreationResult:
        bundle = await asyncio.to_thread(self.source_resolver.resolve, request.source_items)
        generated = await self.generate_validated_cards(bundle, request)
        return self._persist_generated_cards(
            bundle=bundle,
            request=request,
            generated=generated,
            regenerate_from_pack_id=regenerate_from_pack_id,
            expected_regenerate_version=expected_regenerate_version,
        )

    async def generate_validated_cards(
        self,
        bundle: StudySourceBundle,
        request: StudyPackCreateJobRequest,
    ) -> StudyPackGenerationResult:
        system_prompt = self._build_generation_system_prompt(request)
        user_prompt = self._build_generation_user_prompt(bundle=bundle, request=request)
        raw_response = await self._call_generation_model(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        try:
            cards = self._parse_and_validate_response(raw_response, bundle=bundle)
            return StudyPackGenerationResult(cards=cards, raw_response=raw_response, repair_attempted=False)
        except (StudyPackMalformedResponseError, StudyPackValidationError) as exc:
            logger.warning(
                "Study-pack generation output failed strict validation, attempting a single repair pass: {}",
                exc,
            )

        repaired_response = await self._repair_generation_response(
            broken_response=raw_response,
            bundle=bundle,
            request=request,
        )
        cards = self._parse_and_validate_response(repaired_response, bundle=bundle)
        return StudyPackGenerationResult(cards=cards, raw_response=repaired_response, repair_attempted=True)

    def _persist_generated_cards(
        self,
        *,
        bundle: StudySourceBundle,
        request: StudyPackCreateJobRequest,
        generated: StudyPackGenerationResult,
        regenerate_from_pack_id: int | None,
        expected_regenerate_version: int | None,
    ) -> StudyPackCreationResult:
        expected_supersede_version = self._expected_supersede_version(
            regenerate_from_pack_id,
            explicit_version=expected_regenerate_version,
        )
        cleaned_base_name = _clean_text(request.title) or "Study Pack"
        card_template_payloads = [
            {
                "front": card.front,
                "back": card.back,
                "notes": card.notes,
                "extra": card.extra,
                "model_type": card.model_type,
                "tags_json": json.dumps(card.tags) if card.tags else None,
            }
            for card in generated.cards
        ]

        for _ in range(_MAX_DECK_NAME_ATTEMPTS):
            deck_name = self._next_destination_deck_name(request.title)
            try:
                with self.note_db.transaction():
                    deck_id = self.note_db.add_deck(
                        deck_name,
                        workspace_id=request.workspace_id,
                        scheduler_type="sm2_plus",
                        scheduler_settings=get_default_scheduler_settings(),
                    )
                    card_payloads = [
                        {
                            **card_payload,
                            "deck_id": deck_id,
                        }
                        for card_payload in card_template_payloads
                    ]
                    card_uuids = self.note_db.add_flashcards_bulk(card_payloads)
                    pack_id = self.note_db.create_study_pack(
                        title=request.title,
                        workspace_id=request.workspace_id,
                        deck_id=deck_id,
                        source_bundle_json=_serialize_bundle(bundle),
                        generation_options_json={
                            "deck_mode": request.deck_mode,
                            "provider": self.provider,
                            "model": self.model,
                            "repair_attempted": generated.repair_attempted,
                        },
                    )
                    inserted_memberships = self.note_db.add_study_pack_cards(pack_id, card_uuids)
                    if inserted_memberships != len(card_uuids):
                        raise CharactersRAGDBError(
                            f"Expected {len(card_uuids)} study-pack membership rows, inserted {inserted_memberships}"
                        )

                    for card_uuid, card in zip(card_uuids, generated.cards, strict=True):
                        self.provenance_store.persist_flashcard_citations(
                            card_uuid,
                            _serialize_citations(card.citations),
                        )

                    if regenerate_from_pack_id is not None:
                        self.note_db.supersede_study_pack(
                            regenerate_from_pack_id,
                            superseded_by_pack_id=pack_id,
                            expected_version=expected_supersede_version,
                        )

                return StudyPackCreationResult(
                    pack_id=pack_id,
                    deck_id=deck_id,
                    deck_name=deck_name,
                    card_uuids=card_uuids,
                    cards=generated.cards,
                    repair_attempted=generated.repair_attempted,
                    regenerated_from_pack_id=regenerate_from_pack_id,
                )
            except ConflictError as exc:
                if exc.entity != "decks":
                    raise
                continue

        raise CharactersRAGDBError(f"Could not allocate a unique deck name for '{cleaned_base_name}'")  # noqa: TRY003

    def _expected_supersede_version(
        self,
        pack_id: int | None,
        *,
        explicit_version: int | None = None,
    ) -> int | None:
        if explicit_version is not None:
            return int(explicit_version)
        if pack_id is None:
            return None
        original_pack = self.note_db.get_study_pack(pack_id)
        if not original_pack:
            return None
        version = original_pack.get("version")
        return int(version) if version is not None else None

    def _next_destination_deck_name(self, base_name: str) -> str:
        cleaned_base_name = _clean_text(base_name) or "Study Pack"
        existing_names = {
            _clean_text(deck.get("name"))
            for deck in self.note_db.list_decks(limit=10_000, include_deleted=True, include_workspace_items=True)
            if _clean_text(deck.get("name"))
        }
        if cleaned_base_name not in existing_names:
            return cleaned_base_name

        for suffix in range(2, _MAX_DECK_NAME_ATTEMPTS + 1):
            candidate_name = f"{cleaned_base_name} ({suffix})"
            if candidate_name not in existing_names:
                return candidate_name

        raise CharactersRAGDBError(f"Could not allocate a unique deck name for '{cleaned_base_name}'")  # noqa: TRY003

    async def _call_generation_model(self, *, system_prompt: str, user_prompt: str) -> str:
        response = await perform_chat_api_call_async(
            messages=[{"role": "user", "content": user_prompt}],
            api_provider=self.provider,
            model=self.model,
            app_config=self.app_config,
            system_message=system_prompt,
            max_tokens=4000,
            temperature=0.2,
        )
        return _clean_text(extract_openai_content(response))

    async def _repair_generation_response(
        self,
        *,
        broken_response: str,
        bundle: StudySourceBundle,
        request: StudyPackCreateJobRequest,
    ) -> str:
        repair_system_prompt = (
            "You repair malformed study-pack JSON. Return valid JSON only. "
            "Do not add commentary or markdown fences."
        )
        repair_user_prompt = (
            "Rewrite the malformed response so it exactly matches this contract:\n"
            '{"cards":[{"front":"string","back":"string","model_type":"basic|basic_reverse|cloze",'
            '"notes":"optional string","extra":"optional string","tags":["optional"],'
            '"citations":[{"source_type":"note|media|message","source_id":"bundle id","citation_text":"string",'
            '"locator":{"optional":"canonical bundle locator"}}]}]}\n\n'
            f"Allowed source bundle:\n{json.dumps(_serialize_bundle(bundle), sort_keys=True)}\n\n"
            f"Study-pack title: {request.title}\n\n"
            f"Malformed response:\n{broken_response}"
        )
        repaired_response = broken_response
        for _ in range(_MAX_REPAIR_ATTEMPTS):
            repaired_response = await self._call_generation_model(
                system_prompt=repair_system_prompt,
                user_prompt=repair_user_prompt,
            )
        return repaired_response

    def _build_generation_system_prompt(self, request: StudyPackCreateJobRequest) -> str:
        return (
            "Generate a study pack from the provided source bundle.\n"
            "Return valid JSON only with top-level object shape "
            '{"cards":[{"front":"string","back":"string","model_type":"basic|basic_reverse|cloze",'
            '"notes":"optional string","extra":"optional string","tags":["optional"],'
            '"citations":[{"source_type":"note|media|message","source_id":"bundle id","citation_text":"string",'
            '"locator":{"optional":"canonical bundle locator"}}]}]}\n'
            "Every card must include at least one citation.\n"
            "Every citation must reference a source_id from the allowed bundle exactly.\n"
            "Do not cite sources outside the bundle. Do not emit markdown fences or prose.\n"
            f"Requested deck title: {request.title}"
        )

    def _build_generation_user_prompt(
        self,
        *,
        bundle: StudySourceBundle,
        request: StudyPackCreateJobRequest,
    ) -> str:
        return (
            f"Study-pack request:\n{json.dumps({'title': request.title, 'deck_mode': request.deck_mode}, sort_keys=True)}\n\n"
            f"Allowed source bundle:\n{json.dumps(_serialize_bundle(bundle), sort_keys=True)}\n\n"
            "Create concise flashcards grounded only in the supplied sources."
        )

    def _parse_and_validate_response(
        self,
        raw_response: str,
        *,
        bundle: StudySourceBundle,
    ) -> list[StudyPackCardDraft]:
        payload = self._extract_json_payload(raw_response)
        if isinstance(payload, list):
            raw_cards = payload
        elif isinstance(payload, Mapping):
            raw_cards = payload.get("cards")
        else:
            raise StudyPackMalformedResponseError("Study-pack response must decode to a JSON object or list")

        if not isinstance(raw_cards, list):
            raise StudyPackMalformedResponseError("Study-pack response must include a cards array")
        if not raw_cards:
            raise StudyPackValidationError("Study-pack generation returned no cards")

        bundle_lookup = {
            _bundle_item_key(item.source_type, item.source_id): item
            for item in bundle.items
        }
        validated_cards = [
            self._validate_card_payload(raw_card, bundle_lookup=bundle_lookup)
            for raw_card in raw_cards
        ]
        if not validated_cards:
            raise StudyPackValidationError("Study-pack generation returned no valid cards")
        return validated_cards

    def _validate_card_payload(
        self,
        raw_card: Any,
        *,
        bundle_lookup: Mapping[tuple[str, str], StudySourceBundleItem],
    ) -> StudyPackCardDraft:
        if not isinstance(raw_card, Mapping):
            raise StudyPackMalformedResponseError("Each study-pack card must be a JSON object")

        raw_citations = raw_card.get("citations")
        if not isinstance(raw_citations, list):
            raise StudyPackMalformedResponseError("Each study-pack card must include a citations array")
        if not raw_citations:
            raise StudyPackValidationError("Each study-pack card must include at least one citation")

        prepared_citations: list[dict[str, Any]] = []
        for raw_citation in raw_citations:
            prepared_citations.append(
                self._validate_citation_payload(raw_citation, bundle_lookup=bundle_lookup)
            )

        normalized_citations = normalize_citations_for_persistence(prepared_citations)
        if not normalized_citations:
            raise StudyPackValidationError("Each study-pack card must include at least one citation")

        citations = [
            StudyCitationDraft(
                source_type=citation["source_type"],
                source_id=citation["source_id"],
                citation_text=citation["citation_text"],
                locator=_coerce_locator_mapping(citation.get("locator")),
            )
            for citation in normalized_citations
        ]
        tags = raw_card.get("tags")
        if tags is None:
            normalized_tags: list[str] = []
        elif isinstance(tags, Sequence) and not isinstance(tags, (str, bytes, bytearray)):
            normalized_tags = [_clean_text(tag) for tag in tags if _clean_text(tag)]
        else:
            raise StudyPackMalformedResponseError("tags must be an array when provided")

        return StudyPackCardDraft(
            front=raw_card.get("front"),
            back=raw_card.get("back"),
            citations=citations,
            model_type=_clean_text(raw_card.get("model_type")).lower() or "basic",
            notes=_clean_text(raw_card.get("notes")) or None,
            extra=_clean_text(raw_card.get("extra")) or None,
            tags=normalized_tags,
        )

    def _validate_citation_payload(
        self,
        raw_citation: Any,
        *,
        bundle_lookup: Mapping[tuple[str, str], StudySourceBundleItem],
    ) -> dict[str, Any]:
        if not isinstance(raw_citation, Mapping):
            raise StudyPackMalformedResponseError("Each study-pack citation must be a JSON object")

        citation_key = _bundle_item_key(
            raw_citation.get("source_type"),
            raw_citation.get("source_id"),
        )
        bundle_item = bundle_lookup.get(citation_key)
        if bundle_item is None:
            raise StudyPackValidationError("Study-pack citations must reference the allowed source bundle")

        citation_text = _clean_text(raw_citation.get("citation_text"))
        if not citation_text:
            raise StudyPackValidationError("Study-pack citations must include citation_text")

        locator = self._merge_citation_locator(bundle_item, raw_citation.get("locator"))
        return {
            "source_type": bundle_item.source_type,
            "source_id": bundle_item.source_id,
            "citation_text": citation_text,
            "locator": locator,
        }

    def _merge_citation_locator(
        self,
        bundle_item: StudySourceBundleItem,
        raw_locator: Any,
    ) -> dict[str, Any]:
        locator = dict(bundle_item.locator)
        if isinstance(raw_locator, Mapping):
            for key, value in raw_locator.items():
                if value not in (None, "", [], {}):
                    key_text = str(key)
                    existing_value = locator.get(key_text)
                    if key_text not in locator or existing_value in (None, "", [], {}):
                        locator[key_text] = value
        else:
            raw_locator_text = _clean_text(raw_locator)
            if raw_locator_text:
                locator["locator"] = raw_locator_text
        if not locator:
            raise StudyPackValidationError(
                f"Study-pack citation for {bundle_item.source_type}:{bundle_item.source_id} is missing a canonical locator"
            )
        return locator

    def _extract_json_payload(self, raw_response: str) -> Any:
        cleaned_response = _clean_text(raw_response)
        if not cleaned_response:
            raise StudyPackMalformedResponseError("Study-pack generation returned an empty response")

        candidate_text = _JSON_FENCE_RE.sub("", cleaned_response).strip()
        direct_error: json.JSONDecodeError | None = None
        try:
            return json.loads(candidate_text)
        except json.JSONDecodeError as exc:
            direct_error = exc

        decoder = json.JSONDecoder()
        for index, char in enumerate(candidate_text):
            if char not in "[{":
                continue
            try:
                payload, _ = decoder.raw_decode(candidate_text[index:])
                return payload
            except json.JSONDecodeError:
                continue
        raise StudyPackMalformedResponseError("Study-pack generation did not return valid JSON") from direct_error


async def create_study_pack_from_request(
    *,
    note_db: CharactersRAGDB,
    media_db: MediaDatabase | Any,
    request: StudyPackCreateJobRequest,
    regenerate_from_pack_id: int | None = None,
    expected_regenerate_version: int | None = None,
    provider: str | None,
    model: str | None,
) -> StudyPackCreationResult:
    """Resolve sources, generate validated cards, and persist a study pack atomically."""
    service = StudyPackGenerationService(
        note_db=note_db,
        media_db=media_db,
        provider=provider,
        model=model,
    )
    return await service.create_from_request(
        request,
        regenerate_from_pack_id=regenerate_from_pack_id,
        expected_regenerate_version=expected_regenerate_version,
    )


__all__ = [
    "StudyPackGenerationError",
    "StudyPackMalformedResponseError",
    "StudyPackValidationError",
    "StudyPackGenerationService",
    "create_study_pack_from_request",
]

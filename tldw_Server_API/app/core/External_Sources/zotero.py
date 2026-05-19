from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import time
import uuid
from functools import partial
from typing import Any
from urllib.parse import parse_qs, quote, urlencode

from tldw_Server_API.app.core.http_client import afetch
from tldw_Server_API.app.core.Utils.metadata_utils import normalize_safe_metadata

from .connector_base import BaseConnector
from .reference_manager_adapter import ReferenceManagerAdapter
from .reference_manager_types import (
    NormalizedReferenceCollection,
    NormalizedReferenceItem,
    ReferenceAttachmentCandidate,
)

_ZOTERO_API_BASE = "https://api.zotero.org"
_ZOTERO_API_VERSION = "3"
_ZOTERO_OAUTH_REQUEST_URL = "https://www.zotero.org/oauth/request"
_ZOTERO_OAUTH_ACCESS_URL = "https://www.zotero.org/oauth/access"
_ZOTERO_OAUTH_AUTHORIZE_URL = "https://www.zotero.org/oauth/authorize"
_PDF_MIME_TYPES = {"application/pdf", "application/x-pdf"}
_IMPORTABLE_ATTACHMENT_MODES = {"imported_file", "imported_url", "linked_file"}
_OAUTH1_SHA1 = partial(hashlib.sha1, usedforsecurity=False)


class ZoteroConnector(BaseConnector, ReferenceManagerAdapter):
    name = "zotero"

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        redirect_base: str | None = None,
    ):
        super().__init__(
            client_id=client_id or os.getenv("CONNECTOR_ZOTERO_CLIENT_ID"),
            client_secret=client_secret or os.getenv("CONNECTOR_ZOTERO_CLIENT_SECRET"),
            redirect_base=redirect_base or os.getenv("CONNECTOR_REDIRECT_BASE_URL"),
        )

    def authorize_url(
        self,
        state: str | None = None,
        scopes: list[str] | None = None,
        redirect_path: str = "/api/v1/connectors/providers/zotero/callback",
    ) -> str:
        if not self.client_id or not self.client_secret:
            raise ValueError("Zotero authorization requires configured client credentials.")
        oauth_token: str | None = None
        params = {
            "name": "tldw_server",
            "library_access": "1",
            "notes_access": "0",
            "write_access": "0",
        }
        for scope in scopes or []:
            if "=" not in scope:
                continue
            key, value = scope.split("=", 1)
            if key == "oauth_token":
                oauth_token = value
            elif key:
                params[key] = value
        if not oauth_token:
            raise ValueError("Zotero authorization requires an oauth_token from request_temporary_credential().")
        params["oauth_token"] = oauth_token
        if state:
            params["state"] = state
        return f"{_ZOTERO_OAUTH_AUTHORIZE_URL}?{urlencode(params)}"

    async def exchange_code(self, code: str, redirect_uri: str) -> dict[str, Any]:
        if not self.client_id or not self.client_secret:
            raise ValueError("Zotero OAuth exchange requires configured client credentials.")
        token_payload = self._parse_exchange_code_payload(code)
        oauth_token = token_payload.get("oauth_token")
        oauth_token_secret = token_payload.get("oauth_token_secret")
        oauth_verifier = token_payload.get("oauth_verifier")
        if not oauth_token or not oauth_token_secret:
            raise ValueError("Zotero OAuth exchange requires oauth_token and oauth_token_secret.")
        headers = {
            "Authorization": self._build_oauth_header(
                method="POST",
                url=_ZOTERO_OAUTH_ACCESS_URL,
                oauth_token=oauth_token,
                oauth_token_secret=oauth_token_secret,
                oauth_verifier=oauth_verifier,
            ),
            "Content-Type": "application/x-www-form-urlencoded",
        }
        resp = await afetch(
            method="POST",
            url=_ZOTERO_OAUTH_ACCESS_URL,
            headers=headers,
            timeout=30,
        )
        try:
            resp.raise_for_status()
            payload = self._parse_form_encoded_response(resp.content)
        finally:
            close = getattr(resp, "aclose", None)
            if callable(close):
                await close()
        token_type = "_".join((self.name, "api", "key"))
        return dict(
            access_token=payload.get("oauth_token_secret"),
            refresh_token=None,
            token_type=token_type,
            provider=self.name,
            display_name="Zotero Account",
            email=None,
            provider_user_id=payload.get("userID"),
            username=payload.get("username"),
            request_token=payload.get("oauth_token"),
            redirect_uri=redirect_uri,
        )

    @staticmethod
    def _access_token_from_account(account: dict[str, Any]) -> str | None:
        return (account.get("tokens") or {}).get("access_token") or account.get("access_token")

    @staticmethod
    def _provider_user_id_from_account(account: dict[str, Any]) -> str | None:
        for key in ("provider_user_id", "userID", "user_id"):
            value = str(account.get(key) or "").strip()
            if value:
                return value
        return None

    def _library_path(self, account: dict[str, Any]) -> str:
        provider_user_id = self._provider_user_id_from_account(account)
        if not provider_user_id:
            raise ValueError("Zotero account is missing provider_user_id/userID.")
        return f"users/{provider_user_id}"

    @staticmethod
    def _percent_encode(value: Any) -> str:
        return quote(str(value), safe="~-._")

    @classmethod
    def _normalize_oauth_parameters(cls, params: dict[str, Any]) -> str:
        pairs = []
        for key in sorted(params):
            pairs.append(f"{cls._percent_encode(key)}={cls._percent_encode(params[key])}")
        return "&".join(pairs)

    @classmethod
    def _build_oauth_signature(
        cls,
        *,
        method: str,
        url: str,
        params: dict[str, Any],
        consumer_secret: str,
        token_secret: str = "",
    ) -> str:
        base_string = "&".join(
            [
                method.upper(),
                cls._percent_encode(url),
                cls._percent_encode(cls._normalize_oauth_parameters(params)),
            ]
        )
        signing_key = f"{cls._percent_encode(consumer_secret)}&{cls._percent_encode(token_secret)}"
        # Zotero's OAuth 1.0a handshake mandates HMAC-SHA1. Mark it non-security
        # critical so static analyzers do not treat it as a general-purpose hash.
        digest = hmac.new(signing_key.encode("utf-8"), base_string.encode("utf-8"), _OAUTH1_SHA1).digest()
        return base64.b64encode(digest).decode("ascii")

    def _build_oauth_header(
        self,
        *,
        method: str,
        url: str,
        oauth_token: str | None = None,
        oauth_token_secret: str = "",
        oauth_callback: str | None = None,
        oauth_verifier: str | None = None,
    ) -> str:
        if not self.client_id or not self.client_secret:
            raise ValueError("Zotero OAuth operations require configured client credentials.")
        oauth_params: dict[str, Any] = {
            "oauth_consumer_key": self.client_id,
            "oauth_nonce": uuid.uuid4().hex,
            "oauth_signature_method": "HMAC-SHA1",
            "oauth_timestamp": str(int(time.time())),
            "oauth_version": "1.0",
        }
        if oauth_token:
            oauth_params["oauth_token"] = oauth_token
        if oauth_callback:
            oauth_params["oauth_callback"] = oauth_callback
        if oauth_verifier:
            oauth_params["oauth_verifier"] = oauth_verifier
        oauth_params["oauth_signature"] = self._build_oauth_signature(
            method=method,
            url=url,
            params=oauth_params,
            consumer_secret=self.client_secret,
            token_secret=oauth_token_secret,
        )
        header_parts = [
            f'{self._percent_encode(key)}="{self._percent_encode(value)}"'
            for key, value in sorted(oauth_params.items())
        ]
        return "OAuth " + ", ".join(header_parts)

    @staticmethod
    def _parse_exchange_code_payload(code: str) -> dict[str, str]:
        raw_code = str(code or "").strip()
        if not raw_code:
            return {}
        if raw_code.startswith("{"):
            data = json.loads(raw_code)
            return {str(key): str(value) for key, value in data.items() if value not in (None, "")}
        parsed = parse_qs(raw_code, keep_blank_values=False)
        return {key: values[0] for key, values in parsed.items() if values}

    @staticmethod
    def _parse_form_encoded_response(content: bytes | bytearray | str | None) -> dict[str, str]:
        if isinstance(content, (bytes, bytearray)):
            raw_content = content.decode("utf-8")
        else:
            raw_content = str(content or "")
        parsed = parse_qs(raw_content, keep_blank_values=False)
        return {key: values[0] for key, values in parsed.items() if values}

    async def request_temporary_credential(self, callback_url: str) -> dict[str, str]:
        headers = {
            "Authorization": self._build_oauth_header(
                method="POST",
                url=_ZOTERO_OAUTH_REQUEST_URL,
                oauth_callback=callback_url,
            ),
            "Content-Type": "application/x-www-form-urlencoded",
        }
        resp = await afetch(
            method="POST",
            url=_ZOTERO_OAUTH_REQUEST_URL,
            headers=headers,
            timeout=30,
        )
        try:
            resp.raise_for_status()
            return self._parse_form_encoded_response(resp.content)
        finally:
            close = getattr(resp, "aclose", None)
            if callable(close):
                await close()

    def _build_headers(self, account: dict[str, Any]) -> dict[str, str]:
        api_key = self._access_token_from_account(account)
        if not api_key:
            raise ValueError("Zotero account is missing an API key.")
        return {
            "Zotero-API-Version": _ZOTERO_API_VERSION,
            "Zotero-API-Key": api_key,
        }

    @staticmethod
    def _build_paging_params(cursor: str | None, page_size: int) -> tuple[dict[str, Any], int]:
        start = 0
        if cursor:
            start = max(0, int(cursor))
        limit = max(1, min(int(page_size), 100))
        return {"format": "json", "limit": limit, "start": start}, start

    @staticmethod
    def _next_cursor(headers: dict[str, str], start: int, count: int) -> str | None:
        total_results = str(headers.get("Total-Results") or "").strip()
        if not total_results:
            return None
        try:
            total = int(total_results)
        except (TypeError, ValueError):
            return None
        next_start = start + count
        if next_start >= total:
            return None
        return str(next_start)

    @staticmethod
    def _creator_name(creator: dict[str, Any]) -> str | None:
        if not isinstance(creator, dict):
            return None
        first_name = str(creator.get("firstName") or "").strip()
        last_name = str(creator.get("lastName") or "").strip()
        if first_name and last_name:
            return f"{first_name} {last_name}"
        if last_name:
            return last_name
        name = str(creator.get("name") or "").strip()
        return name or None

    def _format_authors(self, raw_item: dict[str, Any]) -> str | None:
        data = raw_item.get("data") or {}
        creators = data.get("creators") or []
        author_names: list[str] = []
        for creator in creators:
            creator_type = str((creator or {}).get("creatorType") or "").strip().lower()
            if creator_type and creator_type != "author":
                continue
            name = self._creator_name(creator)
            if name:
                author_names.append(name)
        if author_names:
            return ", ".join(author_names)
        creator_summary = str((raw_item.get("meta") or {}).get("creatorSummary") or "").strip()
        return creator_summary or None

    @staticmethod
    def _parse_year(date_value: str | None) -> str | None:
        if not date_value:
            return None
        match = re.search(r"\b(1[5-9]\d{2}|20\d{2}|21\d{2})\b", str(date_value).strip())
        if match:
            return match.group(1)
        return None

    @staticmethod
    def _source_url(raw_item: dict[str, Any]) -> str | None:
        links = raw_item.get("links") or {}
        alternate = links.get("alternate") or {}
        href = str(alternate.get("href") or "").strip()
        return href or None

    @staticmethod
    def _attachment_source_url(raw_attachment: dict[str, Any]) -> str | None:
        links = raw_attachment.get("links") or {}
        alternate = links.get("alternate") or {}
        href = str(alternate.get("href") or "").strip()
        if href:
            return href
        data = raw_attachment.get("data") or {}
        url = str(data.get("url") or "").strip()
        return url or None

    @staticmethod
    def _is_attachment(raw_item: dict[str, Any]) -> bool:
        data = raw_item.get("data") or {}
        return str(data.get("itemType") or "").strip().lower() == "attachment"

    def _normalize_attachment_candidate(
        self,
        raw_attachment: dict[str, Any],
        *,
        provider_item_key: str,
    ) -> ReferenceAttachmentCandidate | None:
        data = raw_attachment.get("data") or {}
        link_mode = str(data.get("linkMode") or "").strip().lower()
        mime_type = str(data.get("contentType") or "").strip().lower() or None
        if link_mode not in _IMPORTABLE_ATTACHMENT_MODES:
            return None
        if mime_type not in _PDF_MIME_TYPES:
            return None
        attachment_key = str(data.get("key") or raw_attachment.get("key") or "").strip()
        if not attachment_key:
            return None
        size_bytes = data.get("filesize")
        try:
            size_value = int(size_bytes) if size_bytes is not None else None
        except (TypeError, ValueError):
            size_value = None
        return ReferenceAttachmentCandidate(
            provider=self.name,
            provider_item_key=provider_item_key,
            attachment_key=attachment_key,
            title=str(data.get("title") or "").strip() or None,
            source_url=self._attachment_source_url(raw_attachment),
            mime_type=mime_type,
            size_bytes=size_value,
            metadata={
                "link_mode": link_mode,
                "filename": str(data.get("filename") or "").strip() or None,
            },
        )

    async def normalize_reference_item(
        self,
        raw_item: dict[str, Any],
        raw_attachments: list[dict[str, Any]],
        *,
        collection_key: str | None = None,
        collection_name: str | None = None,
        provider_library_id: str | None = None,
    ) -> NormalizedReferenceItem:
        data = raw_item.get("data") or {}
        provider_item_key = str(data.get("key") or raw_item.get("key") or "").strip()
        if not provider_item_key:
            raise ValueError("Zotero item is missing a key.")
        try:
            parsed_metadata = normalize_safe_metadata({"doi": data.get("DOI") or data.get("doi")})
        except ValueError:
            parsed_metadata = {}
        doi = parsed_metadata.get("doi")
        item_collection_key = collection_key
        if not item_collection_key:
            collections = data.get("collections") or []
            if collections:
                item_collection_key = str(collections[0] or "").strip() or None
        date_value = str(data.get("date") or "").strip() or None
        attachments: list[ReferenceAttachmentCandidate] = []
        for raw_attachment in raw_attachments:
            candidate = self._normalize_attachment_candidate(
                raw_attachment,
                provider_item_key=provider_item_key,
            )
            if candidate is not None:
                attachments.append(candidate)
        return NormalizedReferenceItem(
            provider=self.name,
            provider_item_key=provider_item_key,
            provider_library_id=(
                provider_library_id
                or self._provider_user_id_from_account(raw_item.get("account") or {})
                or None
            ),
            collection_key=item_collection_key,
            collection_name=collection_name,
            doi=doi,
            title=str(data.get("title") or "").strip() or None,
            authors=self._format_authors(raw_item),
            publication_date=date_value,
            year=self._parse_year(date_value),
            journal=str(data.get("publicationTitle") or "").strip() or None,
            abstract=str(data.get("abstractNote") or "").strip() or None,
            source_url=self._source_url(raw_item),
            attachments=attachments,
            metadata={
                "provider_version": raw_item.get("version"),
                "item_type": data.get("itemType"),
            },
        )

    async def list_collections(
        self,
        account: dict[str, Any],
        *,
        cursor: str | None = None,
        page_size: int = 100,
    ) -> tuple[list[NormalizedReferenceCollection], str | None]:
        params, start = self._build_paging_params(cursor, page_size)
        resp = await afetch(
            method="GET",
            url=f"{_ZOTERO_API_BASE}/{self._library_path(account)}/collections",
            headers=self._build_headers(account),
            params=params,
            timeout=30,
        )
        try:
            resp.raise_for_status()
            payload = resp.json() or []
        finally:
            close = getattr(resp, "aclose", None)
            if callable(close):
                await close()

        collections: list[NormalizedReferenceCollection] = []
        for raw_collection in payload:
            if not isinstance(raw_collection, dict):
                continue
            data = raw_collection.get("data") or {}
            collection_key = str(data.get("key") or raw_collection.get("key") or "").strip()
            if not collection_key:
                continue
            collections.append(
                NormalizedReferenceCollection(
                    provider=self.name,
                    provider_library_id=self._provider_user_id_from_account(account),
                    collection_key=collection_key,
                    collection_name=str(data.get("name") or "").strip() or None,
                    source_url=self._source_url(raw_collection),
                    metadata={
                        "provider_version": raw_collection.get("version"),
                        "parent_collection": data.get("parentCollection") or None,
                    },
                )
            )
        next_cursor = self._next_cursor(resp.headers, start, len(payload))
        return collections, next_cursor

    async def list_collection_items(
        self,
        account: dict[str, Any],
        collection_key: str,
        *,
        collection_name: str | None = None,
        cursor: str | None = None,
        page_size: int = 100,
    ) -> tuple[list[NormalizedReferenceItem], str | None]:
        params, start = self._build_paging_params(cursor, page_size)
        resp = await afetch(
            method="GET",
            url=f"{_ZOTERO_API_BASE}/{self._library_path(account)}/collections/{collection_key}/items/top",
            headers=self._build_headers(account),
            params=params,
            timeout=30,
        )
        try:
            resp.raise_for_status()
            payload = resp.json() or []
        finally:
            close = getattr(resp, "aclose", None)
            if callable(close):
                await close()

        items: list[NormalizedReferenceItem] = []
        for raw_item in payload:
            if not isinstance(raw_item, dict) or self._is_attachment(raw_item):
                continue
            normalized_item = await self.normalize_reference_item(
                raw_item,
                [],
                collection_key=collection_key,
                collection_name=collection_name,
                provider_library_id=self._provider_user_id_from_account(account),
            )
            items.append(normalized_item)
        next_cursor = self._next_cursor(resp.headers, start, len(payload))
        return items, next_cursor

    async def list_item_attachments(
        self,
        account: dict[str, Any],
        provider_item_key: str,
    ) -> list[ReferenceAttachmentCandidate]:
        resp = await afetch(
            method="GET",
            url=f"{_ZOTERO_API_BASE}/{self._library_path(account)}/items/{provider_item_key}/children",
            headers=self._build_headers(account),
            params={"format": "json"},
            timeout=30,
        )
        try:
            resp.raise_for_status()
            payload = resp.json() or []
        finally:
            close = getattr(resp, "aclose", None)
            if callable(close):
                await close()

        attachments: list[ReferenceAttachmentCandidate] = []
        for raw_attachment in payload:
            if not isinstance(raw_attachment, dict):
                continue
            candidate = self._normalize_attachment_candidate(
                raw_attachment,
                provider_item_key=provider_item_key,
            )
            if candidate is not None:
                attachments.append(candidate)
        return attachments

    async def download_file(
        self,
        account: dict[str, Any],
        file_id: str,
        **kwargs: Any,
    ) -> bytes:
        _ = kwargs
        resp = await afetch(
            method="GET",
            url=f"{_ZOTERO_API_BASE}/{self._library_path(account)}/items/{quote(file_id, safe='')}/file",
            headers=self._build_headers(account),
            timeout=60,
        )
        try:
            resp.raise_for_status()
            return resp.content
        finally:
            close = getattr(resp, "aclose", None)
            if callable(close):
                await close()

    async def resolve_attachment_download(
        self,
        account: dict[str, Any],
        attachment: ReferenceAttachmentCandidate,
    ) -> bytes:
        return await self.download_file(
            account,
            attachment.attachment_key,
            mime_type=attachment.mime_type,
        )

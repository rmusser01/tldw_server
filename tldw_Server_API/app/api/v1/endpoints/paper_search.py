# paper_search.py
# Provider-specific paper search endpoints under /api/v1/paper-search/*

import asyncio
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.encoders import jsonable_encoder
from loguru import logger

try:
    import httpx
except ImportError:
    httpx = None

try:
    import aiohttp
except ImportError:
    aiohttp = None

import contextlib

from tldw_Server_API.app.api.v1.API_Deps.backpressure import guard_backpressure_and_quota
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.endpoints._pagination_utils import build_page_pagination_meta
from tldw_Server_API.app.api.v1.schemas.paper_search_schemas import (
    BioRxivFunderPaper,
    BioRxivFunderSearchRequestForm,
    BioRxivFunderSearchResponse,
    BioRxivPaper,
    BioRxivPublishedRecord,
    BioRxivPublisherSearchRequestForm,
    BioRxivPubSearchRequestForm,
    BioRxivPubsSearchRequestForm,
    BioRxivPubsSearchResponse,
    BioRxivSearchRequestForm,
    BioRxivSearchResponse,
    BioRxivSummaryRequestForm,
    BioRxivSummaryResponse,
    BioRxivUsageRequestForm,
    BioRxivUsageResponse,
    OSFSearchRequestForm,
    PMCOAIdentifyResponse,
    PMCOAIHeader,
    PMCOAIIdentifiersResponse,
    PMCOAIIdentifyResponse,
    PMCOAIListResponse,
    PMCOAIListSetsResponse,
    PMCOAIRecord,
    PMCOAQueryResponse,
    PMCOARecord,
    PubMedPaper,
    PubMedSearchRequestForm,
    PubMedSearchResponse,
)
from tldw_Server_API.app.api.v1.schemas.research_schemas import (
    ArxivPaper,
    ArxivSearchRequestForm,
    ArxivSearchResponse,
    SemanticScholarPaper,
    SemanticScholarSearchRequestForm,
    SemanticScholarSearchResponse,
)
from tldw_Server_API.app.core.DB_Management.media_db.api import (
    MediaDbSession,
    MediaWriterLike,
    get_media_repository,
)
from tldw_Server_API.app.core.http_client import (
    RetryPolicy as _RetryPolicy,
)
from tldw_Server_API.app.core.http_client import (
    adownload as _http_adownload,
)
from tldw_Server_API.app.core.http_client import (
    afetch as _http_afetch,
)
from tldw_Server_API.app.core.http_client import (
    create_client as _create_http_client,
)
from tldw_Server_API.app.core.Third_Party import PMC_OA as PMC_OA
from tldw_Server_API.app.core.Third_Party import PMC_OAI as PMC_OAI
from tldw_Server_API.app.core.Third_Party import Arxiv as Arxiv
from tldw_Server_API.app.core.Third_Party import BioRxiv as BioRxiv
from tldw_Server_API.app.core.Third_Party import PubMed as PubMed
from tldw_Server_API.app.core.Third_Party import Semantic_Scholar as Semantic_Scholar
from tldw_Server_API.app.core.Utils.pydantic_compat import model_dump_compat

router = APIRouter()


def _raw_passthrough_responses(*media_types: str) -> dict[int, dict[str, Any]]:
    return {
        200: {
            "description": "Raw upstream provider response.",
            "content": {media_type: {} for media_type in media_types},
        },
    }


_RAW_JSON_XML_HTML_RESPONSES = _raw_passthrough_responses("application/json", "application/xml", "text/html")
_RAW_JSON_CSV_RESPONSES = _raw_passthrough_responses("application/json", "text/csv")
_RAW_JSON_XML_RESPONSES = _raw_passthrough_responses("application/json", "application/xml")
_RAW_XML_RESPONSES = _raw_passthrough_responses("application/xml")
_RAW_JSON_RESPONSES = _raw_passthrough_responses("application/json")
_RAW_PDF_RESPONSES = _raw_passthrough_responses("application/pdf")
_RAW_HAL_RESPONSES = _raw_passthrough_responses(
    "application/json",
    "application/xml",
    "application/octet-stream",
    "text/csv",
    "text/plain",
    "application/atom+xml",
    "application/rss+xml",
)

_PROVIDER_TIMEOUT_DETAIL = "Upstream provider request timed out"
_PROVIDER_ERROR_DETAIL = "Upstream provider request failed"
_PROVIDER_UNEXPECTED_DETAIL = "Unexpected provider error"
_PROVIDER_NOT_CONFIGURED_DETAIL = "Upstream provider not configured"

_HTTPX_ERROR = getattr(httpx, "HTTPError", None) if httpx is not None else None
_AIOHTTP_ERROR = getattr(aiohttp, "ClientError", None) if aiohttp is not None else None
_OPTIONAL_HTTP_EXCEPTIONS = tuple(
    exc for exc in (_HTTPX_ERROR, _AIOHTTP_ERROR) if exc is not None
)

_PAPER_SEARCH_NONCRITICAL_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    AttributeError,
    ConnectionError,
    TimeoutError,
    json.JSONDecodeError,
    re.error,
    asyncio.TimeoutError,
) + _OPTIONAL_HTTP_EXCEPTIONS


def _raise_http_error_from_exception(
    exc: Exception,
    *,
    not_found_detail: Optional[str] = None,
    default_status: int = 502,
    force_status: Optional[int] = None,
) -> None:
    """Raise HTTPException when an exception carries a response status code."""
    response = getattr(exc, "response", None)
    if response is None:
        return
    status = getattr(response, "status_code", None)
    try:
        status_code = int(status)
    except (TypeError, ValueError):
        if default_status is None:
            return
        status_code = default_status

    if status_code == 404 and not_found_detail:
        raise HTTPException(status_code=404, detail=not_found_detail) from exc
    if force_status is not None:
        raise HTTPException(status_code=force_status, detail=_PROVIDER_ERROR_DETAIL) from exc
    raise HTTPException(status_code=status_code, detail=_PROVIDER_ERROR_DETAIL) from exc


def _ingest_paper_search_media(
    *,
    media_db: MediaDbSession | MediaWriterLike,
    url: str,
    title: str,
    media_type: str,
    content: str,
    keywords: list[str],
    overwrite: bool = False,
    **kwargs: Any,
) -> tuple[Any, Any, Any]:
    """Route paper-search ingest through the repository API for real DB sessions."""
    try:
        media_writer = get_media_repository(media_db)
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS:
        media_writer = media_db
    return media_writer.add_media_with_keywords(
        url=url,
        title=title,
        media_type=media_type,
        content=content,
        keywords=keywords,
        overwrite=overwrite,
        **kwargs,
    )


async def _download_pdf_bytes(
    url: str,
    *,
    headers: Optional[dict[str, str]] = None,
    timeout: int = 60,
    enforce_pdf: bool = False,
) -> bytes:
    """HEAD + atomic download to bytes using centralized http client.

    - Performs a HEAD to validate content-type when available.
    - Streams to a temp file via adownload with retries and atomic rename.
    - Returns the downloaded file as bytes and removes the temp file.
    """
    # Test-friendly fallback: if a module-level `_http_session()` factory is present
    # (older tests monkeypatch this), use it to GET the URL and return its content.
    try:
        _sess_factory = globals().get("_http_session")
        if callable(_sess_factory):
            _sess = _sess_factory()
            if hasattr(_sess, "get"):
                _resp = _sess.get(url, timeout=timeout)
                # Best-effort PDF validation for enforce_pdf: accept Content-Disposition *.pdf
                if enforce_pdf:
                    try:
                        disp = str((_resp.headers or {}).get("Content-Disposition") or "").lower()
                        if (".pdf" not in disp) and not (isinstance(getattr(_resp, "content", b""), (bytes, bytearray)) and (getattr(_resp, "content", b"") or b"").startswith(b"%PDF")):
                            raise HTTPException(status_code=415, detail="Expected PDF content")
                    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS:
                        # If header inspection fails, fall through to content check below
                        pass
                data_b = getattr(_resp, "content", b"") or b""
                if not isinstance(data_b, (bytes, bytearray)) or not data_b:
                    raise HTTPException(status_code=502, detail="PDF download returned empty content")  # noqa: TRY301
                return bytes(data_b)
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS:
        # Fallback to standard async client path below
        pass

    # 1) Preflight using GET with Range: bytes=0-0 (prefer over HEAD)
    try:
        _hdrs = dict(headers or {})
        _hdrs.setdefault("Range", "bytes=0-0")
        r = await _http_afetch(method="GET", url=url, headers=_hdrs, timeout=timeout)
        try:
            ctype = (r.headers.get("content-type") or "").lower()
        finally:
            with contextlib.suppress(_PAPER_SEARCH_NONCRITICAL_EXCEPTIONS):
                await r.aclose()
        if ctype:
            is_pdf = ("application/pdf" in ctype) or ctype.endswith("/pdf")
            if enforce_pdf and not is_pdf:
                raise HTTPException(status_code=415, detail=f"Expected application/pdf; got {ctype}")
            if not is_pdf:
                logger.warning(f"PDF download content-type not 'application/pdf': {ctype}; continuing")
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        # Preflight may fail; log and proceed to GET download
        logger.debug(f"Preflight GET Range failed for {url}: {e}")

    # 2) Stream download to a temp path, then read bytes
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        dest_path = Path(tmp.name)
    try:
        await _http_adownload(url=url, dest=dest_path, headers=headers or {}, timeout=timeout, retry=_RetryPolicy())
        data = dest_path.read_bytes()
        if not data:
            raise HTTPException(status_code=502, detail="PDF download returned empty content")
        return data
    finally:
        try:
            if dest_path.exists():
                os.unlink(dest_path)
        except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS:
            pass


@router.get(
    "/arxiv",
    response_model=ArxivSearchResponse,
    summary="Search arXiv papers",
    tags=["paper-search"],
)
async def paper_search_arxiv(
    search_params: ArxivSearchRequestForm = Depends(),
    token: Optional[str] = Query(None, alias="X-Token"),
):
    """Provider-specific search for arXiv papers with pagination."""
    start_index = (search_params.page - 1) * search_params.results_per_page
    logger.info(
        f"/paper-search/arxiv: query='{search_params.query}', author='{search_params.author}', year='{search_params.year}', page={search_params.page}, size={search_params.results_per_page}"
    )
    loop = asyncio.get_running_loop()
    try:
        papers_list, total_results_from_api, error_message = await loop.run_in_executor(
            None,
            Arxiv.search_arxiv_custom_api,
            search_params.query,
            search_params.author,
            search_params.year,
            start_index,
            search_params.results_per_page,
        )
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected arXiv search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_PROVIDER_UNEXPECTED_DETAIL) from e

    if error_message:
        _handle_provider_error(error_message)
    if papers_list is None:
        raise HTTPException(status_code=500, detail="arXiv search failed to return paper data.")

    total_pages = math.ceil(total_results_from_api / search_params.results_per_page) if search_params.results_per_page > 0 else 0
    if total_results_from_api == 0:
        total_pages = 0
    return ArxivSearchResponse(
        query_echo={
            "query": search_params.query,
            "author": search_params.author,
            "year": search_params.year,
        },
        items=[ArxivPaper(**paper_data) for paper_data in papers_list],
        total_results=total_results_from_api,
        page=search_params.page,
        results_per_page=search_params.results_per_page,
        total_pages=total_pages,
        pagination=build_page_pagination_meta(
            page=search_params.page,
            per_page=search_params.results_per_page,
            total=total_results_from_api,
            total_pages=total_pages,
        ),
    )


@router.get(
    "/biorxiv",
    response_model=BioRxivSearchResponse,
    summary="Search BioRxiv/MedRxiv papers",
    tags=["paper-search"],
)
async def paper_search_biorxiv(
    search_params: BioRxivSearchRequestForm = Depends(),
):
    """Provider-specific search for BioRxiv/MedRxiv with pagination."""
    offset = (search_params.page - 1) * search_params.results_per_page
    logger.info(
        f"/paper-search/biorxiv: q='{search_params.q}', server='{search_params.server}', page={search_params.page}, size={search_params.results_per_page}"
    )
    loop = asyncio.get_running_loop()
    try:
        items, total_results, error_message = await loop.run_in_executor(
            None,
            BioRxiv.search_biorxiv,
            search_params.q,
            search_params.server,
            search_params.from_date,
            search_params.to_date,
            search_params.category,
            offset,
            search_params.results_per_page,
            search_params.recent_days,
            search_params.recent_count,
        )
        if error_message:
            _handle_provider_error(error_message)
        if items is None:
            raise HTTPException(status_code=500, detail="BioRxiv search failed to return data.")  # noqa: TRY301
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected BioRxiv search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_PROVIDER_UNEXPECTED_DETAIL) from e

    total_pages = math.ceil(total_results / search_params.results_per_page) if search_params.results_per_page > 0 else 0
    if total_results == 0:
        total_pages = 0
    return BioRxivSearchResponse(
        query_echo={
            "q": search_params.q,
            "server": search_params.server,
            "from_date": search_params.from_date,
            "to_date": search_params.to_date,
            "category": search_params.category,
        },
        items=[BioRxivPaper(**it) for it in items],
        total_results=total_results,
        page=search_params.page,
        results_per_page=search_params.results_per_page,
        total_pages=total_pages,
        pagination=build_page_pagination_meta(
            page=search_params.page,
            per_page=search_params.results_per_page,
            total=total_results,
            total_pages=total_pages,
        ),
    )


# -------------------- MedRxiv Convenience Endpoints (aliases to BioRxiv adapter) --------------------

@router.get(
    "/medrxiv",
    response_model=BioRxivSearchResponse,
    summary="Search MedRxiv papers",
    tags=["paper-search"],
)
async def paper_search_medrxiv(
    search_params: BioRxivSearchRequestForm = Depends(),
):
    # Force server to medrxiv and delegate to same logic
    search_params.server = "medrxiv"
    return await paper_search_biorxiv(search_params)


@router.get(
    "/medrxiv/by-doi",
    response_model=BioRxivPaper,
    summary="Get MedRxiv paper by DOI",
    tags=["paper-search"],
)
async def paper_search_medrxiv_by_doi(
    doi: str = Query(..., description="MedRxiv DOI, e.g., 10.1101/2020.03.24.20042964"),
):
    loop = asyncio.get_running_loop()
    try:
        item, error_message = await loop.run_in_executor(None, BioRxiv.get_biorxiv_by_doi, doi, "medrxiv")
        if error_message:
            _handle_provider_error(error_message)
        if not item:
            raise HTTPException(status_code=404, detail="Paper not found for DOI")  # noqa: TRY301
        return BioRxivPaper(**item)
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected MedRxiv DOI error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_PROVIDER_UNEXPECTED_DETAIL) from e


@router.get(
    "/medrxiv/raw/details",
    summary="Raw passthrough for MedRxiv details endpoint (json|xml|html)",
    response_class=Response,
    responses=_RAW_JSON_XML_HTML_RESPONSES,
    tags=["paper-search"],
)
async def medrxiv_raw_details(
    from_date: Optional[str] = Query(None),
    to_date: Optional[str] = Query(None),
    recent_days: Optional[int] = Query(None),
    recent_count: Optional[int] = Query(None),
    doi: Optional[str] = Query(None),
    cursor: int = Query(0, ge=0),
    category: Optional[str] = Query(None),
    format: str = Query("json", description="json|xml|html"),
):
    loop = asyncio.get_running_loop()
    content, media_type, err = await loop.run_in_executor(
        None,
        BioRxiv.raw_details,
        "medrxiv",
        from_date,
        to_date,
        recent_days,
        recent_count,
        doi,
        cursor,
        category,
        format,
    )
    if err:
        _handle_provider_error(err)
    if not content:
        raise HTTPException(status_code=404, detail="No content")
    return Response(content=content, media_type=media_type or "application/octet-stream")


@router.get(
    "/medrxiv/raw/pubs",
    summary="Raw passthrough for MedRxiv pubs endpoint (json|csv)",
    response_class=Response,
    responses=_RAW_JSON_CSV_RESPONSES,
    tags=["paper-search"],
)
async def medrxiv_raw_pubs(
    from_date: Optional[str] = Query(None),
    to_date: Optional[str] = Query(None),
    recent_days: Optional[int] = Query(None),
    recent_count: Optional[int] = Query(None),
    doi: Optional[str] = Query(None),
    cursor: int = Query(0, ge=0),
    format: str = Query("json", description="json|csv"),
):
    loop = asyncio.get_running_loop()
    content, media_type, err = await loop.run_in_executor(
        None,
        BioRxiv.raw_pubs,
        "medrxiv",
        from_date,
        to_date,
        recent_days,
        recent_count,
        doi,
        cursor,
        format,
    )
    if err:
        _handle_provider_error(err)
    if not content:
        raise HTTPException(status_code=404, detail="No content")
    return Response(content=content, media_type=media_type or "application/octet-stream")


@router.get(
    "/medrxiv/raw/pub",
    summary="Raw passthrough for MedRxiv pub endpoint (json|csv)",
    response_class=Response,
    responses=_RAW_JSON_CSV_RESPONSES,
    tags=["paper-search"],
)
async def medrxiv_raw_pub(
    from_date: Optional[str] = Query(None),
    to_date: Optional[str] = Query(None),
    recent_days: Optional[int] = Query(None),
    recent_count: Optional[int] = Query(None),
    cursor: int = Query(0, ge=0),
    format: str = Query("json", description="json|csv"),
):
    loop = asyncio.get_running_loop()
    content, media_type, err = await loop.run_in_executor(
        None,
        BioRxiv.raw_pub,
        from_date,
        to_date,
        recent_days,
        recent_count,
        cursor,
        format,
    )
    if err:
        _handle_provider_error(err)
    if not content:
        raise HTTPException(status_code=404, detail="No content")
    return Response(content=content, media_type=media_type or "application/octet-stream")


# -------------------- PMC OAI-PMH Endpoints --------------------

@router.get(
    "/pmc-oai/identify",
    response_model=PMCOAIIdentifyResponse,
    summary="PMC OAI-PMH Identify",
    tags=["paper-search"],
)
async def pmc_oai_identify():
    loop = asyncio.get_running_loop()
    try:
        info, error_message = await loop.run_in_executor(None, PMC_OAI.pmc_oai_identify)
        if error_message:
            logger.error(f"PMC OAI Identify error: {error_message}")
            if "timed out" in error_message.lower():
                raise HTTPException(status_code=504, detail=_PROVIDER_TIMEOUT_DETAIL)  # noqa: TRY301
            raise HTTPException(status_code=502, detail=_PROVIDER_ERROR_DETAIL)  # noqa: TRY301
        if info is None:
            raise HTTPException(status_code=500, detail="PMC OAI Identify returned no data.")  # noqa: TRY301
        return PMCOAIIdentifyResponse(info=info)
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected PMC OAI Identify error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_PROVIDER_UNEXPECTED_DETAIL) from e


@router.get(
    "/pmc-oai/list-sets",
    response_model=PMCOAIListSetsResponse,
    summary="PMC OAI-PMH ListSets",
    tags=["paper-search"],
)
async def pmc_oai_list_sets(resumptionToken: Optional[str] = Query(None)):  # noqa: N803
    loop = asyncio.get_running_loop()
    try:
        items, next_token, error_message = await loop.run_in_executor(None, PMC_OAI.pmc_oai_list_sets, resumptionToken)
        if error_message:
            logger.error(f"PMC OAI ListSets error: {error_message}")
            if "timed out" in error_message.lower():
                raise HTTPException(status_code=504, detail=_PROVIDER_TIMEOUT_DETAIL)  # noqa: TRY301
            raise HTTPException(status_code=502, detail=_PROVIDER_ERROR_DETAIL)  # noqa: TRY301
        if items is None:
            items = []
        return PMCOAIListSetsResponse(query_echo={"resumptionToken": resumptionToken}, items=[
            {
                "setSpec": it.get("setSpec"),
                "setName": it.get("setName"),
            } for it in items
        ], resumption_token=next_token)
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected PMC OAI ListSets error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_PROVIDER_UNEXPECTED_DETAIL) from e


@router.get(
    "/pmc-oai/list-identifiers",
    response_model=PMCOAIIdentifiersResponse,
    summary="PMC OAI-PMH ListIdentifiers",
    tags=["paper-search"],
)
async def pmc_oai_list_identifiers(
    metadataPrefix: str = Query("oai_dc"),  # noqa: N803
    from_date: Optional[str] = Query(None, alias="from"),
    until_date: Optional[str] = Query(None, alias="until"),
    set_name: Optional[str] = Query(None, alias="set"),
    resumptionToken: Optional[str] = Query(None),  # noqa: N803
):
    loop = asyncio.get_running_loop()
    try:
        items, next_token, error_message = await loop.run_in_executor(
            None, PMC_OAI.pmc_oai_list_identifiers, metadataPrefix, from_date, until_date, set_name, resumptionToken
        )
        if error_message:
            logger.error(f"PMC OAI ListIdentifiers error: {error_message}")
            if "timed out" in error_message.lower():
                raise HTTPException(status_code=504, detail=_PROVIDER_TIMEOUT_DETAIL)  # noqa: TRY301
            raise HTTPException(status_code=502, detail=_PROVIDER_ERROR_DETAIL)  # noqa: TRY301
        if items is None:
            items = []
        return PMCOAIIdentifiersResponse(
            query_echo={"metadataPrefix": metadataPrefix, "from": from_date, "until": until_date, "set": set_name, "resumptionToken": resumptionToken},
            items=[PMCOAIHeader(**it) for it in items],
            resumption_token=next_token,
        )
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected PMC OAI ListIdentifiers error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_PROVIDER_UNEXPECTED_DETAIL) from e


@router.get(
    "/pmc-oai/list-records",
    response_model=PMCOAIListResponse,
    summary="PMC OAI-PMH ListRecords",
    tags=["paper-search"],
)
async def pmc_oai_list_records(
    metadataPrefix: str = Query("oai_dc"),  # noqa: N803
    from_date: Optional[str] = Query(None, alias="from"),
    until_date: Optional[str] = Query(None, alias="until"),
    set_name: Optional[str] = Query(None, alias="set"),
    resumptionToken: Optional[str] = Query(None),  # noqa: N803
):
    loop = asyncio.get_running_loop()
    try:
        items, next_token, error_message = await loop.run_in_executor(
            None, PMC_OAI.pmc_oai_list_records, metadataPrefix, from_date, until_date, set_name, resumptionToken
        )
        if error_message:
            logger.error(f"PMC OAI ListRecords error: {error_message}")
            if "timed out" in error_message.lower():
                raise HTTPException(status_code=504, detail=_PROVIDER_TIMEOUT_DETAIL)  # noqa: TRY301
            raise HTTPException(status_code=502, detail=_PROVIDER_ERROR_DETAIL)  # noqa: TRY301
        if items is None:
            items = []
        return PMCOAIListResponse(
            query_echo={"metadataPrefix": metadataPrefix, "from": from_date, "until": until_date, "set": set_name, "resumptionToken": resumptionToken},
            items=[PMCOAIRecord(**it) for it in items],
            resumption_token=next_token,
        )
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected PMC OAI ListRecords error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_PROVIDER_UNEXPECTED_DETAIL) from e


@router.get(
    "/pmc-oai/get-record",
    response_model=PMCOAIRecord,
    summary="PMC OAI-PMH GetRecord",
    tags=["paper-search"],
)
async def pmc_oai_get_record(identifier: str = Query(...), metadataPrefix: str = Query("oai_dc")):  # noqa: N803
    loop = asyncio.get_running_loop()
    try:
        item, error_message = await loop.run_in_executor(None, PMC_OAI.pmc_oai_get_record, identifier, metadataPrefix)
        if error_message:
            logger.error(f"PMC OAI GetRecord error: {error_message}")
            if "timed out" in error_message.lower():
                raise HTTPException(status_code=504, detail=_PROVIDER_TIMEOUT_DETAIL)  # noqa: TRY301
            raise HTTPException(status_code=502, detail=_PROVIDER_ERROR_DETAIL)  # noqa: TRY301
        if not item:
            raise HTTPException(status_code=404, detail="Record not found")  # noqa: TRY301
        return PMCOAIRecord(**item)
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected PMC OAI GetRecord error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_PROVIDER_UNEXPECTED_DETAIL) from e


# -------------------- PMC OA Web Service Endpoints --------------------

@router.get(
    "/pmc-oa/identify",
    response_model=PMCOAIdentifyResponse,
    summary="PMC OA Web Service identification",
    tags=["paper-search"],
)
async def pmc_oa_identify():
    loop = asyncio.get_running_loop()
    try:
        info, error_message = await loop.run_in_executor(None, PMC_OA.pmc_oa_identify)
        if error_message:
            logger.error(f"PMC OA Identify error: {error_message}")
            if "timed out" in error_message.lower():
                raise HTTPException(status_code=504, detail=_PROVIDER_TIMEOUT_DETAIL)  # noqa: TRY301
            raise HTTPException(status_code=502, detail=_PROVIDER_ERROR_DETAIL)  # noqa: TRY301
        if info is None:
            raise HTTPException(status_code=500, detail="PMC OA Identify returned no data.")  # noqa: TRY301
        return PMCOAIdentifyResponse(info=info)
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected PMC OA Identify error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_PROVIDER_UNEXPECTED_DETAIL) from e


@router.get(
    "/pmc-oa/query",
    response_model=PMCOAQueryResponse,
    summary="PMC OA Web Service query with resumption",
    tags=["paper-search"],
)
async def pmc_oa_query(
    from_date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    until_date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    format: Optional[str] = Query(None, description="'pdf' or 'tgz'"),
    resumptionToken: Optional[str] = Query(None),  # noqa: N803
    id: Optional[str] = Query(None, description="PMC ID like PMC5334499"),
    pdf_only: bool = Query(False, description="Filter results to those with PDF links"),
    license_contains: Optional[str] = Query(None, description="Case-insensitive substring match on license"),
):
    loop = asyncio.get_running_loop()
    try:
        items, next_token, error_message = await loop.run_in_executor(None, PMC_OA.pmc_oa_query, from_date, until_date, format, resumptionToken, id)
        if error_message:
            logger.error(f"PMC OA query error: {error_message}")
            if "timed out" in error_message.lower():
                raise HTTPException(status_code=504, detail=_PROVIDER_TIMEOUT_DETAIL)  # noqa: TRY301
            raise HTTPException(status_code=502, detail=_PROVIDER_ERROR_DETAIL)  # noqa: TRY301
        if items is None:
            items = []
        # Server-side filters
        if pdf_only:
            def has_pdf(it: dict) -> bool:
                for lk in it.get("links", []) or []:
                    if (lk.get("format") or "").lower() == "pdf" or (lk.get("href") or "").lower().endswith(".pdf"):
                        return True
                return False
            items = [it for it in items if has_pdf(it)]
        if license_contains and isinstance(license_contains, str):
            lc = license_contains.lower()
            items = [it for it in items if (it.get("license") or "").lower().find(lc) >= 0]
        return PMCOAQueryResponse(
            query_echo={"from": from_date, "until": until_date, "format": format, "resumptionToken": resumptionToken, "id": id, "pdf_only": pdf_only, "license_contains": license_contains},
            items=[PMCOARecord(**it) for it in items],
            resumption_token=next_token,
        )
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected PMC OA query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_PROVIDER_UNEXPECTED_DETAIL) from e


import contextlib  # noqa: E402, F811
import io  # noqa: E402

from fastapi.responses import Response, StreamingResponse  # noqa: E402, F811


@router.get(
    "/pmc-oa/fetch-pdf",
    summary="Fetch PMC PDF by PMCID and return as attachment",
    response_class=StreamingResponse,
    responses=_RAW_PDF_RESPONSES,
    tags=["paper-search"],
)
async def pmc_oa_fetch_pdf(pmcid: str = Query(..., description="PMCID numeric or with 'PMC' prefix")):
    loop = asyncio.get_running_loop()
    try:
        content, filename, error_message = await loop.run_in_executor(None, PMC_OA.download_pmc_pdf, pmcid)
        if error_message:
            logger.error(f"PMC OA fetch-pdf error: {error_message}")
            if "timed out" in error_message.lower():
                raise HTTPException(status_code=504, detail=_PROVIDER_TIMEOUT_DETAIL)  # noqa: TRY301
            if "http error" in error_message.lower():
                nums = [int(s) for s in error_message.split() if s.isdigit() and 400 <= int(s) < 600]
                code = nums[0] if nums else 502
                raise HTTPException(status_code=code, detail=_PROVIDER_ERROR_DETAIL)  # noqa: TRY301
            raise HTTPException(status_code=502, detail=_PROVIDER_ERROR_DETAIL)  # noqa: TRY301
        if not content:
            raise HTTPException(status_code=404, detail="PDF not found for PMCID")  # noqa: TRY301
        stream = io.BytesIO(content)
        resp = StreamingResponse(stream, media_type="application/pdf")
        resp.headers["Content-Disposition"] = f"attachment; filename=\"{filename or 'article.pdf'}\""
        return resp  # noqa: TRY300
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected PMC OA fetch-pdf error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_PROVIDER_UNEXPECTED_DETAIL) from e


@router.post(
    "/pmc-oa/ingest-pdf",
    summary="Download PMC PDF by PMCID, process, and persist to DB",
    tags=["paper-search"],
)
async def pmc_oa_ingest_pdf(
    pmcid: str = Query(..., description="PMCID numeric or with 'PMC' prefix"),
    title: Optional[str] = Query(None, description="Optional title override"),
    author: Optional[str] = Query(None, description="Optional author override"),
    keywords: Optional[str] = Query(None, description="Optional comma-separated keywords"),
    perform_chunking: bool = Query(True, description="Enable chunking during processing"),
    parser: Optional[str] = Query("pymupdf4llm", description="PDF parsing backend"),
    # Analysis
    api_name: Optional[str] = Query(None, description="LLM API name for analysis"),
    custom_prompt: Optional[str] = Query(None, description="Custom prompt for analysis"),
    system_prompt: Optional[str] = Query(None, description="System prompt for analysis"),
    enable_ocr: bool = Query(False, description="Enable OCR for scanned PDFs"),
    ocr_backend: Optional[str] = Query(None, description="OCR backend (e.g., 'tesseract', 'auto')"),
    ocr_lang: Optional[str] = Query("eng", description="OCR language code"),
    ocr_dpi: int = Query(300, ge=72, le=600, description="OCR render DPI"),
    ocr_mode: Optional[str] = Query("fallback", description="OCR mode: 'always' or 'fallback'"),
    ocr_min_page_text_chars: int = Query(40, ge=0, le=2000, description="Min text chars/page to skip OCR"),
    ocr_output_format: Optional[str] = Query(None, description="OCR output format: text|markdown|json"),
    ocr_prompt_preset: Optional[str] = Query(None, description="OCR prompt preset (e.g., 'general', 'doc', 'table', 'spotting', 'json')"),
    chunk_method: Optional[str] = Query(None, description="Chunking method (e.g., 'sentences', 'semantic')"),
    chunk_size: int = Query(500, ge=50, le=4000, description="Target chunk size"),
    chunk_overlap: int = Query(200, ge=0, le=1000, description="Chunk overlap"),
    perform_analysis: bool = Query(True, description="Run analysis/summarization"),
    summarize_recursively: bool = Query(False, description="Enable recursive summarization"),
    enrich_metadata: bool = Query(True, description="Enrich with PMC OAI-PMH oai_dc metadata"),
    db: MediaDbSession = Depends(get_media_db_for_user),
):
    """Convenience endpoint: fetch PMC PDF and ingest into user's Media DB."""
    loop = asyncio.get_running_loop()
    try:
        # 1) Download PDF
        content, filename, error_message = await loop.run_in_executor(None, PMC_OA.download_pmc_pdf, pmcid)
        if error_message:
            logger.error(f"PMC OA download error: {error_message}")
            if "timed out" in error_message.lower():
                raise HTTPException(status_code=504, detail=_PROVIDER_TIMEOUT_DETAIL)  # noqa: TRY301
            if "http error" in error_message.lower():
                nums = [int(s) for s in error_message.split() if s.isdigit() and 400 <= int(s) < 600]
                code = nums[0] if nums else 502
                raise HTTPException(status_code=code, detail=_PROVIDER_ERROR_DETAIL)  # noqa: TRY301
            raise HTTPException(status_code=502, detail=_PROVIDER_ERROR_DETAIL)  # noqa: TRY301
        if not content:
            raise HTTPException(status_code=404, detail="PDF not found for PMCID")  # noqa: TRY301

        # 2) Process PDF bytes
        # Import locally to ease testing/monkeypatching
        from tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.PDF_Processing_Lib import process_pdf_task

        kw_list = None
        if keywords and isinstance(keywords, str):
            kw_list = [k.strip() for k in keywords.split(',') if k.strip()]

        result = await process_pdf_task(
            file_bytes=content,
            filename=filename or f"{pmcid}.pdf",
            parser=parser or "pymupdf4llm",
            title_override=title,
            author_override=author,
            keywords=kw_list,
            perform_chunking=perform_chunking,
            enable_ocr=enable_ocr or None,
            ocr_backend=ocr_backend or None,
            ocr_lang=ocr_lang or None,
            chunk_method=chunk_method,
            max_chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            perform_analysis=perform_analysis,
            api_name=api_name,
            custom_prompt=custom_prompt,
            system_prompt=system_prompt,
            summarize_recursively=summarize_recursively,
            ocr_dpi=ocr_dpi,
            ocr_mode=ocr_mode,
            ocr_min_page_text_chars=ocr_min_page_text_chars,
            ocr_output_format=ocr_output_format,
            ocr_prompt_preset=ocr_prompt_preset,
        )

        # 3) Optional OAI-PMH metadata enrichment (oai_dc)
        enriched_oai = None
        if enrich_metadata:
            try:
                pmcid_num = str(pmcid).strip().lstrip("PMC")
                oai_id = f"oai:pubmedcentral.nih.gov:{pmcid_num}"
                oai_item, oai_err = await loop.run_in_executor(None, PMC_OAI.pmc_oai_get_record, oai_id, "oai_dc")
                if not oai_err and oai_item and isinstance(oai_item.get("metadata"), dict):
                    enriched_oai = oai_item["metadata"]
            except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as _oai_ex:
                logger.warning(f"OAI-PMH enrichment failed: {_oai_ex}")

        # 4) Persist to DB
        content_for_db = result.get('transcript') or result.get('content')
        analysis_for_db = result.get('summary') or result.get('analysis')
        metadata_for_db = result.get('metadata') or {}

        # Merge in OAI metadata for title/author/ids if available
        if enriched_oai:
            if enriched_oai.get("title") and not title:
                metadata_for_db["title"] = enriched_oai.get("title")
            creators = enriched_oai.get("creators") or []
            if creators and not author:
                metadata_for_db["author"] = "; ".join(creators)
            for key in ("pmcid", "pmid", "doi", "date", "license_urls", "rights"):
                if enriched_oai.get(key) is not None:
                    metadata_for_db[key] = enriched_oai.get(key)
            try:
                import json
                enrich_txt = json.dumps({"oai_dc": enriched_oai}, ensure_ascii=False, indent=2)
                analysis_for_db = (analysis_for_db + "\n\nOAI Metadata:\n" + enrich_txt) if analysis_for_db else ("OAI Metadata:\n" + enrich_txt)
            except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS:
                pass
        extracted_keywords = metadata_for_db.get('keywords') or result.get('keywords') or []
        combined_keywords = set(kw_list or [])
        if isinstance(extracted_keywords, list):
            combined_keywords.update(k.strip().lower() for k in extracted_keywords if k and isinstance(k, str))
        final_keywords = sorted(combined_keywords)

        db_id = None
        media_uuid = None
        db_msg = "Skipped (no content)"
        if content_for_db:
            try:
                safe_metadata_json = None
                if enriched_oai:
                    try:
                        import json as _json
                        safe_metadata_json = _json.dumps({"oai_dc": enriched_oai}, ensure_ascii=False)
                    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS:
                        safe_metadata_json = None
                # Build plaintext chunks for chunk-level FTS if chunking enabled
                chunks_for_sql = None
                try:
                    if perform_chunking:
                        from tldw_Server_API.app.core.Chunking.chunker import Chunker as _Chunker
                        _ck = _Chunker()
                        _flat = _ck.chunk_text_hierarchical_flat(
                            content_for_db,
                            method=chunk_method or "sentences",
                            max_size=chunk_size,
                            overlap=chunk_overlap,
                        )
                        _kind_map = {
                            'paragraph': 'text', 'list_unordered': 'list', 'list_ordered': 'list',
                            'code_fence': 'code', 'table_md': 'table', 'header_line': 'heading', 'header_atx': 'heading'
                        }
                        chunks_for_sql = []
                        for _it in _flat:
                            _md = _it.get('metadata') or {}
                            _ctype = _kind_map.get(str(_md.get('paragraph_kind') or '').lower(), 'text')
                            _small = {}
                            if _md.get('ancestry_titles'):
                                _small['ancestry_titles'] = _md.get('ancestry_titles')
                            if _md.get('section_path'):
                                _small['section_path'] = _md.get('section_path')
                            chunks_for_sql.append({
                                'text': _it.get('text',''),
                                'start_char': _md.get('start_offset'),
                                'end_char': _md.get('end_offset'),
                                'chunk_type': _ctype,
                                'metadata': _small,
                            })
                except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS:
                    chunks_for_sql = None

                db_id, media_uuid, db_msg = await loop.run_in_executor(
                    None,
                    lambda: _ingest_paper_search_media(
                        media_db=db,
                        url=f"pmcid:{pmcid}",
                        title=metadata_for_db.get('title') or title or (filename or pmcid),
                        media_type="pdf",
                        content=content_for_db,
                        keywords=final_keywords,
                        prompt=None,
                        analysis_content=analysis_for_db,
                        transcription_model=metadata_for_db.get('parser_used') or 'Imported',
                        author=metadata_for_db.get('author') or author,
                        safe_metadata=safe_metadata_json,
                        overwrite=False,
                        chunk_options={
                            "method": chunk_method if chunk_method else "sentences",
                            "max_size": chunk_size,
                            "overlap": chunk_overlap,
                        } if perform_chunking else None,
                        chunks=chunks_for_sql,
                    )
                )
            except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
                logger.error(f"DB persistence failed for PMCID {pmcid}: {e}", exc_info=True)
                db_msg = f"DB error: {e}"

        return {
            "pmcid": pmcid,
            "filename": filename,
            "status": result.get("status", "Unknown"),
            "result": result,
            "db_id": db_id,
            "media_uuid": media_uuid,
            "db_message": db_msg,
        }
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected PMC OA ingest error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_PROVIDER_UNEXPECTED_DETAIL) from e


def _http_session():
    """Create a centralized HTTP client for outbound fetches in research endpoints.

    Uses the shared http_client factory (trust_env=False) and sets basic headers.
    """
    client = _create_http_client(timeout=30)
    with contextlib.suppress(_PAPER_SEARCH_NONCRITICAL_EXCEPTIONS):
        client.headers.update({"Accept-Encoding": "gzip, deflate"})
    return client


@router.post(
    "/arxiv/ingest",
    summary="Download arXiv PDF by arXiv ID, process, and persist",
    tags=["paper-search"],
    dependencies=[Depends(guard_backpressure_and_quota)],
)
async def arxiv_ingest(
    arxiv_id: str = Query(..., description="arXiv ID, e.g., 1706.03762"),
    keywords: Optional[str] = Query(None, description="Comma-separated keywords"),
    perform_chunking: bool = Query(True),
    parser: Optional[str] = Query("pymupdf4llm"),
    chunk_method: Optional[str] = Query(None),
    chunk_size: int = Query(500, ge=50, le=4000),
    chunk_overlap: int = Query(200, ge=0, le=1000),
    perform_analysis: bool = Query(True),
    custom_prompt: Optional[str] = Query(None),
    system_prompt: Optional[str] = Query(None),
    api_name: Optional[str] = Query(None),
    enable_ocr: bool = Query(False),
    ocr_backend: Optional[str] = Query(None),
    ocr_lang: Optional[str] = Query("eng"),
    ocr_dpi: int = Query(300, ge=72, le=600),
    ocr_mode: Optional[str] = Query("fallback"),
    ocr_min_page_text_chars: int = Query(40, ge=0, le=2000),
    ocr_output_format: Optional[str] = Query(None, description="OCR output format: text|markdown|json"),
    ocr_prompt_preset: Optional[str] = Query(None, description="OCR prompt preset (e.g., 'general', 'doc', 'table', 'spotting', 'json')"),
    db: MediaDbSession = Depends(get_media_db_for_user),
):
    loop = asyncio.get_running_loop()
    try:
        """Download an arXiv PDF, process it, and persist to the Media DB.

        Example:
          POST /api/v1/paper-search/arxiv/ingest?arxiv_id=1706.03762&perform_analysis=true
        """
        # Metadata & PDF URL
        xml_text = Arxiv.fetch_arxiv_xml(arxiv_id) or ""
        meta = {}
        if xml_text:
            try:
                parsed = Arxiv.parse_arxiv_feed(xml_text.encode("utf-8"))
                if parsed:
                    meta = parsed[0]
            except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS:
                meta = {}
        pdf_url = Arxiv.fetch_arxiv_pdf_url(arxiv_id)
        if not pdf_url:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        content = await _download_pdf_bytes(pdf_url)

        # Process PDF
        from tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.PDF_Processing_Lib import process_pdf_task
        kw_list = [k.strip() for k in (keywords or '').split(',') if k.strip()] if keywords else None
        result = await process_pdf_task(
            file_bytes=content,
            filename=f"{arxiv_id}.pdf",
            parser=parser or "pymupdf4llm",
            keywords=kw_list,
            perform_chunking=perform_chunking,
            chunk_method=chunk_method,
            max_chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            perform_analysis=perform_analysis,
            api_name=api_name,
            custom_prompt=custom_prompt,
            system_prompt=system_prompt,
            enable_ocr=enable_ocr or None,
            ocr_backend=ocr_backend or None,
            ocr_lang=ocr_lang or None,
            ocr_dpi=ocr_dpi,
            ocr_mode=ocr_mode,
            ocr_min_page_text_chars=ocr_min_page_text_chars,
            ocr_output_format=ocr_output_format,
            ocr_prompt_preset=ocr_prompt_preset,
        )

        # Persist with safe_metadata
        content_for_db = result.get('transcript') or result.get('content')
        if not content_for_db:
            raise HTTPException(status_code=500, detail="Processing did not produce content")  # noqa: TRY301
        from tldw_Server_API.app.core.Utils.metadata_utils import normalize_safe_metadata
        sm = normalize_safe_metadata({
            "arxiv_id": arxiv_id,
            "title": meta.get('title'),
            "authors": meta.get('authors'),
            "date": meta.get('published_date'),
            "pdf_url": meta.get('pdf_url') or pdf_url,
            "source": "arxiv",
        })
        import json as _json
        smj = _json.dumps({k: v for k, v in sm.items() if v}, ensure_ascii=False)
        analysis_for_db = result.get('summary') or result.get('analysis')
        title_for_db = meta.get('title') or arxiv_id
        author_for_db = meta.get('authors')

        # Build plaintext chunks for chunk-level FTS if chunking enabled
        chunks_for_sql = None
        try:
            if perform_chunking:
                from tldw_Server_API.app.core.Chunking.chunker import Chunker as _Chunker
                _ck = _Chunker()
                _flat = _ck.chunk_text_hierarchical_flat(
                    content_for_db,
                    method=chunk_method or "sentences",
                    max_size=chunk_size,
                    overlap=chunk_overlap,
                )
                _kind_map = {
                    'paragraph': 'text', 'list_unordered': 'list', 'list_ordered': 'list',
                    'code_fence': 'code', 'table_md': 'table', 'header_line': 'heading', 'header_atx': 'heading'
                }
                chunks_for_sql = []
                for _it in _flat:
                    _md = _it.get('metadata') or {}
                    _ctype = _kind_map.get(str(_md.get('paragraph_kind') or '').lower(), 'text')
                    _small = {}
                    if _md.get('ancestry_titles'):
                        _small['ancestry_titles'] = _md.get('ancestry_titles')
                    if _md.get('section_path'):
                        _small['section_path'] = _md.get('section_path')
                    chunks_for_sql.append({
                        'text': _it.get('text',''),
                        'start_char': _md.get('start_offset'),
                        'end_char': _md.get('end_offset'),
                        'chunk_type': _ctype,
                        'metadata': _small,
                    })
        except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS:
            chunks_for_sql = None

        media_id, media_uuid, msg = await loop.run_in_executor(
            None,
            lambda: _ingest_paper_search_media(
                media_db=db,
                url=f"arxiv:{arxiv_id}",
                title=title_for_db,
                media_type="pdf",
                content=content_for_db,
                keywords=kw_list or [],
                prompt=custom_prompt,
                analysis_content=analysis_for_db,
                safe_metadata=smj,
                transcription_model='Imported',
                author=author_for_db,
                overwrite=False,
                chunk_options={"method": chunk_method or "sentences", "max_size": chunk_size, "overlap": chunk_overlap} if perform_chunking else None,
                chunks=chunks_for_sql,
            )
        )
        return {"message": msg, "media_id": media_id, "media_uuid": media_uuid}  # noqa: TRY300
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        _raise_http_error_from_exception(
            e,
            not_found_detail="arXiv PDF returned 404",
            force_status=502,
        )
        logger.error(f"arXiv ingest error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="arXiv ingest failed") from e


@router.post(
    "/earthrxiv/ingest",
    summary="Download EarthArXiv PDF by OSF ID, process, and persist",
    tags=["paper-search"],
    dependencies=[Depends(guard_backpressure_and_quota)],
)
async def eartharxiv_ingest(
    osf_id: str = Query(..., description="EarthArXiv OSF ID (e.g., 12345)"),
    keywords: Optional[str] = Query(None, description="Comma-separated keywords"),
    perform_chunking: bool = Query(True),
    parser: Optional[str] = Query("pymupdf4llm"),
    chunk_method: Optional[str] = Query(None),
    chunk_size: int = Query(500, ge=50, le=4000),
    chunk_overlap: int = Query(200, ge=0, le=1000),
    perform_analysis: bool = Query(True),
    custom_prompt: Optional[str] = Query(None),
    system_prompt: Optional[str] = Query(None),
    api_name: Optional[str] = Query(None),
    enable_ocr: bool = Query(False),
    ocr_backend: Optional[str] = Query(None),
    ocr_lang: Optional[str] = Query("eng"),
    ocr_dpi: int = Query(300, ge=72, le=600),
    ocr_mode: Optional[str] = Query("fallback"),
    ocr_min_page_text_chars: int = Query(40, ge=0, le=2000),
    ocr_output_format: Optional[str] = Query(None, description="OCR output format: text|markdown|json"),
    ocr_prompt_preset: Optional[str] = Query(None, description="OCR prompt preset (e.g., 'general', 'doc', 'table', 'spotting', 'json')"),
    db: MediaDbSession = Depends(get_media_db_for_user),
):
    """Download an EarthArXiv PDF by OSF ID, process, and persist to the Media DB."""
    loop = asyncio.get_running_loop()
    try:
        # 1) Metadata (best-effort) and PDF URL
        item, _err = await loop.run_in_executor(None, EarthRxiv.get_item_by_id, osf_id)
        title_meta = (item or {}).get('title') if isinstance(item, dict) else None
        doi_meta = (item or {}).get('doi') if isinstance(item, dict) else None
        pdf_url = f"https://eartharxiv.org/{osf_id}/download"
        content = await _download_pdf_bytes(pdf_url)

        # 2) Process
        from tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.PDF_Processing_Lib import process_pdf_task
        kw_list = [k.strip() for k in (keywords or '').split(',') if k.strip()] if keywords else None
        result = await process_pdf_task(
            file_bytes=content,
            filename=f"{osf_id}.pdf",
            parser=parser or "pymupdf4llm",
            keywords=kw_list,
            perform_chunking=perform_chunking,
            chunk_method=chunk_method,
            max_chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            perform_analysis=perform_analysis,
            api_name=api_name,
            custom_prompt=custom_prompt,
            system_prompt=system_prompt,
            enable_ocr=enable_ocr or None,
            ocr_backend=ocr_backend or None,
            ocr_lang=ocr_lang or None,
            ocr_dpi=ocr_dpi,
            ocr_mode=ocr_mode,
            ocr_min_page_text_chars=ocr_min_page_text_chars,
            ocr_output_format=ocr_output_format,
            ocr_prompt_preset=ocr_prompt_preset,
        )
        content_for_db = result.get('transcript') or result.get('content') or result.get('text')
        if not content_for_db:
            raise HTTPException(status_code=500, detail="Processing did not produce content")  # noqa: TRY301

        from tldw_Server_API.app.core.Utils.metadata_utils import normalize_safe_metadata
        sm = normalize_safe_metadata({
            "osf_id": osf_id,
            "doi": doi_meta,
            "title": title_meta,
            "pdf_url": pdf_url,
            "source": "eartharxiv",
        })
        import json as _json
        smj = _json.dumps({k: v for k, v in sm.items() if v}, ensure_ascii=False)
        analysis_for_db = result.get('summary') or result.get('analysis')
        title_for_db = title_meta or osf_id
        author_for_db = None  # Not fetched in lightweight adapter

        # Build plaintext chunks for chunk-level FTS if chunking enabled
        chunks_for_sql = None
        try:
            if perform_chunking:
                from tldw_Server_API.app.core.Chunking.chunker import Chunker as _Chunker
                _ck = _Chunker()
                _flat = _ck.chunk_text_hierarchical_flat(
                    content_for_db,
                    method=chunk_method or "sentences",
                    max_size=chunk_size,
                    overlap=chunk_overlap,
                )
                _kind_map = {
                    'paragraph': 'text', 'list_unordered': 'list', 'list_ordered': 'list',
                    'code_fence': 'code', 'table_md': 'table', 'header_line': 'heading', 'header_atx': 'heading'
                }
                chunks_for_sql = []
                for _it in _flat:
                    _md = _it.get('metadata') or {}
                    _ctype = _kind_map.get(str(_md.get('paragraph_kind') or '').lower(), 'text')
                    _small = {}
                    if _md.get('ancestry_titles'):
                        _small['ancestry_titles'] = _md.get('ancestry_titles')
                    if _md.get('section_path'):
                        _small['section_path'] = _md.get('section_path')
                    chunks_for_sql.append({
                        'text': _it.get('text',''),
                        'start_char': _md.get('start_offset'),
                        'end_char': _md.get('end_offset'),
                        'chunk_type': _ctype,
                        'metadata': _small,
                    })
        except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS:
            chunks_for_sql = None

        media_id, media_uuid, msg = await loop.run_in_executor(
            None,
            lambda: _ingest_paper_search_media(
                media_db=db,
                url=f"eartharxiv:{osf_id}",
                title=title_for_db,
                media_type="pdf",
                content=content_for_db,
                keywords=kw_list or [],
                prompt=custom_prompt,
                analysis_content=analysis_for_db,
                safe_metadata=smj,
                transcription_model='Imported',
                author=author_for_db,
                overwrite=False,
                chunk_options={"method": chunk_method or "sentences", "max_size": chunk_size, "overlap": chunk_overlap} if perform_chunking else None,
                chunks=chunks_for_sql,
            )
        )
        return {"message": msg, "media_id": media_id, "media_uuid": media_uuid}  # noqa: TRY300
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        _raise_http_error_from_exception(
            e,
            not_found_detail="EarthArXiv PDF returned 404",
            force_status=502,
        )
        logger.error(f"EarthArXiv ingest error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="EarthArXiv ingest failed") from e


@router.post(
    "/pubmed/ingest",
    summary="Download PubMed-linked PMC PDF by PMID, process, and persist",
    tags=["paper-search"],
)
async def pubmed_ingest(
    pmid: str = Query(..., description="PubMed PMID"),
    keywords: Optional[str] = Query(None),
    perform_chunking: bool = Query(True),
    parser: Optional[str] = Query("pymupdf4llm"),
    chunk_method: Optional[str] = Query(None),
    chunk_size: int = Query(500, ge=50, le=4000),
    chunk_overlap: int = Query(200, ge=0, le=1000),
    perform_analysis: bool = Query(True),
    custom_prompt: Optional[str] = Query(None),
    system_prompt: Optional[str] = Query(None),
    api_name: Optional[str] = Query(None),
    enable_ocr: bool = Query(False),
    ocr_backend: Optional[str] = Query(None),
    ocr_lang: Optional[str] = Query("eng"),
    ocr_dpi: int = Query(300, ge=72, le=600),
    ocr_mode: Optional[str] = Query("fallback"),
    ocr_min_page_text_chars: int = Query(40, ge=0, le=2000),
    ocr_output_format: Optional[str] = Query(None, description="OCR output format: text|markdown|json"),
    ocr_prompt_preset: Optional[str] = Query(None, description="OCR prompt preset (e.g., 'general', 'doc', 'table', 'spotting', 'json')"),
    db: MediaDbSession = Depends(get_media_db_for_user),
):
    loop = asyncio.get_running_loop()
    try:
        """Download an OA PMC PDF linked to a PubMed PMID, process it, and persist.

        Example:
          POST /api/v1/paper-search/pubmed/ingest?pmid=12345678&perform_analysis=true
        """
        meta, err = PubMed.get_pubmed_by_id(pmid)
        if err:
            raise HTTPException(status_code=502, detail=_PROVIDER_ERROR_DETAIL)  # noqa: TRY301
        if not meta:
            raise HTTPException(status_code=404, detail="PMID not found")  # noqa: TRY301
        pmcid = meta.get('pmcid')
        if not pmcid:
            raise HTTPException(status_code=400, detail="No PMC Open Access PMCID available for this PMID")  # noqa: TRY301
        content, filename, d_err = await loop.run_in_executor(None, PMC_OA.download_pmc_pdf, pmcid)
        if d_err or not content:
            raise HTTPException(status_code=502, detail=_PROVIDER_ERROR_DETAIL)  # noqa: TRY301

        from tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.PDF_Processing_Lib import process_pdf_task
        kw_list = [k.strip() for k in (keywords or '').split(',') if k.strip()] if keywords else None
        result = await process_pdf_task(
            file_bytes=content,
            filename=filename or f"PMC{pmcid}.pdf",
            parser=parser or "pymupdf4llm",
            keywords=kw_list,
            perform_chunking=perform_chunking,
            chunk_method=chunk_method,
            max_chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            perform_analysis=perform_analysis,
            api_name=api_name,
            custom_prompt=custom_prompt,
            system_prompt=system_prompt,
            enable_ocr=enable_ocr or None,
            ocr_backend=ocr_backend or None,
            ocr_lang=ocr_lang or None,
            ocr_dpi=ocr_dpi,
            ocr_mode=ocr_mode,
            ocr_min_page_text_chars=ocr_min_page_text_chars,
            ocr_output_format=ocr_output_format,
            ocr_prompt_preset=ocr_prompt_preset,
        )
        content_for_db = result.get('transcript') or result.get('content')
        if not content_for_db:
            raise HTTPException(status_code=500, detail="Processing did not produce content")  # noqa: TRY301
        from tldw_Server_API.app.core.Utils.metadata_utils import normalize_safe_metadata
        sm = normalize_safe_metadata({
            "pmid": pmid,
            "pmcid": pmcid,
            "doi": (meta.get('externalIds') or {}).get('DOI') if isinstance(meta.get('externalIds'), dict) else meta.get('doi'),
            "title": meta.get('title'),
            "authors": ', '.join(a.get('name') for a in (meta.get('authors') or []) if a.get('name')) if isinstance(meta.get('authors'), list) else meta.get('authors'),
            "journal": meta.get('journal'),
            "venue": meta.get('journal'),
            "date": meta.get('pub_date'),
            "pmc_url": meta.get('pmc_url'),
            "pdf_url": meta.get('pdf_url'),
            "source": "pubmed",
        })
        import json as _json
        smj = _json.dumps({k: v for k, v in sm.items() if v}, ensure_ascii=False)
        analysis_for_db = result.get('summary') or result.get('analysis')
        title_for_db = meta.get('title') or f"PMID {pmid}"
        author_for_db = sm.get('authors')
        # Build plaintext chunks for chunk-level FTS if chunking enabled
        chunks_for_sql = None
        try:
            if perform_chunking:
                from tldw_Server_API.app.core.Chunking.chunker import Chunker as _Chunker
                _ck = _Chunker()
                _flat = _ck.chunk_text_hierarchical_flat(
                    content_for_db,
                    method=chunk_method or "sentences",
                    max_size=chunk_size,
                    overlap=chunk_overlap,
                )
                _kind_map = {
                    'paragraph': 'text', 'list_unordered': 'list', 'list_ordered': 'list',
                    'code_fence': 'code', 'table_md': 'table', 'header_line': 'heading', 'header_atx': 'heading'
                }
                chunks_for_sql = []
                for _it in _flat:
                    _md = _it.get('metadata') or {}
                    _ctype = _kind_map.get(str(_md.get('paragraph_kind') or '').lower(), 'text')
                    _small = {}
                    if _md.get('ancestry_titles'):
                        _small['ancestry_titles'] = _md.get('ancestry_titles')
                    if _md.get('section_path'):
                        _small['section_path'] = _md.get('section_path')
                    chunks_for_sql.append({
                        'text': _it.get('text',''),
                        'start_char': _md.get('start_offset'),
                        'end_char': _md.get('end_offset'),
                        'chunk_type': _ctype,
                        'metadata': _small,
                    })
        except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS:
            chunks_for_sql = None

        media_id, media_uuid, msg = await loop.run_in_executor(
            None,
            lambda: _ingest_paper_search_media(
                media_db=db,
                url=f"pmid:{pmid}",
                title=title_for_db,
                media_type="pdf",
                content=content_for_db,
                keywords=kw_list or [],
                prompt=custom_prompt,
                analysis_content=analysis_for_db,
                safe_metadata=smj,
                transcription_model='Imported',
                author=author_for_db,
                overwrite=False,
                chunk_options={"method": chunk_method or "sentences", "max_size": chunk_size, "overlap": chunk_overlap} if perform_chunking else None,
                chunks=chunks_for_sql,
            )
        )
        return {"message": msg, "media_id": media_id, "media_uuid": media_uuid}  # noqa: TRY300
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"PubMed ingest error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="PubMed ingest failed") from e


@router.post(
    "/semantic-scholar/ingest",
    summary="Download Semantic Scholar open-access PDF by paperId, process, and persist",
    tags=["paper-search"],
)
async def s2_ingest(
    paper_id: str = Query(..., description="Semantic Scholar paperId"),
    keywords: Optional[str] = Query(None),
    perform_chunking: bool = Query(True),
    parser: Optional[str] = Query("pymupdf4llm"),
    chunk_method: Optional[str] = Query(None),
    chunk_size: int = Query(500, ge=50, le=4000),
    chunk_overlap: int = Query(200, ge=0, le=1000),
    perform_analysis: bool = Query(True),
    custom_prompt: Optional[str] = Query(None),
    system_prompt: Optional[str] = Query(None),
    api_name: Optional[str] = Query(None),
    enable_ocr: bool = Query(False),
    ocr_backend: Optional[str] = Query(None),
    ocr_lang: Optional[str] = Query("eng"),
    ocr_dpi: int = Query(300, ge=72, le=600),
    ocr_mode: Optional[str] = Query("fallback"),
    ocr_min_page_text_chars: int = Query(40, ge=0, le=2000),
    ocr_output_format: Optional[str] = Query(None, description="OCR output format: text|markdown|json"),
    ocr_prompt_preset: Optional[str] = Query(None, description="OCR prompt preset (e.g., 'general', 'doc', 'table', 'spotting', 'json')"),
    db: MediaDbSession = Depends(get_media_db_for_user),
):
    loop = asyncio.get_running_loop()
    try:
        """Download an open-access PDF from Semantic Scholar, process it, and persist.

        Example:
          POST /api/v1/paper-search/semantic-scholar/ingest?paper_id=abcdef&perform_analysis=true
        """
        meta, err = Semantic_Scholar.get_paper_details_semantic_scholar(paper_id)
        if err:
            raise HTTPException(status_code=502, detail=_PROVIDER_ERROR_DETAIL)  # noqa: TRY301
        if not meta:
            raise HTTPException(status_code=404, detail="paperId not found")  # noqa: TRY301
        oap = meta.get('openAccessPdf') or {}
        pdf_url = oap.get('url')
        if not pdf_url:
            raise HTTPException(status_code=400, detail="No open access PDF available for this paper")  # noqa: TRY301
        content = await _download_pdf_bytes(pdf_url)

        from tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.PDF_Processing_Lib import process_pdf_task
        kw_list = [k.strip() for k in (keywords or '').split(',') if k.strip()] if keywords else None
        result = await process_pdf_task(
            file_bytes=content,
            filename=f"{paper_id}.pdf",
            parser=parser or "pymupdf4llm",
            keywords=kw_list,
            perform_chunking=perform_chunking,
            chunk_method=chunk_method,
            max_chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            perform_analysis=perform_analysis,
            api_name=api_name,
            custom_prompt=custom_prompt,
            system_prompt=system_prompt,
            enable_ocr=enable_ocr or None,
            ocr_backend=ocr_backend or None,
            ocr_lang=ocr_lang or None,
            ocr_dpi=ocr_dpi,
            ocr_mode=ocr_mode,
            ocr_min_page_text_chars=ocr_min_page_text_chars,
            ocr_output_format=ocr_output_format,
            ocr_prompt_preset=ocr_prompt_preset,
        )
        content_for_db = result.get('transcript') or result.get('content')
        if not content_for_db:
            raise HTTPException(status_code=500, detail="Processing did not produce content")  # noqa: TRY301
        from tldw_Server_API.app.core.Utils.metadata_utils import normalize_safe_metadata
        sm = normalize_safe_metadata({
            "s2_paper_id": paper_id,
            "title": meta.get('title'),
            "authors": ', '.join(a.get('name') for a in (meta.get('authors') or []) if a.get('name')) if isinstance(meta.get('authors'), list) else None,
            "venue": meta.get('venue'),
            "date": meta.get('publicationDate') or meta.get('year'),
            "doi": (meta.get('externalIds') or {}).get('DOI') if isinstance(meta.get('externalIds'), dict) else None,
            "pdf_url": pdf_url,
            "source": "semantic_scholar",
        })
        import json as _json
        smj = _json.dumps({k: v for k, v in sm.items() if v}, ensure_ascii=False)
        analysis_for_db = result.get('summary') or result.get('analysis')
        title_for_db = meta.get('title') or f"S2 {paper_id}"
        author_for_db = sm.get('authors')
        media_id, media_uuid, msg = await loop.run_in_executor(
            None,
            lambda: _ingest_paper_search_media(
                media_db=db,
                url=f"s2:{paper_id}",
                title=title_for_db,
                media_type="pdf",
                content=content_for_db,
                keywords=kw_list or [],
                prompt=custom_prompt,
                analysis_content=analysis_for_db,
                safe_metadata=smj,
                transcription_model='Imported',
                author=author_for_db,
                overwrite=False,
                chunk_options={"method": chunk_method or "sentences", "max_size": chunk_size, "overlap": chunk_overlap} if perform_chunking else None,
            )
        )
        return {"message": msg, "media_id": media_id, "media_uuid": media_uuid}  # noqa: TRY300
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Semantic Scholar ingest error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Semantic Scholar ingest failed") from e


@router.get(
    "/biorxiv/by-doi",
    response_model=BioRxivPaper,
    summary="Get BioRxiv/MedRxiv paper by DOI",
    tags=["paper-search"],
)
async def paper_search_biorxiv_by_doi(
    doi: str = Query(..., description="Preprint DOI (e.g., 10.1101/2021.11.09.467936)"),
    server: str = Query("biorxiv", description="Server: biorxiv or medrxiv"),
):
    """Fetch a single preprint by DOI."""
    loop = asyncio.get_running_loop()
    try:
        item, error_message = await loop.run_in_executor(
            None,
            BioRxiv.get_biorxiv_by_doi,
            doi,
            server,
        )
        if error_message:
            logger.error(f"BioRxiv DOI provider error: {error_message}")
            if "timed out" in error_message.lower():
                raise HTTPException(status_code=504, detail=_PROVIDER_TIMEOUT_DETAIL)  # noqa: TRY301
            # Map 404 style errors if provided
            if "HTTP Error: 404" in error_message or "HTTP Error: 400" in error_message:
                raise HTTPException(status_code=404, detail="Paper not found for DOI")  # noqa: TRY301
            raise HTTPException(status_code=502, detail=_PROVIDER_ERROR_DETAIL)  # noqa: TRY301
        if not item:
            raise HTTPException(status_code=404, detail="Paper not found for DOI")  # noqa: TRY301
        return BioRxivPaper(**item)
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected BioRxiv DOI error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_PROVIDER_UNEXPECTED_DETAIL) from e


@router.get(
    "/semantic-scholar",
    response_model=SemanticScholarSearchResponse,
    summary="Search Semantic Scholar",
    tags=["paper-search"],
)
async def paper_search_semantic_scholar(
    search_params: SemanticScholarSearchRequestForm = Depends(),
):
    """Thin wrapper routing to existing Semantic Scholar adapter with consistent envelope."""
    offset = (search_params.page - 1) * search_params.results_per_page
    logger.info(
        f"/paper-search/semantic-scholar: query='{search_params.query}', page={search_params.page}, limit={search_params.results_per_page}"
    )
    loop = asyncio.get_running_loop()
    try:
        api_response_data, error_message = await loop.run_in_executor(
            None,
            Semantic_Scholar.search_papers_semantic_scholar,
            search_params.query,
            offset,
            search_params.results_per_page,
            search_params.fields_of_study_list,
            search_params.publication_types_list,
            search_params.year_range,
            search_params.venue_list,
            search_params.min_citations,
            False,
        )
        if error_message:
            logger.error(f"Semantic Scholar provider error: {error_message}")
            if "timed out" in error_message.lower():
                raise HTTPException(status_code=504, detail=_PROVIDER_TIMEOUT_DETAIL)  # noqa: TRY301
            if "HTTP Error" in error_message:
                # Try to extract status code from error string
                nums = [int(s) for s in error_message.split() if s.isdigit() and 400 <= int(s) < 600]
                code = nums[0] if nums else 502
                raise HTTPException(status_code=code, detail=_PROVIDER_ERROR_DETAIL)  # noqa: TRY301
            raise HTTPException(status_code=502, detail=_PROVIDER_ERROR_DETAIL)  # noqa: TRY301
        if api_response_data is None or "data" not in api_response_data:
            raise HTTPException(status_code=500, detail="Semantic Scholar search returned invalid data.")  # noqa: TRY301
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected Semantic Scholar search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_PROVIDER_UNEXPECTED_DETAIL) from e

    total_results_api = api_response_data.get("total", 0)
    actual_offset_api = api_response_data.get("offset", 0)
    next_offset_api = api_response_data.get("next")

    parsed_papers = []
    for paper_data in api_response_data.get("data", []):
        # Remove openAccessPdf if None to avoid Pydantic error
        if paper_data.get("openAccessPdf") is None:
            paper_data.pop("openAccessPdf", None)
        parsed_papers.append(SemanticScholarPaper(**paper_data))

    total_pages = math.ceil(total_results_api / search_params.results_per_page) if search_params.results_per_page > 0 else 0
    if total_results_api == 0:
        total_pages = 0

    return SemanticScholarSearchResponse(
        query_echo={
            "query": search_params.query,
            "fields_of_study": search_params.fields_of_study_list,
            "publication_types": search_params.publication_types_list,
            "year_range": search_params.year_range,
            "venue": search_params.venue_list,
            "min_citations": search_params.min_citations,
        },
        items=parsed_papers,
        total_results=total_results_api,
        offset=actual_offset_api,
        limit=search_params.results_per_page,
        next_offset=next_offset_api,
        page=search_params.page,
        total_pages=total_pages,
        pagination=build_page_pagination_meta(
            page=search_params.page,
            per_page=search_params.results_per_page,
            total=total_results_api,
            total_pages=total_pages,
        ),
    )


@router.get(
    "/biorxiv-pubs",
    response_model=BioRxivPubsSearchResponse,
    summary="Search published article metadata (bioRxiv/medRxiv)",
    tags=["paper-search"],
)
async def paper_search_biorxiv_pubs(
    search_params: BioRxivPubsSearchRequestForm = Depends(),
):
    offset = (search_params.page - 1) * search_params.results_per_page
    logger.info(
        f"/paper-search/biorxiv-pubs: server='{search_params.server}', page={search_params.page}, size={search_params.results_per_page}"
    )
    loop = asyncio.get_running_loop()
    try:
        items, total_results, error_message = await loop.run_in_executor(
            None,
            BioRxiv.search_biorxiv_pubs,
            search_params.server,
            search_params.from_date,
            search_params.to_date,
            offset,
            search_params.results_per_page,
            search_params.recent_days,
            search_params.recent_count,
            search_params.q,
        )
        if error_message:
            logger.error(f"BioRxiv pubs error: {error_message}")
            if "timed out" in error_message.lower():
                raise HTTPException(status_code=504, detail=_PROVIDER_TIMEOUT_DETAIL)  # noqa: TRY301
            raise HTTPException(status_code=502, detail=_PROVIDER_ERROR_DETAIL)  # noqa: TRY301
        if items is None:
            raise HTTPException(status_code=500, detail="BioRxiv pubs search failed to return data.")  # noqa: TRY301
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected BioRxiv pubs search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_PROVIDER_UNEXPECTED_DETAIL) from e

    total_pages = math.ceil(total_results / search_params.results_per_page) if search_params.results_per_page > 0 else 0
    if total_results == 0:
        total_pages = 0
    # Optionally strip abstracts for compact responses
    if not search_params.include_abstracts:
        for it in items:
            if isinstance(it, dict):
                it["preprint_abstract"] = None

    return BioRxivPubsSearchResponse(
        query_echo={
            "server": search_params.server,
            "from_date": search_params.from_date,
            "to_date": search_params.to_date,
            "recent_days": search_params.recent_days,
            "recent_count": search_params.recent_count,
            "q": search_params.q,
            "include_abstracts": search_params.include_abstracts,
        },
        items=[BioRxivPublishedRecord(**it) for it in items],
        total_results=total_results,
        page=search_params.page,
        results_per_page=search_params.results_per_page,
        total_pages=total_pages,
        pagination=build_page_pagination_meta(
            page=search_params.page,
            per_page=search_params.results_per_page,
            total=total_results,
            total_pages=total_pages,
        ),
    )


@router.get(
    "/pubmed",
    response_model=PubMedSearchResponse,
    summary="Search PubMed",
    tags=["paper-search"],
)
async def paper_search_pubmed(
    search_params: PubMedSearchRequestForm = Depends(),
):
    """Provider-specific search for PubMed with pagination via E-utilities."""
    offset = (search_params.page - 1) * search_params.results_per_page
    logger.info(
        f"/paper-search/pubmed: q='{search_params.q}', from_year={search_params.from_year}, to_year={search_params.to_year}, free_full_text={search_params.free_full_text}, page={search_params.page}, size={search_params.results_per_page}"
    )
    loop = asyncio.get_running_loop()
    try:
        items, total_results, error_message = await loop.run_in_executor(
            None,
            PubMed.search_pubmed,
            search_params.q,
            offset,
            search_params.results_per_page,
            search_params.from_year,
            search_params.to_year,
            search_params.free_full_text,
        )
        if error_message:
            logger.error(f"PubMed provider error: {error_message}")
            if "timed out" in error_message.lower():
                raise HTTPException(status_code=504, detail=_PROVIDER_TIMEOUT_DETAIL)  # noqa: TRY301
            if "http error" in error_message.lower():
                # Try to extract status code
                nums = [int(s) for s in error_message.split() if s.isdigit() and 400 <= int(s) < 600]
                code = nums[0] if nums else 502
                raise HTTPException(status_code=code, detail=_PROVIDER_ERROR_DETAIL)  # noqa: TRY301
            raise HTTPException(status_code=502, detail=_PROVIDER_ERROR_DETAIL)  # noqa: TRY301
        if items is None:
            raise HTTPException(status_code=500, detail="PubMed search failed to return data.")  # noqa: TRY301
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected PubMed search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_PROVIDER_UNEXPECTED_DETAIL) from e

    total_pages = math.ceil(total_results / search_params.results_per_page) if search_params.results_per_page > 0 else 0
    if total_results == 0:
        total_pages = 0

    return PubMedSearchResponse(
        query_echo={
            "q": search_params.q,
            "from_year": search_params.from_year,
            "to_year": search_params.to_year,
            "free_full_text": search_params.free_full_text,
        },
        items=[PubMedPaper(**it) for it in items],
        total_results=total_results,
        page=search_params.page,
        results_per_page=search_params.results_per_page,
        total_pages=total_pages,
        pagination=build_page_pagination_meta(
            page=search_params.page,
            per_page=search_params.results_per_page,
            total=total_results,
            total_pages=total_pages,
        ),
    )


@router.get(
    "/biorxiv-pubs/by-doi",
    response_model=BioRxivPublishedRecord,
    summary="Get published metadata by DOI (bioRxiv/medRxiv)",
    tags=["paper-search"],
)
async def paper_search_biorxiv_pubs_by_doi(
    doi: str = Query(..., description="Preprint or published DOI"),
    server: str = Query("biorxiv", description="Server: biorxiv or medrxiv"),
    include_abstracts: bool = Query(True, description="Include preprint_abstract field in result"),
):
    loop = asyncio.get_running_loop()
    try:
        item, error_message = await loop.run_in_executor(
            None,
            BioRxiv.get_biorxiv_published_by_doi,
            doi,
            server,
        )
        if error_message:
            logger.error(f"BioRxiv pubs by-doi error: {error_message}")
            if "timed out" in error_message.lower():
                raise HTTPException(status_code=504, detail=_PROVIDER_TIMEOUT_DETAIL)  # noqa: TRY301
        if not item:
            raise HTTPException(status_code=404, detail="Published record not found for DOI")  # noqa: TRY301
        if not include_abstracts and isinstance(item, dict):
            item["preprint_abstract"] = None
        return BioRxivPublishedRecord(**item)
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected BioRxiv pubs by-doi error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_PROVIDER_UNEXPECTED_DETAIL) from e


# End of paper_search.py

# -------------------- BioRxiv Additional Reports Endpoints --------------------

@router.get(
    "/biorxiv/publisher",
    response_model=BioRxivPubsSearchResponse,
    summary="Search bioRxiv publisher mapping (pubs) by publisher prefix",
    tags=["paper-search"],
)
async def paper_search_biorxiv_publisher(
    search_params: BioRxivPublisherSearchRequestForm = Depends(),
):
    offset = (search_params.page - 1) * search_params.results_per_page
    loop = asyncio.get_running_loop()
    try:
        items, total, err = await loop.run_in_executor(
            None,
            BioRxiv.search_biorxiv_publisher,
            search_params.publisher_prefix,
            search_params.from_date,
            search_params.to_date,
            offset,
            search_params.results_per_page,
            search_params.recent_days,
            search_params.recent_count,
        )
        if err:
            logger.error(f"BioRxiv publisher error: {err}")
            if "timed out" in err.lower():
                raise HTTPException(status_code=504, detail=_PROVIDER_TIMEOUT_DETAIL)  # noqa: TRY301
            raise HTTPException(status_code=502, detail=_PROVIDER_ERROR_DETAIL)  # noqa: TRY301
        if items is None:
            raise HTTPException(status_code=500, detail="BioRxiv publisher search failed to return data.")  # noqa: TRY301
        total_pages = math.ceil(total / search_params.results_per_page) if search_params.results_per_page > 0 else 0
        if total == 0:
            total_pages = 0
        return BioRxivPubsSearchResponse(
            query_echo={
                "publisher_prefix": search_params.publisher_prefix,
                "from_date": search_params.from_date,
                "to_date": search_params.to_date,
                "recent_days": search_params.recent_days,
                "recent_count": search_params.recent_count,
            },
            items=[BioRxivPublishedRecord(**it) for it in items],
            total_results=total,
            page=search_params.page,
            results_per_page=search_params.results_per_page,
            total_pages=total_pages,
            pagination=build_page_pagination_meta(
                page=search_params.page,
                per_page=search_params.results_per_page,
                total=total,
                total_pages=total_pages,
            ),
        )
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected BioRxiv publisher search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_PROVIDER_UNEXPECTED_DETAIL) from e


@router.get(
    "/biorxiv/pub",
    response_model=BioRxivPubsSearchResponse,
    summary="Search bioRxiv published article detail (pub)",
    tags=["paper-search"],
)
async def paper_search_biorxiv_pub(
    search_params: BioRxivPubSearchRequestForm = Depends(),
):
    offset = (search_params.page - 1) * search_params.results_per_page
    loop = asyncio.get_running_loop()
    try:
        items, total, err = await loop.run_in_executor(
            None,
            BioRxiv.search_biorxiv_pub,
            search_params.from_date,
            search_params.to_date,
            offset,
            search_params.results_per_page,
            search_params.recent_days,
            search_params.recent_count,
        )
        if err:
            logger.error(f"BioRxiv pub error: {err}")
            if "timed out" in err.lower():
                raise HTTPException(status_code=504, detail=_PROVIDER_TIMEOUT_DETAIL)  # noqa: TRY301
            raise HTTPException(status_code=502, detail=_PROVIDER_ERROR_DETAIL)  # noqa: TRY301
        if items is None:
            raise HTTPException(status_code=500, detail="BioRxiv pub search failed to return data.")  # noqa: TRY301
        total_pages = math.ceil(total / search_params.results_per_page) if search_params.results_per_page > 0 else 0
        if total == 0:
            total_pages = 0
        return BioRxivPubsSearchResponse(
            query_echo={
                "from_date": search_params.from_date,
                "to_date": search_params.to_date,
                "recent_days": search_params.recent_days,
                "recent_count": search_params.recent_count,
            },
            items=[BioRxivPublishedRecord(**it) for it in items],
            total_results=total,
            page=search_params.page,
            results_per_page=search_params.results_per_page,
            total_pages=total_pages,
            pagination=build_page_pagination_meta(
                page=search_params.page,
                per_page=search_params.results_per_page,
                total=total,
                total_pages=total_pages,
            ),
        )
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected BioRxiv pub search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_PROVIDER_UNEXPECTED_DETAIL) from e


@router.get(
    "/biorxiv/funder",
    response_model=BioRxivFunderSearchResponse,
    summary="Search bioRxiv/medRxiv by funder ROR ID",
    tags=["paper-search"],
)
async def paper_search_biorxiv_funder(
    search_params: BioRxivFunderSearchRequestForm = Depends(),
):
    offset = (search_params.page - 1) * search_params.results_per_page
    loop = asyncio.get_running_loop()
    try:
        items, total, err = await loop.run_in_executor(
            None,
            BioRxiv.search_biorxiv_funder,
            search_params.server,
            search_params.ror_id,
            search_params.from_date,
            search_params.to_date,
            offset,
            search_params.results_per_page,
            search_params.recent_days,
            search_params.recent_count,
            search_params.category,
        )
        if err:
            logger.error(f"BioRxiv funder error: {err}")
            if "timed out" in err.lower():
                raise HTTPException(status_code=504, detail=_PROVIDER_TIMEOUT_DETAIL)  # noqa: TRY301
            raise HTTPException(status_code=502, detail=_PROVIDER_ERROR_DETAIL)  # noqa: TRY301
        if items is None:
            raise HTTPException(status_code=500, detail="BioRxiv funder search failed to return data.")  # noqa: TRY301
        total_pages = math.ceil(total / search_params.results_per_page) if search_params.results_per_page > 0 else 0
        if total == 0:
            total_pages = 0
        return BioRxivFunderSearchResponse(
            query_echo={
                "server": search_params.server,
                "ror_id": search_params.ror_id,
                "from_date": search_params.from_date,
                "to_date": search_params.to_date,
                "recent_days": search_params.recent_days,
                "recent_count": search_params.recent_count,
                "category": search_params.category,
            },
            items=[BioRxivFunderPaper(**it) for it in items],
            total_results=total,
            page=search_params.page,
            results_per_page=search_params.results_per_page,
            total_pages=total_pages,
            pagination=build_page_pagination_meta(
                page=search_params.page,
                per_page=search_params.results_per_page,
                total=total,
                total_pages=total_pages,
            ),
        )
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected BioRxiv funder search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_PROVIDER_UNEXPECTED_DETAIL) from e


@router.get(
    "/biorxiv/reports/summary",
    response_model=BioRxivSummaryResponse,
    summary="Content summary statistics (bioRxiv)",
    tags=["paper-search"],
)
async def paper_search_biorxiv_summary(
    search_params: BioRxivSummaryRequestForm = Depends(),
):
    loop = asyncio.get_running_loop()
    try:
        items, err = await loop.run_in_executor(None, BioRxiv.get_biorxiv_summary, search_params.interval)
        if err:
            logger.error(f"BioRxiv summary error: {err}")
            if "timed out" in err.lower():
                raise HTTPException(status_code=504, detail=_PROVIDER_TIMEOUT_DETAIL)  # noqa: TRY301
            raise HTTPException(status_code=502, detail=_PROVIDER_ERROR_DETAIL)  # noqa: TRY301
        if items is None:
            items = []
        return BioRxivSummaryResponse(query_echo={"interval": search_params.interval}, items=items)  # type: ignore[arg-type]
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected BioRxiv summary error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_PROVIDER_UNEXPECTED_DETAIL) from e


@router.get(
    "/biorxiv/reports/usage",
    response_model=BioRxivUsageResponse,
    summary="Usage summary statistics (bioRxiv)",
    tags=["paper-search"],
)
async def paper_search_biorxiv_usage(
    search_params: BioRxivUsageRequestForm = Depends(),
):
    loop = asyncio.get_running_loop()
    try:
        items, err = await loop.run_in_executor(None, BioRxiv.get_biorxiv_usage, search_params.interval)
        if err:
            logger.error(f"BioRxiv usage error: {err}")
            if "timed out" in err.lower():
                raise HTTPException(status_code=504, detail=_PROVIDER_TIMEOUT_DETAIL)  # noqa: TRY301
            raise HTTPException(status_code=502, detail=_PROVIDER_ERROR_DETAIL)  # noqa: TRY301
        if items is None:
            items = []
        return BioRxivUsageResponse(query_echo={"interval": search_params.interval}, items=items)  # type: ignore[arg-type]
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected BioRxiv usage error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_PROVIDER_UNEXPECTED_DETAIL) from e


# -------------------- BioRxiv Raw Passthrough Endpoints (CSV/XML) --------------------

@router.get(
    "/biorxiv/raw/details",
    summary="Raw passthrough for BioRxiv details endpoint (supports json/xml/html)",
    response_class=Response,
    responses=_RAW_JSON_XML_HTML_RESPONSES,
    tags=["paper-search"],
)
async def biorxiv_raw_details(
    server: str = Query("biorxiv"),
    from_date: Optional[str] = Query(None),
    to_date: Optional[str] = Query(None),
    recent_days: Optional[int] = Query(None),
    recent_count: Optional[int] = Query(None),
    doi: Optional[str] = Query(None),
    cursor: int = Query(0, ge=0),
    category: Optional[str] = Query(None),
    format: str = Query("json", description="json|xml|html"),
):
    loop = asyncio.get_running_loop()
    content, media_type, err = await loop.run_in_executor(
        None,
        BioRxiv.raw_details,
        server,
        from_date,
        to_date,
        recent_days,
        recent_count,
        doi,
        cursor,
        category,
        format,
    )
    if err:
        _handle_provider_error(err)
    if not content:
        raise HTTPException(status_code=404, detail="No content")
    return Response(content=content, media_type=media_type or "application/octet-stream")


@router.get(
    "/biorxiv/raw/pubs",
    summary="Raw passthrough for BioRxiv pubs endpoint (supports json/csv)",
    response_class=Response,
    responses=_RAW_JSON_CSV_RESPONSES,
    tags=["paper-search"],
)
async def biorxiv_raw_pubs(
    server: str = Query("biorxiv"),
    from_date: Optional[str] = Query(None),
    to_date: Optional[str] = Query(None),
    recent_days: Optional[int] = Query(None),
    recent_count: Optional[int] = Query(None),
    doi: Optional[str] = Query(None),
    cursor: int = Query(0, ge=0),
    format: str = Query("json", description="json|csv"),
):
    loop = asyncio.get_running_loop()
    content, media_type, err = await loop.run_in_executor(
        None,
        BioRxiv.raw_pubs,
        server,
        from_date,
        to_date,
        recent_days,
        recent_count,
        doi,
        cursor,
        format,
    )
    if err:
        _handle_provider_error(err)
    if not content:
        raise HTTPException(status_code=404, detail="No content")
    return Response(content=content, media_type=media_type or "application/octet-stream")


@router.get(
    "/biorxiv/raw/pub",
    summary="Raw passthrough for BioRxiv pub endpoint (supports json/csv)",
    response_class=Response,
    responses=_RAW_JSON_CSV_RESPONSES,
    tags=["paper-search"],
)
async def biorxiv_raw_pub(
    from_date: Optional[str] = Query(None),
    to_date: Optional[str] = Query(None),
    recent_days: Optional[int] = Query(None),
    recent_count: Optional[int] = Query(None),
    cursor: int = Query(0, ge=0),
    format: str = Query("json", description="json|csv"),
):
    loop = asyncio.get_running_loop()
    content, media_type, err = await loop.run_in_executor(
        None,
        BioRxiv.raw_pub,
        from_date,
        to_date,
        recent_days,
        recent_count,
        cursor,
        format,
    )
    if err:
        _handle_provider_error(err)
    if not content:
        raise HTTPException(status_code=404, detail="No content")
    return Response(content=content, media_type=media_type or "application/octet-stream")


@router.get(
    "/biorxiv/raw/funder",
    summary="Raw passthrough for BioRxiv funder endpoint (supports json/xml)",
    response_class=Response,
    responses=_RAW_JSON_XML_RESPONSES,
    tags=["paper-search"],
)
async def biorxiv_raw_funder(
    server: str = Query("biorxiv"),
    ror_id: str = Query(...),
    from_date: Optional[str] = Query(None),
    to_date: Optional[str] = Query(None),
    recent_days: Optional[int] = Query(None),
    recent_count: Optional[int] = Query(None),
    cursor: int = Query(0, ge=0),
    category: Optional[str] = Query(None),
    format: str = Query("json", description="json|xml"),
):
    loop = asyncio.get_running_loop()
    content, media_type, err = await loop.run_in_executor(
        None,
        BioRxiv.raw_funder,
        server,
        ror_id,
        from_date,
        to_date,
        recent_days,
        recent_count,
        cursor,
        category,
        format,
    )
    if err:
        _handle_provider_error(err)
    if not content:
        raise HTTPException(status_code=404, detail="No content")
    return Response(content=content, media_type=media_type or "application/octet-stream")


@router.get(
    "/biorxiv/raw/reports/summary",
    summary="Raw passthrough for BioRxiv summary (supports json/csv)",
    response_class=Response,
    responses=_RAW_JSON_CSV_RESPONSES,
    tags=["paper-search"],
)
async def biorxiv_raw_summary(
    interval: str = Query("m"),
    format: str = Query("json", description="json|csv"),
):
    loop = asyncio.get_running_loop()
    content, media_type, err = await loop.run_in_executor(None, BioRxiv.raw_sum, interval, format)
    if err:
        _handle_provider_error(err)
    if not content:
        raise HTTPException(status_code=404, detail="No content")
    return Response(content=content, media_type=media_type or "application/octet-stream")


@router.get(
    "/biorxiv/raw/reports/usage",
    summary="Raw passthrough for BioRxiv usage (supports json/csv)",
    response_class=Response,
    responses=_RAW_JSON_CSV_RESPONSES,
    tags=["paper-search"],
)
async def biorxiv_raw_usage(
    interval: str = Query("m"),
    format: str = Query("json", description="json|csv"),
):
    loop = asyncio.get_running_loop()
    content, media_type, err = await loop.run_in_executor(None, BioRxiv.raw_usage, interval, format)
    if err:
        _handle_provider_error(err)
    if not content:
        raise HTTPException(status_code=404, detail="No content")
    return Response(content=content, media_type=media_type or "application/octet-stream")

# -------------------- Additional Provider Endpoints (Scaffold) --------------------

from tldw_Server_API.app.api.v1.schemas.paper_search_schemas import (  # noqa: E402
    ChemRxivSearchRequestForm,
    DOIRequestForm,
    GenericPaper,
    GenericPageSearchResponse,
    GenericSearchResponse,
    IacrConferenceResponse,
    IEEESearchRequestForm,
    IngestBatchRequest,
    IngestBatchResponse,
    IngestBatchResultItem,
    SimpleVenueSearchForm,
)
from tldw_Server_API.app.core.Third_Party import HAL as HAL  # noqa: E402
from tldw_Server_API.app.core.Third_Party import IACR as IACR  # noqa: E402
from tldw_Server_API.app.core.Third_Party import OSF as OSF  # noqa: E402
from tldw_Server_API.app.core.Third_Party import ChemRxiv as ChemRxiv  # noqa: E402
from tldw_Server_API.app.core.Third_Party import Crossref as Crossref  # noqa: E402
from tldw_Server_API.app.core.Third_Party import EarthRxiv as EarthRxiv  # noqa: E402
from tldw_Server_API.app.core.Third_Party import Elsevier_Scopus as Elsevier_Scopus  # noqa: E402
from tldw_Server_API.app.core.Third_Party import Figshare as Figshare  # noqa: E402
from tldw_Server_API.app.core.Third_Party import IEEE_Xplore as IEEE_Xplore  # noqa: E402
from tldw_Server_API.app.core.Third_Party import OpenAlex as OpenAlex  # noqa: E402
from tldw_Server_API.app.core.Third_Party import RePEc as RePEc  # noqa: E402
from tldw_Server_API.app.core.Third_Party import Springer_Nature as Springer_Nature  # noqa: E402
from tldw_Server_API.app.core.Third_Party import Unpaywall as Unpaywall  # noqa: E402
from tldw_Server_API.app.core.Third_Party import Vixra as Vixra  # noqa: E402
from tldw_Server_API.app.core.Third_Party import Zenodo as Zenodo  # noqa: E402


def _extract_http_status(err: str) -> Optional[int]:
    """Extract HTTP status code from provider error strings.

    Recognizes common formats like:
    - "HTTP Error 404" or "error 404"
    - "status code: 502" or "status code 502"
    - "status 500"

    Args:
        err: Error message string from a provider adapter.

    Returns:
        Extracted HTTP status code (400-599) or None if not found.
        Returns 502 if an extracted code is outside the 4xx/5xx range.
    """
    patterns = (
        r"(?:http\s+)?error\s*[:\s]*(\d{3})",
        r"status\s*code\s*[:\s]*(\d{3})",
        r"status\s+(\d{3})",
    )
    for pattern in patterns:
        match = re.search(pattern, err, re.IGNORECASE)
        if not match:
            continue
        try:
            code = int(match.group(1))
        except (TypeError, ValueError):
            continue
        if 400 <= code < 600:
            return code
        logger.warning(
            "Extracted HTTP status {} outside 4xx/5xx range; using 502",
            match.group(1),
        )
        return 502
    return None


def _handle_provider_error(err: str) -> None:
    """Normalize provider error messages into consistent HTTP responses.

    Recognizes formats like "HTTP Error 404", "status 404 Not Found", or
    "status code: 502" when extracting provider status codes.
    """
    if not err:
        return
    low = err.lower()
    logger.error("Provider error: {}", err)
    if "not configured" in low:
        raise HTTPException(
            status_code=501,
            detail=_PROVIDER_NOT_CONFIGURED_DETAIL,
        )
    if "timed out" in low:
        raise HTTPException(
            status_code=504,
            detail=_PROVIDER_TIMEOUT_DETAIL,
        )
    code = _extract_http_status(err)
    if code is not None:
        raise HTTPException(
            status_code=code,
            detail=_PROVIDER_ERROR_DETAIL,
        )
    raise HTTPException(
        status_code=502,
        detail=_PROVIDER_ERROR_DETAIL,
    )


# ---------------- RePEc / CitEc Endpoints ----------------

from tldw_Server_API.app.api.v1.schemas.paper_search_schemas import (  # noqa: E402
    RepecCitationsResponse,
)


@router.get(
    "/repec/by-handle",
    response_model=GenericPaper,
    summary="Get RePEc record by handle (IDEAS getref)",
    tags=["paper-search"],
)
async def repec_by_handle(handle: str = Query(..., min_length=8)):
    """Lookup a RePEc item by its handle via IDEAS API.

    Requires `REPEC_API_CODE` to be configured in the environment.
    """
    loop = asyncio.get_running_loop()
    try:
        item, err = await loop.run_in_executor(None, RePEc.get_ref_by_handle, handle)
        if err:
            _handle_provider_error(err)
        if not item:
            raise HTTPException(status_code=404, detail="RePEc handle not found")  # noqa: TRY301
        return GenericPaper(**item)
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected RePEc by-handle error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_PROVIDER_UNEXPECTED_DETAIL) from e


@router.get(
    "/repec/citations",
    response_model=RepecCitationsResponse,
    summary="Get CitEc citation summary for a RePEc handle",
    tags=["paper-search"],
)
async def repec_citations(handle: str = Query(..., min_length=8)):
    loop = asyncio.get_running_loop()
    try:
        data, err = await loop.run_in_executor(None, RePEc.get_citations_plain, handle)
        if err:
            _handle_provider_error(err)
        if not data:
            raise HTTPException(status_code=404, detail="CitEc record not found for handle")  # noqa: TRY301
        return RepecCitationsResponse(**data)
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected RePEc/CitEc citations error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_PROVIDER_UNEXPECTED_DETAIL) from e


@router.get(
    "/ieee",
    response_model=GenericPageSearchResponse,
    summary="Search IEEE Xplore (scaffold)",
    tags=["paper-search"],
)
async def paper_search_ieee(search_params: IEEESearchRequestForm = Depends()):
    offset = (search_params.page - 1) * search_params.results_per_page
    loop = asyncio.get_running_loop()
    try:
        items, total, err = await loop.run_in_executor(
            None,
            IEEE_Xplore.search_ieee,
            search_params.q,
            offset,
            search_params.results_per_page,
            search_params.from_year,
            search_params.to_year,
            search_params.publication_title,
            search_params.authors,
        )
        if err:
            _handle_provider_error(err)
        if items is None:
            raise HTTPException(status_code=500, detail="IEEE search failed to return data.")  # noqa: TRY301
        total_pages = math.ceil(total / search_params.results_per_page) if search_params.results_per_page > 0 else 0
        if total == 0:
            total_pages = 0
        return GenericPageSearchResponse(
            query_echo={
                "q": search_params.q,
                "from_year": search_params.from_year,
                "to_year": search_params.to_year,
                "publication_title": search_params.publication_title,
                "authors": search_params.authors,
            },
            items=[GenericPaper(**it) for it in items],
            total_results=total,
            page=search_params.page,
            results_per_page=search_params.results_per_page,
            total_pages=total_pages,
            pagination=build_page_pagination_meta(
                page=search_params.page,
                per_page=search_params.results_per_page,
                total=total,
                total_pages=total_pages,
            ),
        )
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected IEEE search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_PROVIDER_UNEXPECTED_DETAIL) from e


@router.get(
    "/ieee/by-doi",
    response_model=GenericPaper,
    summary="Get IEEE Xplore by DOI (scaffold)",
    tags=["paper-search"],
)
async def paper_search_ieee_by_doi(params: DOIRequestForm = Depends()):
    loop = asyncio.get_running_loop()
    try:
        item, err = await loop.run_in_executor(None, IEEE_Xplore.get_ieee_by_doi, params.doi)
        if err:
            _handle_provider_error(err)
        if not item:
            raise HTTPException(status_code=404, detail="IEEE paper not found for DOI")  # noqa: TRY301
        return GenericPaper(**item)
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected IEEE by-doi error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_PROVIDER_UNEXPECTED_DETAIL) from e


@router.get(
    "/ieee/by-id",
    response_model=GenericPaper,
    summary="Get IEEE Xplore by article number (scaffold)",
    tags=["paper-search"],
)
async def paper_search_ieee_by_id(article_number: str = Query(...)):
    loop = asyncio.get_running_loop()
    try:
        item, err = await loop.run_in_executor(None, IEEE_Xplore.get_ieee_by_id, article_number)
        if err:
            _handle_provider_error(err)
        if not item:
            raise HTTPException(status_code=404, detail="IEEE paper not found for article number")  # noqa: TRY301
        return GenericPaper(**item)
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected IEEE by-id error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_PROVIDER_UNEXPECTED_DETAIL) from e


@router.get(
    "/springer",
    response_model=GenericPageSearchResponse,
    summary="Search Springer Nature (scaffold)",
    tags=["paper-search"],
)
async def paper_search_springer(search_params: SimpleVenueSearchForm = Depends()):
    offset = (search_params.page - 1) * search_params.results_per_page
    loop = asyncio.get_running_loop()
    try:
        items, total, err = await loop.run_in_executor(
            None,
            Springer_Nature.search_springer,
            search_params.q,
            offset,
            search_params.results_per_page,
            search_params.venue,
            search_params.from_year,
            search_params.to_year,
        )
        if err:
            _handle_provider_error(err)
        if items is None:
            raise HTTPException(status_code=500, detail="Springer search failed to return data.")  # noqa: TRY301
        total_pages = math.ceil(total / search_params.results_per_page) if search_params.results_per_page > 0 else 0
        if total == 0:
            total_pages = 0
        return GenericPageSearchResponse(
            query_echo={
                "q": search_params.q,
                "journal": search_params.venue,
                "from_year": search_params.from_year,
                "to_year": search_params.to_year,
            },
            items=[GenericPaper(**it) for it in items],
            total_results=total,
            page=search_params.page,
            results_per_page=search_params.results_per_page,
            total_pages=total_pages,
            pagination=build_page_pagination_meta(
                page=search_params.page,
                per_page=search_params.results_per_page,
                total=total,
                total_pages=total_pages,
            ),
        )
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected Springer search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_PROVIDER_UNEXPECTED_DETAIL) from e


@router.get(
    "/springer/by-doi",
    response_model=GenericPaper,
    summary="Get Springer Nature by DOI (scaffold)",
    tags=["paper-search"],
)
async def paper_search_springer_by_doi(params: DOIRequestForm = Depends()):
    loop = asyncio.get_running_loop()
    try:
        item, err = await loop.run_in_executor(None, Springer_Nature.get_springer_by_doi, params.doi)
        if err:
            _handle_provider_error(err)
        if not item:
            raise HTTPException(status_code=404, detail="Springer paper not found for DOI")  # noqa: TRY301
        return GenericPaper(**item)
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected Springer by-doi error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_PROVIDER_UNEXPECTED_DETAIL) from e


@router.get(
    "/scopus",
    response_model=GenericPageSearchResponse,
    summary="Search Elsevier Scopus (scaffold)",
    tags=["paper-search"],
)
async def paper_search_scopus(
    q: Optional[str] = Query(None),
    from_year: Optional[int] = Query(None, ge=1800, le=2100),
    to_year: Optional[int] = Query(None, ge=1800, le=2100),
    open_access_only: bool = Query(False),
    page: int = Query(1, ge=1),
    results_per_page: int = Query(10, ge=1, le=100),
):
    offset = (page - 1) * results_per_page
    loop = asyncio.get_running_loop()
    try:
        items, total, err = await loop.run_in_executor(
            None,
            Elsevier_Scopus.search_scopus,
            q,
            offset,
            results_per_page,
            from_year,
            to_year,
            open_access_only,
        )
        if err:
            _handle_provider_error(err)
        if items is None:
            raise HTTPException(status_code=500, detail="Scopus search failed to return data.")  # noqa: TRY301
        total_pages = math.ceil(total / results_per_page) if results_per_page > 0 else 0
        if total == 0:
            total_pages = 0
        return GenericPageSearchResponse(
            query_echo={
                "q": q,
                "from_year": from_year,
                "to_year": to_year,
                "open_access_only": open_access_only,
            },
            items=[GenericPaper(**it) for it in items],
            total_results=total,
            page=page,
            results_per_page=results_per_page,
            total_pages=total_pages,
            pagination=build_page_pagination_meta(
                page=page,
                per_page=results_per_page,
                total=total,
                total_pages=total_pages,
            ),
        )
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected Scopus search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_PROVIDER_UNEXPECTED_DETAIL) from e


@router.get(
    "/scopus/by-doi",
    response_model=GenericPaper,
    summary="Get Scopus by DOI (scaffold)",
    tags=["paper-search"],
)
async def paper_search_scopus_by_doi(params: DOIRequestForm = Depends()):
    loop = asyncio.get_running_loop()
    try:
        item, err = await loop.run_in_executor(None, Elsevier_Scopus.get_scopus_by_doi, params.doi)
        if err:
            _handle_provider_error(err)
        if not item:
            raise HTTPException(status_code=404, detail="Scopus record not found for DOI")  # noqa: TRY301
        return GenericPaper(**item)
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected Scopus by-doi error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_PROVIDER_UNEXPECTED_DETAIL) from e


@router.get(
    "/acm",
    response_model=GenericPageSearchResponse,
    summary="Search ACM Digital Library via aggregators (scaffold)",
    tags=["paper-search"],
)
async def paper_search_acm(search_params: SimpleVenueSearchForm = Depends()):
    offset = (search_params.page - 1) * search_params.results_per_page
    loop = asyncio.get_running_loop()
    try:
        items, total, err = await loop.run_in_executor(
            None,
            OpenAlex.search_openalex,
            search_params.q,
            offset,
            search_params.results_per_page,
            search_params.venue or "ACM",
            search_params.from_year,
            search_params.to_year,
        )
        if err:
            # Degrade gracefully (e.g., OpenAlex 403) by returning empty results
            logger.warning(f"ACM search upstream error; returning empty results: {err}")
            items, total = [], 0
        if items is None:
            items, total = [], 0
        total_pages = math.ceil(total / search_params.results_per_page) if search_params.results_per_page > 0 else 0
        if total == 0:
            total_pages = 0
        return GenericPageSearchResponse(
            query_echo={
                "q": search_params.q,
                "venue": search_params.venue or "ACM",
                "from_year": search_params.from_year,
                "to_year": search_params.to_year,
            },
            items=[GenericPaper(**it) for it in items],
            total_results=total,
            page=search_params.page,
            results_per_page=search_params.results_per_page,
            total_pages=total_pages,
            pagination=build_page_pagination_meta(
                page=search_params.page,
                per_page=search_params.results_per_page,
                total=total,
                total_pages=total_pages,
            ),
        )
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected ACM search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_PROVIDER_UNEXPECTED_DETAIL) from e


@router.get(
    "/acm/by-doi",
    response_model=GenericPaper,
    summary="Get ACM via Crossref/OpenAlex by DOI (scaffold)",
    tags=["paper-search"],
)
async def paper_search_acm_by_doi(params: DOIRequestForm = Depends()):
    loop = asyncio.get_running_loop()
    try:
        item, err = await loop.run_in_executor(None, Crossref.get_crossref_by_doi, params.doi)
        if err:
            _handle_provider_error(err)
        if not item:
            raise HTTPException(status_code=404, detail="ACM record not found for DOI")  # noqa: TRY301
        return GenericPaper(**item)
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected ACM by-doi error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_PROVIDER_UNEXPECTED_DETAIL) from e


@router.get(
    "/wiley",
    response_model=GenericPageSearchResponse,
    summary="Search Wiley via aggregators (scaffold)",
    tags=["paper-search"],
)
async def paper_search_wiley(search_params: SimpleVenueSearchForm = Depends()):
    offset = (search_params.page - 1) * search_params.results_per_page
    loop = asyncio.get_running_loop()
    try:
        items, total, err = await loop.run_in_executor(
            None,
            OpenAlex.search_openalex,
            search_params.q,
            offset,
            search_params.results_per_page,
            search_params.venue or "Wiley",
            search_params.from_year,
            search_params.to_year,
        )
        if err:
            logger.warning(f"Wiley search upstream error; returning empty results: {err}")
            items, total = [], 0
        if items is None:
            items, total = [], 0
        total_pages = math.ceil(total / search_params.results_per_page) if search_params.results_per_page > 0 else 0
        if total == 0:
            total_pages = 0
        return GenericPageSearchResponse(
            query_echo={
                "q": search_params.q,
                "venue": search_params.venue or "Wiley",
                "from_year": search_params.from_year,
                "to_year": search_params.to_year,
            },
            items=[GenericPaper(**it) for it in items],
            total_results=total,
            page=search_params.page,
            results_per_page=search_params.results_per_page,
            total_pages=total_pages,
            pagination=build_page_pagination_meta(
                page=search_params.page,
                per_page=search_params.results_per_page,
                total=total,
                total_pages=total_pages,
            ),
        )
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected Wiley search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_PROVIDER_UNEXPECTED_DETAIL) from e


@router.get(
    "/wiley/by-doi",
    response_model=GenericPaper,
    summary="Get Wiley via Crossref/OpenAlex by DOI (scaffold)",
    tags=["paper-search"],
)
async def paper_search_wiley_by_doi(params: DOIRequestForm = Depends()):
    loop = asyncio.get_running_loop()
    try:
        item, err = await loop.run_in_executor(None, Crossref.get_crossref_by_doi, params.doi)
        if err:
            _handle_provider_error(err)
        if not item:
            raise HTTPException(status_code=404, detail="Wiley record not found for DOI")  # noqa: TRY301
        return GenericPaper(**item)
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected Wiley by-doi error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_PROVIDER_UNEXPECTED_DETAIL) from e


@router.post(
    "/ingest/by-doi",
    summary="Resolve OA by DOI via Unpaywall, download PDF, process, and persist",
    tags=["paper-search"],
)
async def ingest_by_doi(
    doi: str = Query(..., description="DOI to ingest via OA when available"),
    title: Optional[str] = Query(None, description="Optional title override"),
    author: Optional[str] = Query(None, description="Optional author override"),
    keywords: Optional[str] = Query(None, description="Optional comma-separated keywords"),
    perform_chunking: bool = Query(True, description="Enable chunking during processing"),
    parser: Optional[str] = Query("pymupdf4llm", description="PDF parsing backend"),
    # Analysis
    api_name: Optional[str] = Query(None, description="LLM API name for analysis"),
    custom_prompt: Optional[str] = Query(None, description="Custom prompt for analysis"),
    system_prompt: Optional[str] = Query(None, description="System prompt for analysis"),
    enable_ocr: bool = Query(False, description="Enable OCR for scanned PDFs"),
    ocr_backend: Optional[str] = Query(None, description="OCR backend"),
    ocr_lang: Optional[str] = Query("eng", description="OCR language code"),
    ocr_dpi: int = Query(300, ge=72, le=600, description="OCR render DPI"),
    ocr_mode: Optional[str] = Query("fallback", description="OCR mode: 'always' or 'fallback'"),
    ocr_min_page_text_chars: int = Query(40, ge=0, le=2000, description="Min text chars/page to skip OCR"),
    ocr_output_format: Optional[str] = Query(None, description="OCR output format: text|markdown|json"),
    ocr_prompt_preset: Optional[str] = Query(None, description="OCR prompt preset (e.g., 'general', 'doc', 'table', 'spotting', 'json')"),
    chunk_method: Optional[str] = Query(None, description="Chunking method (e.g., 'sentences', 'semantic')"),
    chunk_size: int = Query(500, ge=50, le=4000, description="Target chunk size"),
    chunk_overlap: int = Query(200, ge=0, le=1000, description="Chunk overlap"),
    perform_analysis: bool = Query(True, description="Run analysis/summarization"),
    summarize_recursively: bool = Query(False, description="Enable recursive summarization"),
    db: MediaDbSession = Depends(get_media_db_for_user),
):
    """Best-effort OA ingestion using Unpaywall DOI resolution.

    If OA PDF cannot be resolved, returns 404 with a helpful message.
    """
    loop = asyncio.get_running_loop()
    try:
        # 1) Resolve OA PDF URL
        pdf_url, err = await loop.run_in_executor(None, Unpaywall.resolve_oa_pdf, doi)
        if err:
            _handle_provider_error(err)
        if not pdf_url:
            raise HTTPException(status_code=404, detail="No Open Access PDF found for DOI")  # noqa: TRY301

        # 2) Download PDF (HEAD check + atomic download)
        content = await _download_pdf_bytes(pdf_url)

        # 3) Process PDF bytes
        from tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.PDF_Processing_Lib import process_pdf_task

        kw_list = [k.strip() for k in (keywords or '').split(',') if k.strip()] if keywords else None
        result = await process_pdf_task(
            file_bytes=content,
            filename=f"{doi.replace('/', '_')}.pdf",
            parser=parser or "pymupdf4llm",
            title_override=title,
            author_override=author,
            keywords=kw_list,
            perform_chunking=perform_chunking,
            enable_ocr=enable_ocr or None,
            ocr_backend=ocr_backend or None,
            ocr_lang=ocr_lang or None,
            chunk_method=chunk_method,
            max_chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            perform_analysis=perform_analysis,
            api_name=api_name,
            custom_prompt=custom_prompt,
            system_prompt=system_prompt,
            summarize_recursively=summarize_recursively,
            ocr_dpi=ocr_dpi,
            ocr_mode=ocr_mode,
            ocr_min_page_text_chars=ocr_min_page_text_chars,
            ocr_output_format=ocr_output_format,
            ocr_prompt_preset=ocr_prompt_preset,
        )

        content_for_db = None
        analysis_for_db = None
        meta = {}
        smj = {"provider": "unpaywall", "doi": doi, "pdf_url": pdf_url}
        try:
            if result.get("chunks"):
                content_for_db = "\n\n".join([ch.get("text", "") for ch in result.get("chunks", [])])
            else:
                content_for_db = result.get("text") or None
            analysis_for_db = result.get("analysis") or None
            meta = result.get("metadata") or {}
        except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS:
            pass

        title_for_db = title or meta.get("title") or doi
        author_for_db = author or meta.get("author") or None

        # Prepare chunk rows if present
        chunks_for_sql = None
        try:
            if perform_chunking and result.get("chunks"):
                from tldw_Server_API.app.core.Chunking.chunker import Chunker as _Chunker
                _ck = _Chunker()
                _flat = _ck.chunk_text_hierarchical_flat(
                    content_for_db,
                    method=chunk_method or "sentences",
                    max_size=chunk_size,
                    overlap=chunk_overlap,
                )
                _kind_map = {
                    'paragraph': 'text', 'list_unordered': 'list', 'list_ordered': 'list',
                    'code_fence': 'code', 'table_md': 'table', 'header_line': 'heading', 'header_atx': 'heading'
                }
                chunks_for_sql = []
                for _it in _flat:
                    _md = _it.get('metadata') or {}
                    _ctype = _kind_map.get(str(_md.get('paragraph_kind') or '').lower(), 'text')
                    _small = {}
                    if _md.get('ancestry_titles'):
                        _small['ancestry_titles'] = _md.get('ancestry_titles')
                    if _md.get('section_path'):
                        _small['section_path'] = _md.get('section_path')
                    chunks_for_sql.append({
                        'text': _it.get('text',''),
                        'start_char': _md.get('start_offset'),
                        'end_char': _md.get('end_offset'),
                        'chunk_type': _ctype,
                        'metadata': _small,
                    })
        except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS:
            chunks_for_sql = None

        media_id, media_uuid, msg = await loop.run_in_executor(
            None,
            lambda: _ingest_paper_search_media(
                media_db=db,
                url=f"doi:{doi}",
                title=title_for_db,
                media_type="pdf",
                content=content_for_db,
                keywords=kw_list or [],
                prompt=custom_prompt,
                analysis_content=analysis_for_db,
                safe_metadata=smj,
                transcription_model='Imported',
                author=author_for_db,
                overwrite=False,
                chunk_options={"method": chunk_method or "sentences", "max_size": chunk_size, "overlap": chunk_overlap} if perform_chunking else None,
                chunks=chunks_for_sql,
            )
        )
        return {"message": msg, "media_id": media_id, "media_uuid": media_uuid, "source_pdf": pdf_url}  # noqa: TRY300
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        _raise_http_error_from_exception(
            e,
            not_found_detail="Unpaywall PDF returned 404",
            force_status=502,
        )
        logger.error(f"OA ingest by DOI error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="OA ingest by DOI failed") from e


@router.post(
    "/ingest/batch",
    response_model=IngestBatchResponse,
    summary="Batch ingest PDFs by pdf_url, DOI (OA), PMCID, or arXiv ID",
    tags=["paper-search"],
)
async def ingest_batch(
    payload: IngestBatchRequest,
    db: MediaDbSession = Depends(get_media_db_for_user),
):
    loop = asyncio.get_running_loop()
    results: list[IngestBatchResultItem] = []
    # Local import for processing
    from tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.PDF_Processing_Lib import process_pdf_task

    async def _process_one(it: dict) -> IngestBatchResultItem:
        doi = it.get("doi")
        pdf_url = it.get("pdf_url")
        pmcid = it.get("pmcid") or it.get("PMCID")
        arxiv_id = it.get("arxiv_id") or it.get("arXiv") or it.get("ArXiv")
        title = it.get("title")
        author = it.get("author")
        kw_list = it.get("keywords") or []
        try:
            # 1) Direct pdf_url path
            if pdf_url:
                content = await _download_pdf_bytes(pdf_url)
                result = await process_pdf_task(
                    file_bytes=content,
                    filename=(title or doi or arxiv_id or pmcid or "document").replace('/', '_') + ".pdf",
                    parser=payload.parser or "pymupdf4llm",
                    title_override=title,
                    author_override=author,
                    keywords=kw_list,
                    perform_chunking=payload.perform_chunking,
                    enable_ocr=payload.enable_ocr or None,
                    ocr_backend=payload.ocr_backend or None,
                    ocr_lang=payload.ocr_lang or None,
                    chunk_method=payload.chunk_method,
                    max_chunk_size=payload.chunk_size,
                    chunk_overlap=payload.chunk_overlap,
                    perform_analysis=payload.perform_analysis,
                    api_name=payload.api_name,
                    custom_prompt=payload.custom_prompt,
                    system_prompt=payload.system_prompt,
                    summarize_recursively=False,
                    ocr_dpi=payload.ocr_dpi,
                    ocr_mode=payload.ocr_mode,
                    ocr_min_page_text_chars=payload.ocr_min_page_text_chars,
                    ocr_output_format=payload.ocr_output_format,
                    ocr_prompt_preset=payload.ocr_prompt_preset,
                )
                content_for_db = result.get("transcript") or result.get("content") or result.get("text")
                analysis_for_db = result.get("summary") or result.get("analysis")
                meta = result.get("metadata") or {}
                title_for_db = title or meta.get("title") or (doi or arxiv_id or pmcid or "Document")
                author_for_db = author or meta.get("author")
                # Prepare chunks (optional)
                chunks_for_sql = None
                try:
                    if payload.perform_chunking and content_for_db:
                        from tldw_Server_API.app.core.Chunking.chunker import Chunker as _Chunker
                        _ck = _Chunker()
                        _flat = _ck.chunk_text_hierarchical_flat(
                            content_for_db,
                            method=payload.chunk_method or "sentences",
                            max_size=payload.chunk_size,
                            overlap=payload.chunk_overlap,
                        )
                        _kind_map = {
                            'paragraph': 'text', 'list_unordered': 'list', 'list_ordered': 'list',
                            'code_fence': 'code', 'table_md': 'table', 'header_line': 'heading', 'header_atx': 'heading'
                        }
                        chunks_for_sql = []
                        for _it in _flat:
                            _md = _it.get('metadata') or {}
                            _ctype = _kind_map.get(str(_md.get('paragraph_kind') or '').lower(), 'text')
                            _small = {}
                            if _md.get('ancestry_titles'):
                                _small['ancestry_titles'] = _md.get('ancestry_titles')
                            if _md.get('section_path'):
                                _small['section_path'] = _md.get('section_path')
                            chunks_for_sql.append({
                                'text': _it.get('text',''),
                                'start_char': _md.get('start_offset'),
                                'end_char': _md.get('end_offset'),
                                'chunk_type': _ctype,
                                'metadata': _small,
                            })
                except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS:
                    chunks_for_sql = None
                import json as _json
                smj = _json.dumps({"provider": "batch", "doi": doi, "pdf_url": pdf_url, "pmcid": pmcid, "arxiv_id": arxiv_id}, ensure_ascii=False)
                media_id, media_uuid, msg = await loop.run_in_executor(
                    None,
                    lambda: _ingest_paper_search_media(
                        media_db=db,
                        url=f"{('doi:'+doi) if doi else ('arxiv:'+arxiv_id) if arxiv_id else ('pmcid:'+pmcid) if pmcid else (pdf_url or 'unknown')}",
                        title=title_for_db,
                        media_type="pdf",
                        content=content_for_db,
                        keywords=kw_list or [],
                        prompt=payload.custom_prompt,
                        analysis_content=analysis_for_db,
                        safe_metadata=smj,
                        transcription_model='Imported',
                        author=author_for_db,
                        overwrite=False,
                        chunk_options={"method": payload.chunk_method or "sentences", "max_size": payload.chunk_size, "overlap": payload.chunk_overlap} if payload.perform_chunking else None,
                        chunks=chunks_for_sql,
                    )
                )
                return IngestBatchResultItem(doi=doi, pdf_url=pdf_url, pmcid=pmcid, arxiv_id=arxiv_id, success=True, media_id=media_id, media_uuid=media_uuid)

            # 2) PMCID path (PMC OA)
            if pmcid and not pdf_url:
                pmcid_norm = str(pmcid).strip()
                if not pmcid_norm.upper().startswith('PMC'):
                    pmcid_norm = f"PMC{pmcid_norm}"
                content, filename, d_err = await loop.run_in_executor(None, PMC_OA.download_pmc_pdf, pmcid_norm)
                if d_err:
                    return IngestBatchResultItem(
                        pmcid=pmcid_norm,
                        success=False,
                        error=f"PMC download error: {d_err}",
                    )
                if not content:
                    return IngestBatchResultItem(
                        pmcid=pmcid_norm,
                        success=False,
                        error="PMC download returned empty content",
                    )
                result = await process_pdf_task(
                    file_bytes=content,
                    filename=filename or f"{pmcid_norm}.pdf",
                    parser=payload.parser or "pymupdf4llm",
                    title_override=title,
                    author_override=author,
                    keywords=kw_list,
                    perform_chunking=payload.perform_chunking,
                    enable_ocr=payload.enable_ocr or None,
                    ocr_backend=payload.ocr_backend or None,
                    ocr_lang=payload.ocr_lang or None,
                    chunk_method=payload.chunk_method,
                    max_chunk_size=payload.chunk_size,
                    chunk_overlap=payload.chunk_overlap,
                    perform_analysis=payload.perform_analysis,
                    api_name=payload.api_name,
                    custom_prompt=payload.custom_prompt,
                    system_prompt=payload.system_prompt,
                    summarize_recursively=False,
                    ocr_dpi=payload.ocr_dpi,
                    ocr_mode=payload.ocr_mode,
                    ocr_min_page_text_chars=payload.ocr_min_page_text_chars,
                    ocr_output_format=payload.ocr_output_format,
                    ocr_prompt_preset=payload.ocr_prompt_preset,
                )
                content_for_db = result.get('transcript') or result.get('content') or result.get('text')
                analysis_for_db = result.get('summary') or result.get('analysis')
                meta = result.get('metadata') or {}
                title_for_db = title or meta.get('title') or pmcid_norm
                author_for_db = author or meta.get('author')
                # Chunks
                chunks_for_sql = None
                try:
                    if payload.perform_chunking and content_for_db:
                        from tldw_Server_API.app.core.Chunking.chunker import Chunker as _Chunker
                        _ck = _Chunker()
                        _flat = _ck.chunk_text_hierarchical_flat(
                            content_for_db,
                            method=payload.chunk_method or "sentences",
                            max_size=payload.chunk_size,
                            overlap=payload.chunk_overlap,
                        )
                        _kind_map = {
                            'paragraph': 'text', 'list_unordered': 'list', 'list_ordered': 'list',
                            'code_fence': 'code', 'table_md': 'table', 'header_line': 'heading', 'header_atx': 'heading'
                        }
                        chunks_for_sql = []
                        for _it in _flat:
                            _md = _it.get('metadata') or {}
                            _ctype = _kind_map.get(str(_md.get('paragraph_kind') or '').lower(), 'text')
                            _small = {}
                            if _md.get('ancestry_titles'):
                                _small['ancestry_titles'] = _md.get('ancestry_titles')
                            if _md.get('section_path'):
                                _small['section_path'] = _md.get('section_path')
                            chunks_for_sql.append({
                                'text': _it.get('text',''),
                                'start_char': _md.get('start_offset'),
                                'end_char': _md.get('end_offset'),
                                'chunk_type': _ctype,
                                'metadata': _small,
                            })
                except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS:
                    chunks_for_sql = None
                import json as _json
                smj = _json.dumps({"provider": "batch", "pmcid": pmcid_norm}, ensure_ascii=False)
                media_id, media_uuid, msg = await loop.run_in_executor(
                    None,
                    lambda: _ingest_paper_search_media(
                        media_db=db,
                        url=f"pmcid:{pmcid_norm}",
                        title=title_for_db,
                        media_type="pdf",
                        content=content_for_db,
                        keywords=kw_list or [],
                        prompt=payload.custom_prompt,
                        analysis_content=analysis_for_db,
                        safe_metadata=smj,
                        transcription_model='Imported',
                        author=author_for_db,
                        overwrite=False,
                        chunk_options={"method": payload.chunk_method or "sentences", "max_size": payload.chunk_size, "overlap": payload.chunk_overlap} if payload.perform_chunking else None,
                        chunks=chunks_for_sql,
                    )
                )
                return IngestBatchResultItem(pmcid=pmcid_norm, success=True, media_id=media_id, media_uuid=media_uuid)

            # 3) arXiv path
            if arxiv_id and not pdf_url:
                # Best-effort XML for title/author
                xml_text = None
                try:
                    xml_text = Arxiv.fetch_arxiv_xml(arxiv_id) or ""
                except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS:
                    xml_text = None
                meta = {}
                if xml_text:
                    try:
                        parsed = Arxiv.parse_arxiv_feed(xml_text.encode("utf-8"))
                        if parsed:
                            meta = parsed[0]
                    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS:
                        meta = {}
                pdf_guess = None
                try:
                    pdf_guess = Arxiv.fetch_arxiv_pdf_url(arxiv_id)
                except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS:
                    pdf_guess = None
                if not pdf_guess:
                    pdf_guess = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                content = await _download_pdf_bytes(pdf_guess)
                result = await process_pdf_task(
                    file_bytes=content,
                    filename=f"{arxiv_id}.pdf",
                    parser=payload.parser or "pymupdf4llm",
                    title_override=title,
                    author_override=author,
                    keywords=kw_list,
                    perform_chunking=payload.perform_chunking,
                    enable_ocr=payload.enable_ocr or None,
                    ocr_backend=payload.ocr_backend or None,
                    ocr_lang=payload.ocr_lang or None,
                    chunk_method=payload.chunk_method,
                    max_chunk_size=payload.chunk_size,
                    chunk_overlap=payload.chunk_overlap,
                    perform_analysis=payload.perform_analysis,
                    api_name=payload.api_name,
                    custom_prompt=payload.custom_prompt,
                    system_prompt=payload.system_prompt,
                    summarize_recursively=False,
                    ocr_dpi=payload.ocr_dpi,
                    ocr_mode=payload.ocr_mode,
                    ocr_min_page_text_chars=payload.ocr_min_page_text_chars,
                    ocr_output_format=payload.ocr_output_format,
                    ocr_prompt_preset=payload.ocr_prompt_preset,
                )
                content_for_db = result.get('transcript') or result.get('content') or result.get('text')
                analysis_for_db = result.get('summary') or result.get('analysis')
                meta_res = result.get('metadata') or {}
                title_for_db = title or meta.get('title') or meta_res.get('title') or arxiv_id
                author_for_db = author or meta_res.get('author') or meta.get('authors')
                chunks_for_sql = None
                try:
                    if payload.perform_chunking and content_for_db:
                        from tldw_Server_API.app.core.Chunking.chunker import Chunker as _Chunker
                        _ck = _Chunker()
                        _flat = _ck.chunk_text_hierarchical_flat(
                            content_for_db,
                            method=payload.chunk_method or "sentences",
                            max_size=payload.chunk_size,
                            overlap=payload.chunk_overlap,
                        )
                        _kind_map = {
                            'paragraph': 'text', 'list_unordered': 'list', 'list_ordered': 'list',
                            'code_fence': 'code', 'table_md': 'table', 'header_line': 'heading', 'header_atx': 'heading'
                        }
                        chunks_for_sql = []
                        for _it in _flat:
                            _md = _it.get('metadata') or {}
                            _ctype = _kind_map.get(str(_md.get('paragraph_kind') or '').lower(), 'text')
                            _small = {}
                            if _md.get('ancestry_titles'):
                                _small['ancestry_titles'] = _md.get('ancestry_titles')
                            if _md.get('section_path'):
                                _small['section_path'] = _md.get('section_path')
                            chunks_for_sql.append({
                                'text': _it.get('text',''),
                                'start_char': _md.get('start_offset'),
                                'end_char': _md.get('end_offset'),
                                'chunk_type': _ctype,
                                'metadata': _small,
                            })
                except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS:
                    chunks_for_sql = None
                import json as _json
                smj = _json.dumps({"provider": "batch", "arxiv_id": arxiv_id, "pdf_url": pdf_guess}, ensure_ascii=False)
                media_id, media_uuid, msg = await loop.run_in_executor(
                    None,
                    lambda: _ingest_paper_search_media(
                        media_db=db,
                        url=f"arxiv:{arxiv_id}",
                        title=title_for_db,
                        media_type="pdf",
                        content=content_for_db,
                        keywords=kw_list or [],
                        prompt=payload.custom_prompt,
                        analysis_content=analysis_for_db,
                        safe_metadata=smj,
                        transcription_model='Imported',
                        author=author_for_db,
                        overwrite=False,
                        chunk_options={"method": payload.chunk_method or "sentences", "max_size": payload.chunk_size, "overlap": payload.chunk_overlap} if payload.perform_chunking else None,
                        chunks=chunks_for_sql,
                    )
                )
                return IngestBatchResultItem(arxiv_id=arxiv_id, success=True, media_id=media_id, media_uuid=media_uuid)

            # 4) DOI path via Unpaywall
            if not pdf_url and doi:
                _, err = (None, None)
                pdf_url, err = await loop.run_in_executor(None, Unpaywall.resolve_oa_pdf, doi)
                if err:
                    return IngestBatchResultItem(
                        doi=doi,
                        pdf_url=None,
                        pmcid=pmcid,
                        arxiv_id=arxiv_id,
                        success=False,
                        error=f"DOI resolution error: {err}",
                    )
            if not pdf_url:
                return IngestBatchResultItem(
                    doi=doi,
                    pdf_url=None,
                    pmcid=pmcid,
                    arxiv_id=arxiv_id,
                    success=False,
                    error="DOI resolution returned no PDF URL",
                )
            # Download PDF
            content = await _download_pdf_bytes(pdf_url)
            # Process & persist
            result = await process_pdf_task(
                file_bytes=content,
                filename=(title or doi or "document").replace('/', '_') + ".pdf",
                parser=payload.parser or "pymupdf4llm",
                title_override=title,
                author_override=author,
                keywords=kw_list,
                perform_chunking=payload.perform_chunking,
                enable_ocr=payload.enable_ocr or None,
                ocr_backend=payload.ocr_backend or None,
                ocr_lang=payload.ocr_lang or None,
                chunk_method=payload.chunk_method,
                max_chunk_size=payload.chunk_size,
                chunk_overlap=payload.chunk_overlap,
                perform_analysis=payload.perform_analysis,
                api_name=payload.api_name,
                custom_prompt=payload.custom_prompt,
                system_prompt=payload.system_prompt,
                summarize_recursively=False,
                ocr_dpi=payload.ocr_dpi,
                ocr_mode=payload.ocr_mode,
                ocr_min_page_text_chars=payload.ocr_min_page_text_chars,
                ocr_output_format=payload.ocr_output_format,
                ocr_prompt_preset=payload.ocr_prompt_preset,
            )
            content_for_db = None
            analysis_for_db = None
            smj = {"provider": "batch", "doi": doi, "pdf_url": pdf_url}
            try:
                if result.get("chunks"):
                    content_for_db = "\n\n".join([ch.get("text", "") for ch in result.get("chunks", [])])
                else:
                    content_for_db = result.get("text") or None
                analysis_for_db = result.get("analysis") or None
            except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS:
                pass
            title_for_db = title or doi or (pdf_url.split('/')[-1] if pdf_url else "Document")
            author_for_db = author or None
            media_id, media_uuid, msg = await loop.run_in_executor(
                None,
                lambda: _ingest_paper_search_media(
                    media_db=db,
                    url=f"doi:{doi}" if doi else pdf_url,
                    title=title_for_db,
                    media_type="pdf",
                    content=content_for_db,
                    keywords=kw_list,
                    prompt=payload.custom_prompt,
                    analysis_content=analysis_for_db,
                    safe_metadata=smj,
                    transcription_model='Imported',
                    author=author_for_db,
                    overwrite=False,
                    chunk_options={"method": payload.chunk_method or "sentences", "max_size": payload.chunk_size, "overlap": payload.chunk_overlap} if payload.perform_chunking else None,
                    chunks=None,
                )
            )
            return IngestBatchResultItem(doi=doi, pdf_url=pdf_url, pmcid=pmcid, arxiv_id=arxiv_id, success=True, media_id=media_id, media_uuid=media_uuid)
        except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Batch ingest error for doi={doi} pmcid={pmcid} arxiv={arxiv_id} pdf={pdf_url}: {e}", exc_info=True)
            return IngestBatchResultItem(
                doi=doi,
                pdf_url=pdf_url,
                pmcid=pmcid,
                arxiv_id=arxiv_id,
                success=False,
                error=f"Batch ingest error: {e}",
            )

    for it in payload.items:
        try:
            item_dict = model_dump_compat(it)
        except TypeError:
            encoded_item = jsonable_encoder(it)
            if not isinstance(encoded_item, dict):
                raise HTTPException(status_code=400, detail="Invalid batch ingest payload")  # noqa: B904
            item_dict = encoded_item
        results.append(await _process_one(item_dict))

    succeeded = sum(1 for r in results if r.success)
    failed = len(results) - succeeded
    return IngestBatchResponse(results=results, succeeded=succeeded, failed=failed)


# -------------------- ChemRxiv Endpoints --------------------

@router.get(
    "/chemrxiv/items",
    response_model=GenericPageSearchResponse,
    summary="Search ChemRxiv items",
    tags=["paper-search"],
)
async def chemrxiv_items(search: ChemRxivSearchRequestForm = Depends()):
    loop = asyncio.get_running_loop()
    try:
        items, total, err = await loop.run_in_executor(
            None,
            ChemRxiv.search_items,
            search.term,
            search.skip,
            search.limit,
            search.sort,
            search.author,
            search.searchDateFrom,
            search.searchDateTo,
            search.searchLicense,
            search.categoryIds_list,
            search.subjectIds_list,
        )
        if err:
            _handle_provider_error(err)
        if items is None:
            raise HTTPException(status_code=500, detail="ChemRxiv search failed to return data.")  # noqa: TRY301
        page = (search.skip // max(1, search.limit)) + 1
        total_pages = math.ceil(total / search.limit) if search.limit > 0 else 0
        return GenericPageSearchResponse(
            query_echo={
                "term": search.term,
                "skip": search.skip,
                "limit": search.limit,
                "sort": search.sort,
                "author": search.author,
            },
            items=[GenericPaper(**it) for it in items],
            total_results=total,
            page=page,
            results_per_page=search.limit,
            total_pages=total_pages,
            pagination=build_page_pagination_meta(
                page=page,
                per_page=search.limit,
                total=total,
                total_pages=total_pages,
            ),
        )
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected ChemRxiv search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_PROVIDER_UNEXPECTED_DETAIL) from e


@router.get(
    "/chemrxiv/items/by-id",
    response_model=GenericPaper,
    summary="Get ChemRxiv item by ID",
    tags=["paper-search"],
)
async def chemrxiv_item_by_id(itemId: str = Query(..., min_length=3)):  # noqa: N803
    loop = asyncio.get_running_loop()
    try:
        item, err = await loop.run_in_executor(None, ChemRxiv.get_item_by_id, itemId)
        if err:
            _handle_provider_error(err)
        if not item:
            raise HTTPException(status_code=404, detail="ChemRxiv item not found")  # noqa: TRY301
        return GenericPaper(**item)
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Unexpected ChemRxiv by-id error")
        raise HTTPException(status_code=500, detail=_PROVIDER_UNEXPECTED_DETAIL) from e


@router.get(
    "/chemrxiv/items/by-doi",
    response_model=GenericPaper,
    summary="Get ChemRxiv item by DOI",
    tags=["paper-search"],
)
async def chemrxiv_item_by_doi(doi: str = Query(..., min_length=3)):
    loop = asyncio.get_running_loop()
    try:
        item, err = await loop.run_in_executor(None, ChemRxiv.get_item_by_doi, doi)
        if err:
            _handle_provider_error(err)
        if not item:
            raise HTTPException(status_code=404, detail="ChemRxiv item not found for DOI")  # noqa: TRY301
        return GenericPaper(**item)
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected ChemRxiv by-doi error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_PROVIDER_UNEXPECTED_DETAIL) from e


@router.get(
    "/chemrxiv/categories",
    summary="Get ChemRxiv categories",
    tags=["paper-search"],
)
async def chemrxiv_categories():
    loop = asyncio.get_running_loop()
    data, err = await loop.run_in_executor(None, ChemRxiv.get_categories)
    if err:
        _handle_provider_error(err)
    return data or {"categories": []}


@router.get(
    "/chemrxiv/licenses",
    summary="Get ChemRxiv licenses",
    tags=["paper-search"],
)
async def chemrxiv_licenses():
    loop = asyncio.get_running_loop()
    data, err = await loop.run_in_executor(None, ChemRxiv.get_licenses)
    if err:
        _handle_provider_error(err)
    return data or {"licenses": []}


@router.get(
    "/chemrxiv/version",
    summary="Get ChemRxiv API version",
    tags=["paper-search"],
)
async def chemrxiv_version():
    loop = asyncio.get_running_loop()
    data, err = await loop.run_in_executor(None, ChemRxiv.get_version)
    if err:
        _handle_provider_error(err)
    return data or {}


@router.get(
    "/chemrxiv/oai",
    summary="ChemRxiv OAI-PMH passthrough (XML)",
    response_class=Response,
    responses=_RAW_XML_RESPONSES,
    tags=["paper-search"],
)
async def chemrxiv_oai(
    verb: str = Query(..., description="OAI verb"),
    identifier: Optional[str] = Query(None),
    metadataPrefix: Optional[str] = Query(None),  # noqa: N803
    from_date: Optional[str] = Query(None, alias="from"),
    until_date: Optional[str] = Query(None, alias="until"),
    resumptionToken: Optional[str] = Query(None),  # noqa: N803
    set_name: Optional[str] = Query(None, alias="set"),
):
    loop = asyncio.get_running_loop()
    params: dict[str, Any] = {"verb": verb}
    if identifier:
        params["identifier"] = identifier
    if metadataPrefix:
        params["metadataPrefix"] = metadataPrefix
    if from_date:
        params["from"] = from_date
    if until_date:
        params["until"] = until_date
    if resumptionToken:
        params["resumptionToken"] = resumptionToken
    if set_name:
        params["set"] = set_name
    content, media, err = await loop.run_in_executor(None, ChemRxiv.oai_raw, params)
    if err:
        _handle_provider_error(err)
    if not content:
        raise HTTPException(status_code=404, detail="No content")
    return Response(content=content, media_type=media or "application/xml")


# -------------------- IACR Endpoints --------------------

@router.get(
    "/iacr/conf",
    response_model=IacrConferenceResponse,
    summary="IACR conference metadata (by venue + year)",
    tags=["paper-search"],
)
async def iacr_conf(
    venue: str = Query(..., description="crypto|eurocrypt|asiacrypt|fse|ches|tcc|pkc"),
    year: int = Query(..., ge=1970, le=2100),
):
    loop = asyncio.get_running_loop()
    data, err = await loop.run_in_executor(None, IACR.fetch_conference, venue, year)
    if err:
        _handle_provider_error(err)
    if data is None:
        raise HTTPException(status_code=404, detail="Conference not found")
    return IacrConferenceResponse(query_echo={"venue": venue, "year": year}, data=data)


@router.get(
    "/iacr/conf/raw",
    summary="IACR conference metadata raw (download)",
    response_class=Response,
    responses=_RAW_JSON_RESPONSES,
    tags=["paper-search"],
)
async def iacr_conf_raw(
    venue: str = Query(..., description="crypto|eurocrypt|asiacrypt|fse|ches|tcc|pkc"),
    year: int = Query(..., ge=1970, le=2100),
):
    loop = asyncio.get_running_loop()
    content, media, err = await loop.run_in_executor(None, IACR.fetch_conference_raw, venue, year)
    if err:
        _handle_provider_error(err)
    if not content:
        raise HTTPException(status_code=404, detail="No content")
    return Response(content=content, media_type=media or "application/json")


@router.get(
    "/earthrxiv",
    response_model=GenericPageSearchResponse,
    summary="Search EarthArXiv (OSF) preprints",
    tags=["paper-search"],
)
async def earthrxiv_search(
    term: Optional[str] = Query(None, description="Search query (OSF 'q' parameter)"),
    from_date: Optional[str] = Query(None, description="Filter by date_created >= YYYY-MM-DD"),
    page: int = Query(1, ge=1),
    results_per_page: int = Query(10, ge=1, le=100),
):
    loop = asyncio.get_running_loop()
    try:
        items, total, err = await loop.run_in_executor(None, EarthRxiv.search_items, term, page, results_per_page, from_date)
        if err:
            _handle_provider_error(err)
        if items is None:
            raise HTTPException(status_code=500, detail="EarthArXiv search failed to return data.")  # noqa: TRY301
        total_pages = math.ceil(total / results_per_page) if results_per_page > 0 else 0
        return GenericPageSearchResponse(
            query_echo={"term": term, "from_date": from_date},
            items=[GenericPaper(**it) for it in items],
            total_results=total,
            page=page,
            results_per_page=results_per_page,
            total_pages=total_pages,
            pagination=build_page_pagination_meta(
                page=page,
                per_page=results_per_page,
                total=total,
                total_pages=total_pages,
            ),
        )
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected EarthArXiv search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_PROVIDER_UNEXPECTED_DETAIL) from e


@router.get(
    "/earthrxiv/by-id",
    response_model=GenericPaper,
    summary="Get EarthArXiv item by OSF ID",
    tags=["paper-search"],
)
async def earthrxiv_by_id(osf_id: str = Query(..., min_length=3)):
    loop = asyncio.get_running_loop()
    try:
        item, err = await loop.run_in_executor(None, EarthRxiv.get_item_by_id, osf_id)
        if err:
            _handle_provider_error(err)
        if not item:
            raise HTTPException(status_code=404, detail="EarthArXiv item not found")  # noqa: TRY301
        return GenericPaper(**item)
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Unexpected EarthArXiv by-id error")
        raise HTTPException(status_code=500, detail=_PROVIDER_UNEXPECTED_DETAIL) from e


@router.get(
    "/earthrxiv/by-doi",
    response_model=GenericPaper,
    summary="Get EarthArXiv item by DOI",
    tags=["paper-search"],
)
async def earthrxiv_by_doi(doi: str = Query(..., min_length=3)):
    loop = asyncio.get_running_loop()
    try:
        item, err = await loop.run_in_executor(None, EarthRxiv.get_item_by_doi, doi)
        if err:
            _handle_provider_error(err)
        if not item:
            raise HTTPException(status_code=404, detail="EarthArXiv item not found for DOI")  # noqa: TRY301
        return GenericPaper(**item)
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected EarthArXiv by-doi error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_PROVIDER_UNEXPECTED_DETAIL) from e


# -------------------- OSF Preprints (Generic) --------------------

@router.get(
    "/osf",
    response_model=GenericPageSearchResponse,
    summary="Search OSF preprints (all providers or a specific provider)",
    tags=["paper-search"],
)
async def osf_search(search: OSFSearchRequestForm = Depends()):
    loop = asyncio.get_running_loop()
    try:
        items, total, err = await loop.run_in_executor(
            None,
            OSF.search_preprints,
            search.term,
            search.page,
            search.results_per_page,
            search.provider,
            search.from_date,
        )
        if err:
            _handle_provider_error(err)
        if items is None:
            raise HTTPException(status_code=500, detail="OSF search failed to return data.")  # noqa: TRY301
        total_pages = math.ceil(total / search.results_per_page) if search.results_per_page > 0 else 0
        return GenericPageSearchResponse(
            query_echo={
                "term": search.term,
                "provider": search.provider,
                "from_date": search.from_date,
            },
            items=[GenericPaper(**it) for it in items],
            total_results=total,
            page=search.page,
            results_per_page=search.results_per_page,
            total_pages=total_pages,
            pagination=build_page_pagination_meta(
                page=search.page,
                per_page=search.results_per_page,
                total=total,
                total_pages=total_pages,
            ),
        )
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected OSF search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_PROVIDER_UNEXPECTED_DETAIL) from e


@router.get(
    "/osf/by-id",
    response_model=GenericPaper,
    summary="Get OSF preprint by ID",
    tags=["paper-search"],
)
async def osf_by_id(osf_id: str = Query(..., min_length=3)):
    loop = asyncio.get_running_loop()
    try:
        item, err = await loop.run_in_executor(None, OSF.get_preprint_by_id, osf_id)
        if err:
            _handle_provider_error(err)
        if not item:
            raise HTTPException(status_code=404, detail="OSF preprint not found")  # noqa: TRY301
        return GenericPaper(**item)
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Unexpected OSF by-id error")
        raise HTTPException(status_code=500, detail=_PROVIDER_UNEXPECTED_DETAIL) from e


@router.get(
    "/osf/by-doi",
    response_model=GenericPaper,
    summary="Get OSF preprint by DOI",
    tags=["paper-search"],
)
async def osf_by_doi(doi: str = Query(..., min_length=3)):
    loop = asyncio.get_running_loop()
    try:
        item, err = await loop.run_in_executor(None, OSF.get_preprint_by_doi, doi)
        if err:
            _handle_provider_error(err)
        if not item:
            raise HTTPException(status_code=404, detail="OSF preprint not found for DOI")  # noqa: TRY301
        return GenericPaper(**item)
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected OSF by-doi error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_PROVIDER_UNEXPECTED_DETAIL) from e


@router.post(
    "/osf/ingest",
    summary="Download OSF preprint primary-file PDF by OSF ID, process, and persist",
    tags=["paper-search"],
)
async def osf_ingest(
    osf_id: str = Query(..., description="OSF preprint ID"),
    keywords: Optional[str] = Query(None, description="Comma-separated keywords"),
    perform_chunking: bool = Query(True),
    parser: Optional[str] = Query("pymupdf4llm"),
    chunk_method: Optional[str] = Query(None),
    chunk_size: int = Query(500, ge=50, le=4000),
    chunk_overlap: int = Query(200, ge=0, le=1000),
    perform_analysis: bool = Query(True),
    custom_prompt: Optional[str] = Query(None),
    system_prompt: Optional[str] = Query(None),
    api_name: Optional[str] = Query(None),
    enable_ocr: bool = Query(False),
    ocr_backend: Optional[str] = Query(None),
    ocr_lang: Optional[str] = Query("eng"),
    ocr_dpi: int = Query(300, ge=72, le=600),
    ocr_mode: Optional[str] = Query("fallback"),
    ocr_min_page_text_chars: int = Query(40, ge=0, le=2000),
    ocr_output_format: Optional[str] = Query(None, description="OCR output format: text|markdown|json"),
    ocr_prompt_preset: Optional[str] = Query(None, description="OCR prompt preset (e.g., 'general', 'doc', 'table', 'spotting', 'json')"),
    db: MediaDbSession = Depends(get_media_db_for_user),
):
    loop = asyncio.get_running_loop()
    try:
        # Resolve primary file download URL
        pdf_url, perr = await loop.run_in_executor(None, OSF.get_primary_file_download_url, osf_id)
        if perr:
            _handle_provider_error(perr)
        if not pdf_url:
            raise HTTPException(status_code=404, detail="No primary file download URL found for this preprint")  # noqa: TRY301

        content = await _download_pdf_bytes(pdf_url)

        # Fetch minimal metadata for title/doi if possible
        meta, _ = await loop.run_in_executor(None, OSF.get_preprint_by_id, osf_id)
        title = (meta or {}).get("title") if isinstance(meta, dict) else None
        doi = (meta or {}).get("doi") if isinstance(meta, dict) else None

        # Process PDF
        from tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.PDF_Processing_Lib import process_pdf_task
        kw_list = [k.strip() for k in (keywords or '').split(',') if k.strip()] if keywords else None
        result = await process_pdf_task(
            file_bytes=content,
            filename=f"{osf_id}.pdf",
            parser=parser or "pymupdf4llm",
            keywords=kw_list,
            perform_chunking=perform_chunking,
            chunk_method=chunk_method,
            max_chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            perform_analysis=perform_analysis,
            api_name=api_name,
            custom_prompt=custom_prompt,
            system_prompt=system_prompt,
            enable_ocr=enable_ocr or None,
            ocr_backend=ocr_backend or None,
            ocr_lang=ocr_lang or None,
            ocr_dpi=ocr_dpi,
            ocr_mode=ocr_mode,
            ocr_min_page_text_chars=ocr_min_page_text_chars,
            ocr_output_format=ocr_output_format,
            ocr_prompt_preset=ocr_prompt_preset,
        )

        content_for_db = result.get('transcript') or result.get('content')
        if not content_for_db:
            raise HTTPException(status_code=500, detail="Processing did not produce content")  # noqa: TRY301

        # Safe metadata
        from tldw_Server_API.app.core.Utils.metadata_utils import normalize_safe_metadata
        sm = normalize_safe_metadata({
            "osf_id": osf_id,
            "title": title,
            "doi": doi,
            "source": "osf",
        })
        import json as _json
        smj = None
        try:
            smj = _json.dumps(sm, ensure_ascii=False)
        except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS:
            smj = None

        # Optional chunk capture for SQL
        chunks_for_sql = None
        try:
            if perform_chunking:
                from tldw_Server_API.app.core.Chunking.chunker import Chunker as _Chunker
                _ck = _Chunker()
                _flat = _ck.chunk_text_hierarchical_flat(
                    content_for_db,
                    method=chunk_method or "sentences",
                    max_size=chunk_size,
                    overlap=chunk_overlap,
                )
                _kind_map = {
                    'paragraph': 'text', 'list_unordered': 'list', 'list_ordered': 'list',
                    'code_fence': 'code', 'table_md': 'table', 'header_line': 'heading', 'header_atx': 'heading'
                }
                chunks_for_sql = []
                for _it in _flat:
                    _md = _it.get('metadata') or {}
                    _ctype = _kind_map.get(str(_md.get('paragraph_kind') or '').lower(), 'text')
                    _small = {}
                    if _md.get('ancestry_titles'):
                        _small['ancestry_titles'] = _md.get('ancestry_titles')
                    if _md.get('section_path'):
                        _small['section_path'] = _md.get('section_path')
                    chunks_for_sql.append({
                        'text': _it.get('text',''),
                        'start_char': _md.get('start_offset'),
                        'end_char': _md.get('end_offset'),
                        'chunk_type': _ctype,
                        'metadata': _small,
                    })
        except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS:
            chunks_for_sql = None

        # Persist
        media_id, media_uuid, msg = await loop.run_in_executor(
            None,
            lambda: _ingest_paper_search_media(
                media_db=db,
                url=f"osf:{osf_id}",
                title=title or osf_id,
                media_type="pdf",
                content=content_for_db,
                keywords=kw_list or [],
                prompt=custom_prompt,
                analysis_content=result.get('summary') or result.get('analysis'),
                transcription_model='Imported',
                author=None,
                safe_metadata=smj,
                overwrite=False,
                chunk_options={"method": chunk_method or "sentences", "max_size": chunk_size, "overlap": chunk_overlap} if perform_chunking else None,
                chunks=chunks_for_sql,
            )
        )
        return {"message": msg, "media_id": media_id, "media_uuid": media_uuid}  # noqa: TRY300
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        _raise_http_error_from_exception(e)
        logger.error(f"OSF ingest error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="OSF ingest failed") from e


@router.get(
    "/osf/raw",
    summary="Raw passthrough for OSF preprints list (JSON)",
    response_class=Response,
    responses=_RAW_JSON_RESPONSES,
    tags=["paper-search"],
)
async def osf_raw(
    term: Optional[str] = Query(None, alias="q"),
    provider: Optional[str] = Query(None, alias="filter[provider]"),
    date_gte: Optional[str] = Query(None, alias="filter[date_created][gte]"),
    page_size: Optional[int] = Query(None, alias="page[size]"),
    page_number: Optional[int] = Query(None, alias="page[number]"),
):
    loop = asyncio.get_running_loop()
    params: dict[str, Any] = {}
    if term:
        params["q"] = term
    if provider:
        params["filter[provider]"] = provider
    if date_gte:
        params["filter[date_created][gte]"] = date_gte
    if page_size is not None:
        params["page[size]"] = max(1, min(int(page_size), 100))
    if page_number is not None:
        params["page[number]"] = max(1, int(page_number))
    content, media, err = await loop.run_in_executor(None, OSF.raw_preprints, params)
    if err:
        _handle_provider_error(err)
    if not content:
        raise HTTPException(status_code=404, detail="No content")
    return Response(content=content, media_type=media or "application/json")


@router.get(
    "/osf/raw/by-id",
    summary="Raw passthrough for a single OSF preprint (JSON)",
    response_class=Response,
    responses=_RAW_JSON_RESPONSES,
    tags=["paper-search"],
)
async def osf_raw_by_id(osf_id: str = Query(..., min_length=3)):
    loop = asyncio.get_running_loop()
    content, media, err = await loop.run_in_executor(None, OSF.raw_by_id, osf_id)
    if err:
        _handle_provider_error(err)
    if not content:
        raise HTTPException(status_code=404, detail="No content")
    return Response(content=content, media_type=media or "application/json")



# -------------------- Zenodo Endpoints --------------------

@router.get(
    "/zenodo",
    response_model=GenericPageSearchResponse,
    summary="Search Zenodo published records",
    tags=["paper-search"],
)
async def zenodo_search(
    q: Optional[str] = Query(None, description="Search query"),
    page: int = Query(1, ge=1),
    results_per_page: int = Query(10, ge=1, le=100),
    type: Optional[str] = Query(None, description="Filter by type (Publication, Dataset, Software, etc.)"),
    subtype: Optional[str] = Query(None, description="Subtype (Journal article, Preprint, etc.)"),
    communities: Optional[str] = Query(None, description="Community identifier"),
):
    loop = asyncio.get_running_loop()
    try:
        items, total, err = await loop.run_in_executor(None, Zenodo.search_records, q, page, results_per_page, type, subtype, communities)
        if err:
            _handle_provider_error(err)
        if items is None:
            raise HTTPException(status_code=500, detail="Zenodo search failed to return data.")  # noqa: TRY301
        total_pages = math.ceil(total / results_per_page) if results_per_page > 0 else 0
        return GenericPageSearchResponse(
            query_echo={"q": q, "type": type, "subtype": subtype, "communities": communities},
            items=[GenericPaper(**it) for it in items],
            total_results=total,
            page=page,
            results_per_page=results_per_page,
            total_pages=total_pages,
            pagination=build_page_pagination_meta(
                page=page,
                per_page=results_per_page,
                total=total,
                total_pages=total_pages,
            ),
        )
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected Zenodo search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_PROVIDER_UNEXPECTED_DETAIL) from e


@router.get(
    "/zenodo/by-id",
    response_model=GenericPaper,
    summary="Get Zenodo record by ID",
    tags=["paper-search"],
)
async def zenodo_by_id(record_id: str = Query(..., min_length=1)):
    loop = asyncio.get_running_loop()
    try:
        item, err = await loop.run_in_executor(None, Zenodo.get_record_by_id, record_id)
        if err:
            _handle_provider_error(err)
        if not item:
            raise HTTPException(status_code=404, detail="Zenodo record not found")  # noqa: TRY301
        return GenericPaper(**item)
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected Zenodo by-id error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_PROVIDER_UNEXPECTED_DETAIL) from e


@router.get(
    "/zenodo/by-doi",
    response_model=GenericPaper,
    summary="Get Zenodo record by DOI",
    tags=["paper-search"],
)
async def zenodo_by_doi(doi: str = Query(..., min_length=3)):
    loop = asyncio.get_running_loop()
    try:
        item, err = await loop.run_in_executor(None, Zenodo.get_record_by_doi, doi)
        if err:
            _handle_provider_error(err)
        if not item:
            raise HTTPException(status_code=404, detail="Zenodo record not found for DOI")  # noqa: TRY301
        return GenericPaper(**item)
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected Zenodo by-doi error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_PROVIDER_UNEXPECTED_DETAIL) from e


@router.get(
    "/zenodo/oai",
    summary="Zenodo OAI-PMH passthrough (XML)",
    response_class=Response,
    responses=_RAW_XML_RESPONSES,
    tags=["paper-search"],
)
async def zenodo_oai(
    verb: str = Query(..., description="OAI verb"),
    identifier: Optional[str] = Query(None),
    metadataPrefix: Optional[str] = Query(None),  # noqa: N803
    from_date: Optional[str] = Query(None, alias="from"),
    until_date: Optional[str] = Query(None, alias="until"),
    resumptionToken: Optional[str] = Query(None),  # noqa: N803
    set_name: Optional[str] = Query(None, alias="set"),
):
    loop = asyncio.get_running_loop()
    params: dict[str, Any] = {"verb": verb}
    if identifier:
        params["identifier"] = identifier
    if metadataPrefix:
        params["metadataPrefix"] = metadataPrefix
    if from_date:
        params["from"] = from_date
    if until_date:
        params["until"] = until_date
    if resumptionToken:
        params["resumptionToken"] = resumptionToken
    if set_name:
        params["set"] = set_name
    content, media, err = await loop.run_in_executor(None, Zenodo.oai_raw, params)
    if err:
        _handle_provider_error(err)
    if not content:
        raise HTTPException(status_code=404, detail="No content")
    return Response(content=content, media_type=media or "application/xml")


@router.post(
    "/zenodo/ingest",
    summary="Download Zenodo PDF by record ID, process, and persist",
    tags=["paper-search"],
)
async def zenodo_ingest(
    record_id: str = Query(..., description="Zenodo record ID (e.g., 1234567)"),
    keywords: Optional[str] = Query(None, description="Comma-separated keywords"),
    perform_chunking: bool = Query(True),
    parser: Optional[str] = Query("pymupdf4llm"),
    chunk_method: Optional[str] = Query(None),
    chunk_size: int = Query(500, ge=50, le=4000),
    chunk_overlap: int = Query(200, ge=0, le=1000),
    perform_analysis: bool = Query(True),
    custom_prompt: Optional[str] = Query(None),
    system_prompt: Optional[str] = Query(None),
    api_name: Optional[str] = Query(None),
    enable_ocr: bool = Query(False),
    ocr_backend: Optional[str] = Query(None),
    ocr_lang: Optional[str] = Query("eng"),
    ocr_dpi: int = Query(300, ge=72, le=600),
    ocr_mode: Optional[str] = Query("fallback"),
    ocr_min_page_text_chars: int = Query(40, ge=0, le=2000),
    ocr_output_format: Optional[str] = Query(None, description="OCR output format: text|markdown|json"),
    ocr_prompt_preset: Optional[str] = Query(None, description="OCR prompt preset (e.g., 'general', 'doc', 'table', 'spotting', 'json')"),
    db: MediaDbSession = Depends(get_media_db_for_user),
):
    """Download a PDF from a Zenodo record (if available), process, and persist.

    Notes:
      - Not all Zenodo records have PDFs; if no PDF is found in files, returns 404.
    """
    loop = asyncio.get_running_loop()
    try:
        # 1) Fetch record metadata and locate a PDF link
        item, err = await loop.run_in_executor(None, Zenodo.get_record_by_id, record_id)
        if err:
            _handle_provider_error(err)
        if not item:
            raise HTTPException(status_code=404, detail="Zenodo record not found")  # noqa: TRY301
        title_meta = item.get('title')
        doi_meta = item.get('doi')
        pdf_url = item.get('pdf_url')
        # Fallback: inspect raw record to find a PDF if not present
        if not pdf_url:
            raw, raw_err = await loop.run_in_executor(None, Zenodo.get_record_raw, record_id)
            if raw_err:
                _handle_provider_error(raw_err)
            if raw and isinstance(raw, dict):
                pdf_url = Zenodo.extract_pdf_from_raw(raw)
        if not pdf_url:
            raise HTTPException(status_code=404, detail="No PDF link found for this Zenodo record")  # noqa: TRY301

        # 2) Download PDF
        content = await _download_pdf_bytes(pdf_url)

        # 3) Process PDF
        from tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.PDF_Processing_Lib import process_pdf_task
        kw_list = [k.strip() for k in (keywords or '').split(',') if k.strip()] if keywords else None
        result = await process_pdf_task(
            file_bytes=content,
            filename=f"zenodo_{record_id}.pdf",
            parser=parser or "pymupdf4llm",
            keywords=kw_list,
            perform_chunking=perform_chunking,
            chunk_method=chunk_method,
            max_chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            perform_analysis=perform_analysis,
            api_name=api_name,
            custom_prompt=custom_prompt,
            system_prompt=system_prompt,
            enable_ocr=enable_ocr or None,
            ocr_backend=ocr_backend or None,
            ocr_lang=ocr_lang or None,
            ocr_dpi=ocr_dpi,
            ocr_mode=ocr_mode,
            ocr_min_page_text_chars=ocr_min_page_text_chars,
            ocr_output_format=ocr_output_format,
            ocr_prompt_preset=ocr_prompt_preset,
        )
        content_for_db = result.get('transcript') or result.get('content') or result.get('text')
        if not content_for_db:
            raise HTTPException(status_code=500, detail="Processing did not produce content")  # noqa: TRY301

        from tldw_Server_API.app.core.Utils.metadata_utils import normalize_safe_metadata
        sm = normalize_safe_metadata({
            "zenodo_record_id": record_id,
            "doi": doi_meta,
            "title": title_meta,
            "pdf_url": pdf_url,
            "source": "zenodo",
        })
        import json as _json
        smj = _json.dumps({k: v for k, v in sm.items() if v}, ensure_ascii=False)
        analysis_for_db = result.get('summary') or result.get('analysis')
        title_for_db = title_meta or f"Zenodo {record_id}"
        author_for_db = None

        # 4) Optional: Build chunks for chunk-level FTS
        chunks_for_sql = None
        try:
            if perform_chunking:
                from tldw_Server_API.app.core.Chunking.chunker import Chunker as _Chunker
                _ck = _Chunker()
                _flat = _ck.chunk_text_hierarchical_flat(
                    content_for_db,
                    method=chunk_method or "sentences",
                    max_size=chunk_size,
                    overlap=chunk_overlap,
                )
                _kind_map = {
                    'paragraph': 'text', 'list_unordered': 'list', 'list_ordered': 'list',
                    'code_fence': 'code', 'table_md': 'table', 'header_line': 'heading', 'header_atx': 'heading'
                }
                chunks_for_sql = []
                for _it in _flat:
                    _md = _it.get('metadata') or {}
                    _ctype = _kind_map.get(str(_md.get('paragraph_kind') or '').lower(), 'text')
                    _small = {}
                    if _md.get('ancestry_titles'):
                        _small['ancestry_titles'] = _md.get('ancestry_titles')
                    if _md.get('section_path'):
                        _small['section_path'] = _md.get('section_path')
                    chunks_for_sql.append({
                        'text': _it.get('text',''),
                        'start_char': _md.get('start_offset'),
                        'end_char': _md.get('end_offset'),
                        'chunk_type': _ctype,
                        'metadata': _small,
                    })
        except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS:
            chunks_for_sql = None

        media_id, media_uuid, msg = await loop.run_in_executor(
            None,
            lambda: _ingest_paper_search_media(
                media_db=db,
                url=f"zenodo:{record_id}",
                title=title_for_db,
                media_type="pdf",
                content=content_for_db,
                keywords=kw_list or [],
                prompt=custom_prompt,
                analysis_content=analysis_for_db,
                safe_metadata=smj,
                transcription_model='Imported',
                author=author_for_db,
                overwrite=False,
                chunk_options={"method": chunk_method or "sentences", "max_size": chunk_size, "overlap": chunk_overlap} if perform_chunking else None,
                chunks=chunks_for_sql,
            )
        )
        return {"message": msg, "media_id": media_id, "media_uuid": media_uuid}  # noqa: TRY300
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        _raise_http_error_from_exception(e)
        logger.error(f"Zenodo ingest error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Zenodo ingest failed") from e


# -------------------- Figshare Endpoints --------------------

@router.get(
    "/figshare",
    response_model=GenericPageSearchResponse,
    summary="Search Figshare records",
    tags=["paper-search"],
)
async def figshare_search(
    q: Optional[str] = Query(None, description="Free text to search in metadata"),
    search_for: Optional[str] = Query(None, description="Advanced fielded search string, e.g., :title: frog"),
    page: int = Query(1, ge=1),
    results_per_page: int = Query(10, ge=1, le=1000),
    order: Optional[str] = Query(None, description="Sort key, e.g., published_date"),
    order_direction: Optional[str] = Query("desc", description="asc|desc"),
):
    loop = asyncio.get_running_loop()
    try:
        items, total, err = await loop.run_in_executor(None, Figshare.search_articles, q, page, results_per_page, order, order_direction, search_for)
        if err:
            _handle_provider_error(err)
        if items is None:
            raise HTTPException(status_code=500, detail="Figshare search failed to return data.")  # noqa: TRY301
        total_pages = math.ceil(total / results_per_page) if results_per_page > 0 else 0
        return GenericPageSearchResponse(
            query_echo={"q": q, "search_for": search_for, "order": order, "order_direction": order_direction},
            items=[GenericPaper(**it) for it in items],
            total_results=total,
            page=page,
            results_per_page=results_per_page,
            total_pages=total_pages,
            pagination=build_page_pagination_meta(
                page=page,
                per_page=results_per_page,
                total=total,
                total_pages=total_pages,
            ),
        )
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected Figshare search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_PROVIDER_UNEXPECTED_DETAIL) from e


@router.get(
    "/figshare/by-id",
    response_model=GenericPaper,
    summary="Get Figshare item by article ID",
    tags=["paper-search"],
)
async def figshare_by_id(article_id: str = Query(..., min_length=1)):
    loop = asyncio.get_running_loop()
    try:
        item, err = await loop.run_in_executor(None, Figshare.get_article_by_id, article_id)
        if err:
            _handle_provider_error(err)
        if not item:
            raise HTTPException(status_code=404, detail="Figshare article not found")  # noqa: TRY301
        return GenericPaper(**item)
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected Figshare by-id error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_PROVIDER_UNEXPECTED_DETAIL) from e


@router.get(
    "/figshare/by-doi",
    response_model=GenericPaper,
    summary="Get Figshare item by DOI (best-effort)",
    tags=["paper-search"],
)
async def figshare_by_doi(doi: str = Query(..., min_length=3)):
    loop = asyncio.get_running_loop()
    try:
        item, err = await loop.run_in_executor(None, Figshare.get_article_by_doi, doi)
        if err:
            _handle_provider_error(err)
        if not item:
            raise HTTPException(status_code=404, detail="Figshare item not found for DOI")  # noqa: TRY301
        return GenericPaper(**item)
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected Figshare by-doi error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_PROVIDER_UNEXPECTED_DETAIL) from e


@router.get(
    "/figshare/oai",
    summary="Figshare OAI-PMH passthrough (XML)",
    response_class=Response,
    responses=_RAW_XML_RESPONSES,
    tags=["paper-search"],
)
async def figshare_oai(
    verb: str = Query(..., description="OAI verb"),
    identifier: Optional[str] = Query(None),
    metadataPrefix: Optional[str] = Query(None),  # noqa: N803
    from_date: Optional[str] = Query(None, alias="from"),
    until_date: Optional[str] = Query(None, alias="until"),
    resumptionToken: Optional[str] = Query(None),  # noqa: N803
    set_name: Optional[str] = Query(None, alias="set"),
):
    loop = asyncio.get_running_loop()
    params: dict[str, Any] = {"verb": verb}
    if identifier:
        params["identifier"] = identifier
    if metadataPrefix:
        params["metadataPrefix"] = metadataPrefix
    if from_date:
        params["from"] = from_date
    if until_date:
        params["until"] = until_date
    if resumptionToken:
        params["resumptionToken"] = resumptionToken
    if set_name:
        params["set"] = set_name
    content, media, err = await loop.run_in_executor(None, Figshare.oai_raw, params)
    if err:
        _handle_provider_error(err)
    if not content:
        raise HTTPException(status_code=404, detail="No content")
    return Response(content=content, media_type=media or "application/xml")


@router.post(
    "/figshare/ingest",
    summary="Download a Figshare PDF by Article ID, process, and persist",
    tags=["paper-search"],
)
async def figshare_ingest(
    article_id: str = Query(..., description="Figshare article ID (numeric)"),
    keywords: Optional[str] = Query(None, description="Comma-separated keywords"),
    perform_chunking: bool = Query(True),
    parser: Optional[str] = Query("pymupdf4llm"),
    chunk_method: Optional[str] = Query(None),
    chunk_size: int = Query(500, ge=50, le=4000),
    chunk_overlap: int = Query(200, ge=0, le=1000),
    perform_analysis: bool = Query(True),
    custom_prompt: Optional[str] = Query(None),
    system_prompt: Optional[str] = Query(None),
    api_name: Optional[str] = Query(None),
    enable_ocr: bool = Query(False),
    ocr_backend: Optional[str] = Query(None),
    ocr_lang: Optional[str] = Query("eng"),
    ocr_dpi: int = Query(300, ge=72, le=600),
    ocr_mode: Optional[str] = Query("fallback"),
    ocr_min_page_text_chars: int = Query(40, ge=0, le=2000),
    ocr_output_format: Optional[str] = Query(None, description="OCR output format: text|markdown|json"),
    ocr_prompt_preset: Optional[str] = Query(None, description="OCR prompt preset (e.g., 'general', 'doc', 'table', 'spotting', 'json')"),
    db: MediaDbSession = Depends(get_media_db_for_user),
):
    loop = asyncio.get_running_loop()
    try:
        # 1) Fetch raw article and determine a PDF download URL
        raw, err = await loop.run_in_executor(None, Figshare.get_article_raw, article_id)
        if err:
            _handle_provider_error(err)
        if not raw:
            raise HTTPException(status_code=404, detail="Figshare article not found")  # noqa: TRY301
        pdf_url = Figshare.extract_pdf_download_url(raw)
        title_meta = raw.get("title")
        doi_meta = raw.get("doi")
        if not pdf_url:
            raise HTTPException(status_code=404, detail="No PDF link found for this Figshare article")  # noqa: TRY301

        # 2) Download PDF
        content = await _download_pdf_bytes(pdf_url)

        # 3) Process PDF
        from tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.PDF_Processing_Lib import process_pdf_task
        kw_list = [k.strip() for k in (keywords or '').split(',') if k.strip()] if keywords else None
        result = await process_pdf_task(
            file_bytes=content,
            filename=f"figshare_{article_id}.pdf",
            parser=parser or "pymupdf4llm",
            keywords=kw_list,
            perform_chunking=perform_chunking,
            chunk_method=chunk_method,
            max_chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            perform_analysis=perform_analysis,
            api_name=api_name,
            custom_prompt=custom_prompt,
            system_prompt=system_prompt,
            enable_ocr=enable_ocr or None,
            ocr_backend=ocr_backend or None,
            ocr_lang=ocr_lang or None,
            ocr_dpi=ocr_dpi,
            ocr_mode=ocr_mode,
            ocr_min_page_text_chars=ocr_min_page_text_chars,
            ocr_output_format=ocr_output_format,
            ocr_prompt_preset=ocr_prompt_preset,
        )
        content_for_db = result.get('transcript') or result.get('content') or result.get('text')
        if not content_for_db:
            raise HTTPException(status_code=500, detail="Processing did not produce content")  # noqa: TRY301

        from tldw_Server_API.app.core.Utils.metadata_utils import normalize_safe_metadata
        sm = normalize_safe_metadata({
            "figshare_article_id": article_id,
            "doi": doi_meta,
            "title": title_meta,
            "pdf_url": pdf_url,
            "source": "figshare",
        })
        import json as _json
        smj = _json.dumps({k: v for k, v in sm.items() if v}, ensure_ascii=False)
        analysis_for_db = result.get('summary') or result.get('analysis')
        title_for_db = title_meta or f"Figshare {article_id}"
        author_for_db = None

        # 4) Optional chunk objects
        chunks_for_sql = None
        try:
            if perform_chunking:
                from tldw_Server_API.app.core.Chunking.chunker import Chunker as _Chunker
                _ck = _Chunker()
                _flat = _ck.chunk_text_hierarchical_flat(
                    content_for_db,
                    method=chunk_method or "sentences",
                    max_size=chunk_size,
                    overlap=chunk_overlap,
                )
                _kind_map = {
                    'paragraph': 'text', 'list_unordered': 'list', 'list_ordered': 'list',
                    'code_fence': 'code', 'table_md': 'table', 'header_line': 'heading', 'header_atx': 'heading'
                }
                chunks_for_sql = []
                for _it in _flat:
                    _md = _it.get('metadata') or {}
                    _ctype = _kind_map.get(str(_md.get('paragraph_kind') or '').lower(), 'text')
                    _small = {}
                    if _md.get('ancestry_titles'):
                        _small['ancestry_titles'] = _md.get('ancestry_titles')
                    if _md.get('section_path'):
                        _small['section_path'] = _md.get('section_path')
                    chunks_for_sql.append({
                        'text': _it.get('text',''),
                        'start_char': _md.get('start_offset'),
                        'end_char': _md.get('end_offset'),
                        'chunk_type': _ctype,
                        'metadata': _small,
                    })
        except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS:
            chunks_for_sql = None

        media_id, media_uuid, msg = await loop.run_in_executor(
            None,
            lambda: _ingest_paper_search_media(
                media_db=db,
                url=f"figshare:{article_id}",
                title=title_for_db,
                media_type="pdf",
                content=content_for_db,
                keywords=kw_list or [],
                prompt=custom_prompt,
                analysis_content=analysis_for_db,
                safe_metadata=smj,
                transcription_model='Imported',
                author=author_for_db,
                overwrite=False,
                chunk_options={"method": chunk_method or "sentences", "max_size": chunk_size, "overlap": chunk_overlap} if perform_chunking else None,
                chunks=chunks_for_sql,
            )
        )
        return {"message": msg, "media_id": media_id, "media_uuid": media_uuid}  # noqa: TRY300
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        _raise_http_error_from_exception(e)
        logger.error(f"Figshare ingest error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Figshare ingest failed") from e


@router.post(
    "/figshare/ingest-by-doi",
    summary="Resolve Figshare record by DOI and ingest PDF",
    tags=["paper-search"],
)
async def figshare_ingest_by_doi(
    doi: str = Query(..., description="DOI, e.g., 10.6084/m9.figshare.5616409.v3"),
    keywords: Optional[str] = Query(None, description="Comma-separated keywords"),
    perform_chunking: bool = Query(True),
    parser: Optional[str] = Query("pymupdf4llm"),
    chunk_method: Optional[str] = Query(None),
    chunk_size: int = Query(500, ge=50, le=4000),
    chunk_overlap: int = Query(200, ge=0, le=1000),
    perform_analysis: bool = Query(True),
    custom_prompt: Optional[str] = Query(None),
    system_prompt: Optional[str] = Query(None),
    api_name: Optional[str] = Query(None),
    enable_ocr: bool = Query(False),
    ocr_backend: Optional[str] = Query(None),
    ocr_lang: Optional[str] = Query("eng"),
    ocr_dpi: int = Query(300, ge=72, le=600),
    ocr_mode: Optional[str] = Query("fallback"),
    ocr_min_page_text_chars: int = Query(40, ge=0, le=2000),
    ocr_output_format: Optional[str] = Query(None, description="OCR output format: text|markdown|json"),
    ocr_prompt_preset: Optional[str] = Query(None, description="OCR prompt preset (e.g., 'general', 'doc', 'table', 'spotting', 'json')"),
    db: MediaDbSession = Depends(get_media_db_for_user),
):
    """Convenience endpoint: resolves DOI to Figshare article, downloads its PDF, and ingests it."""
    loop = asyncio.get_running_loop()
    try:
        item, err = await loop.run_in_executor(None, Figshare.get_article_by_doi, doi)
        if err:
            _handle_provider_error(err)
        if not item or not item.get("id"):
            raise HTTPException(status_code=404, detail="Figshare item not found for DOI")  # noqa: TRY301
        # Reuse ingestion logic using article_id
        article_id = str(item["id"])  # normalized GenericPaper id is a string

        # Fetch raw and proceed as in figshare_ingest
        raw, err2 = await loop.run_in_executor(None, Figshare.get_article_raw, article_id)
        if err2:
            _handle_provider_error(err2)
        if not raw:
            raise HTTPException(status_code=404, detail="Figshare article not found")  # noqa: TRY301
        pdf_url = Figshare.extract_pdf_download_url(raw)
        title_meta = raw.get("title")
        doi_meta = raw.get("doi")
        if not pdf_url:
            raise HTTPException(status_code=404, detail="No PDF link found for this Figshare article")  # noqa: TRY301

        content = await _download_pdf_bytes(pdf_url)

        from tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.PDF_Processing_Lib import process_pdf_task
        kw_list = [k.strip() for k in (keywords or '').split(',') if k.strip()] if keywords else None
        result = await process_pdf_task(
            file_bytes=content,
            filename=f"figshare_{article_id}.pdf",
            parser=parser or "pymupdf4llm",
            keywords=kw_list,
            perform_chunking=perform_chunking,
            chunk_method=chunk_method,
            max_chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            perform_analysis=perform_analysis,
            api_name=api_name,
            custom_prompt=custom_prompt,
            system_prompt=system_prompt,
            enable_ocr=enable_ocr or None,
            ocr_backend=ocr_backend or None,
            ocr_lang=ocr_lang or None,
            ocr_dpi=ocr_dpi,
            ocr_mode=ocr_mode,
            ocr_min_page_text_chars=ocr_min_page_text_chars,
            ocr_output_format=ocr_output_format,
            ocr_prompt_preset=ocr_prompt_preset,
        )
        content_for_db = result.get('transcript') or result.get('content') or result.get('text')
        if not content_for_db:
            raise HTTPException(status_code=500, detail="Processing did not produce content")  # noqa: TRY301

        from tldw_Server_API.app.core.Utils.metadata_utils import normalize_safe_metadata
        sm = normalize_safe_metadata({
            "figshare_article_id": article_id,
            "doi": doi_meta or doi,
            "title": title_meta,
            "pdf_url": pdf_url,
            "source": "figshare",
        })
        import json as _json
        smj = _json.dumps({k: v for k, v in sm.items() if v}, ensure_ascii=False)
        analysis_for_db = result.get('summary') or result.get('analysis')
        title_for_db = title_meta or f"Figshare {article_id}"
        author_for_db = None

        chunks_for_sql = None
        try:
            if perform_chunking:
                from tldw_Server_API.app.core.Chunking.chunker import Chunker as _Chunker
                _ck = _Chunker()
                _flat = _ck.chunk_text_hierarchical_flat(
                    content_for_db,
                    method=chunk_method or "sentences",
                    max_size=chunk_size,
                    overlap=chunk_overlap,
                )
                _kind_map = {
                    'paragraph': 'text', 'list_unordered': 'list', 'list_ordered': 'list',
                    'code_fence': 'code', 'table_md': 'table', 'header_line': 'heading', 'header_atx': 'heading'
                }
                chunks_for_sql = []
                for _it in _flat:
                    _md = _it.get('metadata') or {}
                    _ctype = _kind_map.get(str(_md.get('paragraph_kind') or '').lower(), 'text')
                    _small = {}
                    if _md.get('ancestry_titles'):
                        _small['ancestry_titles'] = _md.get('ancestry_titles')
                    if _md.get('section_path'):
                        _small['section_path'] = _md.get('section_path')
                    chunks_for_sql.append({
                        'text': _it.get('text',''),
                        'start_char': _md.get('start_offset'),
                        'end_char': _md.get('end_offset'),
                        'chunk_type': _ctype,
                        'metadata': _small,
                    })
        except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS:
            chunks_for_sql = None

        media_id, media_uuid, msg = await loop.run_in_executor(
            None,
            lambda: _ingest_paper_search_media(
                media_db=db,
                url=f"figshare:{article_id}",
                title=title_for_db,
                media_type="pdf",
                content=content_for_db,
                keywords=kw_list or [],
                prompt=custom_prompt,
                analysis_content=analysis_for_db,
                safe_metadata=smj,
                transcription_model='Imported',
                author=author_for_db,
                overwrite=False,
                chunk_options={"method": chunk_method or "sentences", "max_size": chunk_size, "overlap": chunk_overlap} if perform_chunking else None,
                chunks=chunks_for_sql,
            )
        )
        return {"message": msg, "media_id": media_id, "media_uuid": media_uuid}  # noqa: TRY300
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        _raise_http_error_from_exception(e)
        logger.error(f"Figshare ingest-by-doi error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Figshare ingest-by-doi failed") from e

# -------------------- HAL.science Endpoints --------------------

@router.get(
    "/hal",
    response_model=GenericPageSearchResponse,
    summary="Search HAL (Solr-like)",
    tags=["paper-search"],
)
async def hal_search(
    q: str = Query("*:*", description="Solr-like query (e.g., title_t:japon)"),
    page: int = Query(1, ge=1),
    results_per_page: int = Query(10, ge=1, le=1000),
    fl: Optional[str] = Query(None, description="Fields list (fl=...)"),
    fq: Optional[str] = Query(None, description="Filter query (repeat with &fq=...)"),
    sort: Optional[str] = Query(None, description="Sort (e.g., docid asc)"),
    scope: Optional[str] = Query(None, description="Portal or COLLECTION scope segment, e.g., 'tel' or 'FRANCE-GRILLES'"),
):
    loop = asyncio.get_running_loop()
    start = (page - 1) * results_per_page
    fqs = [fq] if fq else None
    try:
        items, total, err = await loop.run_in_executor(None, HAL.search, q, start, results_per_page, fl, fqs, sort, scope)
        if err:
            _handle_provider_error(err)
        if items is None:
            raise HTTPException(status_code=500, detail="HAL search failed to return data.")  # noqa: TRY301
        total_pages = math.ceil(total / results_per_page) if results_per_page > 0 else 0
        return GenericPageSearchResponse(
            query_echo={"q": q, "fl": fl, "fq": fqs, "sort": sort, "scope": scope},
            items=[GenericPaper(**it) for it in items],
            total_results=total,
            page=page,
            results_per_page=results_per_page,
            total_pages=total_pages,
            pagination=build_page_pagination_meta(
                page=page,
                per_page=results_per_page,
                total=total,
                total_pages=total_pages,
            ),
        )
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected HAL search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_PROVIDER_UNEXPECTED_DETAIL) from e


@router.get(
    "/hal/raw",
    summary="HAL raw passthrough (wt=json|xml|xml-tei|csv|bibtex|endnote|atom|rss)",
    response_class=Response,
    responses=_RAW_HAL_RESPONSES,
    tags=["paper-search"],
)
async def hal_raw(
    q: str = Query("*:*"),
    wt: str = Query("json"),
    fl: Optional[str] = Query(None),
    rows: int = Query(10, ge=0, le=10000),
    start: int = Query(0, ge=0),
    fq: Optional[str] = Query(None),
    sort: Optional[str] = Query(None),
    indent: Optional[bool] = Query(None),
    scope: Optional[str] = Query(None, description="Portal or COLLECTION scope segment"),
):
    loop = asyncio.get_running_loop()
    params: dict[str, Any] = {"q": q, "wt": wt, "rows": rows, "start": start}
    if fl:
        params["fl"] = fl
    if fq:
        params["fq"] = fq
    if sort:
        params["sort"] = sort
    if indent is not None:
        params["indent"] = "true" if indent else "false"
    content, media, err = await loop.run_in_executor(None, HAL.raw, params, scope)
    if err:
        _handle_provider_error(err)
    if not content:
        raise HTTPException(status_code=404, detail="No content")
    return Response(content=content, media_type=media or "application/octet-stream")


@router.get(
    "/hal/by-id",
    response_model=GenericPaper,
    summary="Get HAL document by docid",
    tags=["paper-search"],
)
async def hal_by_id(docid: str = Query(..., min_length=1), scope: Optional[str] = Query(None)):
    loop = asyncio.get_running_loop()
    try:
        item, err = await loop.run_in_executor(None, HAL.by_docid, docid, None, scope)
        if err:
            _handle_provider_error(err)
        if not item:
            raise HTTPException(status_code=404, detail="HAL doc not found")  # noqa: TRY301
        return GenericPaper(**item)
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected HAL by-id error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_PROVIDER_UNEXPECTED_DETAIL) from e


@router.post(
    "/hal/ingest",
    summary="Download a HAL PDF by docid, process, and persist",
    tags=["paper-search"],
)
async def hal_ingest(
    docid: str = Query(..., description="HAL docid"),
    scope: Optional[str] = Query(None, description="Portal or COLLECTION scope segment"),
    keywords: Optional[str] = Query(None),
    perform_chunking: bool = Query(True),
    parser: Optional[str] = Query("pymupdf4llm"),
    chunk_method: Optional[str] = Query(None),
    chunk_size: int = Query(500, ge=50, le=4000),
    chunk_overlap: int = Query(200, ge=0, le=1000),
    perform_analysis: bool = Query(True),
    custom_prompt: Optional[str] = Query(None),
    system_prompt: Optional[str] = Query(None),
    api_name: Optional[str] = Query(None),
    enable_ocr: bool = Query(False),
    ocr_backend: Optional[str] = Query(None),
    ocr_lang: Optional[str] = Query("eng"),
    ocr_dpi: int = Query(300, ge=72, le=600),
    ocr_mode: Optional[str] = Query("fallback"),
    ocr_min_page_text_chars: int = Query(40, ge=0, le=2000),
    ocr_output_format: Optional[str] = Query(None, description="OCR output format: text|markdown|json"),
    ocr_prompt_preset: Optional[str] = Query(None, description="OCR prompt preset (e.g., 'general', 'doc', 'table', 'spotting', 'json')"),
    db: MediaDbSession = Depends(get_media_db_for_user),
):
    loop = asyncio.get_running_loop()
    try:
        item, err = await loop.run_in_executor(None, HAL.by_docid, docid, None, scope)
        if err:
            _handle_provider_error(err)
        if not item:
            raise HTTPException(status_code=404, detail="HAL doc not found")  # noqa: TRY301
        raw_item = item  # already normalized includes pdf_url best-effort
        pdf_url = (raw_item or {}).get("pdf_url") or None
        title_meta = (raw_item or {}).get("title")
        doi_meta = (raw_item or {}).get("doi")
        if not pdf_url:
            raise HTTPException(status_code=404, detail="No PDF link found for this HAL document")  # noqa: TRY301

        content = await _download_pdf_bytes(pdf_url)

        from tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.PDF_Processing_Lib import process_pdf_task
        kw_list = [k.strip() for k in (keywords or '').split(',') if k.strip()] if keywords else None
        result = await process_pdf_task(
            file_bytes=content,
            filename=f"hal_{docid}.pdf",
            parser=parser or "pymupdf4llm",
            keywords=kw_list,
            perform_chunking=perform_chunking,
            chunk_method=chunk_method,
            max_chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            perform_analysis=perform_analysis,
            api_name=api_name,
            custom_prompt=custom_prompt,
            system_prompt=system_prompt,
            enable_ocr=enable_ocr or None,
            ocr_backend=ocr_backend or None,
            ocr_lang=ocr_lang or None,
            ocr_dpi=ocr_dpi,
            ocr_mode=ocr_mode,
            ocr_min_page_text_chars=ocr_min_page_text_chars,
            ocr_output_format=ocr_output_format,
            ocr_prompt_preset=ocr_prompt_preset,
        )
        content_for_db = result.get('transcript') or result.get('content') or result.get('text')
        if not content_for_db:
            raise HTTPException(status_code=500, detail="Processing did not produce content")  # noqa: TRY301

        from tldw_Server_API.app.core.Utils.metadata_utils import normalize_safe_metadata
        sm = normalize_safe_metadata({
            "hal_docid": docid,
            "doi": doi_meta,
            "title": title_meta,
            "pdf_url": pdf_url,
            "source": "hal",
        })
        import json as _json
        smj = _json.dumps({k: v for k, v in sm.items() if v}, ensure_ascii=False)
        analysis_for_db = result.get('summary') or result.get('analysis')
        title_for_db = title_meta or f"HAL {docid}"
        author_for_db = (raw_item or {}).get("authors")

        chunks_for_sql = None
        try:
            if perform_chunking:
                from tldw_Server_API.app.core.Chunking.chunker import Chunker as _Chunker
                _ck = _Chunker()
                _flat = _ck.chunk_text_hierarchical_flat(
                    content_for_db,
                    method=chunk_method or "sentences",
                    max_size=chunk_size,
                    overlap=chunk_overlap,
                )
                _kind_map = {
                    'paragraph': 'text', 'list_unordered': 'list', 'list_ordered': 'list',
                    'code_fence': 'code', 'table_md': 'table', 'header_line': 'heading', 'header_atx': 'heading'
                }
                chunks_for_sql = []
                for _it in _flat:
                    _md = _it.get('metadata') or {}
                    _ctype = _kind_map.get(str(_md.get('paragraph_kind') or '').lower(), 'text')
                    _small = {}
                    if _md.get('ancestry_titles'):
                        _small['ancestry_titles'] = _md.get('ancestry_titles')
                    if _md.get('section_path'):
                        _small['section_path'] = _md.get('section_path')
                    chunks_for_sql.append({
                        'text': _it.get('text',''),
                        'start_char': _md.get('start_offset'),
                        'end_char': _md.get('end_offset'),
                        'chunk_type': _ctype,
                        'metadata': _small,
                    })
        except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS:
            chunks_for_sql = None

        media_id, media_uuid, msg = await loop.run_in_executor(
            None,
            lambda: _ingest_paper_search_media(
                media_db=db,
                url=f"hal:{docid}",
                title=title_for_db,
                media_type="pdf",
                content=content_for_db,
                keywords=kw_list or [],
                prompt=custom_prompt,
                analysis_content=analysis_for_db,
                safe_metadata=smj,
                transcription_model='Imported',
                author=author_for_db,
                overwrite=False,
                chunk_options={"method": chunk_method or "sentences", "max_size": chunk_size, "overlap": chunk_overlap} if perform_chunking else None,
                chunks=chunks_for_sql,
            )
        )
        return {"message": msg, "media_id": media_id, "media_uuid": media_uuid}  # noqa: TRY300
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        _raise_http_error_from_exception(e)
        logger.error(f"HAL ingest error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="HAL ingest failed") from e

# -------------------- viXra Endpoints --------------------

@router.get(
    "/vixra/by-id",
    response_model=GenericPaper,
    summary="Resolve a viXra ID to minimal metadata and PDF when possible",
    tags=["paper-search"],
)
async def vixra_by_id(vid: str = Query(..., description="viXra ID, e.g., 1901.0001")):
    loop = asyncio.get_running_loop()
    item, err = await loop.run_in_executor(None, Vixra.get_vixra_by_id, vid)
    if err:
        _handle_provider_error(err)
    if not item:
        raise HTTPException(status_code=404, detail="viXra item not found")
    return GenericPaper(**item)


@router.post(
    "/vixra/ingest",
    summary="Download viXra PDF by ID, process, and persist",
    tags=["paper-search"],
)
async def vixra_ingest(
    vid: str = Query(..., description="viXra ID, e.g., 1901.0001"),
    keywords: Optional[str] = Query(None),
    perform_chunking: bool = Query(True),
    parser: Optional[str] = Query("pymupdf4llm"),
    chunk_method: Optional[str] = Query(None),
    chunk_size: int = Query(500, ge=50, le=4000),
    chunk_overlap: int = Query(200, ge=0, le=1000),
    perform_analysis: bool = Query(True),
    custom_prompt: Optional[str] = Query(None),
    system_prompt: Optional[str] = Query(None),
    api_name: Optional[str] = Query(None),
    enable_ocr: bool = Query(False),
    ocr_backend: Optional[str] = Query(None),
    ocr_lang: Optional[str] = Query("eng"),
    ocr_dpi: int = Query(300, ge=72, le=600),
    ocr_mode: Optional[str] = Query("fallback"),
    ocr_min_page_text_chars: int = Query(40, ge=0, le=2000),
    ocr_output_format: Optional[str] = Query(None, description="OCR output format: text|markdown|json"),
    ocr_prompt_preset: Optional[str] = Query(None, description="OCR prompt preset (e.g., 'general', 'doc', 'table', 'spotting', 'json')"),
    db: MediaDbSession = Depends(get_media_db_for_user),
):
    loop = asyncio.get_running_loop()
    try:
        item, err = await loop.run_in_executor(None, Vixra.get_vixra_by_id, vid)
        if err:
            _handle_provider_error(err)
        if not item or not item.get('pdf_url'):
            raise HTTPException(status_code=404, detail="viXra PDF not found for ID")  # noqa: TRY301
        pdf_url = item['pdf_url']
        # Download PDF (set Accept header to prefer PDF)
        content = await _download_pdf_bytes(pdf_url, headers={"Accept": "application/pdf, */*"})

        # Process
        from tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.PDF_Processing_Lib import process_pdf_task
        kw_list = [k.strip() for k in (keywords or '').split(',') if k.strip()] if keywords else None
        result = await process_pdf_task(
            file_bytes=content,
            filename=f"vixra_{vid}.pdf",
            parser=parser or "pymupdf4llm",
            keywords=kw_list,
            perform_chunking=perform_chunking,
            chunk_method=chunk_method,
            max_chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            perform_analysis=perform_analysis,
            api_name=api_name,
            custom_prompt=custom_prompt,
            system_prompt=system_prompt,
            enable_ocr=enable_ocr or None,
            ocr_backend=ocr_backend or None,
            ocr_lang=ocr_lang or None,
            ocr_dpi=ocr_dpi,
            ocr_mode=ocr_mode,
            ocr_min_page_text_chars=ocr_min_page_text_chars,
            ocr_output_format=ocr_output_format,
            ocr_prompt_preset=ocr_prompt_preset,
        )
        content_for_db = result.get('transcript') or result.get('content') or result.get('text')
        if not content_for_db:
            raise HTTPException(status_code=500, detail="Processing did not produce content")  # noqa: TRY301
        from tldw_Server_API.app.core.Utils.metadata_utils import normalize_safe_metadata
        sm = normalize_safe_metadata({
            "vixra_id": vid,
            "title": item.get('title'),
            "pdf_url": pdf_url,
            "source": "vixra",
        })
        import json as _json
        smj = _json.dumps({k: v for k, v in sm.items() if v}, ensure_ascii=False)
        analysis_for_db = result.get('summary') or result.get('analysis')
        title_for_db = item.get('title') or vid
        author_for_db = None

        chunks_for_sql = None
        try:
            if perform_chunking:
                from tldw_Server_API.app.core.Chunking.chunker import Chunker as _Chunker
                _ck = _Chunker()
                _flat = _ck.chunk_text_hierarchical_flat(
                    content_for_db,
                    method=chunk_method or "sentences",
                    max_size=chunk_size,
                    overlap=chunk_overlap,
                )
                _kind_map = {
                    'paragraph': 'text', 'list_unordered': 'list', 'list_ordered': 'list',
                    'code_fence': 'code', 'table_md': 'table', 'header_line': 'heading', 'header_atx': 'heading'
                }
                chunks_for_sql = []
                for _it in _flat:
                    _md = _it.get('metadata') or {}
                    _ctype = _kind_map.get(str(_md.get('paragraph_kind') or '').lower(), 'text')
                    _small = {}
                    if _md.get('ancestry_titles'):
                        _small['ancestry_titles'] = _md.get('ancestry_titles')
                    if _md.get('section_path'):
                        _small['section_path'] = _md.get('section_path')
                    chunks_for_sql.append({
                        'text': _it.get('text',''),
                        'start_char': _md.get('start_offset'),
                        'end_char': _md.get('end_offset'),
                        'chunk_type': _ctype,
                        'metadata': _small,
                    })
        except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS:
            chunks_for_sql = None

        media_id, media_uuid, msg = await loop.run_in_executor(
            None,
            lambda: _ingest_paper_search_media(
                media_db=db,
                url=f"vixra:{vid}",
                title=title_for_db,
                media_type="pdf",
                content=content_for_db,
                keywords=kw_list or [],
                prompt=custom_prompt,
                analysis_content=analysis_for_db,
                safe_metadata=smj,
                transcription_model='Imported',
                author=author_for_db,
                overwrite=False,
                chunk_options={"method": chunk_method or "sentences", "max_size": chunk_size, "overlap": chunk_overlap} if perform_chunking else None,
                chunks=chunks_for_sql,
            )
        )
        return {"message": msg, "media_id": media_id, "media_uuid": media_uuid}  # noqa: TRY300
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        _raise_http_error_from_exception(e)
        logger.error(f"viXra ingest error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="viXra ingest failed") from e


@router.get(
    "/vixra/search",
    response_model=GenericPageSearchResponse,
    summary="Best-effort viXra search (HTML scrape)",
    tags=["paper-search"],
)
async def vixra_search(
    term: str = Query(..., min_length=2, description="Search term"),
    page: int = Query(1, ge=1),
    results_per_page: int = Query(10, ge=1, le=50),
):
    loop = asyncio.get_running_loop()
    try:
        items, total, err = await loop.run_in_executor(None, Vixra.search, term, page, results_per_page)
        if err:
            _handle_provider_error(err)
        items = items or []
        total_count = int(total or 0)
        total_pages = math.ceil(total_count / results_per_page) if total_count else 0
        return GenericPageSearchResponse(
            query_echo={"term": term},
            items=[GenericPaper(**it) for it in items],
            total_results=total,
            page=page,
            results_per_page=results_per_page,
            total_pages=total_pages,
            pagination=build_page_pagination_meta(
                page=page,
                per_page=results_per_page,
                total=total,
                total_pages=total_pages,
            ),
        )
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected viXra search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_PROVIDER_UNEXPECTED_DETAIL) from e

@router.get(
    "/arxiv/by-id",
    response_model=ArxivPaper,
    summary="Get arXiv paper by arXiv ID",
    tags=["paper-search"],
)
async def paper_search_arxiv_by_id(
    id: str = Query(..., description="arXiv ID, e.g., 1706.03762"),
):
    loop = asyncio.get_running_loop()
    try:
        item, error_message = await loop.run_in_executor(None, Arxiv.get_arxiv_by_id, id)
        if error_message:
            logger.error(f"arXiv by-id provider error: {error_message}")
            if "timed out" in error_message.lower():
                raise HTTPException(status_code=504, detail=_PROVIDER_TIMEOUT_DETAIL)  # noqa: TRY301
            raise HTTPException(status_code=502, detail=_PROVIDER_ERROR_DETAIL)  # noqa: TRY301
        if not item:
            raise HTTPException(status_code=404, detail="Paper not found for arXiv ID")  # noqa: TRY301
        return ArxivPaper(**item)
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Unexpected arXiv by-id error")
        raise HTTPException(status_code=500, detail=_PROVIDER_UNEXPECTED_DETAIL) from e


@router.get(
    "/semantic-scholar/by-id",
    response_model=SemanticScholarPaper,
    summary="Get Semantic Scholar paper by paperId",
    tags=["paper-search"],
)
async def paper_search_semantic_scholar_by_id(
    paper_id: str = Query(..., description="Semantic Scholar paperId"),
):
    loop = asyncio.get_running_loop()
    try:
        data, error_message = await loop.run_in_executor(
            None,
            Semantic_Scholar.get_paper_details_semantic_scholar,
            paper_id,
        )
        if error_message:
            logger.error(f"Semantic Scholar by-id provider error: {error_message}")
            if "timed out" in error_message.lower():
                raise HTTPException(status_code=504, detail=_PROVIDER_TIMEOUT_DETAIL)  # noqa: TRY301
            if "HTTP Error" in error_message:
                nums = [int(s) for s in error_message.split() if s.isdigit() and 400 <= int(s) < 600]
                code = nums[0] if nums else 502
                raise HTTPException(status_code=code, detail=_PROVIDER_ERROR_DETAIL)  # noqa: TRY301
            raise HTTPException(status_code=502, detail=_PROVIDER_ERROR_DETAIL)  # noqa: TRY301
        if not data:
            raise HTTPException(status_code=404, detail="Paper not found for paperId")  # noqa: TRY301
        if data.get("openAccessPdf") is None:
            data.pop("openAccessPdf", None)
        return SemanticScholarPaper(**data)
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Unexpected Semantic Scholar by-id error")
        raise HTTPException(status_code=500, detail=_PROVIDER_UNEXPECTED_DETAIL) from e


@router.get(
    "/pubmed/by-id",
    response_model=PubMedPaper,
    summary="Get PubMed record by PMID (with abstract)",
    tags=["paper-search"],
)
async def paper_search_pubmed_by_id(
    pmid: str = Query(..., description="PubMed PMID, e.g., 12345678"),
):
    loop = asyncio.get_running_loop()
    try:
        item, error_message = await loop.run_in_executor(
            None,
            PubMed.get_pubmed_by_id,
            pmid,
        )
        if error_message:
            logger.error(f"PubMed by-id provider error: {error_message}")
            if "timed out" in error_message.lower():
                raise HTTPException(status_code=504, detail=_PROVIDER_TIMEOUT_DETAIL)  # noqa: TRY301
            if "http error" in error_message.lower():
                nums = [int(s) for s in error_message.split() if s.isdigit() and 400 <= int(s) < 600]
                code = nums[0] if nums else 502
                raise HTTPException(status_code=code, detail=_PROVIDER_ERROR_DETAIL)  # noqa: TRY301
        if not item:
            raise HTTPException(status_code=404, detail="Record not found for PMID")  # noqa: TRY301
        return PubMedPaper(**item)
    except HTTPException:
        raise
    except _PAPER_SEARCH_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Unexpected PubMed by-id error")
        raise HTTPException(status_code=500, detail=_PROVIDER_UNEXPECTED_DETAIL) from e

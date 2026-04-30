# Server_API/app/api/v1/endpoints/research.py
#
#
# Imports
import asyncio
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from threading import Lock
from typing import Any, Optional

#
# 3rd-Party Libraries
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user

#
# Local Imports
from tldw_Server_API.app.api.v1.endpoints._pagination_utils import build_page_pagination_meta
from tldw_Server_API.app.api.v1.schemas.research_schemas import (
    ArxivPaper,
    ArxivSearchRequestForm,
    ArxivSearchResponse,
    SemanticScholarPaper,
    SemanticScholarSearchRequestForm,
    SemanticScholarSearchResponse,
)
from tldw_Server_API.app.api.v1.schemas.websearch_schemas import (
    WebSearchAggregateResponse,
    WebSearchRawResponse,
    WebSearchRequest,
)
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user, User
from tldw_Server_API.app.core.Chat.Chat_Deps import ChatConfigurationError
from tldw_Server_API.app.core.DB_Management.media_db.api import get_media_repository
from tldw_Server_API.app.core.DB_Management.media_db.api import managed_media_database
from tldw_Server_API.app.core.Third_Party.Arxiv import convert_xml_to_markdown, fetch_arxiv_xml, search_arxiv_custom_api
from tldw_Server_API.app.core.Third_Party.Semantic_Scholar import search_papers_semantic_scholar
from tldw_Server_API.app.core.Web_Scraping import WebSearch_APIs

_WEBSEARCH_EXECUTOR: ThreadPoolExecutor | None = None
_WEBSEARCH_EXECUTOR_LOCK = Lock()


def generate_and_search(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Wrapper that defers to the web search module, keeping monkeypatches effective."""
    return WebSearch_APIs.generate_and_search(*args, **kwargs)


async def analyze_and_aggregate(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Async wrapper around the aggregation stage for the same monkeypatch semantics."""
    return await WebSearch_APIs.analyze_and_aggregate(*args, **kwargs)


def _get_websearch_executor() -> ThreadPoolExecutor:
    """Lazily initialize and return the shared websearch executor."""
    global _WEBSEARCH_EXECUTOR
    if _WEBSEARCH_EXECUTOR is None:
        with _WEBSEARCH_EXECUTOR_LOCK:
            if _WEBSEARCH_EXECUTOR is None:
                _WEBSEARCH_EXECUTOR = ThreadPoolExecutor(
                    max_workers=4,
                    thread_name_prefix="ThreadPoolExecutor",
                )
    return _WEBSEARCH_EXECUTOR


def shutdown_websearch_executor(*, wait: bool = True, cancel_futures: bool = True) -> None:
    """Shut down the shared websearch executor, if initialized."""
    global _WEBSEARCH_EXECUTOR
    executor = _WEBSEARCH_EXECUTOR
    _WEBSEARCH_EXECUTOR = None
    if executor is not None:
        executor.shutdown(wait=wait, cancel_futures=cancel_futures)


def _get_phase1_fatal_error(phase1: dict[str, Any]) -> HTTPException | None:
    """Convert fatal provider failures with no results into an HTTP error."""
    web_results = phase1.get("web_search_results_dict")
    if not isinstance(web_results, dict):
        return None

    results = web_results.get("results")
    if isinstance(results, list) and results:
        return None

    error_message = web_results.get("processing_error") or web_results.get("error")
    if not error_message:
        warnings = web_results.get("warnings")
        if isinstance(warnings, list):
            for warning in warnings:
                if isinstance(warning, dict) and warning.get("phase") == "provider" and warning.get("message"):
                    error_message = warning["message"]
                    break

    if not error_message:
        return None

    return HTTPException(status_code=502, detail="Websearch failed")

#
#########################################################################################################################
#
# FastAPI Router

router = APIRouter()

##############################################################################
#
# Arxiv Search Endpoint
@router.get(
    "/arxiv-search",
    response_model=ArxivSearchResponse,
    summary="DEPRECATED: Use /api/v1/paper-search/arxiv",
    tags=["Research Tools - arXiv"],
    deprecated=True,
)
async def arxiv_search_endpoint(
        # user: User = Depends(get_request_user), # Optional: if endpoint requires auth
        search_params: ArxivSearchRequestForm = Depends(),
        Token: Optional[str] = Query(None, alias="X-Token"),
        db: Any = Depends(get_media_db_for_user),
):
    """
    Searches the arXiv repository based on query, author, and year.
    Returns a paginated list of matching papers.
    """
    logger.warning("Deprecated endpoint /api/v1/research/arxiv-search called. Prefer /api/v1/paper-search/arxiv")
    start_index = (search_params.page - 1) * search_params.results_per_page

    logger.info(
        f"arXiv search requested with params: query='{search_params.query}', author='{search_params.author}', year='{search_params.year}', page={search_params.page}, size={search_params.results_per_page}")

    # Use run_in_executor if search_arxiv_custom_api uses synchronous requests
    loop = asyncio.get_running_loop()
    try:
        # search_arxiv_custom_api now returns: papers_list, total_results_from_api, error_message
        papers_list, total_results_from_api, error_message = await loop.run_in_executor(
            None,
            search_arxiv_custom_api,
            search_params.query,
            search_params.author,
            search_params.year,
            start_index,
            search_params.results_per_page
        )

        if error_message:
            # Log the error from the library and raise an HTTPException
            logger.error(f"arXiv search failed: {error_message}")
            if "timed out" in error_message.lower():
                raise HTTPException(status_code=504, detail=f"arXiv API request timed out: {error_message}")
            else:
                raise HTTPException(status_code=502, detail=f"arXiv API error: {error_message}")

        if papers_list is None:  # Should be caught by error_message but as a safeguard
            logger.error("arXiv search returned None for papers_list without an error message.")
            raise HTTPException(status_code=500, detail="arXiv search failed to return paper data.")

    except HTTPException:  # Re-raise if it's already an HTTPException
        raise
    except Exception as e:
        logger.error("Unexpected error during arXiv search execution")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while searching arXiv",
        ) from e

    total_pages_calculated = math.ceil(
        total_results_from_api / search_params.results_per_page) if search_params.results_per_page > 0 else 0
    if total_results_from_api == 0:
        total_pages_calculated = 0  # Ensure total_pages is 0 if no results

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
        total_pages=total_pages_calculated,
        pagination=build_page_pagination_meta(
            page=search_params.page,
            per_page=search_params.results_per_page,
            total=total_results_from_api,
            total_pages=total_pages_calculated,
        ),
    )

# FIXME - This needs to be updated/Integrated
def process_and_ingest_arxiv_paper(paper_id, additional_keywords):
    try:
        xml_content = fetch_arxiv_xml(paper_id)
        markdown, title, authors, categories = convert_xml_to_markdown(xml_content)

        keywords = f"arxiv,{','.join(categories)}"
        if additional_keywords:
            keywords += f",{additional_keywords}"

        # Persist via MediaDatabase instance
        with managed_media_database(
            client_id="research_ingest",
            initialize=False,
        ) as db_instance:
            get_media_repository(db_instance).add_media_with_keywords(
                url=f"https://arxiv.org/abs/{paper_id}",
                title=title,
                media_type='document',
                content=markdown,
                keywords=[kw.strip() for kw in keywords.split(',') if kw.strip()] if isinstance(keywords, str) else (keywords or []),
                prompt='No prompt for arXiv papers',
                analysis_content='arXiv paper ingested from XML',
                transcription_model='None',
                author=', '.join(authors),
                ingestion_date=datetime.now().strftime('%Y-%m-%d')
            )

        return f"arXiv paper '{title}' ingested successfully."
    except Exception as e:
        logger.error("Error processing arXiv paper")
        return "Error processing arXiv paper"
#
# End of arxiv_search_endpoint
###########################################################################


##############################################################################
#
# Semantic Scholar Search Endpoint

@router.get(
    "/semantic-scholar-search",
    response_model=SemanticScholarSearchResponse,
    summary="DEPRECATED: Use /api/v1/paper-search/semantic-scholar",
    tags=["Research Tools - Semantic Scholar"],
    deprecated=True,
)
async def semantic_scholar_search_endpoint(
        # user: User = Depends(get_request_user), # Optional: if endpoint requires auth
        search_params: SemanticScholarSearchRequestForm = Depends()
):
    """
    Searches the Semantic Scholar database for papers based on various criteria.
    """
    logger.warning("Deprecated endpoint /api/v1/research/semantic-scholar-search called. Prefer /api/v1/paper-search/semantic-scholar")
    offset = (search_params.page - 1) * search_params.results_per_page

    logger.info(
        f"Semantic Scholar search: query='{search_params.query}', page={search_params.page}, limit={search_params.results_per_page}")

    loop = asyncio.get_running_loop()
    try:
        # Call the modified search function from your library
        api_response_data, error_message = await loop.run_in_executor(
            None,
            search_papers_semantic_scholar,  # Use the renamed function
            search_params.query,
            offset,
            search_params.results_per_page,
            search_params.fields_of_study_list,
            search_params.publication_types_list,
            search_params.year_range,
            search_params.venue_list,
            search_params.min_citations,
            False  # open_access_only - client side filtering for now based on presence of openAccessPdf field
        )

        if error_message:
            logger.error(f"Semantic Scholar search failed: {error_message}")
            if "timed out" in error_message.lower():
                raise HTTPException(status_code=504, detail=f"Semantic Scholar API request timed out: {error_message}")
            elif "HTTP Error" in error_message:  # Check for HTTP errors
                # Attempt to parse status code if present
                status_code_match = [int(s) for s in error_message.split() if s.isdigit() and 400 <= int(s) < 600]
                api_status_code = status_code_match[0] if status_code_match else 502
                raise HTTPException(status_code=api_status_code, detail=f"Semantic Scholar API error: {error_message}")
            else:
                raise HTTPException(status_code=502, detail=f"Semantic Scholar API error: {error_message}")

        if api_response_data is None or "data" not in api_response_data:
            logger.error("Semantic Scholar search returned invalid data structure or None without error message.")
            raise HTTPException(status_code=500, detail="Semantic Scholar search returned invalid data.")

    except HTTPException:  # Re-raise if it's already an HTTPException from above
        raise
    except Exception as e:
        logger.error("Unexpected error during Semantic Scholar search execution")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while searching Semantic Scholar",
        ) from e

    total_results_api = api_response_data.get("total", 0)
    actual_offset_api = api_response_data.get("offset", 0)  # S2 returns the actual offset used
    next_offset_api = api_response_data.get("next")  # This is the *next* offset value, not next page number

    # Filter for open access if requested (client-side, as S2 search API is limited here)
    # papers_to_return = api_response_data["data"]
    # if search_params.open_access_only:
    #     papers_to_return = [p for p in papers_to_return if p.get("openAccessPdf")]

    parsed_papers = []
    for paper_data in api_response_data.get("data", []):
        # The S2 API might return null for openAccessPdf if not available
        if paper_data.get("openAccessPdf") is None:
            paper_data.pop("openAccessPdf", None)  # Remove if None to avoid Pydantic error if model expects dict
        parsed_papers.append(SemanticScholarPaper(**paper_data))

    total_pages_calculated = math.ceil(
        total_results_api / search_params.results_per_page) if search_params.results_per_page > 0 else 0
    if total_results_api == 0:
        total_pages_calculated = 0

    return SemanticScholarSearchResponse(
        query_echo={
            "query": search_params.query,
            "fields_of_study": search_params.fields_of_study_list,
            "publication_types": search_params.publication_types_list,
            "year_range": search_params.year_range,
            "venue": search_params.venue_list,
            "min_citations": search_params.min_citations,
            # "open_access_only": search_params.open_access_only,
        },
        items=parsed_papers,
        total_results=total_results_api,
        offset=actual_offset_api,
        limit=search_params.results_per_page,  # The limit we requested
        next_offset=next_offset_api,
        page=search_params.page,  # The page number we calculated offset from
        total_pages=total_pages_calculated,
        pagination=build_page_pagination_meta(
            page=search_params.page,
            per_page=search_params.results_per_page,
            total=total_results_api,
            total_pages=total_pages_calculated,
        ),
    )

#
# End of semantic_scholar_search_endpoint
###############################################################################


##############################################################################
#
# Web Search Endpoint

from typing import Union


@router.post(
    "/websearch",
    response_model=Union[WebSearchRawResponse, WebSearchAggregateResponse],
    summary="Web search across providers with optional aggregation",
    tags=["research"],
)
async def websearch_endpoint(
    payload: WebSearchRequest,
    request: Request,
    current_user: User = Depends(get_request_user),
    db: Any = Depends(get_media_db_for_user),
):
    """
    Runs the websearch pipeline: optional subqueries + provider search. If aggregate=True,
    runs relevance evaluation and generates a final answer.
    """
    try:
        resolved_relevance_llm = payload.relevance_analysis_llm
        if payload.aggregate and not resolved_relevance_llm:
            # Default to final answer provider when available; otherwise fail fast with a clear
            # configuration error instead of silently degrading to empty relevance output.
            if payload.final_answer_llm:
                resolved_relevance_llm = payload.final_answer_llm
            else:
                raise ValueError(
                    "aggregate=true requires `relevance_analysis_llm` or `final_answer_llm`."
                )

        search_params = {
            "engine": payload.engine,
            "content_country": payload.content_country,
            "search_lang": payload.search_lang,
            "output_lang": payload.output_lang,
            "result_count": payload.result_count,
            "date_range": payload.date_range,
            "safesearch": payload.safesearch,
            "site_blacklist": payload.site_blacklist,
            "exactTerms": payload.exactTerms,
            "excludeTerms": payload.excludeTerms,
            "filter": payload.filter,
            "geolocation": payload.geolocation,
            "search_result_language": payload.search_result_language,
            "sort_results_by": payload.sort_results_by,
            "searx_url": payload.searx_url,
            "searx_json_mode": payload.searx_json_mode,
            "google_domain": payload.google_domain,
            "boards": payload.boards,
            "max_threads_per_board": payload.max_threads_per_board,
            "max_archived_threads_per_board": payload.max_archived_threads_per_board,
            "include_archived": payload.include_archived,
            "subquery_generation": payload.subquery_generation,
            "subquery_generation_llm": payload.subquery_generation_llm,
            "user_review": payload.user_review,
            "relevance_analysis_llm": resolved_relevance_llm,
            "final_answer_llm": payload.final_answer_llm,
        }

        # Run potentially blocking provider calls off the event loop
        # Use an explicit ThreadPoolExecutor for stable thread naming expected by tests
        loop = asyncio.get_running_loop()
        phase1 = await loop.run_in_executor(
            _get_websearch_executor(),
            generate_and_search,
            payload.query,
            search_params,
        )

        phase1_fatal_error = _get_phase1_fatal_error(phase1)
        if phase1_fatal_error is not None:
            raise phase1_fatal_error

        if payload.aggregate:
            # Cancellation propagates if client disconnects
            cancel_event = asyncio.Event()

            async def _watch_disconnect():
                if request is None:
                    return
                try:
                    # Poll for client disconnect, then signal cancellation
                    while not cancel_event.is_set():
                        try:
                            if await request.is_disconnected():
                                cancel_event.set()
                                break
                        except Exception:
                            # If the underlying server doesn't support disconnect checks, stop monitoring
                            break
                        await asyncio.sleep(0.5)
                except Exception as monitor_error:
                    # Never let the monitor crash the endpoint
                    logger.debug("Research request disconnect monitor failed; stopping monitor task", exc_info=monitor_error)

            monitor = asyncio.create_task(_watch_disconnect())
            aggregated = await analyze_and_aggregate(
                phase1["web_search_results_dict"],
                phase1["sub_query_dict"],
                search_params,
                cancel_event=cancel_event,
            )
            monitor.cancel()
            # Merge in sub_query_dict for completeness
            aggregated["sub_query_dict"] = phase1["sub_query_dict"]
            return aggregated

        return phase1

    except HTTPException:
        raise
    except (ValueError, ChatConfigurationError) as e:
        logger.warning(f"websearch endpoint config validation failed: {e}")
        raise HTTPException(status_code=422, detail=f"Websearch configuration error: {str(e)}") from e
    except Exception as e:
        logger.error("websearch endpoint failed")
        raise HTTPException(status_code=500, detail="Websearch failed") from e


#
# End of research.py
########################################################################################################################

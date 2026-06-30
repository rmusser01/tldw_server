# paper_search_schemas.py
# Pydantic models and FastAPI request forms for provider-specific paper search.

from typing import Any, Optional
from uuid import UUID

from fastapi import Query
from pydantic import BaseModel
from tldw_Server_API.app.api.v1.schemas.pagination import PagePaginationMeta


class BioRxivPaper(BaseModel):
    doi: str
    title: str
    authors: Optional[str] = None
    category: Optional[str] = None
    date: Optional[str] = None  # YYYY-MM-DD
    abstract: Optional[str] = None
    server: Optional[str] = None  # biorxiv | medrxiv
    version: Optional[int] = None
    url: Optional[str] = None
    pdf_url: Optional[str] = None


class BioRxivSearchResponse(BaseModel):
    query_echo: dict[str, Any]
    items: list[BioRxivPaper]
    total_results: int
    page: int
    results_per_page: int
    total_pages: int
    pagination: PagePaginationMeta


class BioRxivSearchRequestForm:
    def __init__(
        self,
        q: Optional[str] = Query(None, description="Keyword query to search titles/abstracts."),
        server: str = Query("biorxiv", description="Source server: biorxiv or medrxiv"),
        from_date: Optional[str] = Query(
            None, description="Start date YYYY-MM-DD. Defaults to last 30 days if omitted."
        ),
        to_date: Optional[str] = Query(
            None, description="End date YYYY-MM-DD. Defaults to today if omitted."
        ),
        category: Optional[str] = Query(None, description="Optional subject category to filter."),
        recent_days: Optional[int] = Query(None, ge=1, description="Use most recent N days (alternative to date range)."),
        recent_count: Optional[int] = Query(None, ge=1, description="Use most recent N posts (alternative to date range)."),
        page: int = Query(1, ge=1, description="Page number (1-indexed)."),
        results_per_page: int = Query(10, ge=1, le=100, description="Results per page (max 100)."),
    ):
        self.q = q
        self.server = server
        self.from_date = from_date
        self.to_date = to_date
        self.category = category
        self.recent_days = recent_days
        self.recent_count = recent_count
        self.page = page
        self.results_per_page = results_per_page


class BioRxivPublishedRecord(BaseModel):
    biorxiv_doi: str
    published_doi: Optional[str] = None
    published_journal: Optional[str] = None
    preprint_platform: Optional[str] = None
    preprint_title: Optional[str] = None
    preprint_authors: Optional[str] = None
    preprint_category: Optional[str] = None
    preprint_date: Optional[str] = None
    published_date: Optional[str] = None
    preprint_abstract: Optional[str] = None
    preprint_author_corresponding: Optional[str] = None
    preprint_author_corresponding_institution: Optional[str] = None


class BioRxivPubsSearchResponse(BaseModel):
    query_echo: dict[str, Any]
    items: list[BioRxivPublishedRecord]
    total_results: int
    page: int
    results_per_page: int
    total_pages: int
    pagination: PagePaginationMeta


class BioRxivPubsSearchRequestForm:
    def __init__(
        self,
        server: str = Query("biorxiv", description="Server: biorxiv or medrxiv"),
        from_date: Optional[str] = Query(None, description="YYYY-MM-DD start (if recent params omitted)"),
        to_date: Optional[str] = Query(None, description="YYYY-MM-DD end (if recent params omitted)"),
        recent_days: Optional[int] = Query(None, ge=1, description="Use most recent N days."),
        recent_count: Optional[int] = Query(None, ge=1, description="Use most recent N posts."),
        q: Optional[str] = Query(None, description="Client-side filter over title/abstract/authors"),
        include_abstracts: bool = Query(True, description="Include preprint_abstract field in results"),
        page: int = Query(1, ge=1),
        results_per_page: int = Query(10, ge=1, le=100),
    ):
        self.server = server
        self.from_date = from_date
        self.to_date = to_date
        self.recent_days = recent_days
        self.recent_count = recent_count
        self.q = q
        self.include_abstracts = include_abstracts
        self.page = page
        self.results_per_page = results_per_page


# End of paper_search_schemas.py

class PubMedPaper(BaseModel):
    pmid: str
    title: str
    authors: Optional[str] = None
    journal: Optional[str] = None
    pub_date: Optional[str] = None
    abstract: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    pmcid: Optional[str] = None
    pmc_url: Optional[str] = None
    pdf_url: Optional[str] = None


class PubMedSearchResponse(BaseModel):
    query_echo: dict[str, Any]
    items: list[PubMedPaper]
    total_results: int
    page: int
    results_per_page: int
    total_pages: int
    pagination: PagePaginationMeta


class PubMedSearchRequestForm:
    def __init__(
        self,
        q: str = Query(..., min_length=1, description="PubMed search query (supports normal PubMed syntax)."),
        from_year: Optional[int] = Query(None, ge=1800, le=2100, description="Filter by publication year start (YYYY)."),
        to_year: Optional[int] = Query(None, ge=1800, le=2100, description="Filter by publication year end (YYYY)."),
        free_full_text: bool = Query(False, description="Restrict to 'Free full text' articles."),
        page: int = Query(1, ge=1, description="Page number (1-indexed)."),
        results_per_page: int = Query(10, ge=1, le=100, description="Results per page (max 100)."),
    ):
        self.q = q
        self.from_year = from_year
        self.to_year = to_year
        self.free_full_text = free_full_text
        self.page = page
        self.results_per_page = results_per_page


# ---------------- PMC OAI-PMH Schemas ----------------

class PMCOAIHeader(BaseModel):
    identifier: Optional[str] = None
    datestamp: Optional[str] = None
    setSpecs: Optional[list[str]] = None


class PMCOAIMetadata(BaseModel):
    title: Optional[str] = None
    creators: Optional[list[str]] = None
    identifiers: Optional[list[str]] = None
    rights: Optional[list[str]] = None
    license_urls: Optional[list[str]] = None
    date: Optional[str] = None
    pmcid: Optional[str] = None
    pmid: Optional[str] = None
    doi: Optional[str] = None


class PMCOAIRecord(BaseModel):
    header: Optional[PMCOAIHeader] = None
    metadata: Optional[PMCOAIMetadata] = None
    raw_xml: Optional[str] = None


class PMCOAIListResponse(BaseModel):
    query_echo: dict[str, Any]
    items: list[PMCOAIRecord]
    resumption_token: Optional[str] = None


class PMCOAIIdentifiersResponse(BaseModel):
    query_echo: dict[str, Any]
    items: list[PMCOAIHeader]
    resumption_token: Optional[str] = None


class PMCOAISet(BaseModel):
    setSpec: Optional[str] = None
    setName: Optional[str] = None


class PMCOAIListSetsResponse(BaseModel):
    query_echo: dict[str, Any]
    items: list[PMCOAISet]
    resumption_token: Optional[str] = None


class PMCOAIIdentifyResponse(BaseModel):
    info: dict[str, Any]


# ---------------- PMC OA Web Service Schemas ----------------

class PMCOALink(BaseModel):
    format: Optional[str] = None
    updated: Optional[str] = None
    href: Optional[str] = None


class PMCOARecord(BaseModel):
    id: str
    citation: Optional[str] = None
    license: Optional[str] = None
    retracted: Optional[bool] = None
    links: list[PMCOALink] = []


class PMCOAQueryResponse(BaseModel):
    query_echo: dict[str, Any]
    items: list[PMCOARecord]
    resumption_token: Optional[str] = None


class PMCOAIdentifyResponse(BaseModel):
    info: dict[str, Any]


# ---------------- Additional Provider Schemas (Scaffold) ----------------

class GenericPaper(BaseModel):
    id: Optional[str] = None
    title: str
    authors: Optional[str] = None
    journal: Optional[str] = None
    pub_date: Optional[str] = None
    abstract: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    provider: Optional[str] = None


class GenericSearchResponse(BaseModel):
    query_echo: dict[str, Any]
    items: list[GenericPaper]
    total_results: int
    page: int
    results_per_page: int
    total_pages: int


class GenericPageSearchResponse(BaseModel):
    query_echo: dict[str, Any]
    items: list[GenericPaper]
    total_results: int
    page: int
    results_per_page: int
    total_pages: int
    pagination: PagePaginationMeta


class IEEESearchRequestForm:
    def __init__(
        self,
        q: Optional[str] = Query(None, description="Keyword query"),
        from_year: Optional[int] = Query(None, ge=1800, le=2100),
        to_year: Optional[int] = Query(None, ge=1800, le=2100),
        publication_title: Optional[str] = Query(None, description="IEEE publication title filter"),
        authors: Optional[str] = Query(None, description="Author name(s)"),
        page: int = Query(1, ge=1),
        results_per_page: int = Query(10, ge=1, le=100),
    ):
        self.q = q
        self.from_year = from_year
        self.to_year = to_year
        self.publication_title = publication_title
        self.authors = authors
        self.page = page
        self.results_per_page = results_per_page


class SimpleVenueSearchForm:
    def __init__(
        self,
        q: Optional[str] = Query(None, description="Keyword query"),
        venue: Optional[str] = Query(None, description="Venue or journal name filter"),
        from_year: Optional[int] = Query(None, ge=1800, le=2100),
        to_year: Optional[int] = Query(None, ge=1800, le=2100),
        page: int = Query(1, ge=1),
        results_per_page: int = Query(10, ge=1, le=100),
    ):
        self.q = q
        self.venue = venue
        self.from_year = from_year
        self.to_year = to_year
        self.page = page
        self.results_per_page = results_per_page


class DOIRequestForm:
    def __init__(self, doi: str = Query(..., min_length=3)):
        self.doi = doi


# ---------------- RePEc / CitEc Schemas ----------------

class RepecCitationsResponse(BaseModel):
    handle: str
    cited_by: int
    cites: int
    uri: Optional[str] = None
    date: Optional[str] = None

# ---------------- BioRxiv Reports Schemas ----------------

class BioRxivFunderPaper(BaseModel):
    doi: str
    title: str
    authors: Optional[str] = None
    category: Optional[str] = None
    date: Optional[str] = None
    abstract: Optional[str] = None
    server: Optional[str] = None
    version: Optional[int] = None
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    funder: Optional[Any] = None  # raw funder block from API (name/id/id-type/award)


class BioRxivFunderSearchResponse(BaseModel):
    query_echo: dict[str, Any]
    items: list[BioRxivFunderPaper]
    total_results: int
    page: int
    results_per_page: int
    total_pages: int
    pagination: PagePaginationMeta


class BioRxivFunderSearchRequestForm:
    def __init__(
        self,
        server: str = Query("biorxiv", description="Server: biorxiv or medrxiv"),
        ror_id: str = Query(..., min_length=5, description="Funder ROR ID (final 9-char segment)"),
        from_date: Optional[str] = Query(None, description="YYYY-MM-DD start"),
        to_date: Optional[str] = Query(None, description="YYYY-MM-DD end"),
        category: Optional[str] = Query(None, description="Optional category filter"),
        recent_days: Optional[int] = Query(None, ge=1, description="Most recent N days"),
        recent_count: Optional[int] = Query(None, ge=1, description="Most recent N posts"),
        page: int = Query(1, ge=1),
        results_per_page: int = Query(10, ge=1, le=100),
    ):
        self.server = server
        self.ror_id = ror_id
        self.from_date = from_date
        self.to_date = to_date
        self.category = category
        self.recent_days = recent_days
        self.recent_count = recent_count
        self.page = page
        self.results_per_page = results_per_page


class BioRxivPublisherSearchRequestForm:
    def __init__(
        self,
        publisher_prefix: str = Query(..., min_length=4, description="Publisher DOI prefix, e.g., '10.15252'"),
        from_date: Optional[str] = Query(None, description="YYYY-MM-DD start"),
        to_date: Optional[str] = Query(None, description="YYYY-MM-DD end"),
        recent_days: Optional[int] = Query(None, ge=1, description="Most recent N days"),
        recent_count: Optional[int] = Query(None, ge=1, description="Most recent N articles"),
        page: int = Query(1, ge=1),
        results_per_page: int = Query(10, ge=1, le=100),
    ):
        self.publisher_prefix = publisher_prefix
        self.from_date = from_date
        self.to_date = to_date
        self.recent_days = recent_days
        self.recent_count = recent_count
        self.page = page
        self.results_per_page = results_per_page


class BioRxivPubSearchRequestForm:
    def __init__(
        self,
        from_date: Optional[str] = Query(None, description="YYYY-MM-DD start"),
        to_date: Optional[str] = Query(None, description="YYYY-MM-DD end"),
        recent_days: Optional[int] = Query(None, ge=1, description="Most recent N days"),
        recent_count: Optional[int] = Query(None, ge=1, description="Most recent N articles"),
        page: int = Query(1, ge=1),
        results_per_page: int = Query(10, ge=1, le=100),
    ):
        self.from_date = from_date
        self.to_date = to_date
        self.recent_days = recent_days
        self.recent_count = recent_count
        self.page = page
        self.results_per_page = results_per_page


class BioRxivSummaryItem(BaseModel):
    month: str
    new_papers: Optional[int] = None
    new_papers_cumulative: Optional[int] = None
    revised_papers: Optional[int] = None
    preprint_date: Optional[str] = None
    revised_papers_cumulative: Optional[int] = None


class BioRxivSummaryResponse(BaseModel):
    query_echo: dict[str, Any]
    items: list[BioRxivSummaryItem]


class BioRxivSummaryRequestForm:
    def __init__(self, interval: str = Query("m", description="Interval: m (monthly) or y (yearly)")):
        self.interval = interval


class BioRxivUsageItem(BaseModel):
    month: str
    abstract_views: Optional[int] = None
    full_text_views: Optional[int] = None
    pdf_downloads: Optional[int] = None
    abstract_cumulative: Optional[int] = None
    full_text_cumulative: Optional[int] = None
    pdf_cumulative: Optional[int] = None


class BioRxivUsageResponse(BaseModel):
    query_echo: dict[str, Any]
    items: list[BioRxivUsageItem]


class BioRxivUsageRequestForm:
    def __init__(self, interval: str = Query("m", description="Interval: m (monthly) or y (yearly)")):
        self.interval = interval


# ---------------- ChemRxiv Schemas ----------------

class ChemRxivSearchRequestForm:
    def __init__(
        self,
        term: Optional[str] = Query(None, description="Search term"),
        skip: int = Query(0, ge=0, description="Offset for results"),
        limit: int = Query(10, ge=1, le=50, description="Results per page (max 50)"),
        sort: Optional[str] = Query("PUBLISHED_DATE_DESC", description="Sort order (ChemRxiv enum)"),
        author: Optional[str] = Query(None, description="Author filter"),
        searchDateFrom: Optional[str] = Query(None, description="YYYY-MM-DD or ISO"),
        searchDateTo: Optional[str] = Query(None, description="YYYY-MM-DD or ISO"),
        searchLicense: Optional[str] = Query(None, description="License filter"),
        categoryIds: Optional[str] = Query(None, description="Comma-separated category IDs"),
        subjectIds: Optional[str] = Query(None, description="Comma-separated subject IDs"),
    ):
        self.term = term
        self.skip = skip
        self.limit = limit
        self.sort = sort
        self.author = author
        self.searchDateFrom = searchDateFrom
        self.searchDateTo = searchDateTo
        self.searchLicense = searchLicense
        self.categoryIds_list = [s.strip() for s in categoryIds.split(',')] if categoryIds else None
        self.subjectIds_list = [s.strip() for s in subjectIds.split(',')] if subjectIds else None


# ---------------- IACR Schemas ----------------

class IacrConferenceResponse(BaseModel):
    query_echo: dict[str, Any]
    data: dict[str, Any]


# ---------------- Ingest Batch Schemas ----------------

class IngestBatchItem(BaseModel):
    doi: Optional[str] = None
    pdf_url: Optional[str] = None
    pmcid: Optional[str] = None
    arxiv_id: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None
    keywords: Optional[list[str]] = None


class IngestBatchRequest(BaseModel):
    items: list[IngestBatchItem]
    perform_chunking: bool = True
    parser: Optional[str] = "pymupdf4llm"
    chunk_method: Optional[str] = None
    chunk_size: int = 500
    chunk_overlap: int = 200
    perform_analysis: bool = False
    api_name: Optional[str] = None
    custom_prompt: Optional[str] = None
    system_prompt: Optional[str] = None
    enable_ocr: bool = False
    ocr_backend: Optional[str] = None
    ocr_lang: Optional[str] = "eng"
    ocr_dpi: int = 300
    ocr_mode: Optional[str] = "fallback"
    ocr_min_page_text_chars: int = 40
    ocr_output_format: Optional[str] = None
    ocr_prompt_preset: Optional[str] = None


class IngestBatchResultItem(BaseModel):
    doi: Optional[str] = None
    pdf_url: Optional[str] = None
    pmcid: Optional[str] = None
    arxiv_id: Optional[str] = None
    success: bool
    media_id: Optional[int] = None
    media_uuid: Optional[UUID] = None
    error: Optional[str] = None


class IngestBatchResponse(BaseModel):
    results: list[IngestBatchResultItem]
    succeeded: int
    failed: int


# ---------------- OSF Schemas ----------------

class OSFSearchRequestForm:
    def __init__(
        self,
        term: Optional[str] = Query(None, description="Search text (OSF 'q')"),
        provider: Optional[str] = Query(None, description="Preprints provider key (e.g., 'osf', 'eartharxiv', 'socarxiv')"),
        from_date: Optional[str] = Query(None, description="Filter by created date >= YYYY-MM-DD"),
        page: int = Query(1, ge=1),
        results_per_page: int = Query(10, ge=1, le=100),
    ):
        self.term = term
        self.provider = provider
        self.from_date = from_date
        self.page = page
        self.results_per_page = results_per_page

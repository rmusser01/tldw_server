# research_schemas.py
#
#
# Imports
from typing import Any, Optional

#
# # 3rd-Party Libraries
from fastapi import Query
from pydantic import BaseModel

#
# Local Imports
from tldw_Server_API.app.api.v1.schemas.pagination import PagePaginationMeta
from tldw_Server_API.app.core.Third_Party.Arxiv import ARXIV_DEFAULT_PAGE_SIZE
from tldw_Server_API.app.core.Third_Party.Semantic_Scholar import FIELDS_OF_STUDY_CHOICES, PUBLICATION_TYPE_CHOICES


#
#########################################################################################################################
#
# Functions:
# --- Pydantic Models for arXiv Search ---
class ArxivPaper(BaseModel):
    id: Optional[str] = None
    title: Optional[str] = None
    authors: Optional[str] = None
    published_date: Optional[str] = None
    abstract: Optional[str] = None
    pdf_url: Optional[str] = None


class ArxivSearchResponse(BaseModel):
    query_echo: dict[str, Any]
    items: list[ArxivPaper]
    total_results: int
    page: int
    results_per_page: int
    total_pages: int
    pagination: PagePaginationMeta


class ArxivSearchRequestForm:
    def __init__(
            self,
            query: Optional[str] = Query(None, description="Search query for title, abstract, etc."),
            author: Optional[str] = Query(None, description="Filter by author name."),
            year: Optional[str] = Query(None, description="Filter by submission year (e.g., '2023')."),
            page: int = Query(1, ge=1, description="Page number for pagination."),
            results_per_page: int = Query(ARXIV_DEFAULT_PAGE_SIZE, ge=1, le=100,
                                          description="Number of results per page.")  # Max 100 for sanity
    ):
        self.query = query
        self.author = author
        self.year = year
        self.page = page
        self.results_per_page = results_per_page

#
# End of Arxiv Schemas
###############################################################################


###############################################################################
#
# --- Pydantic Models for Semantic Scholar Search ---

class SemanticScholarAuthor(BaseModel):
    authorId: Optional[str] = None
    name: str

class SemanticScholarOpenAccessPdf(BaseModel):
    url: str
    status: str # e.g., "GREEN", "GOLD"

class SemanticScholarPaper(BaseModel):
    paperId: str
    title: str
    abstract: Optional[str] = None
    year: Optional[int] = None
    citationCount: Optional[int] = None
    authors: Optional[list[SemanticScholarAuthor]] = []
    venue: Optional[str] = None
    openAccessPdf: Optional[SemanticScholarOpenAccessPdf] = None
    url: Optional[str] = None # Link to Semantic Scholar page
    publicationTypes: Optional[list[str]] = None
    publicationDate: Optional[str] = None # Typically YYYY-MM-DD
    externalIds: Optional[dict[str, str]] = None # e.g., {"DOI": "...", "ArXiv": "..."}


class SemanticScholarSearchResponse(BaseModel):
    query_echo: dict[str, Any]
    items: list[SemanticScholarPaper]
    total_results: int
    offset: int
    limit: int
    next_offset: Optional[int] = None # Semantic Scholar provides 'next' which is the next offset
    page: int # Calculated for user convenience
    total_pages: int # Calculated
    pagination: PagePaginationMeta

class SemanticScholarSearchRequestForm:
    def __init__(
        self,
        query: str = Query(..., min_length=1, description="Search query (e.g., paper title, keywords, author name)."),
        fields_of_study: Optional[str] = Query(None, description=f"Comma-separated list of fields of study. Choices include: {', '.join(FIELDS_OF_STUDY_CHOICES[:5])}, etc."),
        publication_types: Optional[str] = Query(None, description=f"Comma-separated list of publication types. Choices include: {', '.join(PUBLICATION_TYPE_CHOICES[:5])}, etc."),
        year_range: Optional[str] = Query(None, description="Filter by publication year or range (e.g., '2020' or '2019-2021')."),
        venue: Optional[str] = Query(None, description="Comma-separated list of publication venues (e.g., 'Nature,CVPR')."),
        min_citations: Optional[int] = Query(None, ge=0, description="Minimum number of citations."),
        # open_access_only: bool = Query(False, description="Filter for papers with Open Access PDFs. (Note: this filters client-side after fetching based on openAccessPdf field presence)"), # This is tricky with S2 API
        page: int = Query(1, ge=1, description="Page number for pagination (1-indexed)."),
        results_per_page: int = Query(10, ge=1, le=100, description="Number of results per page.")
    ):
        self.query = query
        self.fields_of_study_list = [fos.strip() for fos in fields_of_study.split(',')] if fields_of_study else None
        self.publication_types_list = [pt.strip() for pt in publication_types.split(',')] if publication_types else None
        self.year_range = year_range
        self.venue_list = [v.strip() for v in venue.split(',')] if venue else None
        self.min_citations = min_citations
        # self.open_access_only = open_access_only
        self.page = page
        self.results_per_page = results_per_page

#
# End of Semantic Scholar Schemas
#################################################################


#
# End of research_schemas.py
########################################################################################################################

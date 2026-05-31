import pytest
from pydantic import BaseModel, Field, ValidationError, model_validator

from tldw_Server_API.app.api.v1.endpoints._pagination_utils import (
    build_link_header,
    build_offset_pagination_meta,
    build_page_pagination_meta,
    build_pagination_link_header,
    resolve_page_pagination_metadata,
)
from tldw_Server_API.app.api.v1.schemas.admin_schemas import WebhookListResponse
from tldw_Server_API.app.api.v1.schemas.character_schemas import CharacterListQueryResponse
from tldw_Server_API.app.api.v1.schemas.pagination import (
    CursorPaginationMeta,
    OffsetPaginationMeta,
    PagePaginationMeta,
)
from tldw_Server_API.app.api.v1.schemas import pagination as pagination_schemas
from tldw_Server_API.app.api.v1.schemas.reading_schemas import ReadingItemsListResponse
from tldw_Server_API.app.api.v1.schemas.writing_manuscript_schemas import ManuscriptProjectListResponse


class OffsetAliasResponse(BaseModel):
    """Test-only response proving default offset alias behavior."""

    items: list[int]
    limit: int | None = Field(default=None, ge=1)
    offset: int | None = Field(default=None, ge=0)
    total: int | None = Field(default=None, ge=0)
    has_more: bool | None = None
    next_offset: int | None = Field(default=None, ge=0)
    pagination: OffsetPaginationMeta

    @model_validator(mode="after")
    def default_aliases(self) -> "OffsetAliasResponse":
        return pagination_schemas.default_offset_pagination_aliases(self)


class StrictOffsetAliasResponse(OffsetAliasResponse):
    """Test-only response proving strict offset alias validation behavior."""

    @model_validator(mode="after")
    def validate_aliases(self) -> "StrictOffsetAliasResponse":
        return pagination_schemas.validate_offset_pagination_aliases(self)


class PageAliasResponse(BaseModel):
    """Test-only response proving default page alias behavior."""

    items: list[int]
    page: int | None = Field(default=None, ge=1)
    per_page: int | None = Field(default=None, ge=1)
    total: int | None = Field(default=None, ge=0)
    total_pages: int | None = Field(default=None, ge=0)
    has_more: bool | None = None
    pagination: PagePaginationMeta

    @model_validator(mode="after")
    def default_aliases(self) -> "PageAliasResponse":
        return pagination_schemas.default_page_pagination_aliases(self)


class CursorAliasResponse(BaseModel):
    """Test-only response proving default cursor alias behavior."""

    items: list[int]
    limit: int | None = Field(default=None, ge=1)
    cursor: str | None = None
    next_cursor: str | None = None
    has_more: bool | None = None
    pagination: CursorPaginationMeta

    @model_validator(mode="after")
    def default_aliases(self) -> "CursorAliasResponse":
        return pagination_schemas.default_cursor_pagination_aliases(self)


class OptionalPaginationAliasResponse(BaseModel):
    """Test-only response proving alias helpers tolerate missing metadata."""

    items: list[int]
    has_more: bool | None = None
    next_offset: int | None = Field(default=None, ge=0)
    pagination: OffsetPaginationMeta | None = None

    @model_validator(mode="after")
    def default_aliases(self) -> "OptionalPaginationAliasResponse":
        return pagination_schemas.default_offset_pagination_aliases(self)


class StrictOptionalPaginationAliasResponse(OptionalPaginationAliasResponse):
    """Test-only response proving strict helper also tolerates missing metadata."""

    @model_validator(mode="after")
    def validate_aliases(self) -> "StrictOptionalPaginationAliasResponse":
        return pagination_schemas.validate_offset_pagination_aliases(self)


def test_build_offset_pagination_meta_computes_has_more_and_next_offset() -> None:
    """Offset metadata reports a next page when returned count does not exhaust total."""
    pagination = build_offset_pagination_meta(
        limit=25,
        offset=50,
        total=123,
        count=25,
    )

    assert pagination.mode == "offset"
    assert pagination.limit == 25
    assert pagination.offset == 50
    assert pagination.total == 123
    assert pagination.has_more is True
    assert pagination.next_offset == 75


def test_build_offset_pagination_meta_ends_without_next_offset() -> None:
    """Offset metadata omits next_offset when the requested page reaches the total."""
    pagination = build_offset_pagination_meta(
        limit=25,
        offset=100,
        total=123,
        count=23,
    )

    assert pagination.has_more is False
    assert pagination.next_offset is None


def test_build_page_pagination_meta_derives_has_more_from_total() -> None:
    """Page metadata uses total and per_page when total_pages is not provided."""
    pagination = build_page_pagination_meta(page=2, per_page=10, total=25)

    assert pagination.has_more is True
    assert pagination.total_pages is None


def test_resolve_page_pagination_metadata_safely_defaults_bad_numbers() -> None:
    """Page pagination normalization does not raise on malformed storage metadata."""
    metadata = resolve_page_pagination_metadata(
        {
            "page": "not-a-number",
            "per_page": "0",
            "total": "-5",
            "total_pages": "also-bad",
        },
        page=2,
        per_page=10,
        item_count=7,
    )

    assert metadata == {
        "page": 2,
        "per_page": 1,
        "total": 0,
        "total_pages": 0,
    }


def test_pagination_meta_modes_are_fixed_discriminators() -> None:
    """Pagination mode fields reject contradictory discriminator values."""
    with pytest.raises(ValidationError):
        OffsetPaginationMeta(mode="cursor", limit=10, offset=0, total=20, has_more=True, next_offset=10)
    with pytest.raises(ValidationError):
        CursorPaginationMeta(mode="offset", limit=10, cursor=None, next_cursor=None, has_more=False)
    with pytest.raises(ValidationError):
        PagePaginationMeta(mode="offset", page=1, per_page=10, total=20, total_pages=2, has_more=True)


def test_character_list_backfills_has_more_alias_when_omitted() -> None:
    """Character list responses derive omitted top-level aliases from canonical metadata."""
    response = CharacterListQueryResponse(
        items=[],
        total=2,
        page=1,
        page_size=1,
        pagination=build_offset_pagination_meta(limit=1, offset=0, total=2, count=1),
    )

    assert response.has_more is True
    assert response.next_offset == 1


def test_reading_list_rejects_contradictory_pagination_aliases() -> None:
    """Reading responses reject drift between legacy aliases and canonical metadata."""
    pagination = build_offset_pagination_meta(limit=1, offset=0, total=2, count=1)

    with pytest.raises(ValidationError, match="has_more alias mismatch"):
        ReadingItemsListResponse(
            items=[],
            total=2,
            page=1,
            size=1,
            has_more=False,
            pagination=pagination,
        )

    with pytest.raises(ValidationError, match="next_offset alias mismatch"):
        ReadingItemsListResponse(
            items=[],
            total=2,
            page=1,
            size=1,
            next_offset=99,
            pagination=pagination,
        )


def test_manuscript_project_list_rejects_negative_pagination_aliases() -> None:
    """Manuscript list responses constrain newly restored offset fields."""
    pagination = build_offset_pagination_meta(limit=10, offset=0, total=0, count=0)

    with pytest.raises(ValidationError):
        ManuscriptProjectListResponse(projects=[], total=0, limit=0, offset=0, pagination=pagination)
    with pytest.raises(ValidationError):
        ManuscriptProjectListResponse(projects=[], total=0, limit=10, offset=-1, pagination=pagination)


def test_webhook_list_backfills_legacy_limit_offset_aliases() -> None:
    """Webhook list responses preserve legacy top-level limit and offset fields."""
    pagination = build_offset_pagination_meta(limit=25, offset=50, total=100, count=25)

    response = WebhookListResponse(items=[], total=100, pagination=pagination)

    assert response.limit == 25
    assert response.offset == 50
    assert response.has_more is True
    assert response.next_offset == 75


def test_shared_offset_alias_helper_defaults_all_present_alias_fields() -> None:
    """Shared offset helper backfills every legacy alias field declared by a response."""
    pagination = OffsetPaginationMeta(limit=10, offset=20, total=35, has_more=True, next_offset=30)

    response = OffsetAliasResponse(items=[], pagination=pagination)

    assert response.limit == 10
    assert response.offset == 20
    assert response.total == 35
    assert response.has_more is True
    assert response.next_offset == 30


def test_shared_offset_alias_validator_accepts_matching_explicit_aliases() -> None:
    """Strict offset helper accepts aliases that already agree with canonical metadata."""
    pagination = OffsetPaginationMeta(limit=10, offset=20, total=35, has_more=True, next_offset=30)

    response = StrictOffsetAliasResponse(
        items=[],
        limit=10,
        offset=20,
        total=35,
        has_more=True,
        next_offset=30,
        pagination=pagination,
    )

    assert response.pagination == pagination


def test_shared_offset_alias_validator_rejects_contradictory_aliases() -> None:
    """Strict offset helper rejects drift between legacy aliases and canonical metadata."""
    pagination = OffsetPaginationMeta(limit=10, offset=20, total=35, has_more=True, next_offset=30)

    with pytest.raises(ValidationError, match="limit alias mismatch"):
        StrictOffsetAliasResponse(items=[], limit=5, pagination=pagination)


def test_shared_alias_helpers_skip_missing_canonical_pagination() -> None:
    """Shared alias helpers remain compatibility-safe when metadata is absent."""
    defaulted = OptionalPaginationAliasResponse(items=[])
    strict = StrictOptionalPaginationAliasResponse(items=[])

    assert defaulted.has_more is None
    assert defaulted.next_offset is None
    assert strict.has_more is None
    assert strict.next_offset is None


def test_shared_page_alias_helper_defaults_all_present_alias_fields() -> None:
    """Shared page helper backfills every legacy page alias field declared by a response."""
    pagination = PagePaginationMeta(page=2, per_page=25, total=80, total_pages=4, has_more=True)

    response = PageAliasResponse(items=[], pagination=pagination)

    assert response.page == 2
    assert response.per_page == 25
    assert response.total == 80
    assert response.total_pages == 4
    assert response.has_more is True


def test_shared_cursor_alias_helper_defaults_all_present_alias_fields() -> None:
    """Shared cursor helper backfills every legacy cursor alias field declared by a response."""
    pagination = CursorPaginationMeta(limit=25, cursor="input-token", next_cursor="next-token", has_more=True)

    response = CursorAliasResponse(items=[], pagination=pagination)

    assert response.limit == 25
    assert response.cursor == "input-token"
    assert response.next_cursor == "next-token"
    assert response.has_more is True


def test_build_pagination_link_header_uses_offset_metadata() -> None:
    """Link header construction uses canonical offset pagination metadata."""
    pagination = build_offset_pagination_meta(
        limit=25,
        offset=50,
        total=123,
        count=25,
    )

    link_header = build_pagination_link_header(
        base_path="/api/v1/skills",
        common_params=[("include_hidden", "false")],
        pagination=pagination,
    )

    assert link_header == (
        '</api/v1/skills?include_hidden=false&limit=25&offset=75>; rel="next", '
        '</api/v1/skills?include_hidden=false&limit=25&offset=25>; rel="prev", '
        '</api/v1/skills?include_hidden=false&limit=25&offset=0>; rel="first"'
    )


def test_build_link_header_offset_compatibility_signature_stays_stable() -> None:
    """Legacy offset Link header callers keep the same output shape."""
    link_header = build_link_header(
        base_path="/api/v1/workflows/runs",
        common_params=[("status", "running")],
        limit=25,
        offset=50,
        has_more=True,
    )

    assert link_header == (
        '</api/v1/workflows/runs?status=running&limit=25&offset=75>; rel="next", '
        '</api/v1/workflows/runs?status=running&limit=25&offset=25>; rel="prev", '
        '</api/v1/workflows/runs?status=running&limit=25&offset=0>; rel="first"'
    )

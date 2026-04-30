from tldw_Server_API.app.api.v1.endpoints._pagination_utils import (
    build_link_header,
    build_offset_pagination_meta,
    build_pagination_link_header,
)


def test_build_offset_pagination_meta_computes_has_more_and_next_offset():
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


def test_build_offset_pagination_meta_ends_without_next_offset():
    pagination = build_offset_pagination_meta(
        limit=25,
        offset=100,
        total=123,
        count=23,
    )

    assert pagination.has_more is False
    assert pagination.next_offset is None


def test_build_pagination_link_header_uses_offset_metadata():
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


def test_build_link_header_offset_compatibility_signature_stays_stable():
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

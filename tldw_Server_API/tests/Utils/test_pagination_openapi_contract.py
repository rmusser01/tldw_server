from __future__ import annotations

from pathlib import Path


MATRIX_PATH = Path("Docs/Design/Pagination_Completion_Matrix.md")
UNRESOLVED_STATUSES = {"migration-candidate", "needs-confirmation"}
CANONICAL_STATUS = "canonical-present-or-custom-pagination"
HTTP_METHODS = ("GET", "POST", "PUT", "PATCH", "DELETE")


def _clean_cell(value: str) -> str:
    return value.strip().strip("`")


def _matrix_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in MATRIX_PATH.read_text().splitlines():
        if not line.startswith(tuple(f"| `{method}`" for method in HTTP_METHODS)):
            continue
        columns = [column.strip() for column in line.strip().strip("|").split("|")]
        if len(columns) < 12:
            continue
        rows.append(
            {
                "method": _clean_cell(columns[0]),
                "path": _clean_cell(columns[1]),
                "endpoint": _clean_cell(columns[3]),
                "response_model": _clean_cell(columns[4]),
                "response_fields": _clean_cell(columns[6]),
                "status": _clean_cell(columns[8]),
                "tranche": _clean_cell(columns[10]),
            }
        )
    return rows


def _component_name(response_model: str) -> str | None:
    if response_model == "unknown" or response_model.startswith("list["):
        return None
    if "[" in response_model or "|" in response_model:
        return None
    return response_model.rsplit(".", 1)[-1]


def _pagination_paths(response_fields: str) -> list[str]:
    return [
        field.strip()
        for field in response_fields.split(",")
        if field.strip() == "pagination" or field.strip().endswith(".pagination")
    ]


def _resolve_schema(schema: dict[str, object], components: dict[str, dict[str, object]]) -> dict[str, object]:
    ref = schema.get("$ref")
    if isinstance(ref, str):
        return components[ref.rsplit("/", 1)[-1]]

    for combiner in ("anyOf", "oneOf", "allOf"):
        candidates = schema.get(combiner)
        if isinstance(candidates, list):
            for candidate in candidates:
                if isinstance(candidate, dict) and candidate.get("type") != "null":
                    return _resolve_schema(candidate, components)

    return schema


def _has_property_path(
    schema: dict[str, object],
    path: str,
    components: dict[str, dict[str, object]],
) -> bool:
    current = schema
    for part in path.split("."):
        current = _resolve_schema(current, components)
        properties = current.get("properties")
        if not isinstance(properties, dict) or part not in properties:
            return False
        next_schema = properties[part]
        if not isinstance(next_schema, dict):
            return False
        current = next_schema
    return True


def test_pagination_matrix_has_no_unresolved_candidates() -> None:
    """Every inventoried route should be either canonicalized or explicitly exempt."""
    unresolved = [
        f"{row['endpoint']} ({row['path']}) -> {row['status']}"
        for row in _matrix_rows()
        if row["status"] in UNRESOLVED_STATUSES
    ]

    assert unresolved == []


def test_openapi_exposes_canonical_pagination_components() -> None:
    """The public schema should expose all canonical pagination metadata shapes."""
    from tldw_Server_API.app.main import app

    components = app.openapi()["components"]["schemas"]

    expected_properties = {
        "OffsetPaginationMeta": {"mode", "limit", "offset", "total", "has_more", "next_offset"},
        "PagePaginationMeta": {"mode", "page", "per_page", "total", "total_pages", "has_more"},
        "CursorPaginationMeta": {"mode", "limit", "cursor", "next_cursor", "has_more"},
    }
    for component_name, fields in expected_properties.items():
        assert component_name in components
        assert fields <= set(components[component_name].get("properties", {}))


def test_canonical_matrix_response_models_expose_pagination_when_openapi_resolvable() -> None:
    """Canonical matrix rows should point at response models exposing their matrix path."""
    from tldw_Server_API.app.main import app

    components = app.openapi()["components"]["schemas"]
    mismatches: list[str] = []
    for row in _matrix_rows():
        if row["status"] != CANONICAL_STATUS:
            continue
        component_name = _component_name(row["response_model"])
        if component_name is None or component_name not in components:
            continue
        paths = _pagination_paths(row["response_fields"])
        if not paths or not any(
            _has_property_path(components[component_name], path, components)
            for path in paths
        ):
            mismatches.append(f"{row['endpoint']} -> {component_name}")

    assert mismatches == []

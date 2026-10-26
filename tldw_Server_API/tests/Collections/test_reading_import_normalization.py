from tldw_Server_API.app.core.Collections.reading_importers import (
    ReadingImportItem,
    normalize_import_items,
)


def test_normalize_import_items_merges_equivalent_urls():
    items = [
        ReadingImportItem(
            url="https://example.com/article?utm_source=a",
            title=None,
            tags=["AI"],
            status="saved",
            favorite=False,
            notes=None,
            read_at=None,
            metadata={"import_source": "pocket"},
        ),
        ReadingImportItem(
            url="https://example.com/article?utm_source=b",
            title="Explicit Title",
            tags=["ai", "Research"],
            status="read",
            favorite=True,
            notes="important",
            read_at="2026-02-20T10:00:00+00:00",
            metadata={"import_source": "instapaper"},
        ),
    ]

    normalized = normalize_import_items(items)
    assert len(normalized) == 1
    merged = normalized[0]
    assert merged.url == "https://example.com/article"
    assert merged.title == "Explicit Title"
    assert merged.status == "read"
    assert merged.favorite is True
    assert merged.tags == ["AI", "Research"]
    assert merged.notes == "important"
    assert merged.metadata["import_normalized_url"] == "https://example.com/article"


def test_normalize_import_items_derives_title_from_url_when_missing():
    items = [
        ReadingImportItem(
            url="https://example.com/deep-dive_on-testing",
            title=None,
            tags=[],
            status=None,
            favorite=False,
            notes=None,
            read_at=None,
            metadata={},
        )
    ]

    normalized = normalize_import_items(items)
    assert len(normalized) == 1
    assert normalized[0].title == "deep dive on testing"
    assert normalized[0].status == "saved"


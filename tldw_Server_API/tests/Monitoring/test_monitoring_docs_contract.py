from __future__ import annotations

from pathlib import Path


def test_monitoring_docs_describe_current_notification_contract() -> None:
    product_doc = Path("Docs/Product/Completed/Topic_Monitoring_Watchlists.md").read_text(
        encoding="utf-8"
    )
    readme_doc = Path("tldw_Server_API/app/core/Monitoring/README.md").read_text(
        encoding="utf-8"
    )

    lowered_product = product_doc.lower()
    assert "best-effort webhook/email attempts" in lowered_product
    assert "generic notifications use the jsonl sink plus optional webhook dispatch" in lowered_product
    assert "mutation responses include the authoritative merged alert state" in lowered_product
    assert "`read` marks the runtime alert as read without setting `acknowledged_at`" in lowered_product

    lowered_readme = readme_doc.lower()
    assert "generic notifications only use the jsonl sink plus optional webhook dispatch" in lowered_readme
    assert "`flush_digest()` currently clears buffered items and returns the count only" in lowered_readme
    assert "public alert mutation endpoints return the authoritative merged alert state" in lowered_readme
    assert "`acknowledge` records `acknowledged_at`" in lowered_readme

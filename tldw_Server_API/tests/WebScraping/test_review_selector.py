from tldw_Server_API.app.core.Web_Scraping.WebSearch_APIs import review_and_select_results


def test_review_and_select_results_selector():
    data = {
        "results": [
            {"title": "keep", "url": "u1", "content": "a"},
            {"title": "drop", "url": "u2", "content": "b"},
        ]
    }

    sel = lambda r: r.get("title") == "keep"
    selected = review_and_select_results(data, selector=sel)
    assert list(selected.keys()) == ["0"]
    assert selected["0"]["title"] == "keep"

    # No selector should keep all
    selected_all = review_and_select_results(data)
    assert len(selected_all) == 2


def test_review_and_select_results_preserves_mapping_keys_without_selector():
    data = {
        "keep": {"title": "keep", "url": "u1", "content": "a"},
        "drop": {"title": "drop", "url": "u2", "content": "b"},
    }

    selected = review_and_select_results(data)

    assert list(selected.keys()) == ["keep", "drop"]
    assert selected["keep"]["title"] == "keep"


def test_review_and_select_results_preserves_mapping_keys_with_selector():
    data = {
        "keep": {"title": "keep", "url": "u1", "content": "a"},
        "drop": {"title": "drop", "url": "u2", "content": "b"},
    }

    selected = review_and_select_results(data, selector=lambda result: result.get("title") == "keep")

    assert list(selected.keys()) == ["keep"]
    assert selected["keep"]["url"] == "u1"

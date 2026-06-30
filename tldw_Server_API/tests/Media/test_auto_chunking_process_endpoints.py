import pytest


pytestmark = pytest.mark.unit


def test_process_pdfs_auto_chunking_adds_plan_metadata(
    client_with_single_user,
    bypass_api_limits,
    monkeypatch,
):
    client, _ = client_with_single_user

    import tldw_Server_API.app.api.v1.endpoints.media as media_mod
    import tldw_Server_API.app.core.Chunking as chunking_mod

    captured: dict[str, object] = {}

    async def _stub_process_pdf_task(**kwargs):
        captured["initial_chunk_method"] = kwargs.get("chunk_method")
        return {
            "status": "Success",
            "content": "# Heading\n\nThis PDF content has enough structure to prefer sections.",
            "metadata": {"title": "stub-pdf"},
        }

    def _stub_chunking(text, options):
        captured["final_chunk_options"] = dict(options)
        return [{"text": text, "metadata": {"chunk_num": 0}}]

    monkeypatch.setattr(media_mod.pdf_lib, "process_pdf_task", _stub_process_pdf_task)
    monkeypatch.setattr(chunking_mod, "improved_chunking_process", _stub_chunking)

    response = client.post(
        "/api/v1/media/process-pdfs",
        data={
            "perform_chunking": "true",
            "chunking_mode": "auto",
            "auto_chunking_goal": "navigation_summary",
            "auto_chunking_use_llm": "true",
            "chunk_method": "words",
            "chunk_size": "333",
            "chunk_overlap": "1",
        },
        files=[("files", ("paper.pdf", b"%PDF-1.4\n", "application/pdf"))],
    )

    assert response.status_code == 200, response.text
    result = response.json()["results"][0]
    plan = result["metadata"]["chunking_plan"]
    assert plan["mode"] == "auto"
    assert plan["goal"] == "navigation_summary"
    assert plan["method"] == "structure_aware"
    assert plan["used_llm"] is False
    assert "ai_assist_unavailable" in plan["fallback_reason"]
    assert captured["final_chunk_options"]["method"] == "structure_aware"
    assert captured["final_chunk_options"]["max_size"] != 333

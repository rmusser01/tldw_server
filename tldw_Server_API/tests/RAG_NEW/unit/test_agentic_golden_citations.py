import json
from pathlib import Path

import pytest

import tldw_Server_API.app.core.RAG.rag_service.agentic_chunker as ac
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document


def _make_doc(doc_id: str, text: str, title: str) -> Document:
    return Document(id=doc_id, content=text, metadata={"title": title}, source=DataSource.MEDIA_DB, score=0.9)


def _load_first_fixture(path_rel: str):
    p = Path(path_rel)
    data = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return data[0]


@pytest.mark.asyncio
async def test_golden_sentence_level_citations_offsets(monkeypatch):
    # Load NQ-like fixture
    fx = _load_first_fixture("tldw_Server_API/tests/fixtures/rag_agentic/nq_golden.jsonl")
    ctx = fx["contexts"][0]
    doc = _make_doc(ctx["id"], ctx["text"], ctx["title"])  # contains exact sentences

    class FakeRetriever:
        def __init__(self, *args, **kwargs):
            pass

        async def retrieve(self, *args, **kwargs):
            return [doc]

    monkeypatch.setattr(ac, "MultiDatabaseRetriever", FakeRetriever)

    # Patch generator to echo exact sentence from the source so offsets should be exact
    class FakeAnswerGenerator:
        def __init__(self, *args, **kwargs):
            pass

        async def generate(self, *, query: str, context: str, prompt_template=None, max_tokens=None, temperature=None):  # noqa: ARG002
            # Return two sentences, first repeats exact first sentence of ctx
            return {"answer": "We ran 42 experiments. The findings were consistent across trials."}

    # Patch the generation module's AnswerGenerator (agentic_chunker imports it inside the function)
    import tldw_Server_API.app.core.RAG.rag_service.generation as gen_mod
    monkeypatch.setattr(gen_mod, "AnswerGenerator", FakeAnswerGenerator, raising=False)

    res = await ac.agentic_rag_pipeline(
        query=fx["question"],
        sources=["media_db"],
        search_mode="fts",
        agentic=ac.AgenticConfig(top_k_docs=1),
        enable_generation=True,
        require_hard_citations=True,
        enable_numeric_fidelity=True,
        numeric_fidelity_behavior="continue",
        debug_mode=True,
    )

    md = res.metadata or {}
    # Hard citations present
    hc = md.get("hard_citations")
    assert isinstance(hc, dict) and isinstance(hc.get("sentences"), list)
    sents = hc.get("sentences")
    assert len(sents) >= 2
    # Verify offsets slice exact text from the synthetic chunk (agentic hard citations map to synthetic doc)
    syn = res.documents[0]
    chunk = syn.content if not isinstance(syn, dict) else syn.get("content", "")
    for entry in sents:
        txt = entry.get("text")
        cites = entry.get("citations") or []
        if not txt or not cites:
            continue
        c0 = cites[0]
        st, en = int(c0.get("start", 0)), int(c0.get("end", 0))
        assert st >= 0 and en > st
        # Slice must equal full sentence or a contiguous substring (if windowed)
        segment = (chunk or "")[st:en]
        assert segment.strip() in {txt.strip(), txt.strip()[: len(segment.strip())]}

    # Numeric fidelity should include '42'
    nf = md.get("numeric_fidelity") or {}
    assert any("42" in s for s in (nf.get("present") or []))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer,supported",
    [
        ("Residual connections help gradient flow in deep networks.", True),
        ("Residual connections help gradient flow. They enable deeper networks.", False),
    ],
    ids=["exact-source-span", "unsupported-claims"],
)
async def test_golden_multihop_merge_and_citations(monkeypatch, answer, supported):
    # Load HotpotQA-like fixture with two contexts
    fx = _load_first_fixture("tldw_Server_API/tests/fixtures/rag_agentic/hotpotqa_golden.jsonl")
    ctxs = fx["contexts"]
    docs = [_make_doc(c["id"], c["text"], c["title"]) for c in ctxs]

    class FakeRetriever:
        def __init__(self, *args, **kwargs):
            pass

        async def retrieve(self, *args, **kwargs):
            return docs

    monkeypatch.setattr(ac, "MultiDatabaseRetriever", FakeRetriever)

    # Exercise exact evidence and unsupported claims through the same pipeline
    class FakeAnswerGenerator:
        def __init__(self, *args, **kwargs):
            pass

        async def generate(self, *, query: str, context: str, prompt_template=None, max_tokens=None, temperature=None):  # noqa: ARG002
            return {"answer": answer}

    import tldw_Server_API.app.core.RAG.rag_service.generation as gen_mod
    monkeypatch.setattr(gen_mod, "AnswerGenerator", FakeAnswerGenerator, raising=False)

    res = await ac.agentic_rag_pipeline(
        query=fx["question"],
        sources=["media_db"],
        search_mode="fts",
        agentic=ac.AgenticConfig(top_k_docs=2, enable_tools=True, enable_query_decomposition=True, subgoal_max=2),
        enable_generation=True,
        require_hard_citations=True,
        debug_mode=True,
    )

    md = res.metadata or {}
    hc = md.get("hard_citations") or {}
    assert (hc.get("coverage", 0.0) > 0.0) is supported
    # Ensure the assembled chunk contains the cited sentence
    syn = res.documents[0]
    chunk = syn.content if not isinstance(syn, dict) else syn.get("content", "")
    assert "Residual connections help gradient flow" in chunk

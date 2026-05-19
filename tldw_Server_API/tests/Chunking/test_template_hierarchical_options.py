from tldw_Server_API.app.core.Chunking.templates import (
    TemplateProcessor,
    ChunkingTemplate,
    TemplateStage,
)


def test_template_hierarchical_forwards_method_options():

    template = ChunkingTemplate(
        name="hierarchical_options_test",
        base_method="sentences",
        stages=[
            TemplateStage(
                name="chunk",
                operations=[
                    {
                        "method": "sentences",
                        "max_size": 1,
                        "overlap": 0,
                        "config": {"hierarchical": True},
                        "params": {"combine_short": True, "min_sentence_length": 20},
                    }
                ],
            )
        ],
    )

    processor = TemplateProcessor()
    text = "Hi. Ok. This is a longer sentence."
    chunks = processor.process_template(text, template)

    assert chunks, "Expected hierarchical chunking to return chunks"
    assert chunks[0]["text"].startswith("Hi. Ok."), "Expected short sentences to be combined via method options"


def test_template_string_false_does_not_enable_hierarchical():
    template = ChunkingTemplate(
        name="hierarchical_string_false_test",
        base_method="words",
        stages=[
            TemplateStage(
                name="chunk",
                operations=[
                    {
                        "method": "words",
                        "max_size": 50,
                        "overlap": 0,
                        "config": {"hierarchical": "false"},
                    }
                ],
            )
        ],
    )

    processor = TemplateProcessor()
    text = "# Title\n\nPara one.\n\nPara two."
    chunks = processor.process_template(text, template)

    assert chunks
    assert all("paragraph_kind" not in (chunk.get("metadata") or {}) for chunk in chunks)
    assert all("ancestry_titles" not in (chunk.get("metadata") or {}) for chunk in chunks)

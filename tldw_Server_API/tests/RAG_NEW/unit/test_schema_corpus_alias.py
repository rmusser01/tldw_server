from tldw_Server_API.app.api.v1.schemas.rag_schemas_unified import UnifiedRAGRequest, UnifiedBatchRequest


def test_unified_request_corpus_alias_maps_to_index_namespace():


    req = UnifiedRAGRequest(query="q", sources=["media_db"], corpus="my_corpus")
    assert req.corpus == "my_corpus"
    assert req.index_namespace == "my_corpus"


def test_unified_batch_corpus_alias_maps_to_index_namespace():


    req = UnifiedBatchRequest(queries=["q1", "q2"], corpus="space_corpus")
    assert req.corpus == "space_corpus"
    assert req.index_namespace == "space_corpus"


def test_unified_request_accepts_all_public_knowledge_sources_and_aliases() -> None:
    req = UnifiedRAGRequest(
        query="canonical source contract",
        sources=[
            "media",
            "notes_db",
            "chat_history",
            "character_cards",
            "task_boards",
            "prompts_db",
            "worldbooks",
            "chat_dictionaries",
        ],
    )

    assert req.sources == [
        "media_db",
        "notes",
        "chats",
        "characters",
        "kanban",
        "prompts",
        "world_books",
        "dictionaries",
    ]


def test_unified_batch_accepts_all_public_knowledge_sources_and_aliases() -> None:
    req = UnifiedBatchRequest(
        queries=["canonical source contract"],
        sources=[
            "media",
            "notes_db",
            "chat_history",
            "character_cards",
            "task_boards",
            "prompts_db",
            "worldbooks",
            "chat_dictionaries",
        ],
    )

    assert req.sources == [
        "media_db",
        "notes",
        "chats",
        "characters",
        "kanban",
        "prompts",
        "world_books",
        "dictionaries",
    ]


def test_unified_request_sources_description_names_all_public_knowledge_sources() -> None:
    description = UnifiedRAGRequest.model_fields["sources"].description or ""

    for source in (
        "media_db",
        "notes",
        "chats",
        "characters",
        "kanban",
        "prompts",
        "world_books",
        "dictionaries",
    ):
        assert source in description

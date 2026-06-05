from pathlib import Path

TARGET_FILES = [
    "tldw_Server_API/app/api/v1/endpoints/prompt_studio/prompt_studio_optimization.py",
    "tldw_Server_API/app/api/v1/endpoints/prompt_studio/prompt_studio_projects.py",
    "tldw_Server_API/app/api/v1/endpoints/prompt_studio/prompt_studio_test_cases.py",
    "tldw_Server_API/app/api/v1/endpoints/paper_search.py",
    "tldw_Server_API/app/api/v1/endpoints/vector_stores_openai.py",
    "tldw_Server_API/app/api/v1/endpoints/prompt_studio/prompt_studio_prompts.py",
    "tldw_Server_API/app/api/v1/endpoints/chunking_templates.py",
    "tldw_Server_API/app/api/v1/endpoints/rag_unified.py",
    "tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_unified.py",
    "tldw_Server_API/app/api/v1/endpoints/setup.py",
    "tldw_Server_API/app/api/v1/endpoints/sync.py",
    "tldw_Server_API/app/core/Setup/install_manager.py",
    "tldw_Server_API/app/core/TTS/tts_config.py",
    "tldw_Server_API/app/core/TTS/adapter_registry.py",
    "tldw_Server_API/app/core/Embeddings/worker_config.py",
]


def test_no_deprecated_dict_usage():


    repo_root = Path(__file__).resolve().parents[3]
    offenders = []

    for rel_path in TARGET_FILES:
        file_path = repo_root / rel_path
        if not file_path.exists():
            raise AssertionError(f"Target file missing for lint: {rel_path}")
        contents = file_path.read_text(encoding="utf-8")
        if ".dict(" in contents:
            offenders.append(rel_path)

    assert not offenders, f"Deprecated .dict() usage found in: {', '.join(offenders)}"

import pytest

from tldw_Server_API.app.core.Prompt_Management.prompt_studio.job_processor import (
    JobProcessor,
)


class _RecordingPromptStudioDB:
    client_id = "prompt-studio-test-client"

    def __init__(self):
        self.updates = []

    def get_evaluation(self, evaluation_id):
        return {
            "id": evaluation_id,
            "prompt_id": 10,
            "project_id": 20,
        }

    def ensure_prompt_stub(self, **kwargs):
        return None

    def update_evaluation(self, evaluation_id, updates):
        self.updates.append((evaluation_id, dict(updates)))
        return dict(updates)


@pytest.mark.asyncio
async def test_evaluation_job_failure_stores_sanitized_error_message(monkeypatch):
    db = _RecordingPromptStudioDB()
    processor = JobProcessor(db)

    async def fail_execute_test_case(*args, **kwargs):
        raise RuntimeError("runner failed at /private/prompt-studio/eval.sqlite")

    monkeypatch.setattr(processor, "_execute_test_case", fail_execute_test_case)

    with pytest.raises(RuntimeError, match="runner failed"):
        await processor.process_evaluation_job(
            {
                "prompt_id": 10,
                "test_case_ids": [1],
                "model_configs": [{"provider": "test"}],
                "request_id": "req-test",
            },
            entity_id=99,
        )

    failed_update = db.updates[-1][1]
    assert failed_update["status"] == "failed"
    assert failed_update["error_message"] == "Prompt Studio evaluation job failed"
    assert "private" not in failed_update["error_message"]
    assert "eval.sqlite" not in failed_update["error_message"]

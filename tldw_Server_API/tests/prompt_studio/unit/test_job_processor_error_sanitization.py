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
            "test_case_ids": [1],
            "model_configs": [{"provider": "test"}],
        }

    def get_project(self, project_id, *, include_deleted=False):
        del include_deleted
        return {"id": project_id, "deleted": False}

    def get_prompt_with_project(self, prompt_id, *, include_deleted=False):
        del include_deleted
        return {
            "id": prompt_id,
            "project_id": 20,
            "deleted": False,
        }

    def get_test_case(self, test_case_id, *, include_deleted=False):
        del include_deleted
        return {
            "id": test_case_id,
            "project_id": 20,
            "deleted": False,
        }

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

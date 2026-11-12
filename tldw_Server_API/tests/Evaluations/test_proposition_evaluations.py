import os
import pytest
from fastapi.testclient import TestClient

# Import centralized test configuration
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from test_config import test_config

# Set up test environment before any app imports
test_config.setup_test_environment()
test_config.reset_settings()

from fastapi import FastAPI
from tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_unified import router as eval_router

# Build a minimal app that mounts only the evaluations router to avoid importing unrelated modules
app = FastAPI()
app.include_router(eval_router, prefix="/api/v1")


@pytest.fixture(scope="function")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth_headers():
    return test_config.get_auth_headers()


class TestPropositionEvaluationEndpoint:
    def test_proposition_evaluation_basic(self, client, auth_headers):
        payload = {
            "extracted": [
                "Alice founded Acme Corp in 2020",
                "Bob joined Acme in 2021",
            ],
            "reference": [
                "Alice founded Acme Corp in 2020",
                "Carol raised funding for Acme in 2022",
            ],
            "method": "jaccard",
            "threshold": 0.5
        }
        resp = client.post("/api/v1/evaluations/propositions", json=payload, headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert set(["precision", "recall", "f1"]).issubset(data.keys())
        assert data["total_extracted"] == 2
        assert data["total_reference"] == 2

    def test_proposition_evaluation_headers_present(self, client, auth_headers):

        payload = {
            "extracted": [
                "Alice founded Acme Corp in 2020",
                "Bob joined Acme in 2021",
            ],
            "reference": [
                "Alice founded Acme Corp in 2020",
                "Carol raised funding for Acme in 2022",
            ],
            "method": "jaccard",
            "threshold": 0.5,
        }
        resp = client.post("/api/v1/evaluations/propositions", json=payload, headers=auth_headers)
        assert resp.status_code == 200
        # Assert rate-limit headers are present
        headers = resp.headers
        for key in [
            "X-RateLimit-Tier",
            "X-RateLimit-PerMinute-Limit",
            "X-RateLimit-Reset",
        ]:
            assert key in headers, f"Missing header: {key}"


class TestPropositionRunFlow:
    def test_create_and_run_proposition_evaluation(self, client, auth_headers):
             # Create evaluation with inline dataset
        create_eval_req = {
            "name": "prop_eval_test",
            "description": "Proposition extraction evaluation",
            "eval_type": "proposition_extraction",
            "eval_spec": {
                "method": "jaccard",
                "threshold": 0.6
            },
            "dataset": [
                {
                    "input": {
                        "extracted": ["A is B", "C is D"],
                        "reference": ["A is B"]
                    },
                    "expected": {}
                },
                {
                    "input": {
                        "extracted": ["X equals Y"],
                        "reference": ["X equals Z"]
                    },
                    "expected": {}
                }
            ]
        }

        eval_resp = client.post("/api/v1/evaluations", json=create_eval_req, headers=auth_headers)
        assert eval_resp.status_code == 201
        eval_id = eval_resp.json()["id"]

        # Create a run
        run_req = {
            "target_model": "n/a",
            "config": {"batch_size": 10, "max_workers": 2}
        }
        run_resp = client.post(f"/api/v1/evaluations/{eval_id}/runs", json=run_req, headers=auth_headers)
        assert run_resp.status_code in [202, 200]
        run_data = run_resp.json()
        assert run_data.get("id") is not None
        assert run_data.get("eval_id") == eval_id
        run_id = run_data.get("id")

        # Poll for completion
        import time as _time
        for _ in range(50):  # up to ~5s
            status_resp = client.get(f"/api/v1/evaluations/runs/{run_id}", headers=auth_headers)
            assert status_resp.status_code == 200
            sdata = status_resp.json()
            if sdata.get("status") in ["completed", "failed", "cancelled"]:
                break
            _time.sleep(0.1)
        assert sdata.get("status") in ["completed", "failed", "cancelled"]

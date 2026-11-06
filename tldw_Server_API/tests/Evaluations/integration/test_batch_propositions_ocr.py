import os

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.main import app

pytestmark = [pytest.mark.integration, pytest.mark.evaluations]


@pytest.fixture(autouse=True)
def _setup_env(monkeypatch):
    # Open the admin gate; requests authenticate with the real single-user key.
    monkeypatch.setenv("EVALS_HEAVY_ADMIN_ONLY", "false")


@pytest.fixture(scope="function")
def client():
    with TestClient(app, headers={"X-API-KEY": get_settings().SINGLE_USER_API_KEY}) as c:
        yield c


def test_batch_propositions_basic(client):


    body = {
        "evaluation_type": "propositions",
        "items": [
            {
                "extracted": ["Alice founded Acme in 2020", "Bob joined in 2021"],
                "reference": ["Alice founded Acme in 2020"],
                "method": "jaccard",
                "threshold": 0.5,
            },
            {
                "extracted": ["X equals Y"],
                "reference": ["X equals Y"],
                "method": "semantic",
                "threshold": 0.7,
            },
        ],
        "parallel_workers": 1,
        "continue_on_error": True,
    }
    r = client.post("/api/v1/evaluations/batch", json=body)
    assert r.status_code == 200
    j = r.json()
    assert len(j.get("results", [])) == 2


def test_batch_ocr_text_items(client):


    body = {
        "evaluation_type": "ocr",
        "items": [
            {
                # For batch OCR, items list is embedded in each eval item
                "items": [
                    {"id": "d1", "extracted_text": "hello world", "ground_truth_text": "hello world"}
                ],
                "metrics": ["cer", "wer"],
            }
        ],
        "parallel_workers": 1,
    }
    r = client.post("/api/v1/evaluations/batch", json=body)
    assert r.status_code == 200
    j = r.json()
    assert len(j.get("results", [])) == 1

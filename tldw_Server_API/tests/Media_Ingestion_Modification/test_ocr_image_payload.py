"""vLLM OCR backends with *_USE_DATA_URL=0 send a temp-file path (TASK-13307).

The file must still exist when the request naming it is sent; with delete=True it was
unlinked first, every page came back "" and the run reported success with no content.
"""

from __future__ import annotations

import importlib
import os

import pytest

_BACKENDS = "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends."

CASES = [
    ("dots_ocr", "_ocr_via_vllm", {"DOTS_VLLM_URL": "http://localhost:8000", "DOTS_VLLM_USE_DATA_URL": "0"}, ()),
    ("hunyuan_ocr", "_ocr_via_vllm", {"HUNYUAN_VLLM_URL": "http://localhost:8000", "HUNYUAN_VLLM_USE_DATA_URL": "0"}, ()),
    ("dolphin_ocr", "_ocr_via_openai", {"DOLPHIN_URL": "http://localhost:8000", "DOLPHIN_USE_DATA_URL": "0"}, ()),
    ("nemotron_parse", "_ocr_via_vllm", {"NEMOTRON_VLLM_URL": "http://localhost:8000", "NEMOTRON_VLLM_USE_DATA_URL": "0"}, (True,)),
]


@pytest.mark.unit
@pytest.mark.parametrize(("module", "func", "env", "extra"), CASES, ids=[c[0] for c in CASES])
def test_path_based_image_survives_until_the_request(monkeypatch, module, func, env, extra):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    seen: dict[str, object] = {}

    def fake_fetch_json(*, method, url, json, timeout):
        path = json["messages"][0]["content"][1]["image_url"]["url"]
        seen["path"] = path
        with open(path, "rb") as fh:
            seen["bytes"] = fh.read()
        return {"choices": [{"message": {"content": "HELLO"}}]}

    monkeypatch.setattr("tldw_Server_API.app.core.http_client.fetch_json", fake_fetch_json)

    text = getattr(importlib.import_module(_BACKENDS + module), func)(b"png-bytes", "ocr", *extra)

    assert text == "HELLO"
    assert seen["bytes"] == b"png-bytes"
    assert not os.path.exists(seen["path"]), "temp image left behind after the request"

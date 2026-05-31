from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script_module():
    script_path = (
        Path(__file__).resolve().parents[3]
        / "Helper_Scripts"
        / "download_embedding_models.py"
    )
    spec = importlib.util.spec_from_file_location(
        "download_embedding_models_script",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_progress_messages_are_ascii_safe(monkeypatch, tmp_path, capsys) -> None:
    module = _load_script_module()

    calls: list[dict[str, object]] = []

    def _fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(kwargs["local_dir"])

    monkeypatch.setattr(module, "snapshot_download", _fake_snapshot_download)

    module.download_models(
        ["sentence-transformers/all-MiniLM-L6-v2"],
        tmp_path,
    )

    out = capsys.readouterr().out

    assert "Downloading 'sentence-transformers/all-MiniLM-L6-v2' into" in out
    assert "Ready:" in out
    assert all(ord(ch) < 128 for ch in out)
    assert len(calls) == 1

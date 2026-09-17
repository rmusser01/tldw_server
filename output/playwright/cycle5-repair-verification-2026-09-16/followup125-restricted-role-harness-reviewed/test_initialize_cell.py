"""Initialization proof tests with a fake module; no application imports or DBs."""

import importlib
import json
import runpy
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest


@pytest.mark.parametrize("outcome", ["normal", "exit-zero", "interrupt", "failure"])
def test_receipt_requires_normal_initializer_return(tmp_path, monkeypatch, outcome):
    source = tmp_path / "frozen"
    receipt = tmp_path / "attempt.completed.private.json"
    request = {
        "sourceRoot": str(source),
        "receiptPath": str(receipt),
        "token": uuid4().hex,
        "preparationHash": "matching-preparation",
    }
    monkeypatch.setenv("MATRIX_INIT_REQUEST", json.dumps(request))
    calls = []

    async def initialize(*, non_interactive):
        calls.append(non_interactive)
        if outcome == "exit-zero":
            raise SystemExit(0)
        if outcome == "interrupt":
            raise KeyboardInterrupt
        if outcome == "failure":
            raise RuntimeError("synthetic failure")

    def fake_import(name):
        assert name == "tldw_Server_API.app.core.AuthNZ.initialize"
        return SimpleNamespace(__file__=str(source / "initialize.py"), main=initialize)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    wrapper = Path(__file__).with_name("initialize-cell.py")
    if outcome == "normal":
        runpy.run_path(str(wrapper), run_name="__main__")
        assert json.loads(receipt.read_text()) == {
            "status": "completed",
            "token": request["token"],
            "preparationHash": request["preparationHash"],
        }
        assert receipt.stat().st_mode & 0o777 == 0o600
    else:
        expected = {"exit-zero": SystemExit, "interrupt": KeyboardInterrupt, "failure": RuntimeError}[outcome]
        with pytest.raises(expected):
            runpy.run_path(str(wrapper), run_name="__main__")
        assert not receipt.exists()
    assert calls == [True]


def test_initializer_from_outside_frozen_root_is_rejected_before_call(tmp_path, monkeypatch):
    receipt = tmp_path / "attempt.completed.private.json"
    monkeypatch.setenv(
        "MATRIX_INIT_REQUEST",
        json.dumps(
            {
                "sourceRoot": str(tmp_path / "frozen"),
                "receiptPath": str(receipt),
                "token": uuid4().hex,
                "preparationHash": "owner",
            }
        ),
    )
    monkeypatch.setattr(
        importlib, "import_module", lambda name: SimpleNamespace(__file__=str(tmp_path / "mutable/initialize.py"))
    )
    with pytest.raises(RuntimeError, match="frozen"):
        runpy.run_path(str(Path(__file__).with_name("initialize-cell.py")), run_name="__main__")
    assert not receipt.exists()

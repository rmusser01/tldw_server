"""
Watchlists suite configuration.

For these tests we need the full app profile (not the minimal test app) so
the watchlists router is registered. This fixture flips the relevant env vars
before any tests import the FastAPI app.
"""

import os
import sys
import importlib
import types
import importlib.machinery

import pytest

from tldw_Server_API.tests.helpers.app_main_state import (
    clear_app_main,
    restore_app_main,
    snapshot_app_main,
)

# Stub heavyweight audio deps before importing full app modules in this suite.
if "torch" not in sys.modules:
    _fake_torch = types.ModuleType("torch")
    _fake_torch.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)
    _fake_torch.Tensor = object
    _fake_torch.nn = types.SimpleNamespace(Module=object)
    sys.modules["torch"] = _fake_torch

if "faster_whisper" not in sys.modules:
    _fake_fw = types.ModuleType("faster_whisper")
    _fake_fw.__spec__ = importlib.machinery.ModuleSpec("faster_whisper", loader=None)

    class _StubWhisperModel:
        def __init__(self, *args, **kwargs):
            pass

    _fake_fw.WhisperModel = _StubWhisperModel
    _fake_fw.BatchedInferencePipeline = _StubWhisperModel
    sys.modules["faster_whisper"] = _fake_fw

if "transformers" not in sys.modules:
    _fake_tf = types.ModuleType("transformers")
    _fake_tf.__spec__ = importlib.machinery.ModuleSpec("transformers", loader=None)

    class _StubProcessor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class _StubModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    _fake_tf.AutoProcessor = _StubProcessor
    _fake_tf.Qwen2AudioForConditionalGeneration = _StubModel
    sys.modules["transformers"] = _fake_tf


@pytest.fixture(scope="session", autouse=True)
def enable_full_app_for_watchlists():
    orig_minimal = os.getenv("MINIMAL_TEST_APP")
    orig_disable = os.getenv("ROUTES_DISABLE")
    orig_enable = os.getenv("ROUTES_ENABLE")
    previous_main = snapshot_app_main()

    # Force full router set so /api/v1/watchlists/* endpoints are available
    os.environ["MINIMAL_TEST_APP"] = "0"

    # Ensure watchlists routes are not disabled and explicitly enabled
    disable = os.getenv("ROUTES_DISABLE", "")
    disable_parts = [p.strip() for p in disable.replace(" ", ",").split(",") if p.strip()]
    disable_parts = [p for p in disable_parts if p.lower() != "watchlists"]
    os.environ["ROUTES_DISABLE"] = ",".join(dict.fromkeys(disable_parts))

    enable = os.getenv("ROUTES_ENABLE", "")
    enable_parts = [p.strip() for p in enable.replace(" ", ",").split(",") if p.strip()]
    if "watchlists" not in {p.lower() for p in enable_parts}:
        enable_parts.append("watchlists")
    os.environ["ROUTES_ENABLE"] = ",".join(dict.fromkeys(enable_parts))

    # Reload app module so the new env is observed even if it was imported earlier
    clear_app_main()
    importlib.invalidate_caches()

    yield

    # Restore prior env so other suites keep their defaults
    if orig_minimal is None:
        os.environ.pop("MINIMAL_TEST_APP", None)
    else:
        os.environ["MINIMAL_TEST_APP"] = orig_minimal

    if orig_disable is None:
        os.environ.pop("ROUTES_DISABLE", None)
    else:
        os.environ["ROUTES_DISABLE"] = orig_disable

    if orig_enable is None:
        os.environ.pop("ROUTES_ENABLE", None)
    else:
        os.environ["ROUTES_ENABLE"] = orig_enable

    restore_app_main(previous_main)
    importlib.invalidate_caches()

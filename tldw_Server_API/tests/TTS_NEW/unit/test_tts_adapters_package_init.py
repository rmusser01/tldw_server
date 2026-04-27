import json
import subprocess  # nosec B404
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit


def test_tts_adapters_package_import_is_lazy():
    repo_root = Path(__file__).resolve().parents[4]
    script = """
import importlib
import json
import sys

adapters_module_name = "tldw_Server_API.app.core.TTS.adapters"
base_module_name = "tldw_Server_API.app.core.TTS.adapters.base"
adapter_submodule_name = "tldw_Server_API.app.core.TTS.adapters.omnivoice_adapter"

sys.modules.pop(adapter_submodule_name, None)
sys.modules.pop(base_module_name, None)
sys.modules.pop(adapters_module_name, None)

adapters = importlib.import_module(adapters_module_name)
before = base_module_name in sys.modules
audio_format = adapters.AudioFormat
adapter_submodule = adapters.omnivoice_adapter

print(json.dumps({
    "base_loaded_before_attr_access": before,
    "audio_format_name": audio_format.__name__,
    "submodule_name": adapter_submodule.__name__,
    "base_loaded_after_attr_access": base_module_name in sys.modules,
}))
""".strip()

    result = subprocess.run(  # nosec B603
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(repo_root),
    )
    payload = json.loads(result.stdout.strip())

    assert payload["base_loaded_before_attr_access"] is False  # nosec B101
    assert payload["audio_format_name"] == "AudioFormat"  # nosec B101
    assert payload["submodule_name"] == "tldw_Server_API.app.core.TTS.adapters.omnivoice_adapter"  # nosec B101
    assert payload["base_loaded_after_attr_access"] is True  # nosec B101

from pathlib import Path

import pytest

from tldw_Server_API.app.core.TTS.adapter_registry import TTSAdapterRegistry
from tldw_Server_API.app.core.TTS.adapters.pocket_tts_adapter import PocketTTSOnnxAdapter
from tldw_Server_API.app.core.TTS.tts_config import TTSConfigManager


@pytest.mark.unit
def test_yaml_module_path_reaches_pocket_tts_adapter(tmp_path):
    module_path = tmp_path / "pocket-runtime"
    yaml_path = tmp_path / "tts.yaml"
    yaml_path.write_text(
        "providers:\n  pocket_tts:\n    enabled: true\n"
        f"    module_path: {module_path}\n"
        f"    model_path: {tmp_path / 'models' / 'english'}\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.txt"
    config_path.write_text("", encoding="utf-8")
    manager = TTSConfigManager(yaml_path=yaml_path, config_txt_path=config_path)
    registry = TTSAdapterRegistry({"providers": {}}, include_defaults=False)
    registry.config_manager = manager

    adapter = PocketTTSOnnxAdapter(config=registry._get_provider_config("pocket_tts"))

    assert adapter.module_path == Path(module_path)  # nosec B101 - Verify documented runtime path survives actual configuration parsing.

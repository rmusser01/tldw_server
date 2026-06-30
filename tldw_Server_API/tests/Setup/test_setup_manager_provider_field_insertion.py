from configparser import ConfigParser
from pathlib import Path

from tldw_Server_API.app.core.Setup import setup_manager


def _write_config(tmp_path: Path, content: str) -> Path:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "config.txt"
    config_path.write_text(content, encoding="utf-8")
    return config_path


def _read_config(config_path: Path) -> ConfigParser:
    parser = ConfigParser()
    parser.optionxform = str
    parser.read(config_path, encoding="utf-8")
    return parser


def test_provider_api_key_field_is_inserted_inside_existing_api_section(monkeypatch, tmp_path):
    config_path = _write_config(
        tmp_path,
        "[API]\n"
        "default_api = openai\n"
        "\n"
        "[Local-API]\n"
        "ollama_api_IP = http://127.0.0.1:11434/v1\n",
    )
    monkeypatch.setenv("TLDW_CONFIG_DIR", str(config_path.parent))

    setup_manager.update_config(
        {"API": {"openai_api_key": "sk-abcdefghijklmnopqrstuvwxyz"}},
        create_backup=False,
    )

    parser = _read_config(config_path)
    assert parser.get("API", "openai_api_key") == "sk-abcdefghijklmnopqrstuvwxyz"
    assert not parser.has_option("Local-API", "openai_api_key")


def test_provider_local_api_field_is_inserted_inside_existing_local_api_section(
    monkeypatch,
    tmp_path,
):
    config_path = _write_config(
        tmp_path,
        "[API]\n"
        "default_api = openai\n"
        "\n"
        "[Local-API]\n"
        "ollama_api_IP = http://127.0.0.1:11434/v1\n"
        "\n"
        "[Other]\n"
        "value = keep\n",
    )
    monkeypatch.setenv("TLDW_CONFIG_DIR", str(config_path.parent))

    setup_manager.update_config(
        {"Local-API": {"kobold_api_IP": "http://127.0.0.1:5001/api/v1/generate"}},
        create_backup=False,
    )

    parser = _read_config(config_path)
    assert parser.get("Local-API", "kobold_api_IP") == "http://127.0.0.1:5001/api/v1/generate"
    assert not parser.has_option("Other", "kobold_api_IP")


def test_provider_field_update_creates_missing_section_at_eof(monkeypatch, tmp_path):
    config_path = _write_config(
        tmp_path,
        "[API]\n"
        "default_api = openai\n",
    )
    monkeypatch.setenv("TLDW_CONFIG_DIR", str(config_path.parent))

    setup_manager.update_config(
        {"Local-API": {"kobold_api_IP": "http://127.0.0.1:5001/api/v1/generate"}},
        create_backup=False,
    )

    parser = _read_config(config_path)
    assert parser.get("Local-API", "kobold_api_IP") == "http://127.0.0.1:5001/api/v1/generate"

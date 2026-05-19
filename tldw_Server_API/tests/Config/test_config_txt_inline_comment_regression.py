from __future__ import annotations

import configparser
from pathlib import Path


def test_config_txt_avoids_inline_value_comments() -> None:
    config_path = Path(__file__).resolve().parents[2] / "Config_Files" / "config.txt"
    offending_lines = []

    for line_number, line in enumerate(config_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.lstrip()
        if not stripped or stripped.startswith(("#", ";", "[")):
            continue
        if "  #" in line:
            offending_lines.append((line_number, line))

    assert offending_lines == []


def test_config_txt_enables_ingestion_sources_route_for_sources_ui() -> None:
    config_path = Path(__file__).resolve().parents[2] / "Config_Files" / "config.txt"
    parser = configparser.ConfigParser()
    parser.read(config_path, encoding="utf-8")

    enabled_routes = {
        item.strip().lower()
        for item in parser.get("API-Routes", "enable", fallback="").split(",")
        if item.strip()
    }

    assert "ingestion-sources" in enabled_routes

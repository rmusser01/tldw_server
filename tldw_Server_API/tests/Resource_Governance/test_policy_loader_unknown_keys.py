"""The file-backed policy loader must warn about top-level/route_map keys it silently ignores."""

from pathlib import Path

import pytest
import yaml
from loguru import logger

from tldw_Server_API.app.core.Resource_Governance.policy_loader import PolicyLoader, PolicyReloadConfig

pytestmark = [pytest.mark.unit, pytest.mark.rate_limit, pytest.mark.asyncio]


async def test_loader_warns_on_keys_it_ignores(tmp_path: Path) -> None:
    """Unknown top-level and route_map keys warn; consumed keys (templates, schema_version) don't."""
    path = tmp_path / "rg.yaml"
    path.write_text(yaml.safe_dump({"version": 1, "policies": {}, "templates": {}, "schema_version": 1, "bogus": 1, "route_map": {"by_path": {}, "by_route": {}}}), encoding="utf-8")
    messages = []
    sink = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        await PolicyLoader(path, PolicyReloadConfig(enabled=False)).load_once()
    finally:
        logger.remove(sink)
    text = "\n".join(messages)
    assert "bogus" in text and "by_route" in text
    assert "templates" not in text and "schema_version" not in text

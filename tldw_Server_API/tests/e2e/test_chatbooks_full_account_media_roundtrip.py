from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = [pytest.mark.critical, pytest.mark.multi_user]


SCRIPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "Helper_Scripts"
    / "Testing-related"
    / "chatbooks_full_account_uat_fixture.py"
)


def _load_fixture_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("chatbooks_full_account_uat_fixture", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load fixture helper: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_full_account_archive_restores_media_bytes_vectors_and_profile_to_clean_user(
    tmp_path: Path,
) -> None:
    original_env = os.environ.copy()
    fixture = _load_fixture_module()
    try:
        prepared = await fixture.prepare(tmp_path)
        reset = await fixture.reset_destination(tmp_path)

        assert prepared["source_user_id"] != reset["destination_user_id"]

        import_result = await fixture.import_archive(tmp_path, Path(prepared["archive_path"]))
        assert import_result["success"] is True, import_result
        assert import_result["imported_items"]["account_profile"] == 1
        assert import_result["imported_items"]["account_settings"] == 1
        assert import_result["imported_items"]["character"] == 1
        assert import_result["imported_items"]["media"] == 1
        assert import_result["imported_items"]["embedding"] >= 2, import_result

        report = await fixture.verify(tmp_path)

        assert report["source_user_id"] != report["destination_user_id"]
        assert report["profile"] == report["expected"]["profile"]
        assert report["settings"] == report["expected"]["settings"]
        assert [item["name"] for item in report["characters"]] == [
            item["name"] for item in report["expected"]["characters"]
        ]
        assert report["media"]["transcript_count"] == 1
        assert report["media"]["chunk_count"] == 2
        assert report["media"]["artifact_sha256"] == report["expected"]["media"]["artifact_sha256"]
        assert report["media"]["vector_sha256"] == report["expected"]["media"]["vector_sha256"]
        assert report["embeddings"]["collection_ids"] == report["expected"]["embeddings"]["collection_ids"]

        restored_artifact = Path(report["media"]["artifact_path"])
        restored_artifact.write_bytes(b"corrupted after import")
        with pytest.raises(fixture.FixtureVerificationError, match="SHA-256"):
            await fixture.verify(tmp_path)
    finally:
        await fixture.reset_runtime_state()
        os.environ.clear()
        os.environ.update(original_env)
        fixture.reset_settings()
        fixture.clear_config_cache()

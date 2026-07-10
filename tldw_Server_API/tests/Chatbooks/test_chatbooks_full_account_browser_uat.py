from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[3] / "Helper_Scripts" / "Testing-related" / "chatbooks_full_account_browser_uat.py"
)
SPEC = importlib.util.spec_from_file_location("chatbooks_full_account_browser_uat", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load browser UAT runner from {SCRIPT_PATH}")
browser_uat = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = browser_uat
SPEC.loader.exec_module(browser_uat)


REQUIRED_PAYLOADS = {
    "json/account_profile.json": b'{"schema_version":"1.0","values":{}}',
    "json/account_settings.json": b'{"schema_version":"1.0","values":{}}',
    "media/full-account-uat.bin": b"browser-downloaded-media-bytes",
}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_archive(path: Path, *, name: str, payloads: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    inventory = [
        {
            "path": payload_path,
            "media_type": "application/octet-stream",
            "size_bytes": len(payload),
            "integrity": {
                "status": "verified",
                "algorithm": "sha256",
                "value": f"sha256:{_sha256(payload)}",
            },
            "role": "payload",
            "content_item_ids": [],
        }
        for payload_path, payload in sorted(payloads.items())
    ]
    manifest = {
        "version": "1.1.0",
        "name": name,
        "description": "browser UAT archive",
        "file_inventory": inventory,
        "account_inventory_summary": {"post_write_verification": True},
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for payload_path, payload in payloads.items():
            archive.writestr(payload_path, payload)
        archive.writestr("manifest.json", json.dumps(manifest, sort_keys=True))


def _write_expected(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "expected.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "source_user_id": 1,
                "destination_user_id": 2,
                "media": {
                    "archive_path": "media/full-account-uat.bin",
                    "artifact_sha256": _sha256(REQUIRED_PAYLOADS["media/full-account-uat.bin"]),
                    "vector_sha256": _sha256(b"destination-vector"),
                },
                "embeddings": {
                    "collection_name": "chatbooks_full_account_uat",
                    "collection_ids": ["uat-chunk-001", "uat-chunk-002"],
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def test_decode_json_output_returns_complete_final_object() -> None:
    output = 'fixture log\n{\n  "source_user_id": 1,\n  "media": {"count": 2}\n}\n'

    assert browser_uat._decode_json_output(output, "fixture") == {
        "source_user_id": 1,
        "media": {"count": 2},
    }


@pytest.mark.parametrize("surface", ["webui", "extension"])
def test_plan_enforces_distinct_two_phase_browser_round_trip(
    tmp_path: Path,
    surface: str,
) -> None:
    config = browser_uat.BrowserUatConfig(
        surface=surface,
        root=tmp_path,
        api_port=18001,
        web_port=18269,
    )

    plan = browser_uat.build_uat_plan(config)

    assert [step.step_id for step in plan] == [
        "prepare-source",
        "start-source-api",
        "browser-export",
        "inspect-browser-archive",
        "stop-source-api",
        "reset-destination",
        "start-destination-api",
        "browser-import",
        "stop-destination-api",
        "verify-destination",
    ]
    assert config.source_root != config.destination_root
    assert config.downloaded_archive != config.fixture_archive
    assert plan[2].archive_path == config.downloaded_archive
    assert plan[7].archive_path == config.downloaded_archive
    assert plan[1].phase == "source"
    assert plan[6].phase == "destination"


def test_archive_inspection_rejects_fixture_substitution_even_after_copy(
    tmp_path: Path,
) -> None:
    config = browser_uat.BrowserUatConfig(
        surface="webui",
        root=tmp_path,
        api_port=18001,
        web_port=18269,
    )
    _write_expected(tmp_path)
    _write_archive(
        config.fixture_archive,
        name="fixture archive",
        payloads=REQUIRED_PAYLOADS,
    )
    config.downloaded_archive.parent.mkdir(parents=True, exist_ok=True)
    config.downloaded_archive.write_bytes(config.fixture_archive.read_bytes())

    with pytest.raises(browser_uat.BrowserUatError, match="fixture archive"):
        browser_uat.inspect_browser_archive(config)


def test_api_scope_preflight_rejects_wrong_source_or_dirty_destination() -> None:
    source_scope = {
        "categories": [
            {"category": "account_settings", "count": 2},
            {"category": "characters", "count": 2},
            {"category": "media_records", "count": 1},
            {"category": "media_stored_artifacts", "count": 1},
            {"category": "embeddings", "count": 2},
        ]
    }
    destination_scope = {
        "categories": [
            {"category": "media_records", "count": 0},
            {"category": "media_stored_artifacts", "count": 0},
            {"category": "embeddings", "count": 0},
        ]
    }

    browser_uat.validate_phase_scope("source", source_scope)
    browser_uat.validate_phase_scope("destination", destination_scope)

    source_scope["categories"][2]["count"] = 0
    with pytest.raises(browser_uat.BrowserUatError, match="source API scope"):
        browser_uat.validate_phase_scope("source", source_scope)

    destination_scope["categories"][0]["count"] = 1
    with pytest.raises(browser_uat.BrowserUatError, match="destination API scope"):
        browser_uat.validate_phase_scope("destination", destination_scope)


@pytest.mark.parametrize(
    ("missing_path", "message"),
    [
        ("json/account_profile.json", "account profile"),
        ("json/account_settings.json", "account settings"),
        ("media/full-account-uat.bin", "bundled media"),
    ],
)
def test_archive_inspection_requires_verified_full_account_inventory(
    tmp_path: Path,
    missing_path: str,
    message: str,
) -> None:
    config = browser_uat.BrowserUatConfig(
        surface="extension",
        root=tmp_path,
        api_port=18011,
    )
    _write_expected(tmp_path)
    _write_archive(
        config.fixture_archive,
        name="fixture archive",
        payloads={"fixture.txt": b"fixture"},
    )
    _write_archive(
        config.downloaded_archive,
        name="browser archive",
        payloads={path: payload for path, payload in REQUIRED_PAYLOADS.items() if path != missing_path},
    )

    with pytest.raises(browser_uat.BrowserUatError, match=message):
        browser_uat.inspect_browser_archive(config)


class _FakeRuntime:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.import_archive: Path | None = None

    def prepare_source(self, config: browser_uat.BrowserUatConfig) -> dict[str, int]:
        self.events.append("prepare-source")
        _write_expected(config.root)
        _write_archive(
            config.fixture_archive,
            name="fixture archive",
            payloads={"fixture.txt": b"fixture"},
        )
        return {"source_user_id": 1}

    def start_api(self, config: browser_uat.BrowserUatConfig, phase: str) -> None:
        self.events.append(f"start-{phase}-api")

    def run_browser(
        self,
        config: browser_uat.BrowserUatConfig,
        phase: str,
        archive_path: Path,
    ) -> None:
        self.events.append(f"browser-{phase}")
        if phase == "export":
            _write_archive(
                archive_path,
                name="browser archive",
                payloads=REQUIRED_PAYLOADS,
            )
        else:
            self.import_archive = archive_path

    def inspect_archive(
        self,
        config: browser_uat.BrowserUatConfig,
    ) -> dict[str, object]:
        self.events.append("inspect-browser-archive")
        return browser_uat.inspect_browser_archive(config)

    def stop_api(self, phase: str) -> None:
        self.events.append(f"stop-{phase}-api")

    def reset_destination(self, config: browser_uat.BrowserUatConfig) -> dict[str, int]:
        self.events.append("reset-destination")
        return {"destination_user_id": 2}

    def verify_destination(
        self,
        config: browser_uat.BrowserUatConfig,
    ) -> dict[str, object]:
        self.events.append("verify-destination")
        return {
            "media": {
                "artifact_sha256": _sha256(REQUIRED_PAYLOADS["media/full-account-uat.bin"]),
                "vector_sha256": _sha256(b"destination-vector"),
            },
            "embeddings": {
                "collection_name": "chatbooks_full_account_uat",
                "collection_ids": ["uat-chunk-001", "uat-chunk-002"],
            },
        }

    def close(self) -> None:
        self.events.append("close")


def test_fake_runtime_imports_exact_browser_download_and_verifies_destination(
    tmp_path: Path,
) -> None:
    config = browser_uat.BrowserUatConfig(
        surface="webui",
        root=tmp_path,
        api_port=18001,
        web_port=18269,
    )
    runtime = _FakeRuntime()

    result = browser_uat.run_browser_uat(config, runtime=runtime)

    assert runtime.events == [
        "prepare-source",
        "start-source-api",
        "browser-export",
        "inspect-browser-archive",
        "stop-source-api",
        "reset-destination",
        "start-destination-api",
        "browser-import",
        "stop-destination-api",
        "verify-destination",
        "close",
    ]
    assert runtime.import_archive == config.downloaded_archive
    assert result["downloaded_archive_path"] == str(config.downloaded_archive)
    assert result["source_root"] == str(config.source_root)
    assert result["destination_root"] == str(config.destination_root)
    assert result["archive"]["format_version"] == "1.1.0"

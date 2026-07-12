from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Chatbooks.chatbook_account_inventory import (
    ACCOUNT_DATA_INVENTORY,
)

SCRIPT_PATH = (
    Path(__file__).resolve().parents[3] / "Helper_Scripts" / "Testing-related" / "chatbooks_full_account_browser_uat.py"
)
SPEC = importlib.util.spec_from_file_location("chatbooks_full_account_browser_uat", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load browser UAT runner from {SCRIPT_PATH}")
browser_uat = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = browser_uat
SPEC.loader.exec_module(browser_uat)


CHARACTER_PATH = "content/characters/character_1.json"
EMBEDDING_PATH = "content/embeddings/collection_chatbooks_full_account_uat.json"
EMBEDDING_ROWS = [
    {
        "id": "uat-chunk-001",
        "embedding": [0.1, 0.2, 0.3],
        "metadata": {"media_id": "1", "media_uuid": "media-uuid", "chunk_index": 0},
    },
    {
        "id": "uat-chunk-002",
        "embedding": [0.4, 0.5, 0.6],
        "metadata": {"media_id": "1", "media_uuid": "media-uuid", "chunk_index": 1},
    },
]
REQUIRED_PAYLOADS = {
    "json/account_profile.json": json.dumps(
        {
            "schema_version": "1.0",
            "profile": {"identity.email": "chatbooks-backup-source@example.com"},
        },
        sort_keys=True,
    ).encode("utf-8"),
    "json/account_settings.json": json.dumps(
        {
            "schema_version": "1.0",
            "overrides": {"preferences.ui.theme": "paper"},
        },
        sort_keys=True,
    ).encode("utf-8"),
    CHARACTER_PATH: b'{"name":"Chatbooks UAT Archivist"}',
    "media/full-account-uat.bin": b"browser-downloaded-media-bytes",
    EMBEDDING_PATH: json.dumps(
        {
            "embedding_set_id": "chatbooks_full_account_uat",
            "item_count": 2,
            "chunks": EMBEDDING_ROWS,
        },
        sort_keys=True,
    ).encode("utf-8"),
}
ACCOUNT_COUNTS = {
    row.category: (
        2
        if row.category in {"media_chunks", "embeddings"}
        else 1
        if row.category
        in {
            "account_profile",
            "account_settings",
            "characters",
            "media_records",
            "media_transcripts",
            "media_stored_artifacts",
            "media_pointers",
        }
        else 0
    )
    for row in ACCOUNT_DATA_INVENTORY
}
CLEAN_DESTINATION_COUNTS = {
    row.category: 1
    if row.category in {"account_profile", "account_settings", "characters"}
    else 0
    for row in ACCOUNT_DATA_INVENTORY
}


def test_runner_prioritizes_current_worktree_imports() -> None:
    assert Path(browser_uat.sys.path[0]).resolve() == browser_uat.PROJECT_ROOT


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_archive(
    path: Path,
    *,
    name: str,
    payloads: dict[str, bytes],
    account_categories: set[str] | None = None,
) -> None:
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
    account_inventory = [
        row.to_summary()
        for row in ACCOUNT_DATA_INVENTORY
        if account_categories is None or row.category in account_categories
    ]
    manifest = {
        "version": "1.1.0",
        "name": name,
        "description": "browser UAT archive",
        "file_inventory": inventory,
        "account_inventory": account_inventory,
        "account_inventory_summary": {
            "post_write_verification": True,
            "counts": {
                row.manifest_count_key: ACCOUNT_COUNTS[row.category]
                for row in ACCOUNT_DATA_INVENTORY
            },
        },
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
                "profile": {"identity.email": "chatbooks-backup-source@example.com"},
                "settings": {"preferences.ui.theme": "paper"},
                "characters": [
                    {"name": "Chatbooks UAT Archivist", "archive_path": CHARACTER_PATH}
                ],
                "media": {
                    "title": "Chatbooks full-account stored media",
                    "archive_path": "media/full-account-uat.bin",
                    "artifact_sha256": _sha256(REQUIRED_PAYLOADS["media/full-account-uat.bin"]),
                    "vector_sha256": _sha256(b"destination-vector"),
                    "transcript_count": 1,
                    "chunk_count": 2,
                },
                "embeddings": {
                    "collection_name": "chatbooks_full_account_uat",
                    "collection_ids": ["uat-chunk-001", "uat-chunk-002"],
                    "archive_path": EMBEDDING_PATH,
                    "row_count": 2,
                    "rows": EMBEDDING_ROWS,
                },
                "account_inventory": {
                    "source_counts": ACCOUNT_COUNTS,
                    "clean_destination_counts": CLEAN_DESTINATION_COUNTS,
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


def test_archive_inspection_rejects_sensitive_or_source_path_leaks(
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
        payloads={"fixture.txt": b"fixture"},
    )
    password_hash = hashlib.sha256(b"chatbooks-uat-disabled-login:chatbooks-backup-source").hexdigest()
    _write_archive(
        config.downloaded_archive,
        name="browser archive",
        payloads={
            **REQUIRED_PAYLOADS,
            "json/leak.json": json.dumps(
                {
                    "password_hash": password_hash,
                    "source_root": str(config.source_root),
                }
            ).encode("utf-8"),
        },
    )

    with pytest.raises(browser_uat.BrowserUatError, match="sensitive data"):
        browser_uat.inspect_browser_archive(config)


def test_sensitive_scan_streams_binary_members_and_detects_chunk_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = browser_uat.BrowserUatConfig(
        surface="webui",
        root=tmp_path,
        api_port=18001,
        web_port=18269,
    )
    forbidden = hashlib.sha256(
        b"chatbooks-uat-disabled-login:chatbooks-backup-source"
    ).hexdigest().encode("ascii")
    monkeypatch.setattr(browser_uat, "ARCHIVE_SCAN_CHUNK_SIZE", 32, raising=False)
    archive_path = tmp_path / "boundary.chatbook"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("binary/payload.bin", b"\x00\xff" + b"x" * 27 + forbidden + b"\x00")

    with zipfile.ZipFile(archive_path) as archive:
        monkeypatch.setattr(
            archive,
            "read",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("privacy scan must not materialize archive members")
            ),
        )
        with pytest.raises(browser_uat.BrowserUatError, match="sensitive data"):
            browser_uat._verify_no_sensitive_archive_leaks(archive, config)


def test_api_scope_preflight_requires_every_category_and_clean_destination() -> None:
    source_scope = {
        "categories": [
            {"category": category, "count": count}
            for category, count in ACCOUNT_COUNTS.items()
        ]
    }
    destination_scope = {
        "categories": [
            {"category": category, "count": count}
            for category, count in CLEAN_DESTINATION_COUNTS.items()
        ]
    }

    browser_uat.validate_phase_scope("source", source_scope, ACCOUNT_COUNTS)
    browser_uat.validate_phase_scope(
        "destination",
        destination_scope,
        CLEAN_DESTINATION_COUNTS,
    )

    source_scope["categories"] = source_scope["categories"][:-1]
    with pytest.raises(browser_uat.BrowserUatError, match="source API scope"):
        browser_uat.validate_phase_scope("source", source_scope, ACCOUNT_COUNTS)

    dirty_category = next(
        row for row in destination_scope["categories"] if row["category"] == "characters"
    )
    dirty_category["count"] = 2
    with pytest.raises(browser_uat.BrowserUatError, match="destination API scope"):
        browser_uat.validate_phase_scope(
            "destination",
            destination_scope,
            CLEAN_DESTINATION_COUNTS,
        )


def test_phase_environment_enables_multi_user_app_mode(tmp_path: Path) -> None:
    config = browser_uat.BrowserUatConfig(
        surface="webui",
        root=tmp_path,
        api_port=18001,
        web_port=18269,
    )

    environment = browser_uat._phase_environment(config, "source")

    assert environment["AUTH_MODE"] == "multi_user"
    assert environment["APP_MODE"] == "multi"


def test_browser_timeout_preserves_partial_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = browser_uat.BrowserUatConfig(
        surface="webui",
        root=tmp_path,
        api_port=18001,
        web_port=18269,
        timeout_seconds=1,
    )
    runtime = browser_uat.RealBrowserUatRuntime()
    runtime._tokens["source"] = "synthetic-token"
    monkeypatch.setattr(runtime, "_browser_command", lambda _config: (tmp_path, ["playwright"]))

    def raise_timeout(*_args: object, **_kwargs: object) -> None:
        raise subprocess.TimeoutExpired(
            cmd=["playwright"],
            timeout=1,
            output="partial stdout",
            stderr="partial stderr",
        )

    monkeypatch.setattr(browser_uat.subprocess, "run", raise_timeout)

    with pytest.raises(browser_uat.BrowserUatError, match="timed out after 1 seconds"):
        runtime.run_browser(config, "export", config.downloaded_archive)

    browser_log = tmp_path / "logs" / "webui-export.log"
    assert browser_log.read_text(encoding="utf-8") == "partial stdout\npartial stderr"


def test_extension_browser_respects_explicit_headless_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = browser_uat.BrowserUatConfig(
        surface="extension",
        root=tmp_path,
        api_port=18011,
    )
    runtime = browser_uat.RealBrowserUatRuntime()
    runtime._tokens["source"] = "synthetic-token"
    monkeypatch.setenv("TLDW_E2E_EXTENSION_HEADLESS", "0")
    monkeypatch.setattr(runtime, "_browser_command", lambda _config: (tmp_path, ["playwright"]))
    captured: dict[str, object] = {}

    def complete(*_args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["env"] = kwargs["env"]
        return subprocess.CompletedProcess(["playwright"], 0, stdout="", stderr="")

    monkeypatch.setattr(browser_uat.subprocess, "run", complete)

    runtime.run_browser(config, "export", config.downloaded_archive)

    assert isinstance(captured["env"], dict)
    assert captured["env"]["TLDW_E2E_EXTENSION_HEADLESS"] == "0"


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


def test_archive_inspection_rejects_incomplete_account_category_inventory(
    tmp_path: Path,
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
    incomplete_counts = dict(ACCOUNT_COUNTS)
    incomplete_counts.pop("characters")
    _write_archive(
        config.downloaded_archive,
        name="browser archive",
        payloads=REQUIRED_PAYLOADS,
        account_categories=set(incomplete_counts),
    )

    with pytest.raises(browser_uat.BrowserUatError, match="account inventory"):
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
        return {
            "destination_user_id": 2,
            "counts": CLEAN_DESTINATION_COUNTS,
        }

    def verify_destination(
        self,
        config: browser_uat.BrowserUatConfig,
    ) -> dict[str, object]:
        self.events.append("verify-destination")
        return {
            "profile": {"identity.email": "chatbooks-backup-source@example.com"},
            "settings": {"preferences.ui.theme": "paper"},
            "characters": [{"name": "Chatbooks UAT Archivist"}],
            "media": {
                "title": "Chatbooks full-account stored media",
                "transcript_count": 1,
                "chunk_count": 2,
                "artifact_sha256": _sha256(REQUIRED_PAYLOADS["media/full-account-uat.bin"]),
                "vector_sha256": _sha256(b"destination-vector"),
            },
            "embeddings": {
                "collection_name": "chatbooks_full_account_uat",
                "collection_ids": ["uat-chunk-001", "uat-chunk-002"],
                "row_count": 2,
                "rows": EMBEDDING_ROWS,
            },
            "account_inventory_counts": ACCOUNT_COUNTS,
        }

    def close(self) -> None:
        self.events.append("close")


@pytest.mark.parametrize(
    ("section", "key", "bad_value"),
    [
        ("profile", "identity.email", "preexisting@example.com"),
        ("settings", "preferences.ui.theme", "preexisting"),
        ("characters", 0, {"name": "Pre-existing character"}),
        ("embeddings", "row_count", 3),
    ],
)
def test_destination_comparison_rejects_any_restored_account_mismatch(
    tmp_path: Path,
    section: str,
    key: object,
    bad_value: object,
) -> None:
    config = browser_uat.BrowserUatConfig(
        surface="webui",
        root=tmp_path,
        api_port=18001,
        web_port=18269,
    )
    _write_expected(tmp_path)
    destination = _FakeRuntime().verify_destination(config)
    destination[section][key] = bad_value

    with pytest.raises(browser_uat.BrowserUatError, match="Destination"):
        browser_uat._assert_destination_matches_expected(config, destination)


def test_destination_comparison_rejects_unexpected_inventory_count(tmp_path: Path) -> None:
    config = browser_uat.BrowserUatConfig(
        surface="webui",
        root=tmp_path,
        api_port=18001,
        web_port=18269,
    )
    _write_expected(tmp_path)
    destination = _FakeRuntime().verify_destination(config)
    destination["account_inventory_counts"]["notes"] = 1

    with pytest.raises(browser_uat.BrowserUatError, match="Destination account inventory"):
        browser_uat._assert_destination_matches_expected(config, destination)


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

"""Opt-in candidate-build evidence, never a production allowlist update.

No model or executable is downloaded. Only newly created temporary profiles run.
"""

import asyncio
import json
import os
import socket
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from tldw_Server_API.app.core.Local_LLM.llamacpp_process_runner import LlamaCppProcessRunner
from tldw_Server_API.app.core.Local_LLM.llamacpp_runtime_models import LlamaCppProfile
from tldw_Server_API.app.core.Local_LLM.llamacpp_snapshot_compatibility import hash_file_stable
from tldw_Server_API.app.core.Local_LLM.llamacpp_snapshot_models import SnapshotRequest
from tldw_Server_API.app.core.Local_LLM.llamacpp_snapshot_operations import SnapshotOperations
from tldw_Server_API.app.core.Local_LLM.llamacpp_snapshot_store import SnapshotStore
from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Schemas import LlamaCppConfig

pytestmark = pytest.mark.local_llm_service


def completion_metrics(response: dict) -> dict[str, int]:
    """Require native runtime counters; missing counters cannot prove reuse."""
    # Source-derived (not live evidence): llama.cpp 4d9176092d00586775af140581bb0b558ddc4389,
    # tools/server/server-common.cpp:67-71. Top-level tokens_cached is the final
    # slot cache size, not reused tokens (server-context.cpp:2118).
    cached = response.get("timings", {}).get("cache_n")
    processed = response.get("timings", {}).get("prompt_n")
    if any(type(value) is not int or value < 0 for value in (cached, processed)):
        raise ValueError("Runtime must report integer timings.cache_n and timings.prompt_n counters")
    return {"cached_tokens": cached, "processed_tokens": processed}


def free_local_port() -> int:
    """Choose a loopback port; the runner rechecks availability before launch."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def live_server_args() -> dict[str, object]:
    """Explicit operator choices, independent of model names or architecture guesses."""
    full = os.environ.get("TLDW_SNAPSHOT_SWA_FULL", "0")
    if full not in {"0", "1"}:
        raise ValueError("TLDW_SNAPSHOT_SWA_FULL must be 0 or 1")
    context = int(os.environ.get("TLDW_SNAPSHOT_CTX_SIZE", "16384"))
    if not 1 <= context <= 1048576:
        raise ValueError("TLDW_SNAPSHOT_CTX_SIZE must be between 1 and 1048576; check memory before running")
    return {"ctx_size": context, "parallel": 1, "n_gpu_layers": 0, "swa_full": full == "1"}


@pytest.mark.skipif(
    os.environ.get("TLDW_SNAPSHOT_LIVE") != "1",
    reason="Set TLDW_SNAPSHOT_LIVE=1 with operator-supplied assets and disposable-profile consent",
)
@pytest.mark.asyncio
async def test_real_snapshot_reuse_after_restart_against_cold_process(tmp_path: Path, record_property):
    """Exercise real runner/store/operations with a candidate build, text only.

    This is an isolated characterization harness, not HTTP Admin/Chatbook proof.
    Candidate admission is injected only into this private test service.
    """
    if os.environ.get("TLDW_SNAPSHOT_DISPOSABLE") != "YES":
        pytest.fail("TLDW_SNAPSHOT_DISPOSABLE=YES is required; only fresh temporary profiles are supported")
    executable = Path(os.environ.get("TLDW_SNAPSHOT_EXECUTABLE", ""))
    model = Path(os.environ.get("TLDW_SNAPSHOT_MODEL", ""))
    for label, path in (("executable", executable), ("model", model)):
        if not path.is_absolute() or not path.is_file() or path.is_symlink():
            pytest.fail(f"Supply an absolute regular non-symlink {label} path")
    if not os.access(executable, os.X_OK):
        pytest.fail("Supplied runtime is not executable")
    server_args = live_server_args()
    executable_hash = hash_file_stable(executable)
    model_hash = hash_file_stable(model)
    config = LlamaCppConfig(
        executable_path=executable,
        models_dir=model.parent,
        allowed_paths=[model.parent],
        default_host="127.0.0.1",
        default_ctx_size=server_args["ctx_size"],
        default_n_gpu_layers=0,
        readiness_timeout=180,
        http_timeout=600,
        log_output_file=None,
    )
    profile = LlamaCppProfile(
        profile_id=f"disposable-snapshot-{uuid4().hex}",
        name="Disposable snapshot evidence",
        model_path=str(model),
        host="127.0.0.1",
        port=free_local_port(),
        snapshots_enabled=True,
        snapshot_retention=1,
        autostart=False,
        server_args=server_args,
    )
    store = SnapshotStore(tmp_path / "private-snapshots")
    service = SnapshotOperations(store, supported_builds={executable_hash})
    runner = LlamaCppProcessRunner(config, profile.profile_id)
    runner.snapshot_store = store
    cold = LlamaCppProcessRunner(config, f"disposable-cold-{uuid4().hex}")
    cold.snapshot_store = store
    # Public synthetic text only. Never include a private prompt in this report.
    prefix = "\n".join(
        f"Record {index}: The observatory measures clear skies and checks each instrument daily."
        for index in range(160)
    )
    prompt = prefix + "\nSummarize the observatory routine in one sentence."

    async def completion(target, text):
        endpoint = target.status().endpoint
        assert endpoint and endpoint.startswith("http://127.0.0.1:")
        async with httpx.AsyncClient(timeout=600, trust_env=False, follow_redirects=False) as client:
            response = await client.post(
                endpoint + "/completion",
                json={
                    "prompt": text,
                    "n_predict": 1,
                    "cache_prompt": True,
                    "id_slot": 0,
                    "temperature": 0,
                    "seed": 42,
                    "stream": False,
                },
            )
            response.raise_for_status()
            return completion_metrics(response.json())

    async def operation(kind, snapshot_id=None):
        observed = await service.slots(profile, runner)
        assert observed["capability"] == "ready", observed["reason"]
        request = SnapshotRequest(
            slot_id=0,
            expected_launch_generation=runner.snapshot_generation,
            request_id=observed["request_id"],
            replace_confirmed=kind == "restore",
        )
        receipt = await service.admit(profile, runner, request, "disposable-live-harness", kind, snapshot_id)
        await asyncio.gather(*list(service.tasks.values()))
        result = await service.operation(profile.profile_id, receipt.operation_id)
        assert result.state == "complete", result.error_code
        return result

    try:
        await runner.start(model, profile)
        effective_options = list(runner.snapshot_options)
        seed_metrics = await completion(runner, prefix)
        saved = await operation("save")
        assert saved.token_count and saved.token_count >= 1024
        original_generation = runner.snapshot_generation
        await runner.stop()
        assert runner.snapshot_process.returncode is not None
        store.cleanup_launch(profile.profile_id, original_generation)
        await runner.start(model, profile)
        assert runner.snapshot_generation != original_generation
        assert runner.snapshot_options == effective_options
        restored = await operation("restore", saved.snapshot_id)
        assert restored.token_count == saved.token_count
        warm_metrics = await completion(runner, prompt)
        await runner.stop()
        cold_profile = profile.model_copy(update={"profile_id": cold.profile_id, "port": free_local_port()})
        await cold.start(model, cold_profile)
        assert cold.snapshot_options == effective_options
        cold_metrics = await completion(cold, prompt)
        report = {
            "coverage": "candidate build, text, native /completion with id_slot=0; no Chatbook/Admin browser evidence",
            "executable_sha256": executable_hash,
            "model_sha256": model_hash,
            "effective_options": effective_options,
            "saved_tokens": saved.token_count,
            "seed": seed_metrics,
            "restored_request": warm_metrics,
            "cold_control": cold_metrics,
        }
        record_property("snapshot_live_evidence", json.dumps(report, sort_keys=True))
        print(json.dumps(report, sort_keys=True))
        # Publish measured evidence before assertions, including negative results.
        # Similar output, HTTP 200 or an artifact alone cannot satisfy these checks.
        assert warm_metrics["cached_tokens"] >= saved.token_count * 0.8
        assert warm_metrics["processed_tokens"] < cold_metrics["processed_tokens"] * 0.25
        assert cold_metrics["processed_tokens"] >= 1024
        assert warm_metrics["cached_tokens"] > cold_metrics["cached_tokens"]
        assert hash_file_stable(executable) == executable_hash
        assert hash_file_stable(model) == model_hash
    finally:
        await service.drain()
        await runner.stop()
        await cold.stop()
        for target in (runner, cold):
            if (
                target.snapshot_generation
                and target.snapshot_process
                and target.snapshot_process.returncode is not None
            ):
                store.cleanup_launch(target.profile_id, target.snapshot_generation)
        store.close()


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"tokens_cached": 20, "timings": {"prompt_n": 3}},
        {"timings": {"cache_n": True, "prompt_n": 3}},
        {"timings": {"cache_n": 2, "prompt_n": -1}},
    ],
)
def test_live_metrics_reject_missing_or_invalid_runtime_evidence(response):
    with pytest.raises(ValueError):
        completion_metrics(response)


def test_live_metrics_keep_measured_processed_and_cached_counts():
    assert completion_metrics({"tokens_cached": 9000, "timings": {"cache_n": 2048, "prompt_n": 7}}) == {
        "cached_tokens": 2048,
        "processed_tokens": 7,
    }


@pytest.mark.parametrize("value", ["true", "false", "", "2"])
def test_live_cache_mode_rejects_ambiguous_environment(monkeypatch, value):
    monkeypatch.setenv("TLDW_SNAPSHOT_SWA_FULL", value)
    with pytest.raises(ValueError):
        live_server_args()


def test_live_options_allow_explicit_cache_mode_and_context(monkeypatch):
    monkeypatch.setenv("TLDW_SNAPSHOT_SWA_FULL", "1")
    monkeypatch.setenv("TLDW_SNAPSHOT_CTX_SIZE", "8192")
    assert live_server_args() == {"ctx_size": 8192, "parallel": 1, "n_gpu_layers": 0, "swa_full": True}


@pytest.mark.parametrize("value", ["0", "-1", "unbounded", "1048577"])
def test_live_context_rejects_invalid_size(monkeypatch, value):
    monkeypatch.setenv("TLDW_SNAPSHOT_CTX_SIZE", value)
    with pytest.raises(ValueError):
        live_server_args()

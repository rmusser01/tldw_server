
"""Utilities to execute backend installation plans after the setup wizard."""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import inspect
import json
import os
import shutil
import subprocess  # nosec B404 - setup provisioning intentionally invokes vetted command lists without a shell
import sys
import tempfile
from collections.abc import Iterable
from configparser import ConfigParser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from tldw_Server_API.app.api.v1.endpoints.audio import audio_health
from tldw_Server_API.app.core.config import load_and_log_configs
from tldw_Server_API.app.core.Setup import audio_profile_service
from tldw_Server_API.app.core.Setup import audio_readiness_store
from tldw_Server_API.app.core.Setup import setup_manager
from tldw_Server_API.app.core.Setup.audio_bundle_catalog import (
    AUDIO_BUNDLE_CATALOG_VERSION,
    AudioBundle,
    AudioResourceProfile,
    AudioBundleStep,
    AutomationTier,
    DEFAULT_AUDIO_RESOURCE_PROFILE,
    build_audio_selection_key,
    get_audio_bundle_catalog,
)
from tldw_Server_API.app.core.Setup.install_schema import DEFAULT_WHISPER_MODELS, InstallPlan
from tldw_Server_API.app.core.Utils.pydantic_compat import model_dump_compat

CONFIG_ROOT = setup_manager.CONFIG_RELATIVE_PATH.parent
STATUS_FILENAME = 'setup_install_status.json'


_LATEST_STATUS_DATA: dict[str, Any] | None = None
_INSTALLED_DEPENDENCIES: set[str] = set()


def _candidate_status_dirs() -> list[Path]:
    candidates: list[Path] = []
    override = os.getenv('TLDW_INSTALL_STATE_DIR')
    if override:
        candidates.append(Path(override))
    candidates.append(CONFIG_ROOT)
    try:
        home = Path.home()
    except Exception:  # noqa: BLE001
        home = None
    if home:
        candidates.append(home / '.cache' / 'tldw_server')
    candidates.append(Path(tempfile.gettempdir()) / 'tldw_server')
    return candidates


def _resolve_status_file() -> Path | None:
    """Return a writable path for persisting install status, or ``None`` if unavailable."""

    for root in _candidate_status_dirs():
        if not root:
            continue
        try:
            root.mkdir(parents=True, exist_ok=True)
            probe = root / '.write_test'
            with probe.open('w', encoding='utf-8') as handle:
                handle.write('ok')
            with contextlib.suppress(FileNotFoundError):
                probe.unlink()
            return root / STATUS_FILENAME
        except Exception:  # noqa: BLE001
            logger.debug('Install status directory not writable')

    logger.warning('No writable location found for setup install status; running without persistence.')
    return None


def _install_dependencies(plan: InstallPlan, status: InstallationStatus, errors: list[str]) -> None:
    """Install required Python packages for selected backends."""

    processed_backends: set[str] = set()

    for entry in plan.stt:
        key = f"stt:{entry.engine}"
        if key not in processed_backends:
            with contextlib.suppress(PipInstallBlockedError):
                _install_backend_dependencies('stt', entry.engine, status, errors)
            processed_backends.add(key)

    if plan.stt:
        key = "stt:silero_vad"
        if key not in processed_backends:
            with contextlib.suppress(PipInstallBlockedError):
                _install_backend_dependencies('stt', 'silero_vad', status, errors)
            processed_backends.add(key)

    for entry in plan.tts:
        key = f"tts:{entry.engine}"
        if key not in processed_backends:
            with contextlib.suppress(PipInstallBlockedError):
                _install_backend_dependencies('tts', entry.engine, status, errors)
            processed_backends.add(key)

    if plan.embeddings.huggingface:
        with contextlib.suppress(PipInstallBlockedError):
            _install_embedding_dependencies('huggingface', status, errors)
    if plan.embeddings.custom:
        with contextlib.suppress(PipInstallBlockedError):
            _install_embedding_dependencies('custom', status, errors)
    if plan.embeddings.onnx:
        with contextlib.suppress(PipInstallBlockedError):
            _install_embedding_dependencies('onnx', status, errors)


def _install_backend_dependencies(category: str, engine: str, status: InstallationStatus, errors: list[str]) -> None:
    requirements: list[PipRequirement] = []
    if category == 'stt':
        requirements = STT_DEPENDENCIES.get(engine, [])
    elif category == 'tts':
        requirements = TTS_DEPENDENCIES.get(engine, [])

    if not requirements:
        return

    step_name = f"deps:{category}:{engine}"
    status.step(step_name, 'in_progress')

    try:
        _install_requirement_list(requirements)
        status.step(step_name, 'completed')
    except PipInstallBlockedError as exc:
        status.step(step_name, 'skipped', str(exc))
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Dependency install failed for {}:{}", category, engine)
        status.step(step_name, 'failed', str(exc))
        errors.append(f"{engine} dependencies: {exc}")
        raise


def _install_embedding_dependencies(target: str, status: InstallationStatus, errors: list[str]) -> None:
    requirements = EMBEDDING_DEPENDENCIES.get(target)
    if not requirements:
        return

    step_name = f"deps:embeddings:{target}"
    status.step(step_name, 'in_progress')

    try:
        _install_requirement_list(requirements)
        status.step(step_name, 'completed')
    except PipInstallBlockedError as exc:
        status.step(step_name, 'skipped', str(exc))
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Dependency install failed for embeddings:{}", target)
        status.step(step_name, 'failed', str(exc))
        errors.append(f"embeddings:{target} deps: {exc}")
        raise


def _install_requirement_list(requirements: Iterable[PipRequirement]) -> None:
    for requirement in requirements:
        _ensure_requirement(requirement)


def _ensure_requirement(requirement: PipRequirement) -> None:
    package_name = _select_package(requirement)
    if not package_name:
        return

    platforms = requirement.platforms
    if platforms and sys.platform not in platforms:
        logger.info('Skipping {} due to platform restriction', package_name)
        return

    if package_name in _INSTALLED_DEPENDENCIES:
        logger.debug('Requirement {} already processed this session', package_name)
        return

    import_name = requirement.import_name
    if import_name and importlib.util.find_spec(import_name) is not None:
        logger.info('Dependency {} already available (import {})', package_name, import_name)
        _INSTALLED_DEPENDENCIES.add(package_name)
        return

    if not _pip_allowed():
        raise PipInstallBlockedError('Package installs disabled via TLDW_SETUP_SKIP_PIP')

    logger.info('Installing dependency {}', package_name)
    # Prefer python -m pip when available; fall back to `uv pip` if pip isn't available
    def _pip_available() -> bool:
        try:
            probe = subprocess.run(  # nosec B603 - static interpreter/pip probe with fixed argv
                [sys.executable, '-m', 'pip', '--version'],
                check=False,
                capture_output=True,
                text=True,
            )
            return probe.returncode == 0
        except Exception:
            return False

    def _pip_cmd(pkg: str) -> list[str]:
        cmd = [sys.executable, '-m', 'pip', 'install', '--upgrade', '--no-input', pkg]
        idx = os.getenv('TLDW_SETUP_PIP_INDEX_URL')
        if idx:
            cmd.extend(['--index-url', idx])
        return cmd

    def _uv_pip_cmd(pkg: str) -> list[str] | None:
        uv = shutil.which('uv')
        if not uv:
            return None
        cmd = [uv, 'pip', 'install', '--upgrade', pkg]
        idx = os.getenv('TLDW_SETUP_PIP_INDEX_URL')
        if idx:
            cmd.extend(['--index-url', idx])
        return cmd

    tried_commands: list[list[str]] = []
    if _pip_available():
        tried_commands.append(_pip_cmd(package_name))
        uv_cmd = _uv_pip_cmd(package_name)
        if uv_cmd:
            tried_commands.append(uv_cmd)
    else:
        uv_cmd = _uv_pip_cmd(package_name)
        if uv_cmd:
            tried_commands.append(uv_cmd)
        tried_commands.append(_pip_cmd(package_name))

    last_err: Exception | None = None
    for cmd in tried_commands:
        try:
            logger.info('Attempting installer: {}', ' '.join(cmd[:3]))
            _run_subprocess(cmd)
            last_err = None
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            logger.warning('Installer command failed: {}', exc)
    if last_err is not None:
        raise last_err
    _INSTALLED_DEPENDENCIES.add(package_name)


def _select_package(requirement: PipRequirement) -> str | None:
    package = requirement.package
    if requirement.gpu_package or requirement.cpu_package:
        if _cuda_available() and requirement.gpu_package:
            package = requirement.gpu_package
        elif requirement.cpu_package:
            package = requirement.cpu_package
    return package


def _cuda_available() -> bool:
    """
    Detect whether CUDA should be treated as available for dependency selection.

    Priority:
      1. Environment overrides:
         - TLDW_SETUP_FORCE_GPU=1 -> force GPU packages
         - TLDW_SETUP_FORCE_CPU=1 -> force CPU-only packages
      2. Conservative environment/tool detection (safe default: CPU)
    """

    def _truthy(value: str | None) -> bool:
        if not value:
            return False
        return value.strip().lower() not in {"0", "false", "no", "off"}

    # Explicit overrides from setup scripts (e.g., Kokoro installer)
    if _truthy(os.getenv("TLDW_SETUP_FORCE_CPU")):
        return False
    if _truthy(os.getenv("TLDW_SETUP_FORCE_GPU")):
        return True

    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return False

    try:
        probe = subprocess.run(  # nosec B603 - vetted command list, no shell
            [nvidia_smi, "-L"],
            capture_output=True,
            check=False,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("CUDA probe via nvidia-smi failed: {}", exc)
        return False

    if probe.returncode != 0:
        logger.debug(
            "CUDA probe via nvidia-smi returned non-zero exit code {}",
            probe.returncode,
        )
        return False

    stdout = (probe.stdout or "").strip()
    if not stdout or "GPU" not in stdout:
        logger.debug("CUDA probe via nvidia-smi did not report any GPUs")
        return False
    return True


def cuda_available() -> bool:
    """Public compatibility wrapper for machine-profile detection helpers."""
    return _cuda_available()


def _resolve_kitten_tts_prefetch_settings() -> dict[str, str | None]:
    cache_dir = "cache/kitten_tts"

    try:
        from tldw_Server_API.app.core.TTS.tts_config import get_tts_config

        cfg = get_tts_config()
        provider_cfg = getattr(cfg, "providers", {}).get("kitten_tts")
        if provider_cfg:
            extra_params = getattr(provider_cfg, "extra_params", {}) or {}
            configured_cache_dir = extra_params.get("cache_dir")
            if configured_cache_dir:
                cache_dir = str(configured_cache_dir)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Unable to load KittenTTS prefetch settings from config: {}", exc)

    return {"cache_dir": cache_dir, "revision": None}


def _resolve_primary_stt_target(selected_profile: Any) -> tuple[str | None, str | None]:
    primary_stt_engine = None
    primary_stt_target = None
    if selected_profile.stt_plan and isinstance(selected_profile.stt_plan[0], dict):
        primary_stt_engine = selected_profile.stt_plan[0].get("engine")
        primary_stt_target = ((selected_profile.stt_plan[0].get("models") or [None])[0]) or primary_stt_engine
    return primary_stt_engine, primary_stt_target


def _extract_tts_provider_details(tts_health: dict[str, Any] | None, provider_name: str | None) -> dict[str, Any]:
    if not provider_name or not isinstance(tts_health, dict):
        return {}

    providers = tts_health.get("providers", {})
    if not isinstance(providers, dict):
        return {}

    details = providers.get("details", {})
    if isinstance(details, dict):
        provider_details = details.get(provider_name)
        if isinstance(provider_details, dict):
            return provider_details

    fallback_details = providers.get(provider_name)
    if isinstance(fallback_details, dict):
        return fallback_details
    return {}


def _tts_provider_usable(
    provider_name: str | None,
    provider_details: dict[str, Any],
    tts_health: dict[str, Any] | None,
) -> bool:
    if not provider_name:
        return False

    status = str(provider_details.get("status") or provider_details.get("availability") or "").strip().lower()
    if status in {"healthy", "enabled", "available", "ready", "ok"}:
        return True
    if status in {"failed", "disabled", "unavailable", "error", "unhealthy"}:
        return False

    if isinstance(provider_details.get("failed"), bool):
        return not provider_details["failed"]
    if isinstance(provider_details.get("initialized"), bool) and provider_details["initialized"]:
        return True

    if provider_name == "kokoro":
        providers = (tts_health or {}).get("providers", {})
        kokoro_info = providers.get("kokoro") if isinstance(providers, dict) else None
        return isinstance(kokoro_info, dict)
    return False

class InstallationStatus:
    """Persist installation progress to a status file."""

    def __init__(self, plan: InstallPlan) -> None:
        self.path = _resolve_status_file()
        self._persist_failed = False
        self.data: dict[str, Any] = {
            'plan': model_dump_compat(plan),
            'status': 'in_progress',
            'started_at': _utc_now(),
            'completed_at': None,
            'steps': [],
            'errors': [],
        }
        self._save()

    def step(self, name: str, status: str, detail: str | None = None) -> None:
        entry = {
            'name': name,
            'status': status,
            'detail': detail,
            'timestamp': _utc_now(),
        }
        self.data.setdefault('steps', []).append(entry)
        if status == 'failed' and detail:
            self.data.setdefault('errors', []).append(detail)
        self._save()

    def complete(self) -> None:
        self.data['status'] = 'completed'
        self.data['completed_at'] = _utc_now()
        self._save()

    def fail(self, message: str) -> None:
        self.data['status'] = 'failed'
        self.data['completed_at'] = _utc_now()
        self.data.setdefault('errors', []).append(message)
        self._save()

    def _save(self) -> None:
        if not self.path:
            return

        try:
            self.path.write_text(json.dumps(self.data, indent=2), encoding='utf-8')
        except Exception:  # noqa: BLE001
            if not self._persist_failed:
                logger.warning(
                    'Failed to persist setup install status to {}; continuing in-memory.',
                    self.path,
                    exc_info=True,
                )
            self._persist_failed = True
            self.path = None
        _record_latest_status(self.data)


class DownloadBlockedError(RuntimeError):
    """Raised when installer downloads are disabled or unavailable."""


def _downloads_allowed() -> bool:
    flag = os.getenv('TLDW_SETUP_SKIP_DOWNLOADS')
    if not flag:
        return True
    return flag.strip().lower() not in {'1', 'true', 'yes', 'y', 'on'}


def downloads_allowed() -> bool:
    """Public wrapper for setup download-policy checks."""
    return _downloads_allowed()


def _ensure_downloads_allowed(target: str) -> None:
    if _downloads_allowed():
        return
    raise DownloadBlockedError(
        f'Downloads disabled via TLDW_SETUP_SKIP_DOWNLOADS; skipped {target}.',
    )


class PipInstallBlockedError(RuntimeError):
    """Raised when pip installation is disabled for the setup installer."""


def _pip_allowed() -> bool:
    flag = os.getenv('TLDW_SETUP_SKIP_PIP')
    if not flag:
        return True
    return flag.strip().lower() not in {'1', 'true', 'yes', 'y', 'on'}


def _record_latest_status(data: dict[str, Any]) -> None:
    global _LATEST_STATUS_DATA
    _LATEST_STATUS_DATA = json.loads(json.dumps(data))


def _persist_install_status_snapshot(data: dict[str, Any]) -> None:
    """Persist an install status snapshot with the same shape used by InstallationStatus."""

    path = _resolve_status_file()
    if path:
        try:
            path.write_text(json.dumps(data, indent=2), encoding='utf-8')
        except Exception:  # noqa: BLE001
            logger.warning('Failed to persist qualified setup install status to {}', path, exc_info=True)
    _record_latest_status(data)


# --- HTTPX network error detection -------------------------------------------
def _is_httpx_network_error(exc: Exception) -> bool:
    """Return True if the exception is an httpx HTTP/network error.

    Uses module/name checks to avoid a hard dependency at module import time.
    """
    module = getattr(exc.__class__, "__module__", "") or ""
    if not module.startswith("httpx"):
        return False
    name = exc.__class__.__name__
    return name.endswith("Error") or name.endswith("Exception")


def _is_requests_network_error(exc: Exception) -> bool:
    """Return True if the exception looks like a requests/huggingface network error."""
    name = type(exc).__name__
    module = getattr(exc, "__module__", "") or ""
    if module.startswith("requests."):
        return True
    if name in {"RequestException", "HTTPError", "ConnectionError", "Timeout"} or name.endswith("RequestException"):
        return True
    if "HTTPError" in name:
        return True
    msg = str(exc).lower()
    return bool("timeout" in msg or "connection" in msg or "dns" in msg or "network" in msg)


def get_install_status_snapshot() -> dict[str, Any] | None:
    """Return the most recent install status if available."""

    for root in _candidate_status_dirs():
        path = root / STATUS_FILENAME if root else None
        if not path or not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            _record_latest_status(data)
            return json.loads(json.dumps(data))
        except Exception:  # noqa: BLE001
            logger.exception('Failed to read install status from {}', path)

    if _LATEST_STATUS_DATA is not None:
        return json.loads(json.dumps(_LATEST_STATUS_DATA))

    return None

def _utc_now() -> str:
    return datetime.utcnow().isoformat() + 'Z'

def execute_install_plan(plan_payload: dict[str, Any]) -> dict[str, Any] | None:
    """Background entry point to execute an installation plan."""
    try:
        validate = getattr(InstallPlan, 'model_validate', None) or getattr(InstallPlan, 'parse_obj', None)
        if not validate:
            raise TypeError('No compatible Pydantic validation method found on InstallPlan')
        plan = validate(plan_payload)
    except Exception:  # noqa: BLE001
        logger.exception("Received invalid install plan")
        return

    if plan.is_empty():
        logger.info("Install plan empty; nothing to install.")
        return

    readiness = audio_readiness_store.get_audio_readiness_store()
    readiness.update(
        status='provisioning',
        remediation_items=[],
        last_verification=None,
    )

    status = InstallationStatus(plan)
    errors: list[str] = []

    try:
        _install_dependencies(plan, status, errors)
        _install_stt(plan, status, errors)
        _install_tts(plan, status, errors)
        _install_embeddings(plan, status, errors)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Install plan execution failed")
        errors.append(str(exc))

    if errors:
        status.fail('; '.join(errors))
        readiness.update(
            status='failed',
            remediation_items=errors,
        )
    else:
        status.complete()
        readiness.update(
            status='partial',
            remediation_items=['Run audio verification to confirm readiness.'],
        )
    return json.loads(json.dumps(status.data))


def build_install_plan_from_bundle(
    bundle_id: str,
    resource_profile: str = DEFAULT_AUDIO_RESOURCE_PROFILE,
    tts_choice: str | None = None,
) -> InstallPlan:
    """Expand a curated bundle into the existing installer plan schema."""

    bundle = get_audio_bundle_catalog().bundle_by_id(bundle_id)
    selected_profile = bundle.profile_by_id(resource_profile)
    canonical_tts_choice = _resolve_curated_tts_choice(selected_profile, tts_choice)
    return InstallPlan.model_validate(
        {
            "stt": selected_profile.stt_plan,
            "tts": selected_profile.tts_plan_for_choice(canonical_tts_choice),
            "embeddings": selected_profile.embeddings_plan,
        }
    )


def _resolve_curated_tts_choice(selected_profile: Any, tts_choice: str | None) -> str | None:
    """Normalize a curated TTS choice and treat invalid choices as request errors."""

    try:
        return selected_profile.canonical_tts_choice(tts_choice)
    except KeyError as exc:
        raise ValueError(str(exc)) from exc


def _entry_suffix(entry: Any, attribute: str, default: str = "default") -> str:
    values = getattr(entry, attribute, None) or []
    normalized = [str(value).strip().replace(" ", "_") for value in values if str(value).strip()]
    return "-".join(normalized) if normalized else default


def _plan_step_names(
    plan: InstallPlan,
    *,
    bundle_id: str,
    resource_profile: str,
    tts_choice: str | None = None,
    catalog_version: str = AUDIO_BUNDLE_CATALOG_VERSION,
) -> set[str]:
    step_names: set[str] = set()
    selection_key = build_audio_selection_key(
        bundle_id,
        resource_profile,
        catalog_version,
        tts_choice=tts_choice,
    )

    for entry in plan.stt:
        step_names.add(f"{selection_key}:deps:stt:{entry.engine}")
        step_names.add(f"{selection_key}:stt:{entry.engine}:{_entry_suffix(entry, 'models')}")
    if plan.stt:
        step_names.add(f"{selection_key}:deps:stt:silero_vad")
        step_names.add(f"{selection_key}:stt:silero_vad")

    for entry in plan.tts:
        step_names.add(f"{selection_key}:deps:tts:{entry.engine}")
        step_names.add(f"{selection_key}:tts:{entry.engine}:{_entry_suffix(entry, 'variants')}")

    if plan.embeddings.huggingface:
        step_names.add(f"{selection_key}:deps:embeddings:huggingface")
        step_names.add(f"{selection_key}:embeddings:huggingface")
    if plan.embeddings.custom:
        step_names.add(f"{selection_key}:deps:embeddings:custom")
        step_names.add(f"{selection_key}:embeddings:custom")
    if plan.embeddings.onnx:
        step_names.add(f"{selection_key}:deps:embeddings:onnx")
        step_names.add(f"{selection_key}:embeddings:onnx")

    return step_names


def _completed_step_names(snapshot: dict[str, Any] | None) -> set[str]:
    if not snapshot:
        return set()
    completed: set[str] = set()
    for step in snapshot.get("steps", []):
        if step.get("status") == "completed" and step.get("name"):
            completed.add(str(step["name"]))
    return completed


def _system_prerequisite_status(
    bundle_step: AudioBundleStep,
    machine_profile: audio_profile_service.MachineProfile,
) -> tuple[str, str | None]:
    if bundle_step.step_id == "ffmpeg" and machine_profile.ffmpeg_available:
        return "completed", "FFmpeg already available."
    if bundle_step.step_id == "espeak_ng" and machine_profile.espeak_available:
        return "completed", "eSpeak already available."
    if bundle_step.automation_tier == AutomationTier.GUIDED:
        return "guided_action_required", bundle_step.detail
    if bundle_step.automation_tier == AutomationTier.MANUAL_BLOCKED:
        return "failed", bundle_step.detail
    return "pending", bundle_step.detail


def _qualify_install_step_name(
    name: str,
    plan: InstallPlan,
    *,
    bundle_id: str,
    resource_profile: str,
    tts_choice: str | None,
    catalog_version: str,
) -> str:
    selection_key = build_audio_selection_key(
        bundle_id,
        resource_profile,
        catalog_version,
        tts_choice=tts_choice,
    )
    if name.startswith("deps:") or name.startswith("embeddings:") or name == "stt:silero_vad":
        return f"{selection_key}:{name}"
    if name.startswith("stt:"):
        engine = name.split(":", 1)[1]
        entry = next((candidate for candidate in plan.stt if candidate.engine == engine), None)
        return f"{selection_key}:stt:{engine}:{_entry_suffix(entry, 'models') if entry else 'default'}"
    if name.startswith("tts:"):
        engine = name.split(":", 1)[1]
        entry = next((candidate for candidate in plan.tts if candidate.engine == engine), None)
        return f"{selection_key}:tts:{engine}:{_entry_suffix(entry, 'variants') if entry else 'default'}"
    return f"{selection_key}:{name}"


def _qualify_install_result(
    install_result: dict[str, Any],
    plan: InstallPlan,
    *,
    bundle_id: str,
    resource_profile: str,
    tts_choice: str | None,
    catalog_version: str,
) -> dict[str, Any]:
    qualified = json.loads(json.dumps(install_result))
    qualified["steps"] = [
        {
            **step,
            "name": _qualify_install_step_name(
                step.get("name", ""),
                plan,
                bundle_id=bundle_id,
                resource_profile=resource_profile,
                tts_choice=tts_choice,
                catalog_version=catalog_version,
            ),
        }
        for step in install_result.get("steps", [])
    ]
    return qualified


def execute_audio_bundle(
    bundle_id: str,
    *,
    resource_profile: str = DEFAULT_AUDIO_RESOURCE_PROFILE,
    tts_choice: str | None = None,
    safe_rerun: bool = False,
) -> dict[str, Any]:
    """Provision a curated audio bundle using the existing installer substrate."""

    bundle = get_audio_bundle_catalog().bundle_by_id(bundle_id)
    selected_profile = bundle.profile_by_id(resource_profile)
    canonical_tts_choice = _resolve_curated_tts_choice(selected_profile, tts_choice)
    plan = build_install_plan_from_bundle(
        bundle_id,
        resource_profile=resource_profile,
        tts_choice=canonical_tts_choice,
    )
    plan_payload = model_dump_compat(plan)
    machine_profile = audio_profile_service.detect_machine_profile()
    readiness = audio_readiness_store.get_audio_readiness_store()
    selection_key = build_audio_selection_key(
        bundle_id,
        resource_profile,
        bundle.catalog_version,
        tts_choice=canonical_tts_choice,
    )
    readiness.update(
        status="provisioning",
        selected_bundle_id=bundle_id,
        selected_resource_profile=resource_profile,
        tts_choice=canonical_tts_choice,
        catalog_version=bundle.catalog_version,
        selection_key=selection_key,
        machine_profile=machine_profile.model_dump(),
        remediation_items=[],
        last_verification=None,
    )

    result_steps: list[dict[str, Any]] = []
    for prerequisite in bundle.system_prerequisites:
        status_value, detail = _system_prerequisite_status(prerequisite, machine_profile)
        result_steps.append(
            {
                "name": f"{selection_key}:system:{prerequisite.step_id}",
                "status": status_value,
                "detail": detail,
            }
        )

    expected_steps = _plan_step_names(
        plan,
        bundle_id=bundle_id,
        resource_profile=resource_profile,
        tts_choice=canonical_tts_choice,
        catalog_version=bundle.catalog_version,
    )
    completed_steps = _completed_step_names(get_install_status_snapshot()) if safe_rerun else set()
    if safe_rerun and expected_steps and expected_steps.issubset(completed_steps):
        result_steps.extend(
            {
                "name": step_name,
                "status": "skipped",
                "detail": "Already satisfied by a previous successful install run.",
            }
            for step_name in sorted(expected_steps)
        )
        readiness.update(
            status="partial",
            remediation_items=["Run audio verification to confirm readiness."],
        )
        return {
            "bundle_id": bundle_id,
            "resource_profile": resource_profile,
            "tts_choice": canonical_tts_choice,
            "selection_key": selection_key,
            "safe_rerun": True,
            "install_plan": plan_payload,
            "steps": result_steps,
            "status": "partial",
        }

    install_result = execute_install_plan(plan_payload) or {"steps": [], "status": "failed", "errors": []}
    qualified_install_result = _qualify_install_result(
        install_result,
        plan,
        bundle_id=bundle_id,
        resource_profile=resource_profile,
        tts_choice=canonical_tts_choice,
        catalog_version=bundle.catalog_version,
    )
    if qualified_install_result:
        qualified_install_result["plan"] = plan_payload
        _persist_install_status_snapshot(qualified_install_result)
    result_steps.extend(qualified_install_result.get("steps", []))
    return {
        "bundle_id": bundle_id,
        "resource_profile": resource_profile,
        "tts_choice": canonical_tts_choice,
        "selection_key": selection_key,
        "safe_rerun": safe_rerun,
        "install_plan": plan_payload,
        "steps": result_steps,
        "status": qualified_install_result.get("status"),
    }


async def _resolve_health_call(result: Any) -> Any:
    if inspect.isawaitable(result):
        return await result
    return result


def _remediation_item(code: str, message: str, *, action: str = "safe_rerun") -> dict[str, str]:
    return {
        "code": code,
        "message": message,
        "action": action,
    }


def _sanitize_health_payload(value: Any) -> Any:
    """Drop nested debug-only exception details from user-visible health payloads."""
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"details", "exception", "traceback", "stack", "stack_trace"}:
                continue
            sanitized[key] = _sanitize_health_payload(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_health_payload(item) for item in value]
    return value


async def verify_audio_bundle_async_for_profile(
    bundle: Any,
    selected_profile: Any,
    *,
    tts_choice: str | None = None,
) -> dict[str, Any]:
    """Verify the primary STT/TTS paths for a selected bundle profile."""

    bundle_id = bundle.bundle_id
    resource_profile = selected_profile.profile_id
    machine_profile = audio_profile_service.detect_machine_profile()
    canonical_tts_choice = _resolve_curated_tts_choice(selected_profile, tts_choice)
    selection_key = build_audio_selection_key(
        bundle_id,
        resource_profile,
        bundle.catalog_version,
        tts_choice=canonical_tts_choice,
    )
    primary_stt_engine, primary_stt_target = _resolve_primary_stt_target(selected_profile)
    stt_health = _sanitize_health_payload(
        await _resolve_health_call(audio_health.collect_setup_stt_health(model=primary_stt_target))
    )
    tts_health = _sanitize_health_payload(
        await _resolve_health_call(audio_health.collect_setup_tts_health())
    )

    primary_tts_plan = selected_profile.tts_plan_for_choice(canonical_tts_choice)
    primary_tts_engine = primary_tts_plan[0]["engine"] if primary_tts_plan else None
    remediation_items: list[dict[str, str]] = []
    warning_items: list[dict[str, str]] = []

    stt_usable = bool((stt_health or {}).get("usable", False))
    primary_tts_details = _extract_tts_provider_details(tts_health, primary_tts_engine)
    tts_usable = _tts_provider_usable(primary_tts_engine, primary_tts_details, tts_health)

    if not machine_profile.ffmpeg_available:
        remediation_items.append(
            _remediation_item(
                "FFMPEG_MISSING",
                "Install FFmpeg and rerun verification.",
            )
        )

    providers = (tts_health or {}).get("providers", {})
    kokoro_info = providers.get("kokoro") if isinstance(providers, dict) else None
    if primary_tts_engine == "kokoro" and not bool((kokoro_info or {}).get("espeak_lib_exists", False)):
        remediation_items.append(
            _remediation_item(
                "KOKORO_ESPEAK_MISSING",
                "Install espeak-ng and rerun verification.",
            )
        )
        tts_usable = False

    if not stt_usable:
        remediation_items.append(
            _remediation_item(
                "STT_UNUSABLE",
                "Primary STT path is not usable. Rerun provisioning or inspect model downloads.",
            )
        )
    if not tts_usable:
        remediation_items.append(
            _remediation_item(
                "TTS_UNHEALTHY",
                "Primary TTS path is not healthy. Rerun provisioning or inspect TTS logs.",
            )
        )

    provider_details = providers.get("details", {}) if isinstance(providers, dict) else {}
    if isinstance(provider_details, dict):
        for provider_name, details in provider_details.items():
            if provider_name == primary_tts_engine or not isinstance(details, dict):
                continue
            if str(details.get("status", details.get("availability", ""))).lower() == "failed":
                warning_items.append(
                    _remediation_item(
                        "SECONDARY_PROVIDER_FAILED",
                        f"Secondary TTS provider {provider_name} is unavailable.",
                        action="advisory",
                    )
                )

    if remediation_items:
        status = "partial" if (stt_usable or tts_usable) else "failed"
    elif warning_items:
        status = "ready_with_warnings"
    else:
        status = "ready"

    verified_at = _utc_now()
    result = {
        "bundle_id": bundle_id,
        "selected_resource_profile": resource_profile,
        "tts_choice": canonical_tts_choice,
        "selection_key": selection_key,
        "status": status,
        "machine_profile": machine_profile.model_dump(),
        "stt_health": stt_health,
        "tts_health": tts_health,
        "remediation_items": remediation_items + warning_items,
        "verified_at": verified_at,
    }

    audio_readiness_store.get_audio_readiness_store().update(
        status=status,
        selected_bundle_id=bundle_id,
        selected_resource_profile=resource_profile,
        tts_choice=canonical_tts_choice,
        catalog_version=bundle.catalog_version,
        selection_key=selection_key,
        machine_profile=machine_profile.model_dump(),
        last_verification={
            "bundle_id": bundle_id,
            "selected_resource_profile": resource_profile,
            "tts_choice": canonical_tts_choice,
            "selection_key": selection_key,
            "verified_at": verified_at,
            "stt_health": stt_health,
            "tts_health": tts_health,
        },
        remediation_items=result["remediation_items"],
    )
    return result


async def verify_audio_bundle_async(
    bundle_id: str,
    *,
    resource_profile: str = DEFAULT_AUDIO_RESOURCE_PROFILE,
    tts_choice: str | None = None,
) -> dict[str, Any]:
    """Verify the primary STT/TTS paths for a curated audio bundle."""

    bundle = get_audio_bundle_catalog().bundle_by_id(bundle_id)
    selected_profile = bundle.profile_by_id(resource_profile)
    return await verify_audio_bundle_async_for_profile(
        bundle,
        selected_profile,
        tts_choice=tts_choice,
    )


def verify_audio_bundle(
    bundle_id: str,
    *,
    resource_profile: str = DEFAULT_AUDIO_RESOURCE_PROFILE,
    tts_choice: str | None = None,
) -> dict[str, Any]:
    """Synchronous wrapper for setup verification tests and scripts."""

    return asyncio.run(
        verify_audio_bundle_async(
            bundle_id,
            resource_profile=resource_profile,
            tts_choice=tts_choice,
        )
    )

def _install_stt(plan: InstallPlan, status: InstallationStatus, errors: list[str]) -> None:
    any_stt = False
    for entry in plan.stt:
        any_stt = True
        step_name = f"stt:{entry.engine}"
        status.step(step_name, 'in_progress')
        try:
            if entry.engine == 'faster_whisper':
                _install_faster_whisper(entry.models)
            elif entry.engine == 'qwen2_audio':
                _install_qwen2_audio()
            elif entry.engine == 'nemo_parakeet_standard':
                _install_nemo_parakeet('standard')
            elif entry.engine == 'nemo_parakeet_onnx':
                _install_nemo_parakeet('onnx')
            elif entry.engine == 'nemo_parakeet_mlx':
                _install_nemo_parakeet('mlx')
            elif entry.engine == 'nemo_canary':
                _install_nemo_canary()
            status.step(step_name, 'completed')
        except DownloadBlockedError as exc:
            logger.info('Skipping STT install {}: {}', entry.engine, exc)
            status.step(step_name, 'skipped', str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.exception("STT install failed for {}", entry.engine)
            status.step(step_name, 'failed', str(exc))
            errors.append(f"{entry.engine}: {exc}")

    # Install Silero VAD once when any STT engine is selected so unified streaming
    # turn detection has its dependency ready by default.
    if any_stt:
        step_name = "stt:silero_vad"
        status.step(step_name, "in_progress")
        try:
            _install_silero_vad()
            status.step(step_name, "completed")
        except DownloadBlockedError as exc:
            logger.info("Skipping Silero VAD install: {}", exc)
            status.step(step_name, "skipped", str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Silero VAD install failed")
            status.step(step_name, "failed", str(exc))
            errors.append(f"silero_vad: {exc}")

def _install_tts(plan: InstallPlan, status: InstallationStatus, errors: list[str]) -> None:
    for entry in plan.tts:
        step_name = f"tts:{entry.engine}"
        status.step(step_name, 'in_progress')
        try:
            if entry.engine == 'kokoro':
                _install_kokoro(entry.variants)
            elif entry.engine == 'kitten_tts':
                _install_kitten_tts(entry.variants)
            elif entry.engine == 'dia':
                _install_dia()
            elif entry.engine == 'higgs':
                _install_higgs()
            elif entry.engine == 'vibevoice':
                _install_vibevoice(entry.variants)
            elif entry.engine == 'omnivoice':
                _install_omnivoice()
            status.step(step_name, 'completed')
        except DownloadBlockedError as exc:
            logger.info('Skipping TTS install {}: {}', entry.engine, exc)
            status.step(step_name, 'skipped', str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.exception("TTS install failed for {}", entry.engine)
            status.step(step_name, 'failed', str(exc))
            errors.append(f"{entry.engine}: {exc}")

def _install_embeddings(plan: InstallPlan, status: InstallationStatus, errors: list[str]) -> None:
    if plan.embeddings.huggingface:
        step_name = 'embeddings:huggingface'
        status.step(step_name, 'in_progress')
        try:
            _download_huggingface_models(plan.embeddings.huggingface)
            status.step(step_name, 'completed')
        except DownloadBlockedError as exc:
            logger.info('Skipping Hugging Face embeddings download: {}', exc)
            status.step(step_name, 'skipped', str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to download huggingface embeddings")
            status.step(step_name, 'failed', str(exc))
            errors.append(f"embeddings:huggingface: {exc}")
    if plan.embeddings.custom:
        step_name = 'embeddings:custom'
        status.step(step_name, 'in_progress')
        try:
            _download_huggingface_models(plan.embeddings.custom)
            _append_trusted_embeddings(plan.embeddings.custom)
            status.step(step_name, 'completed')
        except DownloadBlockedError as exc:
            logger.info('Skipping custom embedding downloads: {}', exc)
            status.step(step_name, 'skipped', str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to process custom embedding models")
            status.step(step_name, 'failed', str(exc))
            errors.append(f"embeddings:custom: {exc}")

# --- Individual installers -------------------------------------------------

def _install_faster_whisper(models: list[str]) -> None:
    _ensure_downloads_allowed('faster-whisper checkpoints')
    try:
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib import WhisperModel
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError('faster-whisper not available. Ensure dependency is installed.') from exc

    for model_name in models or DEFAULT_WHISPER_MODELS:
        logger.info("Downloading faster-whisper checkpoint {}", model_name)
        try:
            instance = WhisperModel(model_name, device='cpu')
            del instance
        except Exception as exc:  # noqa: BLE001
            if _is_httpx_network_error(exc):
                raise DownloadBlockedError(f'Network unavailable while downloading {model_name}.') from exc
            raise


def _install_silero_vad() -> None:
    """
    Install Silero VAD model assets used for streaming turn detection.

    This leverages the existing `_lazy_import_silero_vad` helper, which is responsible
    for configuring the cache directory (under `models/torch_home` when available),
    calling `torch.hub.load('snakers4/silero-vad', 'silero_vad', ...)` or a locally
    checked-out repo, and validating the returned `(model, utils)` tuple.

    Downloads are gated by TLDW_SETUP_SKIP_DOWNLOADS in the same way as other model
    installers; when disabled, this step is marked as skipped.
    """
    _ensure_downloads_allowed("Silero VAD model")
    try:
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.VAD_Lib import (
            _lazy_import_silero_vad,
            get_silero_vad_unavailable_reason,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Silero VAD helper not available; ensure audio dependencies are installed.") from exc

    model, utils = _lazy_import_silero_vad()
    if not model or not utils:
        reason = get_silero_vad_unavailable_reason()
        if reason:
            raise RuntimeError(f"Silero VAD model could not be loaded: {reason}")
        raise RuntimeError("Silero VAD model could not be loaded; check network or torch hub configuration.")


def _install_qwen2_audio() -> None:
    _ensure_downloads_allowed('Qwen2Audio model assets')
    try:
        import torch
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError('transformers (with Qwen2Audio) is required for Qwen installs.') from exc

    repo = 'Qwen/Qwen2-Audio-7B-Instruct'
    repo_revision = _resolve_hf_revision(repo)
    logger.info("Fetching Qwen2Audio assets from {}", repo)
    try:
        AutoProcessor.from_pretrained(repo, revision=repo_revision)  # nosec B615
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        Qwen2AudioForConditionalGeneration.from_pretrained(  # nosec B615
            repo,
            revision=repo_revision,
            torch_dtype=dtype,
            device_map='cpu',
        )
    except Exception as exc:  # noqa: BLE001
        if _is_httpx_network_error(exc):
            raise DownloadBlockedError(f'Network unavailable while downloading {repo}.') from exc
        raise


def _install_nemo_parakeet(variant: str) -> None:
    _ensure_downloads_allowed(f'NeMo Parakeet {variant} weights')
    try:
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo import (
            load_parakeet_model,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError('nemo_toolkit is required for NeMo installations.') from exc

    logger.info("Loading NeMo Parakeet variant {} to trigger download", variant)
    model = load_parakeet_model(variant)
    if model is None:
        raise RuntimeError(f'Failed to load Parakeet variant {variant}; check nemo dependencies.')


def _install_nemo_canary() -> None:
    _ensure_downloads_allowed('NeMo Canary model')
    try:
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo import load_canary_model
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError('nemo_toolkit is required for NeMo installations.') from exc

    logger.info('Loading NeMo Canary to trigger download')
    model = load_canary_model()
    if model is None:
        raise RuntimeError('Failed to load NeMo Canary model; verify nemo_toolkit installation.')


def _install_kokoro(variants: list[str]) -> None:
    targets = set(variants or ['onnx'])
    config = _load_config()
    # Default to v1.0 ONNX layout
    default_model_path = Path('models/kokoro/onnx/model.onnx')
    model_path = Path(config.get('TTS-Settings', {}).get('kokoro_model_path', str(default_model_path)))
    # Destination for voices directory (used by v1.0 ONNX and PyTorch variant)
    default_voices_dir = model_path.parent.parent / 'voices'
    voices_dir = Path(config.get('TTS-Settings', {}).get('kokoro_voices_json', str(default_voices_dir)))

    # Ensure destination directories exist
    model_path.parent.mkdir(parents=True, exist_ok=True)
    voices_dir.mkdir(parents=True, exist_ok=True)

    # Source repo for v1.0 ONNX
    onnx_repo = 'onnx-community/Kokoro-82M-v1.0-ONNX-timestamped'

    if 'onnx' in targets:
        # Download main ONNX model (user may replace with fp16/quantized variant later)
        _download_hf_file(onnx_repo, 'onnx/model.onnx', model_path)
    if 'voices' in targets:
        # Download the voices directory
        _download_hf_dir(onnx_repo, 'voices', voices_dir)


def _install_kitten_tts(variants: list[str]) -> None:
    _ensure_downloads_allowed('KittenTTS model assets')
    try:
        from tldw_Server_API.app.core.TTS.vendors.kittentts_compat import download_model_assets
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError('KittenTTS compatibility runtime is required for asset downloads.') from exc

    settings = _resolve_kitten_tts_prefetch_settings()
    variant_to_repo = {
        'nano': 'KittenML/kitten-tts-nano-0.8',
        'nano-int8': 'KittenML/kitten-tts-nano-0.8-int8',
        'micro': 'KittenML/kitten-tts-micro-0.8',
        'mini': 'KittenML/kitten-tts-mini-0.8',
    }
    selected = [str(variant).strip().lower() for variant in (variants or ['nano']) if str(variant).strip()]
    if not selected:
        selected = ['nano']
    unknown = [variant for variant in selected if variant not in variant_to_repo]
    if unknown:
        raise ValueError(f"Unsupported KittenTTS variants: {', '.join(unknown)}")
    for variant in selected:
        repo_id = variant_to_repo[variant]
        logger.info('Prefetching KittenTTS assets for {}', repo_id)
        try:
            download_model_assets(
                repo_id,
                cache_dir=settings.get("cache_dir"),
                auto_download=True,
                revision=settings.get("revision"),
            )
        except Exception as exc:  # noqa: BLE001
            if _is_httpx_network_error(exc) or _is_requests_network_error(exc):
                raise DownloadBlockedError(f'Network unavailable while downloading {repo_id}.') from exc
            raise


def _install_dia() -> None:
    logger.info('Downloading Dia dialogue TTS model (nari-labs/dia)')
    _snapshot_repo('nari-labs/dia')


def _install_higgs() -> None:
    logger.info('Downloading Higgs Audio V2 model')
    _snapshot_repo('bosonai/higgs-audio-v2-generation-3B-base')
    _snapshot_repo('bosonai/higgs-audio-v2-tokenizer')


def _install_vibevoice(variants: list[str]) -> None:
    _ensure_downloads_allowed('VibeVoice assets')
    selected = set(variants or ['1.5B'])
    if '1.5B' in selected:
        _snapshot_repo('microsoft/VibeVoice-1.5B')
    if '7B' in selected:
        # Official 7B repository
        _snapshot_repo('vibevoice/VibeVoice-7B')
    if '7B-Q8' in selected:
        # Community 8-bit quantized 7B variant (reduced VRAM usage)
        _snapshot_repo('FabioSarracino/VibeVoice-Large-Q8')


def _install_omnivoice() -> None:
    from Helper_Scripts.TTS_Installers import install_tts_omnivoice_sidecar as installer

    _ensure_downloads_allowed("OmniVoice sidecar runtime")
    repo_root = Path(__file__).resolve().parents[4]
    configured_model_path = os.getenv("TLDW_OMNIVOICE_MODEL_PATH")
    if not configured_model_path:
        raise RuntimeError("TLDW_OMNIVOICE_MODEL_PATH is required to enable OmniVoice")
    candidate_model_path = Path(configured_model_path).expanduser()
    if not candidate_model_path.is_absolute():
        candidate_model_path = repo_root / candidate_model_path
    try:
        model_path = installer.validate_local_model_path(candidate_model_path)
    except SystemExit as exc:
        raise RuntimeError(str(exc)) from exc

    runtime_base = repo_root / "models" / "omnivoice_sidecar"
    source_checkout = installer.resolve_source_checkout(repo_root=repo_root)
    layout = installer.build_runtime_layout(runtime_base, repo_root=repo_root)
    installer.create_runtime_layout(layout)
    installer.clone_repository(installer.DEFAULT_REPO_URL, source_checkout)
    installer.create_virtualenv(layout.venv_dir)
    installer.install_sidecar_runtime(
        interpreter_path=layout.interpreter_path,
        repo_root=repo_root,
        source_checkout=source_checkout,
    )
    missing = installer.validate_runtime_layout(layout)
    if missing:
        raise RuntimeError(f"OmniVoice runtime layout incomplete: {', '.join(missing)}")
    config_patched = installer.patch_tts_config(
        config_path=repo_root / installer.DEFAULT_CONFIG_PATH,
        layout=layout,
        source_checkout=source_checkout,
        model_path=model_path,
        repo_root=repo_root,
    )
    if not config_patched:
        logger.error("OmniVoice provider configuration was not updated after runtime installation")
        raise RuntimeError("OmniVoice provider configuration could not be updated")


def _download_huggingface_models(models: list[str]) -> None:
    for model_id in models:
        logger.info('Downloading embedding model {}', model_id)
        _snapshot_repo(model_id)


def _append_trusted_embeddings(models: list[str]) -> None:
    if not models:
        return

    parser = ConfigParser()
    config_path = setup_manager.get_config_file_path()
    parser.read(config_path)
    section = 'Embeddings'
    current = []
    if parser.has_option(section, 'trusted_hf_remote_code_models'):
        current = [value.strip() for value in parser.get(section, 'trusted_hf_remote_code_models').split(',') if value.strip()]
    merged = sorted(set(current + models), key=str.lower)
    setup_manager.update_config({section: {'trusted_hf_remote_code_models': ', '.join(merged)}})


def _load_config() -> dict[str, dict[str, Any]]:
    try:
        configs = load_and_log_configs()
        if isinstance(configs, dict):
            return configs
    except Exception:
        logger.debug('Falling back to empty config snapshot for installer metadata')
    return {}


def _download_hf_file(repo_id: str, filename: str, destination: Path) -> None:
    _ensure_downloads_allowed(f'{repo_id}/{filename}')
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError('huggingface_hub package is required for model downloads.') from exc

    force = _force_downloads()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        logger.info('Skip existing file {}', destination)
        return
    try:
        # Download into cache, then copy to the exact destination path
        repo_revision = _resolve_hf_revision(repo_id)
        src_fp = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=repo_revision,
            force_download=force,
        )  # nosec B615
        shutil.copy2(src_fp, destination)
    except Exception as exc:  # noqa: BLE001
        if _is_httpx_network_error(exc) or _is_requests_network_error(exc):
            raise DownloadBlockedError(f'Network unavailable while downloading {repo_id}/{filename}.') from exc
        raise


def _download_hf_dir(repo_id: str, subdir: str, destination: Path) -> None:
    """Download a directory from a HuggingFace repo via snapshot and copy the subdir to destination."""
    _ensure_downloads_allowed(f'{repo_id}/{subdir} directory')
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError('huggingface_hub package is required for model downloads.') from exc

    force = _force_downloads()
    if destination.exists() and any(destination.iterdir()) and not force:
        logger.info('Skip existing directory {}', destination)
        return

    try:
        # Download snapshot into a temporary folder then copy requested subdir
        repo_revision = _resolve_hf_revision(repo_id)
        import tempfile
        with tempfile.TemporaryDirectory(prefix="tldw_hf_") as _td:
            snapshot_path = Path(snapshot_download(
                repo_id=repo_id,
                local_dir=str(_td),
                revision=repo_revision,
                allow_patterns=[f"{subdir}", f"{subdir}/*", f"{subdir}/**"],
                force_download=force,
            ))  # nosec B615
            src = snapshot_path / subdir
            if not src.exists():
                raise FileNotFoundError(f'Subdirectory {subdir!r} not found in snapshot of {repo_id}')
            # Prepare destination directory
            if destination.exists() and force:
                if destination.is_dir():
                    shutil.rmtree(destination)
                else:
                    destination.unlink()
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Copy directory tree while tempdir is alive
            shutil.copytree(src, destination, dirs_exist_ok=True)
    except Exception as exc:  # noqa: BLE001
        if _is_httpx_network_error(exc) or _is_requests_network_error(exc):
            raise DownloadBlockedError(f'Network unavailable while downloading {repo_id}/{subdir}.') from exc
        raise

def _force_downloads() -> bool:
    """Whether to force re-download/overwrite. Controlled via env flags."""
    for key in ('TLDW_SETUP_FORCE_DOWNLOADS', 'TLDW_SETUP_FORCE', 'TLDW_FORCE'):
        v = os.getenv(key)
        if v and v not in ('0', 'false', 'False', 'no', 'NO'):
            return True
    return False


def _resolve_hf_revision(repo_id: str) -> str | None:
    """Resolve optional pinned HF revision from environment."""
    normalized = "".join(ch if ch.isalnum() else "_" for ch in str(repo_id)).upper()
    return (
        os.getenv(f"HF_REVISION_{normalized}")
        or os.getenv("HF_DEFAULT_REVISION")
        or None
    )

def _snapshot_repo(repo_id: str) -> None:
    """Prefetch a HuggingFace repository snapshot into the local cache.

    The download respects the setup download policy, optional pinned
    repository revisions, and force-download environment flags. Network errors
    are normalized to DownloadBlockedError so install-plan status reporting can
    present a clear remediation message.
    """
    _ensure_downloads_allowed(f'{repo_id} snapshot')
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError('huggingface_hub package is required for model downloads.') from exc

    try:
        # Prefetch into cache; no local_dir required and no symlink flag
        snapshot_download(  # nosec B615
            repo_id=repo_id,
            revision=_resolve_hf_revision(repo_id),
            force_download=_force_downloads(),
        )
    except Exception as exc:  # noqa: BLE001
        if _is_httpx_network_error(exc) or _is_requests_network_error(exc):
            raise DownloadBlockedError(f'Network unavailable while downloading {repo_id}.') from exc
        raise


def _run_subprocess(command: list[str]) -> None:
    logger.info('Running command: {}', ' '.join(command))
    result = subprocess.run(  # nosec B603 - command argv is assembled from trusted setup requirement mappings
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f'Command failed with exit code {result.returncode}')
    if result.stdout:
        logger.debug(result.stdout)
@dataclass(frozen=True)
class PipRequirement:
    package: str
    import_name: str | None = None
    gpu_package: str | None = None
    cpu_package: str | None = None
    platforms: set[str] | None = None


# Dependency manifests keyed by backend type
STT_DEPENDENCIES: dict[str, list[PipRequirement]] = {
    'faster_whisper': [
        PipRequirement(package='faster-whisper>=1.0.0', import_name='faster_whisper', cpu_package='faster-whisper>=1.0.0'),
    ],
    'silero_vad': [
        PipRequirement(package='torch>=2.2.0', import_name='torch'),
        PipRequirement(package='torchaudio>=2.2.0', import_name='torchaudio'),
    ],
    'qwen2_audio': [
        PipRequirement(package='torch>=2.2.0', import_name='torch'),
        PipRequirement(package='transformers>=4.41.0', import_name='transformers'),
        PipRequirement(package='accelerate>=0.28.0', import_name='accelerate'),
        PipRequirement(package='soundfile>=0.12.1', import_name='soundfile'),
        PipRequirement(package='sentencepiece>=0.1.99', import_name='sentencepiece'),
    ],
    'nemo_parakeet_standard': [
        PipRequirement(package='nemo_toolkit[asr]>=1.23.0', import_name='nemo'),
    ],
    'nemo_parakeet_onnx': [
        PipRequirement(package='onnxruntime>=1.16.0', gpu_package='onnxruntime-gpu>=1.16.0', import_name='onnxruntime'),
        PipRequirement(package='huggingface_hub>=0.23.0', import_name='huggingface_hub'),
        PipRequirement(package='soundfile>=0.12.1', import_name='soundfile'),
        PipRequirement(package='librosa>=0.10.0', import_name='librosa'),
        PipRequirement(package='numpy>=1.24.0', import_name='numpy'),
    ],
    'nemo_parakeet_mlx': [
        PipRequirement(package='mlx>=0.9.0', import_name='mlx', platforms={'darwin'}),
        PipRequirement(package='mlx-lm>=0.1.0', import_name='mlx_lm', platforms={'darwin'}),
    ],
    'nemo_canary': [
        PipRequirement(package='nemo_toolkit[asr]>=1.23.0', import_name='nemo'),
    ],
}

TTS_DEPENDENCIES: dict[str, list[PipRequirement]] = {
    'kokoro': [
        # PyTorch + ONNX support for Kokoro:
        # - `kokoro` provides KModel/KPipeline (PyTorch backend)
        # - `kokoro-onnx` + onnxruntime enable the ONNX backend
        PipRequirement(package='kokoro>=0.1.0', import_name='kokoro'),
        PipRequirement(package='kokoro-onnx>=0.3.0', import_name='kokoro_onnx'),
        PipRequirement(package='onnxruntime>=1.16.0', gpu_package='onnxruntime-gpu>=1.16.0', import_name='onnxruntime'),
        PipRequirement(package='phonemizer>=3.2.1', import_name='phonemizer'),
        PipRequirement(package='espeak-phonemizer>=1.0.1', import_name='espeak_phonemizer'),
        PipRequirement(package='huggingface_hub>=0.23.0', import_name='huggingface_hub'),
    ],
    'kitten_tts': [
        PipRequirement(package='onnxruntime>=1.16.0', gpu_package='onnxruntime-gpu>=1.16.0', import_name='onnxruntime'),
        PipRequirement(package='phonemizer-fork~=3.3.2'),
        PipRequirement(package='espeakng_loader', import_name='espeakng_loader'),
        PipRequirement(package='huggingface_hub>=0.23.0', import_name='huggingface_hub'),
    ],
    'dia': [
        PipRequirement(package='torch>=2.2.0', import_name='torch'),
        PipRequirement(package='transformers>=4.41.0', import_name='transformers'),
        PipRequirement(package='accelerate>=0.28.0', import_name='accelerate'),
        PipRequirement(package='safetensors>=0.4.0', import_name='safetensors'),
        PipRequirement(package='sentencepiece>=0.1.99', import_name='sentencepiece'),
        PipRequirement(package='soundfile>=0.12.1', import_name='soundfile'),
        PipRequirement(package='huggingface_hub>=0.23.0', import_name='huggingface_hub'),
    ],
    'higgs': [
        PipRequirement(package='git+https://github.com/boson-ai/higgs-audio.git', import_name='boson_multimodal'),
        PipRequirement(package='torch>=2.2.0', import_name='torch'),
        PipRequirement(package='torchaudio>=2.2.0', import_name='torchaudio'),
        PipRequirement(package='soundfile>=0.12.1', import_name='soundfile'),
        PipRequirement(package='huggingface_hub>=0.23.0', import_name='huggingface_hub'),
    ],
    'vibevoice': [
        PipRequirement(package='git+https://github.com/vibevoice-community/VibeVoice.git', import_name='vibevoice'),
        PipRequirement(package='torch>=2.2.0', import_name='torch'),
        PipRequirement(package='torchaudio>=2.2.0', import_name='torchaudio'),
        PipRequirement(package='sentencepiece>=0.1.99', import_name='sentencepiece'),
        PipRequirement(package='soundfile>=0.12.1', import_name='soundfile'),
        PipRequirement(package='huggingface_hub>=0.23.0', import_name='huggingface_hub'),
    ],
    'pocket_tts': [
        PipRequirement(package='onnxruntime>=1.16.0', gpu_package='onnxruntime-gpu>=1.16.0', import_name='onnxruntime'),
        PipRequirement(package='soundfile>=0.12.1', import_name='soundfile'),
        PipRequirement(package='sentencepiece>=0.1.99', import_name='sentencepiece'),
        PipRequirement(package='scipy>=1.10.0', import_name='scipy'),
        PipRequirement(package='huggingface_hub>=0.21.0', import_name='huggingface_hub'),
    ],
    'neutts': [
        PipRequirement(package='neucodec>=0.0.4', import_name='neucodec'),
        PipRequirement(package='librosa>=0.10.0', import_name='librosa'),
        PipRequirement(package='phonemizer>=3.2.1', import_name='phonemizer'),
        PipRequirement(package='transformers>=4.41.0', import_name='transformers'),
        PipRequirement(package='huggingface_hub>=0.23.0', import_name='huggingface_hub'),
        PipRequirement(package='soundfile>=0.12.1', import_name='soundfile'),
    ],
    'echo_tts': [
        PipRequirement(package='torch>=2.2.0', import_name='torch'),
        PipRequirement(package='torchaudio>=2.2.0', import_name='torchaudio'),
        PipRequirement(package='torchcodec>=0.8.1', import_name='torchcodec'),
        PipRequirement(package='huggingface_hub>=0.23.0', import_name='huggingface_hub'),
        PipRequirement(package='safetensors>=0.4.2', import_name='safetensors'),
        PipRequirement(package='einops>=0.8.0', import_name='einops'),
    ],
    'omnivoice': [],
}

EMBEDDING_DEPENDENCIES: dict[str, list[PipRequirement]] = {
    'huggingface': [
        PipRequirement(package='sentence-transformers>=2.6.0', import_name='sentence_transformers'),
        PipRequirement(package='torch>=2.2.0', import_name='torch'),
    ],
    'custom': [
        PipRequirement(package='sentence-transformers>=2.6.0', import_name='sentence_transformers'),
        PipRequirement(package='torch>=2.2.0', import_name='torch'),
    ],
    'onnx': [
        PipRequirement(package='onnxruntime>=1.16.0', gpu_package='onnxruntime-gpu>=1.16.0', import_name='onnxruntime'),
        PipRequirement(package='numpy>=1.24.0', import_name='numpy'),
    ],
}

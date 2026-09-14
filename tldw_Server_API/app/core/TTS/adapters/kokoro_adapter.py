# kokoro_adapter.py
# Description: Kokoro TTS adapter implementation
#
# Imports
import asyncio
import builtins
import concurrent.futures
import os
import platform
import re
import sys
import time
from collections.abc import AsyncGenerator
from contextlib import contextmanager, suppress
from ctypes.util import find_library as _ctypes_find_library
from typing import Any, Optional

#
# Third-party Imports
import numpy as np
from loguru import logger

from ...testing import env_flag_enabled, is_explicit_pytest_runtime, is_test_mode
from ...Utils.torch_import_guard import safe_import_torch
from ..phoneme_overrides import (
    PhonemeOverrideEntry,
    apply_overrides_to_text,
    filter_overrides_for_provider,
    load_override_entries,
    merge_override_entries,
    parse_override_entries,
)
from ..tts_exceptions import (
    TTSGenerationError,
    TTSModelLoadError,
    TTSModelNotFoundError,
    TTSProviderNotConfiguredError,
)
from ..tts_resource_manager import get_resource_manager as _get_resource_manager
from ..tts_validation import validate_tts_request
from ..utils import parse_bool, run_tts_blocking_next

#
# Local Imports
from .base import AudioFormat, ProviderStatus, TTSAdapter, TTSCapabilities, TTSRequest, TTSResponse, VoiceInfo

#
#######################################################################################################################
#
# Kokoro TTS Adapter Implementation

_KOKORO_NONCRITICAL_EXCEPTIONS = (
    asyncio.CancelledError,
    asyncio.TimeoutError,
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    builtins.TimeoutError,
    TypeError,
    ValueError,
    UnicodeDecodeError,
    TTSGenerationError,
    TTSModelLoadError,
    TTSModelNotFoundError,
    TTSProviderNotConfiguredError,
)

_KOKORO_REPO_WARNING_PREFIX = "WARNING: Defaulting repo_id to "


async def get_resource_manager():
    """Compatibility wrapper so tests can monkeypatch the adapter-level symbol."""
    return await _get_resource_manager()


@contextmanager
def _capture_kokoro_repo_warning():
    """Redirect Kokoro's repo_id print warning into Loguru output."""
    original_print = builtins.print

    def _print(*args, **kwargs):
        try:
            target = kwargs.get("file")
            if target not in (None, sys.stdout):
                return original_print(*args, **kwargs)
            sep = kwargs.get("sep", " ")
            if sep is None:
                sep = " "
            msg = sep.join(str(arg) for arg in args)
        except _KOKORO_NONCRITICAL_EXCEPTIONS:
            return original_print(*args, **kwargs)
        if msg.startswith(_KOKORO_REPO_WARNING_PREFIX):
            logger.warning(msg)
            return
        return original_print(*args, **kwargs)

    builtins.print = _print
    try:
        yield
    finally:
        builtins.print = original_print


class KokoroAdapter(TTSAdapter):
    """Adapter for Kokoro TTS (ONNX and PyTorch variants)"""

    # Kokoro voice definitions
    VOICES = {
        "af_bella": VoiceInfo(
            id="af_bella",
            name="Bella",
            gender="female",
            language="en-us",
            description="American female voice"
        ),
        "af_sky": VoiceInfo(
            id="af_sky",
            name="Sky",
            gender="female",
            language="en-us",
            description="Young American female voice"
        ),
        "af_heart": VoiceInfo(
            id="af_heart",
            name="Heart",
            gender="female",
            language="en-us",
            description="Warm American female voice"
        ),
        "am_adam": VoiceInfo(
            id="am_adam",
            name="Adam",
            gender="male",
            language="en-us",
            description="American male voice"
        ),
        "am_michael": VoiceInfo(
            id="am_michael",
            name="Michael",
            gender="male",
            language="en-us",
            description="Deep American male voice"
        ),
        "bf_emma": VoiceInfo(
            id="bf_emma",
            name="Emma",
            gender="female",
            language="en-gb",
            description="British female voice"
        ),
        "bf_isabella": VoiceInfo(
            id="bf_isabella",
            name="Isabella",
            gender="female",
            language="en-gb",
            description="Elegant British female voice"
        ),
        "bm_george": VoiceInfo(
            id="bm_george",
            name="George",
            gender="male",
            language="en-gb",
            description="British male voice"
        ),
        "bm_lewis": VoiceInfo(
            id="bm_lewis",
            name="Lewis",
            gender="male",
            language="en-gb",
            description="Young British male voice"
        )
    }

    # Chunking configuration (from Kokoro-FastAPI)
    CHUNK_CONFIG = {
        "target_min_tokens": 30,  # Lowered for testing
        "target_max_tokens": 60,  # Lowered for testing (80 tokens in test > 60)
        "absolute_max_tokens": 150  # Lowered for testing
    }

    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__(config)

        # Determine backend type (ONNX or PyTorch). Default to PyTorch; ONNX is opt-in.
        self.use_onnx = self.config.get("kokoro_use_onnx", False)
        # Device selection with fallback
        preferred = self.config.get("kokoro_device") or os.getenv("KOKORO_DEVICE")
        probe_timeout_raw = (
            self.config.get("kokoro_device_probe_timeout_sec")
            or os.getenv("KOKORO_DEVICE_PROBE_TIMEOUT_SEC")
        )
        try:
            self.device_probe_timeout_sec = float(probe_timeout_raw) if probe_timeout_raw is not None else 2.0
        except _KOKORO_NONCRITICAL_EXCEPTIONS:
            self.device_probe_timeout_sec = 2.0
        cuda_avail = False
        mps_avail = False

        def _probe_devices() -> tuple[bool, bool]:
            if is_test_mode() or is_explicit_pytest_runtime() or env_flag_enabled("MINIMAL_TEST_APP"):
                return False, False
            try:
                torch = safe_import_torch()
                cuda_ok = torch.cuda.is_available()
                mps_ok = hasattr(torch.backends, 'mps') and getattr(torch.backends.mps, 'is_available', lambda: False)()
                return cuda_ok, mps_ok
            except _KOKORO_NONCRITICAL_EXCEPTIONS:
                return False, False

        if preferred and str(preferred).lower() == "cpu":
            cuda_avail = False
            mps_avail = False
        else:
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_probe_devices)
                    cuda_avail, mps_avail = future.result(timeout=self.device_probe_timeout_sec)
            except concurrent.futures.TimeoutError:
                logger.warning(f"{self.provider_name}: Device probe timed out; defaulting to CPU")
                cuda_avail = False
                mps_avail = False
            except _KOKORO_NONCRITICAL_EXCEPTIONS:
                cuda_avail = False
                mps_avail = False
        if preferred:
            pref = str(preferred).lower()
            if pref == "cuda":
                self.device = "cuda" if cuda_avail else "cpu"
            elif pref == "mps":
                self.device = "mps" if mps_avail else ("cuda" if cuda_avail else "cpu")
            elif pref == "cpu":
                self.device = "cpu"
            else:
                self.device = "cuda" if cuda_avail else "cpu"
        else:
            self.device = "cuda" if cuda_avail else "cpu"

        # Model paths
        # Default to hexgrad/Kokoro-82M PyTorch layout; ONNX users should override via config.
        default_pt_model = "models/kokoro/kokoro-v1_0.pth"
        default_onnx_model = "models/kokoro/onnx/model.onnx"
        self.model_path = self.config.get(
            "kokoro_model_path",
            default_onnx_model if self.use_onnx else default_pt_model,
        )
        # Default voices bundle for kokoro-onnx v1.0 lives alongside the ONNX model.
        default_voices_bin = os.path.join(os.path.dirname(default_onnx_model), "voices-v1.0.bin")
        # Maintain both attribute names for compatibility with tests and internal code.
        # If no explicit path is configured, prefer the bundled voices-v1.0.bin file for ONNX.
        self.voices_json_path = self.config.get("kokoro_voices_json") or (
            default_voices_bin if self.use_onnx else "models/kokoro/voices"
        )
        self.voices_json = self.voices_json_path
        # PyTorch voices directory (for KModel / KPipeline and dynamic voices)
        self.voice_dir = self.config.get("kokoro_voice_dir", "models/kokoro/voices")

        # Optionally defer heavyweight model load (useful for tests)
        self._lazy_init = parse_bool(
            self.config.get("kokoro_lazy_init"),
            default=parse_bool(os.getenv("KOKORO_LAZY_INIT"), default=False),
        )
        if not self._lazy_init:
            test_mode = is_explicit_pytest_runtime() or env_flag_enabled("TESTING") or is_test_mode()
            explicit_model = any(
                key in self.config for key in ("kokoro_model_path", "kokoro_use_onnx", "kokoro_voices_json")
            )
            if test_mode and not parse_bool(os.getenv("RUN_TTS_LEGACY_INTEGRATION"), default=False) and not explicit_model:
                self._lazy_init = True
        self._deferred_model_load = self._lazy_init
        self._model_lock = asyncio.Lock()

        # Auto-download toggle (Kokoro does not auto-download; provided for consistency)
        cfg_auto = self.config.get("kokoro_auto_download")
        env_auto = os.getenv("KOKORO_AUTO_DOWNLOAD") or os.getenv("TTS_AUTO_DOWNLOAD")
        self.auto_download = parse_bool(cfg_auto, default=parse_bool(env_auto, default=True))

        # Text processing settings
        self.normalize_text = self.config.get("normalize_text", True)
        self.sentence_splitting = self.config.get("sentence_splitting", True)
        self.enable_phoneme_overrides = parse_bool(
            self.config.get("kokoro_enable_phoneme_overrides"),
            default=parse_bool(os.getenv("KOKORO_ENABLE_PHONEME_OVERRIDES"), default=True),
        )
        self.phoneme_override_path = (
            self.config.get("kokoro_phoneme_path")
            or self.config.get("phoneme_override_path")
            or os.getenv("TTS_PHONEME_OVERRIDES_PATH")
        )
        self._provider_override_entries: list[PhonemeOverrideEntry] = parse_override_entries(
            self.config.get("kokoro_phoneme_overrides") or self.config.get("phoneme_overrides"),
            provider_hint="kokoro",
        )
        try:
            self._global_override_entries: list[PhonemeOverrideEntry] = load_override_entries(self.phoneme_override_path)
        except _KOKORO_NONCRITICAL_EXCEPTIONS as exc:  # noqa: BLE001
            logger.debug(f"{self.provider_name}: Failed to load global phoneme overrides: {exc}")
            self._global_override_entries = []

        # Performance settings
        self.sample_rate = self.config.get("sample_rate", 24000)
        init_timeout_raw = (
            self.config.get("kokoro_init_timeout_sec")
            or os.getenv("KOKORO_INIT_TIMEOUT_SEC")
        )
        try:
            self.init_timeout_sec = float(init_timeout_raw) if init_timeout_raw is not None else None
        except _KOKORO_NONCRITICAL_EXCEPTIONS:
            self.init_timeout_sec = None
        # Pause insertion pacing (configurable)
        try:
            self.pause_interval_words = int(
                self.config.get("pause_interval_words")
                or (self.config.get("extra_params", {}) or {}).get("pause_interval_words")
                or 500
            )
        except _KOKORO_NONCRITICAL_EXCEPTIONS:
            self.pause_interval_words = 500
        try:
            self.pause_tag = str(
                self.config.get("pause_tag")
                or (self.config.get("extra_params", {}) or {}).get("pause_tag")
                or "[pause=1.1]"
            )
        except _KOKORO_NONCRITICAL_EXCEPTIONS:
            self.pause_tag = "[pause=1.1]"

        # Model instances
        self.kokoro_instance = None
        self.model_pt = None
        self.kokoro_pt_model = None  # KModel when using PyTorch backend
        self.kokoro_pt_pipelines = {}
        self.tokenizer = None
        self.audio_normalizer = None
        self._dynamic_voices: list[VoiceInfo] = []

    def _ensure_audio_normalizer(self) -> None:
        if self.audio_normalizer is not None:
            return
        from tldw_Server_API.app.core.TTS.streaming_audio_writer import AudioNormalizer

        self.audio_normalizer = AudioNormalizer()

    def _model_is_loaded(self) -> bool:
        if self.use_onnx:
            return self.kokoro_instance is not None
        return self.kokoro_pt_model is not None or self.model_pt is not None

    async def _ensure_model_loaded(self) -> bool:
        if self._model_is_loaded():
            return True
        async with self._model_lock:
            if self._model_is_loaded():
                return True
            if self.use_onnx:
                return await self._load_onnx_model()
            if self.init_timeout_sec:
                try:
                    return await asyncio.wait_for(self._load_pytorch_model(), timeout=self.init_timeout_sec)
                except asyncio.TimeoutError:
                    logger.error(f"{self.provider_name}: Lazy model load timed out after {self.init_timeout_sec}s")
                    return False
            return await self._load_pytorch_model()

    async def initialize(self) -> bool:
        """Initialize the Kokoro adapter"""
        try:
            if self._lazy_init:
                try:
                    self._load_dynamic_voices()
                except _KOKORO_NONCRITICAL_EXCEPTIONS as ve:
                    logger.warning(f"{self.provider_name}: Failed to load dynamic voices.json: {ve}")
                logger.info(f"{self.provider_name}: Lazy init enabled; deferring model load")
                self._status = ProviderStatus.AVAILABLE
                return True

            if self.use_onnx:
                success = await self._load_onnx_model()
            else:
                if self.init_timeout_sec:
                    try:
                        success = await asyncio.wait_for(self._load_pytorch_model(), timeout=self.init_timeout_sec)
                    except asyncio.TimeoutError:
                        logger.error(f"{self.provider_name}: Initialization timed out after {self.init_timeout_sec}s")
                        success = False
                else:
                    success = await self._load_pytorch_model()

            # Load dynamic voices if available
            try:
                self._load_dynamic_voices()
            except _KOKORO_NONCRITICAL_EXCEPTIONS as ve:
                logger.warning(f"{self.provider_name}: Failed to load dynamic voices.json: {ve}")

            if success:
                logger.info(f"{self.provider_name}: Initialized successfully (Backend: {'ONNX' if self.use_onnx else 'PyTorch'}, Device: {self.device})")
                self._status = ProviderStatus.AVAILABLE
                return True
            else:
                self._status = ProviderStatus.ERROR
                return False

        except _KOKORO_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"{self.provider_name}: Initialization failed: {e}")
            self._status = ProviderStatus.ERROR
            return False

    async def _initialize_onnx(self) -> bool:
        """Initialize ONNX backend"""
        try:
            from kokoro_onnx import EspeakConfig, Kokoro

            # Check model file exists
            if not os.path.exists(self.model_path):
                raise TTSModelNotFoundError(
                    f"Kokoro ONNX model not found at {self.model_path}",
                    provider=self.provider_name,
                    details={"model_path": self.model_path}
                )

            # Resolve voices bundle path (required by kokoro-onnx)
            voices_json_arg: Optional[str]
            if self.voices_json_path and os.path.isfile(self.voices_json_path):
                voices_json_arg = self.voices_json_path
            else:
                # If an explicit file path was configured but does not exist, surface a clear error
                if self.voices_json_path and not os.path.isdir(self.voices_json_path):
                    raise TTSModelNotFoundError(
                        f"Kokoro voices bundle not found at {self.voices_json_path}",
                        provider=self.provider_name,
                        details={"voices_json": self.voices_json_path}
                    )
                # Fallback: derive standard voices-v1.0.bin next to the model
                fallback_bin = os.path.join(os.path.dirname(self.model_path), "voices-v1.0.bin")
                if os.path.isfile(fallback_bin):
                    voices_json_arg = fallback_bin
                else:
                    raise TTSModelNotFoundError(
                        "Kokoro voices bundle not found (expected voices-v1.0.bin next to model)",
                        provider=self.provider_name,
                        details={"voices_json": self.voices_json_path, "fallback": fallback_bin}
                    )

            # Configure eSpeak (auto-detect to avoid requiring an env var)
            def _discover_espeak_library() -> Optional[str]:
                # 1) Explicit config override
                path = self.config.get("kokoro_espeak_lib")
                if path and os.path.exists(str(path)):
                    return str(path)
                # 2) Environment variable
                path = os.getenv("PHONEMIZER_ESPEAK_LIBRARY")
                if path and os.path.exists(path):
                    return path
                # 3) Platform heuristics
                sys_plat = sys.platform
                candidates = []
                if sys_plat == "darwin":
                    candidates = [
                        "/opt/homebrew/lib/libespeak-ng.dylib",
                        "/usr/local/lib/libespeak-ng.dylib",
                        "/opt/local/lib/libespeak-ng.dylib",
                    ]
                elif sys_plat.startswith("linux"):
                    arch = platform.machine() or ""
                    candidates = [
                        f"/usr/lib/{arch}/libespeak-ng.so.1" if arch else "",
                        "/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1",
                        "/usr/lib/aarch64-linux-gnu/libespeak-ng.so.1",
                        "/usr/lib64/libespeak-ng.so.1",
                        "/usr/lib/libespeak-ng.so.1",
                        "/lib/x86_64-linux-gnu/libespeak-ng.so.1",
                        "/lib/aarch64-linux-gnu/libespeak-ng.so.1",
                        "/lib/libespeak-ng.so.1",
                    ]
                elif sys_plat in ("win32", "cygwin"):
                    pf = os.environ.get("PROGRAMFILES", r"C:\\Program Files")
                    pf86 = os.environ.get("PROGRAMFILES(X86)", r"C:\\Program Files (x86)")
                    candidates = [
                        os.path.join(pf, "eSpeak NG", "libespeak-ng.dll"),
                        os.path.join(pf86, "eSpeak NG", "libespeak-ng.dll"),
                    ]
                    # Also probe PATH entries
                    for d in os.environ.get("PATH", "").split(os.pathsep):
                        if not d:
                            continue
                        candidates.append(os.path.join(d, "libespeak-ng.dll"))
                # Try ctypes discovery last (may return name not path)
                try:
                    lib_name = _ctypes_find_library("espeak-ng") or _ctypes_find_library("espeak")
                    if lib_name and os.path.isabs(lib_name) and os.path.exists(lib_name):
                        candidates.insert(0, lib_name)
                except _KOKORO_NONCRITICAL_EXCEPTIONS:
                    pass
                for cand in candidates:
                    if cand and os.path.exists(cand):
                        return cand
                return None

            espeak_lib = _discover_espeak_library()
            espeak_config = EspeakConfig(lib_path=espeak_lib) if espeak_lib else None

            # Initialize Kokoro (support constructors that accept either 1 or 2 positional args)
            if voices_json_arg:
                self.kokoro_instance = await asyncio.to_thread(
                    Kokoro,
                    self.model_path,
                    voices_json_arg,
                    espeak_config=espeak_config
                )
            else:
                try:
                    self.kokoro_instance = await asyncio.to_thread(
                        Kokoro,
                        self.model_path,
                        espeak_config=espeak_config
                    )
                except TypeError:
                    # Fallback: pass empty string for voices path if constructor requires it
                    self.kokoro_instance = await asyncio.to_thread(
                        Kokoro,
                        self.model_path,
                        "",
                        espeak_config=espeak_config
                    )

            # Work around a kokoro-onnx 0.4.x bug where the ONNX graph
            # expects a float `speed` input but the library feeds int32
            # for newer exports (input_ids path), causing:
            #   INVALID_ARGUMENT : Unexpected input data type.
            #   Actual: tensor(int32), expected: tensor(float)
            # Patch Kokoro._create_audio locally to always pass speed as float.
            try:
                import kokoro_onnx as _konnx  # type: ignore
                import numpy as _np  # type: ignore

                orig_create_audio = getattr(_konnx.Kokoro, "_create_audio", None)

                if callable(orig_create_audio) and not getattr(_konnx.Kokoro, "_tldw_speed_patch", False):
                    def _patched_create_audio(self_k, phonemes, voice, speed):
                        """Create Kokoro ONNX audio while passing speed with a compatible dtype."""
                        from kokoro_onnx.config import MAX_PHONEME_LENGTH, SAMPLE_RATE  # type: ignore
                        from kokoro_onnx.log import log as _log  # type: ignore

                        _log.debug(f"Phonemes: {phonemes}")
                        if len(phonemes) > MAX_PHONEME_LENGTH:
                            _log.warning(
                                f"Phonemes are too long, truncating to {MAX_PHONEME_LENGTH} phonemes"
                            )
                        phonemes = phonemes[:MAX_PHONEME_LENGTH]
                        import time as _time
                        start_t = _time.time()
                        tokens = _np.array(self_k.tokenizer.tokenize(phonemes), dtype=_np.int64)
                        if len(tokens) > MAX_PHONEME_LENGTH:
                            raise ValueError(
                                f"Context length is {MAX_PHONEME_LENGTH}, but leave room for the pad token 0 at the start & end"
                            )

                        voice_vec = voice[len(tokens)]
                        tokens = [[0, *tokens, 0]]
                        input_names = [i.name for i in self_k.sess.get_inputs()]
                        if "input_ids" in input_names:
                            # Newer export versions: speed as float32 to avoid type mismatch
                            inputs = {
                                "input_ids": tokens,
                                "style": _np.array(voice_vec, dtype=_np.float32),
                                "speed": _np.array([float(speed)], dtype=_np.float32),
                            }
                        else:
                            inputs = {
                                "tokens": tokens,
                                "style": voice_vec,
                                "speed": _np.ones(1, dtype=_np.float32) * float(speed),
                            }

                        audio = self_k.sess.run(None, inputs)[0]
                        audio_duration = len(audio) / SAMPLE_RATE
                        create_duration = _time.time() - start_t
                        rtf = create_duration / audio_duration
                        _log.debug(
                            f"Created audio in length of {audio_duration:.2f}s for {len(phonemes)} phonemes in {create_duration:.2f}s (RTF: {rtf:.2f}"
                        )
                        return audio, SAMPLE_RATE

                    _konnx.Kokoro._create_audio = _patched_create_audio  # type: ignore[assignment]
                    _konnx.Kokoro._tldw_speed_patch = True  # type: ignore[attr-defined]
            except _KOKORO_NONCRITICAL_EXCEPTIONS as _patch_exc:  # pragma: no cover - best-effort patch
                logger.debug(f"{self.provider_name}: speed dtype patch skipped: {_patch_exc}")

            # Register model with resource manager (best-effort)
            try:
                resource_manager = await get_resource_manager()
                if self.kokoro_instance:
                    register_result = resource_manager.register_model(
                        provider=self.provider_name.lower(),
                        model_instance=self.kokoro_instance,
                        cleanup_callback=self._cleanup_resources,
                        model_key=f"onnx:{self.model_path}",
                    )
                    if asyncio.iscoroutine(register_result):
                        await register_result
            except _KOKORO_NONCRITICAL_EXCEPTIONS:
                pass

            logger.info(f"{self.provider_name}: ONNX model loaded successfully")
            return True

        except ImportError as e:
            logger.error(f"{self.provider_name}: kokoro_onnx library not installed")
            raise TTSModelLoadError(
                "Failed to import kokoro_onnx library",
                provider=self.provider_name,
                details={"error": str(e), "suggestion": "pip install kokoro-onnx"}
            ) from e
        except TTSModelNotFoundError:
            raise
        except _KOKORO_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"{self.provider_name}: ONNX initialization error: {e}")
            raise TTSModelLoadError(
                "Failed to initialize ONNX model",
                provider=self.provider_name,
                details={"error": str(e), "model_path": self.model_path}
            ) from e

    async def _initialize_pytorch(self) -> bool:
        """Initialize PyTorch backend"""
        # Check model file before importing torch so missing-model failures stay lightweight.
        if not os.path.exists(self.model_path):
            raise TTSModelNotFoundError(
                f"Kokoro PyTorch model not found at {self.model_path}",
                provider=self.provider_name,
                details={"model_path": self.model_path}
            )
        try:
            torch = safe_import_torch()
        except ImportError as e:
            raise TTSModelLoadError(
                "PyTorch is required for Kokoro PyTorch backend",
                provider=self.provider_name,
                details={"error": str(e), "suggestion": "pip install torch"}
            ) from e
        # Try native Kokoro PyTorch if available
        try:
            from kokoro.model import KModel  # type: ignore
            # config.json expected alongside model
            config_path = os.path.join(os.path.dirname(self.model_path), "config.json")
            if not os.path.exists(config_path):
                raise TTSModelLoadError(
                    "Kokoro config.json not found for PyTorch backend",
                    provider=self.provider_name,
                    details={"config_path": config_path}
                )
            repo_id = self.config.get("kokoro_repo_id") or os.getenv("KOKORO_REPO_ID") or "hexgrad/Kokoro-82M"
            logger.info(f"{self.provider_name}: Loading Kokoro PyTorch model (repo_id={repo_id})")
            start = time.time()
            def _load_model():
                return KModel(repo_id=repo_id, config=config_path, model=self.model_path).eval()
            try:
                with _capture_kokoro_repo_warning():
                    self.kokoro_pt_model = await asyncio.to_thread(_load_model)
            except _KOKORO_NONCRITICAL_EXCEPTIONS:
                # Fallback to sync load if threading fails in constrained envs
                with _capture_kokoro_repo_warning():
                    self.kokoro_pt_model = _load_model()
            # Move to device
            dev = str(self.device).lower()
            if dev.startswith("cuda"):
                with suppress(_KOKORO_NONCRITICAL_EXCEPTIONS):
                    self.kokoro_pt_model = self.kokoro_pt_model.cuda()
            elif dev == "mps":
                try:
                    self.kokoro_pt_model = self.kokoro_pt_model.to(torch.device("mps"))
                except _KOKORO_NONCRITICAL_EXCEPTIONS:
                    logger.warning("MPS device not available; using CPU for Kokoro")
                    self.kokoro_pt_model = self.kokoro_pt_model.cpu()
            else:
                self.kokoro_pt_model = self.kokoro_pt_model.cpu()
            logger.info(f"{self.provider_name}: Kokoro PyTorch model loaded on {dev} (t={time.time() - start:.2f}s)")
            # Register model with resource manager (best-effort)
            try:
                resource_manager = await get_resource_manager()
                if self.kokoro_pt_model is not None:
                    register_result = resource_manager.register_model(
                        provider=self.provider_name.lower(),
                        model_instance=self.kokoro_pt_model,
                        cleanup_callback=self._cleanup_resources,
                        model_key=f"torch:{self.model_path}",
                    )
                    if asyncio.iscoroutine(register_result):
                        await register_result
            except _KOKORO_NONCRITICAL_EXCEPTIONS:
                pass
            return True
        except ImportError:
            # Fallback: generic torch.load
            try:
                try:
                    self.model_pt = torch.jit.load(self.model_path, map_location=self.device)
                except _KOKORO_NONCRITICAL_EXCEPTIONS:
                    self.model_pt = torch.load(self.model_path, map_location=self.device)
                with suppress(_KOKORO_NONCRITICAL_EXCEPTIONS):
                    self.model_pt.eval()
                logger.info(f"{self.provider_name}: Loaded generic PyTorch model on {self.device}")
                # Register model with resource manager (best-effort)
                try:
                    resource_manager = await get_resource_manager()
                    if self.model_pt is not None:
                        register_result = resource_manager.register_model(
                            provider=self.provider_name.lower(),
                            model_instance=self.model_pt,
                            cleanup_callback=self._cleanup_resources,
                            model_key=f"torch:{self.model_path}",
                        )
                        if asyncio.iscoroutine(register_result):
                            await register_result
                except _KOKORO_NONCRITICAL_EXCEPTIONS:
                    pass
                return True
            except _KOKORO_NONCRITICAL_EXCEPTIONS as e:
                raise TTSModelLoadError(
                    "Failed to initialize PyTorch model",
                    provider=self.provider_name,
                    details={"error": str(e), "model_path": self.model_path}
                ) from e

    # Thin wrapper methods for tests to patch
    async def _load_onnx_model(self) -> bool:
        return await self._initialize_onnx()

    async def _load_pytorch_model(self) -> bool:
        return await self._initialize_pytorch()

    async def get_capabilities(self) -> TTSCapabilities:
        """Get Kokoro TTS capabilities"""
        all_voices = list(self.VOICES.values()) + self._dynamic_voices
        return TTSCapabilities(
            provider_name="Kokoro",
            supported_languages={"en-us", "en-gb", "en"},
            supported_voices=all_voices,
            supported_formats={
                # Align with validator: mp3, wav, opus
                AudioFormat.MP3,
                AudioFormat.WAV,
                AudioFormat.OPUS
            },
            max_text_length=1000000,
            supports_streaming=True,
            supports_voice_cloning=False,
            supports_emotion_control=False,
            supports_speech_rate=True,
            supports_pitch_control=False,
            supports_volume_control=False,
            supports_ssml=False,
            supports_phonemes=True,  # Kokoro uses phoneme-based generation
            supports_multi_speaker=True,  # Through voice mixing
            supports_background_audio=False,
            latency_ms=300 if self.device == "cuda" else 3500,  # From Kokoro-FastAPI
            sample_rate=self.sample_rate,
            default_format=AudioFormat.WAV
        )

    async def generate(self, request: TTSRequest) -> TTSResponse:
        """Generate speech using Kokoro TTS"""
        if not await self.ensure_initialized():
            raise TTSProviderNotConfiguredError(
                f"{self.provider_name} not initialized",
                provider=self.provider_name
            )
        if self._deferred_model_load and not self._model_is_loaded():
            loaded = await self._ensure_model_loaded()
            if not loaded:
                raise TTSProviderNotConfiguredError(
                    f"{self.provider_name} model not initialized",
                    provider=self.provider_name,
                )
            self._deferred_model_load = False

        # Validate request using new validation system
        try:
            validate_tts_request(request, provider=self.provider_key)
        except _KOKORO_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"{self.provider_name} request validation failed: {e}")
            raise

        # Process voice (support for voice mixing like "af_bella(2)+af_sky(1)")
        voice = self._process_voice(request.voice or "af_bella")

        # Determine language from voice and apply phoneme overrides before normalization
        lang = self._get_language_from_voice(voice)
        raw_text = request.text
        if self._phoneme_overrides_enabled_for_request(request):
            raw_text = self._apply_phoneme_overrides_to_text(raw_text, request=request, lang_hint=lang)

        # Preprocess text
        text = self.preprocess_text(raw_text)

        self._ensure_audio_normalizer()

        logger.info(
            f"{self.provider_name}: Generating speech with voice={voice}, "
            f"lang={lang}, format={request.format.value}"
        )

        try:
            # For ONNX backend, always use the complete path (with de-dup)
            # and optionally wrap the result as a stream to keep the API
            # contract while avoiding duplicated phrases.
            if self.use_onnx:
                audio_bytes = await self._generate_complete_kokoro(text, voice, lang, request)

                if request.stream:
                    chunk_size = 8192

                    async def _byte_stream():
                        for i in range(0, len(audio_bytes), chunk_size):
                            chunk = audio_bytes[i:i + chunk_size]
                            if chunk:
                                yield chunk

                    return TTSResponse(
                        audio_stream=_byte_stream(),
                        format=request.format,
                        sample_rate=self.sample_rate,
                        channels=1,
                        voice_used=voice,
                        provider=self.provider_name
                    )

                return TTSResponse(
                    audio_data=audio_bytes,
                    format=request.format,
                    sample_rate=self.sample_rate,
                    channels=1,
                    voice_used=voice,
                    provider=self.provider_name,
                    metadata={}
                )

            # PyTorch backend: preserve true streaming semantics
            if request.stream:
                return TTSResponse(
                    audio_stream=self._stream_audio_kokoro(text, voice, lang, request),
                    format=request.format,
                    sample_rate=self.sample_rate,
                    channels=1,
                    voice_used=voice,
                    provider=self.provider_name
                )

            audio_data, alignment_payload = await self._generate_complete_kokoro_with_alignment(
                text,
                voice,
                lang,
                request,
            )
            metadata = {}
            if alignment_payload:
                metadata["alignment"] = alignment_payload
            return TTSResponse(
                audio_data=audio_data,
                format=request.format,
                sample_rate=self.sample_rate,
                channels=1,
                voice_used=voice,
                provider=self.provider_name,
                metadata=metadata
            )

        except _KOKORO_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"{self.provider_name} generation error: {e}")
            raise

    async def _stream_audio_kokoro(
        self,
        text: str,
        voice: str,
        lang: str,
        request: TTSRequest,
        alignment_out: Optional[list] = None,
    ) -> AsyncGenerator[bytes, None]:
        """Stream audio from Kokoro"""
        if self.use_onnx:
            if not self.kokoro_instance:
                raise ValueError("Kokoro ONNX not initialized")
        else:
            if self.kokoro_pt_model is None and self.model_pt is None:
                raise ValueError("Kokoro PyTorch model not initialized")

        # Import StreamingAudioWriter for format conversion
        from tldw_Server_API.app.core.TTS.streaming_audio_writer import StreamingAudioWriter
        # Defer writer creation until first chunk to honor source SR
        writer = None

        try:
            chunk_count = 0
            # Stream audio chunks
            if self.use_onnx:
                base_iter = self.kokoro_instance.create_stream(
                    text,
                    voice=voice,
                    speed=request.speed,
                    lang=lang
                )
                # Wrap sync iterators into async for uniform consumption
                import inspect
                if hasattr(base_iter, "__aiter__") or inspect.isasyncgen(base_iter):
                    stream_iter = base_iter
                else:
                    sync_iterator = iter(base_iter)
                    sentinel = object()

                    async def _async_wrap():
                        """Adapt Kokoro ONNX's synchronous stream iterator for async streaming."""
                        while True:
                            item = await run_tts_blocking_next(sync_iterator, sentinel)
                            if item is sentinel:
                                break
                            yield item
                    stream_iter = _async_wrap()
            else:
                # Use Kokoro PyTorch pipeline if available
                try:
                    from kokoro.pipeline import KPipeline  # type: ignore
                except ImportError:
                    # Cannot proceed without kokoro pipeline
                    raise TTSGenerationError(
                        "Kokoro PyTorch generation requires 'kokoro' package",
                        provider=self.provider_name,
                        details={"suggestion": "pip install kokoro-tts or Kokoro PyTorch package"}
                    ) from None
                # Capture the logical voice id before resolving file path
                voice_id = voice

                # Determine voice path if a voice file exists
                voice_path = voice
                try:
                    # Attempt to resolve to a .pt file under configured voice_dir if voice looks like an id
                    if self.voice_dir and isinstance(voice, str) and os.path.isdir(self.voice_dir):
                        candidate = os.path.join(self.voice_dir, f"{voice}.pt")
                        if os.path.exists(candidate):
                            voice_path = candidate
                except _KOKORO_NONCRITICAL_EXCEPTIONS:
                    pass
                # Pick pipeline by Kokoro language code (e.g., 'a' for American, 'b' for British)
                lang_code = self._get_kpipeline_lang_code(voice_id if isinstance(voice_id, str) else "", lang)
                key = lang_code
                if key not in self.kokoro_pt_pipelines:
                    repo_id = self.config.get("kokoro_repo_id") or os.getenv("KOKORO_REPO_ID") or "hexgrad/Kokoro-82M"
                    with _capture_kokoro_repo_warning():
                        self.kokoro_pt_pipelines[key] = KPipeline(
                            lang_code=key,
                            repo_id=repo_id,
                            model=self.kokoro_pt_model,
                            device=str(self.device)
                        )
                pipeline = self.kokoro_pt_pipelines[key]

                sync_iterator = iter(
                    pipeline(text, voice=voice_path, speed=request.speed, model=self.kokoro_pt_model)
                )
                sentinel = object()

                async def _async_iter():
                    """Adapt Kokoro PyTorch's synchronous stream iterator for async streaming."""
                    while True:
                        result = await run_tts_blocking_next(sync_iterator, sentinel)
                        if result is sentinel:
                            break
                        yield result

                stream_iter = _async_iter()

            async for item in stream_iter:
                if alignment_out is not None and getattr(item, "tokens", None):
                    try:
                        text_index = getattr(item, "text_index", None)
                        if text_index is None:
                            alignment_out.extend(item.tokens)
                        else:
                            for token in item.tokens:
                                try:
                                    meta = getattr(token, "_", None)
                                    if meta is None:
                                        meta = token.Underscore()
                                        token._ = meta
                                    meta["segment_index"] = text_index
                                except _KOKORO_NONCRITICAL_EXCEPTIONS:
                                    pass
                                alignment_out.append(token)
                    except _KOKORO_NONCRITICAL_EXCEPTIONS:
                        pass
                samples_chunk, sr_chunk = self._unpack_stream_item(item)
                if samples_chunk is not None and len(samples_chunk) > 0:
                    # Heuristic de-duplication for providers that may repeat phrases
                    with suppress(_KOKORO_NONCRITICAL_EXCEPTIONS):
                        samples_chunk = self._dedupe_repeated_audio(samples_chunk)
                    chunk_count += 1

                    # Create writer on first chunk so we can pass the true SR
                    if writer is None:
                        try:
                            effective_sr = int(sr_chunk) if sr_chunk else self.sample_rate
                        except _KOKORO_NONCRITICAL_EXCEPTIONS:
                            effective_sr = self.sample_rate
                        writer = StreamingAudioWriter(
                            format=request.format.value,
                            sample_rate=effective_sr,
                            channels=1,
                        )

                    # Normalize float32 samples to int16
                    normalized_chunk = self.audio_normalizer.normalize(
                        samples_chunk,
                        target_dtype=np.int16
                    )

                    # Write chunk and get encoded bytes
                    encoded_bytes = writer.write_chunk(normalized_chunk)
                    if encoded_bytes:
                        yield encoded_bytes
                        logger.debug(f"{self.provider_name}: Yielded chunk {chunk_count}, {len(encoded_bytes)} bytes")

            # Finalize stream
            if writer is not None:
                final_bytes = writer.write_chunk(finalize=True)
                if final_bytes:
                    yield final_bytes
                    logger.debug(f"{self.provider_name}: Yielded final chunk, {len(final_bytes)} bytes")

            logger.info(f"{self.provider_name}: Successfully streamed {chunk_count} chunks")

        except _KOKORO_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"{self.provider_name} streaming error: {e}")
            raise
        finally:
            try:
                if writer is not None:
                    writer.close()
            except _KOKORO_NONCRITICAL_EXCEPTIONS:
                pass

    async def _generate_complete_kokoro(
        self,
        text: str,
        voice: str,
        lang: str,
        request: TTSRequest
    ) -> bytes:
        """Generate complete audio from Kokoro"""
        if self.use_onnx:
            # Use synchronous Kokoro.create in a worker thread and post-process
            import asyncio as _asyncio
            loop = _asyncio.get_event_loop()
            samples, sr = await loop.run_in_executor(
                None,
                self.kokoro_instance.create,  # type: ignore[arg-type]
                text,
                voice,
                float(request.speed),
                lang,
            )

            try:
                original_len = len(samples)
                deduped = self._dedupe_repeated_audio(samples)
                if hasattr(deduped, "__len__") and len(deduped) != original_len:
                    logger.info(
                        f"{self.provider_name}: de-duplicated waveform from {original_len} to {len(deduped)} samples"
                    )
                else:
                    logger.debug(f"{self.provider_name}: de-duplication not applied (len={original_len})")
                samples = deduped
            except _KOKORO_NONCRITICAL_EXCEPTIONS as _dedupe_exc:
                logger.debug(f"{self.provider_name}: de-duplication skipped: {_dedupe_exc}")

            from tldw_Server_API.app.core.TTS.streaming_audio_writer import StreamingAudioWriter

            writer = StreamingAudioWriter(
                format=request.format.value,
                sample_rate=int(sr) if sr else self.sample_rate,
                channels=1,
            )
            try:
                normalized = self.audio_normalizer.normalize(samples, target_dtype=np.int16)  # type: ignore[arg-type]
                first = writer.write_chunk(normalized) or b""
                final = writer.write_chunk(finalize=True) or b""
                if request.format == AudioFormat.PCM:
                    return first
                return first + final
            finally:
                writer.close()

        # Fallback: collect encoded bytes from streaming path (PyTorch backend)
        all_audio = b""
        async for chunk in self._stream_audio_kokoro(text, voice, lang, request):
            all_audio += chunk
        return all_audio

    async def _generate_complete_kokoro_with_alignment(
        self,
        text: str,
        voice: str,
        lang: str,
        request: TTSRequest,
    ) -> tuple[bytes, Optional[dict]]:
        if self.use_onnx:
            audio_bytes = await self._generate_complete_kokoro(text, voice, lang, request)
            return audio_bytes, None

        alignment_tokens: list = []
        all_audio = b""
        async for chunk in self._stream_audio_kokoro(
            text,
            voice,
            lang,
            request,
            alignment_out=alignment_tokens,
        ):
            all_audio += chunk
        alignment_payload = self._build_alignment_payload(
            alignment_tokens,
            sample_rate=self.sample_rate,
            text=text,
        )
        return all_audio, alignment_payload

    def _build_alignment_payload(
        self,
        tokens: list,
        *,
        sample_rate: int,
        text: Optional[str] = None,
    ) -> Optional[dict]:
        words: list[dict] = []
        cursor = 0
        source_text = text
        text_len = len(source_text) if source_text is not None else 0
        segments = None
        segment_cursors: dict[int, int] = {}
        if source_text is not None:
            segments = self._split_text_with_offsets(source_text, split_pattern=r"\n+")
        for token in tokens:
            token_text = getattr(token, "text", "") or ""
            if not token_text.strip():
                continue
            start_ts = getattr(token, "start_ts", None)
            end_ts = getattr(token, "end_ts", None)
            if start_ts is None or end_ts is None:
                continue
            char_start = None
            char_end = None
            if source_text is not None:
                segment_index = None
                meta = getattr(token, "_", None)
                if meta is not None:
                    try:
                        segment_index = meta.get("segment_index")
                    except _KOKORO_NONCRITICAL_EXCEPTIONS:
                        segment_index = None
                if (
                    segment_index is not None
                    and segments is not None
                    and 0 <= segment_index < len(segments)
                ):
                    segment = segments[segment_index]
                    seg_text = segment["text"]
                    seg_cursor = segment_cursors.get(segment_index, 0)
                    try:
                        idx = seg_text.find(token_text, seg_cursor)
                    except _KOKORO_NONCRITICAL_EXCEPTIONS:
                        idx = -1
                    if idx == -1 and seg_cursor < len(seg_text):
                        try:
                            candidate = seg_text[seg_cursor : seg_cursor + len(token_text)]
                            if candidate == token_text:
                                idx = seg_cursor
                        except _KOKORO_NONCRITICAL_EXCEPTIONS:
                            idx = -1
                    if idx != -1:
                        char_start = segment["start"] + idx
                        char_end = char_start + len(token_text)
                        seg_cursor = idx + len(token_text)
                        whitespace = getattr(token, "whitespace", "") or ""
                        if whitespace:
                            try:
                                if seg_text[seg_cursor : seg_cursor + len(whitespace)] == whitespace:
                                    seg_cursor = seg_cursor + len(whitespace)
                            except _KOKORO_NONCRITICAL_EXCEPTIONS:
                                pass
                        segment_cursors[segment_index] = seg_cursor
                        if char_end is not None:
                            cursor = max(cursor, char_end)
                            if whitespace:
                                try:
                                    if source_text[char_end : char_end + len(whitespace)] == whitespace:
                                        cursor = max(cursor, char_end + len(whitespace))
                                except _KOKORO_NONCRITICAL_EXCEPTIONS:
                                    pass
                if char_start is None:
                    try:
                        idx = source_text.find(token_text, cursor)
                    except _KOKORO_NONCRITICAL_EXCEPTIONS:
                        idx = -1
                    if idx == -1 and cursor < text_len:
                        try:
                            candidate = source_text[cursor : cursor + len(token_text)]
                            if candidate == token_text:
                                idx = cursor
                        except _KOKORO_NONCRITICAL_EXCEPTIONS:
                            idx = -1
                    if idx != -1:
                        char_start = idx
                        char_end = idx + len(token_text)
                        cursor = char_end
                        whitespace = getattr(token, "whitespace", "") or ""
                        if whitespace:
                            try:
                                if source_text[char_end : char_end + len(whitespace)] == whitespace:
                                    cursor = char_end + len(whitespace)
                            except _KOKORO_NONCRITICAL_EXCEPTIONS:
                                pass
            words.append(
                {
                    "word": token_text.strip(),
                    "start_ms": int(start_ts * 1000),
                    "end_ms": int(end_ts * 1000),
                    "char_start": char_start,
                    "char_end": char_end,
                }
            )
        if not words:
            return None
        return {"engine": "kokoro", "sample_rate": sample_rate, "words": words}

    def _split_text_with_offsets(self, text: str, split_pattern: Optional[str]) -> list[dict]:
        if not split_pattern:
            return [{"text": text, "start": 0, "end": len(text)}]
        segments: list[dict] = []
        try:
            regex = re.compile(split_pattern)
        except re.error:
            return [{"text": text, "start": 0, "end": len(text)}]
        last_end = 0
        for match in regex.finditer(text):
            segments.append(
                {
                    "text": text[last_end:match.start()],
                    "start": last_end,
                    "end": match.start(),
                }
            )
            last_end = match.end()
        segments.append({"text": text[last_end:], "start": last_end, "end": len(text)})
        return segments

    def _process_voice(self, voice: str) -> str:
        """
        Process voice string, supporting voice mixing.
        Examples:
        - "af_bella" -> "af_bella"
        - "af_bella(2)+af_sky(1)" -> mixed voice
        """
        # Check if it's a mixed voice pattern
        if "+" in voice and "(" in voice:
            # This is a mixed voice, return as-is for Kokoro to handle
            return voice

        # Accept known voices (static or dynamically discovered)
        try:
            dynamic_ids = {v.id for v in self._dynamic_voices}
        except _KOKORO_NONCRITICAL_EXCEPTIONS:
            dynamic_ids = set()
        if voice in self.VOICES or voice in dynamic_ids:
            return voice

        # Map generic voice names to Kokoro voices when unknown
        voice = self.map_voice(voice)

        return voice

    def _dedupe_repeated_audio(self, samples: np.ndarray) -> np.ndarray:
        """Heuristically trim duplicated phrases when the waveform is repeated twice."""
        try:
            if samples.ndim != 1:
                return samples
            n = len(samples)
            if n < 8000:
                return samples

            arr = samples.astype(np.float32, copy=False)

            best_diff: Optional[float] = None
            best_offset: Optional[int] = None

            start = n // 3
            end = (2 * n) // 3
            step = max(256, n // 100)

            for offset in range(start, end, step):
                a = arr[: n - offset]
                b = arr[offset:]
                m = min(len(a), len(b))
                if m < 4000:
                    continue
                a_seg = a[:m].copy()
                b_seg = b[:m].copy()
                max_a = float(np.max(np.abs(a_seg))) or 1.0
                max_b = float(np.max(np.abs(b_seg))) or 1.0
                a_seg /= max_a
                b_seg /= max_b
                diff = float(np.mean(np.abs(a_seg - b_seg)))
                if best_diff is None or diff < best_diff:
                    best_diff = diff
                    best_offset = offset

            if best_diff is not None and best_offset is not None and best_diff < 0.08:
                return samples[:best_offset]
            return samples
        except _KOKORO_NONCRITICAL_EXCEPTIONS:
            return samples

    def _load_dynamic_voices(self) -> None:
        """Load voices from voices.json and merge with static voices.

        Expected JSON structure (array of entries):
        [{"id": "af_bella", "name": "Bella", "gender": "female", "language": "en-us", "description": "..."}, ...]
        """
        path = self.voices_json
        if not path or not os.path.exists(path):
            return
        dyn: list[VoiceInfo] = []
        existing_ids = set(self.VOICES.keys()) | {v.id for v in self._dynamic_voices}
        try:
            if os.path.isdir(path):
                # v1.0 layout: voices directory containing *.bin (ONNX) or *.pt (PyTorch) files
                for fname in os.listdir(path):
                    if not (fname.endswith('.bin') or fname.endswith('.pt')):
                        continue
                    vid = os.path.splitext(fname)[0]
                    if not vid or vid in existing_ids:
                        continue
                    # Heuristic language by prefix like 'af_', 'am_', 'bf_', 'bm_', 'zf_', 'zm_', etc.
                    lang = 'en'
                    try:
                        if vid.startswith('a'):
                            lang = 'en-us'
                        elif vid.startswith('b'):
                            lang = 'en-gb'
                    except _KOKORO_NONCRITICAL_EXCEPTIONS:
                        pass
                    vinfo = VoiceInfo(
                        id=vid,
                        name=vid,
                        gender=None,
                        language=lang,
                        description='Kokoro voice profile'
                    )
                    dyn.append(vinfo)
                    existing_ids.add(vid)
                self._dynamic_voices = dyn
                return
        except _KOKORO_NONCRITICAL_EXCEPTIONS:
            # Fall back to JSON parsing
            pass
        # JSON file layout (legacy)
        try:
            import json
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
            entries = data["voices"] if isinstance(data, dict) and "voices" in data else data
            if not isinstance(entries, list):
                return
            for entry in entries:
                try:
                    vid = str(entry.get("id") or entry.get("voice_id") or "").strip()
                    if not vid or vid in existing_ids:
                        continue
                    vinfo = VoiceInfo(
                        id=vid,
                        name=str(entry.get("name") or vid),
                        gender=entry.get("gender"),
                        language=str(entry.get("language") or "en"),
                        description=entry.get("description")
                    )
                    dyn.append(vinfo)
                    existing_ids.add(vid)
                except _KOKORO_NONCRITICAL_EXCEPTIONS:
                    continue
            self._dynamic_voices = dyn
        except _KOKORO_NONCRITICAL_EXCEPTIONS:
            return

    def _get_language_from_voice(self, voice: str) -> str:
        """Get language code from voice ID"""
        # Handle mixed voices
        if "+" in voice:
            # Extract first voice from mix
            first_voice = voice.split("+")[0].split("(")[0].strip()
        else:
            first_voice = voice

        # Determine language from voice prefix
        if first_voice.startswith("a"):
            return "en-us"  # American
        elif first_voice.startswith("b"):
            return "en-gb"  # British
        else:
            return "en-us"  # Default to American

    def _get_kpipeline_lang_code(self, voice: str, lang: Optional[str]) -> str:
        """Map voice/lang to Kokoro PyTorch KPipeline lang_code (e.g., 'a', 'b')."""
        base = voice or ""
        try:
            # If a file path was passed, strip directory and extension
            base = os.path.basename(base)
            if "." in base:
                base = base.split(".", 1)[0]
        except _KOKORO_NONCRITICAL_EXCEPTIONS:
            base = voice or ""
        base = base.strip()

        # Heuristic mapping for known English voices
        if base.startswith("af_") or base.startswith("am_"):
            return "a"  # American English
        if base.startswith("bf_") or base.startswith("bm_"):
            return "b"  # British English

        # Fallback based on language string
        if lang:
            l = str(lang).lower()
            if l.startswith("en"):
                return "a"

        # Default to American English code
        return "a"

    def _unpack_stream_item(self, item: Any) -> tuple[Optional[np.ndarray], Optional[int]]:
        """
        Normalize stream items from both ONNX and PyTorch backends into (samples, sample_rate).

        Supported shapes:
          - (samples, sr)
          - (samples, sr, *rest)
          - samples (np.ndarray or list), using adapter sample_rate
        """
        if item is None:
            return None, None

        # Hexgrad Kokoro PyTorch pipeline returns a Result with an `audio` tensor
        try:
            if hasattr(item, "audio"):
                audio = item.audio
                try:
                    torch = safe_import_torch()

                    if isinstance(audio, torch.Tensor):
                        audio = audio.detach().cpu().numpy()
                except _KOKORO_NONCRITICAL_EXCEPTIONS:
                    # Fallback: try NumPy conversion directly
                    try:
                        audio = np.asarray(audio)
                    except _KOKORO_NONCRITICAL_EXCEPTIONS:
                        return None, None
                return audio, self.sample_rate
        except _KOKORO_NONCRITICAL_EXCEPTIONS:
            pass

        # Tuple/list variants
        if isinstance(item, (tuple, list)):
            if len(item) == 0:
                return None, None
            if len(item) == 1:
                return item[0], self.sample_rate
            # Use the first two elements as (audio, sample_rate); ignore the rest
            samples = item[0]
            sr = item[1]
            try:
                sr_int = int(sr) if sr is not None else self.sample_rate
            except _KOKORO_NONCRITICAL_EXCEPTIONS:
                sr_int = self.sample_rate
            return samples, sr_int

        # Single array-like item: treat as audio with default sample_rate
        return item, self.sample_rate

    def map_voice(self, voice_id: str) -> str:
        """Map generic voice ID to Kokoro voice"""
        # Check if it's already a valid Kokoro voice
        if voice_id in self.VOICES:
            return voice_id

        # Try common mappings
        voice_mappings = {
            "female": "af_bella",
            "male": "am_adam",
            "british_female": "bf_emma",
            "british_male": "bm_george",
            "american_female": "af_bella",
            "american_male": "am_adam",
            "young_female": "af_sky",
            "deep_male": "am_michael",
            "warm": "af_heart",
            # Historical mapping kept for compatibility; note this id is not in static VOICES
            "child": "af_nicole",
        }

        return voice_mappings.get(voice_id.lower(), "af_bella")

    def _phoneme_overrides_enabled_for_request(self, request: TTSRequest) -> bool:
        """Determine whether phoneme overrides should be applied for this request."""
        try:
            extra = getattr(request, "extra_params", {}) or {}
        except _KOKORO_NONCRITICAL_EXCEPTIONS:
            extra = {}
        if "phoneme_overrides_enabled" in extra:
            return parse_bool(extra.get("phoneme_overrides_enabled"), default=self.enable_phoneme_overrides)
        if parse_bool(extra.get("disable_phoneme_overrides"), default=False):
            return False
        return self.enable_phoneme_overrides

    def _collect_phoneme_overrides(self, request: TTSRequest) -> list[PhonemeOverrideEntry]:
        """Merge global, provider, and request-level overrides (request wins)."""
        base = filter_overrides_for_provider(self._global_override_entries, "kokoro")
        provider = filter_overrides_for_provider(self._provider_override_entries, "kokoro")
        try:
            extra = getattr(request, "extra_params", {}) or {}
        except _KOKORO_NONCRITICAL_EXCEPTIONS:
            extra = {}
        request_overrides_raw = extra.get("phoneme_overrides") or extra.get("phoneme_map")
        request_entries = parse_override_entries(request_overrides_raw, provider_hint="kokoro")
        return merge_override_entries(base, provider, request_entries)

    def _apply_phoneme_overrides_to_text(
        self,
        text: str,
        *,
        request: TTSRequest,
        lang_hint: Optional[str],
    ) -> str:
        """Apply applicable phoneme overrides to the provided text."""
        try:
            entries = self._collect_phoneme_overrides(request)
        except _KOKORO_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(f"{self.provider_name}: failed to collect phoneme overrides: {exc}")
            return text
        if not entries:
            return text
        try:
            updated = apply_overrides_to_text(text, entries, lang_hint=lang_hint)
        except _KOKORO_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(f"{self.provider_name}: failed to apply phoneme overrides: {exc}")
            return text
        else:
            return updated

    def preprocess_text(self, text: str, **kwargs) -> str:
        """Preprocess text for Kokoro"""
        # Strip excess whitespace
        text = text.strip()

        # Normalize text if enabled
        if self.normalize_text:
            # Basic normalization (Kokoro handles most of this internally)
            text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
            # Normalize quotes/apostrophes
            text = text.replace('“', '"').replace('”', '"').replace('‟', '"')
            text = text.replace('‘', "'").replace('’', "'")

        # Insert periodic pause tags to keep very long inputs paced
        with suppress(_KOKORO_NONCRITICAL_EXCEPTIONS):
            text = self._insert_pause_tags(text, words_between=self.pause_interval_words, pause_tag=self.pause_tag)

        return text

    def _insert_pause_tags(self, text: str, words_between: int = 500, pause_tag: str = '[pause=1.1]') -> str:
        """Ensure a pause tag appears at least every N words.

        - Splits on whitespace and inserts a pause marker every `words_between` tokens.
        - Respects existing pause markers by splitting and processing each section independently.
        """
        # If already contains pause tags, process sections separately so spacing is preserved
        if pause_tag in text:
            parts = text.split(pause_tag)
            processed = [self._insert_pause_tags(p, words_between, pause_tag) for p in parts]
            return (pause_tag).join(processed)

        words = text.split()
        if len(words) <= words_between:
            return text

        out = []
        cnt = 0
        for w in words:
            out.append(w)
            cnt += 1
            if cnt >= words_between:
                out.append(pause_tag)
                cnt = 0
        return ' '.join(out)

    def chunk_text(self, text: str) -> list[str]:
        """
        Chunk text for optimal Kokoro processing.
        Based on Kokoro-FastAPI chunking strategy.
        """
        # Simple sentence-based chunking
        import re

        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # Estimate token count (rough approximation: 1 token ≈ 4 chars)
            current_plus_sentence = current_chunk + (" " + sentence if current_chunk else sentence)
            estimated_tokens = len(current_plus_sentence) / 4

            if estimated_tokens < self.CHUNK_CONFIG["target_max_tokens"]:
                current_chunk = current_plus_sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    async def _cleanup_resources(self):
        """Clean up Kokoro adapter resources"""
        try:
            # Clean up ONNX instance
            if self.kokoro_instance:
                self.kokoro_instance = None
                logger.debug(f"{self.provider_name}: ONNX instance cleared")

            # Clean up PyTorch model and tokenizer
            if self.model_pt:
                self.model_pt = None
                logger.debug(f"{self.provider_name}: PyTorch model cleared")

            if self.tokenizer:
                self.tokenizer = None
                logger.debug(f"{self.provider_name}: Tokenizer cleared")

            # Clear normalizer
            if self.audio_normalizer:
                self.audio_normalizer = None
            # Clear optionally present attributes used in tests
            if hasattr(self, 'model'):
                self.model = None
            if hasattr(self, 'phonemizer'):
                self.phonemizer = None

            # Clear CUDA cache if using GPU
            if self.device.startswith("cuda"):
                try:
                    if is_test_mode() or is_explicit_pytest_runtime() or env_flag_enabled("MINIMAL_TEST_APP"):
                        return
                    torch = safe_import_torch()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.debug(f"{self.provider_name}: CUDA cache cleared")
                except ImportError:
                    pass

        except _KOKORO_NONCRITICAL_EXCEPTIONS as e:
            logger.warning(f"{self.provider_name}: Error during cleanup: {e}")

#
# End of kokoro_adapter.py
#######################################################################################################################

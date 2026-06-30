# OmniVoice Sidecar Smoke Helper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a repo-native helper that verifies the managed OmniVoice sidecar path can produce real non-silent WAV audio, and document how operators configure and run it.

**Architecture:** Keep runtime orchestration on the existing `OmniVoiceSidecarSupervisor` and `OmniVoiceAdapter` path so the helper exercises the same managed sidecar integration as the API. Keep pure helpers importable for unit tests: CLI argument parsing, provider config construction, output path resolution, and WAV validation. Do not add another direct OmniVoice runtime path.

**Tech Stack:** Python argparse, asyncio, stdlib `wave`/`audioop`, existing TTS adapter dataclasses, pytest unit tests, Bandit.

**Backlog:** `TASK-488`

---

## Stage 1: Plan And Test Surface
**Goal:** Lock the helper scope and add failing tests for behavior that does not require real OmniVoice dependencies.
**Success Criteria:** Tests fail because `smoke_test_omnivoice_sidecar` does not exist yet or lacks the expected helpers.
**Tests:** `python -m pytest tldw_Server_API/tests/TTS_NEW/unit/test_omnivoice_sidecar_smoke.py -q`
**Status:** Complete

### Task 1: Add Failing Unit Tests

**Files:**
- Create: `tldw_Server_API/tests/TTS_NEW/unit/test_omnivoice_sidecar_smoke.py`
- Create later: `Helper_Scripts/TTS_Installers/smoke_test_omnivoice_sidecar.py`

- [x] **Step 1: Write failing tests for CLI parsing and config construction**

```python
def test_smoke_helper_builds_sidecar_provider_config(tmp_path):
    config = build_sidecar_provider_config(...)
    assert config["extra_params"]["python_path"] == str(sidecar_python)
```

- [x] **Step 2: Write failing tests for WAV validation**

```python
def test_smoke_helper_rejects_silent_wav():
    with pytest.raises(SystemExit, match="silent"):
        validate_wav_audio(silent_wav_bytes)
```

- [x] **Step 3: Run the focused tests to verify RED**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/TTS_NEW/unit/test_omnivoice_sidecar_smoke.py -q`
Expected: FAIL from missing module/helper functions.

---

## Stage 2: Smoke Helper Implementation
**Goal:** Implement the CLI and importable helpers with no direct model imports.
**Success Criteria:** Unit tests pass, `--help` works, and the script validates generated audio before reporting success.
**Tests:** `python -m pytest tldw_Server_API/tests/TTS_NEW/unit/test_omnivoice_sidecar_smoke.py -q`
**Status:** Complete

### Task 2: Implement Helper Module

**Files:**
- Create: `Helper_Scripts/TTS_Installers/smoke_test_omnivoice_sidecar.py`
- Test: `tldw_Server_API/tests/TTS_NEW/unit/test_omnivoice_sidecar_smoke.py`

- [x] **Step 1: Implement argument parsing**

Arguments: `--model-path`, `--sidecar-python`, optional `--repo-root`, `--output`, `--text`, `--port`, `--num-step`, `--speed`, `--timeout`.

- [x] **Step 2: Implement config and request construction**

Build provider config with `runtime: "sidecar"`, model path, interpreter path, runtime/scratch paths, loopback host, selected port, and short health/startup timeouts suitable for a smoke test.

- [x] **Step 3: Implement async synthesis path**

Instantiate `OmniVoiceSidecarSupervisor`, attach it to `OmniVoiceAdapter`, generate a non-streaming WAV `TTSRequest`, write the output file, and always shut the supervisor down.

- [x] **Step 4: Implement WAV validation and summary output**

Require parseable WAV bytes, 24 kHz sample rate, mono channel count, nonzero frames, and non-silent PCM samples.

- [x] **Step 5: Run tests to verify GREEN**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/TTS_NEW/unit/test_omnivoice_sidecar_smoke.py -q`
Expected: PASS.

---

## Stage 3: Documentation And Verification
**Goal:** Make setup and real-runtime verification discoverable to operators.
**Success Criteria:** The setup guide includes model-path examples, the sidecar smoke command, expected output, and common failure notes.
**Tests:** focused tests, helper `--help`, docs grep, Bandit on touched helper.
**Status:** Complete

### Task 3: Update Setup Guide And Verify

**Files:**
- Modify: `Docs/STT-TTS/TTS-SETUP-GUIDE.md`
- Modify: `backlog/tasks/task-487 - Add-OmniVoice-managed-sidecar-smoke-test-helper.md`

- [x] **Step 1: Document OmniVoice setup and verification**

Add model cache examples, sidecar Python guidance, smoke helper command, and notes for `torchaudio`, sandbox loopback restrictions, and the opt-in pytest runtime dependency caveat.

- [x] **Step 2: Run verification**

Run:
```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/TTS_NEW/unit/test_omnivoice_sidecar_smoke.py -q
python -m pytest tldw_Server_API/tests/TTS_NEW/unit/test_omnivoice_installer.py tldw_Server_API/tests/TTS_NEW/unit/test_omnivoice_sidecar_supervisor.py -q
python Helper_Scripts/TTS_Installers/smoke_test_omnivoice_sidecar.py --help
python -m bandit -r Helper_Scripts/TTS_Installers/smoke_test_omnivoice_sidecar.py -f json -o /tmp/bandit_omnivoice_sidecar_smoke_helper.json
python -m bandit -r tldw_Server_API/tests/TTS_NEW/unit/test_omnivoice_sidecar_smoke.py -s B101 -f json -o /tmp/bandit_omnivoice_sidecar_smoke_tests.json
git diff --check
```

Observed:
- `test_omnivoice_sidecar_smoke.py`: 10 passed.
- Adjacent OmniVoice installer/supervisor unit tests: 40 passed.
- Helper `--help`: exit 0.
- Bandit helper report: 0 findings.
- Bandit test report with pytest assertion check skipped (`B101`): 0 findings.
- `git diff --check`: exit 0.
- Real managed sidecar smoke run using the local OmniVoice model snapshot and sidecar venv: exit 0, output `/private/tmp/omnivoice-helper-sidecar-smoke-recheck.wav`, 155084 bytes, 24000 Hz mono, 77520 frames, RMS 2284.84, peak 16384. The underlying runtime still emitted a shutdown `resource_tracker` semaphore warning after successful synthesis.
- After rebasing onto `origin/dev`, repeated the focused test, adjacent OmniVoice tests, helper `--help`, Bandit, `git diff --check`, and real managed sidecar smoke. The post-rebase real smoke wrote `/private/tmp/omnivoice-helper-sidecar-smoke-rebase.wav`, 158924 bytes, 24000 Hz mono, 79440 frames, RMS 2425.56, peak 16384, with the same underlying shutdown `resource_tracker` semaphore warning after successful synthesis.

- [x] **Step 3: Update Backlog and commit**

Record touched files and verification in `TASK-487`, then commit with `feat: add omnivoice sidecar smoke helper`.

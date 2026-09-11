# PR2761: Whisper alerts from the F7 Python rescan

Scope: alerts **2678** and **2679**, CodeQL Python analysis **1759185673**, scanned commit `f7c8af3af397a0eeec355a1cea333996d2e8287c`. The exact SARIF was downloaded from the analysis API to `/tmp/pr2761-python-f7.sarif.json`; extracted results are `/tmp/pr2761-2678-trace.json` and `/tmp/pr2761-2679-trace.json`. Both fresh instance lists contain only `refs/pull/2761/head` at that commit. No alert state changes are part of this work.

## 2678: remove unintended home-account lookup

Sink: `Audio_Transcription_Lib.py:693`, the newly introduced `path.expanduser()` in `_resolve_whisper_model_path`. Four SARIF flows originate from media/add.py:57 and audio_health.py:967, through model validation and the local-path resolver. Two use the local-path branch at line755; two use the CWD-under-root branch at line728.

Home expansion was unnecessary for managed model paths and added behavior absent before the earlier Whisper repair: a `~account/model` input queried the operating system account database before model-root containment. A deterministic reproduction with a controlled account resolver showed the same input raising `RuntimeError` for an unknown account and `ValueError` for an existing account. The public STT health validator catches only `ValueError`, so the distinction also creates an error-handling regression. This is an account-lookup/error distinction, not proof of arbitrary file disclosure. Probe: `/tmp/pr2761-whisper2678-home-probe.json`.

The minimal repair replaces `candidate = path.expanduser()` with `candidate = path`. Relative model paths again resolve under the administrator-configured model root, as they did before the earlier repair. Root configuration expansion remains separate. Lexical containment, parent-first directory symlink rejection, canonical containment, remote alias/Hub cache resolution and constructor options remain unchanged. A literal `~probe-account/model` directory under the managed root remains a usable local model path; input does not select an OS account's home.

Three failing regressions were added before the source edit: both known and unknown account cases must perform no account lookup and produce normal validation rejection; a literal-tilde managed directory must resolve inside the model root. All three failed against F7 and pass after removing expansion. The account-database cases require the POSIX `pwd` module; the literal-path and existing boundary tests remain portable.

## 2679: infeasible relative-Hub-ID branch

Sink: `Audio_Transcription_Lib.py:2547`, `normalized_path.exists()` inside `check_model_exists`. All four exact SARIF flows take the `_is_hf_model_id(raw)` branch and return `raw` at line739, twice through normalization before reaching the reported sink. Origins are media/add.py:57, audio_health.py:967, and audio_transcriptions.py:491 and :1430.

This branch accepts the full relative `organization/model` grammar and excludes leading `/`, `.`, `~` and Windows drive forms. Its output is never an absolute filesystem path. The reported sink is dominated by `if normalized_path.is_absolute()` at line2543, so none of these four flows can execute that existence probe. Real absolute local paths use the other normalization branches, which enforce managed-root lexical/canonical containment and reject directory symlinks before probing. Remote Hub IDs instead pass through confined cache candidates and do not invoke network downloads in `check_model_exists`.

Disposition: **false positive for the exact reported flows**. No source change made merely to alter this alert. Two new behavioral tests run the actual `check_model_exists` with representative Hub IDs and record every `Path.exists` call: neither ID enters the absolute-path existence sink, both return unavailable for an empty managed cache, and neither starts a download. Existing tests separately cover valid local directories, outside paths, directory links, aliases and offline downloads. The parser is not claimed to validate every Hugging Face repository naming rule; upstream download validation remains responsible for those rules.

## Verification

- Red: **3 failed**, `/tmp/pr2761-whisper2678-red.log`, before the one-line change.
- Green: **105 passed**, four focused transcription/model suites, `/tmp/pr2761-whisper-rescan-green.log`. This includes the five newly added regressions/evidence tests.
- Ruff passes for the touched source and test file; `git diff --check` passes.
- Bandit on the source plus tests, excluding pytest assertion rule B101, reports the same six pre-existing low production findings and two pre-existing synthetic-token test findings as the corresponding HEAD baselines. No new findings. Artifacts: `/tmp/pr2761-whisper-rescan-bandit.json`, `/tmp/pr2761-whisper-rescan-bandit-baseline.json`, `/tmp/pr2761-whisper-tests-baseline-bandit.json`.

- Independent reviewer: **29 model-resolution tests passed**, `/tmp/pr2761-whisper-rescan-independent.log`. The reviewer confirmed the one-line compatibility repair and independently checked all four infeasible 2679 traces; no actionable finding.

Limits: model/cache roots remain administrator controlled; validation does not claim an atomic defense against a privileged process racing filesystem replacement. The wrapper's previously documented ignored `files` argument remains outside this repair. Hosted analysis must confirm the source-fix result for 2678; runtime tests alone do not establish analyzer closure.

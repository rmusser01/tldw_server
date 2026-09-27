# PR2761: Whisper alerts from the F7 Python rescan

Scope: alerts **2678** and **2679**, CodeQL Python analysis **1759185673**, scanned commit `f7c8af3af397a0eeec355a1cea333996d2e8287c`. The exact SARIF was downloaded from the analysis API to `/tmp/pr2761-python-f7.sarif.json`; extracted results are `/tmp/pr2761-2678-trace.json` and `/tmp/pr2761-2679-trace.json`. Both fresh instance lists contain only `refs/pull/2761/head` at that commit. After this proof was committed in `6b671f22bc`, alert 2679 was independently dismissed as a false positive; alert 2678 remains a source repair awaiting analysis. See the [disposition ledger](PR2761-codeql-dispositions.md).

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

## 2682–2684: Windows model-root boundary follow-up

Exact Python analysis **1759306505**, scanned commit `009505c4154ca5d4c8c58312cd6ad527a1b4d045`, SARIF `/tmp/pr2761-python-009505.sarif.json`. The relevant Whisper and media-path helper source was unchanged at review HEAD `49cce1c8530252a6581e8013db6277a42f0288fa`. Fresh per-alert instance reads showed only `refs/pull/2761/head` at the scanned commit; no main instance was returned. No dismissals are proposed.

Each alert has four exact flows, originating from media/add.py:57, audio_health.py:967 and audio_transcriptions.py:491 and :1430:

| Alert | Scanned sink and trace | Disposition |
| --- | --- | --- |
| 2682 | `_normalize_whisper_model_identifier`:729, `local_path.is_dir()`. The CWD-under-root branch at 727–728 passes through `_resolve_whisper_model_path` and media `resolve_safe_local_path` before this sink. | Source repair. The branch and resolver used Windows case-insensitive `relative_to` checks, allowing an absolute case-only sibling to reach the directory probe. |
| 2683 | Same normalizer:756, `safe_path.is_dir()`, through the explicit local-path branch at 755 and the same resolver chain. | Source repair. A relative `..\cache\SecretModel` could escape the configured `C:\Cache` to a distinct case-sensitive sibling while still passing the case-folding checks. |
| 2684 | `check_model_exists`:2558, `candidate.is_dir()`, after the relative Hub-ID return branch at 739, cache-name replacement at 2552 and resolver call at 2555. | Covered by the shared Whisper boundary repair. An exact canonical postcheck now prevents a returned case-only sibling from reaching this cache probe, including the underscore cache candidate in the reported traces. This alert is not being dismissed as an impossible branch. |

The existing lexical guard `Path.relative_to` and the media helper's `ntpath.commonpath` comparison both fold Windows case. An execution of the actual current normalizer and media helper, using Windows path calculations, showed configured root `C:\Cache` accepting both `C:\cache\SecretModel` and `..\cache\SecretModel`. The link walk examined `C:\Cache\SecretModel`, but the final directory probe and returned model selected `C:\cache\SecretModel`. Proof: `/tmp/pr2761-whisper2682-2683-case-proof.json`. The proof was run with `PYTHONPATH=.` to select this worktree rather than the environment's editable-install checkout.

The repair changes `_resolve_whisper_model_path` and the normalizer’s CWD-path admission. The resolver requires exact lexical root equality or a separator-aware prefix before probing candidate components, retains the parent-first directory-link rejection, and checks exact canonical equality/prefix before returning a path. Canonical comparisons preserve spelling rather than lowercasing the path. Alternative root-case spellings deliberately fail closed; normal mixed-case descendants and canonical root/absolute/relative paths remain supported. No change was made to the shared media path helper or unrelated media-root selection.

Seven regressions/controls were added before the source edit: two outside case-only sibling cases, a canonical local result outside by case, an underscore Hub-cache canonical result outside by case, and three valid root/absolute/relative mixed-case controls. Before the fix **4 failed and 3 controls passed**; afterward all seven pass. The canonical-outcome tests emulate the filesystem resolver returning a distinct case-only sibling; they verify the final boundary and are not a native Windows reparse-point exploit demonstration. The earlier proof establishes the ordinary absolute/relative case-only sibling defect directly through Windows path calculations. No native NTFS execution was available, so native Windows compatibility is not claimed.

Verification:

- **114 passed**, the four focused transcription/model suites, `/tmp/pr2761-whisper-windows-final-green.log`.
- Red controls: `/tmp/pr2761-whisper-windows-red.log`.
- Scoped Ruff clean: `/tmp/pr2761-whisper-windows-ruff.log`.
- Scoped Bandit, excluding pytest assertion rule B101, exactly matches the HEAD baseline: six existing low production findings and two existing synthetic-token test findings, **zero new findings or scan errors**. Comparison ignores shifted source line numbers but retains rule, severity, confidence, issue text and source snippets. `/tmp/pr2761-whisper-windows-bandit.json`, `/tmp/pr2761-whisper-windows-baseline-bandit.json`.
- `git diff --check` passes. Original alias/Hub/offline/download-option tests remain in the passing suite. Hosted CodeQL confirmation remains required after source publication.

Per-alert trace and instance evidence: `/tmp/pr2761-{2682,2683,2684}-trace-review.json` and `/tmp/pr2761-{2682,2683,2684}-instances-review.json`. Machine-readable source-repair dispositions and final file hashes: `/tmp/pr2761-whisper-windows-final-batch.json`.

Independent review identified one compatibility case in the first repair: when CWD is the case-only sibling `C:\cache` of root `C:\Cache`, the original case-folding CWD admission sent opaque aliases and Hub IDs into the strict local resolver and rejected them. Two added tests failed first (`/tmp/pr2761-whisper-windows-cwd-red.log`). CWD admission now compares exact lexical strings as well, ignoring that outside CWD for `tiny.en` and `org/model` without probing it, while explicit outside paths still fail in the resolver. The final 114-test run includes both corrections. Shared `_path_is_within` behavior for unrelated media paths was not edited.

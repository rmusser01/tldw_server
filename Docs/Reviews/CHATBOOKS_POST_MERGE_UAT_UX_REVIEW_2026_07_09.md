# Chatbooks Post-Merge Backup And Import UX Review

Date: 2026-07-09  
Build reviewed: `origin/dev` at `440478b6cb`  
Plan: `Docs/superpowers/plans/2026-07-09-chatbooks-post-merge-uat-remediation-plan.md`  
Scope: WebUI and packaged-extension backup, download, archive preview, restore, job status, and recovery.

## Verdict

The merged work materially improves discoverability, but the workflow is not acceptance-ready.

- Backup all: possible and understandable, but not yet easy because it requires invented name/description metadata and reports incomplete terminal state.
- WebUI restore: not possible through the intended background-job path because a stale worker rejects the archive defaults before the implemented media/embedding restore service runs.
- Jobs: visible, but not trustworthy or efficient because completed jobs show `0%`, verification is blank, timestamps have ambiguous timezone semantics, and the detailed table competes with a duplicate tracker.
- Browser extension: buildable, but not certified end to end because the packaged E2E harness did not expose a service-worker target during UAT.

Overall post-merge UX health: **17/40, Poor**. The work should not be described as fully validated until the P0 round trip and P1 trust/accessibility findings pass live UAT.

## Method And Evidence

The review used:

- the completed full-account WebUI export and attempted round-trip import;
- desktop and 390px responsive live inspection of `/chatbooks`;
- accessibility-tree inspection for labels, roles, and state;
- computed dark-theme styles for Ant Design upload, progress, and empty-state components;
- cognitive walkthrough and Nielsen's 10 usability heuristics;
- three personas: first-time self-hoster, repeat backup power user, and data-migration operator;
- an independent design assessment and deterministic source scan.

Primary evidence:

- `apps/packages/ui/src/components/Option/Chatbooks/ChatbooksPlaygroundPage.tsx`
- `tldw_Server_API/app/api/v1/endpoints/chatbooks.py`
- `tldw_Server_API/app/services/core_jobs_worker.py`
- `/tmp/chatbooks-uat-export-completed.png`
- `/tmp/chatbooks-uat-import-tab.png`
- `/tmp/chatbooks-uat-import-uploaded.png`
- `/tmp/chatbooks-uat-import-failed-job.png`
- `/tmp/chatbooks-uat-jobs-tab.png`

The deterministic command was:

```bash
npx impeccable --json apps/packages/ui/src/components/Option/Chatbooks/ChatbooksPlaygroundPage.tsx
```

Impeccable `3.2.1` returned `[]`. This is not evidence that the live page is clean: the scan did not catch runtime theme inheritance, contradictory API state, accessibility names, or the 1,069px table rendered in a 694px main column. The installed version does not support the documented `impeccable live` command, so a browser overlay was unavailable.

Standards references:

- [NN/g: 10 Usability Heuristics for User Interface Design](https://www.nngroup.com/articles/ten-usability-heuristics/) defines the evaluation framework used below, including system-status visibility, real-world language, error prevention, recognition over recall, minimalist design, and constructive recovery.
- [NN/g: Visibility of System Status](https://www.nngroup.com/articles/visibility-system-status/) supports treating truthful progress, completion, and verification feedback as trust-critical rather than decorative status.
- [NN/g: Error-Message Guidelines](https://www.nngroup.com/articles/error-message-guidelines/) supports replacing raw multipart field names with a precise, plain-language problem and an immediate recovery action.
- [NN/g: Progressive Disclosure](https://www.nngroup.com/articles/progressive-disclosure/) supports keeping common backup and restore decisions primary while deferring conflict, prefix, and execution mechanics until needed.

## Anti-Pattern Verdict

The page avoids obvious AI-app styling: no gradients, glass, glow, hero metrics, or decorative spectacle. The restrained dark palette and operational language suit a self-hosted data tool.

The weaker pattern is assembly without enough product editing. Raw Ant Design defaults leak into the dark theme; cards are nested in the import preview; the Job tracker, Polling card, and Jobs tables repeat state; and several controls expose implementation mechanics instead of user decisions. The result feels credible at first glance but loses trust under task pressure.

## Nielsen Heuristic Score

| # | Heuristic | Score | Main evidence |
|---|---|---:|---|
| 1 | Visibility of system status | 2 | Jobs are visible, but `completed` and `failed` show `0%`; verification is `—`. |
| 2 | Match between system and real world | 2 | Backup/import language is improved; errors expose `import_media` and `import_embeddings`. |
| 3 | User control and freedom | 2 | Jobs can be cancelled/removed, but failed imports have no recovery action and deletion has no confirmation. |
| 4 | Consistency and standards | 2 | `Include all` plus `Selected: 0`; `completed` plus `0%`; two cleanup terms for different destructive operations. |
| 5 | Error prevention | 1 | The default WebUI archive path queues a job the live worker predictably rejects. |
| 6 | Recognition rather than recall | 1 | Recovery requires translating backend field names and identifying imports by UUID. |
| 7 | Flexibility and efficiency | 2 | Selective export and background jobs exist, but Backup all requires repeated metadata entry and Jobs is cramped. |
| 8 | Aesthetic and minimalist design | 2 | Clear tabs and restrained styling, offset by duplicate job surfaces and advanced controls on the common path. |
| 9 | Error recognition and recovery | 1 | The error is specific but not actionable in the interface. |
| 10 | Help and documentation | 2 | Scope/preview help exists; source-specific docs and tests still contradict archive restore behavior. |
| **Total** |  | **17/40** | **Poor** |

## Cognitive Load

Five of eight checklist items fail, which is high cognitive load.

- Single focus: fail. The permanent tracker and polling panel compete with the current Export, Import, or Jobs task.
- Chunking: fail. Backup scope exposes five metrics, category chips, and three full warnings; Export Jobs exposes ten columns.
- Grouping: pass. Tabs and major cards generally group related concepts.
- Visual hierarchy: fail. Primary actions compete with Refresh, cleanup, Remove, polling, and advanced switches.
- One thing at a time: fail. Import exposes conflict handling, prefixing, background execution, inclusion, and submission together.
- Minimal choices: pass. Individual selectors stay at four or fewer options.
- Working memory: fail. Failed recovery requires remembering the file and translating backend flags; the job row provides no direct route back.
- Progressive disclosure: pass with caveats. File preview gates content pickers, but conflict/prefix/background mechanics remain visible too early.

The emotional journey begins calm, peaks negatively when a supposedly restorable archive fails, and ends without recovery. The peak-end memory is therefore: "the backup completed, but I cannot trust that it restores."

## What Works

1. **Information scent is substantially better.** `Chatbooks Backup & Import`, the Export/Import/Jobs tabs, and `Backup all account data` match the user's task.
2. **The system exposes scope before commitment.** Backup inventory counts, pointer-only warnings, automatic archive preview, and source selection are the right foundations for a data-safety workflow.
3. **The operational model is viable.** Background jobs, status tags, cancellation, download, and a dedicated Jobs view are appropriate for large self-hosted archives.

## Priority Findings

### P0: The Intended Full-Archive Restore Path Fails

`chatbooks.py` correctly defaults Chatbook archives to restoring media and embeddings. `core_jobs_worker.py` then rejects those same flags before calling the working archive restore service.

Impact: the product promises complete restore, accepts the archive, queues the job, and fails without user error. This is a release blocker and a data-trust failure.

Fix: remove the stale live-worker guard for Chatbook archives and pass media/embedding flags through to the implemented restore service. Keep the OpenWebUI source-specific guard. Do not "fix" this by silently disabling media or embeddings; a full-account restore must restore all restorable archive data.

### P1: Terminal State Is Contradictory And Incomplete

Completed export: `completed`, `0%`, post-write verification `—`. Failed import: `failed`, `0%`. The API record leaves terminal progress at defaults, and the UI trusts the stale numeric zero before terminal status.

Impact: users cannot tell whether the archive was fully written, whether verification ran, or whether a failed job made partial progress.

Fix: persist 100% and final counts on successful completion, persist archive verification metadata, serialize timezone-aware timestamps, and let terminal status override historical stale progress in the UI.

### P1: Essential Dark-Theme Text Is Unreadable

The upload title uses `rgba(0,0,0,.88)` and its hint uses `rgba(0,0,0,.45)` on a near-black surface. The measured effective contrast is approximately 1.1:1. The same raw Ant Design foreground leaks into progress percentages and empty-state descriptions.

Impact: users with typical vision, low vision, or glare cannot read the primary upload instruction or status. This fails WCAG AA and the product's own design context.

Fix: use shared semantic foreground tokens for Upload, Progress, and Empty descendants in both themes; add automated token assertions and live contrast checks.

### P1: Controls Are Visually Present But Not Programmatically Named

The accessibility tree exposes unnamed switches for Run in background, Prefix imported, and Include all. Export mode, Tags, and Categories are visually preceded by text but their comboboxes do not receive an accessible name.

Impact: keyboard users can reach much of the workflow, but screen-reader users cannot reliably identify or predict these controls.

Fix: associate persistent visible labels with controls through native label structure or `aria-labelledby`; verify the complete flow with keyboard and accessibility snapshots.

### P1: Backup All Is Still A Form, Not A Safety Action

Backup all refuses to submit until users type both a name and description. Author, tags, categories, mode, background execution, a five-metric scope panel, and three warnings appear before the action.

Impact: the most common safety task is slower than necessary and repeatedly asks users to invent metadata.

Fix: generate an editable localized name/description when empty, keep customization optional, collapse detailed warnings, and move nonessential execution controls under Advanced options.

### P1: Restore Preview Understates Impact

The archive preview lists `Characters · 2`, but the full-account scope also includes account profile/settings and restore policy metadata. An enabled Include all switch simultaneously reports `Selected: 0`.

Impact: users cannot accurately predict destination changes and may interpret zero as nothing selected.

Fix: show a compact `What will be restored` account-inventory summary, verification state, sensitive-category summary, and warnings. Report `All 2` or `All in archive` when Include all is enabled.

### P1: Jobs Is Not A Usable Recovery Center

At 1280px, the export table is approximately 1,069px wide inside a 694px main column. The permanent 320px Job tracker duplicates the same jobs and leaves Conflicts/Actions outside the visible area. Failed import rows lead with a wrapping UUID and expose internal flags; the only action is Remove.

Impact: users cannot efficiently inspect, download, diagnose, or recover from jobs at the exact point where trust matters.

Fix: hide the side tracker on the Jobs tab, use the full content width, lead import rows with archive/Chatbook name, keep UUID copyable as secondary metadata, and add plain-language recovery actions.

### P1: Destructive Cleanup Has No Consequence Preview

`Cleanup exports`, `Remove finished`, and row-level Remove call destructive handlers directly. The UI does not say whether it deletes the downloadable archive, job history, or both.

Impact: a user can destroy the only server copy of a backup while trying to tidy history.

Fix: use distinct terms for expired archive cleanup versus history removal, state exact scope, and confirm bulk/file deletion. Add undo only where the backend can support it honestly.

### P2: Operational Details Compete With User Decisions

`Async by default` duplicates the Run as background job switch. A permanent Polling card explains 3-to-30-second backoff. Conflict resolution and prefixing are shown before the normal archive path needs them.

Impact: implementation mechanics make the product feel more complex and reduce the prominence of the primary action.

Fix: use `Advanced options` for conflict, prefix, and execution mode; replace polling implementation detail with a compact updating status or hide it outside diagnostics.

### P2: Extension And Broad Acceptance Coverage Are Not Yet Trustworthy

The production extension build succeeded, but the Chatbooks E2E did not complete because no service-worker target appeared. The broad backend run also exposed stale integration/docs assumptions.

Impact: WebUI mechanics can pass while the packaged extension and round-trip restore remain broken.

Fix: make the extension harness seed configuration from an extension page when an idle MV3 service worker is absent, require real export/download/import completion, and align integration/docs tests with the full-account contract.

## Persona Red Flags

### First-Time Self-Hoster

- Cannot read the primary upload instruction in dark mode.
- Encounters pointer-only, embedding, conflict, and async terminology before the core action is established.
- Receives backend form-field names instead of a recovery path.
- Cannot tell what Remove deletes.

### Repeat Backup Power User

- Must type name and description for every safety backup.
- Cannot repeat a previous backup configuration in one action.
- Sees duplicate job status and a cramped table rather than one efficient history view.
- Cannot trust `completed · 0%` or blank verification.

### Data-Migration Operator

- Preview does not summarize all account-level destination impact.
- Failed restore has no retry/review action and no partial-result summary.
- Timestamps have ambiguous timezone semantics.
- Full-account archive compatibility is discovered only after commitment.

## Recommended Execution Order

1. Fix the live async archive restore path and add a real media-bearing round-trip test.
2. Normalize terminal progress, verification, counts, and timestamps.
3. Make the common backup/restore path one-action, progressively disclosed, contrast-safe, and fully labeled.
4. Make Jobs full-width, human-readable, recoverable, and explicit about destructive consequences.
5. Repair the packaged-extension harness and require its full round trip.
6. Align stale integration/docs contracts and rerun the full UAT matrix.

The detailed TDD steps and release gate are in `Docs/superpowers/plans/2026-07-09-chatbooks-post-merge-uat-remediation-plan.md`.

## Remediation UAT Addendum - 2026-07-10

### Updated Verdict

The WebUI full-account backup and restore workflow is now possible,
straightforward, and sufficiently low-friction for routine use. The exact
browser workflow exported a v1.1 archive, downloaded it through the WebUI,
stopped the source services, started a distinct clean destination, imported
that exact download, and verified destination state rather than trusting job
completion alone.

The packaged browser extension is not certified. Its production Chrome MV3
build and focused unit tests pass, but Playwright cannot establish a usable
persistent extension context on this host. The failure occurs before the
Chatbooks page executes, so it is not evidence that extension backup/restore is
broken; it is also not acceptable evidence that extension parity works.

Updated status:

- WebUI Backup all: **easy enough for routine safety backups**. Metadata can be
  generated, scope is explicit, terminal state is truthful, and the archive is
  downloadable and verified.
- WebUI restore: **straightforward and predictable**. Preview now includes
  account profile, account settings, stored media artifacts, and verification
  state before commitment.
- Jobs: **usable as an operational history and recovery surface**. Completed
  state, identity, cleanup consequences, and recovery actions are no longer
  contradictory.
- Packaged extension: **unknown pending a working host launch path**.

### Acceptance Evidence

- Exact archive:
  `/private/tmp/chatbooks-full-account-browser-uat/webui/browser-downloads/webui-full-account.chatbook`
- Archive version: `1.1.0`; post-write verification passed.
- Archive SHA-256:
  `45fd5c40f5ff8cdb8226fc35fe7e68fb511bdc26efb673613986b2ef5b25ad1d`.
- Stored media SHA-256 matched after restore:
  `6fc4135fef28f9c56af8e075adb6275f55000736c44e8a3551b97b55e730375f`.
- Clean destination restored account profile, locale/theme settings, character,
  media record, transcript, two chunks, bundled media bytes, and vectors
  `uat-chunk-001` and `uat-chunk-002`.
- Sensitive-data inspection found no fixture password hash, raw server storage
  path, or unredacted sensitive payload in browser-visible metadata/log output.
- Backend in-process matrix: 436 passed, 9 documented prerequisite skips.
- Host-spawning legacy E2E subset: 3 documented skips because the Python
  Playwright Chromium binary is absent; no startup or product error remained.
- Focused remediation regressions: 18 Python, 29 runtime-bootstrap, and 7
  extension-path tests passed. WebUI typecheck, extension compile, production
  Chrome MV3 build, token sync, `git diff --check`, and production-scope Bandit
  all passed.

Live UAT also found and corrected four acceptance-harness/product-contract
defects: manual multi-user tokens were removed during WebUI bootstrap, the
preview API omitted account inventory fields that the UI already knew how to
render, loose preview dictionaries could echo undeclared manifest fields, and
the frontend readiness fixture read only the first 4 KiB of a larger Next.js
document. The final exact browser round trip was rerun after the redacted
preview response models were added.

### Updated Nielsen Score - WebUI

The score below applies to the remediated WebUI workflow. The extension is not
scored because its current screen/workflow could not be reached reliably.

| # | Heuristic | Score | Current evidence |
|---|---|---:|---|
| 1 | Visibility of system status | 3 | Progress and verification are truthful; notification requests still fail silently during bootstrap. |
| 2 | Match between system and real world | 4 | Backup, restore-impact, archive identity, and cleanup language describe user outcomes. |
| 3 | User control and freedom | 4 | Advanced options are disclosed, recovery is available, and destructive actions confirm scope. |
| 4 | Consistency and standards | 4 | Completion, counts, Include all, and archive verification now agree. |
| 5 | Error prevention | 4 | The default full-archive path restores media and embeddings instead of queuing a predictable failure. |
| 6 | Recognition rather than recall | 4 | Preview categories and human-readable archive identity remove UUID/flag translation. |
| 7 | Flexibility and efficiency | 4 | Backup all supports generated metadata while selective and advanced workflows remain available. |
| 8 | Aesthetic and minimalist design | 3 | The common path is clearer, though this remains a dense operational surface. |
| 9 | Error recognition and recovery | 3 | Known failures have recovery actions; extension launch and notification authorization lack in-product diagnosis. |
| 10 | Help and documentation | 3 | Source-specific contracts are aligned; environment prerequisites remain fragmented across runners. |
| **Total** |  | **36/40** | **Good; WebUI acceptance-ready, extension not certified.** |

### Remaining Findings

#### P1: Packaged-Extension Parity Is Still Unproven

Three evidence-based launch attempts exhausted the repository retry limit. The
initial headless run exposed no service worker or extension target. After
deterministic manifest-key staging, the extension URL was blocked because the
extension still was not loaded. Headed persistent-context launches then timed
out, including a final 120-second attempt, before Playwright established its
debugging pipe.

Recommendation: separate extension-process certification from Chatbooks UAT.
Create a small host capability gate that proves Chrome loads the unpacked MV3
package, exposes an extension page, and can read/write the storage sentinel.
Only then run the existing Chatbooks export/import phases. Preserve the current
fail-closed rule: build success or a derived extension ID must not count as UAT.

#### P2: Notification Authorization Fails Without User Feedback

The exact source WebUI run issued four notification list/count/stream requests,
all returning `403`, and did not later show a successful notification request.
The Chatbooks workflow still passed, but the shell gives no visible explanation
that notification status may be unavailable.

Recommendation: defer notification startup until authentication bootstrap is
ready, or stop/retry with bounded backoff after an authorization transition.
If notifications remain unavailable, expose a compact nonblocking status in
the notification surface instead of failing only in logs.

#### P2: Extension Build Warnings Remain A Runtime Risk

The production build succeeds but reports duplicate imports, circular
cross-chunk re-exports, unresolved runtime font URLs, and large chunks. These
warnings predate the Chatbooks changes and do not explain the failed debugging
pipe, but the circular execution-order warnings are material for extension
reliability and startup performance.

Recommendation: track these separately from Chatbooks. Resolve direct imports
for the warned `tldwClient`/`tldwModels` consumers, then establish a startup
budget and a smoke test for options-page first interaction.

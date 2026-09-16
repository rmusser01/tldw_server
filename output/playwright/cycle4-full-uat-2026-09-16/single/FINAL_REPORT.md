# Cycle4 single-user fresh UAT — final report

TASK13260. Execution ended 2026-09-16 02:03 UTC. Product remained `7c9409fad2` throughout; documentation commits did not change runtime source. No full acceptance sign-off.

## Provenance and limits

- Fresh isolated config/data/browser: API18300, WebUI18380, session `cycle4-single-20260916`. Existing installed dependencies were reused; this is not a clean-machine installation test.
- Branch `codex/fresh-install-uat-fixes` includes corrected dev `2e1a5e58d3` via merge `267c00cab1`. The task originally missed32 already-fetched dev commits. During this frozen run, remote dev advanced another6 commits to `59049e094e`; those are not part of this tested product.
- Real local model: custom_openai / discovered Gemma at9099. No mocked inference or API-seeded content counts as workflow acceptance. Runtime credentials remain in private0600 manifests; retained browser output redacts them.
- One operator workaround restarted only the unchanged single API after setup (PID86970→23507), refreshing stale model inventory107. Source/config hash provenance is retained. Normal Chat succeeds after this workaround; the original setup-to-Chat failure remains.
- User journeys come from the frontend E2E/shared workflow inventory in the global tracker. No verified literal A/B/C labels exist; named journeys are used explicitly.

## Workflow matrix

| Workflow | Outcome and evidence limits |
| --- | --- |
| Fresh setup/provider discovery/first real Chat | PASS with explicit optional-lane deferral. Blank-model discovery finds Gemma; validation/save and real first response200 (`Hello!`) pass. Manual API-key setup restores access. Audio, RAG/storage and MCP deferred through UI. |
| Authentication/reload/offline disconnect/reconnect | PASS for the single-user key contract. Offline Disconnect clears key and blocks Media in other tabs; normal reconnect/Test Connection gives Core reachable/RAG healthy, and reload retains the key. Online4second control also clears the key. Existing local Chat transcript remains visible by the current connection-only single-user design; no transcript deletion/lock or cross-account isolation claim. |
| Ordinary two-turn saved Chat/reload | FAIL106/107 initially. Fresh Chat selects unrelated Ollama/gemma3 and returns502; advertised configured Gemma returns400 until API restart. After restart both ASTER-42 turns return200, but saved reload shows7 rows for5 canonical rows103. Subsequent prompt request carries duplicate users in context. Retry also forwards display-error JSON108. |
| File ingest/search/cited QA/source Chat | File/FTS search and cited QA PASS; ingest minimize FAIL104. Standard source has260 correct characters and meaningful analysis. Balanced specific-media QA returns correct Mira/date answer with1source/1citation; actual citation preview/Open in Media work. Media→Chat transfers exact text. Question about volunteers/shed later hits the streaming limit with explicit interruption/recovery, so direct source Chat has no completed-answer pass. |
| Exact Wikipedia URL/search/Chat | EXTERNAL BLOCK with honest failure feedback. Exact `https://en.wikipedia.org/wiki/Playwright_(software)` returns Access blocked,0success/1failure. No article was saved; dependent source search/Chat are blocked, not substituted. |
| Note→five generated cards/save/review/reload | Functional PASS; a11y112 fails. Saved/reopened Aster Note transfers exact five facts/provenance. Real generation returns five correct cards; all saved. Five deck cards plus two manually-created undecked cards receive exactly7 distinct Easy reviews200. Completed session/reload counts7; Cram rating does not change scheduled totals. Two additional manual cards prove explicit early-End after1review. |
| Prompt create/save/back/apply/request/answer | Functional PASS after114 recovery. Local save and Back preserve prompt; six-tab request starvation causes failed automatic sync. Closing3completed tabs and visible Retry sync create project/prompt201. Use in Chat→System Instruction sends exact pirate system prompt; real answer says `Ahoy, ye garden-tending scallywags, ARRR!`. Failure notification emits a static-context warning. |
| Character replacement/real Chat/history | Explicit replacement PASS, reload FAIL113. New Guide4 clears pirate prompt/old history. Correct real Mira/date answer completes/persists200. Creation-route reload shows greeting only while canonical3rows remain. Native Server history reopening restores all3. Later source handoff reveals duplicate-user mirror103 also affects character history. |
| Chat→Note/backlink; Chat→card→Study | Note save201 preserves exact clean pirate answer/origin; backlink FAIL111 (false unsaved-Chat guard). Character answer opens reviewed Flashcard draft with clean answer and required question, then saves201. Its actual canonical message ID matches the server transcript. Saved card plus remaining early-End card receive2distinct reviews200 and completed-session reload. Actual practice-card Note link opens original Aster Note/title/body. |
| Media analysis/Review/reanalysis/reload | Original ingestion analysis/native Inspector and Review/reselection/reload PASS. Analyze failure produces dev overlay110. Selecting Gemma in Analysis reverts to gemma3 and submits the wrong provider115, including after backend restart; no completed reanalysis claim. Source/previous analysis preserved. Review displays analysis as plain Markdown source; rendering consistency remains an observation for triage. |
| Sole-source delete/Trash/restore | PASS. Explicit soft-delete204 leaves reachable Trash in empty library; valid deletion date shown. Restore200 and reopen retain exact source/prior analysis; final independent GET200 corroborates. No permanent deletion. |
| Multi-user administration/isolation | Not applicable here; independently exercised in the multi-user report. |

## Canonical synthetic records

- Media1: `cycle4-aster`,260characters.
- Original Note: `4377bd3b-6d92-4c1c-a7d2-430ea3efedc0`.
- Successful normal Chat: `94a50532-cc5c-4f6e-baca-761e25ce295b`. Earlier failed Chat `a88e82b1-b2d0-48bd-9875-0a8e86bed582` intentionally remains; it is not a duplicate-conversation regression.
- Chat Note: `1420d30b-a7e9-4292-99bc-c0ac31ed050e`, canonical message `b24f2d31-f18b-4c87-b205-92cc1a2441b8`.
- Character4 Chat: `ce67520e-883b-42ff-b565-c2872616c048`, saved assistant message `pa_baec-8ac1-caa-6966` (verified server ID despite its prefix).
- Reviewed Chat card: `7f077a43-b4ba-487e-8dd9-edf1a4c4a3dc`; saved source Note `5b492a7b-c771-4720-92fe-a89773d5dc9d`.
- Due sessions:7complete,1explicitly ended,2complete; total10scheduled reviews. Cram practice is separate.

## Findings and controls

Single observations confirm103/104/106/107/108/110/111/112/113/114/115. Multi-only findings/candidates are independently retained in its report. Global tracker owns numbering and repair status.

-114: before closing tabs, browser health/Media/Prompt requests abort while independent API200. After closing3tabs, ordinary requests and prompt sync recover. Per-tab hidden notification streams are source/probe-confirmed; exact socket occupancy is inferred rather than measured.
-115: separated native model selection and1second settled read both show gemma3 again; captured streaming/fallback bodies use Ollama. Actual component/storage/shared-owner probe reproduces the overwrite.
- Reopened058: Prompts/Characters titles blank. Settings reconnect briefly adopts previous Chat title; later reload restores Server Settings.
- Reopened067 candidate: New character emits `useForm` not-connected warning; native creation/selection still succeed. Exact filtered console line retained for diagnosis.
- AntD hidden loading icon remains in accessible names112 despite enabled controls. This is not a stuck save. Native observed-control clicks create cards.
- Planning/admin policy issues are not inferred from single mode. No unauthenticated server read/write success or cross-account leak was observed.

## Harness corrections and environmental observations

Response observers initially assumed `/notes` instead of actual `/chat/knowledge/save`, and `/prompts` instead of Prompt Studio endpoints; requests succeeded where independently shown. Open in Media opens a new tab, so the initial same-tab URL waiter timed out. AntD option/radio selection and hover-replaced actions required current visible controls. One shell URL needed quoting. Developer-tools badge intercepted the bottom Settings button; visible header Settings worked. These are not additional application acceptance failures.

Initial Cram observer expected a scheduled review POST, but Cram intentionally does not change scheduling; corrected observations show its actual behavior. First mixed-review observer matched hidden Manage text; subsequent native reviews scope the active-card test ID and wait for the distinct next card. No repeated rating is counted as a distinct card in this report.

Backend restart causes expected transient connection errors in open tabs. Both exact Wikipedia attempts remained within the original site/outbound restrictions. Original source/data/evidence are preserved.

## Evidence

`evidence-manifest.json` indexes retained nonempty sanitized captures with SHA256. Some `.json` filenames contain the CLI's Markdown envelope; raw API `.json` files are distinguished by their content. Empty timed-out observer output is excluded. `RUNNING_TRACKER.md` preserves earlier chronology; this report supersedes its pending rows. Private diagnosis artifacts are retained separately after integrity/credential checks.

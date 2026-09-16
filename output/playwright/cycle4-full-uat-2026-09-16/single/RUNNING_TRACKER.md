# Cycle4 single-user fresh UAT — running tracker

**Execution closed 2026-09-16 02:03 UTC.** The [final report](FINAL_REPORT.md) supersedes pending rows below and records all actual outcomes, failures and coverage limits. Product source remained frozen at7c9409fad2. The branch includes dev2e1a5e58d3; origin/dev subsequently advanced6commits during execution.

Parent: TASK13260. Product freeze: `7c9409fad2`; documentation checkpoint: `9f17739b4a`. Verified `origin/dev` tip `2e1a5e58d3` is an ancestor, with zero dev-only commits. The original task did not begin from that latest-dev tip; this fresh pass uses the corrected branch.

New profile: `/private/tmp/tldw-onboarding-uat-cycle4-single-20260916`; API18300, UI18380; browser session `cycle4-single-20260916`. Normal AuthNZ initialization succeeded. Configuration/data/browser state are fresh; existing project dependencies are reused. No clean-machine dependency installation claim. No product edits during the run.

## Workflow results

| Workflow | Status |
| --- | --- |
| Fresh setup, provider discovery and real first Chat | PASS: blank-model discovery, validation/save and real first response200; manual API-key access recovery succeeds |
| Normal saved Chat, second turn and reload | FAIL106 initial unrelated model502; FAIL107 selected advertised configured model400. Isolated API restart workaround performed; normal recovery/reload pending |
| File ingest, search, grounded Chat/QA and citations | In progress: Standard file saved with correct260-character content and meaningful real analysis; background minimize FAIL104; search/QA pending |
| Exact Wikipedia URL, search and grounded Chat | Not started |
| Notes, generated Flashcards, save, review and reload | Functional PASS: saved/reopened Note, five real grounded cards saved, mixed seven-card review and reload count correct; accessibility FAIL112 |
| Save/apply Prompt, actual request and real response | Not started |
| Character Chat, explicit replacement and saved history | Not started |
| Chat to Note/backlink and reviewed card to Study | Mixed Study PASS: five generated plus two undecked, seven distinct ratings200, completed session and reload7. Cram advances without changing scheduled totals; separate early-End control in progress. Chat/Note handoff pending |
| Media analysis, review, re-analysis and reload | Original source/Markdown analysis native PASS; reanalysis request failure triggers dev overlay110. Recovery/Review pending |
| Delete, Trash and restore | Not started |
| Disconnect, offline logout and reconnect | Not started |
| Admin/ordinary-user isolation | Not applicable to single-user mode |

## Findings and observations

- Fresh profile preflight verifies no users database before initialization, blank RAG provider/model defaults, no inherited provider environment overrides and private configuration permissions0600.
- Existing real local inference model discovery succeeds at `http://127.0.0.1:9099/v1/models`. Visible setup selected `../../../Working/Language_Models/gemma-4-26B-A4B/gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q4_K_M.gguf` under custom_openai. First-chat response200 contains `Hello!`. This differs from the earlier cycle3 model.
- Audio, RAG/storage and MCP explicitly deferred using the UI. Manual access key UI restores Home successfully. Expected missing-key setup warnings precede manual key entry; no setup error observed.
- Home File entry, native upload of `cycle4-aster.txt`260 bytes, Standard preset: Next with blank provider remains Configure and focuses the provider with a clear validation alert (UAT056 control PASS). Selected configured custom_openai and started processing.
- **UAT104 / TASK13260.45:** Minimize to Background leaves the active dialog visible at0:53. Capture `/private/tmp/uat-cycle4-single-minimize-control.txt`. Continued independent Chat in a normal new tab, without cancelling the job.
- Authenticated read of media1 returns200 and correct source plus meaningful Markdown Analysis covering the six Aster facts. This is a positive control against multi-user UAT105, not yet a native Media review/reanalysis pass. Evidence `/private/tmp/uat-cycle4-single-aster-media.json`.
- API18300 alone restarted unchanged after offline107 diagnosis: oldPID86970 →23507; docs200; config hash/provenance retained. This is an operator workaround, not a clean setup acceptance result. No product source edits.
- Note UUID4377bd3b-6d92-4c1c-a7d2-430ea3efedc0 saved exact five facts and real generation returned five distinct correct question/answer pairs. Saved to Cycle4 Aster facts; independent GET200 verifies cards. Two UI-created undecked cards used public synthetic flag/chair facts.
- Mixed due review session1 completed with exactly seven distinct cards rated Easy and seven reviews; reload retains Completed / All decks / 7 cards reviewed, new0 and next review in4days. Evidence study-complete/reloaded and review2-through7 captures. First rating succeeded; its observer initially matched both hidden Manage and visible Study text, corrected by scoping the active-card test ID. No repeated rating.
- Cram practice advances to the second card after one rating and leaves scheduled reviewed-today7. Its lack of a schedule-review POST is an observed practice contract, not a failure; the initial observer expected the wrong endpoint and timed out. Separate early-End control uses two new UI-created cards.
-112 is accessible naming, not stuck submission: hidden exit-animation loading icon remains in the accessibility tree although buttons are enabled. Native observed-button click works. Wiki provider selection initially targeted a hidden title node; corrected native option selection remains pending, with no extraction yet claimed.

# Native trust contradiction repair

- Base: `2911b98006fcd45738f3a125011d82a1b089d0a1`, isolated worktree `/Users/macbook-dev/.codex/worktrees/knowledge-ux-review/tldw_server2`, latest-dev base `025627214c3aeda1b2e9af5c6a9f85c636a2ec02`.
- Binding requirements: `native-trust-brief.md`, Ruling16; first narrow production repair, no subagents/reviewers. TASK-13453.2 was In Progress before edits and remains In Progress for controller acceptance.
- Controller owns final combined checks/types/builds, actual native/browser proof, independent scoped review, design/plan/review assets and runtime. No upstream, runtime, dependency, backend or API changes.

## Root cause and contract trace

`rag_result_to_response` in `app/core/RAG/rag_service/response_mapping.py` classifies backend trust from the returned result citations or chunk citations. `trust_contracts.py` deliberately checks structural source correspondence and inspectability, not semantic answer support. `unified_pipeline.py` exposes generated academic/chunk bibliography, `inline_citations` and `citation_map` metadata separately from generated answer text. `guardrails.py:build_hard_citations` can heuristically map each answer sentence to offsets in multiple top documents; quote citations separately carry their verified flag. The API schema describes academic/chunk citations and required hard-span metadata; it does not make all these records inline answer references.

Knowledge's `CitationRef` contract is an index actually shown in answer `[N]` text and its returned document ID. The provider's parser creates those refs; scope validation filters them against accepted original source indexes. The native response has no such answer markers, so its visible citation list is empty. The shared normalizer previously returned backend `cited_answer` immediately, while AnswerPanel independently detected zero citations and showed a degraded recovery warning. Saved-thread hydration independently bypassed the normalizer when a stored trust state existed.

All normalizer call sites were traced: final nonstream/stream completion after source filtering, streaming contexts/deltas, retrySync, and history hydration. Saved context creation carries the normalized current state into both `trust_state` and `knowledge_trust`; history snapshots, AnswerPanel, Notes/export metadata, and Research prefill consume that state. Restore/shared-thread/branch hydration share `deriveThreadHydrationState`. Existing sync/transport behavior and owner/request fences are unchanged.

## Minimal change and exact precedence

Only two production files changed:

1. `apps/packages/ui/src/components/Option/KnowledgeQA/trustState.ts`: transport failure wins, then sync failure, then every valid non-positive backend state (including unknown). Positive backend cited trust now follows the same returned-citation checks as positive local inference. Missing visible refs yields `uncited_degraded_answer`/`missing_citations`; refs whose document ID is absent from returned results yield `citation_source_not_returned`; absent excerpts or the existing SourceCard unavailable status/reason yields `no_answer_insufficient_evidence`/`missing_inspectable_evidence`. Existing source text/unavailability helpers are reused. Backend reason codes and evidence origin are retained for these citation qualifications. Inspectable returned inline refs preserve the cited state. Legacy input with neither backend nor required trust metadata remains unknown.
2. `apps/packages/ui/src/components/Option/KnowledgeQA/KnowledgeQAProvider.tsx`: saved-thread hydration supplies its stored trust metadata to the same normalizer instead of accepting it directly. No old records are migrated or rewritten.

No answer citations are invented from bibliography, `inline_citations` dictionary entries, hard spans, quote spans or citation-map metadata. Those backend contracts remain untouched; this view qualifies its own positive claim against the answer refs it can render and inspect. This is conservative UI trust qualification, not a backend semantic-grounding change.

## RED, GREEN and covering evidence

All commands below ran in `apps/packages/ui` unless specified. Tests use a safe constructed fixture with the actual native response structure: two generated sentences without inline tokens, canonical Notes UUID and excerpt, backend positive trust, two hard spans, metadata `[1]` map/citation map, academic bibliography and background chunk record. No runtime response or native mock was changed.

```sh
./node_modules/.bin/vitest run src/components/Option/KnowledgeQA/__tests__/trustState.test.ts src/components/Option/KnowledgeQA/__tests__/KnowledgeQAProvider.native-trust.test.tsx --maxWorkers=1
```

- RED before production edits: `/private/tmp/knowledge-native-trust-red.log`, exit1, **8 failed / 19 passed**, 1.79s. Live native shape, hidden-source positive trust, saved cited/no-ref hydration and five missing/mismatched/uninspectable source cases all incorrectly remained cited. The valid inline and qualified/unknown cases already passed.
- GREEN: `/private/tmp/knowledge-native-trust-green.log`, exit0, **27 passed**, 2.20s.
- A preliminary fixture run initially returned unsynced because its RAG-context success body did not match the existing POST contract. Corrected test fixture success/body/method before recording the meaningful RED above; no production change preceded RED. A first relative-path file write also failed without editing a file. These are test setup corrections, not extra product repair attempts.

The integration test mounts the real provider, AnswerPanel and ExportDialog. It verifies rendered trust/count/reason agreement; canonical result ID; current/local history; actual persisted RAG-context body; reopening that newly persisted context via History; Research handoff via the real builder; unsupported export acknowledgement and rendered Markdown; and the real Notes provenance reader on the saved content. It separately retains valid inline support, excludes out-of-scope references, and covers legacy stored cited, unknown, qualified and missing trust metadata. Unit cases retain all authoritative non-positive states and existing sync/transport overrides.

```sh
./node_modules/.bin/vitest run src/components/Option/KnowledgeQA/__tests__/trustState.test.ts src/components/Option/KnowledgeQA/__tests__/KnowledgeQAProvider.native-trust.test.tsx src/components/Option/KnowledgeQA/__tests__/KnowledgeQAProvider.history.test.tsx src/components/Option/KnowledgeQA/__tests__/KnowledgeQAProvider.persistence.test.tsx src/components/Option/KnowledgeQA/__tests__/KnowledgeQAProvider.streaming.test.tsx src/components/Option/KnowledgeQA/__tests__/KnowledgeQAProvider.scope-handoff.test.tsx src/components/Option/KnowledgeQA/__tests__/AnswerPanel.states.test.tsx src/components/Option/KnowledgeQA/__tests__/AnswerPanel.workspace-handoff.test.tsx src/components/Option/KnowledgeQA/__tests__/ExportDialog.a11y.test.tsx src/components/Option/KnowledgeQA/__tests__/trustSummary.test.ts src/components/Option/KnowledgeQA/__tests__/scopeValidation.test.ts --maxWorkers=1
```

`/private/tmp/knowledge-native-trust-covering.log`, exit0: **205 passed / 11 files**, 9.64s, after changed-block formatting and the fresh-save reopen assertion. Expected negative-path logging and the existing Node localStorage warning remain; no skipped tests or unresolved failures. No semantic edits followed this covering run. Controller owns broader final suites and official types/builds.

## Formatting, hooks and security applicability

Used the already installed Prettier 3.8.1 via Node resolution from `apps/tldw-frontend` for the new test and changed ranges only in the existing files. No dependency installation or link changes. `git diff --check` passed.

Repository-root command (original venv activated first):

```sh
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pre_commit run --files apps/packages/ui/src/components/Option/KnowledgeQA/trustState.ts apps/packages/ui/src/components/Option/KnowledgeQA/KnowledgeQAProvider.tsx apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/trustState.test.ts apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/KnowledgeQAProvider.native-trust.test.tsx 'backlog/tasks/task-13453.2 - Preserve-research-evidence-and-complete-saved-output-review.md'
python -m bandit -r apps/packages/ui/src/components/Option/KnowledgeQA/trustState.ts apps/packages/ui/src/components/Option/KnowledgeQA/KnowledgeQAProvider.tsx -f json -o /private/tmp/knowledge-native-trust-bandit.json
```

`/private/tmp/knowledge-native-trust-precommit.log`: all applicable hooks pass, exit0; Python syntax/Black/Ruff are inapplicable and skipped. Bandit exits0 but reports **two AST syntax errors because these files are TypeScript, not Python** (`knowledge-native-trust-bandit.json` and `.log`). This is a recorded applicability limitation, not a successful TypeScript security scan. No Python changed in this wave. Self-review confirms the change only restricts positive trust; it leaves all auth/request/scope boundaries intact.

## Commit scope and handoff

Exactly five files belong to this repair: the two production files, `trustState.test.ts`, new `KnowledgeQAProvider.native-trust.test.tsx`, and canonical TASK-13453.2. The task was updated via original-venv `PYTHONPATH=tools/backlog-py/src python -m backlog_py --cwd "$PWD" task edit TASK-13453.2 --append-notes ...`; it stays In Progress.

Controller-owned dirty design/plan, untracked final review/assets, Documents build traces, UI node_modules symlink and Next build output remain excluded. No private runtime descriptors/auth/profile/logs were inspected. No new state owner, persistence layer, migration or API. Commit SHA and final status appended below.

Final commit: `03f4e0627b5ae2e38ffa3eba2947688fab89452e` (`fix(knowledge): qualify cited trust against inspectable answer references`). Exactly the five scoped files committed, no hook bypass. Post-commit index is empty; status contains only the controller-owned files/assets/build outputs enumerated above. Git emitted the existing automatic-gc/unreachable-object maintenance warning; no cleanup attempted. No unresolved narrow product concern. Final native acceptance, official type/build gates and independent review remain controller-owned.

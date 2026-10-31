# UAT367 Conversation System Prompt repair

**Task:** TASK-13260.277.17
**Goal:** Preserve a directly edited Conversation System Prompt through Save and the next ordinary Chat request while preserving owner boundaries and reset behavior.
**Scope:** ConversationTab, CurrentChatModelSettings and directly related tests; coordinate any needed store/hook changes with root. UAT356 source and shared runtimes stay frozen.

## Stage 1: Isolate loss point
**Goal:** Test editor, Save/model-scope transition, and later readiness/account events separately.
**Success Criteria:** A causal failing DOM test identifies where the saved value diverges from visible/form/active settings state; native labels008–015 remain intact.
**Tests:** Real AntD form and model store; immediate send settings; stable model scope versus inferred canonical scope; same-owner config event versus owner invalidation.
**Status:** Complete

## Stage 2: Minimal repair
**Goal:** Repair only the confirmed editor/settings path using existing form and model-setting conventions.
**Success Criteria:** Current owner’s entered value survives Save/reopen and feeds the next request; Reset updates visible and effective values; Character/template behavior remains intact.
**Tests:** Direct edit/Save; initial value and reset; ordinary and Character/template controls; account invalidation rejects stale work.
**Status:** Complete

## Stage 3: Verification and handoff
**Goal:** Run relevant settings, prompt assembly, owner controls, lint/type comparison and applicable Bandit check.
**Success Criteria:** Targeted tests pass, no new diagnostics, independent review; leave native SQLite/PostgreSQL acceptance to root and task In Progress.
**Tests:** Installed node entrypoints only; matched-baseline lint/type checks; git diff whitespace check.
**Status:** In Progress

## Initial hypotheses

1. The named Form.Item controls a wrapper div instead of the textarea. Live onChange still writes the model store, but initial/reset values may not reach the visible editor.
2. Save switches active model settings scope before applying allowed settings, and excludes systemPrompt. A direct override written to the previous scope could disappear during this transition.
3. Provider restart/readiness/auth notifications may invalidate a same-owner value. Isolate this from Save before changing any account security behavior.

## Diagnosis

Real AntD editor and live model store reproduced loss immediately at Save without any outage: inferred model scope changes from the cockpit route key to canonical llama.cpp/llamacpp key, and generic Save excludes the live system prompt. Explicit unchanged scope and same-owner config notification controls pass. Separately, the named Form.Item controlled its wrapper div, so existing values and Reset never reached the visible textarea. Initial semantic run:4failed/2passed in /tmp/uat367-editor-red.log.

The final Save logic keeps the existing active scope when its canonical provider/model identity matches the selected model. It preserves explicit scope precedence and genuinely different-model targets. Selected provider/model comparisons use existing normalization helpers, including catalog providers and duplicate model IDs across providers. This leaves one live prompt record for Save and Reset, rather than copying private prompt text between aliases. Generic cached-form prompt saving remains excluded, preserving account invalidation. A nested noStyle Form.Item controls the textarea directly; initialValues hydrates a newly mounted form from the live prompt when React Query reuses cached config.

## Causal regression evidence

- Immediate Save, initial/reset binding:4red/2controls pass, /tmp/uat367-editor-red.log.
- Reset/reactivate:2red under the provisional copy approach, /tmp/uat367-scope-reset-red.log. Read-only original HEAD source overrides also produce2red specifically at old prompt resurrection, /tmp/uat367-original-alias-reset-red.log. This is relevant source evidence for UAT353; its full native Character/ordinary flow is not accepted by this repair.
- Catalog lmstudio/llamafile scopes:2red, /tmp/uat367-catalog-scope-red.log.
- Duplicate model IDs across providers:1red/1explicit-provider control passes, /tmp/uat367-duplicate-model-red.log.
- Actual dialog remount with retained query cache and legacy alias:2red, /tmp/uat367-reopen-red.log; internal-prefix-before-provider alias:1red/1control passes, /tmp/uat367-internal-alias-red.log.

The final DOM suite covers Save, Reset, remount, same-owner readiness, account invalidation, no cached-prompt restoration, Character/template preservation, and actual ordinary prompt assembly retaining exact instruction and PNG content. It stops before network/persistence; native SQLite/PostgreSQL acceptance remains with root.

## Verification

- 14 suites /239 tests pass, including25 new real-modal tests: /tmp/uat367-final-verified.log.
- Matched frontend TypeScript current versus HEAD content overrides:93 baseline diagnostics,93 current, no added/removed or touched-source diagnostics: /tmp/uat367-type-comparison.json.
- Configured ESLint:0errors and2existing warnings unchanged: /tmp/uat367-lint-comparison.json. Scoped diff whitespace check passes.
- Bandit was invoked; TSX is not supported. Recursive Settings-scope scan has0PythonLOC: /tmp/bandit_uat367_scope.json. No Python source is touched.
- Independent reviewer cleared the final source findings and passed6suites/46tests: /private/tmp/uat367-independent-final.log. Stage3/task remain In Progress until native SQLite/PostgreSQL first-send acceptance.

## Native candidate1 correction

Native SQLite010/013/014 and PostgreSQL009 under .tmp/uat-frontend-repair1-20260920 failed Save/reopen and actual image request again. Initial modal fixture seeded apiProvider=llama.cpp, unlike a fresh model store. With apiProvider undefined, the actual cockpit summary returns raw selected tldw:model and Playground3139 activates it; modal Save resolves the catalog canonical key and treats the internal prefix as a different provider. A read-only fixture override reproduces1inferred Save failure with1explicit-scope control passing: /tmp/uat367-providerless-red.log. Root approved recognizing the active raw selected-model scope when provider is absent, retaining foreign-provider boundaries. Add real caller/store/modal integration and actual image assembly with the fresh default before retrying native acceptance.

Candidate2 now recognizes the exact raw current selected-model key with no provider preference, while preserving provider-qualified separation. Real Playground/store/modal integration with the native model path reproduced the failure; two normalChatMode image assembly cases also omitted the instruction. Those3causal failures now pass. The composer surface in the caller test is reduced to dialog open/close, while Playground's scope effect, store and dialog are real. Its two modal renders need a test-specific15s timeout (observed7.9s); no product timeout changed.

Final candidate2 verification:15suites/272tests pass (/tmp/uat367-candidate2-final.log), including28modal/image cases. Matched frontend type diagnostics remain93→93 with no new/removed/touched diagnostics; expanded lint scope has0errors/4existing warnings unchanged. Whitespace check passes. Bandit Settings scan has0PythonLOC (/tmp/bandit_uat367_candidate2.json). Independent review and native SQLite/PostgreSQL candidate2 acceptance remain pending; source is frozen.

Independent candidate2 review is complete and clear: /root/review_uat363 passed50/50caller/dialog tests in /private/tmp/uat367-review-candidate2.log and verified raw-scope, explicit/foreign-provider, and owner controls. Only native candidate2 acceptance remains pending.

# UAT A10 flashcard prompt contract clarification

Related task: TASK-13260.278.18.83.45. Root recorded this new unit before edits and owns the parent task.

## Evidence and approved boundary

An actual five-card request returned five literal Q/A drafts; none were saved. UI, hook and service preserve source/count/type/provider/model. The actual adapter embeds value-bearing Q/A sample arrays in both legacy and planned system prompts. Source is supplied in the user message, within the existing 8000-character cap. Canonical chat/Ollama routing preserves both messages and the explicit target. The normalizer accepts nonblank fields and existing planned counts; real ClaimsEngine uses question/answer relationship units and source-only evidence. Model copying is an inference, and actual judge responses were not inspected; neither model-copy internals nor a verifier defect is proven.

Approved production scope: ONLY replace the two value-bearing sample prompt strings with JSON field/type descriptions requiring distinct source-supported questions and answers. No parser, blacklist, duplication rejection, verifier, count policy, source/target, request/wire, dependency, auth, storage or frontend change. Current source baseline SHA256 afc770de0650a2cc0f8dddc9ef825dac243f0584c61542e1af922cf34e307197.

The new endpoint-function tests call the actual endpoint, actual workflow generation adapter, normalizer, flashcard unit construction and ClaimsEngine. Only external generation/judge/NLI model seams are replaced. Independent fixed biology cards and judge verdicts do not depend on the new phrase. Tests establish an outgoing prompt contract, not actual model quality, full HTTP/auth acceptance, or saved catalog A10 completion.

ADR required: no. Existing [ADR-025](../ADR/025-llm-provider-adapter-routing-and-overrides.md) governs unchanged provider routing. No durable integration, wire or verification-policy decision changes.

## Stage 1: Causal prompt-contract regression
**Goal**: Establish the value-bearing sample requested by unchanged production.
**Success Criteria**: Outgoing real adapter prompt/source/count/target, legacy and planned types, and independent source-card response preservation are checked; real gate refuses NEI/refuted/empty responses.
**Tests**: New Flashcards module with normal configured Python 3.12 fixtures/plugins; outer 600 / per-case 300; at most 3 issue attempts. Red is prompt-contract evidence only.
**Status**: Complete

## Stage 2: Narrow prompt clarification
**Goal**: Replace only both sample instruction strings.
**Success Criteria**: Field/type descriptions and distinct source-derived questions/answers; existing source-only instruction, generation/parser/type/count/gate behavior stays intact.
**Tests**: New causal module and genuinely affected existing flashcard adapter/grounding scopes, avoiding prior frontend/diagnostic replay.
**Status**: Complete

## Stage 3: Scoped qualification and root handoff
**Goal**: Freeze reviewable source/tests with qualified static results.
**Success Criteria**: Scoped baseline/current Ruff, compile and Bandit including new tests; exact hashes, attempts and limitations; independent root review plus root-owned runtime handoff.
**Tests**: Existing activated Python 3.12 for tests/Ruff/compile; existing primary venv STATIC Bandit only. No runtime/browser/profile/log/auth/model/env/install/parent-task/Git/proof-file actions.
**Status**: In Progress

## Attempt record and oracle reassessment

Attempt 1 (normal configured Python 3.12, outer 600 / case 300): five genuine outgoing-prompt-contract reds at the copyable sample-array assertion, refuted/empty controls passed, one NEI oracle failure; total 6 failed / 2 passed, 4.465s outer elapsed (pytest 2.27s). No generation fixture is conditional on the new phrase.

Attempt 2: only the unexpected NEI control was diagnosed, unchanged production, 2.906s outer elapsed. Exact mismatch: actual failed vs expected needs_revision. Source decision-tree read explains it: front Q is an explicit factual unit, doc-only NEI with unsupported evidence produces hallucination; the relationship back is semantic-required and remains unverified. The artifact's failed verdict is stricter and still rejects 422. Correct only expected verdict, retaining status/code/no-grounded/source/gate controls. This is not a new production verifier defect or a causal prompt red.

Baseline Ruff's initial reporting wrapper failed after the configured cache attempted a protected worktree write; the diagnostic confirms that permission cause. Normal configured rules with --no-cache then report 0 findings. This is an operator/static-cache failure, not a behavioral test failure. Existing primary venv STATIC scanner verified Python 3.11.13 / Bandit 1.9.3; baseline touched production plus new tests has 0 findings/errors, 28 individually explained B101 assertion annotations and no broad nosec.

Only the two literal sample strings changed in production. A full parsed AST comparison restored just those values and confirmed identical remaining AST. No other owner change occurred between baseline hash and patch. Third bounded pytest invocation will be the final issue attempt, covering new tests and genuinely affected existing flashcard adapter/prompt behavior.

## Frozen qualification and coverage limits

Attempt 3 / final bounded behavioral invocation PASS: 16 passed, 0 failed/skipped; outer 4.031s / pytest 1.79s, 5 configured warnings (same count as the initial new-module run; details suppressed by normal repository addopts). Existing exact activated Python 3.12.13, normal configured plugins, outer 600 / per-case 300. Scope: new 8 plus existing adapter valid/different-types/legacy-generation-metadata/focus-topics/planned-count/planned-count-mismatch 6 and both existing source-only generation prompt cases 2. No prior frontend/diagnostic/HTTP/model controls were replayed.

Current Ruff with normal configured rules and --no-cache: 0 findings, matching baseline 0. In-memory Python 3.12 compile PASS for touched production/test. Existing primary venv STATIC Bandit current production+test 0 findings/errors, matching baseline 0; total 1023 AST lines, 28 individually explained B101 assertion skips, 0 broad nosec. No source security exception or suppression was added.

Frozen SHA256: tldw_Server_API/app/core/Workflows/adapters/content/generation.py c3a40621beeaa9d777cd728eb25a324feabdbad6c9eb2f0533ca3bfa82fdcdbd; tldw_Server_API/tests/Flashcards/test_flashcard_prompt_contract.py 484f23d21d5135c30181d6b7ca324cce71caad812203e7f7f37a9476dcea07b1. Only the two approved prompt string constants differ from the production baseline; exact-hash and full normalized AST guards confirmed scope. New tests do not condition generated cards/judge verdicts on production wording.

Root independent review requested at this freeze. Stage 3 remains In Progress for independent review and root-controlled load/live recovery. Source/tests are frozen. The test seam supplies an authenticated User directly to the real endpoint function: this establishes its generation/verification path, not full HTTP/auth/CSRF qualification or live model behavior. No production UI/backend runtime, browser/profile/auth/log/model/environment/install/task/index/commit/publication action occurred. Root's one corrected F-BIOLOGY five-card generation with prefilled provenance, subsequent review/save/review/export remains the actual A10 acceptance path; this prompt-contract qualification does not claim that acceptance.

## Independent root review

Root read both complete changed prompt strings and all eight complete new test bodies. Independent review CLEAR in the narrow outgoing prompt-contract scope: count/type/source/target/gates remain unchanged, and tests do not claim real model causal red/green. Source/tests remain frozen. Stage 3 stays In Progress for root's controlled combined backend load and one real F-BIOLOGY recovery/acceptance; no test or review replay was performed.

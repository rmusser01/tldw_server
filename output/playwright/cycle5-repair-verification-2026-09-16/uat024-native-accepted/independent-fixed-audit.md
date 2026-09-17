# Independent UAT024 native rejection audit — TASK13260.6

## Verdict

**PASS for the bounded native claim-verification rejection UX.** The same deterministic provider-boundary negative control still produces a real backend HTTP422 `claim_verification_failed` / `numerical_error`. The corrected UI retains the source/options, presents actionable guidance, exposes no generated draft/save control, and has no development runtime overlay in the retained final capture. This is not a real-model quality result or full UAT completion.

## Before / after evidence

| Observation | Before | Corrected |
| --- | --- | --- |
| Native generate request | 2026-09-16T23:47:54.104Z | 2026-09-16T23:57:42.851Z |
| Native response | HTTP422 at23:47:54.144Z | HTTP422 at23:57:42.904Z |
| Backend verdict | claim_verification_failed; numerical_error | Same |
| Source input | Original 10% sentence retained | Exact same sentence retained |
| Presentation | Visually inspected Next runtime overlay exposes raw report JSON | Visually inspected actionable inline source guidance, no overlay |

The corrected response reports one numerical error for the generated answer “The trial response rate was 90%.” against the provided source stating10%. It includes a failed unit `flashcard:1:back`, numeric mismatch rationale, and the original source evidence. The backend access log independently corroborates `POST /api/v1/flashcards/generate -> 422` at local16:57:42.882 (UTC23:57:42.882), line7630 of the owned `backend-1789602387491.private.log`. No raw runtime log or credentials are copied here.

The fixture uses the existing Mock OpenAI server and deliberately returns false generated content plus low-confidence verifier output through successful provider responses. Its retained log contains four HTTP200 chat-completion responses across the two native attempts. The prepared browser monitor only observes requests/responses/page errors; it does not fulfill or replace HTTP responses. The backend performs the actual generation/verification/rejection. This is documented fault injection at the provider boundary, not a fabricated application422.

## Corrected UI checks

`native-fixed-result.txt` records at23:58:01.461Z:

- Exact source retained: `UAT024 deterministic fixture source: The trial response rate was 10%.`
- Count1 and explicit fixture provider/model remain selected.
- Generate is enabled after rejection; save-control count0; dialog count0.
- Events contain the generate POST and its422, with no save request or pageerror in that observed interval.
- Body contains actionable source guidance and no generated-card review area. The screenshot independently shows retained source and form, inline guidance, enabled Generate, and no Next overlay.

The earlier screenshot visibly contains the Next runtime overlay even though its request observer also counted zero pageerrors. Therefore pageerror absence alone is not used as proof; the visual before/after comparison is essential. The final browser report still lists console errors/warnings, so this audit does not claim a clean console. “No draft” is scoped to the observed UI/no-save behavior; this audit did not query canonical DB state.

## Source and regression corroboration

Current disk bytes and retained review snapshots both match the author manifest:

- `useFlashcardQueries.ts`: `ea45ab82cd617752cc074c20f2c7e9ba01b4ff3d9b15ef9af6953b6381695e63`.
- `flashcards-generated-save-errors.test.tsx`: `7f3e3ff9a4c9c6ad0da7345ed4107bd5a7ffec25b2d010bfd68e186858b4c254`.
- Manifest SHA256: `2b93c4d9337f3e461154cbf0af50c6d2b1c6154ceab6a247d097948b3977422c`.

The backend working tree has no differences from `d9c76d616f` under `tldw_Server_API`. This verifies source consistency, not a separate runtime bundle attestation. The retained root independent default-config test receipt reports **26 tests /4 suites passed, zero skips**; this auditor did not rerun those tests or operate the browser.

## Artifact hashes

- native-fixed-result.txt: `a62e5bb1f0d87096ad5f05955ef5e825b2e60472a0c2792cfa08e2ed62d43710`
- native-fixed.png: `61f8f8660b8b5f0b59535c017382ac6624286dcd42e5ecb977c0b9c5a5b65d2b`
- native-rejection-result.txt: `e0f095840dbb3dde85830d7f62d378cacd97e3b424e55df78b28260069dc4bbc`
- native-rejection-overlay.png: `0317e6f7a53fb8f8f51058a238b11be6a00da21f8330249ad0f81cb3b3a0b92b`

## Limits

The provider/verifier fixture was still active for this accepted negative path. Subsequent real-model positive generation, temporary config restoration, broader account/profile coverage, and full UAT completion are outside this audit. No browser, runtime, provider/config, production source, DB, or tracker changes were performed; only this review artifact was written.

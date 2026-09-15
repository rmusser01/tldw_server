# Native Chat mirror and Cedar reuse checks — round 8

Targeted verification on the existing isolated multi-user profile, 2026-09-15 20:53–21:16 UTC. The runtime was restarted from merged product `267c00cab1` (docs HEAD `34ba5e82bc`); this is not a new fresh-install run.

## Results

- Normal Alice UI login, then the original Cedar Note loaded exact saved text and Chat origin. Alice is the credential-helper-selected identity; no separate post-login browser-principal response was captured. Independent Alice API reads are corroboration, not proof of browser identity.
- Canonical saved Robot entry hit the new UAT093 render-loop crash. Normal reload of the cleaned Chat URL recovered Robot and its original two messages.
- From that recovered Robot selection, the actual Notes row and More actions → Open conversation restored the original Cedar conversation, Cedar character4, both messages, and Save to Notes/Flashcards actions.
- Two full settled reloads retained the exact original assistant answer. Native read-only IndexedDB inspection found one owned two-message mirror with canonical server IDs, while preserving the old unowned incomplete/empty mirrors. The capture reports owner-marker presence only, not a new independent ownership validation.
- Save to Flashcards opened a required-question editor with the exact visible answer and Save disabled while the question was empty. One normal UI save created card15981a3e-ba31-4bf7-abaf-df5291fb7a2f at21:09:42UTC. Independent Alice GET200 confirms the exact pair, original conversation/message and supporting Note IDs; no save retry or duplicate card.
- Study began with the existing learning card followed by the new Cedar card. First Good200 established session2; explicit early End200 completed session2 with1card. The original Cedar answer was then revealed and rated Good200 in new session3. Automatic End200 completed session3 with1card. Settled reload reports Reviewed today3 (including the older session1) and all three completed one-card sessions.

## Open findings and limits

UAT093 is a separate saved-route-entry failure; recovery and passing Note backlink do not pass that entry path. Chat document title remains blank after reload and header Untitled despite named server metadata (existing058/062, TASK13260.34). The automatic Study discovery hint visibly obscures answer text (new094, TASK13260.35). Single-user native mirror acceptance and the full fresh matrices remain pending.

The one-line UAT093 production correction caused development hot reload during the later card-preparation sequence; the initial open editor closed before filling. Root reopened it after arranging a stable edit window. The Cedar backlink and both reload/IndexedDB checks precede that correction. Subsequent Study behavior was not changed during these checks.

## Capture qualifications

- `cedar-backlink-settled.txt` stops when answer text first appears: assistant label/character picker were still hydrating and its immediate title was empty. `cedar-backlink-actions-state.txt` is the later settled evidence for the correct character and both save actions.
- The save-response callback used a URL constructor unavailable in this CLI execution realm and failed after the actual save click. No browser save status/body was retained. Independent saved-row read and the subsequent native Study provide persistence evidence. The command was not blindly retried.
- `cedar-card-save-source-controls.png` shows the source answer/action popover, not the editor: the screenshot frame preceded the modal paint despite the completed input fill. Editor values and required-question state are in `cedar-card-review.txt`; do not mislabel that PNG as editor evidence.
- A compact More actions chip is replaced on hover; selecting the actual visible hovered action succeeded without a force click. An initial Show answer locator omitted its accessible shortcut suffix and timed out; the actual Show answer (Space) control worked.
- The failed plain-sandbox snapshot command could not resolve the npm registry. Resuming the same established browser with approved execution worked; no application failure is inferred.
- UI rating answer times reflect automation, not measured human recall. No fresh inference was submitted in this sequence.
- Native screenshots were visually inspected. JSON is preserved byte-for-byte. Copied text captures normalize trailing whitespace and the final newline for repository hygiene; original private captures remain unchanged.

All retained files are checked against known isolated-runtime credentials and JWT/private-key patterns, JSON is parsed, and SHA256SUMS indexes the evidence.

# `/chat` — NN/g Heuristic UX Audit #2

**Date:** 2026-09-19
**Branch:** `dev` @ `1dfdd819b6` (356 commits after audit #1)
**Surface:** `/chat` → `apps/packages/ui/src/components/Option/Playground/*`, shared by the Next.js WebUI and the browser extension
**Method:** Live walkthrough against a full stack, no mocking. Two isolated assessments (code review, deterministic measurement) plus an interaction walkthrough.
**Stack under test:** llama.cpp `:9099` (gemma-4-26B-A4B, real inference) → tldw backend `:8001` → Next.js WebUI `:8080`. Extension rebuilt from this commit and loaded in Chromium.
**Personas walked:** first-time user (cold, empty conversation) and experienced power user (model switching, slash commands, interrupt and recover, failure handling).
**Predecessor:** `Docs/Plans/2026-09-15-chat-page-nng-ux-audit.md`

---

## Scorecard — 16/40

| # | Heuristic | Score | Headline |
|---|-----------|-------|----------|
| 1 | Visibility of system status | 2 | Green "Healthy" chip during provider failure |
| 2 | Match system / real world | 1 | sidechannel, cockpit rails, `tldw:` byline, "Temp" |
| 3 | User control and freedom | 2 | Excellent stop/recover, but `?` was unusable |
| 4 | Consistency and standards | 1 | Three "shortcuts" controls, three panels |
| 5 | Error prevention | 2 | Unreachable model offered as usable |
| 6 | Recognition rather than recall | 2 | Copy and Regenerate are 0×0 until hover |
| 7 | Flexibility and efficiency | 3 | Slash commands, casual/pro, branching, thread search |
| 8 | Aesthetic and minimalist | 1 | 63 controls on an empty screen |
| 9 | Error recovery | 1 | One failure, two identical alerts, wrong component blamed |
| 10 | Help and documentation | 1 | Tour button dead, tutorial system unmounted on web |

Audit #1 scored the equivalent of 20/40. **The drop is better detection, not regression.** Both major new findings (the `?` defect and the unmounted tutorial host) were verified to exist at `59049e094e` and were missed. Measured against what audit #1 actually tested, the product moved forward.

---

## P0 — Every question mark typed in the composer was swallowed

**Fixed in this pass. `TASK-13263`, commit `ba6d8e60ba`.**

Typing `?` in the chat composer did nothing: the character was suppressed and a shortcuts panel opened. Verified three ways to exclude a test artifact.

| Input method | Result |
|---|---|
| `Shift+Slash`, the real keystroke | `"Is this ok"` — swallowed, dialog opens |
| Synthetic `type("?")`, no shift flag | `"Is this ok?"` — passes |
| Plain textarea injected into the same page | `"ok?"` — works |

**Cause.** `Playground.tsx` computed an editable-target guard at the top of its global key handler but applied it only further down, after the `Shift+?` branch had already called `preventDefault()`.

**Why the obvious fix was rejected.** Hoisting a blanket early return above the handler body would have broken two working behaviours: `Cmd/Ctrl+F` from the composer (which is focused on load, so in-thread search would become unreachable) and `Escape` from the thread-search input. A modifier chord cannot be produced by typing, so intercepting it in a text field is a product decision, not this bug. The fix guards the `?` branch alone.

**Why it shipped untested.** Synthetic key events do not set the shift flag the condition requires, so the branch never fired under test.

---

## Unfixed from audit #1, reproduced verbatim

| # | Finding | Task |
|---|---|---|
| 1 | "Take a quick tour" is a no-op; DOM byte-identical. Help modal host is mounted only when the layout is headerless, which is false for signed-in users on `/chat`. Three chat tutorials are dead code on web. | `TASK-13270` |
| 2 | Health chip read `Ollama / gemma3:1b Healthy` with two error panels for that provider on screen. Reports only local server liveness, polled every 30s. | `TASK-13264` |
| 3 | Errors blame "your tldw server" when the server returned a correct 502 and the provider was the unreachable hop. | `TASK-13265` |
| 4 | One failure renders two alerts, nine buttons, seven concepts, ~35% of the viewport. | `TASK-13266` |
| 5 | "Continue from partial" offered after a refusal with zero tokens received. | `TASK-13267` |
| 6 | Three "shortcuts" controls open three panels; `Shift+/` opens the page navigator, not what the panel documents. | `TASK-13271` |
| 7 | Model picker lists a refusing provider identically to a running one, with no health signal. | `TASK-13269` |
| 8 | Reasoning never renders. llama.cpp emitted 57 reasoning chunks in a 60-token stream; UI showed a placeholder for **15.5s** with zero disclosure elements. | `TASK-13274` |
| 9 | Copy/Edit/Regenerate measure 0×0 until hover, then 32×32. Two overflow menus per message; the persistent one is 37×22. | `TASK-13279` |
| 10 | Internal vocabulary intact: sidechannel, Context rail, Runtime rail, cockpit, Legacy sheet view, `tldw:` byline, "Temp". | `TASK-13280` |

---

## Genuinely fixed since audit #1 — do not regress

- `document.title` is `"Chat | tldw"` and tracks the conversation.
- Conversations auto-title from the first message; the history recall problem is solved.
- Dictation reports "Dictation unavailable" instead of offering voice input while its health check fails.
- Visible controls on the empty screen: 93 → 63.
- "Start chatting" now acts.
- No horizontal overflow at 390px.
- One contrast failure page-wide, down from three.
- "Still generating response (checkpoint 3)" replaced with plain language in the visible UI.

---

## New findings

| Finding | Evidence | Task |
|---|---|---|
| "Show technical details" expands to three words, "Stream completion failed" | Page text grew 21 characters | `TASK-13268` |
| Recovery chrome renders above the preserved answer | Notice at index 32, prose at index 345 | `TASK-13278` |
| At 400px the composer paints over the empty state | Heading clipped, sentence truncated, 3 controls centre-occluded and unreachable; toolbar wraps to 5 rows; model named twice | `TASK-13276` |
| Composer consumes 25.6% of viewport on desktop, 44–46% at 390–400px; page auto-enters focus mode unannounced below 768px; 768–1023px dead band renders mobile rails plus desktop toolbar | Measured | `TASK-13277` |
| Extension first-run wizard opens with a red error saying it needs a reachable server | Fresh extension build, Options page | `TASK-13272` |
| Sidepanel chat says "Finish setup to open Companion Home" | Fresh extension build | `TASK-13273` |
| "Start chatting" is 3.13:1 where AA needs 4.5:1; only contrast failure on the page | Pixel-sampled and computed | `TASK-13283` |
| Two targets below the 24×24 floor: "Take a quick tour" 118×16, "Select character or persona" 16×16 | Measured at 390×844 with touch | `TASK-13282` |
| Token counter is an unlabelled three-term formula; arithmetic is correct, labelling is not | `0 + ~22 = 22` → `635 + ~0 = 635` | `TASK-13281` |
| `500 GET /api/v1/audio/transcriptions/health` on every cold load; only failing request on the page | Network capture | `TASK-13285` |
| Backend version lookup still raises `PackageNotFoundError` from source, failing every provider call as 502 | Reproduced at runtime on this commit | `TASK-13284` |
| The "is the user typing" predicate is reimplemented 14+ times under 5 names; every other screen guards correctly, `/chat` was the one that drifted | Grep across the shared UI package | `TASK-13286` |

---

## Deterministic scan

Detector exit 2, two warnings. One production match (`border-t-4` on a rounded composer, reachable only in temporary-chat state), one false positive (an `<img>` inside a Jest stub). Regex matching over TSX flags classes behind runtime conditionals regardless of whether the branch renders.

---

## What is genuinely good — keep it

1. **Stop and recover.** Partial prose survives intact with four sensible recovery actions and clear copy.
2. **Slash commands.** Five commands, each with a title and a description line, discoverable from the empty-state hint.
3. **Accessible naming.** All 63 interactive controls carry accessible names; zero unnamed icon buttons.
4. **Collapsed rails leave a real labelled handle** with correct `aria-expanded` and `aria-controls`, rather than vanishing.
5. **The casual/pro split is structural**, changing layout, action visibility and composer size rather than toggling a preference.
6. **`noModelSelected` overrides a misleading green badge**, with the reasoning written into the source. The right instinct, applied narrowly.

---

## Recommended sequencing

**Phase 1 — truthfulness and input.** Days, tiny diffs.
`TASK-13263` (done) → `TASK-13284` → `TASK-13264` → `TASK-13265` → `TASK-13266` → `TASK-13267` → `TASK-13270`

**Phase 2 — density and language.** One to two weeks, layout only.
`TASK-13275` → `TASK-13280` → `TASK-13274` → `TASK-13279` → `TASK-13281`

**Phase 3 — narrow viewports and accessibility.** One week.
`TASK-13276` → `TASK-13282` → `TASK-13283` → `TASK-13271` → `TASK-13269` → `TASK-13272` → `TASK-13273` → `TASK-13277` → `TASK-13278` → `TASK-13285` → `TASK-13286`

Phases 1 and 2 should move the score from 16/40 to roughly 28/40.

---

## Reproduction environment

```bash
# llama.cpp (user-supplied) — http://127.0.0.1:9099, gemma-4-26B-A4B

# backend
TLDW_VERSION=0.1.0 .venv/bin/python -m uvicorn tldw_Server_API.app.main:app \
  --host 127.0.0.1 --port 8001

# frontend
cd apps/tldw-frontend && bun run dev -- -p 8080

# extension
cd apps/extension && bun run build:chrome   # then load .output/chrome-mv3 unpacked
```

`TLDW_VERSION` is still required because of `TASK-13284`. Config changed for the walkthrough: `tldw_Server_API/Config_Files/config.txt` points `llama_api_IP` at `:9099`.

All findings were observed against real inference. No mocks, no stubs, no fixtures.

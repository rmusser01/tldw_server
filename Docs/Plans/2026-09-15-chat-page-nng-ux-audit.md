# `/chat` — NN/g Heuristic UX Audit

**Date:** 2026-09-15
**Branch:** `dev` @ `59049e094e`
**Surface:** `/chat` → `routes/option-chat.tsx` → `components/Option/Playground/*` (shared `packages/ui`, used by both WebUI and extension)
**Method:** Live walkthrough against a full stack, no mocking.
**Stack under test:** llama.cpp `:9099` (gemma-4-26B-A4B, real inference) → tldw backend `:8001` (this branch) → Next.js WebUI `:8080`
**Evaluator perspective:** Sr. Designer / HCI, NN/g 10 usability heuristics + WCAG 2.2 AA
**Personas walked:** first-time user (cold, empty DB) and experienced power user (model switching, slash commands, interrupt/recover, history)

---

## Scorecard

| # | Heuristic | Score | Headline |
|---|-----------|-------|----------|
| H1 | Visibility of system status | **2.0** / 5 | Green "Healthy" badge shown *during* provider failure |
| H2 | Match between system & real world | **2.0** / 5 | MCP, OpenUI, sidechannel, cockpit rails, `tldw:model` |
| H3 | User control & freedom | **4.0** / 5 | Interrupt/resume is genuinely well built |
| H4 | Consistency & standards | **2.0** / 5 | Three "shortcuts" controls → three different panels |
| H5 | Error prevention | **2.5** / 5 | Offers unreachable models as "usable" |
| H6 | Recognition rather than recall | **2.5** / 5 | Hover-gated actions, unlabeled icons, all chats "Untitled" |
| H7 | Flexibility & efficiency of use | **4.0** / 5 | Slash commands and model favorites are strong |
| H8 | Aesthetic & minimalist design | **1.5** / 5 | 93 buttons on the default screen |
| H9 | Help recognize/diagnose/recover | **3.0** / 5 | Good copy, wrong diagnosis, duplicated twice |
| H10 | Help & documentation | **1.5** / 5 | The one onboarding button does nothing |

**Weighted average ≈ 2.5 / 5.** Same band as the prior `/document-workspace` (2.95) and `/world-books` (2.85) audits — but with a worse floor, because `/chat` is the app's front door and its first-run affordance is broken.

---

## The headline

`/chat` has a **strong engine and a confusing cockpit**. Interrupt-and-recover, slash commands, streaming, persistence and the model picker are all above the bar for a self-hosted LLM client — several are better than commercial equivalents. But the surface a first-time user meets is 93 controls deep, speaks in internal vocabulary, and its single "learn this" button is dead. Meanwhile the status system actively lies during failure, which is the most trust-corrosive class of bug in the set.

Nothing here requires re-architecture. The top 6 issues are small, local diffs.

---

## Blocking defect found before the walkthrough could start

### B1 — Every outbound LLM call fails when `tldw-server` isn't pip-installed 🔴

Not a `/chat` issue, but it blocked the whole walkthrough and will hit every dev running from source.

`tldw_Server_API/app/core/http_client.py:604` looks up the package version for the User-Agent:

```python
_CACHED_VERSION = _importlib_metadata.version("tldw-server")
except _HTTPCLIENT_NONCRITICAL_EXCEPTIONS:
```

`importlib.metadata.PackageNotFoundError` inherits `ModuleNotFoundError → ImportError`, which is **not** in `_HTTPCLIENT_NONCRITICAL_EXCEPTIONS` (`http_client.py:130-146`). Verified:

```
MRO: PackageNotFoundError → ModuleNotFoundError → ImportError → Exception
caught by tuple? False
```

So the exception escapes, the HTTP client never builds, and *every* provider call dies as `502 provider_unavailable`. The UI correctly reports "provider unavailable" — the provider is fine.

**Fix (one line):** add `_importlib_metadata.PackageNotFoundError` (or plain `ImportError`) to the tuple. The `pyproject.toml` fallback immediately below already handles the miss correctly — it just never gets reached.

**Workaround used for this audit:** `TLDW_VERSION=0.1.0` short-circuits the lookup.

---

## Issues, ranked

Severity uses NN/g's 0–4 scale (4 = usability catastrophe).

### 1. "Take a quick tour" does nothing — Severity 4 🔴 [H10, H1]

The only onboarding affordance on the empty state. Clicking it produces **zero change**: DOM byte-identical (83,082 → 83,082), no dialog, no overlay, no navigation, no console error.

```
tour enabled: true visible: true
BEFORE: {"dialogs":0,"html":83082}
AFTER : {"dialogs":0,"html":83082,"overlays":0}
```

A new user's one deliberate act of asking for help is silently swallowed. Also fails WCAG 2.2 AA 2.5.8 at 118×16px on mobile.

**Solution:** Wire it to a real tour, or remove it until one exists. A dead help button is worse than no help button — it teaches users the product's help is broken. Shortest honest fix: point it at the existing keyboard-shortcuts dialog + a 4-step coachmark over composer → model chip → slash commands → history.

---

### 2. "Healthy" badge displayed during provider failure — Severity 4 🔴 [H1]

Switched to `gemma3:1b` (Ollama configured, not running) and sent a message. The screen showed two error panels — and the status chip directly beneath them read:

> **Ollama / gemma3:1b** `Healthy`

The single element whose entire job is to report provider state was contradicting two error panels about that exact provider, on the same screen, at the same time.

**Solution:** The chip must consume the same result the request did. On any provider error, drive it to `Unreachable` / `Degraded` immediately and hold until a probe succeeds. If health is cached, show staleness ("checked 4m ago") rather than asserting current health. A stale-but-honest badge beats a fresh-looking lie.

---

### 3. Errors blame the wrong component — Severity 3 🟠 [H9]

Copy reads:

> "Something went wrong while talking to **your tldw server**."
> "Try again in a moment, or open Health & diagnostics to inspect **server health**."

The tldw server was healthy and responded correctly — it was the **Ollama provider** that was unreachable. The user is dispatched to debug the one component that was working.

**Solution:** Name the actual failing hop. "Couldn't reach **Ollama** at `127.0.0.1:11434`" plus the real next action ("Is Ollama running? `ollama serve`"). The backend already distinguishes these — `provider_manager` logged `Provider llama.cpp failure recorded` — the specificity is lost in the UI layer.

---

### 4. Two error surfaces, two vocabularies, one failure — Severity 3 🟠 [H4, H8]

The same failure rendered **twice simultaneously**, with different action sets:

| Inline bubble | Composer banner |
|---|---|
| Retry same model | Retry chat |
| Switch model | Switch provider |
| Try provider fallback | Edit provider |
| Continue from partial | View in Health & Diagnostics |
| | Close |

Nine buttons, seven distinct concepts, for one connection refusal — consuming roughly 40% of the viewport. "Switch model" vs "Switch provider" and "Retry same model" vs "Retry chat" force the user to reason about whether the difference is meaningful.

**Solution:** One error surface — the inline bubble, since it's anchored to the turn that failed. Keep three actions: **Retry**, **Switch model**, **Diagnose**. Delete the composer banner.

---

### 5. "Continue from partial" offered when there is no partial — Severity 2 🟡 [H1, H5]

Shown on a connection failure where zero tokens were received. (It is correct and genuinely useful after a *stop* — see Strengths.)

**Solution:** Gate on `partialLength > 0`.

---

### 6. Three "shortcuts" controls open three different panels — Severity 3 🟠 [H4, H6]

Verified by probing each by exact accessible name:

| Control | Opens |
|---|---|
| **"Show shortcuts"** (left rail) | Page navigator — "Search pages…", ⌘1–⌘8 destinations |
| **"Show keyboard shortcuts"** (top bar) | The real keyboard reference (⌘K, Shift+Esc, Ctrl+e, Alt+w) |
| **"Shortcuts"** (chat toolbar) | A *third*, chat-specific panel |
| **Shift+/** — the binding the toolbar button's own tooltip advertises | The **page navigator**, not the panel that documents it |

And the keyboard reference lists its own binding as `⌘+Shift+?` — a third value for the same action.

**Solution:** One name per concept. Rename the rail control **"Go to page"** (it's navigation, not shortcuts), keep **one** keyboard-shortcuts dialog on one binding, and fold the chat-specific panel into it as a section. Fix the tooltip to match the binding that actually fires.

---

### 7. Model picker presents unreachable models as usable — Severity 3 🟠 [H1, H5]

`gemma3:1b` appeared under the heading **"Usable configured models"** with no health indicator, while Ollama was down. "Configured" is being presented as "usable". The user discovers the truth only after composing and sending.

The panel is also clipped at the viewport bottom with no scroll affordance, and its own tooltip overlaps the last row.

**Solution:** Per-row health dots using the probe that already exists for the active model. Sort unreachable to the bottom, dim them, and label the section **"Configured models"**. Give the list a max-height with visible overflow.

---

### 8. 93 buttons on the default chat screen — Severity 3 🟠 [H8]

Measured on first load, before any conversation exists. The first-time user simultaneously faces:

- Top bar: 10 controls (Search, Temp, Character, Share, Settings, Notifications, theme, help…)
- Left rail: 13 icon-only destinations
- Two chip rows: "Standard chat", "Conversation timeline" / "Focus", "Shortcuts", "Artifacts panel closed"
- Composer toolbar: Modes, MCP, Search & Context, Prompt, Role-play setup, Buddy & Persona, OpenUI, mic, attach, 2× sliders, gauge
- Status row: provider, health, tokens, Saved, Advanced controls
- Edge tabs: "Context rail" and "Runtime rail" on both margins

The actual first-run task — type a sentence, press Enter — is one control.

**Solution:** Progressive disclosure, same approach approved for `/world-books`. Default (first-run) shows composer + model chip + attach + slash hint. Everything else moves behind a single **"Tools"** disclosure, with the full cockpit restored by a persisted "Advanced" toggle. Returning users keep whatever they last had open. This is the highest-leverage change in the audit and is layout-only.

---

### 9. Internal vocabulary on the default surface — Severity 3 🟠 [H2]

Visible without opening anything: `MCP`, `OpenUI`, `Buddy & Persona`, `Context rail`, `Runtime rail`, **"Restore context sidechannel"**, "Artifacts panel closed", "Cockpit rails hidden", "Still generating response (**checkpoint 3**)", **"Legacy sheet view"**, and the assistant byline **`tldw:gemma-4-26B-A4B`**.

Two specific traps:

- **"Temp"** (top bar) means *temporary chat* — but in an LLM UI "Temp" reads as **temperature**. Genuine misparse risk, and the two are both plausible chat settings. Rename to **"Temporary"**.
- **"Restore context sidechannel"** has no plain-English meaning to any user.

**Solution:** Plain-language pass. `tldw:gemma-4-26B-A4B` → `gemma-4-26B-A4B`. "Context rail" → "Sources". "Runtime rail" → "Model settings". "checkpoint 3" → elapsed time. Keep `MCP` (it's a real standard) but add a tooltip gloss on first encounter.

---

### 10. Reasoning output hidden while the user waits — Severity 3 🟠 [H1]

The model emits `reasoning_content` (confirmed directly against llama.cpp). The UI receives it — `utils/streaming-chunks.ts:169` reads `delta.reasoning_content` — and `components/Common/Playground/ReasoningBlock.tsx` exists to render it with a "Thinking…" disclosure.

It never activated. Sampled every 1.5s through a full generation:

```
0.0s: Generating response... start
4.5s: Still generating response (checkpoint 3)
...
10.5s: Still generating response (checkpoint 3)
Saw "Thinking…" / reasoning disclosure? false
```

Ten-plus seconds of a spinner while the model is actively producing text the UI already has in hand.

**Solution:** Connect the streaming `reasoning_content` path to `ReasoningBlock`. The component and the data are both already there; only the wiring between them is missing. This converts the worst dead-air in the product into visible progress.

---

### 11. Message actions are invisible until hover — Severity 2 🟡 [H6, H7]

Copy / Edit / Regenerate render as `hidden group-hover:flex group-focus-within:flex` — computed width `0`, absent from the accessibility tree until hovered or focused. Only `•••` is always visible.

This inverts priority: the two most frequent actions in any chat client (copy, regenerate) are hidden, while the rarely-used overflow menu is permanent. On touch, tapping a message does reveal them (verified on a 390×844 touch context) — but nothing signals that tapping does anything.

**Solution:** Show Copy and Regenerate persistently on the last assistant message; hover-reveal is fine for older turns. Keep `•••` for the long tail.

---

### 12. Conversations are never auto-titled — Severity 3 🟠 [H6]

After a complete exchange the conversation is still **"Untitled"**. Every saved chat lands in history under the same label, so the history list is unusable for recall by the time a power user has ten chats.

**Solution:** Auto-title from the first user message (truncated) immediately, then optionally refine with a cheap model call. Keep it inline-editable.

---

### 13. Mobile layout breaks down — Severity 3 🟠 [H8, H4]

At 390×844 with touch:

- **Composer + toolbars consume ~50% of the viewport**, squeezing the conversation into the top third
- **Toolbar wraps into 5 ungrouped rows**; grouping logic doesn't survive the wrap. Two different unlabeled slider icons end up in separate rows
- **Overlap:** the "Model gemma-4-26B-A4B" chip collides with the floating "N" badge, occluding the word "Model"
- The user avatar is clipped at the right edge
- The page **auto-enters Focus mode** unannounced ("Exit focus" appears without being requested)
- Large dead band between the last message and the composer

Credit: **no horizontal overflow** (`scrollWidth 390 === innerWidth 390`).

**Solution:** On mobile collapse the toolbar to 4 primary actions plus one overflow sheet; never wrap past two rows. Fix the z-index collision on the badge. Announce or remove the automatic Focus-mode entry.

---

### 14. Tap targets below WCAG 2.2 AA — Severity 3 🟠

17 controls under 44×44 on mobile; three under the 24×24 AA floor (2.5.8):

| Control | Size |
|---|---|
| "Select character or persona" | **16 × 16** |
| "Take a quick tour" | 118 × **16** |
| "Skip to main content" | 1 × 1 |

**Solution:** 24px minimum hit area (44px preferred) via padding — no visual size change needed.

---

### 15. Silent failing requests on every session — Severity 2 🟡 [H1]

- `500 GET /api/v1/audio/transcriptions/health` — fires on every load. The **mic button is offered anyway**, so voice input is presented as available while its own health check is failing.
- `404 GET /api/v1/chats/{id}/settings?scope_type=global` — fires on every new chat, for a conversation not yet persisted server-side.

Neither is surfaced; both are pure console noise that masks real errors during debugging.

**Solution:** Fix the 500 (or stop probing when STT is unconfigured) and disable the mic with a reason tooltip when it fails. Skip the settings fetch until the chat exists server-side.

---

### 16. Smaller items — Severity 1–2 🟡

| Issue | Detail | Fix |
|---|---|---|
| **"Start chatting" is a no-op** | Composer already focused before *and* after the click; empty state unchanged. Also **3.13:1** contrast (AA needs 4.5:1) | Make it insert a starter prompt, or delete it |
| **Empty `<title>`** | Document title is `""` — tab and screen-reader identification | Set `Chat · tldw` |
| **Token counter** `0 + ~0 = 0 tokens` | Unlabeled three-term formula; operands never explained | `~0 tokens (prompt 0 + reply 0)` or tooltip |
| **Stop before first token** | Assistant bubble vanishes entirely — no record the turn happened | Keep the bubble with "Stopped before any response" |
| **Recovery banner above content** | After stop, error chrome renders *above* the partial answer | Content first, recovery beneath |
| **"Saved" appears twice** | Top bar badge and composer status row | Keep one |
| **Desktop dead space** | ~350px void between conversation and composer | Let the thread grow downward |
| **9–10px badge text** | "Healthy", "Saved" | 11–12px minimum |

---

## What is genuinely good — keep it

Reported so these aren't lost in refactor:

1. **Slash commands.** `/search`, `/web`, `/vision`, `/generate-image`, `/model` — each with a title *and* a description line, discoverable from the empty-state hint. Better than most commercial clients.
2. **Interrupt and recover.** Stopping mid-stream preserves the partial text and shows "Generation was interrupted. You can retry, switch model, or continue from the partial response." with four recovery actions. Verified the partial prose survives intact. Genuinely excellent H3 work.
3. **Message accessibility semantics.** Messages are focusable with `"Assistant message 2 of 2"` / `"User message 1 of 2"` labels and visible focus outlines. Well done.
4. **Persistence.** Full conversation restored across reload, no user action required.
5. **Model picker mechanics.** Search, provider filter, favorites (stars), "Current" grouping.
6. **Error copy tone.** Plain language, no stack traces, with "Show technical details" progressive disclosure — the *tone* is right even where the *attribution* is wrong.
7. **Contrast.** Only 3 flags across the whole page; the dark theme largely passes AA.
8. **Small touches.** "Draft saved", skip-to-main-content link, aria-labels on every rail icon, `Scroll to latest messages`.

---

## Recommended sequencing

**Phase 1 — Truthfulness (days, tiny diffs).** Highest trust-per-line-changed.
1. Fix `PackageNotFoundError` (B1) — one line, unblocks every from-source dev
2. Health chip must reflect the last real request outcome (#2)
3. Name the failing provider in errors (#3)
4. Delete the duplicate composer error banner (#4)
5. Gate "Continue from partial" (#5)
6. Fix or remove "Take a quick tour" (#1)

**Phase 2 — Density & language (1–2 weeks, layout-only).**
7. Progressive disclosure of the composer toolbar (#8)
8. Plain-language pass incl. "Temp" → "Temporary" (#9)
9. Wire `ReasoningBlock` to the streaming path (#10)
10. Auto-title conversations (#12)
11. Persistent Copy/Regenerate on the last message (#11)

**Phase 3 — Mobile & a11y (1 week).**
12. Mobile toolbar collapse + z-index fix (#13)
13. Tap target minimums (#14)
14. Shortcuts consolidation (#6)
15. Model health dots (#7)
16. Silent request failures (#15)

Phase 1 + 2 alone should move the weighted average from **2.5 → ~3.6**.

---

## Reproduction environment

```bash
# llama.cpp (pre-existing, user-supplied)
#   http://127.0.0.1:9099  — gemma-4-26B-A4B-it Q4_K_M

# backend (this branch)
TLDW_VERSION=0.1.0 .venv/bin/python -m uvicorn tldw_Server_API.app.main:app \
  --host 127.0.0.1 --port 8001

# frontend
cd apps/tldw-frontend && bun run dev -- -p 8080
```

Config changed for the walkthrough (revert if unwanted):
- `tldw_Server_API/Config_Files/config.txt` — `llama_api_IP = http://127.0.0.1:9099`, `llama_model = gemma-4-26B-A4B` (backup at `scratchpad/config.txt.bak`)
- `tldw_Server_API/Config_Files/.env` — created, `AUTH_MODE=single_user`
- `apps/tldw-frontend/.env.local` — created, points at `:8001`

All findings above were observed against real inference — no mocks, no stubs, no fixtures.

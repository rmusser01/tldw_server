# `/chat` Page — Senior UX UAT Review

**Date:** 2026-06-13
**Reviewer role:** Senior UX specialist (User Acceptance Test)
**Build under test:** dev branch (`origin/dev` @ `fd8d6d5c38`), Next.js 16.1.4 frontend on `:8080`, FastAPI backend on `:8000` (single-user mode, 26 providers configured incl. OpenAI/Anthropic/Google).
**Method:** Live browser, driven through real Chromium via Playwright (`apps/tldw-frontend/scripts/chat-uat-driver*.mjs`). Synthetic single-user auth seed mirrors `e2e/smoke/smoke.setup.ts`. Personas: **first-time user** and **power/experienced user**.
**Artifacts:** screenshots + raw observations in `./assets/chat-uat-2026-06-13/`.

---

> **Correction (post-review, 2026-06-13):** An earlier draft of this report stated that `/chat` "cannot send for any first-time user" and blamed a wrong data source (`/config/providers` vs `/llm/providers`). Deeper investigation disproved that. The chat page sources models from `/api/v1/llm/models/metadata` (which the backend serves correctly — 843 KB of models), and the empty-picker state is **intermittent**, not deterministic. Root cause and severity are corrected below. The original wording is retained only in git history.

> **Implementation status (2026-06-13):** Three findings are fixed on branch `uat/chat-page-review` with tests (see the remediation plan for detail):
> - **#1** — model picker now shows a "Loading models…" state during the slow catalog fetch instead of the false "No models available. Connect your server in Settings." error.
> - **#4** — Escape now dismisses the shortcuts help panel (a global capture-phase listener was swallowing Escape; fixed with a capture-phase handler).
> - **#6** — concurrent identical GETs on load are now coalesced in `bgRequest` (`/users/me/profile` 5→1, `/config/providers` 3→1, `/characters` 2→1, `/persona/catalog` 2→1). `fetchWithAuth` endpoints (`/persona/profiles`, `/notifications`) remain a follow-up.
> Findings #2, #3, #5, #7, #8, #9 remain open.

## Summary

**First-time user:** Mostly works, with one intermittent but serious failure. On some loads the model picker populates and chat works end-to-end; on other loads (reproduced with both a hand-rolled seed and the project's own `seedAuth` helper) the picker shows **"No models available. Connect your server in Settings,"** the model-metadata endpoint is never called, and the first Enter is rejected with "Please select a model." A page reload reliably recovers it. The failure is a mount-time race (details in Finding #1): the model fetch runs before the client config is ready, returns empty **without a network call, without an error, and without retrying**. The failure UX is poor — silent, with a contradictory green "Healthy" badge and a dead-end "Open current chat settings" link. Separately, the empty state presents three dense rails of insider terminology ("cockpit", "sidechannel", "composition", "runtime", "MCP tools", "context stack") before any interaction — high cognitive load for a "Start a new chat" screen.

**Power/experienced user:** Strong once models load. The page is feature-dense and well-instrumented: a three-rail cockpit (Context / Runtime), composition preview, artifacts panel, conversation branching, compare mode, role-play, MCP tools, research context, a clean and discoverable tools popover (image-gen, OCR, web search, compare), a rich keyboard-shortcut set, and 44px minimum tap targets. The app was stable throughout — **0 console errors, 0 page errors, 0 failed API requests** across all flows. The main frustrations are the intermittent model-load race above, an overlay panel that ignores Escape, and heavy redundant fetching on every load (see Finding #6).

---

## What works well (keep)

- **Stability.** No uncaught JS errors, no error boundaries, no failed API calls across empty state, model selector, tools popover, composer toggles, shortcuts, send attempt, and mobile reflow.
- **Tools popover** (`tools-button`) is genuinely good: clear grouping of Generate image / Manage in Knowledge / Use OCR / Enable web search / Simple web search / Default-on-for-new-chats / Compare mode. Discoverable and labeled.
- **Keyboard support.** Logical Tab order out of the composer (Send → delivery options → dismiss → collapse rails) and a deep shortcut set (Shift+Esc focus composer, Cmd/Ctrl+F search thread, Shift+/ open shortcuts, Alt+Shift+A artifacts, Alt+Shift+C compare, Alt+Shift+M mode launcher, Alt+Shift+←/→ response variant, Alt+Shift+B/R branch).
- **Accessibility primitives.** Model selector exposes `aria-haspopup="listbox"`, `aria-expanded`, a 44px min hit area, and a status tooltip.
- **Mobile parity.** Composer and send control remain visible; rails collapse into Context/Runtime tabs rather than disappearing.
- **Power-user depth.** Composition preview, runtime inspector, raw-request modal, session insights, cost/token estimation, branching, and compare give experienced users real transparency.

---

## Findings

| # | Feature / Area | Persona | Issue | Severity | Recommendation |
|---|----------------|---------|-------|----------|----------------|
| 1 | Model load race | Both | **Intermittent:** on some `/chat` loads the model picker is empty ("No models available"), `/api/v1/llm/models/metadata` is never called, and the first Enter is rejected with "Please select a model." Reproduced with the project's own `seedAuth`; a reload recovers it. Root cause: the mount-time fetch (`Playground.tsx:301 refreshCharacterChatModels` → `TldwModels.getModels`) hits the gate at `TldwModels.ts:217` `if (!isConfiguredForModels(config)) return cachedModels || []` when `tldwClient.getConfig()` hasn't resolved yet — returning empty with **no network call, no error, and no retry**; the cooldown then keeps it empty until reload. | **High** (intermittent, recoverable by reload) | Re-fetch models when config becomes ready (listen for `tldw:config-updated`, or await config in the mount effect); do **not** cache the "unconfigured" empty result; surface a truthful "connecting…/not connected" state instead of silent empty. |
| 2 | Failure-state UX | First-time | When Finding #1 hits, the only inline recovery — **"Open current chat settings"** — produces no visible result (`14-chat-settings.png` unchanged), so the user appears stuck. | Medium | Make "Open current chat settings" open a working inline model/provider config, and add a retry/reconnect affordance to the empty picker. |
| 3 | State consistency | Both | During the Finding #1 failure, the model selector shows a green **"Healthy"** badge while the list says "No models available" and the right rail reads "No model selected" / "Unavailable." Provider-status (`/config/providers`) and model-availability (`/llm/models/metadata`) are independent paths that can disagree. | Medium | Reconcile the two into one truthful "ready to chat?" signal; don't show "Healthy" when no model is selectable. |
| 4 | Shortcuts help panel | Power | **Escape does not dismiss** the panel (`before=true, afterEsc=true`). Only the explicit "Close" button works. | High | Add Escape-to-close (and click-outside) for the shortcuts panel and other dismissible overlays; standard overlay expectation. |
| 5 | Empty-state cognitive load | First-time | Three rails of insider jargon (cockpit / sidechannel / composition / context stack / runtime / MCP tools) are fully expanded before any interaction. | Medium | Progressive disclosure: composer-first empty state, collapse rails by default for new users, reveal cockpit on demand. |
| 6 | Redundant fetch | Both (perf) | `GET /api/v1/config/providers` (13 KB) fires **3×** on a single `/chat` load. | Medium | Deduplicate/cache via a single shared query (React Query `staleTime`/shared key). |
| 7 | Terminology | First-time | "Cockpit", "sidechannel", "composition", "artifacts", "runtime" don't map to familiar chat concepts. | Medium | Plain-language labels or first-run tooltips ("Context", "Model & settings", "Tools used"). |
| 8 | Empty-state CTA | First-time | When Finding #1 hits, the primary CTA "Start chatting" doesn't resolve the missing-model state; clicking it still can't send. | Low | Gate/relabel the CTA on model readiness, or have it trigger model selection/retry first. |
| 9 | Mobile density | Both | Mobile renders the full desktop cockpit (Context/Runtime tabs + composition card + shortcuts). Functional but heavy on a 390px viewport. | Low | Composer-first mobile layout; defer cockpit rails behind a single "Details" affordance. |

---

## Top 5 prioritized fixes

1. **Fix the intermittent model-load race (#1).** Make the mount-time model fetch await config readiness / re-fetch on `tldw:config-updated`, stop caching the unconfigured-empty result, and show a truthful connecting/not-connected state. *Rationale: highest user impact when it hits, and the failure is currently silent and unrecoverable without a reload. Confirm the exact root cause with a reliable repro first (it is timing-dependent).* 
2. **Escape-to-dismiss overlays (#4).** Start with the shortcuts panel; audit other dismissible panels. *Rationale: deterministic, confirmed, cheap; high-frequency expectation for power users.*
3. **Cut redundant fetching (#6).** `/users/me/profile` ×5, `/persona/profiles` ×4, `/config/providers` ×3, `/characters/` ×2, `/notifications` ×2 on a single load. *Rationale: deterministic, confirmed; meaningful load/perf win.*
4. **Reconcile readiness state (#3) and failure-state UX (#2).** One truthful "ready to chat?" signal; make the empty-picker recovery path actually work. *Rationale: removes the contradictory "Healthy" + dead-end that make #1 feel worse.*
5. **Reduce empty-state load (#5 + #7).** Composer-first empty state, collapse cockpit rails by default, plain-language labels. *Rationale: lowers the first-run cliff without removing power-user depth.*

---

## Coverage & limitations

- **Covered live:** initial load / empty state, model selector (open + menu contents), tools popover, composer advanced/options toggles, shortcuts panel (open + Escape behavior), typing + send attempt, keyboard Tab order, mobile reflow (390px), console/page/network error capture, the `/config/providers` and `/llm/models/metadata` payloads, and a reload-recovery test of the model picker.
- **Could not exercise end-to-end during this pass:** on the runs where Finding #1 hit, deeper flows that require an active conversation — character chat, role-play, voice mode, compare mode, branching, artifacts, image generation — were not reached. On runs where models loaded (and after a reload) these are reachable and should be covered in a follow-up pass now that the model loads reliably via reload.
- **Diagnostic note:** Finding #1 is **timing-dependent** — it reproduced with both a hand-rolled localStorage seed and the project's own `seedAuth`, but a subsequent run loaded models on first paint (`/llm/models/metadata` was called). So it is a real race, not a deterministic block and not purely a seeding artifact. The exact trigger (config-readiness vs. backend cold-start) should be pinned with a reliable repro before the fix.

## Reproduction

```bash
# Backend already running on :8000 (single-user, .env configured).
# Frontend (dev branch worktree):
cd apps/tldw-frontend
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE=advanced bun run dev -- -p 8080
# Drive the review:
node scripts/chat-uat-driver.mjs     # full walkthrough + screenshots
node scripts/chat-uat-driver2.mjs    # model-select + send + Escape-dismiss
```

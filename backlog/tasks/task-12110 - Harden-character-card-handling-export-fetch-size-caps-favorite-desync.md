---
id: TASK-12110
title: Harden character card handling (export SSRF, size caps, favorite desync, variant server-id)
status: Done
labels:
- bug
- medium
- character
- packages-ui
documentation:
- apps/FRONTEND_AUDIT.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**Severity: Medium.** Round-2 audit findings R9 + R10. Paths under apps/packages/ui/src/.

- **PNG export fetches an attacker-controlled URL** — `utils/character-export.ts:269-275` (via `useCharacterCrud.tsx:477`) does `await fetch(record.avatar_url)` when `image_base64` is absent. `avatar_url` is a card field an attacker fully controls via a shared/imported card, so exporting that character fires an outbound request from the victim's browser (tracking beacon / internal-network probe), with no `AbortController`/timeout and no response-size guard.
- **No size caps** — `AvatarField.tsx:118-190` avatar upload has no `file.size` check (unlike `CharacterSelect.tsx:113` `MAX_PERSONA_IMAGE_BYTES`); a huge image autosaves its base64 to localStorage every 30s → `QuotaExceededError`. `Characters/utils.ts:516` + `TldwApiClient.ts:4559` import read the whole file + synchronous `JSON.parse` on the UI thread with no size limit → tab freeze on a large card.
- **Two unsynchronized favorite systems** — `CharacterSelect.tsx:1175-1196` writes favorites to localStorage; `useCharacterCrud.tsx:899-980` writes server-side `extensions.tldw.favorite`; the Manager filter/star use the server flag, the header uses localStorage → persistent desync. Plus the optimistic favorite update writes the wrong React-Query cache key (`:928-936`).
- **Variant server-id (R10)** — `utils/message-variants.ts:75` `applyVariantToMessage` keeps the prior `serverMessageId` when swiping to a not-yet-persisted variant, so a later edit/delete targets the wrong server row.
- Lower: undo-restore guesses `version + 1` (`useCharacterCrud.tsx:721`); soft-deleted row stays interactive for the 10s undo window; unsafe PNG chunk-length parse in the dead `readCharacterFromPNG` (`character-export.ts:359`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 PNG export does not fetch an arbitrary `avatar_url`; either only embed already-local base64, or fetch same-origin/allowlisted only, with an AbortController timeout and a response-size cap.
- [ ] #2 Avatar upload and card import enforce size caps (reuse `MAX_PERSONA_IMAGE_BYTES` / add an import byte cap) and fail gracefully instead of freezing the tab / breaking localStorage autosave.
- [ ] #3 A single source of truth for "favorite" (reconcile localStorage vs server), and the optimistic update targets the correct query key.
- [ ] #4 A swiped, not-yet-persisted variant does not inherit the prior message's `serverMessageId`; edit/delete target the displayed variant's server row (or are disabled until it persists).
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

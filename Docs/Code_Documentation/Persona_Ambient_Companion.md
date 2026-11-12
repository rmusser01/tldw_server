# Persona Ambient Companion

Persona Ambient Companion Stage 1 adds deterministic, idle-only behavior to
the existing raster Persona Buddy. The engine consumes reviewed pack metadata,
browser state, and the current visual intent. It has no LLM/model client and no
arbitrary network transport. Protected raster loading remains a separate,
bounded authenticated service used by the renderer.

## Modes and surfaces

- **Off** disables scheduled ambient actions. Authored click, Space, touch, and
  drag reactions remain available when direct interaction is otherwise safe.
- **Expressive** schedules authored non-moving ambient variants while idle.
- **Roaming** schedules expressive variants and authored horizontal movement.
  Roaming is available only on the full web surface. A sidepanel request is
  safely coerced to Expressive; unsupported surfaces suspend ambient behavior.

Stage 1 never moves vertically and never changes the persisted anchor for an
ambient frame. Roaming is a transient horizontal offset, clamped to the current
viewport bounds. Only an explicit user drag writes the persisted position.

## Preference resolution

The effective mode resolves in this order after both reads succeed:

1. The focused Persona's stored override, when present.
2. The global Buddy preference.
3. Expressive as the default for successfully read but unstored preferences.

A missing Persona override is normal data and falls through to the global
preference. A failed or still-unresolved global/Persona read is different: the
runtime fails closed to Off. Persona changes fence stale reads and mutations,
clear the old behavior timer, and start a full fresh interval after the new
identity is ready.

Preference writes are versioned. Optimistic UI updates retain the previous
version, stale writes conflict, and failed writes refresh or roll back without
letting an earlier Persona request overwrite the current Persona.

## Idle eligibility and precedence

Scheduled ambient behavior is eligible only when the semantic visual state is
`idle`, the document is visible, Buddy controls are closed, focus is not within
the controls, no drag is active, reduced motion is off, and the surface is
supported. Hidden time does not count toward the next interval. Returning from
a semantic state, hidden tab, controls, focus, drag, or reduced motion starts a
fresh interval rather than replaying an overdue action.

Visual intent wins in this order:

1. Error or recovery.
2. Approval needed.
3. Offline.
4. A valid, unexpired runtime override declared by the active pack.
5. The highest-priority matching authored trigger.
6. Active tool work.
7. Wake armed.
8. Listening, thinking, speaking, or another normalized live voice state.
9. Direct interaction.
10. Ambient behavior.
11. Base idle.

Runtime leases use the same safety principle: error, approval, offline, wake,
listening, thinking, speaking, and tool/custom semantic intent preempt ambient
work. Incoming visual overrides are bounded and the current Host caps their
duration at 30 seconds, clearing expiry on its one-second maintenance interval.
Tests that prove semantic resume should emit idle before that cap expires, then
measure the new interval.

## Scheduling and behavior metadata

Ambient intervals are randomized locally between 30 and 90 seconds. Selection
uses declared weights, respects cooldowns, and avoids an immediate repeat when
another eligible entry exists. Invalid runtime metadata is excluded
fail-closed; an empty eligible set stays idle. Actions are token- and
generation-fenced so callbacks from an old Persona, pack, revision, surface, or
completed render cannot mutate the current Buddy.

Pack authors declare behavior beside the manifest:

```json
{
  "schema_version": 1,
  "entries": [{
    "state": "ambient.look",
    "trigger": "ambient",
    "category": "idle_variant",
    "suggested_weight": 3,
    "suggested_cooldown_ms": 45000
  }]
}
```

Missing behavior means base idle only. Version 1 allows at most 128 unique
trigger/state entries. States must be resolvable by the same pack; accepted
triggers are `ambient`, `click`, and `drag`, and accepted categories are
`idle_variant`, `reaction`, and `move`. Author weights are bounded to 0–10,000
and cooldowns to 0–86,400,000ms. Movement is allowed only for `move`, is
horizontal only, and uses start/end ratios from 0 through 1 with start no
greater than end. See [Persona Visual Packs](Persona_Visual_Packs.md) for the
full authoring and review contract.

## Gestures and accessibility

- A single mouse click waits 300ms so a double-click can be distinguished,
  then plays the authored `click` reaction when eligible.
- A double-click or Enter opens Buddy controls without playing the delayed
  single-click reaction.
- Space uses the same authored click reaction.
- A touch tap uses the same delayed click path.
- Movement beyond the 8px drag threshold becomes a drag. Release persists the
  explicit anchor and may play the authored `drag` reaction.

Semantic activity, hidden documents, open controls, and active drags reject
conflicting direct reactions. Reduced-motion users may still request a direct
reaction, but the renderer presents its deterministic static frame rather than
allocating animation work.

## Reduced motion and protected raster assets

Reduced motion selects `preview_frame`, then `preview_asset_id`, then the first
animation frame. The selected asset must be a genuine one-frame `image/png`.
The renderer presents it once and allocates no frame timer or video work.

Visual asset endpoints are authenticated and owner-scoped. Native image URLs
cannot attach the configured API-key or bearer header, so renderers never place
protected endpoint URLs directly in `src`. The shared loader:

1. Fetches through the authenticated tldw client.
2. Enforces the declared and received byte limit (16MiB by default), MIME type,
   byte size, and SHA-256 checksum.
3. Caches immutable assets by asset ID plus checksum and exposes a reference-
   counted Blob URL.
4. Aborts stale generations, retains the previous Blob until the replacement
   is ready, and releases/revokes handles when no longer used.

The content API returns only owner-authorized bytes and uses a private cache
policy. Nested Persona, pack, and asset IDs are all checked; a same-user path
mismatch and a different user both receive 404.

## Review, activation, and licensing

Review validates the normalized manifest and behavior, reachable asset bytes,
static PNG coverage, and renderer contract without rewriting the pack. Its
fingerprint binds the manifest, companion behavior, renderer/converter and
provenance versions, and reachable asset metadata/checksums. Activation is an
atomic switch requiring the exact expected version and current reviewed
fingerprint.

Active pack payloads and assets are immutable. To change one, fork an inactive
revision, edit it, review the new fingerprint, and explicitly activate it.
Import, generated-candidate acceptance, starter-copy, and library reuse all
remain inactive until that review/activation step.

Technical validation proves format, bounds, integrity, and runtime safety. It
does not establish licensing rights. Users and pack authors are responsible for
copyright, trademark, model-output terms, attribution, and any other rights
needed for their assets.

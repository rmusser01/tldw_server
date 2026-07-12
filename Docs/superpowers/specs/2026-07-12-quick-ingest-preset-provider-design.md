# Quick Ingest Preset Provider Design

## Problem

The shared Quick Ingest wizard defaults to the Standard preset. Standard and
Deep enable analysis, but the active wizard builds both presets from hard-coded
defaults whose `advancedValues.api_name` is empty. The analysis-provider guard
therefore blocks those presets. The older Quick Ingest surface and Quick Ingest
Settings already persist customized preset maps under
`quickIngestPresetConfigs`, but the active wizard does not read that storage.
Its configure step also has no way to repair a missing provider.

Because the wizard is shared, the defect affects both the Next.js WebUI and the
browser extension.

## Considered Approaches

### 1. Hydrate the active wizard and add an inline provider control (recommended)

Reuse the existing persisted preset map, make the wizard reducer resolve preset
changes against that map, and expose `api_name` while analysis is enabled. Keep
the existing early provider guard as a final safety check.

This restores the intended Settings contract, provides an immediate recovery
path, and keeps WebUI/extension behavior aligned without backend changes.

### 2. Remove the frontend provider guard

Allow Standard and Deep to submit without `api_name` and rely on backend
defaults. This is smaller but reintroduces ambiguous backend behavior and can
silently skip analysis or fail after ingest begins.

### 3. Disable analysis in Standard and Deep by default

This avoids the error but changes the meaning of both presets and makes their
labels inaccurate. It treats the symptom by removing requested functionality.

## Design

### Preset data flow

`useQuickIngestEvents`, which is mounted by the always-present Quick Ingest
event host, will read `quickIngestPresetConfigs` with the existing Plasmo
storage hook and normalize it through `resolvePresetMap`. It will gate pending
opens on both Zustand session hydration and preset-storage hydration. This
loads settings at app mount instead of showing a blank modal while storage
loads.

A combined readiness flag (`sessionHydrated && presetStorageHydrated`) also
gates event-host activation and modal rendering. A session rehydrated with
`visibility: "visible"` therefore cannot mount the wizard or persist fallback
defaults before preset storage is ready. When readiness completes with an
already-visible idle draft, the hook captures the map and open revision once,
rebases the eligible draft, and only then exposes it to the modal.

On each closed-to-open transition, the event hook will capture one resolved
preset-map snapshot. Changes made in Settings while the modal is open do not
mutate the active run. Reopening captures the new settings. An explicit open
revision will remount/rebase the provider only when the session is an idle
draft and its selected preset is non-custom. Workflow-seeded first-source
drafts are excluded from rebasing: their persisted
`FIRST_SOURCE_QUICK_PRESET_CONFIG` intentionally enables chunking even though
the normal Quick preset does not.

Creating another draft is also a snapshot boundary. The event hook exposes a
`createNewDraft` callback that captures the currently resolved map, increments
the revision, and then replaces the session. "Ingest More" calls this callback
instead of mutating the session store directly, so settings changed while a
completed result remained open apply to the new draft without leaking
session-only provider choices from the prior run.

The provider's reducer will resolve initial non-custom state, preset switches,
custom-option matching, and reset behavior against the supplied map. Custom
sessions keep their persisted full `presetConfig`. Eligible idle, non-custom
drafts use the captured definition for their selected preset, which makes the
existing Settings promise—changes apply the next time Quick Ingest opens—true
for the active wizard.

Processing, interrupted, cancelled, and completed sessions never rebase from
Settings. Their persisted `presetConfig` remains authoritative so reattachment
and historical results retain the configuration with which they started.

All option edits merge into the session's current full `presetConfig`, rather
than reconstructing it from a base preset plus the lossy `customOptions`
delta. The merged full config is then matched against the captured preset map
to determine whether its label remains a named preset or becomes Custom.
`SET_PRESET("custom")` preserves that full current config. This rule applies to
named, custom, and workflow-seeded sessions, so changing OCR, provider, or any
other option cannot discard the first-source chunking override. Clearing an
advanced field removes it from the full config before persistence, so JSON
serialization cannot cause a cleared `api_name` to reappear after reload.

The default reducer behavior remains `DEFAULT_PRESETS` when no map is supplied,
preserving isolated consumers and tests.

### Analysis provider control

When the configure step is visible and analysis is enabled,
`WizardConfigureStep` will show an editable "Analysis provider" combobox bound
to `presetConfig.advancedValues.api_name`.

The control will load configured providers from the existing
`tldwClient.getProvidersStatus()` endpoint and offer them as suggestions. The
suggestion list is the trimmed, deduplicated names from entries whose
`configured` field is true; it does not use `any_configured`, because usable
local providers do not contribute to that cloud-only aggregate. The current
typed value remains available even when absent from the catalog.

The request runs only while the configure step is visible and analysis is
enabled. Effect cleanup ignores stale responses. A temporary provider-catalog
failure leaves the editable combobox usable. Clearing the control removes
`api_name` through the full-config merge path described above.

Provider discovery failure is non-fatal. The control remains editable and the
existing early guard prevents an analysis-enabled run with an empty provider.

Provider selection in this wizard is session-scoped. The label/help copy will
say "For this ingest"; changing the reusable Standard or Deep defaults remains
the responsibility of Quick Ingest Settings. This change does not add an
implicit "remember" behavior or another persistence control.

Suggestion copy will describe entries as "configured", not "available",
because a configured local service may still be unreachable. Locale keys for
the label, "For this ingest" help, placeholder/catalog status, and required
warning will be added to the English locale source and use the existing locale
fallback behavior for untranslated languages. The service helper remains the
validation predicate; the UI renders localized warning copy at the boundary.

### Error handling

No backend fallback is added, and the UI never guesses among multiple
providers. Quick processing validates before entering the processing step. If
Standard or Deep has analysis enabled but no provider, it advances from Add
Content to Configure (step 2), keeps processing idle, focuses the provider
combobox, displays an inline warning linked with `aria-describedby`, and makes
no start request. The warning uses an alert live region so the redirect is
announced. If the late safety guard is reached from Review, it likewise resets
processing to idle, returns to Configure, focuses the control, and renders the
same warning. Neither path synthesizes a failed run or enters a render loop.

### Tests

Focused tests will prove:

- a configured Standard preset is used for initial wizard state;
- switching between Standard and Deep uses the configured map;
- the configure step displays and updates the analysis provider;
- configured Standard/Deep flows pass the provider guard;
- fresh/default Standard and Deep sessions with no provider advance to
  Configure, focus the provider control, remain idle, and make no start request;
- delayed preset storage hydration blocks initialization and auto-processing;
- a preloaded visible draft with delayed preset hydration does not mount or
  persist fallback defaults before the combined readiness gate opens;
- closing, changing preset settings, and reopening rebases only an idle,
  non-custom draft;
- a first-source Quick seed retains its mandatory chunking override after
  storage hydration, reopen, and any option edit;
- processing/reattached and completed sessions retain their persisted config;
- clearing a custom provider, persisting/rehydrating, and changing another
  option does not resurrect the provider;
- provider selection remains scoped to the active session, while Ingest More
  captures current Settings, creates a new session from that map, and does not
  carry the previous session-only provider;
- provider suggestions filter unconfigured entries, deduplicate configured
  names, retain typed local/custom values, and tolerate loading failure;
- the provider control has an associated label, supports keyboard and free-text
  entry, links its warning with `aria-describedby`, exposes the warning through
  an alert live region, and receives focus after a missing-provider redirect.

Shared component tests cover the reducer and modal behavior. Verification also
requires the WebUI and extension typecheck/build gates plus one storage
hydration smoke path under each runtime adapter. The existing targeted WebUI
and extension Quick Ingest browser harnesses will verify the affected boundary,
not merely the shared source in isolation.

## Scope

This change will not alter backend provider selection, add a new provider
registry, redesign unrelated Quick Ingest options, or migrate legacy stored
data. It reuses the existing persisted preset schema and API client.

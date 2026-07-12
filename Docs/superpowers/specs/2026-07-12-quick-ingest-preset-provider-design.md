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

`QuickIngestWizardModal` will read `quickIngestPresetConfigs` with the existing
Plasmo storage hook and normalize it through `resolvePresetMap`. The modal will
use the hook's `isLoading` metadata and will not initialize or auto-process a
preset-dependent draft until storage hydration completes.

On each closed-to-open transition, the modal will capture one resolved preset
map snapshot. Changes made in Settings while the modal is open do not mutate
the active run. Reopening captures the new settings. An explicit open revision
will remount/rebase the provider only when the session is an idle draft and its
selected preset is non-custom. Workflow-seeded first-source drafts are excluded
from rebasing: their persisted `FIRST_SOURCE_QUICK_PRESET_CONFIG` intentionally
enables chunking even though the normal Quick preset does not.

The provider's reducer will resolve initial non-custom state, preset switches,
custom-option matching, and reset behavior against the supplied map. Custom
sessions keep their persisted full `presetConfig`. Eligible idle, non-custom
drafts use the captured definition for their selected preset, which makes the
existing Settings promise—changes apply the next time Quick Ingest opens—true
for the active wizard.

Processing, interrupted, cancelled, and completed sessions never rebase from
Settings. Their persisted `presetConfig` remains authoritative so reattachment
and historical results retain the configuration with which they started.

For custom sessions, subsequent option edits merge into the persisted full
`presetConfig`, rather than reconstructing it from a potentially changed base
preset plus the lossy `customOptions` delta. Clearing an advanced field removes
it from the full config before persistence, so JSON serialization cannot cause
a cleared `api_name` to reappear after reload.

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

### Error handling

No backend fallback is added. Quick processing validates before entering the
processing step and keeps the user on Add Content with the existing warning.
The full configure/review flow validates before any start request; if the late
safety guard is reached, it resets processing to idle, returns to Configure
(step 2), and renders the provider warning beside the editable combobox. It
must not synthesize a failed run or enter a render loop.

### Tests

Focused tests will prove:

- a configured Standard preset is used for initial wizard state;
- switching between Standard and Deep uses the configured map;
- the configure step displays and updates the analysis provider;
- configured Standard/Deep flows pass the provider guard;
- a missing provider remains blocked on the exact recoverable step/status and
  no start request occurs;
- delayed preset storage hydration blocks initialization and auto-processing;
- closing, changing preset settings, and reopening rebases only an idle,
  non-custom draft;
- a first-source Quick seed retains its mandatory chunking override after
  storage hydration and reopen;
- processing/reattached and completed sessions retain their persisted config;
- clearing a custom provider, persisting/rehydrating, and changing another
  option does not resurrect the provider;
- provider suggestions filter unconfigured entries, deduplicate configured
  names, retain typed local/custom values, and tolerate loading failure.

Shared component tests cover the reducer and modal behavior. Verification also
requires the WebUI and extension typecheck/build gates plus one storage
hydration smoke path under each runtime adapter. The existing targeted WebUI
and extension Quick Ingest browser harnesses will verify the affected boundary,
not merely the shared source in isolation.

## Scope

This change will not alter backend provider selection, add a new provider
registry, redesign unrelated Quick Ingest options, or migrate legacy stored
data. It reuses the existing persisted preset schema and API client.

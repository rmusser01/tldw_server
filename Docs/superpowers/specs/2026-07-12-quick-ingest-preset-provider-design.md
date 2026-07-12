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
Plasmo storage hook and normalize it through `resolvePresetMap`. It will pass
the resolved map to `IngestWizardProvider`.

The provider's reducer will resolve initial non-custom state, preset switches,
custom-option matching, and reset behavior against the supplied map. Custom
sessions keep their persisted configuration. Non-custom sessions use the
current saved definition for their selected preset, which makes the existing
Settings promise—changes apply the next time Quick Ingest opens—true for the
active wizard.

The default reducer behavior remains `DEFAULT_PRESETS` when no map is supplied,
preserving isolated consumers and tests.

### Analysis provider control

When analysis is enabled, `WizardConfigureStep` will show an "Analysis
provider" control bound to `presetConfig.advancedValues.api_name`.

The control will load configured providers from the existing
`tldwClient.getProvidersStatus()` endpoint and offer them as suggestions. It
will still accept a typed value so local/custom provider aliases work and a
temporary provider-catalog failure does not make the form unusable. Clearing
the control removes `api_name` through the existing `setCustomOptions` merge
path.

Provider discovery failure is non-fatal. The control remains editable and the
existing early guard prevents an analysis-enabled run with an empty provider.

### Error handling

No backend fallback is added. If analysis remains enabled without a provider,
the wizard stays on or returns to a recoverable pre-processing step with the
existing warning. It must not synthesize a failed run or enter a render loop.

### Tests

Focused tests will prove:

- a configured Standard preset is used for initial wizard state;
- switching between Standard and Deep uses the configured map;
- the configure step displays and updates the analysis provider;
- configured Standard/Deep flows pass the provider guard;
- a missing provider remains blocked without entering processing.

The shared component tests cover both WebUI and extension behavior. Verification
will include focused Vitest, the shared UI/frontend TypeScript check, diff
checks, and browser coverage when the existing targeted harness is available.

## Scope

This change will not alter backend provider selection, add a new provider
registry, redesign unrelated Quick Ingest options, or migrate legacy stored
data. It reuses the existing persisted preset schema and API client.

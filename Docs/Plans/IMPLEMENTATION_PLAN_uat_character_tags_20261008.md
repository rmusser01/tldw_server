# Real UAT Character tag crash repair

Associated task: TASK-13260.278.18.83.45.
ADR assessment: no new ADR; existing AntD form value/label contract remains.

## Stage 1: Real causal regression
**Goal**: Reproduce tag-search corruption using the real CharacterEditorForm, AntD Form/Select and preview in create/edit modes.
**Success Criteria**: Typing alone retains an array; adding/removing and popular tags submit exact arrays without losing other fields.
**Tests**: Focused new actual-component regression, unchanged production red.
**Status**: Complete

## Stage 2: Minimal binding repair
**Goal**: Keep only the inner tag Select bound and associate its label.
**Success Criteria**: Regression and affected expression/first-use/preview controls pass; configured types/lint clear of new findings.
**Tests**: Affected UI/WebUI suites, touched types/lint; Bandit N/A TS-only.
**Status**: Complete

## Stage 3: Real corrected workflow
**Goal**: Build the corrected client and complete the real create/edit tag lifecycle.
**Success Criteria**: Fresh browser creation/save/reload/export/import succeeds with exact fields; independent source review and normal hook/commit qualification remain required before unit closure.
**Tests**: Normal production build and B01 corrected live steps; no replay of qualified unrelated Notes actions.
**Status**: In Progress

2026-10-08 qualification: unchanged-source real AntD tag typing/Enter regression red (2 failed), minimal outer label binding repair green. Affected complete UI and WebUI suites each 115 passed without skips; configured types pass. Lint has only retained original diagnostics, TS-only Bandit N/A. Independent actual-source review CLEAR. New production build and real create/edit/tag/preview/reload passed; actual export completed. Live reimport uncovered a separate existing V3 optional-null exporter/importer defect and remains failed until its causal repair. Malformed JSON error is useful and leaves the four-entry library unchanged. Stage 3 remains open; hooks/commit and all deployment/auth matrix cells remain unaccepted.

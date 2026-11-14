# Media zero-visible page continuation

Task: TASK-13260.278.18.83.45, recorded before edits.
ADR required: no. Preserve the existing authoritative server page count and shared navigation; correct only presentation of a locally filtered current page.
Scope: Media/Pagination.tsx, its real mounted test and this plan. ViewMediaPage visible-count fix and all other WIP remain unchanged.

## Stage 1: Demonstrate missing navigation
**Goal**: Real shared Pagination keeps navigation on zero-visible first and last pages.
**Success Criteria**: New mounted transition cases fail on unchanged Pagination because Next/Previous is absent.
**Tests**: Two causal cases with total40/pageSize20/pages2/count0; original case/default5s/600outer.
**Status**: Complete

## Stage 2: Minimal repair and checks
**Goal**: Retain truthful zero count and existing page controls whenever totalPages>1.
**Success Criteria**: New transitions and genuine-empty preservation pass in UI/Web; prior page-size behavior and original assertions remain.
**Tests**: Full genuinely affected Pagination module, touched baseline/current lint, syntax and combined configured types.
**Status**: Complete

## Stage 3: Review and load
**Goal**: Independently review both actual callers and later load the reviewed client.
**Success Criteria**: Connected-source review clear; current build/live check qualified separately; no complete collection/pagination/UAT claim.
**Tests**: Root owns combined build/runtime and actual live validation; no existing filtered-count or unrelated control replay.
**Status**: In Progress

Causal RED:2selected mounted cases failed naturally in2.094s because both zero-visible pages rendered no navigation;2other cases selector-deselected. Minimal two-hunk shared fix passed full configured UI4/Web4 naturally in2.692s/1.879s with original case/outer limits. Syntax/lint0, original page-size test and previous count unit preserved, diff clear. Independent corrective review and post-change full configured types pending; no build/runtime/live acceptance.

Independent corrective connected-source review CLEAR, reviewed source/test hashes freshly matched. Post-change full normal UI/Web configured types0errors, natural0 in83.936s/54.147s. Stage3 remains open for actual new build/live qualification.

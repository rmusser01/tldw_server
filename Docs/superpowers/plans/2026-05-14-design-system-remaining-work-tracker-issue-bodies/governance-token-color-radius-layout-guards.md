# Design token, color, radius, and layout drift guards

## Purpose

Reduce non-product-state design drift by deciding which token, color, radius, and layout patterns should be guarded next.

## Scope

Included guard, doc, CI, or ownership decision:

- Define the next guardable categories beyond product-state imports and canonical state labels.
- Distinguish product-state migration requirements from allowed AntD mechanics.
- Identify the documentation or verifier changes needed before enforcement.

## Non-Goals

- This track does not replace the current product-state guard.
- This track does not force repo-wide AntD removal.
- This track does not migrate individual product surfaces.

## Done Criteria

- Durable artifact exists and is linked from the epic.
- Verification or review path is documented.
- Follow-up migration tasks know how to use the artifact.

## Delivery and verification

The merged reporter inventories literal and palette colors, custom radii, and
layout values across the shared UI, WebUI, extension, and admin UI. Use the
reporter command in the design-system contract and check the report's source
commit and completion status before reviewing findings.
[Usage and review guidance](https://github.com/rmusser01/tldw_server/blob/dev/Docs/Design/tldw_web_design_system_contract.md#advisory-styling-drift-reports)

For follow-up migrations, review candidates in the owning surface. Prefer an
existing semantic token or shared primitive where it fits, and document narrow
exceptions for intentional styling. Use the separate product-state verifier
for canonical state labels and primitives. The design-system contract also
explains where Ant Design mechanics are allowed.
[Product-state CI path](https://github.com/rmusser01/tldw_server/blob/dev/Docs/Design/tldw_web_design_system_ci_gate_path.md) ·
[Ant Design policy](https://github.com/rmusser01/tldw_server/blob/dev/Docs/Design/tldw_web_design_system_contract.md#ant-design-policy)

- On `dev` at `2053ec5ad4`, the advisory report completed across all four roots.
  It analyzed 3,863 files with zero operational errors, and all 232 focused
  tests passed.
- The hosted advisory workflow generated the report, uploaded the JSON, and
  published the job summary in a run after delivery.
  [Hosted workflow evidence](https://github.com/rmusser01/tldw_server/actions/runs/46053191753/job/140694788637)

The delivered workflow is advisory. Required styling enforcement needs a
separate owner decision and CI policy change.
[ADR-098](https://github.com/rmusser01/tldw_server/blob/dev/Docs/ADR/098-advisory-ui-design-drift-reporting.md)

## Tracking

- Parent epic: https://github.com/rmusser01/tldw_server/issues/1655
- Backlog task: TASK-45.44.16 (`backlog/tasks/task-45.44.16 - Design-token-color-radius-and-layout-drift-guards.md`)
- PRs:

| PR | Notes |
| --- | --- |
| [#3294](https://github.com/rmusser01/tldw_server/pull/3294) | Merged the advisory reporter, tests, documentation, and nonrequired CI workflow. |

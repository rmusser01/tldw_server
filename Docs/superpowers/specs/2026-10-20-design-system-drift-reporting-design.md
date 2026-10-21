# Repository-Wide Design-System Drift Reporting

Date: 2026-10-20
Status: Reviewed; owner requested implementation planning after review fixes
Task: TASK-45.44.16
Public tracker: https://github.com/rmusser01/tldw_server/issues/1673
Parent epic: https://github.com/rmusser01/tldw_server/issues/1655

## Decision and Outcome

Add a standalone, advisory verifier for token, color, radius, and layout drift
across repository-owned product UI. Scan existing and new code on every run.
Publish actionable findings without blocking builds or merges. The owner selected
repo-wide reporting with no blocking on 2026-10-20.

This document defines the implementation contract. This design-only change adds
no executable verifier, CI job, baseline, token changes, or product migrations.

## ADR Check

ADR required: yes. Proposed ADR:
`Docs/ADR/079-advisory-ui-design-drift-reporting.md`.

The decision establishes lasting UI conventions and an advisory reporting policy.
No existing ADR in the current index governs these rules. ADR-059 governs task
editing; the existing design-system contract governs token and primitive ownership.

## Existing Foundations and Approaches

Reuse the semantic tokens in `apps/packages/ui/src/styles/tokens/`, CSS variables
in `apps/packages/ui/src/assets/tailwind-shared.css`, and the surrounding script
and focused-test conventions of the product-state verifier. Preserve its current
rules, baseline, exit behavior, and required checks.

| Approach | Assessment |
| --- | --- |
| Standalone verifier beside the product-state verifier | Selected: one report across code and CSS, with independent advisory behavior. |
| ESLint rules plus a separate CSS checker | More configuration and two reporting paths; unnecessary for this slice. |
| Repository-wide text matching | Cheap inventory, but confuses code, documentation, strings, and actual styling. |

Use installed syntax tooling and existing helpers where their contracts fit.
Avoid refactoring the product-state guard just to share a small file walker.
No automatic rewriting or new dependency is authorized by this design.

## Scan Boundary

Repository-wide means all files owned by these four product UI roots, independent
of the PR diff:

- `apps/packages/ui/src/`
- `apps/tldw-frontend/`
- `apps/extension/`
- `admin-ui/`

Scan `.js`, `.jsx`, `.ts`, `.tsx`, `.mjs`, `.cjs`, and `.css` files for supported
styling syntax. Classify each file as product source, token/configuration
definition, mixed definition/product source, test/fixture, tooling, or generated
output. Report discovered and analyzed file counts by root, plus excluded counts
by reason. An empty source root, missing expected root, unreadable file, or parse
failure makes the scan incomplete and appears in diagnostics.

Exclude dependencies, build/cache/coverage output, vendor code, source maps,
tests, fixtures, snapshots, and tooling scripts from product-drift totals.
Recognize named files/directories rather than arbitrary substring matches.
Do not traverse symlinks or other worktrees. Backend templates, static HTML,
MDX, and runtime-generated styles are outside v1 syntax coverage; state this
limitation in each report instead of implying exhaustive UI conformance.

Select tracked files and untracked files not excluded by Git ignore rules within
the four roots, then apply the source/exclusion rules above. Report the included
untracked source-file count. Ignored local/generated files are outside the scan.

Exempt only recognized token/theme declarations or exported definition objects
and the relevant Tailwind configuration properties, recording their exact paths
and syntactic scope. Report those definitions separately from product findings.
Do not exempt a whole file because it contains definitions. For example, token
declarations in `tailwind-shared.css` are definitions, while its
`.animated-gradient-border` and `.shimmer-text` rules remain product styling
eligible for findings. Apply the same distinction to mixed TypeScript files.
Neither `components/ui/**` nor `design-system/**` receives a blanket exemption.
Product styling added under those roots remains eligible for findings.

## Token Authority

Recognize semantic utilities and CSS-variable references from each surface's
current token/configuration definitions. Shared UI, WebUI, and extension use the
shared design-system vocabulary. Admin UI currently has a separate vocabulary
such as `background`, `foreground`, and `destructive`; honor that vocabulary.
This verifier does not migrate admin UI onto the shared theme.

The implementation must record the exact definition paths used per surface.
Do not infer that an arbitrary class is semantic merely because its name sounds
semantic. Token definitions may legitimately contain literal values; their
presence does not exempt adjacent product CSS or whole component directories.

## Finding Rules

| Rule | Detect in supported styling locations | Classification |
| --- | --- | --- |
| `literal-color` | Hex, functional, and named CSS color literals in CSS declarations, static inline style values, or arbitrary color utilities. | Token-review finding. |
| `palette-color` | Direct Tailwind palette colors, including neutral palettes, variants, opacity modifiers, and standalone black/white. | Token-review finding. |
| `custom-radius` | Arbitrary radius utilities or literal CSS/inline radius declarations outside the current documented scale for that surface. | Radius-review finding. |
| `layout-value` | Arbitrary spacing, gap, dimensions, inset, z-index, or grid-template utilities and explicit corresponding CSS/inline declarations. | Context-dependent review candidate. |

Analyze JSX class attributes, static class-composition arguments and class maps,
inline style objects, and CSS declarations using syntax-aware parsing. Handle
static branches and template fragments conservatively. Never execute imports,
Tailwind config, application code, or user-authored theme content to scan it.
Use a CSS parser for declarations and a bracket-aware utility splitter so
variants, quotes, colons, and spaces inside arbitrary values remain intact.
Also inspect Tailwind `@apply` at-rule arguments with that same utility scanner;
for example, `@apply bg-red-500 rounded-[13px]` produces palette-color and
custom-radius findings. Preserve the source locations of individual utilities.
The implementation plan must identify installed parsers before choosing them.

Do not flag unrelated prose, API payload colors, URL fragments, ordinary numeric
props, or arbitrary strings as styling. Dynamic expressions that cannot be
resolved statically produce coverage diagnostics rather than fabricated findings.

Semantic CSS-variable references, inherited/current colors, transparent values,
CSS-wide keywords, and standard spacing/layout utilities are allowed. Recognize
named colors only in color-valued positions, including shorthand declarations;
`style={{ color: "red" }}` is a finding, while unrelated strings remain excluded.
Radius values already on the surface's documented scale are allowed; an exact
literal duplicate of a token may receive a token suggestion but must not be
mislabeled off-scale.

For colors and radii, report a suggested existing token only when the mapping
is supported by the surface's definitions. A color's numeric similarity cannot
establish whether it means danger, emphasis, or a chart series. Otherwise ask for
semantic review. Never invent token names or apply fixes automatically.

Layout reporting is deliberately an inventory. A pane width, canvas coordinate,
44px touch target, or viewport clamp can be correct. A report must distinguish
these review candidates from color/radius findings and avoid a single combined
"violation count" that treats all of them as confirmed defects.

## Intentional Exceptions

Use a single data manifest next to the new verifier for intentional styling
exceptions. Each record identifies an exact repository-relative path, rule,
normalized value, stable syntactic owner when available, category, and reason.
Allow categories for theme definitions, data visualization, user-controlled
styling, media/editor rendering, and necessary layout geometry. These categories
do not auto-exempt components based on filenames or imports.

Match exceptions against discovered findings, without depending on line numbers.
When identical matches repeat within the same owner, keep the matched occurrence
count visible so an exception cannot conceal newly duplicated styling.
Report accepted matches and reasons separately; retain their locations in JSON.
Report unmatched entries as stale and ambiguous entries as diagnostics.
Invalid schemas or unknown rule names make the scan incomplete.

Do not seed an exception for every existing finding. Existing debt stays visible,
so no suppression baseline or automatic baseline regeneration is needed.
The reviewer decides whether to migrate a value, retain it with a documented
reason, or create a follow-up task. Allowed AntD interaction mechanics remain
allowed; imports alone do not indicate token or styling drift.

## Report and Exit Contract

Proposed entrypoint:
`apps/packages/ui/scripts/verify-design-system-drift.mjs`.
Proposed package command: `report:design-system-drift`.

Emit a human-readable summary and a complete versioned JSON report. JSON contains
the source commit, scanned roots and coverage, findings, accepted exceptions,
stale exceptions, and diagnostics. Each finding contains its rule, surface,
repository-relative path, one-based line/column, value, classification, reason,
and optional token suggestion. Sort deterministically by path, location, and
rule; omit wall-clock timestamps from deterministic finding content.

Report source provenance as `sourceCommit`, `dirtySource`, and
`untrackedSourceFileCount`. `sourceCommit` is the checked-out HEAD, not a claim
that working files equal that commit. Set `dirtySource` when any scanned input
differs from HEAD, including staged/unstaged edits, deletions, included untracked
source files, and changes to the verifier, token/configuration inputs, or
exception manifest. Changes only to unrelated files or generated report output
do not affect it. Show the dirty indicator in the human summary; a dirty report
cannot be reproduced from its commit SHA alone. Failure to establish provenance
is an incomplete-scan diagnostic, not an assumed clean state.

Console and CI summaries may be capped, but must state how many entries were
omitted and point to the complete JSON artifact. Avoid dumping source lines,
user data, or full style objects into reports; report only relevant values.

| Result | Local exit | Report status | CI behavior |
| --- | --- | --- | --- |
| Completed scan, any number of findings or stale exceptions | 0 | Complete | Advisory summary and JSON artifact. |
| Missing roots, read/parse/config errors, or output failure | 1 | Incomplete when writable | Visible diagnostics; job remains nonrequired. |

An incomplete scan must never be described as zero drift. If writing JSON fails,
print the failure to stderr and preserve any available diagnostics in the job
log. Static dynamic-expression coverage gaps are reported limitations, not
fatal scan errors, unless a required file cannot be parsed at all.

## CI and Documentation

The implementation plan will place reporting in an isolated, nonrequired job
triggered for relevant source, token/config, verifier, and workflow changes, plus
manual dispatch. Each invocation scans all four roots. Keep report execution and
artifact upload separate so an incomplete scan can still publish diagnostics.
Use advisory job behavior and always-attempt artifact publication; publish the
commit SHA and dirty indicator to distinguish stale or locally modified reports.
No comment bot or extra service.

Do not connect this command to production builds, required aggregate jobs,
pre-commit hooks, or branch-protection requirements. The existing required
product-state guard and build token-sync checks keep their behavior.

Update the main design-system contract with the report's scope and advisory
status, plus usage guidance for migration tasks. Those tasks should cite the
source SHA, dirty indicator, and specific findings, explain intentional
exceptions, and record appropriate visual QA under the existing checklist when
changing rendered UI.

Future enforcement requires an explicit owner decision, demonstrated rule
accuracy, reviewed exceptions/debt treatment, and a separate change to the ADR
and CI policy. Counts never activate enforcement automatically.

## Verification and Acceptance

Focused verifier tests must exercise all four roots, each rule, variants and
arbitrary-value parsing, CSS and inline styles, unrelated strings, dynamic
coverage diagnostics, semantic values, differing admin tokens, canonical-root
components, intentional/stale/ambiguous exceptions, and deterministic sorting.
Include mixed CSS and TypeScript definition/product files, named colors and
allowed keywords, and semantic/advisory `@apply` cases. Verify clean, staged,
unstaged, deleted, untracked, and ignored source handling, plus unrelated-file and
report-output edits that must not mark source dirty.
CLI checks must verify findings exit 0, scan errors exit 1, and incomplete
reports cannot look clean. A workflow review verifies no required-job/build
dependency and artifact publication after scan failure.

Run the implementation against current product source and review a small sample
from every rule and surface before documenting counts as useful migration data.
Record coverage limits and false positives instead of claiming visual or runtime
correctness. Browser visual QA applies to later UI migrations, not this reporter.

For this design-only slice, validate Markdown links, task canonical formatting,
diff whitespace, and consistency with current token/guard/config ownership.
Bandit and runtime/browser tests do not apply because no executable code changes.

The design task remains In Progress until written review and publication of its
durable artifact are complete. The implementation plan must separately track
delivery of the verifier, report, tests, CI wiring, and usage documentation.

# Design System Product-State Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a guarded, baseline-backed product-state policy check for
`apps/packages/ui/src` so new shared UI product-state code must use the tldw
design-system primitives or declare a temporary migration exception.

**Architecture:** Build a small repo-owned Node guard rather than a custom
ESLint plugin. Keep product-state detection, baseline matching, stale-baseline
reporting, and report formatting in a pure rules module with Vitest coverage;
keep filesystem walking and process exit behavior in a thin CLI wrapper. Use
TypeScript's compiler API for import and JSX detection, and limit text scanning
to literal/context signals that are intentionally fixture-tested.

**Tech Stack:** Bun workspace scripts, Node ESM scripts, TypeScript compiler
API, Vitest, JSON baseline, existing `@tldw/ui` package layout.

---

## Scope

Implement the design from:

- `Docs/superpowers/specs/2026-05-06-design-system-product-state-guard-design.md`

This plan covers the guard infrastructure only. It does not migrate existing
Chat, Playground, Watchlists, Admin, or Settings surfaces off legacy
product-state patterns. Existing findings become explicit baseline debt unless
they are safe to migrate in a later slice.

## File Map

- Create `apps/packages/ui/scripts/design-system-product-state-rules.mjs`
  - Pure policy engine.
  - Exports source analysis, finding ID generation, baseline validation,
    baseline application, stale-baseline detection, and report formatting.
  - Does not read the filesystem or call `process.exit`.

- Create `apps/packages/ui/scripts/verify-design-system-product-state.mjs`
  - Thin CLI wrapper.
  - Walks `apps/packages/ui/src`.
  - Reads `design-system-product-state-baseline.json`.
  - Prints a grouped report.
  - Exits nonzero for unbaselined blocked findings or malformed baseline
    entries.

- Create `apps/packages/ui/scripts/design-system-product-state-baseline.json`
  - Checked-in baseline of current shared UI product-state exceptions.
  - Entries use stable finding IDs and include `owner`, `reason`,
    `replacement`, and `migrationQueue`.
  - Only `allowed_legacy_exception` and `active_migration_target` are valid
    stored states.

- Create `apps/packages/ui/src/design-system/__tests__/product-state-guard.test.ts`
  - Vitest coverage for the rules module and CLI-level result semantics.
  - Keep this under `src/**/__tests__` because `apps/packages/ui/vitest.config.ts`
    only includes that tree.

- Modify `apps/packages/ui/package.json`
  - Add `verify:design-system-state`.
  - Add `typescript` to `devDependencies` if the implementation imports the
    compiler API directly from this package.

- Modify `apps/bun.lock`
  - Only if `bun install` updates the workspace lockfile after adding the
    explicit `@tldw/ui` TypeScript dev dependency.

- Modify `Docs/Design/tldw_web_design_system_inventory.md`
  - Point future shared UI product-state work at the guard command and baseline.

- Update Backlog task `TASK-45.9`
  - Record plan, verification, and final summary for this implementation-plan
    slice.

## Design Decisions

- Use the TypeScript compiler API for JSX import/use detection. This avoids
  broad import-name false positives and directly addresses the reviewed
  product-state context concern.
- Keep canonical roots explicit in code, not as broad `components/ui/**` or
  `design-system/**` directory exemptions. New canonical files must be added
  deliberately.
- Keep adapter exceptions in the JSON baseline for v1. A separate adapter
  allowlist can be introduced only after there are enough real adapters to make
  that split useful.
- Treat stale baseline entries as reportable warnings. They do not fail the
  command in v1, but they must be visible so migration cleanup removes dead
  debt.
- Make `blocked` a computed result state only. Baseline files must never store
  `blocked`.

## Task 1: Pure Rule Engine And Fixture Tests

**Files:**

- Create: `apps/packages/ui/src/design-system/__tests__/product-state-guard.test.ts`
- Create: `apps/packages/ui/scripts/design-system-product-state-rules.mjs`
- Modify: `apps/packages/ui/package.json`
- Modify: `apps/bun.lock` if dependency resolution changes

- [ ] **Step 1: Write the failing rule tests**

Create `apps/packages/ui/src/design-system/__tests__/product-state-guard.test.ts`
with focused fixture strings. Use dynamic import so the test can import an ESM
script module without changing the package build.

```ts
import { describe, expect, it } from "vitest"

const guard = await import("../../../scripts/design-system-product-state-rules.mjs")

const analyze = (relativePath: string, source: string) =>
  guard.analyzeSource({
    relativePath,
    source
  })

describe("design-system product-state guard rules", () => {
  it("flags local product-state wrapper components by filename and symbol", () => {
    const findings = analyze(
      "src/components/Sidepanel/Chat/ConnectionBanner.tsx",
      `
        export function ConnectionBanner() {
          return <div>Disconnected from backend</div>
        }
      `
    )

    expect(findings).toContainEqual(
      expect.objectContaining({
        id: "local-recovery-banner:src/components/Sidepanel/Chat/ConnectionBanner.tsx:ConnectionBanner",
        rule: "local-recovery-banner",
        path: "src/components/Sidepanel/Chat/ConnectionBanner.tsx",
        subject: "ConnectionBanner"
      })
    )
  })

  it("flags local empty, loading, and status wrappers", () => {
    const findings = [
      ...analyze(
        "src/components/Common/FeatureEmpty.tsx",
        `
          export function FeatureEmpty() {
            return <div>No saved items yet</div>
          }
        `
      ),
      ...analyze(
        "src/components/Common/FeatureLoadingState.tsx",
        `
          export function FeatureLoadingState() {
            return <div>Loading workspace</div>
          }
        `
      ),
      ...analyze(
        "src/components/Common/FooStatusBadge.tsx",
        `
          export function FooStatusBadge() {
            return <span>Ready</span>
          }
        `
      )
    ]

    expect(findings).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          rule: "local-empty-state",
          subject: "FeatureEmpty"
        }),
        expect.objectContaining({
          rule: "local-loading-state",
          subject: "FeatureLoadingState"
        }),
        expect.objectContaining({
          rule: "local-status-badge",
          subject: "FooStatusBadge"
        })
      ])
    )
  })

  it("flags AntD product-state JSX only when product-state context is present", () => {
    const findings = analyze(
      "src/components/Option/Settings/health-status.tsx",
      `
        import { Alert, Tag } from "antd"

        export function HealthStatus() {
          return (
            <>
              <Alert type="error" message="Server unavailable" />
              <Tag color="success">Ready</Tag>
            </>
          )
        }
      `
    )

    expect(findings).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          rule: "antd-product-state-import",
          subject: "Alert"
        }),
        expect.objectContaining({
          rule: "antd-product-state-import",
          subject: "Tag"
        })
      ])
    )
  })

  it("allows AntD mechanics and metadata-only tags", () => {
    const findings = analyze(
      "src/components/Option/Models/ProviderTable.tsx",
      `
        import { Table, Tag, Tooltip } from "antd"

        export function ProviderTable() {
          return (
            <Table
              columns={[
                { title: "Provider", render: () => <Tag>OpenAI</Tag> },
                { title: "Info", render: () => <Tooltip title="Model family" /> }
              ]}
            />
          )
        }
      `
    )

    expect(findings).toEqual([])
  })

  it("does not flag metadata-only AntD Tag when another JSX node is product-state", () => {
    const findings = analyze(
      "src/components/Option/Models/ProviderStatusPanel.tsx",
      `
        import { Alert, Tag } from "antd"

        export function ProviderStatusPanel() {
          return (
            <>
              <Alert type="error" message="Server unavailable" />
              <Tag>OpenAI</Tag>
            </>
          )
        }
      `
    )

    expect(findings).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          rule: "antd-product-state-import",
          subject: "Alert"
        })
      ])
    )
    expect(findings).not.toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          rule: "antd-product-state-import",
          subject: "Tag"
        })
      ])
    )
  })

  it("does not exempt new files merely because they live under canonical namespaces", () => {
    const findings = analyze(
      "src/components/ui/feedback/ProjectOfflineBanner.tsx",
      `
        export function ProjectOfflineBanner() {
          return <div>Server unavailable</div>
        }
      `
    )

    expect(findings).toContainEqual(
      expect.objectContaining({
        rule: "local-recovery-banner",
        subject: "ProjectOfflineBanner"
      })
    )
  })

  it("exempts known canonical implementation roots but still scans for unknown canonical siblings", () => {
    const canonicalFindings = analyze(
      "src/components/ui/feedback/LoadingState.tsx",
      `
        export function LoadingState() {
          return <div>Loading</div>
        }
      `
    )

    expect(canonicalFindings).toEqual([])
  })

  it("flags hardcoded canonical state labels outside approved roots", () => {
    const findings = analyze(
      "src/components/Common/StatusBadge.tsx",
      `
        export function StatusBadge() {
          return <span>Setup required</span>
        }
      `
    )

    expect(findings).toContainEqual(
      expect.objectContaining({
        rule: "canonical-state-label",
        subject: "Setup required"
      })
    )
  })
})
```

- [ ] **Step 2: Run the test to verify it fails**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --maxWorkers=1
```

Expected result:

- FAIL because `scripts/design-system-product-state-rules.mjs` does not exist.

- [ ] **Step 3: Add the explicit TypeScript dependency if needed**

If `node -e 'import("typescript").then(() => console.log("ok"))'` fails from
`apps/packages/ui`, add `typescript` to `apps/packages/ui/package.json`
`devDependencies`, matching the version already present in `apps/bun.lock`.

Recommended edit:

```json
"typescript": "^5.9.3"
```

Then run from `apps`:

```bash
bun install
```

Expected result:

- `apps/bun.lock` changes only if Bun needs to record the package-level
  dependency.

- [ ] **Step 4: Implement the minimal pure rules module**

Create `apps/packages/ui/scripts/design-system-product-state-rules.mjs`.
Keep the first implementation explicit and small.

```js
import ts from "typescript"

export const PRODUCT_STATE_ANTD_NAMES = new Set([
  "Alert",
  "Badge",
  "Empty",
  "Result",
  "Spin",
  "Tag"
])

export const VALID_BASELINE_STATES = new Set([
  "allowed_legacy_exception",
  "active_migration_target"
])

export const CANONICAL_ROOTS = [
  "src/components/ui/primitives/Alert.tsx",
  "src/components/ui/primitives/Badge.tsx",
  "src/components/ui/feedback/EmptyState.tsx",
  "src/components/ui/feedback/LoadingState.tsx",
  "src/components/ui/layout/ModalFooter.tsx",
  "src/components/ui/state/ActionGroup.tsx",
  "src/components/ui/state/DiagnosticRow.tsx",
  "src/components/ui/state/PermissionNotice.tsx",
  "src/components/ui/state/RecoveryCallout.tsx",
  "src/components/ui/state/SetupRequiredPanel.tsx",
  "src/components/ui/state/StatePanel.tsx",
  "src/design-system/states.ts",
  "src/design-system/index.ts",
  "src/assets/tailwind.css",
  "src/assets/tailwind-shared.css"
]

const CANONICAL_STATE_LABELS = [
  "Unavailable",
  "Setup required",
  "Sign in required",
  "Permission denied",
  "Degraded",
  "Retrying",
  "Blocked",
  "Ready",
  "Loading"
]

const PRODUCT_STATE_WORDS = [
  "unavailable",
  "degraded",
  "retrying",
  "blocked",
  "setup",
  "sign in",
  "permission denied",
  "retry",
  "diagnostics",
  "reconnect",
  "disconnected",
  "loading",
  "failed"
]

const RECOVERY_COMPONENT_PATTERN =
  /(Error|Connection|Unavailable|Recovery|Offline|Readiness|Permission)Banner$/
const EMPTY_COMPONENT_PATTERN = /(EmptyState|Empty)$/
const LOADING_COMPONENT_PATTERN = /(LoadingState|Loading|Spinner)$/
const STATUS_COMPONENT_PATTERN = /(StatusBadge|StatusTag|StatusChip|StatusDot)$/

export function createFindingId(rule, relativePath, subject) {
  return `${rule}:${relativePath}:${subject}`
}

export function analyzeSource({ relativePath, source }) {
  if (isExcludedPath(relativePath)) {
    return []
  }

  const sourceFile = ts.createSourceFile(
    relativePath,
    source,
    ts.ScriptTarget.Latest,
    true,
    relativePath.endsWith(".tsx") ? ts.ScriptKind.TSX : ts.ScriptKind.TS
  )

  const findings = []
  const localAntdNames = collectAntdProductStateImports(sourceFile)
  const componentNames = collectComponentNames(sourceFile)
  const fileSubject = subjectFromPath(relativePath)
  const textSignals = collectTextSignals(source)

  for (const subject of new Set([fileSubject, ...componentNames])) {
    pushLocalComponentFinding(findings, relativePath, subject)
  }

  pushCanonicalLabelFindings(findings, relativePath, source)
  pushAntdFindings(findings, relativePath, sourceFile, localAntdNames, {
    relativePath,
    componentNames,
    textSignals
  })

  return dedupeFindings(findings)
}
```

The implementation does not need to match this snippet byte-for-byte, but it
must keep these public exports and rule names stable:

- `analyzeSource({ relativePath, source })`
- `createFindingId(rule, relativePath, subject)`
- `VALID_BASELINE_STATES`
- `CANONICAL_ROOTS`

Implementation requirements:

- `isExcludedPath()` must exempt only:
  - explicit `CANONICAL_ROOTS`
  - `src/components/ui/**/index.ts`
  - `src/**/*.test.ts`
  - `src/**/*.test.tsx`
  - `src/**/__tests__/**`
  - `src/assets/locale/**`
  - `src/public/_locales/**`
- `collectAntdProductStateImports()` must map local import aliases, for
  example `import { Alert as AntAlert } from "antd"`.
- `pushAntdFindings()` must require both JSX usage and product-state context.
- `pushAntdFindings()` must evaluate the specific JSX use, not only the file.
  A metadata-only `<Tag>OpenAI</Tag>` in the same file as a product-state
  `<Alert type="error" />` must not become a product-state finding.
- `pushAntdFindings()` must not flag metadata-only `Tag` examples unless that
  specific JSX use has product-state text, state/severity props, recovery
  actions, or product-state component semantics.
- `pushLocalComponentFinding()` must cover `FooConnectionBanner`, `FooEmpty`,
  `FooEmptyState`, `FooLoading`, `FooLoadingState`, `FooStatusBadge`,
  `FooStatusTag`, `FooStatusChip`, and `FooStatusDot`.
- `pushCanonicalLabelFindings()` must use canonical labels as literal signals
  only outside approved roots/tests/locales.
- Findings must have this shape:

```ts
type ProductStateFinding = {
  id: string
  path: string
  rule:
    | "antd-product-state-import"
    | "local-recovery-banner"
    | "local-empty-state"
    | "local-loading-state"
    | "local-status-badge"
    | "canonical-state-label"
  subject: string
  message: string
  line?: number
  replacement: string
}
```

- [ ] **Step 5: Run the rule tests to verify they pass**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --maxWorkers=1
```

Expected result:

- PASS for the rule tests.

- [ ] **Step 6: Commit Task 1**

```bash
git add apps/packages/ui/src/design-system/__tests__/product-state-guard.test.ts \
  apps/packages/ui/scripts/design-system-product-state-rules.mjs \
  apps/packages/ui/package.json \
  apps/bun.lock
git commit -m "test: add product-state guard rule coverage"
```

Only include `apps/bun.lock` if it changed.

## Task 2: Baseline Matching, Stale Reporting, And Report Semantics

**Files:**

- Modify: `apps/packages/ui/src/design-system/__tests__/product-state-guard.test.ts`
- Modify: `apps/packages/ui/scripts/design-system-product-state-rules.mjs`

- [ ] **Step 1: Add failing baseline and reporting tests**

Append tests to `product-state-guard.test.ts`.

```ts
describe("design-system product-state guard baseline handling", () => {
  it("marks unbaselined findings as blocked", () => {
    const findings = analyze(
      "src/components/Common/ConnectionProblemBanner.tsx",
      `
        export function ConnectionProblemBanner() {
          return <div>Server unavailable</div>
        }
      `
    )

    const result = guard.applyBaseline({
      findings,
      baseline: []
    })

    expect(result.blocked).toEqual([
      expect.objectContaining({
        state: "blocked",
        rule: "local-recovery-banner"
      })
    ])
    expect(result.allowedLegacy).toEqual([])
    expect(result.activeMigrationTargets).toEqual([])
  })

  it("allows valid legacy and active-migration baseline entries", () => {
    const findings = analyze(
      "src/components/Common/StatusBadge.tsx",
      `
        export function StatusBadge() {
          return <span>Ready</span>
        }
      `
    )
    const [finding] = findings

    const result = guard.applyBaseline({
      findings,
      baseline: [
        {
          id: finding.id,
          path: finding.path,
          rule: finding.rule,
          subject: finding.subject,
          state: "active_migration_target",
          owner: "design-system",
          reason: "Existing generic status wrapper selected for cleanup.",
          replacement: "Badge with design-system state mapping",
          migrationQueue: "shared-product-state"
        }
      ]
    })

    expect(result.blocked).toEqual([])
    expect(result.activeMigrationTargets).toHaveLength(1)
    expect(result.activeMigrationTargets[0]).toEqual(
      expect.objectContaining({
        state: "active_migration_target",
        owner: "design-system"
      })
    )
  })

  it("does not allow a same-path same-rule finding with a different subject", () => {
    const result = guard.applyBaseline({
      findings: [
        {
          id: "local-status-badge:src/components/Common/StatusBadge.tsx:NewStatusBadge",
          path: "src/components/Common/StatusBadge.tsx",
          rule: "local-status-badge",
          subject: "NewStatusBadge",
          message: "Use Badge with a state registry mapping.",
          replacement: "Badge with design-system state registry mapping"
        }
      ],
      baseline: [
        {
          id: "local-status-badge:src/components/Common/StatusBadge.tsx:OldStatusBadge",
          path: "src/components/Common/StatusBadge.tsx",
          rule: "local-status-badge",
          subject: "OldStatusBadge",
          state: "allowed_legacy_exception",
          owner: "design-system",
          reason: "Existing generic status wrapper before the guard.",
          replacement: "Badge with design-system state registry mapping",
          migrationQueue: "shared-product-state"
        }
      ]
    })

    expect(result.blocked).toEqual([
      expect.objectContaining({
        id: "local-status-badge:src/components/Common/StatusBadge.tsx:NewStatusBadge",
        state: "blocked"
      })
    ])
    expect(result.staleBaseline).toEqual([
      expect.objectContaining({
        id: "local-status-badge:src/components/Common/StatusBadge.tsx:OldStatusBadge"
      })
    ])
  })

  it("rejects malformed baseline entries and stored blocked states", () => {
    const errors = guard.validateBaseline([
      {
        id: "local-status-badge:src/components/Common/StatusBadge.tsx:StatusBadge",
        path: "src/components/Common/StatusBadge.tsx",
        rule: "local-status-badge",
        subject: "StatusBadge",
        state: "blocked",
        owner: "",
        reason: "",
        replacement: "",
        migrationQueue: ""
      }
    ])

    expect(errors).toEqual(
      expect.arrayContaining([
        expect.stringContaining("state must be allowed_legacy_exception or active_migration_target"),
        expect.stringContaining("owner is required"),
        expect.stringContaining("reason is required"),
        expect.stringContaining("replacement is required"),
        expect.stringContaining("migrationQueue is required")
      ])
    )
  })

  it("reports stale baseline entries when no live finding matches the entry id", () => {
    const result = guard.applyBaseline({
      findings: [],
      baseline: [
        {
          id: "local-empty-state:src/components/Option/Old/RemovedEmpty.tsx:RemovedEmpty",
          path: "src/components/Option/Old/RemovedEmpty.tsx",
          rule: "local-empty-state",
          subject: "RemovedEmpty",
          state: "allowed_legacy_exception",
          owner: "design-system",
          reason: "Removed component should no longer need a baseline entry.",
          replacement: "EmptyState",
          migrationQueue: "shared-product-state"
        }
      ]
    })

    expect(result.staleBaseline).toEqual([
      expect.objectContaining({
        id: "local-empty-state:src/components/Option/Old/RemovedEmpty.tsx:RemovedEmpty"
      })
    ])
  })

  it("formats blocked, active migration, legacy, stale, and invalid groups distinctly", () => {
    const report = guard.formatReport({
      blocked: [
        {
          id: "local-recovery-banner:src/components/Common/NewBanner.tsx:NewBanner",
          path: "src/components/Common/NewBanner.tsx",
          rule: "local-recovery-banner",
          subject: "NewBanner",
          message: "Use RecoveryCallout or StatePanel.",
          replacement: "RecoveryCallout or StatePanel",
          state: "blocked"
        }
      ],
      activeMigrationTargets: [
        {
          id: "local-status-badge:src/components/Common/StatusBadge.tsx:StatusBadge",
          path: "src/components/Common/StatusBadge.tsx",
          rule: "local-status-badge",
          subject: "StatusBadge",
          message: "Use Badge with a state registry mapping.",
          replacement: "Badge with design-system state registry mapping",
          state: "active_migration_target",
          owner: "design-system",
          reason: "Selected for the next shared product-state cleanup.",
          migrationQueue: "shared-product-state"
        }
      ],
      allowedLegacy: [
        {
          id: "local-empty-state:src/components/Common/FeatureEmpty.tsx:FeatureEmpty",
          path: "src/components/Common/FeatureEmpty.tsx",
          rule: "local-empty-state",
          subject: "FeatureEmpty",
          message: "Use EmptyState.",
          replacement: "EmptyState",
          state: "allowed_legacy_exception",
          owner: "design-system",
          reason: "Existing empty wrapper before the guard.",
          migrationQueue: "shared-product-state"
        }
      ],
      staleBaseline: [
        {
          id: "local-empty-state:src/components/Option/Old/RemovedEmpty.tsx:RemovedEmpty",
          path: "src/components/Option/Old/RemovedEmpty.tsx",
          rule: "local-empty-state",
          subject: "RemovedEmpty",
          state: "allowed_legacy_exception",
          owner: "design-system",
          reason: "Removed.",
          replacement: "EmptyState",
          migrationQueue: "shared-product-state"
        }
      ],
      baselineErrors: ["baseline[0] owner is required"]
    })

    expect(report).toContain("Blocked product-state findings")
    expect(report).toContain("Stale baseline entries")
    expect(report).toContain("Invalid baseline entries")
    expect(report).toContain("Baseline exceptions: 2")
    expect(report).toContain("local-status-badge: 1")
    expect(report).toContain("local-empty-state: 1")
    expect(report).toContain("shared-product-state: 2")
  })
})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --maxWorkers=1
```

Expected result:

- FAIL because `applyBaseline`, `validateBaseline`, and `formatReport` are not
  implemented.

- [ ] **Step 3: Implement baseline helpers**

Extend `design-system-product-state-rules.mjs` with:

```js
const REQUIRED_BASELINE_FIELDS = [
  "id",
  "path",
  "rule",
  "subject",
  "state",
  "owner",
  "reason",
  "replacement",
  "migrationQueue"
]

export function validateBaseline(baseline) {
  if (!Array.isArray(baseline)) {
    return ["baseline must be a JSON array"]
  }

  const errors = []
  const seenIds = new Set()

  baseline.forEach((entry, index) => {
    for (const field of REQUIRED_BASELINE_FIELDS) {
      if (typeof entry?.[field] !== "string" || entry[field].trim() === "") {
        errors.push(`baseline[${index}] ${field} is required`)
      }
    }

    if (entry?.state && !VALID_BASELINE_STATES.has(entry.state)) {
      errors.push(
        `baseline[${index}] state must be allowed_legacy_exception or active_migration_target`
      )
    }

    if (seenIds.has(entry?.id)) {
      errors.push(`baseline[${index}] duplicate id ${entry.id}`)
    }
    seenIds.add(entry?.id)
  })

  return errors
}

export function applyBaseline({ findings, baseline }) {
  const baselineErrors = validateBaseline(baseline)
  const validBaseline = baselineErrors.length === 0 ? baseline : []
  const byId = new Map(validBaseline.map((entry) => [entry.id, entry]))
  const matchedIds = new Set()

  const blocked = []
  const allowedLegacy = []
  const activeMigrationTargets = []

  for (const finding of findings) {
    const entry = byId.get(finding.id)
    if (!entry) {
      blocked.push({ ...finding, state: "blocked" })
      continue
    }

    matchedIds.add(entry.id)
    const merged = { ...finding, ...entry }
    if (entry.state === "active_migration_target") {
      activeMigrationTargets.push(merged)
    } else {
      allowedLegacy.push(merged)
    }
  }

  const staleBaseline = validBaseline.filter((entry) => !matchedIds.has(entry.id))

  return {
    blocked,
    activeMigrationTargets,
    allowedLegacy,
    staleBaseline,
    baselineErrors
  }
}
```

Add `formatReport(result)` with deterministic headings:

- `Invalid baseline entries`
- `Blocked product-state findings`
- `Active product-state migration targets`
- `Allowed legacy product-state exceptions`
- `Stale baseline entries`
- `No product-state guard issues found`

Formatting requirements:

- Include rule, path, subject, owner, replacement, and migration queue when
  available.
- Include remaining baseline totals grouped by rule and by migration queue.
- Include stale baseline entries even when no blocked findings exist.
- Do not print raw JSON as the primary output; the report should be readable in
  CI logs.

- [ ] **Step 4: Run the tests to verify they pass**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --maxWorkers=1
```

Expected result:

- PASS for rule, baseline, and report tests.

- [ ] **Step 5: Commit Task 2**

```bash
git add apps/packages/ui/src/design-system/__tests__/product-state-guard.test.ts \
  apps/packages/ui/scripts/design-system-product-state-rules.mjs
git commit -m "feat: add product-state guard baseline semantics"
```

## Task 3: CLI, Real Scan, And Initial Baseline

**Files:**

- Create: `apps/packages/ui/scripts/verify-design-system-product-state.mjs`
- Create: `apps/packages/ui/scripts/design-system-product-state-baseline.json`
- Modify: `apps/packages/ui/src/design-system/__tests__/product-state-guard.test.ts`
- Modify: `apps/packages/ui/package.json`

- [ ] **Step 1: Add failing CLI-result tests**

Append tests that exercise the CLI-level behavior through a pure runner helper,
not by spawning a process.

```ts
describe("design-system product-state guard runner", () => {
  it("returns failure when blocked findings or baseline errors exist", async () => {
    const result = await guard.runGuardOnSources({
      sources: [
        {
          relativePath: "src/components/Common/NewRecoveryBanner.tsx",
          source: `
            export function NewRecoveryBanner() {
              return <div>Retry connection</div>
            }
          `
        }
      ],
      baseline: []
    })

    expect(result.exitCode).toBe(1)
    expect(result.report).toContain("Blocked product-state findings")
  })

  it("returns success when findings are baselined and stale entries are only warnings", async () => {
    const sources = [
      {
        relativePath: "src/components/Common/FeatureEmptyState.tsx",
        source: `
          export function FeatureEmptyState() {
            return <div>No results yet</div>
          }
        `
      }
    ]
    const findings = analyze(sources[0].relativePath, sources[0].source)
    const result = await guard.runGuardOnSources({
      sources,
      baseline: [
        {
          id: findings[0].id,
          path: findings[0].path,
          rule: findings[0].rule,
          subject: findings[0].subject,
          state: "allowed_legacy_exception",
          owner: "design-system",
          reason: "Existing empty state before the guard.",
          replacement: "EmptyState",
          migrationQueue: "shared-product-state"
        },
        {
          id: "local-status-badge:src/components/Removed.tsx:Removed",
          path: "src/components/Removed.tsx",
          rule: "local-status-badge",
          subject: "Removed",
          state: "allowed_legacy_exception",
          owner: "design-system",
          reason: "Removed component.",
          replacement: "Badge",
          migrationQueue: "shared-product-state"
        }
      ]
    })

    expect(result.exitCode).toBe(0)
    expect(result.report).toContain("Allowed legacy product-state exceptions")
    expect(result.report).toContain("Stale baseline entries")
  })
})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --maxWorkers=1
```

Expected result:

- FAIL because `runGuardOnSources` does not exist.

- [ ] **Step 3: Implement runner semantics in the pure module**

Add this export to `design-system-product-state-rules.mjs`:

```js
export async function runGuardOnSources({ sources, baseline }) {
  const findings = sources.flatMap(({ relativePath, source }) =>
    analyzeSource({ relativePath, source })
  )
  const result = applyBaseline({ findings, baseline })
  const report = formatReport(result)
  const exitCode =
    result.blocked.length > 0 || result.baselineErrors.length > 0 ? 1 : 0

  return {
    ...result,
    findings,
    report,
    exitCode
  }
}
```

- [ ] **Step 4: Create the CLI wrapper**

Create `apps/packages/ui/scripts/verify-design-system-product-state.mjs`.

```js
#!/usr/bin/env node

import fs from "node:fs/promises"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { runGuardOnSources } from "./design-system-product-state-rules.mjs"

const here = path.dirname(fileURLToPath(import.meta.url))
const packageRoot = path.resolve(here, "..")
const srcRoot = path.resolve(packageRoot, "src")
const baselinePath = path.resolve(
  here,
  "design-system-product-state-baseline.json"
)

async function walkSourceFiles(dir) {
  const entries = await fs.readdir(dir, { withFileTypes: true })
  const files = []

  for (const entry of entries) {
    const absolutePath = path.resolve(dir, entry.name)
    if (entry.isDirectory()) {
      files.push(...(await walkSourceFiles(absolutePath)))
      continue
    }
    if (/\.(ts|tsx)$/.test(entry.name)) {
      files.push(absolutePath)
    }
  }

  return files
}

async function main() {
  const absoluteFiles = await walkSourceFiles(srcRoot)
  const sources = await Promise.all(
    absoluteFiles.map(async (absolutePath) => ({
      relativePath: path
        .relative(packageRoot, absolutePath)
        .split(path.sep)
        .join("/"),
      source: await fs.readFile(absolutePath, "utf8")
    }))
  )
  const baseline = JSON.parse(await fs.readFile(baselinePath, "utf8"))
  const result = await runGuardOnSources({ sources, baseline })

  console.log(result.report)
  process.exitCode = result.exitCode
}

main().catch((error) => {
  console.error(error)
  process.exitCode = 1
})
```

Implementation requirements:

- Keep source paths relative to `apps/packages/ui`, for example
  `src/components/Common/StatusBadge.tsx`.
- Do not scan generated output, `node_modules`, `dist`, or `build`.
- Do not add command-line flags in v1 unless needed for implementation
  debugging. A stable default command is enough.

- [ ] **Step 5: Add the package script**

Modify `apps/packages/ui/package.json`:

```json
"verify:design-system-state": "node scripts/verify-design-system-product-state.mjs"
```

Keep the existing scripts. Do not reformat unrelated `package.json` sections.

- [ ] **Step 6: Create an empty baseline and verify the real scan fails**

Create `apps/packages/ui/scripts/design-system-product-state-baseline.json`:

```json
[]
```

Run from `apps/packages/ui`:

```bash
bun run verify:design-system-state
```

Expected result:

- FAIL with `Blocked product-state findings`.
- The report should include current legacy shared UI examples rather than
  failing because of script errors.

- [ ] **Step 7: Add initial baseline entries for current findings**

Update `design-system-product-state-baseline.json` so all current findings are
baselined. Do not weaken rules to make the command pass.

Baseline entry rules:

- `id`, `path`, `rule`, and `subject` must exactly match the live finding.
- Use `state: "allowed_legacy_exception"` by default.
- Use `state: "active_migration_target"` only for a small next cleanup queue
  if the implementation owner is ready to migrate those files immediately.
- Use `owner: "design-system"` for v1 unless there is a more precise owner.
- Use concrete replacements:
  - recovery/unavailable/readiness: `RecoveryCallout or StatePanel`
  - empty states: `EmptyState`
  - loading states: `LoadingState or StatePanel`
  - status chips: `Badge with design-system state registry mapping`
  - hardcoded canonical labels: `getDesignSystemState(...)`
- Use `migrationQueue: "shared-product-state"` unless a narrower queue is
  already known.

Known likely baseline candidates include, but are not limited to:

- `src/components/Sidepanel/Chat/ConnectionBanner.tsx`
- `src/components/Sidepanel/Chat/empty.tsx`
- `src/components/Sidepanel/Chat/StatusDot.tsx`
- `src/components/Sidepanel/Chat/SaveStatusIcon.tsx`
- `src/components/Option/Playground/PlaygroundEmpty.tsx`
- `src/components/Option/Playground/PlaygroundChatErrorBanner.tsx`
- `src/components/Option/Playground/PlaygroundComposerNotices.tsx`
- `src/components/Option/Playground/ResearchRunStatusStack.tsx`
- `src/components/Option/Playground/VoiceChatIndicator.tsx`
- `src/components/Common/ConnectionProblemBanner.tsx`
- `src/components/Common/ConnectFeatureBanner.tsx`
- `src/components/Common/FeatureEmptyState.tsx`
- `src/components/Common/StatusBadge.tsx`
- `src/components/Common/WorkspaceConnectionGate.tsx`
- `src/components/Common/ChatSidebar/ChatStateBadge.tsx`

If the real scan finds additional current product-state wrappers, baseline
them with clear replacement notes. If the scan flags obvious metadata-only
usage, add a fixture test proving the intended allow case and fix the rule
instead of baselining it.

- [ ] **Step 8: Verify the baselined real scan passes**

Run from `apps/packages/ui`:

```bash
bun run verify:design-system-state
```

Expected result:

- PASS exit code.
- Report contains `Allowed legacy product-state exceptions` if current debt
  remains.
- Report contains `Stale baseline entries` only if an entry no longer matches
  a live finding; remove stale entries unless intentionally kept for a
  near-term branch dependency.

- [ ] **Step 9: Run the focused guard tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --maxWorkers=1
```

Expected result:

- PASS.

- [ ] **Step 10: Commit Task 3**

```bash
git add apps/packages/ui/scripts/verify-design-system-product-state.mjs \
  apps/packages/ui/scripts/design-system-product-state-baseline.json \
  apps/packages/ui/scripts/design-system-product-state-rules.mjs \
  apps/packages/ui/src/design-system/__tests__/product-state-guard.test.ts \
  apps/packages/ui/package.json
git commit -m "feat: add shared UI product-state guard"
```

Include `apps/bun.lock` if it changed during dependency installation.

## Task 4: Documentation, Integration Verification, And Closeout

**Files:**

- Modify: `Docs/Design/tldw_web_design_system_inventory.md`
- Modify:
  `backlog/tasks/task-45.9 - Create-shared-UI-product-state-guard-implementation-plan.md`
  only if executing this plan in the same branch; otherwise update the
  implementation task created for the execution slice.

- [ ] **Step 1: Update the inventory with the guard command**

Add a short note near the shared UI product-state migration queue in
`Docs/Design/tldw_web_design_system_inventory.md`:

```md
### Shared UI Product-State Guard

Run `bun run verify:design-system-state` from `apps/packages/ui` before adding
or changing shared UI product-state surfaces. New recovery, loading, empty,
status, readiness, setup, auth, and permission UI in `apps/packages/ui/src`
should use the design-system primitives under `src/components/ui` and
`src/design-system/states.ts`.

Existing shared UI product-state debt is tracked in
`apps/packages/ui/scripts/design-system-product-state-baseline.json`. Do not add
new baseline entries unless a migration exception has an owner, reason,
replacement, and queue; remove stale baseline entries when a migration removes
the matching finding.
```

- [ ] **Step 2: Run the focused static design-system tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run \
  src/design-system/__tests__/product-state-guard.test.ts \
  src/design-system/__tests__/proof-surface-static-guard.test.ts \
  src/design-system/__tests__/states.test.ts \
  src/design-system/__tests__/state-token-aliases.test.ts \
  --maxWorkers=1
```

Expected result:

- PASS.

- [ ] **Step 3: Run the new guard command**

Run from `apps/packages/ui`:

```bash
bun run verify:design-system-state
```

Expected result:

- PASS.
- No `Blocked product-state findings`.
- Any stale baseline entries are either removed before commit or explicitly
  justified in the Backlog notes.

- [ ] **Step 4: Run diff hygiene**

Run from the repo root:

```bash
git diff --check
```

Expected result:

- PASS with no whitespace errors.

- [ ] **Step 5: Decide Bandit applicability**

This implementation touches TypeScript tests, Node scripts, JSON, package
metadata, and docs only. It does not touch Python runtime code.

Expected result:

- Bandit is not applicable. Record the skip in the Backlog final summary as:
  `Bandit skipped: no Python files touched.`

- [ ] **Step 6: Update Backlog task with verification**

Update the active Backlog task with:

- implementation notes listing touched files
- verification commands and pass/fail results
- Bandit skip rationale
- any known stale-baseline warnings left intentionally

- [ ] **Step 7: Commit Task 4**

```bash
git add Docs/Design/tldw_web_design_system_inventory.md \
  backlog/tasks/task-45.9\ -\ Create-shared-UI-product-state-guard-implementation-plan.md
git commit -m "docs: document product-state guard workflow"
```

If this plan is executed under a separate implementation Backlog task, add that
task file instead of `TASK-45.9`.

## Final Verification For The Full Implementation

Run these from the repo root unless a command says otherwise:

```bash
cd apps/packages/ui
bunx vitest run \
  src/design-system/__tests__/product-state-guard.test.ts \
  src/design-system/__tests__/proof-surface-static-guard.test.ts \
  src/design-system/__tests__/states.test.ts \
  src/design-system/__tests__/state-token-aliases.test.ts \
  --maxWorkers=1
bun run verify:design-system-state
```

Then from the repo root:

```bash
git diff --check
```

Expected final state:

- Focused design-system Vitest tests pass.
- `bun run verify:design-system-state` exits zero.
- The guard report has no blocked findings.
- Stale baseline entries are removed unless explicitly justified.
- `git diff --check` passes.
- Bandit skip is recorded because the implementation does not touch Python.

## Review Checklist

Before opening a PR or marking the implementation done, review these points:

- The guard scans `apps/packages/ui/src`, not the whole repo.
- AntD `Alert`, `Tag`, `Badge`, `Empty`, `Spin`, and `Result` are flagged only
  when used in JSX with product-state context.
- Metadata tags, table/form/modal mechanics, provider/model/file-type labels,
  and static non-state labels are not flagged.
- Canonical roots are explicit file paths or narrow test/locale/index patterns,
  not broad namespace exemptions.
- New product-state-like files under `src/components/ui/**` are still scanned
  unless added deliberately as canonical roots.
- Baseline entries require owner, reason, replacement, and queue.
- Baseline entries cannot store `blocked`.
- Stale baseline entries appear in reports.
- The package script is discoverable from `apps/packages/ui/package.json`.
- Inventory documentation explains how to use the guard and when baseline
  entries are acceptable.

## Handoff Notes

- Use @superpowers:test-driven-development for the implementation slice.
- Use @superpowers:verification-before-completion before claiming the guard
  passes.
- Use @superpowers:receiving-code-review if review comments challenge the rule
  boundary, baseline schema, or canonical-root list.
- Keep the initial rule set conservative. Prefer a small number of clear,
  explainable findings over broad noisy enforcement.
- If real scanning produces excessive false positives, stop after three rule
  tuning attempts, document the failing examples, and narrow the first v1 rule
  set rather than adding a large inaccurate baseline.

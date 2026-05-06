import { describe, expect, it } from "vitest"

const guard = await import("../../../scripts/design-system-product-state-rules.mjs")

const analyze = (relativePath: string, source: string) =>
  guard.analyzeSource({
    relativePath,
    source
  })

describe("design-system product-state guard rules", () => {
  it("exports stable rule metadata for later baseline work", () => {
    expect(guard.VALID_BASELINE_STATES).toEqual(
      new Set(["allowed_legacy_exception", "active_migration_target"])
    )
    expect(guard.CANONICAL_ROOTS).toContain(
      "src/components/ui/feedback/LoadingState.tsx"
    )
  })

  it("flags local recovery wrapper components by filename and symbol", () => {
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
        path: "src/components/Sidepanel/Chat/ConnectionBanner.tsx",
        rule: "local-recovery-banner",
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

  it("maps aliased AntD product-state imports back to their source component", () => {
    const findings = analyze(
      "src/components/Option/Settings/ConnectionAlert.tsx",
      `
        import { Alert as AntAlert } from "antd"

        export function ConnectionAlert() {
          return <AntAlert type="error" message="Server unavailable" />
        }
      `
    )

    expect(findings).toContainEqual(
      expect.objectContaining({
        rule: "antd-product-state-import",
        subject: "Alert"
      })
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

  it("does not let sibling component names turn generic AntD JSX into product-state", () => {
    const source = `
      import { Alert } from "antd"

      export function GenericNotice() {
        return <Alert message="Heads up" />
      }

      export function HealthStatus() {
        return <Alert message="Server unavailable" />
      }
    `
    const findings = analyze(
      "src/components/Option/Settings/HealthPanel.tsx",
      source
    )
    const alertFindings = findings.filter(
      (finding) =>
        finding.rule === "antd-product-state-import" &&
        finding.subject === "Alert"
    )
    const productStateAlertLine =
      source
        .split("\n")
        .findIndex((line) => line.includes('message="Server unavailable"')) + 1

    expect(alertFindings).toHaveLength(1)
    expect(alertFindings[0].line).toBe(productStateAlertLine)
  })

  it("keeps product-state component ownership through lower-case JSX callbacks", () => {
    const findings = analyze(
      "src/components/Option/Settings/HealthPanel.tsx",
      `
        import { Spin } from "antd"

        const items = ["api"]

        export function HealthStatus() {
          const rows = items.map(() => <Spin size="small" />)
          return <>{rows}</>
        }
      `
    )

    expect(findings).toContainEqual(
      expect.objectContaining({
        rule: "antd-product-state-import",
        subject: "Spin"
      })
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

  it("does not flag helper functions with loading-like names as local state wrappers", () => {
    const findings = analyze(
      "src/components/Common/EpubHelpers.tsx",
      `
        function handleEpubLoading() {
          return true
        }

        const useUnifiedLoading = () => true

        export function EpubMetadata() {
          return <span>EPUB</span>
        }
      `
    )

    expect(findings).not.toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          rule: "local-loading-state",
          subject: "handleEpubLoading"
        })
      ])
    )
    expect(findings).not.toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          rule: "local-loading-state",
          subject: "useUnifiedLoading"
        })
      ])
    )
  })

  it("exempts known canonical implementation roots", () => {
    const findings = analyze(
      "src/components/ui/feedback/LoadingState.tsx",
      `
        export function LoadingState() {
          return <div>Loading</div>
        }
      `
    )

    expect(findings).toEqual([])
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

  it("flags hardcoded standalone Ready labels outside approved roots", () => {
    const findings = analyze(
      "src/components/Common/ReadyLabel.tsx",
      `
        export function ReadyLabel() {
          return <span>Ready</span>
        }
      `
    )

    expect(findings).toContainEqual(
      expect.objectContaining({
        rule: "canonical-state-label",
        subject: "Ready"
      })
    )
  })
})

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

    const result = guard.applyBaseline({ findings, baseline: [] })

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
          return <span>Operational</span>
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
          replacement: "Badge with design-system state registry mapping",
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
        expect.stringContaining(
          "state must be allowed_legacy_exception or active_migration_target"
        ),
        expect.stringContaining("owner is required"),
        expect.stringContaining("reason is required"),
        expect.stringContaining("replacement is required"),
        expect.stringContaining("migrationQueue is required")
      ])
    )
  })

  it("rejects non-array baselines", () => {
    expect(guard.validateBaseline({})).toEqual([
      "baseline must be a JSON array"
    ])
  })

  it("rejects duplicate baseline ids", () => {
    const baselineEntry = {
      id: "local-status-badge:src/components/Common/StatusBadge.tsx:StatusBadge",
      path: "src/components/Common/StatusBadge.tsx",
      rule: "local-status-badge",
      subject: "StatusBadge",
      state: "allowed_legacy_exception",
      owner: "design-system",
      reason: "Existing generic status wrapper before the guard.",
      replacement: "Badge with design-system state registry mapping",
      migrationQueue: "shared-product-state"
    }

    expect(guard.validateBaseline([baselineEntry, baselineEntry])).toEqual(
      expect.arrayContaining([
        expect.stringContaining(
          "duplicate baseline id local-status-badge:src/components/Common/StatusBadge.tsx:StatusBadge"
        )
      ])
    )
  })

  it("rejects baseline ids that do not match rule, path, and subject", () => {
    const errors = guard.validateBaseline([
      {
        id: "local-status-badge:src/components/Common/StatusBadge.tsx:StatusBadge",
        path: "src/components/Common/RenamedStatusBadge.tsx",
        rule: "local-status-badge",
        subject: "StatusBadge",
        state: "allowed_legacy_exception",
        owner: "design-system",
        reason: "Path was renamed without updating the stable id.",
        replacement: "Badge with design-system state registry mapping",
        migrationQueue: "shared-product-state"
      }
    ])

    expect(errors).toEqual(
      expect.arrayContaining([
        expect.stringContaining("id must match rule/path/subject")
      ])
    )
  })

  it("blocks findings and skips stale reporting when baseline validation fails", () => {
    const finding = {
      id: "local-status-badge:src/components/Common/StatusBadge.tsx:StatusBadge",
      path: "src/components/Common/StatusBadge.tsx",
      rule: "local-status-badge",
      subject: "StatusBadge",
      message: "Use Badge with a state registry mapping.",
      replacement: "Badge with design-system state registry mapping"
    }

    const result = guard.applyBaseline({
      findings: [finding],
      baseline: [
        {
          id: finding.id,
          path: finding.path,
          rule: finding.rule,
          subject: finding.subject,
          state: "blocked",
          owner: "design-system",
          reason: "Stored blocked states must fail closed.",
          replacement: finding.replacement,
          migrationQueue: "shared-product-state"
        }
      ]
    })

    expect(result.blocked).toEqual([
      expect.objectContaining({
        id: finding.id,
        state: "blocked"
      })
    ])
    expect(result.activeMigrationTargets).toEqual([])
    expect(result.allowedLegacy).toEqual([])
    expect(result.staleBaseline).toEqual([])
    expect(result.baselineErrors).toEqual(
      expect.arrayContaining([
        expect.stringContaining(
          "state must be allowed_legacy_exception or active_migration_target"
        )
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

  it("formats an empty result as no product-state guard issues", () => {
    expect(
      guard.formatReport({
        blocked: [],
        activeMigrationTargets: [],
        allowedLegacy: [],
        staleBaseline: [],
        baselineErrors: []
      })
    ).toBe("No product-state guard issues found")
  })
})

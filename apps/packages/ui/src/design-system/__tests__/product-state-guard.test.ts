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
})

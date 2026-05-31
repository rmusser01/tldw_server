import { readFileSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"

const guard = await import("../../../scripts/design-system-product-state-rules.mjs")

const testDir = path.dirname(fileURLToPath(import.meta.url))
const srcDir = path.resolve(testDir, "../..")

const analyzeLiveSource = (relativePath: string) =>
  guard.analyzeSource({
    relativePath,
    source: readLiveSource(relativePath)
  })

const readLiveSource = (relativePath: string) =>
  readFileSync(path.resolve(srcDir, relativePath.replace(/^src\//, "")), "utf8")

describe("Chat and Playground product-state migration guard", () => {
  it("keeps migrated Chat and Playground state labels out of the legacy baseline", () => {
    const findings = [
      ...analyzeLiveSource("src/components/Option/Playground/PlaygroundForm.tsx"),
      ...analyzeLiveSource("src/routes/sidepanel-chat.tsx")
    ]

    const migratedFindings = findings.filter(
      (finding) =>
        finding.rule === "canonical-state-label" &&
        ((finding.path === "src/components/Option/Playground/PlaygroundForm.tsx" &&
          finding.subject === "Degraded") ||
          (finding.path === "src/routes/sidepanel-chat.tsx" &&
            finding.subject === "Ready"))
    )

    expect(migratedFindings).toEqual([])
  })

  it("does not directly dereference reviewed design-system state labels", () => {
    expect(
      readLiveSource("src/components/Option/Playground/PlaygroundForm.tsx")
    ).not.toContain('getDesignSystemState("degraded").label')
    expect(readLiveSource("src/routes/sidepanel-chat.tsx")).not.toContain(
      'getDesignSystemState("ready").label'
    )
  })

  it("keeps Workspace ACP history load errors on canonical recovery UI", () => {
    const findings = analyzeLiveSource(
      "src/components/Option/ResearchWorkspace/WorkspaceACPHistoryModal.tsx"
    )

    expect(findings).not.toContainEqual(
      expect.objectContaining({
        rule: "antd-product-state-import",
        subject: "Alert"
      })
    )
  })
})

import { readFileSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"

const { analyzeSource } = await import("../../../scripts/design-system-product-state-rules.mjs")
const packageRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../..")

describe("Kanban and Scheduled Tasks product-state migration", () => {
  it.each([
    "src/components/Option/KanbanPlayground/ChecklistSection.tsx",
    "src/components/Option/KanbanPlayground/GenerateBoardDialog.tsx",
    "src/components/Option/KanbanPlayground/ImportPanel.tsx",
    "src/components/Option/ScheduledTasks/ScheduledTaskDetailDrawer.tsx"
  ])("keeps %s on shared product states", (relativePath) => {
    const source = readFileSync(path.join(packageRoot, relativePath), "utf8")
    expect(analyzeSource({ relativePath, source })).toEqual([])
  })

  it("retires the migrated Kanban exception and the stale Models entry", () => {
    const baseline = JSON.parse(readFileSync(
      path.join(packageRoot, "scripts/design-system-product-state-baseline.json"),
      "utf8"
    )) as Array<{ id: string }>
    const ids = baseline.map(({ id }) => id)
    expect(ids).not.toContain(
      "antd-product-state-import:src/components/Option/KanbanPlayground/ImportPanel.tsx:Alert"
    )
    expect(ids).not.toContain(
      "antd-product-state-import:src/components/Option/Models/AvailableModelsList.tsx:Alert"
    )
  })
})

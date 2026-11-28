import { readFileSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { expect, it } from "vitest"

const { analyzeSource } = await import("../../../scripts/design-system-product-state-rules.mjs")
const packageRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../..")
const pagePath = "src/components/Option/ChatWorkflows/ChatWorkflowsPage.tsx"

it("keeps Chat Workflows on shared alerts and retires its legacy alert entries", () => {
  const source = readFileSync(path.join(packageRoot, pagePath), "utf8")
  const baseline = JSON.parse(readFileSync(
    path.join(packageRoot, "scripts/design-system-product-state-baseline.json"),
    "utf8"
  )) as Array<{ path: string; rule: string; subject: string }>
  const isChatWorkflowsAlert = ({ path, rule, subject }: (typeof baseline)[number]) =>
    path === pagePath && rule === "antd-product-state-import" && subject === "Alert"

  expect(analyzeSource({ relativePath: pagePath, source }).filter(isChatWorkflowsAlert)).toEqual([])
  expect(baseline.filter(isChatWorkflowsAlert)).toEqual([])
})

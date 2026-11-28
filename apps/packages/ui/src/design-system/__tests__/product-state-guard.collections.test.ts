import { readFileSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { expect, it } from "vitest"

const { analyzeSource } = await import("../../../scripts/design-system-product-state-rules.mjs")
const packageRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../..")
const pagePaths = [
  "src/components/Option/Collections/ImportExport/ImportExportPanel.tsx",
  "src/components/Option/Collections/Digests/DigestSchedulesPanel.tsx"
]

it.each(pagePaths)("retires legacy Alert and Result use in %s", (pagePath) => {
  const source = readFileSync(path.join(packageRoot, pagePath), "utf8")
  const baseline = JSON.parse(readFileSync(
    path.join(packageRoot, "scripts/design-system-product-state-baseline.json"),
    "utf8"
  )) as Array<{ path: string; rule: string; subject: string }>
  const isLegacyState = ({ path, rule, subject }: (typeof baseline)[number]) =>
    path === pagePath && rule === "antd-product-state-import" && ["Alert", "Result"].includes(subject)

  expect(analyzeSource({ relativePath: pagePath, source }).filter(isLegacyState)).toEqual([])
  expect(baseline.filter(isLegacyState)).toEqual([])
})

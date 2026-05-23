import { readFileSync } from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

const readStage4AxeSource = () =>
  readFileSync(
    path.join(process.cwd(), "e2e/smoke/stage4-axe-high-risk-routes.spec.ts"),
    "utf8"
  )

describe("Stage 4 Axe visual settle contract", () => {
  it("waits for visual settle immediately before Axe analysis", () => {
    const source = readStage4AxeSource()

    expect(source).toContain("waitForVisualSettle")
    expect(source).toMatch(
      /await waitForVisualSettle\(page, LOAD_TIMEOUT\)[\s\S]*?const results = await new AxeBuilder\(\{ page \}\)/
    )
    expect(source).not.toContain('waitForLoadState("networkidle"')
    expect(source).not.toContain("waitForTimeout(250)")
  })
})

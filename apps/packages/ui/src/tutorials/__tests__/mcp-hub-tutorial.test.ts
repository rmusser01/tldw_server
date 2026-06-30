import { describe, expect, it } from "vitest"
import { getTutorialsForRoute, getTutorialById } from "../registry"

describe("MCPHub tutorial registration", () => {
  it("is registered for /mcp-hub route", () => {
    const tutorials = getTutorialsForRoute("/mcp-hub")
    expect(tutorials.length).toBeGreaterThanOrEqual(1)
    expect(tutorials[0].id).toBe("mcp-hub-basics")
  })

  it("has 5 steps", () => {
    const tutorial = getTutorialById("mcp-hub-basics")
    expect(tutorial).toBeDefined()
    expect(tutorial!.steps.length).toBe(5)
  })

  it("every step has a valid data-testid target selector", () => {
    const tutorial = getTutorialById("mcp-hub-basics")!
    for (const step of tutorial.steps) {
      expect(step.target).toMatch(/\[data-testid=/)
    }
  })

  it("targets persistent workflow controls for sections outside the default Setup view", () => {
    const tutorial = getTutorialById("mcp-hub-basics")!
    const targets = tutorial.steps.map((step) => step.target)

    expect(targets).toContain('[data-testid="mcp-hub-workflow-access"]')
    expect(targets).toContain('[data-testid="mcp-hub-workflow-audit"]')
    expect(targets).not.toContain('[data-testid="mcp-hub-tab-profiles"]')
    expect(targets).not.toContain('[data-testid="mcp-hub-tab-audit"]')
  })
})

import { describe, expect, it } from "vitest"
import { getTutorialsForRoute, getTutorialById } from "../registry"

describe("Quiz tutorial registration", () => {
  it("is registered for /quiz route", () => {
    const tutorials = getTutorialsForRoute("/quiz")
    expect(tutorials.length).toBeGreaterThanOrEqual(1)
    expect(tutorials[0].id).toBe("quiz-basics")
  })

  it("has 5 steps", () => {
    const tutorial = getTutorialById("quiz-basics")
    expect(tutorial).toBeDefined()
    expect(tutorial!.steps.length).toBe(5)
  })

  it("every step targets a data-testid selector", () => {
    const tutorial = getTutorialById("quiz-basics")!
    for (const step of tutorial.steps) {
      expect(step.target).toMatch(/\[data-testid=/)
    }
  })
})

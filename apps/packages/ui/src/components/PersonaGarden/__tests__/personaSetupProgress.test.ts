import { describe, expect, it } from "vitest"

import { buildPersonaSetupProgress } from "../personaSetupProgress"

describe("buildPersonaSetupProgress", () => {
  it("treats archetype as the fallback current step for in-progress setups", () => {
    const progress = buildPersonaSetupProgress({
      status: "in_progress",
      current_step: undefined,
      completed_steps: [],
      last_test_type: null
    })

    expect(progress[0]).toMatchObject({
      step: "archetype",
      status: "current"
    })
    expect(progress[1]).toMatchObject({
      step: "persona",
      status: "pending"
    })
  })
})

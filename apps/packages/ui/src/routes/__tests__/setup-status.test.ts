import { describe, expect, it } from "vitest"

import { isSetupStatusRequiringWizard } from "../setup-status"

describe("setup route status helpers", () => {
  it.each([
    undefined,
    null,
    "",
    "not_started",
    "in_progress",
    "blocked",
    "first_chat_complete"
  ])("keeps setup wizard visible for required status %s", (status) => {
    expect(isSetupStatusRequiringWizard(status)).toBe(true)
  })

  it.each(["completed", "skipped", "ready", "unknown"])(
    "lets setup recovery exit for non-required status %s",
    (status) => {
      expect(isSetupStatusRequiringWizard(status)).toBe(false)
    }
  )
})

import { describe, expect, it } from "vitest"

import {
  CHARACTER_CHAT_ONBOARDING_INTENT,
  buildCharacterOnboardingRoute,
  resolveOnboardingEntryIntent
} from "../onboarding-route-intent"

describe("onboarding route intent", () => {
  it("matches only the characters route boundary for returnTo intent", () => {
    expect(
      resolveOnboardingEntryIntent({
        search: "?returnTo=%2Fcharacters-archive"
      })
    ).toBeNull()

    expect(
      resolveOnboardingEntryIntent({
        search: "?returnTo=%2Fcharacters%2Fnew"
      })
    ).toBe(CHARACTER_CHAT_ONBOARDING_INTENT)
  })

  it("does not reuse unrelated character-prefixed return paths", () => {
    expect(
      buildCharacterOnboardingRoute({
        returnTo: "/characters-archive",
        action: "create"
      })
    ).toBe("/characters?from=onboarding&create=true")
  })

  it("preserves real characters routes when adding onboarding actions", () => {
    expect(
      buildCharacterOnboardingRoute({
        returnTo: "/characters/library?sort=name",
        action: "import"
      })
    ).toBe("/characters/library?sort=name&from=onboarding&import=true")
  })

  it("preserves existing characters query context when adding onboarding actions", () => {
    expect(
      buildCharacterOnboardingRoute({
        returnTo: "/characters?from=header-select",
        action: "create"
      })
    ).toBe("/characters?from=header-select&create=true")
  })
})

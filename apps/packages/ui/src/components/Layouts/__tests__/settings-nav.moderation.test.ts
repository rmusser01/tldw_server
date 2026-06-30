import { readFileSync } from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

import { getSettingsNavGroups } from "../settings-nav"
import {
  MODERATION_PLAYGROUND_LEGACY_PATH,
  MODERATION_REVIEW_PATH,
  MODERATION_RULES_PATH
} from "@/routes/route-paths"

const readSourceFromThisTest = (relativePath: string) =>
  readFileSync(path.resolve(__dirname, relativePath), "utf8")

describe("settings nav moderation visibility", () => {
  it("defines canonical moderation route constants", () => {
    expect(MODERATION_REVIEW_PATH).toBe("/moderation")
    expect(MODERATION_RULES_PATH).toBe("/moderation/rules")
    expect(MODERATION_PLAYGROUND_LEGACY_PATH).toBe("/moderation-playground")
  })

  it("includes moderation review and content rules in settings navigation", () => {
    const paths = getSettingsNavGroups(undefined).flatMap((group) =>
      group.items.map((item) => item.to)
    )

    expect(paths).toContain(MODERATION_REVIEW_PATH)
    expect(paths).toContain(MODERATION_RULES_PATH)
    expect(paths).not.toContain(MODERATION_PLAYGROUND_LEGACY_PATH)
  })

  it("registers canonical moderation routes and keeps the legacy alias", () => {
    const source = readSourceFromThisTest("../../../routes/route-registry.tsx")

    expect(source).toContain("MODERATION_REVIEW_PATH")
    expect(source).toContain("MODERATION_RULES_PATH")
    expect(source).toContain("MODERATION_PLAYGROUND_LEGACY_PATH")
  })
})

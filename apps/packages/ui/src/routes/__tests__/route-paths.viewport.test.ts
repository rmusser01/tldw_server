import { describe, expect, it } from "vitest"

import {
  CHAT_PATH,
  VIEWPORT_CONSTRAINED_PATHS
} from "../route-paths"

describe("viewport constrained routes", () => {
  it("keeps the main chat route viewport constrained", () => {
    expect(VIEWPORT_CONSTRAINED_PATHS).toContain(CHAT_PATH)
  })
})

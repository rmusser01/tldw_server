import { describe, expect, it } from "vitest"

import {
  CHAT_PATH,
  RESEARCH_WORKSPACE_PATH,
  VIEWPORT_CONSTRAINED_PATHS
} from "../route-paths"

describe("viewport constrained routes", () => {
  it("keeps the main chat route viewport constrained", () => {
    expect(VIEWPORT_CONSTRAINED_PATHS).toContain(CHAT_PATH)
  })

  it("uses research workspace as the constrained research workspace route", () => {
    expect(RESEARCH_WORKSPACE_PATH).toBe("/research-workspace")
    expect(VIEWPORT_CONSTRAINED_PATHS).toContain(RESEARCH_WORKSPACE_PATH)
    expect(VIEWPORT_CONSTRAINED_PATHS).not.toContain("/workspace-playground")
  })

  it("frames the notes route to the viewport like chat and media (NO-01)", () => {
    // Without this the layout gives /notes a 135vh scrolling document, so the
    // app header scrolls away and an empty band sits under the panels.
    expect(VIEWPORT_CONSTRAINED_PATHS).toContain("/notes")
  })
})

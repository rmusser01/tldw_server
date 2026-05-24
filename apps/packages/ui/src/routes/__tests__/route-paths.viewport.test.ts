import { describe, expect, it } from "vitest"

import {
  DOCUMENT_WORKSPACE_PATH,
  RESEARCH_WORKSPACE_PATH,
  VIEWPORT_CONSTRAINED_PATHS
} from "../route-paths"

describe("viewport constrained routes", () => {
  it("preserves existing non-chat constrained routes", () => {
    expect(VIEWPORT_CONSTRAINED_PATHS).toContain(DOCUMENT_WORKSPACE_PATH)
    expect(VIEWPORT_CONSTRAINED_PATHS).toContain("/media-multi")
  })

  it("uses research workspace as the constrained research workspace route", () => {
    expect(RESEARCH_WORKSPACE_PATH).toBe("/research-workspace")
    expect(VIEWPORT_CONSTRAINED_PATHS).toContain(RESEARCH_WORKSPACE_PATH)
    expect(VIEWPORT_CONSTRAINED_PATHS).not.toContain("/workspace-playground")
  })
})

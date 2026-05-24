import { describe, expect, it } from "vitest"
import {
  getWorkspaceChatNoSourcesHint,
  getWorkspaceSourcesLocationLabel,
  getWorkspaceStudioNoSourcesHint
} from "../source-location-copy"

describe("workspace source location copy", () => {
  it("uses tab wording on mobile", () => {
    expect(getWorkspaceSourcesLocationLabel(true)).toBe("Sources tab")
    expect(getWorkspaceChatNoSourcesHint(true)).toContain("Sources tab")
    expect(getWorkspaceChatNoSourcesHint(true).toLowerCase()).toContain(
      "general chat"
    )
    expect(getWorkspaceStudioNoSourcesHint(true)).toContain("Sources tab")
  })

  it("uses pane wording on desktop", () => {
    expect(getWorkspaceSourcesLocationLabel(false)).toBe("Sources pane")
    expect(getWorkspaceChatNoSourcesHint(false)).toContain("Sources pane")
    expect(getWorkspaceStudioNoSourcesHint(false)).toContain("Sources pane")
  })

  it("does not regress to left-pane wording", () => {
    expect(getWorkspaceChatNoSourcesHint(true).toLowerCase()).not.toContain(
      "left pane"
    )
    expect(getWorkspaceStudioNoSourcesHint(true).toLowerCase()).not.toContain(
      "left pane"
    )
  })
})

import { readFileSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"
import { chatWorkspaceTutorials } from "../definitions/chat-workspace"

const testDir = path.dirname(fileURLToPath(import.meta.url))
const repoRoot = path.resolve(testDir, "../../../../../../")

describe("Chat Workspace behavior copy", () => {
  it("describes Browse as a bounded preview rather than a highlight-only action", () => {
    const step = chatWorkspaceTutorials[0].steps.find(
      (entry) => entry.contentKey === "tutorials:chatWorkspace.basics.sourcesContent"
    )
    expect(step?.contentFallback).toContain("bounded read-only preview")
  })

  describe.each(["User_Guides", "Published/User_Guides"])("%s guide", (directory) => {
    const guide = readFileSync(
      path.resolve(repoRoot, "Docs", directory, "WebUI/Chat_Workspace.md"),
      "utf8"
    )

    it("documents guarded empty-store initialization and canonical manager Open", () => {
      expect(guide).toContain("initializes an empty local workspace")
      expect(guide).toContain("?workspace=<id>")
      expect(guide).not.toContain("without activating that workspace")
    })

    it("documents protected durable recovery without implying a legacy model-switch retry", () => {
      expect(guide).toContain("**Turn needs review**")
      expect(guide).toContain("**Verify saved outcome**")
      expect(guide).toContain("**Reprepare input**")
      expect(guide).toContain("Unknown outcomes are never resent automatically.")
      expect(guide).toContain("Legacy **Retry same model** and **Switch model** actions do not recover selected-durable turns.")
      expect(guide).not.toContain("A failed-turn recovery action can select another model for that retry.")
      expect(guide).not.toContain("The **Switch model** failed-turn recovery action is the exception:")
    })

    it("documents bounded Browse content without claiming an inspector preview", () => {
      expect(guide).toContain("bounded read-only preview")
      expect(guide).toContain("truncated")
      expect(guide).not.toContain("Browse** marks a source")
    })
  })
})

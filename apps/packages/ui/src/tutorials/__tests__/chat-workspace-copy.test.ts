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

    it("documents the failed-turn recovery model picker and durable retry identity", () => {
      expect(guide).toContain("failed-turn recovery")
      expect(guide).toContain("same saved user turn")
    })

    it("documents bounded Browse content without claiming an inspector preview", () => {
      expect(guide).toContain("bounded read-only preview")
      expect(guide).toContain("truncated")
      expect(guide).not.toContain("Browse** marks a source")
    })
  })
})

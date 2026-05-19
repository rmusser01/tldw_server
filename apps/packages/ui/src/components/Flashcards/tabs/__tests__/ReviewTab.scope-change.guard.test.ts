import fs from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

describe("ReviewTab scope-change guard", () => {
  it("lets the scope-change cleanup effect own active review session teardown", () => {
    const source = fs.readFileSync(
      path.resolve(__dirname, "..", "ReviewTab.tsx"),
      "utf8"
    )
    const suggestionBranch =
      source.match(
        /if \(response\.target_service === "flashcards" && targetId\) \{([\s\S]*?)return/
      )?.[1] ?? ""

    expect(suggestionBranch.length).toBeGreaterThan(0)
    expect(source).toContain('setSelectedStudySessionId(null)')
    expect(suggestionBranch).toContain('setSelectedStudySessionId(null)')
    expect(suggestionBranch).toContain("onReviewDeckChange(targetId)")
    expect(suggestionBranch).not.toContain("setActiveReviewSessionId(null)")
  })
})

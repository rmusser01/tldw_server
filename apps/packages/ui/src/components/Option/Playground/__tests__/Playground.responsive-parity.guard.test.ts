import fs from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

describe("Playground responsive parity guard", () => {
  it("keeps compact-device compare/branch notice and artifact indicators", () => {
    const sourcePath = path.resolve(__dirname, "../Playground.tsx")
    const source = fs.readFileSync(sourcePath, "utf8")

    expect(source).toContain("playground-mobile-parity-notice")
    expect(source).toContain("playground:regions.compactFeatureNotice")
    expect(source).toContain("playground-artifacts-trigger")
    expect(source).toContain("playground-artifacts-unread")
    expect(source).toContain("playground:regions.artifactsPinned")
    expect(source).toContain("playground:regions.artifactsCount")
    expect(source).toContain("playground-mobile-artifacts-sheet")
    expect(source).toContain("playground-mobile-artifacts-return")
    expect(source).toContain("closeArtifactsWithFocusReturn")
    expect(source).toContain("playground:regions.returnToTimeline")
    expect(source).toContain("playground:regions.closeArtifactsDrawer")
    expect(source).toContain("resolvePlaygroundShortcutAction")
    expect(source).toContain("artifactsTriggerRef")
    expect(source).toContain("tldw:focus-artifacts-trigger")
    expect(source).toContain("tldw:toggle-compare-mode")
    expect(source).toContain("tldw:toggle-mode-launcher")
    expect(source).toContain("playground:search.placeholder")
    expect(source).toContain("collectThreadSearchMatches")
    expect(source).toContain("threadSearchInputRef")
    expect(source).toContain("playground-shortcuts-help-trigger")
    expect(source).toContain("playground-shortcuts-help-panel")
    expect(source).toContain("tldw:open-playground-shortcuts")
    expect(source).toContain("event.key === \"?\"")
    expect(source).toContain("event.key.toLowerCase() === \"f\"")
    expect(source).toContain("searchQuery={threadSearchQuery.trim()}")
    expect(source).toContain("compositionPreviewSummary")
    expect(source).toContain("buildPlaygroundCompositionPreviewSummary")
  })
})

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

  it("keeps the mobile chat surface compact by hiding secondary chrome", () => {
    const playgroundSourcePath = path.resolve(__dirname, "../Playground.tsx")
    const formSourcePath = path.resolve(__dirname, "../PlaygroundForm.tsx")
    const statusSourcePath = path.resolve(__dirname, "../PlaygroundStatusStrip.tsx")
    const playgroundSource = fs.readFileSync(playgroundSourcePath, "utf8")
    const formSource = fs.readFileSync(formSourcePath, "utf8")
    const statusSource = fs.readFileSync(statusSourcePath, "utf8")

    expect(playgroundSource).toContain("hidden sm:inline-flex")
    expect(playgroundSource).toContain("px-2 pt-1 sm:px-4 sm:pt-2")
    expect(playgroundSource).toContain("px-2 sm:px-4")
    expect(playgroundSource).toContain('!isMobileViewport ? (')
    expect(formSource).toContain(
      "grid grid-cols-[auto_minmax(0,1fr)] items-end gap-2"
    )
    expect(formSource).toContain(
      "col-span-2 flex shrink-0 justify-end self-end"
    )
    expect(formSource).toContain("px-2 pb-3 sm:px-4 sm:pb-6")
    expect(formSource).toContain("rounded-xl")
    expect(formSource).toContain("p-2 text-text shadow-sm")
    expect(formSource).toContain("sm:rounded-3xl sm:p-3 sm:shadow-card")
    expect(statusSource).toContain("justify-between gap-1 border-t")
    expect(statusSource).toContain("px-2 py-1")
    expect(statusSource).toContain("sm:gap-3 sm:px-3 sm:py-2")
    expect(statusSource).toContain("hidden sm:inline-flex")
    expect(statusSource).toContain("hidden max-w-[18rem] truncate font-medium text-text sm:inline")
  })

  it("keeps cockpit mode controls inside the chat utility row instead of a top rail", () => {
    const playgroundSourcePath = path.resolve(__dirname, "../Playground.tsx")
    const cockpitShellSourcePath = path.resolve(
      __dirname,
      "../PlaygroundCockpitShell.tsx"
    )
    const playgroundSource = fs.readFileSync(playgroundSourcePath, "utf8")
    const cockpitShellSource = fs.readFileSync(cockpitShellSourcePath, "utf8")

    expect(cockpitShellSource).not.toContain("<header")
    expect(cockpitShellSource).toContain(
      'data-testid="playground-cockpit-mode-summary"'
    )
    expect(cockpitShellSource).toContain("sr-only")
    expect(playgroundSource).toContain(
      'data-testid="playground-chat-layout-mode-trigger"'
    )

    const modeTriggerIndex = playgroundSource.indexOf(
      'data-testid="playground-chat-layout-mode-trigger"'
    )
    const shortcutsTriggerIndex = playgroundSource.indexOf(
      'data-testid="playground-shortcuts-help-trigger"'
    )
    expect(modeTriggerIndex).toBeGreaterThan(-1)
    expect(shortcutsTriggerIndex).toBeGreaterThan(-1)
    expect(modeTriggerIndex).toBeLessThan(shortcutsTriggerIndex)
  })
})

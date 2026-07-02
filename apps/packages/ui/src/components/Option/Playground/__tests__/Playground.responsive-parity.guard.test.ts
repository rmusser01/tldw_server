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
    const playgroundSource = fs.readFileSync(playgroundSourcePath, "utf8")
    const formSource = fs.readFileSync(formSourcePath, "utf8")

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
    expect(formSource).toContain(
      "px-2 pb-0 sm:px-4 sm:pb-0"
    )
    expect(formSource).toContain("rounded-xl")
    expect(formSource).toContain("p-2 text-text shadow-sm")
    expect(formSource).toContain("sm:rounded-3xl sm:p-3 sm:shadow-card")
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

  it("keeps cockpit status inside the composer instead of a bottom shell rail", () => {
    const playgroundSourcePath = path.resolve(__dirname, "../Playground.tsx")
    const cockpitShellSourcePath = path.resolve(
      __dirname,
      "../PlaygroundCockpitShell.tsx"
    )
    const formSourcePath = path.resolve(__dirname, "../PlaygroundForm.tsx")
    const toolbarSourcePath = path.resolve(__dirname, "../ComposerToolbar.tsx")
    const contextItemsSourcePath = path.resolve(
      __dirname,
      "../hooks/usePlaygroundContextItems.ts"
    )
    const playgroundSource = fs.readFileSync(playgroundSourcePath, "utf8")
    const cockpitShellSource = fs.readFileSync(cockpitShellSourcePath, "utf8")
    const formSource = fs.readFileSync(formSourcePath, "utf8")
    const toolbarSource = fs.readFileSync(toolbarSourcePath, "utf8")
    const contextItemsSource = fs.readFileSync(contextItemsSourcePath, "utf8")

    expect(playgroundSource).not.toContain("PlaygroundStatusStrip")
    expect(playgroundSource).not.toContain("cockpitStatusStrip")
    expect(playgroundSource).not.toContain("statusStrip={")
    expect(cockpitShellSource).not.toContain("statusStrip")
    expect(cockpitShellSource).not.toContain("playground-cockpit-status-strip")

    expect(playgroundSource).toContain("composerMessageCount")
    expect(formSource).toContain("composerMessageCount")
    expect(formSource).toContain("keepComposerToolbarVisible")
    expect(formSource).toContain("max-h-[480px] opacity-100 visible")
    expect(contextItemsSource).toContain('id: "messageCount"')
    expect(contextItemsSource).toContain('id: "sessionStatus"')
    expect(toolbarSource).toContain("composer-message-count-chip")
    expect(toolbarSource).toContain("composer-session-status-chip")
  })
})

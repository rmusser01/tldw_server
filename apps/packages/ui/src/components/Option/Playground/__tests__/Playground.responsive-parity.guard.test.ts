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
    expect(source).toContain("tldw:open-shortcuts-modal")
    expect(source).not.toContain("playground-shortcuts-help-panel")
    const helpHost = fs.readFileSync(path.resolve(__dirname, "../../../Common/PageHelpModalHost.tsx"), "utf8")
    expect(helpHost).toContain("tldw:open-playground-shortcuts")
    expect(helpHost).toContain("shouldOpenShortcutsHelp(event)")
    expect(source).toContain("event.key.toLowerCase() === \"f\"")
    // Highlights follow the open search bar only (CS-11).
    expect(source).toContain(
      "searchQuery={threadSearchHighlightQuery(threadSearchOpen, threadSearchQuery)}"
    )
    expect(source).toContain("compositionPreviewSummary")
    expect(source).toContain("buildPlaygroundCompositionPreviewSummary")
  })

  it("keeps the mobile chat surface compact by hiding secondary chrome", () => {
    const playgroundSourcePath = path.resolve(__dirname, "../Playground.tsx")
    const formSourcePath = path.resolve(__dirname, "../PlaygroundForm.tsx")
    const playgroundSource = fs.readFileSync(playgroundSourcePath, "utf8")
    const formSource = fs.readFileSync(formSourcePath, "utf8")

    expect(playgroundSource).toContain("hidden sm:inline-flex")
    expect(playgroundSource).toContain("border-b border-border/60 px-2 py-1 sm:px-4")
    expect(playgroundSource).toContain("px-2 sm:px-4")
    expect(playgroundSource).toContain('!isMobileViewport && !focusModeActive ? (')
    const toolbarSource = fs.readFileSync(path.resolve(__dirname, "../ComposerToolbar.tsx"), "utf8")
    expect(formSource).toContain("flex w-full min-w-0 flex-col")
    expect(formSource).toContain("rounded-2xl border border-border bg-surface p-3")
    expect(toolbarSource).toContain('data-testid="composer-bottom-bar"')
    expect(toolbarSource).toContain("flex flex-wrap items-center justify-between")
    expect(toolbarSource).toContain('hidden={!optionsExpanded}')
    expect(toolbarSource).toContain("max-h-[35dvh] overflow-y-auto")
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
    expect(cockpitShellSource).not.toContain("PlaygroundStatusStrip")
    expect(playgroundSource).not.toContain("cockpitStatusStrip")
    expect(playgroundSource).not.toContain("statusStrip={")
    expect(cockpitShellSource).not.toContain("statusStrip")
    expect(cockpitShellSource).not.toContain("playground-cockpit-status-strip")

    expect(playgroundSource).toContain("composerMessageCount")
    expect(formSource).toContain("composerMessageCount")
    expect(formSource).toContain("optionsLoading={composerOptionsLoading}")
    expect(toolbarSource).toContain('data-testid="composer-bottom-bar"')
    expect(contextItemsSource).toContain('id: "messageCount"')
    expect(contextItemsSource).toContain('id: "sessionStatus"')
    expect(toolbarSource).toContain("neutralContextItems.map")
    expect(toolbarSource).toContain("composer-session-status-chip")
  })
})

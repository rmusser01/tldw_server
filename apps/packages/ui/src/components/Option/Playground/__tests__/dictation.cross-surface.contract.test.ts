import fs from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

const resolveSidepanelFormPath = () => {
  const candidates = [
    path.resolve(__dirname, "../../Sidepanel/Chat/form.tsx"),
    path.resolve(process.cwd(), "src/components/Sidepanel/Chat/form.tsx"),
    path.resolve(
      process.cwd(),
      "../packages/ui/src/components/Sidepanel/Chat/form.tsx"
    ),
    path.resolve(
      process.cwd(),
      "apps/packages/ui/src/components/Sidepanel/Chat/form.tsx"
    )
  ]
  return candidates.find((candidate) => fs.existsSync(candidate))
}

const resolveSharedVoiceHookPath = () => {
  const candidates = [
    path.resolve(
      __dirname,
      "../../Chat/composer/hooks/useComposerVoiceChat.ts"
    ),
    path.resolve(
      process.cwd(),
      "src/components/Chat/composer/hooks/useComposerVoiceChat.ts"
    ),
    path.resolve(
      process.cwd(),
      "apps/packages/ui/src/components/Chat/composer/hooks/useComposerVoiceChat.ts"
    )
  ]
  return candidates.find((candidate) => fs.existsSync(candidate))
}

describe("dictation cross-surface contract", () => {
  // After Phase 2 extraction, both surfaces share the dictation orchestration
  // through `useComposerVoiceChat`. The shared hook owns the source-resolution
  // pipeline, the strategy bridge, and the diagnostics snapshot — so the
  // contract assertions live there. We additionally assert that both
  // Playground and Sidepanel surfaces consume the shared hook so the contract
  // can't be silently re-divided.

  it("keeps the shared dictation source path co-located in the composer voice hook", () => {
    const sharedHookPath = resolveSharedVoiceHookPath()
    if (!sharedHookPath) {
      throw new Error("Unable to locate shared composer voice hook")
    }
    const sharedSource = fs.readFileSync(sharedHookPath, "utf8")

    expect(sharedSource).toContain('useAudioSourcePreferences("dictation")')
    expect(sharedSource).toContain("resolveAudioCapturePlan({")
    expect(sharedSource).toContain(
      'dictationModeOverride === "browser" && !browserDictationCompatible'
    )
    expect(sharedSource).toContain(
      'canUseServerStt\n        ? ("server" as const)\n        : ("unavailable" as const)'
    )
    expect(sharedSource).toContain("resolvedModeOverride,")
    expect(sharedSource).toContain("resolvedDictationSourcePreference.sourceKind")
    expect(sharedSource).toContain("audioInputDevices.some(")
    expect(sharedSource).toContain("dictationSourceReady")
    expect(sharedSource).toContain("pendingDictationStart")
    expect(sharedSource).toContain("hasAudioCatalogSettled")
    expect(sharedSource).toContain(
      "serverDictationErrorBridgeRef.current = dictationStrategy.recordServerError"
    )
    expect(sharedSource).toContain(
      "serverDictationSuccessBridgeRef.current = dictationStrategy.recordServerSuccess"
    )
    expect(sharedSource).toContain(
      "const snapshot = dictationDiagnosticsSnapshotRef.current"
    )
    expect(sharedSource).toContain("requestedSourceKind:")
    expect(sharedSource).toContain("resolvedSourceKind:")
  })

  it("requires both Playground and Sidepanel surfaces to consume the shared composer voice hook", () => {
    const playgroundVoiceChatPath = path.resolve(
      __dirname,
      "../hooks/usePlaygroundVoiceChat.ts"
    )
    const sidepanelFormPath = resolveSidepanelFormPath()
    if (!sidepanelFormPath) {
      throw new Error("Unable to locate Sidepanel chat form source")
    }

    const playgroundSource = fs.readFileSync(playgroundVoiceChatPath, "utf8")
    const sidepanelSource = fs.readFileSync(sidepanelFormPath, "utf8")

    expect(playgroundSource).toContain(
      'from "@/components/Chat/composer/hooks/useComposerVoiceChat"'
    )
    expect(playgroundSource).toContain("useComposerVoiceChat(")
    expect(sidepanelSource).toContain(
      'from "@/components/Chat/composer/hooks/useComposerVoiceChat"'
    )
    expect(sidepanelSource).toContain("useComposerVoiceChat(")
  })

  it("routes dictation controls through unified toggle intent handlers in the shared hook", () => {
    const sharedHookPath = resolveSharedVoiceHookPath()
    if (!sharedHookPath) {
      throw new Error("Unable to locate shared composer voice hook")
    }
    const sharedSource = fs.readFileSync(sharedHookPath, "utf8")

    expect(sharedSource).toContain(
      "const handleDictationToggle = React.useCallback(() => {"
    )
    expect(sharedSource).toContain("switch (dictationToggleIntent)")
    expect(sharedSource).toContain(
      "startServerDictation(requestedServerDictationSource)"
    )
  })

  it("keeps transcript insertion attached to the composer message in both surfaces", () => {
    const playgroundVoiceChatPath = path.resolve(
      __dirname,
      "../hooks/usePlaygroundVoiceChat.ts"
    )
    const sidepanelFormPath = resolveSidepanelFormPath()
    if (!sidepanelFormPath) {
      throw new Error("Unable to locate Sidepanel chat form source")
    }

    const playgroundSource = fs.readFileSync(playgroundVoiceChatPath, "utf8")
    const sidepanelSource = fs.readFileSync(sidepanelFormPath, "utf8")

    // Playground keeps its collapse-aware transcript handler.
    expect(playgroundSource).toContain(
      'setMessageValue(text, { collapseLarge: true, forceCollapse: true })'
    )
    // Sidepanel keeps its plain-text transcript handler.
    expect(sidepanelSource).toContain(
      'form.setFieldValue("message", text)'
    )
  })
})

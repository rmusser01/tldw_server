import fs from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

describe("PlaygroundForm signal surface guard", () => {
  it("keeps mention/slash affordance and state-change/degradation signals", () => {
    const formSourcePath = path.resolve(__dirname, "../PlaygroundForm.tsx")
    const chipSourcePath = path.resolve(
      __dirname,
      "../AttachedResearchContextChip.tsx"
    )
    const researchRunsSourcePath = path.resolve(
      __dirname,
      "../../../../../../../tldw-frontend/lib/api/researchRuns.ts"
    )
    const contextPanelSourcePath = path.resolve(
      __dirname,
      "../ContextFootprintPanel.tsx"
    )
    const recommendationsPanelSourcePath = path.resolve(
      __dirname,
      "../ModelRecommendationsPanel.tsx"
    )
    const chipSource = fs.readFileSync(chipSourcePath, "utf8")
    const formSource = fs.readFileSync(formSourcePath, "utf8")
    const researchRunsSource = fs.readFileSync(researchRunsSourcePath, "utf8")
    const contextPanelSource = fs.readFileSync(contextPanelSourcePath, "utf8")
    const recommendationsPanelSource = fs.readFileSync(
      recommendationsPanelSourcePath,
      "utf8"
    )

    expect(formSource).toContain("Type a message... (/ commands, @ mentions)")
    expect(formSource).toContain("Changed since last send:")
    expect(formSource).toContain("playground:composer.providerDegraded")
    expect(formSource).toContain("playground:composer.presetChanged")
    expect(formSource).toContain("playground:composer.jsonModeEnabledNotice")
    expect(formSource).toContain("playground:composer.jsonModeHint")
    expect(formSource).toContain("playground:composer.characterAppliesNextTurn")
    expect(formSource).toContain("playground:composer.characterPendingNotice")
    expect(formSource).toContain("ContextFootprintPanel")
    expect(formSource).toContain("playground:composer.compareActivationTitle")
    expect(formSource).toContain("playground:composer.compareActivationBody")
    expect(formSource).toContain(
      "playground:composer.compareActivationInteroperability"
    )
    expect(formSource).toContain("compare-interoperability-notices")
    expect(formSource).toContain(
      "playground:composer.validationCompareMinModelsInline"
    )
    expect(formSource).toContain("tldw:playground-starter-selected")
    expect(formSource).toContain('{ mode: "voice" }')
    expect(formSource).toContain("useMobileComposerViewport")
    expect(formSource).toContain("data-mobile-keyboard")
    expect(formSource).toContain("scrollMarginBottom")
    expect(formSource).toContain("previousSendStateRef")
    expect(formSource).toContain("onMutate: () => ({")
    expect(formSource).toContain("errorKeyToDismiss: chatErrorBanner?.key")
    expect(formSource).toContain("onSuccess: (data, _variables, context) =>")
    expect(formSource).toContain("const result = normalizeChatSubmitResult(data);")
    expect(formSource).toContain("isChatSubmitSuccess(result)")
    expect(formSource).toContain("onError: (error) =>")
    expect(formSource).toContain("textAreaFocus()")
    expect(formSource).toContain("tldw:focus-composer")
    expect(formSource).toContain("el.focus()")
    expect(formSource).toContain("tldw:toggle-compare-mode")
    expect(formSource).toContain("tldw:toggle-mode-launcher")
    expect(formSource).toContain("playground:composer.context.sessionStatus")
    expect(formSource).toContain("playground:composer.context.truncationRisk")
    expect(formSource).toContain("playground:composer.context.contextMix")
    expect(formSource).toContain("isSessionDegraded")
    expect(formSource).not.toContain('id: "modelCapabilities"')
    expect(formSource).not.toContain('id: "summaryCheckpoint"')
    expect(formSource).not.toContain('id: "conversationState"')
    expect(formSource).not.toContain('id: "imageEventSync"')
    expect(formSource).toContain("playground:composer.conflict.summaryCheckpointBudget")
    expect(formSource).toContain("playground:composer.conflict.contextFootprint")
    expect(formSource).toContain("buildConversationSummaryCheckpointPrompt")
    expect(formSource).toContain("evaluateSummaryCheckpointSuggestion")
    expect(formSource).toContain("resolveTokenBudgetRisk")
    expect(formSource).toContain("playground:tokens.truncationRisk")
    expect(formSource).toContain("playgroundStartupTemplateBundles")
    expect(formSource).toContain('data-testid="startup-template-controls"')
    expect(formSource).toContain("SessionInsightsPanel")
    expect(formSource).toContain("ModelRecommendationsPanel")
    expect(formSource).toContain("buildSessionInsights")
    expect(formSource).toContain("buildModelRecommendations")
    expect(formSource).toContain("buildCompareInteroperabilityNotices")
    expect(formSource).toContain("AttachedResearchContextChip")
    expect(formSource).toContain("attachedResearchContext")
    expect(formSource).toContain("attachedResearchContextPinned")
    expect(formSource).toContain("onRemoveAttachedResearchContext")
    expect(formSource).toContain("onPinAttachedResearchContext")
    expect(formSource).toContain("onRestorePinnedResearchContext")
    expect(formSource).toContain('data-testid="pinned-research-fallback-card"')
    expect(formSource).toContain('data-testid="pinned-research-history-block"')
    expect(formSource).toContain("Use now")
    expect(formSource).toContain(
      "This thread keeps this research as its default context."
    )
    expect(formSource).toContain("researchContext:")
    expect(formSource).toContain("Attached Research Context")
    expect(formSource).toContain("Reset to Attached Run")
    expect(formSource).toContain("Apply")
    expect(formSource).toContain("Follow-up Research")
    expect(formSource).toContain("Follow up on this research:")
    expect(formSource).toContain("Follow up")
    expect(formSource).toContain("Prepare follow-up?")
    expect(formSource).toContain("Prepare follow-up")
    expect(formSource).toContain("Use attached research as background")
    expect(formSource).toContain("Start research")
    expect(researchRunsSource).toContain("follow_up?:")
    expect(chipSource).toContain("Edit attached research")
    expect(chipSource).toContain("Pinned research")
    expect(chipSource).toContain("Unpin")
    expect(chipSource).toContain("Follow up")
    expect(formSource).toContain("playground:insights.modalTitle")
    expect(formSource).toContain("startup-template-preview-modal")
    expect(formSource).toContain(
      "playground:composer.startupTemplatePreviewTitle"
    )
    expect(formSource).toContain("resolveStartupTemplatePrompt")
    expect(formSource).toContain("image-refine-with-llm")
    expect(formSource).toContain("image-prompt-refine-diff")
    expect(formSource).toContain("applyRefinedImagePromptCandidate")
    expect(formSource).toContain("imageGenerationRefine")
    expect(contextPanelSource).toContain("playground:tokens.contextBreakdownTitle")
    expect(recommendationsPanelSource).toContain(
      'data-testid="model-recommendations-panel"'
    )
    expect(recommendationsPanelSource).toContain(
      "playground:composer.recommendationsTitle"
    )
  })

  it("routes the character starter through character selection before scene controls", () => {
    const formSourcePath = path.resolve(__dirname, "../PlaygroundForm.tsx")
    const formSource = fs.readFileSync(formSourcePath, "utf8")
    const characterStarterStart = formSource.indexOf('if (mode === "character")')
    const ragStarterStart = formSource.indexOf('if (mode === "rag" || mode === "knowledge")')
    const characterStarterBlock = formSource.slice(
      characterStarterStart,
      ragStarterStart
    )

    expect(characterStarterStart).toBeGreaterThanOrEqual(0)
    expect(ragStarterStart).toBeGreaterThan(characterStarterStart)
    expect(formSource).toContain("dispatchOpenAssistantSelect")
    expect(characterStarterBlock).toContain('source: "playground-starter"')
    expect(characterStarterBlock).not.toContain("setOpenActorSettings(true)")
  })
})

import { readFileSync } from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

const source = (relativePath: string) =>
  readFileSync(path.resolve(process.cwd(), relativePath), "utf8")

const composerInputSource = source(
  "src/components/Option/Playground/hooks/useComposerInput.tsx"
)
const playgroundFormSource = source(
  "src/components/Option/Playground/PlaygroundForm.tsx"
)
const sidepanelFormSource = source("src/components/Sidepanel/Chat/form.tsx")
const toolbarSource = source(
  "src/components/Option/Playground/ComposerToolbar.tsx"
)
const controlRowSource = source("src/components/Sidepanel/Chat/ControlRow.tsx")
const controlAreaSource = source(
  "src/components/Sidepanel/Chat/SidepanelComposerControlArea.tsx"
)
const queueManagementSource = source(
  "src/components/Option/Playground/hooks/usePlaygroundQueueManagement.ts"
)
const playgroundSubmitSource = source(
  "src/components/Option/Playground/hooks/usePlaygroundSubmit.ts"
)
const currentChatModelSettingsSource = source(
  "src/components/Common/Settings/CurrentChatModelSettings.tsx"
)
const modelBasicsTabSource = source(
  "src/components/Common/Settings/tabs/ModelBasicsTab.tsx"
)
const sidepanelEntrySource = source("src/entries/sidepanel/App.tsx")
const sharedSidepanelAppSource = source("src/entries/shared/sidepanel-app.tsx")
const sidepanelRouteRegistrySource = source(
  "src/routes/sidepanel-route-registry.tsx"
)
const sidepanelChatSource = source("src/routes/sidepanel-chat.tsx")
const quickChatPopoutSource = source("src/routes/option-quick-chat-popout.tsx")

const occurrences = (value: string, fragment: string) =>
  value.split(fragment).length - 1

const promptAssistConfig = (value: string) =>
  value.match(
    /const promptAssistAction = \(\s*<PromptAssistComposerAction([\s\S]*?)\/>\s*\)/
  )?.[1] ?? ""

describe("PromptAssistComposerAction real shell wiring", () => {
  it("projects the shared owner revision through Playground without a second draft", () => {
    expect(composerInputSource).toMatch(
      /const\s*\{[\s\S]*?form,[\s\S]*?messageRevision,[\s\S]*?\}\s*=\s*composerText/
    )
    expect(composerInputSource).toMatch(
      /return \{[\s\S]*messageRevision,[\s\S]*draftSaved,/
    )
    expect(
      occurrences(playgroundFormSource, "<PromptAssistComposerAction")
    ).toBe(1)
    const config = promptAssistConfig(playgroundFormSource)
    expect(config).toContain("form={form}")
    expect(config).toContain("messageRevision={messageRevision}")
    expect(config).toContain("promptAssistMutation={promptAssistMutation}")
    expect(config).toContain(
      "promptAssistSavedAttemptId={promptAssistSavedAttemptId}"
    )
    expect(config).toContain("selected_model: selectedModel")
    expect(config).toMatch(
      /provider_hint:\s*currentChatModelSettings\.apiProvider/
    )
    expect(config).toContain("promptAssistBackendKey={promptAssistBackendKey}")
    expect(config).toContain(
      "promptAssistAuthorizationRevision={promptAssistAuthorizationRevision}"
    )
    expect(config).toMatch(
      /promptAssistContextKey=\{serverChatId[\s\S]*historyId[\s\S]*"local:playground-draft"\}/
    )
    expect(config).toContain("sending={isSending}")
    expect(config).toMatch(/\bsurfaceOpen\s/)
    expect(config).toContain("narrow={isMobileViewport}")
    expect(config).toContain("onReturnFocus={textAreaFocus}")
  })

  it("passes the same existing owner and route into the narrow Sidepanel control area", () => {
    expect(sidepanelFormSource).toMatch(
      /const\s*\{[\s\S]*?form,[\s\S]*?messageRevision,[\s\S]*?textAreaFocus: promptAssistReturnFocus,[\s\S]*?draftSaved,[\s\S]*?clearDraft[\s\S]*?\}\s*=\s*useComposerText/
    )
    expect(
      occurrences(sidepanelFormSource, "<PromptAssistComposerAction")
    ).toBe(1)
    const config = promptAssistConfig(sidepanelFormSource)
    expect(config).toContain("form={form}")
    expect(config).toContain("messageRevision={messageRevision}")
    expect(config).toContain("promptAssistMutation={promptAssistMutation}")
    expect(config).toContain(
      "promptAssistSavedAttemptId={promptAssistSavedAttemptId}"
    )
    expect(config).toContain("selected_model: selectedModel")
    expect(config).toMatch(
      /provider_hint:\s*currentChatApiProvider \?\? undefined/
    )
    expect(config).toContain("promptAssistBackendKey={promptAssistBackendKey}")
    expect(config).toMatch(
      /promptAssistAuthorizationRevision=\{\s*promptAssistAuthorizationRevision\s*\}/
    )
    expect(config).toMatch(
      /promptAssistContextKey=\{serverChatId[\s\S]*historyId[\s\S]*"local:sidepanel-draft"\}/
    )
    expect(config).toContain("sending={isSending || streaming}")
    expect(config).toMatch(/\bsurfaceOpen\s+narrow\s/)
    expect(config).toContain("onReturnFocus={promptAssistReturnFocus}")
  })

  it("completes only the exact current submit attempt", () => {
    expect(playgroundSubmitSource).toMatch(
      /const promptAssistAttemptId = beginPromptAssistReset\(\)[\s\S]*isChatSubmitSuccess[\s\S]*markPromptAssistAttemptSaved\(promptAssistAttemptId\)/
    )
    expect(sidepanelFormSource).toMatch(
      /promptAssistAttemptId = beginPromptAssistReset\(\)[\s\S]*afterSend: \(result\)[\s\S]*isChatSubmitSuccess[\s\S]*markPromptAssistAttemptSaved\(promptAssistAttemptId\)/
    )
    expect(playgroundFormSource).not.toContain("markPromptAssistSaved")
    expect(sidepanelFormSource).not.toContain("markPromptAssistSaved")
  })

  it("creates and completes one exact attempt on queue enqueue", () => {
    expect(playgroundFormSource).toMatch(
      /onEnqueueSuccess: \(\) => \{[\s\S]*beginPromptAssistReset\(\)[\s\S]*markPromptAssistAttemptSaved\(attemptId\)/
    )
    expect(queueManagementSource).toMatch(
      /handleEnqueueSuccess[\s\S]*onEnqueueSuccess\?\.\(\)/
    )
    expect(sidepanelFormSource).toMatch(
      /handleQueueEnqueueSuccess[\s\S]*beginPromptAssistReset\(\)[\s\S]*markPromptAssistAttemptSaved\(promptAssistAttemptId\)/
    )
  })

  it("owns one WebUI action immediately before external Send, outside the toolbar", () => {
    expect(occurrences(toolbarSource, "<PromptAssistComposerAction")).toBe(0)
    expect(occurrences(controlRowSource, "<PromptAssistComposerAction")).toBe(0)
    expect(
      occurrences(playgroundFormSource, "<PromptAssistComposerAction")
    ).toBe(1)
    expect(occurrences(playgroundFormSource, "{promptAssistAction}")).toBe(1)
    expect(playgroundFormSource).toMatch(
      /data-testid="composer-inline-send-control"[\s\S]*?>\s*\{promptAssistAction\}\s*\{sendControl\}\s*<\/div>/
    )
    expect(playgroundFormSource).toMatch(
      /const sendControl = \(\s*<PlaygroundSendControl[\s\S]*?onSubmitForm=\{handleComposerSend\}/
    )
  })

  it("supplies the same extension action to each mutually exclusive Send cluster", () => {
    expect(
      occurrences(sidepanelFormSource, "<PromptAssistComposerAction")
    ).toBe(1)
    expect(occurrences(controlAreaSource, "{promptAssistAction}")).toBe(1)
    expect(controlAreaSource).toMatch(
      /data-testid="sidepanel-send-action-cluster"[\s\S]*?>\s*\{promptAssistAction\}\s*\{children\}/
    )
    const clusters = [
      ...sidepanelFormSource.matchAll(
        /<SidepanelComposerControlArea([\s\S]*?)<\/SidepanelComposerControlArea>/g
      )
    ]
    expect(clusters).toHaveLength(3)
    for (const [, cluster] of clusters) {
      expect(
        occurrences(cluster, "promptAssistAction={promptAssistAction}")
      ).toBe(1)
      expect(cluster).toContain(
        'type={shouldQueuePrimaryAction ? "button" : "submit"}'
      )
      expect(cluster).toContain("void submitForm()")
    }
    const sharedControls =
      sidepanelFormSource.match(
        /const composerControlAreaNode = \(([\s\S]*?)if \(nextgenComposerEnabled\)/
      )?.[1] ?? ""
    expect(sharedControls).toContain("{isProMode ? (")
    expect(occurrences(sharedControls, "<SidepanelComposerControlArea")).toBe(2)
    const variants = [
      ...sidepanelFormSource.matchAll(
        /<ChatComposer\s+variant="([^"]+)"([\s\S]*?)\/>/g
      )
    ]
    expect(variants.map(([, variant]) => variant)).toEqual(["v5", "v3", "v1"])
    for (const [, variant, props] of variants) {
      if (variant === "v5") {
        expect(props).toContain("sendSlot={v5SendSlot}")
        expect(props).not.toContain("bottomBarSlot=")
      } else {
        expect(props).toContain("bottomBarSlot={composerControlAreaNode}")
        expect(props).not.toContain("sendSlot=")
      }
    }
    expect(sidepanelFormSource).toMatch(
      /const v5SendSlot = \(\s*<SidepanelComposerControlArea[\s\S]*?variant="v5"[\s\S]*?sendSlot=\{v5SendSlot\}/
    )
    expect(sidepanelFormSource).toMatch(
      /return \(\s*<>\s*\{composerTextareaShellNode\}\s*\{composerInlineMessagesNode\}\s*\{composerControlAreaNode\}/
    )
  })

  it("wires the extension sidepanel to the shared composer adapter and does not mislabel the separate quick-chat pop-out", () => {
    expect(sidepanelEntrySource).toContain(
      'export { SidepanelApp as default } from "@/entries/shared/sidepanel-app"'
    )
    expect(sharedSidepanelAppSource).toContain("<SidepanelRouteShell />")
    expect(sidepanelRouteRegistrySource).toMatch(
      /path:\s*"\/chat"[\s\S]*element:\s*<SidepanelChat\s*\/>/
    )
    expect(sidepanelChatSource).toContain(
      'import { SidepanelForm } from "~/components/Sidepanel/Chat/form"'
    )
    expect(
      occurrences(sidepanelFormSource, "<PromptAssistComposerAction")
    ).toBe(1)

    // The product's pop-out entry is a separate Quick Chat surface, not a
    // recipe-capable composer. Contract tests must not relabel it as one.
    expect(quickChatPopoutSource).toContain("<QuickChatInput")
    expect(quickChatPopoutSource).not.toContain("PromptAssistComposerAction")
  })

  it("routes Sidepanel model recovery to settings that contain model selection", () => {
    expect(sidepanelFormSource).toContain(
      "onSelectModel={() => setOpenModelSettings(true)}"
    )
    expect(sidepanelFormSource).toMatch(
      /\{openModelSettings && \([\s\S]*<CurrentChatModelSettings/
    )
    expect(currentChatModelSettingsSource).toContain("<ModelBasicsTab")
    expect(modelBasicsTabSource).toContain('defaultValue: "API / model"')
  })
})

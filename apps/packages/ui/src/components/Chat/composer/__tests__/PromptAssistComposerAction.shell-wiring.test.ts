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

const occurrences = (value: string, fragment: string) =>
  value.split(fragment).length - 1

const promptAssistConfig = (value: string, followingProp: string) =>
  value.match(
    new RegExp(
      `promptAssistComposer=\\{\\{([\\s\\S]*?)\\}\\}\\s+${followingProp}`
    )
  )?.[1] ?? ""

describe("PromptAssistComposerAction real shell wiring", () => {
  it("projects the shared owner revision through Playground without a second draft", () => {
    expect(composerInputSource).toMatch(
      /const\s*\{[\s\S]*?form,[\s\S]*?messageRevision,[\s\S]*?\}\s*=\s*composerText/
    )
    expect(composerInputSource).toMatch(
      /return \{[\s\S]*messageRevision,[\s\S]*draftSaved,/
    )
    expect(occurrences(playgroundFormSource, "promptAssistComposer={{")).toBe(1)
    const config = promptAssistConfig(
      playgroundFormSource,
      "showServerPersistenceHint"
    )
    expect(config).toContain("form,")
    expect(config).toContain("messageRevision,")
    expect(config).toContain("promptAssistMutation,")
    expect(config).toContain("promptAssistSavedAttemptId,")
    expect(config).toContain("selected_model: selectedModel")
    expect(config).toMatch(
      /provider_hint:\s*currentChatModelSettings\.apiProvider/
    )
    expect(config).toContain("promptAssistBackendKey,")
    expect(config).toContain("sending: isSending")
    expect(config).toContain("surfaceOpen: true")
    expect(config).toContain("onReturnFocus: textAreaFocus")
  })

  it("passes the same existing owner and route into the narrow Sidepanel control area", () => {
    expect(sidepanelFormSource).toMatch(
      /const\s*\{[\s\S]*?form,[\s\S]*?messageRevision,[\s\S]*?textAreaFocus: promptAssistReturnFocus,[\s\S]*?draftSaved,[\s\S]*?clearDraft[\s\S]*?\}\s*=\s*useComposerText/
    )
    expect(
      occurrences(sidepanelFormSource, "<PromptAssistComposerAction")
    ).toBe(1)
    expect(sidepanelFormSource).toMatch(
      /<SidepanelComposerControlArea[\s\S]*?<PromptAssistComposerAction[\s\S]*?promptAssistMutation=\{promptAssistMutation\}[\s\S]*?promptAssistSavedAttemptId=\{[\s\S]*?promptAssistSavedAttemptId[\s\S]*?\}[\s\S]*?selected_model: selectedModel[\s\S]*?provider_hint:[\s\S]*?currentChatApiProvider \?\? undefined[\s\S]*?sending=\{isSending \|\| streaming\}[\s\S]*?onReturnFocus=\{promptAssistReturnFocus\}[\s\S]*?>[\s\S]*?\{isProMode \?/
    )
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

  it("owns exactly one action in each shared node rather than any variant", () => {
    expect(occurrences(toolbarSource, "<PromptAssistComposerAction")).toBe(1)
    expect(occurrences(controlRowSource, "<PromptAssistComposerAction")).toBe(0)
    expect(playgroundFormSource).not.toContain("<PromptAssistComposerAction")
    expect(
      occurrences(sidepanelFormSource, "<PromptAssistComposerAction")
    ).toBe(1)
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

export type CompareInteroperabilityNotice = {
  id: string
  tone: "neutral" | "warning"
  text: string
}

type Params = {
  t: (key: string, defaultValueOrOptions?: any, options?: any) => string
  characterName?: string | null
  pinnedSourceCount: number
  webSearch: boolean
  hasPromptContext: boolean
  jsonMode: boolean
  voiceChatEnabled: boolean
}

const toText = (value: unknown): string =>
  typeof value === "string" ? value : String(value)

export const buildCompareInteroperabilityNotices = (
  params: Params
): CompareInteroperabilityNotice[] => {
  const {
    t,
    characterName,
    pinnedSourceCount,
    webSearch,
    hasPromptContext,
    jsonMode,
    voiceChatEnabled
  } = params

  const notices: CompareInteroperabilityNotice[] = []

  if (voiceChatEnabled) {
    notices.push({
      id: "voice",
      tone: "warning",
      text: toText(
        t(
          "playground:composer.compareInteropVoice",
          "Voice mode is enabled. Compare playback timing may vary by model, so review text responses before choosing a winner."
        )
      )
    })
  }

  if (characterName && characterName.trim().length > 0) {
    notices.push({
      id: "character",
      tone: "neutral",
      text: toText(
        t(
          "playground:composer.compareInteropCharacter",
          "Character behavior applies to all selected models ({{name}}).",
          { name: characterName.trim() } as any
        )
      )
    })
  }

  if (pinnedSourceCount > 0) {
    notices.push({
      id: "pinned",
      tone: "neutral",
      text: toText(
        t(
          "playground:composer.compareInteropPinned",
          "{{count}} pinned sources are shared across all compare responses.",
          { count: pinnedSourceCount } as any
        )
      )
    })
  }

  if (webSearch) {
    notices.push({
      id: "web-search",
      tone: "neutral",
      text: toText(
        t(
          "playground:composer.compareInteropWebSearch",
          "Web search/tool calls are shared across compare models when enabled."
        )
      )
    })
  }

  if (hasPromptContext) {
    notices.push({
      id: "prompt",
      tone: "neutral",
      text: toText(
        t(
          "playground:composer.compareInteropPrompt",
          "Prompt steering is shared across every selected compare model."
        )
      )
    })
  }

  if (jsonMode) {
    notices.push({
      id: "json",
      tone: "neutral",
      text: toText(
        t(
          "playground:composer.compareInteropJson",
          "JSON mode constrains every compare response."
        )
      )
    })
  }

  return notices
}

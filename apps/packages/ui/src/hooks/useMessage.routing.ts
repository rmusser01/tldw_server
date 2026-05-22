import type { EffectiveAssistantState } from "@/hooks/chat/effective-assistant-state"

export type UseMessageSendMode =
  | "tracked_character"
  | "tracked_persona"
  | "overlay"
  | "plain"

export const resolveUseMessageSendMode = ({
  effectiveMode,
  hasEffectiveAssistant,
  draftAssistantKind
}: {
  effectiveMode: EffectiveAssistantState["mode"]
  hasEffectiveAssistant: boolean
  draftAssistantKind?: "character" | "persona" | null
}): UseMessageSendMode => {
  if (effectiveMode === "tracked_character") {
    return "tracked_character"
  }

  if (effectiveMode === "tracked_persona") {
    return "tracked_persona"
  }

  if (effectiveMode === "overlay") {
    return "overlay"
  }

  if (!hasEffectiveAssistant) {
    return "plain"
  }

  if (draftAssistantKind === "character") {
    return "tracked_character"
  }

  if (draftAssistantKind === "persona") {
    return "tracked_persona"
  }

  return "plain"
}

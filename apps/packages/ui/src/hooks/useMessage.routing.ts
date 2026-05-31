import type { EffectiveAssistantState } from "@/hooks/chat/effective-assistant-state"

export type UseMessageSendMode =
  | "tracked_character"
  | "tracked_persona"
  | "overlay"
  | "plain"

export const resolveUseMessageSendMode = ({
  effectiveMode,
  hasEffectiveAssistant,
  draftAssistantKind,
  draftAssistantSelectionMode
}: {
  effectiveMode: EffectiveAssistantState["mode"]
  hasEffectiveAssistant: boolean
  draftAssistantKind?: "character" | "persona" | null
  draftAssistantSelectionMode?: "tracked" | "overlay" | null
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

  if (draftAssistantSelectionMode === "overlay" && hasEffectiveAssistant) {
    return "overlay"
  }

  if (!hasEffectiveAssistant) {
    return "plain"
  }

  if (
    draftAssistantSelectionMode === "tracked" &&
    draftAssistantKind === "character"
  ) {
    return "tracked_character"
  }

  if (
    draftAssistantSelectionMode === "tracked" &&
    draftAssistantKind === "persona"
  ) {
    return "tracked_persona"
  }

  return "plain"
}

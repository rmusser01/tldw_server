import {
  type AssistantSelection,
  type AssistantKind,
  getAssistantSelectionMode,
  normalizeAssistantSelection
} from "@/types/assistant-selection"
import type { ChatSettingsRecord } from "@/types/chat-session-settings"

type TrackedAssistantMetadata = {
  assistantKind?: string | null
  assistantId?: string | number | null
  characterId?: string | number | null
  displayName?: string | null
  avatarUrl?: string | null
  systemPromptSnapshot?: string | null
}

type ResolveEffectiveAssistantStateInput = {
  tracked?: TrackedAssistantMetadata | null
  settings?: Pick<ChatSettingsRecord, "assistantOverlay"> | null
  draftSelection?: AssistantSelection | null
}

export type EffectiveAssistantState = {
  mode: "tracked_character" | "tracked_persona" | "overlay" | "plain"
  kind: AssistantKind | null
  id: string | null
  displayName: string | null
  avatarUrl: string | null
  systemPromptSnapshot: string | null
  source: "tracked" | "overlay" | "none"
}

export const effectiveAssistantStateToSelection = (
  state: EffectiveAssistantState
): AssistantSelection | null => {
  if (!state.kind || !state.id || state.source === "none") {
    return null
  }

  return normalizeAssistantSelection({
    kind: state.kind,
    id: state.id,
    name:
      state.displayName ??
      (state.kind === "persona" ? "Persona" : "Assistant"),
    avatar_url: state.avatarUrl,
    system_prompt: state.systemPromptSnapshot,
    metadata:
      state.source === "overlay"
        ? { selectionMode: "overlay" }
        : { selectionMode: "tracked" }
  })
}

const normalizeId = (value: unknown): string | null => {
  if (typeof value === "string") {
    const trimmed = value.trim()
    return trimmed.length > 0 ? trimmed : null
  }
  if (typeof value === "number" && Number.isFinite(value)) {
    return String(value)
  }
  return null
}

const normalizeText = (value: unknown): string | null => {
  if (typeof value !== "string") return null
  const trimmed = value.trim()
  return trimmed.length > 0 ? trimmed : null
}

const getMatchingDraftSelection = (
  draftSelection: AssistantSelection | null | undefined,
  kind: AssistantKind,
  id: string
) =>
  draftSelection?.kind === kind && draftSelection.id === id
    ? draftSelection
    : null

export const resolveEffectiveAssistantState = ({
  tracked,
  settings,
  draftSelection
}: ResolveEffectiveAssistantStateInput): EffectiveAssistantState => {
  const trackedAssistantKind = tracked?.assistantKind
  const trackedCharacterId = normalizeId(tracked?.characterId)
  if (trackedAssistantKind === "character" && trackedCharacterId) {
    const matchedDraft = getMatchingDraftSelection(
      draftSelection,
      "character",
      trackedCharacterId
    )
    return {
      mode: "tracked_character",
      kind: "character",
      id: trackedCharacterId,
      displayName:
        normalizeText(tracked?.displayName) ??
        normalizeText(matchedDraft?.name) ??
        "Assistant",
      avatarUrl:
        normalizeText(tracked?.avatarUrl) ??
        normalizeText(matchedDraft?.avatar_url) ??
        null,
      systemPromptSnapshot:
        normalizeText(tracked?.systemPromptSnapshot) ??
        normalizeText(matchedDraft?.system_prompt) ??
        null,
      source: "tracked"
    }
  }

  const trackedAssistantId = normalizeId(tracked?.assistantId)
  if (trackedAssistantKind === "persona" && trackedAssistantId) {
    const matchedDraft = getMatchingDraftSelection(
      draftSelection,
      "persona",
      trackedAssistantId
    )
    return {
      mode: "tracked_persona",
      kind: "persona",
      id: trackedAssistantId,
      displayName:
        normalizeText(tracked?.displayName) ??
        normalizeText(matchedDraft?.name) ??
        "Persona",
      avatarUrl:
        normalizeText(tracked?.avatarUrl) ??
        normalizeText(matchedDraft?.avatar_url) ??
        null,
      systemPromptSnapshot:
        normalizeText(tracked?.systemPromptSnapshot) ??
        normalizeText(matchedDraft?.system_prompt) ??
        null,
      source: "tracked"
    }
  }

  const overlay = settings?.assistantOverlay
  if (overlay) {
    const matchedDraft = getMatchingDraftSelection(
      draftSelection,
      overlay.kind,
      overlay.id
    )
    return {
      mode: "overlay",
      kind: overlay.kind,
      id: overlay.id,
      displayName:
        normalizeText(overlay.name) ??
        normalizeText(matchedDraft?.name) ??
        null,
      avatarUrl:
        normalizeText(overlay.avatar_url) ??
        normalizeText(matchedDraft?.avatar_url) ??
        null,
      systemPromptSnapshot:
        normalizeText(overlay.system_prompt_snapshot) ??
        normalizeText(matchedDraft?.system_prompt) ??
        null,
      source: "overlay"
    }
  }

  if (draftSelection && getAssistantSelectionMode(draftSelection) === "overlay") {
    return {
      mode: "overlay",
      kind: draftSelection.kind,
      id: draftSelection.id,
      displayName: normalizeText(draftSelection.name) ?? null,
      avatarUrl: normalizeText(draftSelection.avatar_url) ?? null,
      systemPromptSnapshot: normalizeText(draftSelection.system_prompt) ?? null,
      source: "overlay"
    }
  }

  return {
    mode: "plain",
    kind: null,
    id: null,
    displayName: null,
    avatarUrl: null,
    systemPromptSnapshot: null,
    source: "none"
  }
}

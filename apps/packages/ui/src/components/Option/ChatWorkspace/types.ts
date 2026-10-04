import type {
  WorkspaceAssistantDefaultDegradedReason,
  WorkspaceSourceType
} from "@/types/workspace"
import { getDesignSystemState } from "@/design-system"

export type StagedSourceAvailability =
  | "ready"
  | "processing"
  | "error"
  | "unavailable"

export type StagedWorkspaceSource = {
  sourceId: string
  mediaId: number | null
  title: string
  type: WorkspaceSourceType
  scopeLabel: string
  availability: StagedSourceAvailability
  statusMessage?: string
}

export type ChatWorkspaceAssistantSource =
  | "explicit"
  | "workspace"
  | "none"
  | "unavailable"

export type ChatWorkspaceRuntimeStatus = {
  backendAvailable: boolean
  workspaceReady: boolean
  hasModelSelected: boolean
  connectionMode?: "live" | "demo" | "bypass"
  streaming: boolean
  sending?: boolean
  historyLoading?: boolean
  historyLoadError?: string | null
  sendError?: string | null
}

export const getChatWorkspaceRuntimeLabel = ({
  backendAvailable,
  workspaceReady,
  connectionMode,
  streaming,
  sending,
  historyLoading,
  historyLoadError,
  sendError,
  hasModelSelected
}: ChatWorkspaceRuntimeStatus): string => {
  if (connectionMode === "demo") return "Demo mode - not live"
  if (connectionMode === "bypass") return "Offline bypass - not verified"
  if (!backendAvailable) return "Server unavailable"
  if (!workspaceReady) return "Loading workspace context"
  if (historyLoading) return "Loading chat history"
  if (historyLoadError) return "Chat history unavailable"
  if (streaming) return "Streaming"
  if (sending) return "Sending"
  if (sendError) return "Send failed"
  if (!hasModelSelected) return "Select a model"
  return getDesignSystemState("ready").label
}

export type ChatWorkspaceRuntimeState = Omit<
  ChatWorkspaceRuntimeStatus,
  "workspaceReady"
> & {
  selectedModelLabel: string
  selectedPersonaLabel: string | null
  assistantSource: ChatWorkspaceAssistantSource
  workspaceAssistantDegradedReason?: WorkspaceAssistantDefaultDegradedReason | null
}

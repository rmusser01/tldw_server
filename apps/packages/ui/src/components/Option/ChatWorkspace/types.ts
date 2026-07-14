import type {
  WorkspaceAssistantDefaultDegradedReason,
  WorkspaceSourceType
} from "@/types/workspace"

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

export type ChatWorkspaceRuntimeState = {
  backendAvailable: boolean
  streaming: boolean
  sendError?: string | null
  selectedModelLabel: string
  hasModelSelected: boolean
  selectedPersonaLabel: string | null
  assistantSource: ChatWorkspaceAssistantSource
  workspaceAssistantDegradedReason?: WorkspaceAssistantDefaultDegradedReason | null
}

import type { WorkspaceSourceType } from "@/types/workspace"

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

export type ChatWorkspaceRuntimeState = {
  backendAvailable: boolean
  streaming: boolean
  selectedModelLabel: string
  selectedPersonaLabel: string | null
}

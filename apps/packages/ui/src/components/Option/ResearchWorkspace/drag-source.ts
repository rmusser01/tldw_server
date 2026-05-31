import type { WorkspaceSourceType } from "@/types/workspace"

export const WORKSPACE_SOURCE_DRAG_TYPE = "application/x-tldw-workspace-source"

export interface WorkspaceSourceDragPayload {
  sourceId: string
  mediaId: number
  title: string
  type: WorkspaceSourceType
}

export const serializeWorkspaceSourceDragPayload = (
  payload: WorkspaceSourceDragPayload
): string => JSON.stringify(payload)

export const parseWorkspaceSourceDragPayload = (
  raw: string | null | undefined
): WorkspaceSourceDragPayload | null => {
  if (!raw) return null
  try {
    const parsed = JSON.parse(raw) as Partial<WorkspaceSourceDragPayload>
    if (
      !parsed ||
      typeof parsed.sourceId !== "string" ||
      typeof parsed.mediaId !== "number" ||
      typeof parsed.title !== "string" ||
      typeof parsed.type !== "string"
    ) {
      return null
    }

    return {
      sourceId: parsed.sourceId,
      mediaId: parsed.mediaId,
      title: parsed.title,
      type: parsed.type
    }
  } catch {
    return null
  }
}

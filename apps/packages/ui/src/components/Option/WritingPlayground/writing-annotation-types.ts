import type {
  ManuscriptAnnotationCategory,
  ManuscriptAnnotationListFilters,
  ManuscriptAnnotationTargetType
} from "@/services/writing-playground"
import type { WritingEditorSelection } from "./writing-editor-adapter"

export const WRITING_ANNOTATION_CATEGORIES: ManuscriptAnnotationCategory[] = [
  "style",
  "clarity",
  "pacing",
  "continuity",
  "character",
  "worldbuilding",
  "structure",
  "research",
  "other"
]

export type WritingAnnotationTargetContext = {
  targetType: ManuscriptAnnotationTargetType
  targetId: string
}

export const resolveWritingAnnotationTargetContext = ({
  projectId,
  activeNodeType,
  activeNodeId,
  activeSceneId
}: {
  projectId?: string | null
  activeNodeType?: "part" | ManuscriptAnnotationTargetType | null
  activeNodeId?: string | null
  activeSceneId?: string | null
}): WritingAnnotationTargetContext | null => {
  if (!projectId) return null
  if (activeNodeType === "scene") {
    const sceneId = activeSceneId || activeNodeId
    return sceneId ? { targetType: "scene", targetId: sceneId } : null
  }
  if (activeNodeType === "chapter") {
    return activeNodeId
      ? { targetType: "chapter", targetId: activeNodeId }
      : null
  }
  return { targetType: "project", targetId: projectId }
}

export type WritingAnnotationFilters = ManuscriptAnnotationListFilters

export type WritingAnnotationSelection = WritingEditorSelection | null

export type WritingAnnotationProviderTarget = {
  provider?: string | null
  model?: string | null
}

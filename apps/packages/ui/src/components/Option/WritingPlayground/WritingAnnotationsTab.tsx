import React from "react"
import { Button, Input, Select, Segmented, Typography } from "antd"
import type {
  ManuscriptAnnotationCategory,
  ManuscriptAnnotationTargetType
} from "@/services/writing-playground"
import type { WritingEditorSelection } from "./writing-editor-adapter"
import {
  WRITING_ANNOTATION_CATEGORIES,
  resolveWritingAnnotationTargetContext,
  type WritingAnnotationProviderTarget
} from "./writing-annotation-types"
import {
  buildSceneRangeAnnotationInput,
  validateSelectedRange
} from "./writing-annotation-anchor-utils"
import {
  WritingAnnotationList
} from "./WritingAnnotationList"
import type {
  ManuscriptAnnotationReviewJobResponse,
  UseWritingAnnotationsResult
} from "./hooks/useWritingAnnotations"

const { TextArea } = Input
const { Text } = Typography

type WritingAnnotationsTabProps = {
  annotationsHook: UseWritingAnnotationsResult
  projectId: string | null
  activeChapterId?: string | null
  activeSceneId?: string | null
  activeSceneVersion?: number | null
  activeSceneText: string
  selection: WritingEditorSelection | null
  canCreateRangeAnnotation: boolean
  isSceneDirty: boolean
  selectedModel?: string | null
  apiProvider?: string | null
} & WritingAnnotationProviderTarget

const resolveTargetId = ({
  targetType,
  projectId,
  activeChapterId,
  activeSceneId
}: {
  targetType: ManuscriptAnnotationTargetType
  projectId: string | null
  activeChapterId?: string | null
  activeSceneId?: string | null
}) => {
  if (targetType === "project") return projectId
  if (targetType === "chapter") return activeChapterId
  return activeSceneId
}

const resolveDefaultTargetType = ({
  projectId,
  activeChapterId,
  activeSceneId
}: {
  projectId: string | null
  activeChapterId?: string | null
  activeSceneId?: string | null
}): ManuscriptAnnotationTargetType => {
  if (activeSceneId) return "scene"
  if (activeChapterId) return "chapter"
  if (projectId) return "project"
  return "scene"
}

export function WritingAnnotationsTab({
  annotationsHook,
  projectId,
  activeChapterId,
  activeSceneId,
  activeSceneVersion,
  activeSceneText,
  selection,
  canCreateRangeAnnotation,
  isSceneDirty,
  selectedModel,
  apiProvider
}: WritingAnnotationsTabProps) {
  const [targetType, setTargetType] =
    React.useState<ManuscriptAnnotationTargetType>(() =>
      resolveDefaultTargetType({ projectId, activeChapterId, activeSceneId })
    )
  const [category, setCategory] =
    React.useState<ManuscriptAnnotationCategory>("other")
  const [body, setBody] = React.useState("")
  const [sceneReviewJob, setSceneReviewJob] =
    React.useState<ManuscriptAnnotationReviewJobResponse | null>(null)
  const sceneReviewContextRef = React.useRef<{
    sceneId: string | null
    sceneVersion: number | null
  }>({
    sceneId: activeSceneId ?? null,
    sceneVersion: activeSceneVersion ?? null
  })
  sceneReviewContextRef.current = {
    sceneId: activeSceneId ?? null,
    sceneVersion: activeSceneVersion ?? null
  }
  const defaultTargetType = resolveDefaultTargetType({
    projectId,
    activeChapterId,
    activeSceneId
  })
  const trimmedBody = body.trim()
  const provider = apiProvider?.trim()
  const model = selectedModel?.trim()
  const selectedRange = React.useMemo(
    () => validateSelectedRange({ documentText: activeSceneText, selection }),
    [activeSceneText, selection]
  )
  const hasValidSelection = selectedRange.ok
  const aiReviewDisabled =
    !provider ||
    !model ||
    !activeSceneId ||
    activeSceneVersion == null ||
    isSceneDirty
  const rangeDisabled =
    !trimmedBody ||
    !canCreateRangeAnnotation ||
    !activeSceneId ||
    activeSceneVersion == null ||
    !hasValidSelection
  const selectionReviewDisabled =
    aiReviewDisabled ||
    !canCreateRangeAnnotation ||
    !hasValidSelection
  const noteTargetId = resolveTargetId({
    targetType,
    projectId,
    activeChapterId,
    activeSceneId
  })
  const noteDisabled = !trimmedBody || !noteTargetId

  React.useEffect(() => {
    setSceneReviewJob(null)
  }, [activeSceneId, activeSceneVersion])

  React.useEffect(() => {
    const context = resolveWritingAnnotationTargetContext({
      projectId,
      activeNodeType: targetType,
      activeNodeId:
        targetType === "chapter"
          ? activeChapterId ?? null
          : targetType === "scene"
            ? activeSceneId ?? null
            : projectId,
      activeSceneId
    })
    if (!context) {
      setTargetType(defaultTargetType)
    }
  }, [activeChapterId, activeSceneId, defaultTargetType, projectId, targetType])

  const addNote = async () => {
    if (!noteTargetId || !trimmedBody) return
    await annotationsHook.createAnnotation({
      target_type: targetType,
      target_id: noteTargetId,
      category,
      body: trimmedBody
    })
    setBody("")
  }

  const addRangeComment = async () => {
    if (!trimmedBody) return
    const input = buildSceneRangeAnnotationInput({
      canCreateRangeAnnotation,
      sceneId: activeSceneId ?? null,
      sceneVersion: activeSceneVersion ?? null,
      documentText: activeSceneText,
      selection,
      category,
      body: trimmedBody
    })
    await annotationsHook.createAnnotation(input)
    setBody("")
  }

  const reviewSelection = async () => {
    if (selectionReviewDisabled || !activeSceneId || !selection) return
    const input = buildSceneRangeAnnotationInput({
      canCreateRangeAnnotation,
      sceneId: activeSceneId,
      sceneVersion: activeSceneVersion ?? null,
      documentText: activeSceneText,
      selection,
      category,
      body: "AI selection review"
    })
    await annotationsHook.reviewSelection({
      sceneId: activeSceneId,
      provider: provider!,
      model: model!,
      scene_version: activeSceneVersion!,
      start: input.start ?? 0,
      end: input.end ?? 0,
      selected_text: input.selected_text ?? "",
      category_hints: [category]
    })
  }

  const reviewScene = async () => {
    if (aiReviewDisabled || !activeSceneId) return
    const requestedSceneId = activeSceneId
    const requestedSceneVersion = activeSceneVersion!
    const job = await annotationsHook.reviewScene({
      sceneId: requestedSceneId,
      provider: provider!,
      model: model!,
      scene_version: requestedSceneVersion,
      max_comments: 8,
      category_filters: [category]
    })
    const currentContext = sceneReviewContextRef.current
    if (
      currentContext.sceneId !== requestedSceneId ||
      currentContext.sceneVersion !== requestedSceneVersion
    ) {
      return
    }
    setSceneReviewJob(job)
  }

  return (
    <div className="flex flex-col gap-3" data-testid="writing-annotations-tab">
      <div className="flex flex-col gap-2">
        <Segmented
          size="small"
          value={targetType}
          options={[
            { label: "Scene", value: "scene" },
            { label: "Chapter", value: "chapter" },
            { label: "Project", value: "project" }
          ]}
          onChange={(value) =>
            setTargetType(value as ManuscriptAnnotationTargetType)
          }
        />
        <Select
          size="small"
          value={category}
          options={WRITING_ANNOTATION_CATEGORIES.map((value) => ({
            value,
            label: value
          }))}
          onChange={(value) => setCategory(value)}
        />
        <TextArea
          aria-label="Annotation body"
          size="small"
          value={body}
          placeholder="Add an annotation..."
          autoSize={{ minRows: 3, maxRows: 5 }}
          onChange={(event) => setBody(event.target.value)}
        />
        {!canCreateRangeAnnotation ? (
          <Text type="secondary" className="text-xs">
            Save the selected scene before adding range comments.
          </Text>
        ) : null}
        <div className="flex flex-wrap gap-1">
          <Button
            size="small"
            type="primary"
            disabled={rangeDisabled}
            loading={annotationsHook.isCreating}
            onClick={() => {
              void addRangeComment()
            }}>
            Add range comment
          </Button>
          <Button
            size="small"
            disabled={noteDisabled}
            loading={annotationsHook.isCreating}
            onClick={() => {
              void addNote()
            }}>
            Add note
          </Button>
        </div>
        <div className="flex flex-wrap gap-1">
          <Button
            size="small"
            disabled={selectionReviewDisabled}
            loading={annotationsHook.isReviewingSelection}
            onClick={() => {
              void reviewSelection()
            }}>
            Review selection with AI
          </Button>
          <Button
            size="small"
            disabled={aiReviewDisabled}
            loading={annotationsHook.isReviewingScene}
            onClick={() => {
              void reviewScene()
            }}>
            Review scene with AI
          </Button>
        </div>
        {sceneReviewJob ? (
          <Text type="secondary" className="text-xs">
            Scene review job {sceneReviewJob.job_id} {sceneReviewJob.status}
          </Text>
        ) : null}
      </div>
      <WritingAnnotationList
        annotations={annotationsHook.annotations}
        onUpdate={annotationsHook.updateAnnotation}
        onDelete={annotationsHook.deleteAnnotation}
        disabled={
          annotationsHook.isUpdating ||
          annotationsHook.isDeleting ||
          annotationsHook.isFetching
        }
      />
    </div>
  )
}

import React from "react"
import { useQuery, useQueryClient } from "@tanstack/react-query"
import type { JSONContent } from "@tiptap/react"

import {
  getManuscriptScene,
  updateManuscriptScene,
  type ManuscriptSceneResponse
} from "@/services/writing-playground"
import { resolveTipTapDocument } from "../writing-tiptap-utils"

type ManuscriptNodeType = "part" | "chapter" | "scene" | null
type ManuscriptEditorMode = "plain" | "tiptap"

export type ActiveManuscriptSceneBinding = {
  scene: ManuscriptSceneResponse | null
  sceneId: string | null
  sceneVersion: number | null
  isSceneBound: boolean
  isSceneLoading: boolean
  isSceneDirty: boolean
  canCreateRangeAnnotation: boolean
  saveScene: () => Promise<ManuscriptSceneResponse | null>
  reloadScene: () => void
}

type UseActiveManuscriptSceneDeps = {
  activeNodeId: string | null
  activeNodeType: ManuscriptNodeType
  editorText: string
  setEditorText: (nextText: string) => void
  tipTapContent: JSONContent | null
  setTipTapContent: (nextContent: JSONContent | null) => void
  editorMode?: ManuscriptEditorMode
  isOnline?: boolean
}

type SceneBindingState = {
  scene: ManuscriptSceneResponse
  savedPlainText: string
  savedContent: JSONContent
  savedContentSignature: string
}

const serializeTipTapContent = (content: JSONContent | null): string =>
  JSON.stringify(content ?? null)

const resolveSceneTipTapContent = (
  scene: ManuscriptSceneResponse
): JSONContent =>
  resolveTipTapDocument(
    scene.content_plain ?? "",
    scene.content as JSONContent | null | undefined
  )

const createSceneBinding = (
  scene: ManuscriptSceneResponse
): SceneBindingState => {
  const savedContent = resolveSceneTipTapContent(scene)
  return {
    scene,
    savedPlainText: scene.content_plain ?? "",
    savedContent,
    savedContentSignature: serializeTipTapContent(savedContent)
  }
}

const getCurrentContentSignature = (
  binding: SceneBindingState,
  tipTapContent: JSONContent | null
): string =>
  tipTapContent
    ? serializeTipTapContent(tipTapContent)
    : binding.savedContentSignature

const resolveSaveContent = (
  binding: SceneBindingState,
  editorText: string,
  tipTapContent: JSONContent | null,
  trackRichContent: boolean
): JSONContent => {
  if (
    trackRichContent &&
    tipTapContent &&
    serializeTipTapContent(tipTapContent) !== binding.savedContentSignature
  ) {
    return tipTapContent
  }
  return resolveTipTapDocument(editorText, null)
}

const isBindingDirty = (
  binding: SceneBindingState | null,
  editorText: string,
  tipTapContent: JSONContent | null,
  trackRichContent: boolean
): boolean => {
  if (!binding) return false
  if (!trackRichContent) return editorText !== binding.savedPlainText
  return (
    editorText !== binding.savedPlainText ||
    getCurrentContentSignature(binding, tipTapContent) !==
      binding.savedContentSignature
  )
}

export function useActiveManuscriptScene({
  activeNodeId,
  activeNodeType,
  editorText,
  setEditorText,
  tipTapContent,
  setTipTapContent,
  editorMode = "tiptap",
  isOnline = true
}: UseActiveManuscriptSceneDeps): ActiveManuscriptSceneBinding {
  const queryClient = useQueryClient()
  const [binding, setBinding] = React.useState<SceneBindingState | null>(null)
  const shouldQueryScene =
    isOnline && activeNodeType === "scene" && Boolean(activeNodeId)

  const sceneQuery = useQuery({
    queryKey: ["manuscript-scene", activeNodeId],
    queryFn: () => getManuscriptScene(activeNodeId ?? ""),
    enabled: shouldQueryScene,
    staleTime: 30_000
  })

  const bindingMatchesActiveScene =
    activeNodeType === "scene" &&
    Boolean(activeNodeId) &&
    binding?.scene.id === activeNodeId
  const trackRichContent = editorMode === "tiptap"
  const isSceneDirty = bindingMatchesActiveScene
    ? isBindingDirty(binding, editorText, tipTapContent, trackRichContent)
    : false

  React.useEffect(() => {
    if (activeNodeType === "scene" || !binding) return
    if (isBindingDirty(binding, editorText, tipTapContent, trackRichContent)) {
      return
    }
    setBinding(null)
  }, [activeNodeType, binding, editorText, tipTapContent, trackRichContent])

  React.useEffect(() => {
    const scene = sceneQuery.data ?? null
    if (!scene || activeNodeType !== "scene" || activeNodeId !== scene.id) {
      return
    }

    const nextBinding = createSceneBinding(scene)
    const isSameSavedScene =
      binding?.scene.id === scene.id &&
      binding.scene.version === scene.version &&
      binding.savedPlainText === nextBinding.savedPlainText &&
      binding.savedContentSignature === nextBinding.savedContentSignature
    if (isSameSavedScene) return

    if (binding?.scene.id === scene.id && isSceneDirty) return

    setEditorText(nextBinding.savedPlainText)
    setTipTapContent(nextBinding.savedContent)
    setBinding(nextBinding)
  }, [
    activeNodeId,
    activeNodeType,
    binding,
    isSceneDirty,
    sceneQuery.data,
    setEditorText,
    setTipTapContent
  ])

  const saveScene = React.useCallback(async () => {
    if (!binding) return null
    if (
      activeNodeType !== "scene" ||
      !activeNodeId ||
      binding.scene.id !== activeNodeId
    ) {
      return null
    }

    const content = resolveSaveContent(
      binding,
      editorText,
      tipTapContent,
      trackRichContent
    )
    const savedScene = (await updateManuscriptScene(
      binding.scene.id,
      {
        content_plain: editorText,
        content
      },
      binding.scene.version
    )) as ManuscriptSceneResponse
    const nextBinding = createSceneBinding(savedScene)

    queryClient.setQueryData(
      ["manuscript-scene", savedScene.id],
      savedScene
    )
    queryClient.invalidateQueries({
      queryKey: ["manuscript-structure", savedScene.project_id]
    })
    setEditorText(nextBinding.savedPlainText)
    setTipTapContent(nextBinding.savedContent)
    setBinding(nextBinding)
    return savedScene
  }, [
    activeNodeId,
    activeNodeType,
    binding,
    editorText,
    queryClient,
    setEditorText,
    setTipTapContent,
    tipTapContent,
    trackRichContent
  ])

  const reloadScene = React.useCallback(() => {
    const targetSceneId =
      bindingMatchesActiveScene && binding
        ? binding.scene.id
        : activeNodeType === "scene"
          ? activeNodeId
          : null
    if (bindingMatchesActiveScene && binding) {
      setEditorText(binding.savedPlainText)
      setTipTapContent(binding.savedContent)
    }
    if (targetSceneId) {
      queryClient.invalidateQueries({
        queryKey: ["manuscript-scene", targetSceneId]
      })
    }
  }, [
    activeNodeId,
    activeNodeType,
    binding,
    bindingMatchesActiveScene,
    queryClient,
    setEditorText,
    setTipTapContent
  ])
  const exposedBinding = bindingMatchesActiveScene ? binding : null

  return {
    scene: exposedBinding?.scene ?? null,
    sceneId: exposedBinding?.scene.id ?? null,
    sceneVersion: exposedBinding?.scene.version ?? null,
    isSceneBound: Boolean(exposedBinding),
    isSceneLoading: sceneQuery.isLoading || sceneQuery.isFetching,
    isSceneDirty,
    canCreateRangeAnnotation: Boolean(exposedBinding) && !isSceneDirty,
    saveScene,
    reloadScene
  }
}

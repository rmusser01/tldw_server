import { useCallback } from "react"
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import {
  getAudiobookProjects,
  getAudiobookProjectById,
  upsertAudiobookProject,
  deleteAudiobookProject,
  duplicateAudiobookProject,
  markProjectOpened,
  storeChapterAudio,
  getChapterAudioBlob,
  serializeChapters,
  createEmptyProject
} from "@/db/dexie/audiobook-projects"
import type { AudiobookProject } from "@/db/dexie/types"
import { useAudiobookStudioStore, type AudioChapter } from "@/store/audiobook-studio"

const PROJECTS_QUERY_KEY = ["audiobook-projects"]
const PROJECT_QUERY_KEY = (id: string) => ["audiobook-project", id]

/**
 * Hook for managing audiobook projects list
 */
export function useAudiobookProjects() {
  const queryClient = useQueryClient()

  // Query all projects
  const {
    data: projects = [],
    isLoading,
    error,
    refetch
  } = useQuery<AudiobookProject[]>({
    queryKey: PROJECTS_QUERY_KEY,
    queryFn: getAudiobookProjects
  })

  // Delete project mutation
  const deleteMutation = useMutation({
    mutationFn: deleteAudiobookProject,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: PROJECTS_QUERY_KEY })
    }
  })

  // Duplicate project mutation
  const duplicateMutation = useMutation({
    mutationFn: ({
      id,
      newTitle
    }: {
      id: string
      newTitle?: string
    }) => duplicateAudiobookProject(id, newTitle),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: PROJECTS_QUERY_KEY })
    }
  })

  return {
    projects,
    isLoading,
    error,
    refetch,
    deleteProject: deleteMutation.mutateAsync,
    isDeleting: deleteMutation.isPending,
    duplicateProject: duplicateMutation.mutateAsync,
    isDuplicating: duplicateMutation.isPending
  }
}

/**
 * Hook for managing the current audiobook project
 */
export function useCurrentProject() {
  const queryClient = useQueryClient()

  // Store selectors
  const projectId = useAudiobookStudioStore((s) => s.currentProjectId)
  const rawContent = useAudiobookStudioStore((s) => s.rawContent)
  const chapters = useAudiobookStudioStore((s) => s.chapters)
  const projectTitle = useAudiobookStudioStore((s) => s.projectTitle)
  const projectAuthor = useAudiobookStudioStore((s) => s.projectAuthor)
  const projectDescription = useAudiobookStudioStore((s) => s.projectDescription)
  const projectCoverImageUrl = useAudiobookStudioStore(
    (s) => s.projectCoverImageUrl
  )
  const defaultVoiceConfig = useAudiobookStudioStore((s) => s.defaultVoiceConfig)
  const getTotalDuration = useAudiobookStudioStore((s) => s.getTotalDuration)

  // Store actions
  const setCurrentProjectId = useAudiobookStudioStore((s) => s.setCurrentProjectId)
  const setRawContent = useAudiobookStudioStore((s) => s.setRawContent)
  const setChapters = useAudiobookStudioStore((s) => s.setChapters)
  const setProjectTitle = useAudiobookStudioStore((s) => s.setProjectTitle)
  const setProjectAuthor = useAudiobookStudioStore((s) => s.setProjectAuthor)
  const setProjectDescription = useAudiobookStudioStore(
    (s) => s.setProjectDescription
  )
  const setProjectCoverImageUrl = useAudiobookStudioStore(
    (s) => s.setProjectCoverImageUrl
  )
  const setDefaultVoiceConfig = useAudiobookStudioStore((s) => s.setDefaultVoiceConfig)
  const clearChapters = useAudiobookStudioStore((s) => s.clearChapters)
  const revokeAllAudioUrls = useAudiobookStudioStore((s) => s.revokeAllAudioUrls)

  // Load project from database into store
  const loadProject = useCallback(
    async (id: string) => {
      const project = await getAudiobookProjectById(id)
      if (!project) return false

      // Clear existing state
      revokeAllAudioUrls()

      // Set project metadata
      setCurrentProjectId(project.id)
      setProjectTitle(project.title)
      setProjectAuthor(project.author)
      setProjectDescription(project.description || "")
      setProjectCoverImageUrl(project.coverImageUrl || null)
      setRawContent(project.rawContent)
      setDefaultVoiceConfig(project.defaultVoiceConfig)

      // Load chapters with their audio blobs
      const chaptersWithAudio: AudioChapter[] = await Promise.all(
        project.chapters.map(async (ch) => {
          const assetId = project.chapterAudioAssetIds[ch.id]
          let audioBlob: Blob | undefined
          let audioUrl: string | undefined

          if (assetId && ch.status === "completed") {
            audioBlob = (await getChapterAudioBlob(assetId)) || undefined
            if (audioBlob) {
              audioUrl = URL.createObjectURL(audioBlob)
            }
          }

          return {
            id: ch.id,
            title: ch.title,
            content: ch.content,
            order: ch.order,
            voiceConfig: ch.voiceConfig,
            status: ch.status,
            audioBlob,
            audioUrl,
            audioDuration: ch.audioDuration,
            errorMessage: ch.errorMessage,
            requestedBackend: ch.requestedBackend,
            actualBackend: ch.actualBackend,
            fallbackUsed: ch.fallbackUsed
          }
        })
      )

      setChapters(chaptersWithAudio)

      // Mark as opened
      await markProjectOpened(id)

      return true
    },
    [
      setCurrentProjectId,
      setProjectTitle,
      setProjectAuthor,
      setRawContent,
      setDefaultVoiceConfig,
      setChapters,
      revokeAllAudioUrls
    ]
  )

  // Save current store state to database
  const saveProject = useCallback(
    async (id?: string) => {
      const targetId = id || projectId || crypto.randomUUID()
      const now = Date.now()
      const existingProject = projectId
        ? await getAudiobookProjectById(targetId)
        : undefined

      // Store chapter audio blobs
      const existingAssetIds = existingProject?.chapterAudioAssetIds || {}
      const chapterAudioAssetIds: Record<string, string> = {}

      for (const chapter of chapters) {
        if (chapter.status !== "completed") continue

        if (chapter.audioBlob) {
          const result = await storeChapterAudio(
            targetId,
            chapter.id,
            chapter.audioBlob
          )
          if (result.assetId) {
            chapterAudioAssetIds[chapter.id] = result.assetId
          } else if (existingAssetIds[chapter.id]) {
            chapterAudioAssetIds[chapter.id] = existingAssetIds[chapter.id]
            console.warn(
              "Audiobook assets storage cap reached; keeping existing asset."
            )
          }
          continue
        }

        if (existingAssetIds[chapter.id]) {
          chapterAudioAssetIds[chapter.id] = existingAssetIds[chapter.id]
        }
      }

      // Determine project status
      const completedCount = chapters.filter((ch) => ch.status === "completed").length
      let status: AudiobookProject["status"] = "draft"
      if (completedCount > 0 && completedCount === chapters.length) {
        status = "completed"
      } else if (completedCount > 0) {
        status = "in_progress"
      }

      const project: AudiobookProject = {
        id: targetId,
        title: projectTitle,
        author: projectAuthor,
        description: projectDescription || undefined,
        coverImageUrl: projectCoverImageUrl || undefined,
        rawContent,
        chapters: serializeChapters(chapters),
        chapterAudioAssetIds,
        defaultVoiceConfig,
        status,
        totalDuration: getTotalDuration(),
        createdAt: existingProject?.createdAt || now,
        updatedAt: now
      }

      await upsertAudiobookProject(project)
      setCurrentProjectId(targetId)

      // Invalidate queries
      queryClient.invalidateQueries({ queryKey: PROJECTS_QUERY_KEY })
      queryClient.invalidateQueries({ queryKey: PROJECT_QUERY_KEY(targetId) })

      return targetId
    },
    [
      projectId,
      rawContent,
      chapters,
      projectTitle,
      projectAuthor,
      projectDescription,
      projectCoverImageUrl,
      defaultVoiceConfig,
      getTotalDuration,
      setCurrentProjectId,
      queryClient
    ]
  )

  // Create a new empty project
  const createNewProject = useCallback(
    async (title?: string) => {
      // Clear current state
      revokeAllAudioUrls()
      clearChapters()
      setRawContent("")
      setProjectTitle(title || "Untitled Audiobook")
      setProjectAuthor("")
      setProjectDescription("")
      setProjectCoverImageUrl(null)
      setDefaultVoiceConfig({})

      // Create new project in DB
      const newProject = createEmptyProject(title)
      await upsertAudiobookProject(newProject)
      setCurrentProjectId(newProject.id)

      // Invalidate queries
      queryClient.invalidateQueries({ queryKey: PROJECTS_QUERY_KEY })

      return newProject.id
    },
    [
      clearChapters,
      setRawContent,
      setProjectTitle,
      setProjectAuthor,
      setProjectDescription,
      setProjectCoverImageUrl,
      setDefaultVoiceConfig,
      setCurrentProjectId,
      revokeAllAudioUrls,
      queryClient
    ]
  )

  // Check if there are unsaved changes
  const hasUnsavedChanges = useCallback(async () => {
    if (!projectId) return rawContent.length > 0 || chapters.length > 0

    const saved = await getAudiobookProjectById(projectId)
    if (!saved) return true

    // Simple check: compare serialized chapters length and raw content
    if (saved.rawContent !== rawContent) return true
    if (saved.chapters.length !== chapters.length) return true
    if (saved.title !== projectTitle) return true
    if (saved.author !== projectAuthor) return true
    if ((saved.description || "") !== projectDescription) return true
    if ((saved.coverImageUrl || "") !== (projectCoverImageUrl || "")) return true

    return false
  }, [
    projectId,
    rawContent,
    chapters,
    projectTitle,
    projectAuthor,
    projectDescription,
    projectCoverImageUrl
  ])

  return {
    projectId,
    loadProject,
    saveProject,
    createNewProject,
    hasUnsavedChanges
  }
}

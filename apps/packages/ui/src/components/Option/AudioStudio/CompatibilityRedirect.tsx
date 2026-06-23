import React from "react"
import { Alert, Spin } from "antd"
import { useNavigate } from "react-router-dom"
import {
  listLegacyAudiobookProjectsForMigration,
  markLegacyAudiobookProjectMigrated,
  serializeLegacyAudiobookProjectForMigration,
  type LegacyAudiobookProjectMigrationPayload
} from "@/db/dexie/audiobook-projects"
import type { AudiobookProject } from "@/db/dexie/types"
import {
  useCommitAudiobookMigration,
  usePreviewAudiobookMigration
} from "@/hooks/useAudioStudioMigration"
import type {
  AudiobookMigrationCounts,
  AudiobookMigrationResponse
} from "@/services/audio-studio"
import { MigrationBanner } from "./MigrationBanner"

export const AUDIOBOOK_COMPATIBILITY_TARGET =
  "/audio-studio?workflow=narration"

const projectTarget = (projectId: string) =>
  `${AUDIOBOOK_COMPATIBILITY_TARGET}&project=${encodeURIComponent(projectId)}`

const extractErrorMessage = (error: unknown): string => {
  if (error instanceof Error && error.message) return error.message
  return "Audiobook migration could not be completed."
}

const createMigrationIdempotencyKey = () =>
  `audiobook-migration-${Date.now()}-${Math.random()
    .toString(36)
    .slice(2, 10)}`

const fallbackCounts = (
  projects: LegacyAudiobookProjectMigrationPayload[]
): AudiobookMigrationCounts => ({
  projects: projects.length,
  chapters: projects.reduce(
    (total, project) => total + project.chapters.length,
    0
  ),
  audio_assets: projects.reduce(
    (total, project) => total + project.audio_assets.length,
    0
  )
})

const countsFromPreviewResponse = (
  response: AudiobookMigrationResponse,
  fallbackProject: LegacyAudiobookProjectMigrationPayload
): AudiobookMigrationCounts => ({
  projects: response.project_count ?? response.counts?.projects ?? 1,
  chapters:
    response.section_count ??
    response.counts?.chapters ??
    fallbackProject.chapters.length,
  audio_assets:
    response.audio_reference_count ??
    response.counts?.audio_assets ??
    response.counts?.audioAssets ??
    fallbackProject.audio_assets.length
})

const addCounts = (
  left: AudiobookMigrationCounts,
  right: AudiobookMigrationCounts
): AudiobookMigrationCounts => ({
  projects: (left.projects ?? 0) + (right.projects ?? 0),
  chapters: (left.chapters ?? 0) + (right.chapters ?? 0),
  audio_assets:
    (left.audio_assets ?? left.audioAssets ?? 0) +
    (right.audio_assets ?? right.audioAssets ?? 0)
})

const committedProjectFromResponse = (
  response: AudiobookMigrationResponse,
  legacyProjectId: string
) => {
  const project = response.project
  if (
    project &&
    typeof project.project_id === "string" &&
    project.project_id.length > 0
  ) {
    return {
      legacy_project_id: legacyProjectId,
      project_id: project.project_id,
      migration_id: response.fingerprint
    }
  }
  return null
}

export const CompatibilityRedirect: React.FC = () => {
  const navigate = useNavigate()
  const previewMigration = usePreviewAudiobookMigration()
  const commitMigration = useCommitAudiobookMigration()
  const [isLoading, setIsLoading] = React.useState(true)
  const [projects, setProjects] = React.useState<AudiobookProject[]>([])
  const [selectedProjectIds, setSelectedProjectIds] = React.useState<string[]>([])
  const [previewCounts, setPreviewCounts] = React.useState<
    AudiobookMigrationCounts | undefined
  >()
  const [errorMessage, setErrorMessage] = React.useState<string | null>(null)

  React.useEffect(() => {
    let isMounted = true

    const loadLegacyProjects = async () => {
      try {
        const localProjects = await listLegacyAudiobookProjectsForMigration()
        if (!isMounted) return

        if (localProjects.length === 0) {
          navigate(AUDIOBOOK_COMPATIBILITY_TARGET, { replace: true })
          return
        }

        setProjects(localProjects)
        setSelectedProjectIds(localProjects.map((project) => project.id))
        setIsLoading(false)
      } catch (error) {
        if (!isMounted) return
        setErrorMessage(extractErrorMessage(error))
        setIsLoading(false)
      }
    }

    void loadLegacyProjects()

    return () => {
      isMounted = false
    }
  }, [navigate])

  const serializeSelectedProjects = React.useCallback(async () => {
    return await Promise.all(
      selectedProjectIds.map((projectId) =>
        serializeLegacyAudiobookProjectForMigration(projectId)
      )
    )
  }, [selectedProjectIds])

  const handleSelectionChange = React.useCallback((projectIds: string[]) => {
    setSelectedProjectIds(projectIds)
    setPreviewCounts(undefined)
    setErrorMessage(null)
  }, [])

  const handlePreview = React.useCallback(async () => {
    if (selectedProjectIds.length === 0) {
      setErrorMessage("Select at least one Audiobook project to preview.")
      return
    }

    try {
      setErrorMessage(null)
      const serializedProjects = await serializeSelectedProjects()
      const previewCounts = await Promise.all(
        serializedProjects.map(async (project) => {
          const response = await previewMigration.mutateAsync({
            legacy_project_id: project.legacy_project_id,
            project_payload: project
          })
          return countsFromPreviewResponse(response, project)
        })
      )
      setPreviewCounts(
        previewCounts.reduce(addCounts, fallbackCounts([]))
      )
    } catch (error) {
      setErrorMessage(extractErrorMessage(error))
    }
  }, [previewMigration, selectedProjectIds.length, serializeSelectedProjects])

  const handleCommit = React.useCallback(async () => {
    if (selectedProjectIds.length === 0) {
      setErrorMessage("Select at least one Audiobook project to migrate.")
      return
    }

    try {
      setErrorMessage(null)
      const serializedProjects = await serializeSelectedProjects()
      const committedProjects = (
        await Promise.all(
          serializedProjects.map(async (project) => {
            const response = await commitMigration.mutateAsync({
              idempotency_key: createMigrationIdempotencyKey(),
              legacy_project_id: project.legacy_project_id,
              project_payload: project
            })
            return committedProjectFromResponse(response, project.legacy_project_id)
          })
        )
      ).filter((project): project is NonNullable<typeof project> => Boolean(project))

      for (const project of committedProjects) {
        await markLegacyAudiobookProjectMigrated(project.legacy_project_id, {
          migrationId: project.migration_id,
          projectId: project.project_id as string
        })
      }

      const targetProjectId = committedProjects[0]?.project_id
      navigate(
        targetProjectId ? projectTarget(targetProjectId) : AUDIOBOOK_COMPATIBILITY_TARGET,
        { replace: true }
      )
    } catch (error) {
      setErrorMessage(extractErrorMessage(error))
    }
  }, [commitMigration, navigate, selectedProjectIds.length, serializeSelectedProjects])

  if (isLoading) {
    return (
      <div className="flex min-h-[160px] items-center justify-center gap-3">
        <Spin />
        <span>Checking local Audiobook projects...</span>
      </div>
    )
  }

  if (errorMessage && projects.length === 0) {
    return (
      <div className="mx-auto max-w-3xl p-4">
        <Alert type="error" showIcon title={errorMessage} />
      </div>
    )
  }

  if (projects.length === 0) {
    return null
  }

  return (
    <div className="mx-auto max-w-3xl p-4">
      <MigrationBanner
        projects={projects}
        selectedProjectIds={selectedProjectIds}
        previewCounts={previewCounts}
        errorMessage={errorMessage}
        isPreviewing={previewMigration.isPending}
        isCommitting={commitMigration.isPending}
        onSelectionChange={handleSelectionChange}
        onPreview={handlePreview}
        onCommit={handleCommit}
      />
    </div>
  )
}

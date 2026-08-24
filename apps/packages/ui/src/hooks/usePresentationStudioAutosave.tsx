import React from "react"

import { tldwClient } from "@/services/tldw/TldwApiClient"
import {
  buildPresentationStudioPatchPayloadFromRecord,
  mergePresentationStudioDraftWithRemote
} from "@/store/presentation-studio"
import { usePresentationStudioStore } from "@/store/presentation-studio"

const AUTOSAVE_DELAY_MS = 800

const toEtag = (version: number | null | undefined): string | null =>
  typeof version === "number" && Number.isFinite(version) ? `W/"v${version}"` : null

const toErrorMessage = (error: unknown): string =>
  error instanceof Error ? error.message || "autosave_failed" : String(error || "autosave_failed")

export const usePresentationStudioAutosave = (): void => {
  const projectId = usePresentationStudioStore((state) => state.projectId)
  const etag = usePresentationStudioStore((state) => state.etag)
  const isDirty = usePresentationStudioStore((state) => state.isDirty)
  const buildPatchPayload = usePresentationStudioStore((state) => state.buildPatchPayload)
  const setAutosaveState = usePresentationStudioStore((state) => state.setAutosaveState)
  const markPersisted = usePresentationStudioStore((state) => state.markPersisted)
  const loadProject = usePresentationStudioStore((state) => state.loadProject)

  React.useEffect(() => {
    if (!projectId || !etag || !isDirty) {
      return
    }

    const timer = window.setTimeout(async () => {
      const localDraft = buildPatchPayload()
      setAutosaveState("saving")
      try {
        const updated = await tldwClient.patchPresentation(projectId, localDraft, {
          ifMatch: etag
        })
        markPersisted(toEtag(updated.version), updated)
      } catch (error) {
        const message = toErrorMessage(error)
        if (message.includes("412") || message.includes("precondition_failed")) {
          try {
            const detail = await tldwClient.getPresentation(projectId)
            if (detail.record.content_kind !== "structured_slides") {
              throw new Error("Structured presentation required")
            }
            if (!detail.etag) {
              throw new Error("Presentation autosave conflict response missing ETag")
            }
            const latest = detail.record
            const mergedProject = mergePresentationStudioDraftWithRemote(latest, localDraft)
            const latestEtag = detail.etag
            loadProject(mergedProject, {
              etag: latestEtag,
              preserveDirty: true
            })
            const updated = await tldwClient.patchPresentation(
              projectId,
              buildPresentationStudioPatchPayloadFromRecord(mergedProject),
              {
                ifMatch: latestEtag
              }
            )
            markPersisted(toEtag(updated.version), updated)
            return
          } catch (reloadError) {
            setAutosaveState("error", toErrorMessage(reloadError))
            return
          }
        }
        setAutosaveState("error", message)
      }
    }, AUTOSAVE_DELAY_MS)

    return () => window.clearTimeout(timer)
  }, [
    buildPatchPayload,
    etag,
    isDirty,
    loadProject,
    markPersisted,
    projectId,
    setAutosaveState
  ])
}

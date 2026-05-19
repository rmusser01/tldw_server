import React from "react"
import { Button, Modal } from "antd"
import type { MessageInstance } from "antd/es/message/interface"
import type { TFunction } from "i18next"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import type { ArtifactType, GeneratedArtifact } from "@/types/workspace"
import {
  WORKSPACE_UNDO_WINDOW_MS,
  scheduleWorkspaceUndoAction,
  undoWorkspaceAction
} from "../../undo-manager"

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

export const SLIDES_EXPORT_FORMATS: { value: string; label: string; ext: string }[] = [
  { value: "revealjs", label: "Reveal.js (ZIP)", ext: "zip" },
  { value: "markdown", label: "Markdown", ext: "md" },
  { value: "pdf", label: "PDF", ext: "pdf" },
  { value: "json", label: "JSON", ext: "json" }
]

const WORKSPACE_DISCUSS_EVENT = "workspace-playground:discuss-artifact"

type ArtifactDiscussDetail = {
  artifactId: string
  artifactType: ArtifactType
  title: string
  content: string
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null

// ─────────────────────────────────────────────────────────────────────────────
// Shared helpers
// ─────────────────────────────────────────────────────────────────────────────

export const downloadBlobFile = (blob: Blob, filename: string) => {
  const url = URL.createObjectURL(blob)
  const a = document.createElement("a")
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

export function getFileExtension(type: ArtifactType): string {
  switch (type) {
    case "summary":
    case "report":
    case "timeline":
      return "md"
    case "quiz":
    case "flashcards":
      return "json"
    case "mindmap":
      return "mmd"
    case "slides":
      return "md"
    case "data_table":
      return "csv"
    case "audio_overview":
      return "mp3"
    default:
      return "txt"
  }
}

export const getResponsiveArtifactModalProps = (
  isMobile: boolean,
  desktopWidth: number
): {
  width: number | string
  style?: React.CSSProperties
  styles?: {
    body?: React.CSSProperties
  }
} => {
  if (!isMobile) {
    return { width: desktopWidth }
  }

  return {
    width: "100%",
    style: { top: 0, paddingBottom: 0 },
    styles: {
      body: {
        maxHeight: "calc(100dvh - 96px)",
        overflowY: "auto"
      }
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Hook interface
// ─────────────────────────────────────────────────────────────────────────────

export interface UseArtifactExportDeps {
  messageApi: MessageInstance
  isMobile: boolean
  generatedArtifacts: GeneratedArtifact[]
  removeArtifact: (id: string) => void
  restoreArtifact: (artifact: GeneratedArtifact, options?: { index?: number }) => void
  captureToCurrentNote: (payload: {
    title: string
    content: string
    mode: "append" | "replace"
  }) => void
  t: TFunction
}

export function useArtifactExport(deps: UseArtifactExportDeps) {
  const {
    messageApi,
    isMobile,
    generatedArtifacts,
    removeArtifact,
    restoreArtifact,
    captureToCurrentNote,
    t,
  } = deps

  const handleDeleteArtifact = React.useCallback(
    (artifact: GeneratedArtifact) => {
      const artifactIndex = generatedArtifacts.findIndex(
        (entry) => entry.id === artifact.id
      )
      Modal.confirm({
        title: t("playground:studio.deleteOutputTitle", "Delete output?"),
        content: t(
          "playground:studio.deleteOutputDescription",
          "This generated output will be permanently removed."
        ),
        okText: t("common:delete", "Delete"),
        cancelText: t("common:cancel", "Cancel"),
        okButtonProps: { danger: true },
        onOk: () => {
          const undoHandle = scheduleWorkspaceUndoAction({
            apply: () => {
              removeArtifact(artifact.id)
            },
            undo: () => {
              restoreArtifact(artifact, { index: artifactIndex })
            }
          })

          const undoMessageKey = `workspace-artifact-undo-${undoHandle.id}`
          const maybeOpen = (messageApi as { open?: (config: unknown) => void })
            .open
          const messageConfig = {
            key: undoMessageKey,
            type: "warning",
            duration: WORKSPACE_UNDO_WINDOW_MS / 1000,
            content: t(
              "playground:studio.undoDeleteOutput",
              "Output deleted."
            ),
            btn: (
              <Button
                size="small"
                type="link"
                onClick={() => {
                  if (undoWorkspaceAction(undoHandle.id)) {
                    messageApi.success(
                      t("playground:studio.outputRestored", "Output restored")
                    )
                  }
                  messageApi.destroy(undoMessageKey)
                }}
              >
                {t("common:undo", "Undo")}
              </Button>
            )
          }
          if (typeof maybeOpen === "function") {
            maybeOpen(messageConfig)
          } else {
            const maybeWarning = (
              messageApi as { warning?: (content: string) => void }
            ).warning
            if (typeof maybeWarning === "function") {
              maybeWarning(
                t("playground:studio.undoDeleteOutput", "Output deleted.")
              )
            }
          }
        }
      })
    },
    [generatedArtifacts, messageApi, removeArtifact, restoreArtifact, t]
  )

  const handleDiscussArtifact = React.useCallback(
    (artifact: GeneratedArtifact) => {
      if (typeof window === "undefined") return
      const content = (artifact.content || "").trim()
      if (!content) {
        messageApi.warning(
          t(
            "playground:studio.discussNoContent",
            "This output has no text content to discuss yet."
          )
        )
        return
      }
      const detail: ArtifactDiscussDetail = {
        artifactId: artifact.id,
        artifactType: artifact.type,
        title: artifact.title,
        content
      }
      window.dispatchEvent(
        new CustomEvent<ArtifactDiscussDetail>(WORKSPACE_DISCUSS_EVENT, { detail })
      )
      messageApi.success(
        t(
          "playground:studio.discussSent",
          "Sent to chat. Ask a follow-up in the chat pane."
        )
      )
    },
    [messageApi, t]
  )

  const handleSaveArtifactToNotes = React.useCallback(
    (
      artifact: GeneratedArtifact,
      mode: "append" | "replace" = "append"
    ) => {
      const content = (artifact.content || "").trim()
      if (!content) {
        messageApi.warning(
          t(
            "playground:studio.notesCaptureNoContent",
            "This output has no text content to save."
          )
        )
        return
      }
      captureToCurrentNote({
        title: artifact.title,
        content,
        mode
      })
      messageApi.success(
        mode === "replace"
          ? t(
              "playground:studio.notesCaptureReplaced",
              "Output replaced the current note draft."
            )
          : t(
              "playground:studio.notesCaptureAppended",
              "Output added to your current note draft."
            )
      )
    },
    [captureToCurrentNote, messageApi, t]
  )

  const handleDownloadArtifact = React.useCallback(
    async (artifact: GeneratedArtifact, format?: string) => {
      // Handle audio download - use the audioUrl blob directly
      if (artifact.type === "audio_overview" && artifact.audioUrl) {
        const a = document.createElement("a")
        a.href = artifact.audioUrl
        a.download = `${artifact.title}.${artifact.audioFormat || "mp3"}`
        a.click()
        return
      }

      // Handle slides download with format selection
      if (artifact.type === "slides" && artifact.presentationId) {
        const exportFormat = (format || "markdown") as "revealjs" | "markdown" | "json" | "pdf"
        const formatConfig = SLIDES_EXPORT_FORMATS.find((f) => f.value === exportFormat)
        try {
          const blob = await tldwClient.exportPresentation(artifact.presentationId, exportFormat)
          const url = URL.createObjectURL(blob)
          const a = document.createElement("a")
          a.href = url
          a.download = `${artifact.title}.${formatConfig?.ext || "md"}`
          a.click()
          URL.revokeObjectURL(url)
          messageApi.success(t("common:downloadSuccess", "Downloaded successfully"))
        } catch (error) {
          messageApi.error(t("common:downloadError", "Download failed"))
        }
        return
      }

      if (artifact.type === "quiz") {
        const questions =
          isRecord(artifact.data) && Array.isArray(artifact.data.questions)
            ? artifact.data.questions
            : null

        if (questions) {
          const quizBlob = new Blob(
            [
              JSON.stringify(
                {
                  title: artifact.title,
                  questions
                },
                null,
                2
              )
            ],
            { type: "application/json" }
          )
          downloadBlobFile(quizBlob, `${artifact.title}.json`)
          return
        }

        if (artifact.content) {
          const quizTextBlob = new Blob([artifact.content], { type: "text/plain" })
          downloadBlobFile(quizTextBlob, `${artifact.title}.txt`)
        }
        return
      }

      if (artifact.type === "flashcards") {
        const flashcards =
          isRecord(artifact.data) && Array.isArray(artifact.data.flashcards)
            ? artifact.data.flashcards
            : null

        if (flashcards) {
          const flashcardsBlob = new Blob(
            [
              JSON.stringify(
                {
                  title: artifact.title,
                  flashcards
                },
                null,
                2
              )
            ],
            { type: "application/json" }
          )
          downloadBlobFile(flashcardsBlob, `${artifact.title}.json`)
          return
        }

        if (artifact.content) {
          const flashcardsTextBlob = new Blob([artifact.content], {
            type: "text/plain"
          })
          downloadBlobFile(flashcardsTextBlob, `${artifact.title}.txt`)
        }
        return
      }

      if (artifact.serverId && artifact.type !== "mindmap") {
        try {
          const blob = await tldwClient.downloadOutput(String(artifact.serverId))
          const url = URL.createObjectURL(blob)
          const a = document.createElement("a")
          a.href = url
          a.download = `${artifact.title}.${getFileExtension(artifact.type)}`
          a.click()
          URL.revokeObjectURL(url)
        } catch {
          messageApi.error(t("common:downloadError", "Download failed"))
        }
      } else if (artifact.content) {
        // Download text content
        const blob = new Blob([artifact.content], { type: "text/plain" })
        const url = URL.createObjectURL(blob)
        const a = document.createElement("a")
        a.href = url
        a.download = `${artifact.title}.${artifact.type === "mindmap" ? "mmd" : "txt"}`
        a.click()
        URL.revokeObjectURL(url)
      }
    },
    [messageApi, t]
  )

  // Show slides export format selection modal
  const handleSlidesDownload = React.useCallback(
    (artifact: GeneratedArtifact) => {
      if (!artifact.presentationId) {
        // Fallback to content download
        handleDownloadArtifact(artifact)
        return
      }

      const modal = Modal.info({
        title: t("playground:studio.selectExportFormat", "Select Export Format"),
        content: (
          <div className="mt-4 space-y-2">
            {SLIDES_EXPORT_FORMATS.map((format) => (
              <button
                key={format.value}
                type="button"
                onClick={() => {
                  modal.destroy()
                  handleDownloadArtifact(artifact, format.value)
                }}
                className="w-full rounded border border-border p-3 text-left hover:bg-surface2"
              >
                <div className="font-medium">{format.label}</div>
                <div className="text-xs text-text-muted">.{format.ext}</div>
              </button>
            ))}
          </div>
        ),
        footer: null,
        icon: null,
        width: 300
      })
    },
    [handleDownloadArtifact, t]
  )

  const handleIconButtonKeyDown = React.useCallback(
    (event: React.KeyboardEvent<HTMLButtonElement>) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault()
        event.currentTarget.click()
      }
    },
    []
  )

  return {
    handleDeleteArtifact,
    handleDiscussArtifact,
    handleSaveArtifactToNotes,
    handleDownloadArtifact,
    handleSlidesDownload,
    handleIconButtonKeyDown,
  }
}

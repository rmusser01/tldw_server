import React from "react"
import { Button, Dropdown } from "antd"
import { useTranslation } from "react-i18next"

import { tldwClient } from "@/services/tldw/TldwApiClient"
import { usePresentationStudioStore } from "@/store/presentation-studio"
import { downloadBlob } from "@/utils/download-blob"
import { SLIDES_EXPORT_FORMATS } from "@/utils/presentation-export-formats"

type Props = { presentationId: string; title: string; isAuthorityCurrent: () => boolean }

export const PresentationExportMenu: React.FC<Props> = ({
  presentationId,
  title,
  isAuthorityCurrent
}) => {
  const { t } = useTranslation(["playground", "common"])
  const isDirty = usePresentationStudioStore((state) => state.isDirty)
  const autosaveState = usePresentationStudioStore((state) => state.autosaveState)
  const [isExporting, setIsExporting] = React.useState(false)
  const [error, setError] = React.useState(false)
  const pending = React.useRef<AbortController | null>(null)

  React.useLayoutEffect(() => {
    setIsExporting(false)
    setError(false)
    return () => {
      pending.current?.abort()
      pending.current = null
    }
  }, [presentationId])

  const handleExport = async (format: (typeof SLIDES_EXPORT_FORMATS)[number]) => {
    const saved = usePresentationStudioStore.getState()
    if (
      !isAuthorityCurrent() ||
      pending.current ||
      saved.projectId !== presentationId ||
      saved.isDirty ||
      saved.autosaveState === "saving"
    )
      return
    const controller = new AbortController()
    pending.current = controller
    setIsExporting(true)
    setError(false)
    try {
      const blob = await tldwClient.exportPresentation(presentationId, format.value, {
        abortSignal: controller.signal
      })
      if (controller.signal.aborted || !isAuthorityCurrent()) return
      const current = usePresentationStudioStore.getState()
      if (current.projectId !== presentationId || current.isDirty || current.etag !== saved.etag) {
        throw new Error("Presentation changed during export")
      }
      downloadBlob(blob, `${title || "presentation"}.${format.ext}`)
    } catch {
      if (!controller.signal.aborted && isAuthorityCurrent()) setError(true)
    } finally {
      if (pending.current === controller) {
        pending.current = null
        setIsExporting(false)
      }
    }
  }

  const unsaved = isDirty || autosaveState === "saving"
  return (
    <div className="space-y-2">
      <Dropdown
        trigger={["click"]}
        disabled={unsaved || isExporting}
        menu={{
          items: SLIDES_EXPORT_FORMATS.map((format) => ({
            key: format.value,
            label: format.label,
            onClick: () => void handleExport(format)
          }))
        }}
      >
        <Button
          aria-label={t("playground:studio.exportSavedPresentation", "Export saved presentation")}
          disabled={unsaved || isExporting}
          loading={isExporting}
        >
          {t("common:export", "Export")}
        </Button>
      </Dropdown>
      {unsaved && (
        <p role="status" className="text-sm text-text-muted">
          {t("playground:studio.saveBeforeExport", "Save changes before exporting.")}
        </p>
      )}
      {error && (
        <p role="alert" className="text-sm text-danger">
          {t(
            "playground:studio.exportRetry",
            "Export failed. Your slides are unchanged. Try again after saving any changes."
          )}
        </p>
      )}
    </div>
  )
}

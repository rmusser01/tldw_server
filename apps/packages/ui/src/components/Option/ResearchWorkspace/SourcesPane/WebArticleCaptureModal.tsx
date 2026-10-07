import { useState } from "react"
import { Button, Modal } from "antd"
import { useTranslation } from "react-i18next"
import type { useResearchWebCapture } from "@/utils/use-research-web-capture"

/** Readable extracted text stays text, including HTML-looking article content. */
export function WebArticleCaptureModal({
  capture
}: {
  capture: ReturnType<typeof useResearchWebCapture>
}) {
  const { t } = useTranslation("playground")
  const [expanded, setExpanded] = useState(false)
  const text =
    capture.pending?.body.content?.full_extract ?? capture.preview?.text
  return (
    <Modal
      open={Boolean(capture.source)}
      title={t("sources.captureArticle", "Capture article")}
      onCancel={capture.cancel}
      footer={null}
      afterClose={() => setExpanded(false)}
    >
      <p className="mb-3">{capture.source?.title}</p>
      <p className="mb-3">
        {t(
          "sources.captureDisclosure",
          "Save creates an additional capture Note and a new source containing the complete extracted article snapshot. Original retrieved evidence stays intact."
        )}
      </p>
      <p className="mb-3 text-sm">
        {t(
          "sources.capturePublicOnly",
          "Public readable text only. Website sign-ins, browser cookies, and private-page access are not used."
        )}
      </p>
      {text != null && (
        <>
          <Button
            onClick={() => setExpanded(!expanded)}
            aria-expanded={expanded}
          >
            {expanded
              ? t("sources.collapseText", "Collapse text")
              : t("sources.expandText", "Expand text")}
          </Button>
          {expanded && (
            <pre className="my-3 max-h-96 overflow-auto whitespace-pre-wrap break-words">
              {text}
            </pre>
          )}
          <p className="my-2 text-sm">
            {t(
              "sources.captureCompleteText",
              "The complete text is saved; it is never silently shortened."
            )}
          </p>
        </>
      )}
      {capture.error && (
        <p role="alert" className="my-3 text-error">
          {capture.error}
        </p>
      )}
      {capture.notice && (
        <p role="status" className="my-3">
          {t("sources.textUnchanged", "Text unchanged")}
        </p>
      )}
      <div className="mt-4 flex justify-end gap-2">
        <Button onClick={capture.cancel}>
          {t("sources.cancelCapture", "Cancel")}
        </Button>
        {text == null ? (
          <Button
            type="primary"
            loading={capture.busy}
            onClick={() => void capture.extract()}
          >
            {capture.source?.webCapture
              ? t("sources.refreshCapture", "Refresh capture")
              : t("sources.captureArticle", "Capture article")}
          </Button>
        ) : (
          <Button
            type="primary"
            loading={capture.busy}
            disabled={Boolean(capture.notice)}
            onClick={() => void capture.save()}
          >
            {capture.pending
              ? t("sources.retryCapture", "Retry capture")
              : t("sources.saveCapture", "Save capture")}
          </Button>
        )}
      </div>
    </Modal>
  )
}

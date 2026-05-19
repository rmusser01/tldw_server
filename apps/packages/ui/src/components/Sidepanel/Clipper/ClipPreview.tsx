import type { PendingClipDraft } from "@/services/web-clipper/pending-draft"
import { useTranslation } from "react-i18next"

type ClipPreviewProps = {
  draft: PendingClipDraft
}

const formatClipType = (value: string) => value.replaceAll("_", " ")

const ClipPreview = ({ draft }: ClipPreviewProps) => {
  const { t } = useTranslation()

  return (
    <section className="rounded-xl border border-border bg-surface2 p-3 shadow-sm">
      <div className="flex flex-wrap items-center gap-2">
        <span className="rounded-full bg-primary/10 px-2 py-1 text-[11px] font-semibold uppercase tracking-[0.12em] text-primary">
          {formatClipType(draft.captureMetadata.actualType)}
        </span>
        <span className="text-xs text-text-muted">
          {t("sidepanel:clipper.previewRequested", "Requested")}{" "}
          {formatClipType(draft.requestedType)}
        </span>
      </div>

      <h2 className="mt-3 text-sm font-semibold text-text">{draft.pageTitle}</h2>
      <p className="mt-1 break-all text-xs text-text-muted">{draft.pageUrl}</p>

      {draft.userVisibleError ? (
        <div className="mt-3 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900">
          {draft.userVisibleError}
        </div>
      ) : null}

      <div className="mt-3 rounded-lg border border-border bg-background p-3">
        <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.12em] text-text-muted">
          {t("sidepanel:clipper.previewLabel", "Clip preview")}
        </div>
        <p className="max-h-40 overflow-y-auto whitespace-pre-wrap text-sm text-text">
          {draft.visibleBody}
        </p>
      </div>
    </section>
  )
}

export default ClipPreview

import React from "react"
import { useTranslation } from "react-i18next"

export const SelectedModelSaveFeedback = ({
  failed,
  onRetry
}: {
  failed: boolean
  onRetry: () => Promise<boolean>
}) => {
  const { t } = useTranslation("common")
  const [saving, setSaving] = React.useState(false)
  if (!failed) return null
  return (
    <div role="alert" className="flex flex-wrap items-center gap-2 text-xs text-warn">
      <span>{t("modelSelect.saveFailed", "This model is selected for this tab. Your preference could not be saved.")}</span>
      <button
        type="button"
        disabled={saving}
        aria-busy={saving}
        className="rounded px-2 py-1 underline disabled:opacity-50 focus-visible:outline focus-visible:outline-2"
        onClick={() => {
          setSaving(true)
          void onRetry().catch(() => undefined).finally(() => setSaving(false))
        }}
      >
        {t("modelSelect.retrySave", "Retry saving model")}
      </button>
    </div>
  )
}

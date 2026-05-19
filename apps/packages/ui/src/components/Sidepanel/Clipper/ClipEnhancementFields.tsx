import { useTranslation } from "react-i18next"

type ClipEnhancementFieldsProps = {
  runOcr: boolean
  runVlm: boolean
  onRunOcrChange: (nextValue: boolean) => void
  onRunVlmChange: (nextValue: boolean) => void
}

const ClipEnhancementFields = ({
  runOcr,
  runVlm,
  onRunOcrChange,
  onRunVlmChange
}: ClipEnhancementFieldsProps) => {
  const { t } = useTranslation()

  return (
    <section className="panel-card p-3">
      <div className="text-[11px] font-semibold uppercase tracking-[0.12em] text-text-muted">
        {t("sidepanel:clipper.enhancementLegend", "Enhancements")}
      </div>

      <div className="mt-3 space-y-3">
        <label className="flex items-start gap-3 text-sm text-text">
          <input
            type="checkbox"
            checked={runOcr}
            onChange={(event) => onRunOcrChange(event.target.checked)}
            className="mt-1"
          />
          <span>{t("sidepanel:clipper.runOcr", "Run OCR")}</span>
        </label>

        <label className="flex items-start gap-3 text-sm text-text">
          <input
            type="checkbox"
            checked={runVlm}
            onChange={(event) => onRunVlmChange(event.target.checked)}
            className="mt-1"
          />
          <span>{t("sidepanel:clipper.runVlm", "Run visual analysis")}</span>
        </label>
      </div>

      <p className="mt-3 text-xs text-text-muted">
        {t(
          "sidepanel:clipper.privacyDisclosure",
          "OCR and vision requests send captured clip content to your configured tldw server."
        )}
      </p>
    </section>
  )
}

export default ClipEnhancementFields

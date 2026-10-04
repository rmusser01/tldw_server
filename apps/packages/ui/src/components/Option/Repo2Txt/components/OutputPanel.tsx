import { useTranslation } from "react-i18next"

type OutputPanelProps = {
  output: string
  busy: boolean
  canGenerate: boolean
  onGenerate: () => void | Promise<void>
  onCopy: () => void | Promise<void>
  onDownload: () => void
}

export function OutputPanel({
  output,
  busy,
  canGenerate,
  onGenerate,
  onCopy,
  onDownload
}: OutputPanelProps) {
  const { t } = useTranslation(["option"])

  return (
    <section className="space-y-3">
      <header className="flex flex-wrap items-center gap-2">
        <h2 className="text-sm font-semibold">
          {t("option:repo2txt.output", { defaultValue: "Output" })}
        </h2>
        <button
          type="button"
          className="rounded border px-3 py-1.5 text-sm"
          onClick={() => void onGenerate()}
          disabled={!canGenerate || busy}
        >
          {t("option:repo2txt.generate", { defaultValue: "Generate Output" })}
        </button>
        <button
          type="button"
          className="rounded border px-3 py-1.5 text-sm"
          onClick={() => void onCopy()}
          disabled={output.trim().length === 0}
        >
          {t("option:repo2txt.copy", { defaultValue: "Copy" })}
        </button>
        <button
          type="button"
          className="rounded border px-3 py-1.5 text-sm"
          onClick={onDownload}
          disabled={output.trim().length === 0}
        >
          {t("option:repo2txt.download", { defaultValue: "Download" })}
        </button>
      </header>
      <textarea
        value={output}
        readOnly
        rows={14}
        className="w-full rounded border border-border bg-surface p-3 text-xs text-text"
        aria-label={t("option:repo2txt.outputPreviewAria", {
          defaultValue: "Repo2txt output preview"
        })}
      />
    </section>
  )
}

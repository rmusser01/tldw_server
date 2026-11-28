import { useEffect, useId, useRef, useState } from "react"
import { extractCompletedIngestJobTerminalData } from "@/services/tldw/ingest-job-results"
import { extractProcessingItems } from "./processing-results"
import type { WizardResultItem } from "./types"

type ExtractedTextViewerProps = {
  item: WizardResultItem
  text: string
  qi: (key: string, defaultValue: string) => string
  onBack: () => void
}

export function ExtractedTextViewer({ item, text, qi, onBack }: ExtractedTextViewerProps) {
  const headingId = useId()
  const heading = useRef<HTMLHeadingElement>(null)
  const [copyStatus, setCopyStatus] = useState<"copied" | "failed" | null>(null)
  useEffect(() => { heading.current?.focus() }, [])
  const origin = item.fileName || item.url
  const payload = extractCompletedIngestJobTerminalData(item.data)
  const articles = extractProcessingItems(payload)
  const article = articles.find(entry => origin && (entry?.url === origin || entry?.input_ref === origin)) ?? articles[0]
  const location = origin || article?.url || article?.input_ref
  const payloadTitle = payload && typeof payload === "object" && "title" in payload ? payload.title : undefined
  const title = [item.title, article?.title, article?.metadata?.title, payloadTitle]
    .find((value): value is string => typeof value === "string" && Boolean(value.trim()) && value !== location)

  const copyText = async () => {
    try {
      await navigator.clipboard.writeText(text)
      setCopyStatus("copied")
    } catch {
      setCopyStatus("failed")
    }
  }

  return (
    <section aria-labelledby={headingId} className="flex min-h-0 min-w-0 flex-1 flex-col"
      onKeyDown={event => {
        if (event.key === "Escape") {
          event.stopPropagation()
          onBack()
        }
      }}>
      <div className="qi-web-scroll">
        <h2 ref={heading} id={headingId} tabIndex={-1} className="text-xl font-semibold text-text">
          {qi("web.extractedText", "Extracted text")}
        </h2>
        {title && <p className="mt-2 break-words font-medium text-text">{title}</p>}
        {(location || !title) && <p className="mt-1 break-words text-sm text-text-muted">{location || item.id}</p>}
        <pre className="my-4 select-text whitespace-pre-wrap break-words border-t border-border pt-4 font-sans text-sm leading-relaxed text-text">
          {text}
        </pre>
      </div>
      <div className="qi-web-footer">
        {copyStatus === "copied" && <p role="status" className="w-full text-sm text-text-muted">{qi("web.textCopied", "Text copied.")}</p>}
        {copyStatus === "failed" && <p role="alert" className="w-full text-sm text-danger">{qi("web.textCopyFailed", "Select the text and copy it manually.")}</p>}
        <button type="button" onClick={onBack} className="rounded-md border border-border bg-surface px-4 py-2 text-sm font-medium text-text hover:bg-surface2">
          {qi("web.backToResults", "Back to results")}
        </button>
        <button type="button" onClick={copyText} className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90">
          {qi("web.copyText", "Copy text")}
        </button>
      </div>
    </section>
  )
}

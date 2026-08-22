import React from "react"

import type { StandaloneHtmlOutline } from "./standalone-html-outline.worker"

type StandaloneHtmlSafeOutlineProps = {
  status: "current" | "stale" | "failed"
  outline: StandaloneHtmlOutline | null
}

const STATUS_LABELS = {
  current: "Current",
  stale: "Stale",
  failed: "Failed"
} as const

export const StandaloneHtmlSafeOutline: React.FC<StandaloneHtmlSafeOutlineProps> = ({
  status,
  outline
}) => (
  <section aria-labelledby="standalone-html-outline-heading" className="space-y-3">
    <header className="flex flex-wrap items-center justify-between gap-2">
      <h2 id="standalone-html-outline-heading" className="text-sm font-semibold text-text">
        Safe outline — text only; code never runs in Studio
      </h2>
      <span role="status" className="rounded-full bg-surface2 px-2 py-1 text-xs text-text-muted">
        {STATUS_LABELS[status]}
      </span>
    </header>
    {status === "failed" ? (
      <p className="rounded-lg border border-warning/40 bg-warning/10 p-3 text-sm text-text">
        Outline unavailable
      </p>
    ) : null}
    {outline?.slides.length ? (
      <div className="space-y-3">
        {outline.slides.map((slide) => (
          <article
            key={slide.index}
            aria-label={`Slide ${slide.index}`}
            className="rounded-lg border border-border bg-bg p-4"
          >
            <h3 className="text-xs font-semibold uppercase tracking-wide text-text-muted">
              Slide {slide.index}
            </h3>
            <div className="mt-3 space-y-2">
              {slide.blocks.map((block, index) => (
                <p
                  key={`${block.kind}-${index}`}
                  dir="auto"
                  className="[unicode-bidi:isolate] whitespace-pre-wrap text-sm text-text"
                >
                  {block.text}
                </p>
              ))}
            </div>
            {slide.notes.length ? (
              <section aria-label={`Slide ${slide.index} speaker notes`} className="mt-4 border-t border-border pt-3">
                <h4 className="text-xs font-semibold text-text-muted">Notes</h4>
                {slide.notes.map((block, index) => (
                  <p
                    key={`${block.kind}-${index}`}
                    dir="auto"
                    className="[unicode-bidi:isolate] mt-1 whitespace-pre-wrap text-sm text-text-muted"
                  >
                    {block.text}
                  </p>
                ))}
              </section>
            ) : null}
          </article>
        ))}
      </div>
    ) : (
      <p className="rounded-lg border border-dashed border-border p-4 text-sm text-text-muted">
        No trusted slide text is available yet.
      </p>
    )}
  </section>
)

export default StandaloneHtmlSafeOutline

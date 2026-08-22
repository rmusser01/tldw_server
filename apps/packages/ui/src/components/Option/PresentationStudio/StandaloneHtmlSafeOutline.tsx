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

const TRUNCATION_MARKER = "... [truncated]"
const TRUNCATION_MARKER_SCALARS = Array.from(TRUNCATION_MARKER).length
const MAX_SLIDE_SCALARS = 20_000
const MAX_TOTAL_SCALARS = 100_000

const hasBlockTruncationMarker = (
  block: StandaloneHtmlOutline["slides"][number]["blocks"][number]
): boolean => block.truncated && block.text.endsWith(TRUNCATION_MARKER)

const hasInlineTruncationMarker = (
  slide: StandaloneHtmlOutline["slides"][number]
): boolean =>
  slide.blocks
    .concat(slide.notes)
    .some(hasBlockTruncationMarker)

const needsCardTruncationMarker = (
  slide: StandaloneHtmlOutline["slides"][number]
): boolean => slide.truncated && !hasInlineTruncationMarker(slide)

const fitOutlineForRendering = (outline: StandaloneHtmlOutline) => {
  const cardMarkerCount = outline.slides.filter(needsCardTruncationMarker).length
  const inlineMarkerCount = outline.slides.reduce(
    (total, slide) =>
      total + slide.blocks.concat(slide.notes).filter(hasBlockTruncationMarker).length,
    0
  )
  const showOutlineMarker =
    outline.truncated &&
    !outline.slides.some(
      (slide) => slide.truncated || hasInlineTruncationMarker(slide)
    )
  let totalContentRemaining = Math.max(
    0,
    MAX_TOTAL_SCALARS -
      (inlineMarkerCount + cardMarkerCount + (showOutlineMarker ? 1 : 0)) *
        TRUNCATION_MARKER_SCALARS
  )

  const slides = outline.slides.map((slide) => {
    const showCardMarker = needsCardTruncationMarker(slide)
    const slideInlineMarkerCount = slide.blocks
      .concat(slide.notes)
      .filter(hasBlockTruncationMarker).length
    let slideContentRemaining = Math.max(
      0,
      MAX_SLIDE_SCALARS -
        (slideInlineMarkerCount + (showCardMarker ? 1 : 0)) *
          TRUNCATION_MARKER_SCALARS
    )
    const fitBlocks = (blocks: typeof slide.blocks) => {
      const fitted: typeof slide.blocks = []
      for (const block of blocks) {
        const hasInlineMarker = hasBlockTruncationMarker(block)
        const content = hasInlineMarker
          ? block.text.slice(0, -TRUNCATION_MARKER.length)
          : block.text
        const scalars = Array.from(content)
        const accepted = Math.min(
          scalars.length,
          slideContentRemaining,
          totalContentRemaining
        )
        if (accepted > 0 || hasInlineMarker) {
          fitted.push({
            ...block,
            text: `${scalars.slice(0, accepted).join("")}${
              hasInlineMarker ? TRUNCATION_MARKER : ""
            }`
          })
        }
        slideContentRemaining -= accepted
        totalContentRemaining -= accepted
      }
      return fitted
    }
    return {
      ...slide,
      blocks: fitBlocks(slide.blocks),
      notes: fitBlocks(slide.notes),
      showCardMarker
    }
  })

  return { slides, showOutlineMarker }
}

export const StandaloneHtmlSafeOutline: React.FC<StandaloneHtmlSafeOutlineProps> = ({
  status,
  outline
}) => {
  const rendered = outline ? fitOutlineForRendering(outline) : null
  return (
    <section aria-labelledby="standalone-html-outline-heading" className="space-y-3">
    <header className="flex flex-wrap items-center justify-between gap-2">
      <h2 id="standalone-html-outline-heading" className="text-sm font-semibold text-text">
        Safe outline: text only; code never runs in Studio
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
    {rendered?.slides.length ? (
      <div className="space-y-3">
        {rendered.slides.map((slide) => (
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
              <details className="mt-4 border-t border-border pt-3">
                <summary className="cursor-pointer text-xs font-semibold text-text-muted">
                  Speaker notes
                </summary>
                <div aria-label={`Slide ${slide.index} speaker notes`}>
                  {slide.notes.map((block, index) => (
                    <p
                      key={`${block.kind}-${index}`}
                      dir="auto"
                      className="[unicode-bidi:isolate] mt-1 whitespace-pre-wrap text-sm text-text-muted"
                    >
                      {block.text}
                    </p>
                  ))}
                </div>
              </details>
            ) : null}
            {slide.showCardMarker ? (
              <p className="mt-3 text-sm text-text-muted">{TRUNCATION_MARKER}</p>
            ) : null}
          </article>
        ))}
        {rendered.showOutlineMarker ? (
          <p className="text-sm text-text-muted">{TRUNCATION_MARKER}</p>
        ) : null}
      </div>
    ) : (
      <p className="rounded-lg border border-dashed border-border p-4 text-sm text-text-muted">
        No trusted slide text is available yet.
      </p>
    )}
    </section>
  )
}

export default StandaloneHtmlSafeOutline

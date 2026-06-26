import React from "react"
import type { ManuscriptAnnotationResponse } from "@/services/writing-playground"
import { cn } from "@/libs/utils"
import {
  type WritingEditorAdapter,
  type WritingEditorRangeMeasurement,
  type WritingEditorSelection
} from "./writing-editor-adapter"
import { codePointOffsetToUtf16Offset } from "./writing-annotation-anchor-utils"
import { WritingAnnotationCard } from "./WritingAnnotationCard"

const CARD_GAP_PX = 8
const COLLAPSED_CARD_HEIGHT_PX = 96
const ACTIVE_CARD_HEIGHT_PX = 152

type MeasuredAnnotation = {
  annotation: ManuscriptAnnotationResponse
  selection: WritingEditorSelection
  measurement: WritingEditorRangeMeasurement
  top: number
}

export type WritingAnnotationMarginRailProps = {
  annotations: ManuscriptAnnotationResponse[]
  adapter: WritingEditorAdapter | null
  documentText: string
  activeAnnotationId?: string | null
  onActiveAnnotationChange?: (annotationId: string | null) => void
  onReviewSuggestedFix?: (annotation: ManuscriptAnnotationResponse) => void
  onCopySuggestedFix?: (annotation: ManuscriptAnnotationResponse) => void
  includeResolved?: boolean
  measurementVersion?: number
  className?: string
}

const resolveAnnotationSelection = (
  annotation: ManuscriptAnnotationResponse,
  documentText: string
): WritingEditorSelection | null => {
  const useDerived =
    annotation.anchor_status === "reattached" &&
    annotation.derived_start !== null &&
    annotation.derived_start !== undefined &&
    annotation.derived_end !== null &&
    annotation.derived_end !== undefined
  const start = useDerived ? annotation.derived_start : annotation.anchor_start
  const end = useDerived ? annotation.derived_end : annotation.anchor_end

  if (start === null || start === undefined || end === null || end === undefined) {
    return null
  }
  if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) {
    return null
  }

  const utf16Start = codePointOffsetToUtf16Offset(documentText, start)
  const utf16End = codePointOffsetToUtf16Offset(documentText, end)
  if (utf16End <= utf16Start) {
    return null
  }
  return { start: utf16Start, end: utf16End }
}

const isRailCandidate = (
  annotation: ManuscriptAnnotationResponse,
  includeResolved: boolean
): boolean => {
  if (!includeResolved && annotation.status === "resolved") return false
  if (annotation.status !== "open" && !includeResolved) return false
  if (annotation.target_type !== "scene" || annotation.scene_level) return false
  return (
    annotation.anchor_status === "attached" ||
    annotation.anchor_status === "reattached" ||
    annotation.anchor_status === "needs_review"
  )
}

const sortMeasuredAnnotations = (
  left: MeasuredAnnotation,
  right: MeasuredAnnotation
): number =>
  left.measurement.top - right.measurement.top ||
  left.annotation.created_at.localeCompare(right.annotation.created_at) ||
  left.annotation.id.localeCompare(right.annotation.id)

export function WritingAnnotationMarginRail({
  annotations,
  adapter,
  documentText,
  activeAnnotationId = null,
  onActiveAnnotationChange,
  onReviewSuggestedFix,
  onCopySuggestedFix,
  includeResolved = false,
  measurementVersion = 0,
  className
}: WritingAnnotationMarginRailProps) {
  const [layoutVersion, setLayoutVersion] = React.useState(0)
  const measureRange = adapter?.measureRange

  React.useEffect(() => {
    if (!measureRange) return undefined

    let frame: number | null = null
    const scheduleLayout = () => {
      if (frame !== null) return
      frame = window.requestAnimationFrame(() => {
        frame = null
        setLayoutVersion((version) => version + 1)
      })
    }

    window.addEventListener("resize", scheduleLayout)
    window.addEventListener("scroll", scheduleLayout, true)

    return () => {
      if (frame !== null) {
        window.cancelAnimationFrame(frame)
      }
      window.removeEventListener("resize", scheduleLayout)
      window.removeEventListener("scroll", scheduleLayout, true)
    }
  }, [measureRange])

  const measuredAnnotations = React.useMemo(() => {
    if (!measureRange) return []

    return annotations
      .filter((annotation) => isRailCandidate(annotation, includeResolved))
      .map((annotation): MeasuredAnnotation | null => {
        const selection = resolveAnnotationSelection(annotation, documentText)
        if (!selection) return null
        const measurement = measureRange(selection)
        if (!measurement) return null
        return {
          annotation,
          selection,
          measurement,
          top: measurement.top
        }
      })
      .filter((entry): entry is MeasuredAnnotation => entry !== null)
      .sort(sortMeasuredAnnotations)
      .reduce<MeasuredAnnotation[]>((entries, entry) => {
        const previous = entries.at(-1)
        if (!previous) {
          entries.push(entry)
          return entries
        }
        const previousHeight =
          previous.annotation.id === activeAnnotationId
            ? ACTIVE_CARD_HEIGHT_PX
            : COLLAPSED_CARD_HEIGHT_PX
        const minTop = previous.top + previousHeight + CARD_GAP_PX
        entries.push({
          ...entry,
          top: Math.max(entry.measurement.top, minTop)
        })
        return entries
      }, [])
  }, [
    activeAnnotationId,
    annotations,
    documentText,
    includeResolved,
    layoutVersion,
    measureRange,
    measurementVersion
  ])

  if (!measureRange || measuredAnnotations.length === 0) return null

  const railHeight = Math.max(
    ...measuredAnnotations.map((entry) => {
      const cardHeight =
        entry.annotation.id === activeAnnotationId
          ? ACTIVE_CARD_HEIGHT_PX
          : COLLAPSED_CARD_HEIGHT_PX
      return entry.top + cardHeight
    })
  )

  return (
    <aside
      aria-label="Manuscript annotations"
      className={cn("relative hidden w-56 shrink-0 xl:block", className)}
      style={{ minHeight: railHeight }}
    >
      {measuredAnnotations.map(({ annotation, selection, top }) => {
        const active = annotation.id === activeAnnotationId
        const cardId = `writing-annotation-margin-card-${annotation.id}`
        const describedById = `${cardId}-body`
        const inspectorRowId = `writing-annotation-inspector-row-${annotation.id}`
        return (
          <div
            key={annotation.id}
            data-testid="writing-annotation-margin-card"
            data-anchor-status={annotation.anchor_status}
            className="absolute left-0 right-0"
            style={{ top }}
          >
            <WritingAnnotationCard
              annotation={annotation}
              active={active}
              warning={annotation.anchor_status === "needs_review"}
              cardId={cardId}
              describedById={describedById}
              inspectorRowId={inspectorRowId}
              onFocus={() => {
                onActiveAnnotationChange?.(annotation.id)
                adapter?.focus()
                adapter?.setSelection(selection)
              }}
              onReviewSuggestedFix={onReviewSuggestedFix}
              onCopySuggestedFix={onCopySuggestedFix}
            />
          </div>
        )
      })}
    </aside>
  )
}

export default WritingAnnotationMarginRail

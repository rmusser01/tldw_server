import { type Change, diffLines, diffWordsWithSpace } from "diff"
import { useTranslation } from "react-i18next"

const WORD_DIFF_MAX_CHARS = 12_000
const LINE_DIFF_MAX_CHARS = 48_000
const LINE_DIFF_MAX_LINES = 800
const MAX_RENDERED_SEGMENTS = 400
const DIFF_TIMEOUT_MS = 50
const DIFF_MAX_EDIT_LENGTH = 1_000

export type PromptDiffProps = {
  original: string
  candidate: string
}

type Comparison = { kind: "segments"; segments: Change[] } | { kind: "plain" }

const comparePromptText = (original: string, candidate: string): Comparison => {
  const combinedLength = original.length + candidate.length
  const combinedLines =
    original.split(/\r?\n/).length + candidate.split(/\r?\n/).length

  if (
    combinedLength > LINE_DIFF_MAX_CHARS ||
    combinedLines > LINE_DIFF_MAX_LINES
  ) {
    return { kind: "plain" }
  }

  const options = {
    timeout: DIFF_TIMEOUT_MS,
    maxEditLength: DIFF_MAX_EDIT_LENGTH
  }
  const rawSegments =
    combinedLength <= WORD_DIFF_MAX_CHARS
      ? diffWordsWithSpace(original, candidate, options)
      : diffLines(original, candidate, options)

  if (!rawSegments) return { kind: "plain" }

  const segments: Change[] = []
  for (let index = 0; index < rawSegments.length; index += 1) {
    const current = rawSegments[index]
    const next = rawSegments[index + 1]
    const previous = segments[segments.length - 1]
    if (
      /^\s+$/.test(current.value) &&
      previous &&
      next &&
      Boolean(previous.added) === Boolean(next.added) &&
      Boolean(previous.removed) === Boolean(next.removed) &&
      (previous.added || previous.removed)
    ) {
      previous.value += current.value + next.value
      index += 1
      continue
    }
    if (
      previous &&
      Boolean(previous.added) === Boolean(current.added) &&
      Boolean(previous.removed) === Boolean(current.removed)
    ) {
      previous.value += current.value
      continue
    }
    segments.push({ ...current })
  }

  return segments.length <= MAX_RENDERED_SEGMENTS
    ? { kind: "segments", segments }
    : { kind: "plain" }
}

export function PromptDiff({ original, candidate }: PromptDiffProps) {
  const { t } = useTranslation(["common"])
  const comparison = comparePromptText(original, candidate)
  const added = t("common:promptAssist.added", "Added")
  const removed = t("common:promptAssist.removed", "Removed")

  if (comparison.kind === "plain") {
    return (
      <div className="space-y-3">
        <p
          role="status"
          aria-live="polite"
          className="text-sm text-muted-foreground">
          {t(
            "common:promptAssist.diffTooLarge",
            "This comparison is too large to highlight safely. Showing the plain candidate."
          )}
        </p>
        <textarea
          aria-label={t(
            "common:promptAssist.plainCandidateLabel",
            "Plain improved prompt candidate"
          )}
          className="min-h-48 w-full resize-y rounded-md border border-border bg-background p-3 font-mono text-sm"
          readOnly
          value={candidate}
        />
      </div>
    )
  }

  return (
    <div className="space-y-3">
      <div
        data-diff-legend
        className="flex flex-wrap gap-4 text-xs text-muted-foreground">
        <span className="font-medium text-danger">− {removed}</span>
        <span className="font-medium text-success underline decoration-2 underline-offset-2">
          + {added}
        </span>
      </div>
      <pre
        aria-label={t("common:promptAssist.changesLabel", "Prompt changes")}
        className="max-h-80 overflow-auto whitespace-pre-wrap break-words rounded-md border border-border bg-muted/30 p-3 font-mono text-sm leading-6">
        {comparison.segments.map((part, index) => {
          if (part.added) {
            return (
              <ins
                key={index}
                data-change="added"
                data-change-label={added}
                aria-label={`${added}: ${part.value}`}
                className="rounded-sm bg-success/10 text-inherit underline decoration-2 underline-offset-2">
                {part.value}
              </ins>
            )
          }
          if (part.removed) {
            return (
              <del
                key={index}
                data-change="removed"
                data-change-label={removed}
                aria-label={`${removed}: ${part.value}`}
                className="rounded-sm bg-danger/10 text-inherit line-through decoration-2">
                {part.value}
              </del>
            )
          }
          return (
            <span key={index} data-change="unchanged">
              {part.value}
            </span>
          )
        })}
      </pre>
    </div>
  )
}

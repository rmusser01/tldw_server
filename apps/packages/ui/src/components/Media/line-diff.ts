import { diffArrays } from 'diff'

export type DiffLine = { type: 'same' | 'add' | 'del'; text: string }
export const DIFF_SYNC_LINE_THRESHOLD = 4000
export const DIFF_HARD_CHAR_THRESHOLD = 300_000
export const DIFF_SAMPLED_CHAR_BUDGET = 120_000
const MAX_INPUT_LINES = 10_000
const SAMPLED_LINES_PER_SIDE = 4000
const OMITTED = '...[sampled middle omitted for performance]...'
const countLines = (text: string): number => String(text || '').split('\n').length

export const shouldUseWorkerDiff = (
  left: string,
  right: string,
  threshold = DIFF_SYNC_LINE_THRESHOLD
): boolean => countLines(left) + countLines(right) > threshold

export const shouldRequireSampling = (
  left: string,
  right: string,
  threshold = DIFF_HARD_CHAR_THRESHOLD
): boolean =>
  left.length + right.length > threshold || countLines(left) + countLines(right) > MAX_INPUT_LINES

export const sampleTextForDiff = (text: string, charBudget = DIFF_SAMPLED_CHAR_BUDGET): string => {
  let sampled = String(text || '')
  const half = Math.floor(charBudget / 2)
  if (sampled.length > charBudget)
    sampled = `${sampled.slice(0, half)}\n${OMITTED}\n${sampled.slice(-half)}`
  const lines = sampled.split('\n')
  if (lines.length > SAMPLED_LINES_PER_SIDE) {
    const halfLines = SAMPLED_LINES_PER_SIDE / 2
    sampled = [...lines.slice(0, halfLines), OMITTED, ...lines.slice(-halfLines)].join('\n')
  }
  return sampled
}

export function computeDiffSync(
  left: string,
  right: string,
  limits: { maxEditLength: number; timeout: number } = { maxEditLength: 1024, timeout: 100 }
): DiffLine[] {
  left = String(left || '')
  right = String(right || '')
  const sample = shouldRequireSampling(left, right)
  const a = (sample ? sampleTextForDiff(left) : left).split('\n')
  const b = (sample ? sampleTextForDiff(right) : right).split('\n')
  // ponytail: bounded edit search; keep common edges around a coarse replacement
  // if the budget runs out. Workers get a larger, still finite budget (TASK-13450).
  const changes = diffArrays(a, b, limits)
  if (!changes) {
    let start = 0
    let end = 0
    while (start < Math.min(a.length, b.length) && a[start] === b[start]) start++
    while (
      end < Math.min(a.length, b.length) - start &&
      a[a.length - 1 - end] === b[b.length - 1 - end]
    )
      end++
    return [
      ...a.slice(0, start).map((text) => ({ type: 'same' as const, text })),
      ...a.slice(start, a.length - end).map((text) => ({ type: 'del' as const, text })),
      ...b.slice(start, b.length - end).map((text) => ({ type: 'add' as const, text })),
      ...a.slice(a.length - end).map((text) => ({ type: 'same' as const, text }))
    ]
  }
  return changes.flatMap((change) =>
    change.value.map((text) => ({
      type: change.added ? ('add' as const) : change.removed ? ('del' as const) : ('same' as const),
      text
    }))
  )
}

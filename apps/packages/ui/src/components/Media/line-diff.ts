import { diffArrays } from 'diff'

export type DiffLine = { type: 'same' | 'add' | 'del'; text: string }
export const DIFF_SYNC_LINE_THRESHOLD = 4000
export const DIFF_HARD_CHAR_THRESHOLD = 300_000
export const DIFF_SAMPLED_CHAR_BUDGET = 120_000
const MAX_INPUT_LINES = 10_000
const SAMPLED_LINES_PER_SIDE = 4000
const OMITTED = '...[sampled middle omitted for performance]...'
const countLines = (text: string) => String(text || '').split('\n').length

export const shouldUseWorkerDiff = (
  left: string,
  right: string,
  threshold = DIFF_SYNC_LINE_THRESHOLD
) => countLines(left) + countLines(right) > threshold

export const shouldRequireSampling = (
  left: string,
  right: string,
  threshold = DIFF_HARD_CHAR_THRESHOLD
) =>
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

export function computeDiffSync(left: string, right: string): DiffLine[] {
  left = String(left || '')
  right = String(right || '')
  const sample = shouldRequireSampling(left, right)
  const a = (sample ? sampleTextForDiff(left) : left).split('\n')
  const b = (sample ? sampleTextForDiff(right) : right).split('\n')
  // ponytail: bounded edit search; show a complete replacement for very divergent
  // inputs instead of allocating an unbounded LCS matrix (TASK-13450).
  const changes = diffArrays(a, b, { maxEditLength: 256, timeout: 100 })
  if (!changes)
    return [
      ...a.map((text) => ({ type: 'del' as const, text })),
      ...b.map((text) => ({ type: 'add' as const, text }))
    ]
  return changes.flatMap((change) =>
    change.value.map((text) => ({
      type: change.added ? ('add' as const) : change.removed ? ('del' as const) : ('same' as const),
      text
    }))
  )
}

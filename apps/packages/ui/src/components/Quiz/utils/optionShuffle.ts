export type ShuffledOptionEntry = {
  originalIndex: number
  label: string
}

const normalizeSeed = (value: number): number => {
  if (!Number.isFinite(value)) return 1
  const normalized = Math.floor(Math.abs(value)) >>> 0
  return normalized === 0 ? 1 : normalized
}

const createMulberry32 = (seed: number) => {
  let state = normalizeSeed(seed)
  return () => {
    state += 0x6d2b79f5
    let t = state
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

const mixQuestionSeed = (sessionSeed: number, questionId: number): number => {
  const seed = normalizeSeed(sessionSeed)
  const question = normalizeSeed(questionId)
  let mixed = seed ^ Math.imul(question, 0x9e3779b1)
  mixed = Math.imul(mixed ^ (mixed >>> 16), 0x45d9f3b)
  mixed = Math.imul(mixed ^ (mixed >>> 16), 0x45d9f3b)
  mixed ^= mixed >>> 16
  return normalizeSeed(mixed)
}

export const buildShuffledIndexOrder = (length: number, seed: number): number[] => {
  const safeLength = Math.max(0, Math.floor(length))
  const order = Array.from({ length: safeLength }, (_, index) => index)
  if (safeLength <= 1) return order

  const random = createMulberry32(seed)
  for (let index = order.length - 1; index > 0; index -= 1) {
    const target = Math.floor(random() * (index + 1))
    const next = order[index]
    order[index] = order[target]
    order[target] = next
  }
  return order
}

export const buildShuffledOptionEntries = (
  options: string[],
  questionId: number,
  sessionSeed: number
): ShuffledOptionEntry[] => {
  if (!Array.isArray(options) || options.length === 0) return []
  if (options.length === 1) {
    return [{ originalIndex: 0, label: options[0] }]
  }

  const seed = mixQuestionSeed(sessionSeed, questionId)
  const order = buildShuffledIndexOrder(options.length, seed)
  return order.map((originalIndex) => ({
    originalIndex,
    label: options[originalIndex] ?? ""
  }))
}

export const drawDeterministicQuestionPool = <T>(
  items: T[],
  drawCount: number,
  seed: number
): T[] => {
  if (!Array.isArray(items) || items.length === 0) return []
  const normalizedDrawCount = Math.min(items.length, Math.max(1, Math.floor(drawCount)))
  if (normalizedDrawCount >= items.length) {
    return [...items]
  }
  const order = buildShuffledIndexOrder(items.length, seed)
  return order.slice(0, normalizedDrawCount).map((index) => items[index] as T)
}

export const drawDeterministicGroupedQuestionPool = <T>(
  items: T[],
  drawCount: number,
  seed: number,
  getGroupKey: (item: T) => string | null | undefined
): T[] => {
  if (!Array.isArray(items) || items.length === 0) return []
  const normalizedDrawCount = Math.min(items.length, Math.max(1, Math.floor(drawCount)))
  if (normalizedDrawCount >= items.length) return [...items]

  const units: T[][] = []
  const groupedUnits = new Map<string, T[]>()
  items.forEach((item) => {
    const groupKey = getGroupKey(item)?.trim() ?? ""
    if (!groupKey) {
      units.push([item])
      return
    }
    const existingUnit = groupedUnits.get(groupKey)
    if (existingUnit) {
      existingUnit.push(item)
      return
    }
    const unit = [item]
    groupedUnits.set(groupKey, unit)
    units.push(unit)
  })

  const selected: T[] = []
  const order = buildShuffledIndexOrder(units.length, seed)
  for (const unitIndex of order) {
    const unit = units[unitIndex]
    if (unit) selected.push(...unit)
    if (selected.length >= normalizedDrawCount) break
  }
  return selected
}

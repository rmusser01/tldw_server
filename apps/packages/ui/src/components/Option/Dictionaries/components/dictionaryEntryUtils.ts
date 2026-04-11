export type InlineEditableEntryField = "pattern" | "replacement"

export type TextDiffSegment = {
  type: "unchanged" | "removed" | "added"
  text: string
}

export function validateRegexPattern(pattern: string): string | null {
  if (!pattern) return null
  try {
    // Check if it looks like a regex pattern (starts and ends with /)
    const regexMatch = pattern.match(/^\/(.*)\/([gimsuvy]*)$/)
    if (regexMatch) {
      new RegExp(regexMatch[1], regexMatch[2])
    } else {
      // Try as plain regex
      new RegExp(pattern)
    }
    return null
  } catch (e: any) {
    return humanizeRegexError(e.message || "Invalid regex pattern")
  }
}

const REGEX_ERROR_PATTERNS: Array<{ pattern: RegExp; message: string; fix: string }> = [
  {
    pattern: /unterminated character class/i,
    message: "Opening bracket [ has no closing ].",
    fix: "Add a closing ] to your character class.",
  },
  {
    pattern: /unterminated group/i,
    message: "Opening parenthesis ( has no closing ).",
    fix: "Add a closing ) to your group.",
  },
  {
    pattern: /nothing to repeat/i,
    message: "A repeat symbol (*, +, ?) has nothing before it.",
    fix: "Put a character or group before the repeat symbol.",
  },
  {
    pattern: /invalid escape/i,
    message: "A backslash \\ is followed by an invalid character.",
    fix: "Use a valid escape like \\d (digit), \\w (word), or \\s (space).",
  },
  {
    pattern: /(?:^|[\s:])(unmatched|lone)(?=$|[\s:)\]}])/i,
    message: "A bracket or parenthesis is not properly paired.",
    fix: "Check that every opening ( [ { has a matching closing ) ] }.",
  },
  {
    pattern: /invalid quantifier/i,
    message: "A repeat range like {n,m} is not formatted correctly.",
    fix: "Use the format {min,max}, e.g., {1,3} for 1 to 3 repeats.",
  },
]

const VALIDATION_CODE_LABELS: Record<string, { label: string; fix?: string }> = {
  regex_catastrophic_backtracking: {
    label: "Slow pattern",
    fix: "This pattern may run very slowly. Simplify nested repeating groups like (.+)+ or (.*)*.",
  },
  regex_invalid: {
    label: "Invalid pattern",
    fix: "The regex pattern could not be compiled. Check syntax.",
  },
  pattern_empty: {
    label: "Empty pattern",
    fix: "Add text to the pattern field.",
  },
  pattern_duplicate: {
    label: "Duplicate pattern",
    fix: "Another entry already uses this pattern. Remove one to avoid conflicts.",
  },
  replacement_empty: {
    label: "Empty replacement",
    fix: "Add text to the replacement field.",
  },
}

export function humanizeRegexError(rawMessage: string): string {
  for (const { pattern, message } of REGEX_ERROR_PATTERNS) {
    if (pattern.test(rawMessage)) {
      return message
    }
  }
  return rawMessage
}

export function humanizeValidationCode(code: string): { label: string; fix?: string } {
  return VALIDATION_CODE_LABELS[code] ?? { label: code }
}

export function toSafeNonNegativeInteger(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    return 0
  }
  return Math.floor(value)
}

export function buildTimedEffectsPayload(
  source: unknown,
  options: { forceObject?: boolean } = {}
): { sticky: number; cooldown: number; delay: number } | undefined {
  const forceObject = options.forceObject === true
  const raw =
    source && typeof source === "object"
      ? (source as Record<string, unknown>)
      : null

  if (!raw && !forceObject) {
    return undefined
  }

  const sticky = toSafeNonNegativeInteger(raw?.sticky)
  const cooldown = toSafeNonNegativeInteger(raw?.cooldown)
  const delay = toSafeNonNegativeInteger(raw?.delay)

  if (!forceObject) {
    const hasInput = Boolean(raw) && ["sticky", "cooldown", "delay"].some((key) => {
      const value = raw?.[key]
      return value !== null && value !== undefined && value !== ""
    })
    if (!hasInput) {
      return undefined
    }
  }

  return {
    sticky,
    cooldown,
    delay
  }
}

export function normalizeProbabilityValue(value: unknown, fallback = 1): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return fallback
  }
  return Math.min(1, Math.max(0, value))
}

export function formatProbabilityFrequencyHint(value: unknown): string {
  const normalizedValue = normalizeProbabilityValue(value, 1)
  const percent = Math.round(normalizedValue * 100)
  const outOfTen = Math.round(normalizedValue * 10)
  return `Fires ~${outOfTen} out of 10 messages (${percent}%).`
}

function tokenizeDiffText(source: string): string[] {
  return source.split(/(\s+)/).filter((token) => token.length > 0)
}

function appendDiffSegment(
  segments: TextDiffSegment[],
  type: TextDiffSegment["type"],
  text: string
) {
  if (!text) return
  const previous = segments[segments.length - 1]
  if (previous && previous.type === type) {
    previous.text += text
    return
  }
  segments.push({ type, text })
}

export function buildTextDiffSegments(
  originalText: string,
  processedText: string
): TextDiffSegment[] {
  const originalTokens = tokenizeDiffText(originalText)
  const processedTokens = tokenizeDiffText(processedText)
  const originalLength = originalTokens.length
  const processedLength = processedTokens.length

  if (originalLength === 0 && processedLength === 0) {
    return []
  }

  const lcs: number[][] = Array.from({ length: originalLength + 1 }, () =>
    Array(processedLength + 1).fill(0)
  )

  for (let originalIndex = originalLength - 1; originalIndex >= 0; originalIndex -= 1) {
    for (let processedIndex = processedLength - 1; processedIndex >= 0; processedIndex -= 1) {
      if (originalTokens[originalIndex] === processedTokens[processedIndex]) {
        lcs[originalIndex][processedIndex] = lcs[originalIndex + 1][processedIndex + 1] + 1
      } else {
        lcs[originalIndex][processedIndex] = Math.max(
          lcs[originalIndex + 1][processedIndex],
          lcs[originalIndex][processedIndex + 1]
        )
      }
    }
  }

  const segments: TextDiffSegment[] = []
  let originalIndex = 0
  let processedIndex = 0

  while (originalIndex < originalLength && processedIndex < processedLength) {
    if (originalTokens[originalIndex] === processedTokens[processedIndex]) {
      appendDiffSegment(segments, "unchanged", originalTokens[originalIndex])
      originalIndex += 1
      processedIndex += 1
      continue
    }

    if (lcs[originalIndex + 1][processedIndex] >= lcs[originalIndex][processedIndex + 1]) {
      appendDiffSegment(segments, "removed", originalTokens[originalIndex])
      originalIndex += 1
      continue
    }

    appendDiffSegment(segments, "added", processedTokens[processedIndex])
    processedIndex += 1
  }

  while (originalIndex < originalLength) {
    appendDiffSegment(segments, "removed", originalTokens[originalIndex])
    originalIndex += 1
  }
  while (processedIndex < processedLength) {
    appendDiffSegment(segments, "added", processedTokens[processedIndex])
    processedIndex += 1
  }

  return segments
}

export function extractRegexSafetyMessage(validationReport: any): string | null {
  const errors = Array.isArray(validationReport?.errors)
    ? validationReport.errors
    : []

  if (errors.length === 0) {
    return null
  }

  const regexIssue = errors.find((issue: any) => {
    const code = String(issue?.code || "").toLowerCase()
    const field = String(issue?.field || "").toLowerCase()
    const message = String(issue?.message || "").toLowerCase()
    return (
      code.startsWith("regex_") ||
      field.endsWith(".pattern") ||
      message.includes("regex")
    )
  })

  if (regexIssue?.message) {
    return String(regexIssue.message)
  }

  const firstError = errors[0]
  if (firstError?.message) {
    return String(firstError.message)
  }

  return "Regex pattern failed server validation."
}

export function buildRestorableDictionaryEntryPayload(entry: any): Record<string, any> {
  const payload: Record<string, any> = {
    pattern: typeof entry?.pattern === "string" ? entry.pattern : "",
    replacement: typeof entry?.replacement === "string" ? entry.replacement : "",
    type: entry?.type === "regex" ? "regex" : "literal",
    enabled: typeof entry?.enabled === "boolean" ? entry.enabled : true,
    case_sensitive:
      typeof entry?.case_sensitive === "boolean" ? entry.case_sensitive : true,
    probability:
      typeof entry?.probability === "number" && Number.isFinite(entry.probability)
        ? Math.min(1, Math.max(0, entry.probability))
        : 1,
    max_replacements:
      Number.isInteger(entry?.max_replacements) && entry.max_replacements >= 0
        ? entry.max_replacements
        : 0
  }
  if (typeof entry?.group === "string" && entry.group.trim()) {
    payload.group = entry.group
  }
  if (entry?.timed_effects && typeof entry.timed_effects === "object") {
    payload.timed_effects = entry.timed_effects
  }
  return payload
}

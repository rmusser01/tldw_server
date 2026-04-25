export type InlineEditableEntryField = "pattern" | "replacement"
export type DictionaryTextLocalizer = (key: string, fallback: string) => string

export type TextDiffSegment = {
  type: "unchanged" | "removed" | "added"
  text: string
}

const localizeDictionaryText = (
  localize: DictionaryTextLocalizer | undefined,
  key: string,
  fallback: string
): string => (localize ? localize(key, fallback) : fallback)

export function validateRegexPattern(
  pattern: string,
  localize?: DictionaryTextLocalizer
): string | null {
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
    return humanizeRegexError(
      e.message || "Invalid regex pattern",
      localize
    )
  }
}

const REGEX_ERROR_PATTERNS: Array<{
  pattern: RegExp
  messageKey: string
  messageDefault: string
  fixKey: string
  fixDefault: string
}> = [
  {
    pattern: /unterminated character class/i,
    messageKey: "option:dictionaries.validation.regex.unterminatedCharacterClass",
    messageDefault: "Opening bracket [ has no closing ].",
    fixKey: "option:dictionaries.validation.regex.unterminatedCharacterClassFix",
    fixDefault: "Add a closing ] to your character class.",
  },
  {
    pattern: /unterminated group/i,
    messageKey: "option:dictionaries.validation.regex.unterminatedGroup",
    messageDefault: "Opening parenthesis ( has no closing ).",
    fixKey: "option:dictionaries.validation.regex.unterminatedGroupFix",
    fixDefault: "Add a closing ) to your group.",
  },
  {
    pattern: /nothing to repeat/i,
    messageKey: "option:dictionaries.validation.regex.nothingToRepeat",
    messageDefault: "A repeat symbol (*, +, ?) has nothing before it.",
    fixKey: "option:dictionaries.validation.regex.nothingToRepeatFix",
    fixDefault: "Put a character or group before the repeat symbol.",
  },
  {
    pattern: /invalid escape/i,
    messageKey: "option:dictionaries.validation.regex.invalidEscape",
    messageDefault: "A backslash \\ is followed by an invalid character.",
    fixKey: "option:dictionaries.validation.regex.invalidEscapeFix",
    fixDefault: "Use a valid escape like \\d (digit), \\w (word), or \\s (space).",
  },
  {
    pattern: /(?:^|[\s:])(unmatched|lone)(?=$|[\s:)\]}])/i,
    messageKey: "option:dictionaries.validation.regex.unmatchedDelimiter",
    messageDefault: "A bracket or parenthesis is not properly paired.",
    fixKey: "option:dictionaries.validation.regex.unmatchedDelimiterFix",
    fixDefault: "Check that every opening ( [ { has a matching closing ) ] }.",
  },
  {
    pattern: /invalid quantifier/i,
    messageKey: "option:dictionaries.validation.regex.invalidQuantifier",
    messageDefault: "A repeat range like {n,m} is not formatted correctly.",
    fixKey: "option:dictionaries.validation.regex.invalidQuantifierFix",
    fixDefault: "Use the format {min,max}, e.g., {1,3} for 1 to 3 repeats.",
  },
]

const VALIDATION_CODE_LABELS: Record<
  string,
  {
    labelKey: string
    labelDefault: string
    fixKey?: string
    fixDefault?: string
  }
> = {
  regex_catastrophic_backtracking: {
    labelKey: "option:dictionaries.validation.code.regexCatastrophicBacktracking.label",
    labelDefault: "Slow pattern",
    fixKey: "option:dictionaries.validation.code.regexCatastrophicBacktracking.fix",
    fixDefault:
      "This pattern may run very slowly. Simplify nested repeating groups like (.+)+ or (.*)*.",
  },
  regex_invalid: {
    labelKey: "option:dictionaries.validation.code.regexInvalid.label",
    labelDefault: "Invalid pattern",
    fixKey: "option:dictionaries.validation.code.regexInvalid.fix",
    fixDefault: "The regex pattern could not be compiled. Check syntax.",
  },
  pattern_empty: {
    labelKey: "option:dictionaries.validation.code.patternEmpty.label",
    labelDefault: "Empty pattern",
    fixKey: "option:dictionaries.validation.code.patternEmpty.fix",
    fixDefault: "Add text to the pattern field.",
  },
  pattern_duplicate: {
    labelKey: "option:dictionaries.validation.code.patternDuplicate.label",
    labelDefault: "Duplicate pattern",
    fixKey: "option:dictionaries.validation.code.patternDuplicate.fix",
    fixDefault:
      "Another entry already uses this pattern. Remove one to avoid conflicts.",
  },
  replacement_empty: {
    labelKey: "option:dictionaries.validation.code.replacementEmpty.label",
    labelDefault: "Empty replacement",
    fixKey: "option:dictionaries.validation.code.replacementEmpty.fix",
    fixDefault: "Add text to the replacement field.",
  },
}

export function humanizeRegexError(
  rawMessage: string,
  localize?: DictionaryTextLocalizer
): string {
  for (const { pattern, messageKey, messageDefault } of REGEX_ERROR_PATTERNS) {
    if (pattern.test(rawMessage)) {
      return localizeDictionaryText(localize, messageKey, messageDefault)
    }
  }
  return rawMessage
}

export function humanizeValidationCode(
  code: string,
  localize?: DictionaryTextLocalizer
): { label: string; fix?: string } {
  const copy = VALIDATION_CODE_LABELS[code]
  if (!copy) {
    return { label: code }
  }

  return {
    label: localizeDictionaryText(localize, copy.labelKey, copy.labelDefault),
    fix: copy.fixKey
      ? localizeDictionaryText(localize, copy.fixKey, copy.fixDefault ?? "")
      : undefined,
  }
}

export function toSafeNonNegativeInteger(value: unknown): number {
  const normalizedValue =
    typeof value === "string" && value.trim() !== "" ? Number(value) : value

  if (
    typeof normalizedValue !== "number" ||
    !Number.isFinite(normalizedValue) ||
    normalizedValue < 0
  ) {
    return 0
  }
  return Math.floor(normalizedValue)
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

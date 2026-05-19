import type {
  BlocklistManagedItem,
  ModerationOverrideRule,
  ModerationUserOverride
} from "@/services/moderation"

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type ModerationScope = "server" | "user"

export interface SettingsDraft {
  piiEnabled: boolean
  categoriesEnabled: string[]
  persist: boolean
}

export type ManagedBlocklistRowKind = "literal" | "regex" | "comment" | "empty"

export interface NormalizedManagedBlocklistRow extends BlocklistManagedItem {
  rowKind: ManagedBlocklistRowKind
  pattern: string
  action: "block" | "redact" | "warn"
  replacement: string | null
  categories: string[]
  isActive: boolean
  isValid: boolean
  statusLabel: "Active" | "Invalid" | "Comment" | "Blank"
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

export const ONBOARDING_KEY = "moderation-playground-onboarded"

export const CATEGORY_SUGGESTIONS = [
  { value: "violence", label: "Violence", description: "Kill, murder, weapon, assault, bomb", severity: "critical" },
  { value: "self_harm", label: "Self-Harm", description: "Suicide, self-harm, cutting, overdose", severity: "critical" },
  { value: "sexual_content", label: "Sexual Content", description: "Sex, porn, nude, erotic, nsfw", severity: "high" },
  { value: "hate_speech", label: "Hate Speech", description: "Racist, sexist, homophobic, bigot", severity: "high" },
  { value: "pii", label: "PII (Personal Info)", description: "SSN, credit cards, phone numbers", severity: "high" },
  { value: "pii_email", label: "Email Addresses", description: "Email address patterns", severity: "medium" },
  { value: "pii_phone", label: "Phone Numbers", description: "US phone number patterns", severity: "medium" },
  { value: "profanity", label: "Profanity", description: "Damn, hell, crap", severity: "low" },
  { value: "drugs_alcohol", label: "Drugs & Alcohol", description: "Marijuana, cocaine, alcohol, drunk", severity: "medium" },
  { value: "gambling", label: "Gambling", description: "Casino, poker, slot machine, lottery", severity: "medium" },
  { value: "confidential", label: "Confidential", description: "Custom confidential content rules", severity: "high" }
] as const

export const ACTION_OPTIONS = [
  { value: "block", label: "Block", description: "Reject the message entirely" },
  { value: "redact", label: "Redact", description: "Replace flagged content with [REDACTED]" },
  { value: "warn", label: "Warn", description: "Allow but record in logs" }
] as const

export const PRESET_PROFILES: Record<
  string,
  { label: string; description: string; payload: ModerationUserOverride }
> = {
  strict: {
    label: "Strict",
    description: "Block risky inputs and redact sensitive outputs.",
    payload: {
      enabled: true,
      input_enabled: true,
      output_enabled: true,
      input_action: "block",
      output_action: "redact"
    }
  },
  balanced: {
    label: "Balanced",
    description: "Warn on inputs, redact outputs.",
    payload: {
      enabled: true,
      input_enabled: true,
      output_enabled: true,
      input_action: "warn",
      output_action: "redact"
    }
  },
  monitor: {
    label: "Monitor",
    description: "Warn only, never block.",
    payload: {
      enabled: true,
      input_enabled: true,
      output_enabled: true,
      input_action: "warn",
      output_action: "warn"
    }
  }
}

// ---------------------------------------------------------------------------
// Pure utility functions
// ---------------------------------------------------------------------------

export const normalizeCategories = (value: unknown): string[] => {
  if (Array.isArray(value)) {
    return value.map((item) => String(item).trim()).filter(Boolean)
  }
  if (typeof value === "string") {
    return value
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean)
  }
  return []
}

export const formatJson = (value: unknown) => {
  try {
    return JSON.stringify(value ?? {}, null, 2)
  } catch {
    return "{}"
  }
}

export const normalizeRuleIsRegex = (value: unknown): boolean | null => {
  if (typeof value === "boolean") return value
  if (value === undefined || value === null) return false
  return null
}

export const formatRulePhase = (phase: ModerationOverrideRule["phase"]): string =>
  phase === "both" ? "Both phases" : `${phase.charAt(0).toUpperCase() + phase.slice(1)} phase`

export const normalizeOverrideRules = (value: unknown): ModerationOverrideRule[] => {
  if (!Array.isArray(value)) return []
  return value
    .map((raw) => {
      if (!raw || typeof raw !== "object") return null
      const candidate = raw as Record<string, unknown>
      const id = String(candidate.id ?? "").trim()
      const pattern = String(candidate.pattern ?? "").trim()
      const isRegex = normalizeRuleIsRegex(candidate.is_regex)
      const action = String(candidate.action ?? "").trim().toLowerCase()
      const phaseRaw = String(candidate.phase ?? "both").trim().toLowerCase()
      const phase: ModerationOverrideRule["phase"] =
        phaseRaw === "input" || phaseRaw === "output" || phaseRaw === "both"
          ? phaseRaw
          : "both"
      if (!id || !pattern) return null
      if (isRegex === null) return null
      if (action !== "block" && action !== "warn") return null
      return {
        id,
        pattern,
        is_regex: isRegex,
        action,
        phase
      } as ModerationOverrideRule
    })
    .filter((rule): rule is ModerationOverrideRule => rule !== null)
}

export const sortOverrideRules = (rules: ModerationOverrideRule[]): ModerationOverrideRule[] =>
  [...rules].sort((left, right) => left.id.localeCompare(right.id))

export const areRulesEquivalent = (
  left: ModerationOverrideRule,
  right: ModerationOverrideRule
): boolean =>
  left.pattern.toLowerCase() === right.pattern.toLowerCase() &&
  left.is_regex === right.is_regex &&
  left.action === right.action &&
  left.phase === right.phase

export const createRuleId = (): string => {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID()
  }
  return `rule-${Date.now()}-${Math.random().toString(16).slice(2)}`
}

export const buildOverridePayload = (draft: ModerationUserOverride): ModerationUserOverride => {
  const payload: ModerationUserOverride = {}
  if (draft.enabled !== undefined) payload.enabled = draft.enabled
  if (draft.input_enabled !== undefined) payload.input_enabled = draft.input_enabled
  if (draft.output_enabled !== undefined) payload.output_enabled = draft.output_enabled
  if (draft.input_action) payload.input_action = draft.input_action
  if (draft.output_action) payload.output_action = draft.output_action
  if (draft.redact_replacement) payload.redact_replacement = draft.redact_replacement
  if (draft.categories_enabled !== undefined) {
    payload.categories_enabled = normalizeCategories(draft.categories_enabled)
  }
  if (draft.rules !== undefined) {
    payload.rules = normalizeOverrideRules(draft.rules)
  }
  return payload
}

export const stableSort = (items: string[]) => [...items].sort((a, b) => a.localeCompare(b))

export const normalizeSettingsDraft = (draft: SettingsDraft) => ({
  piiEnabled: Boolean(draft.piiEnabled),
  categoriesEnabled: stableSort(normalizeCategories(draft.categoriesEnabled)),
  persist: Boolean(draft.persist)
})

export const normalizeOverrideForCompare = (draft: ModerationUserOverride) => {
  const payload = buildOverridePayload(draft)
  if (payload.categories_enabled !== undefined) {
    payload.categories_enabled = stableSort(normalizeCategories(payload.categories_enabled))
  }
  if (payload.rules !== undefined) {
    payload.rules = sortOverrideRules(normalizeOverrideRules(payload.rules))
  }
  return payload
}

export const isEqualJson = (left: unknown, right: unknown) =>
  JSON.stringify(left) === JSON.stringify(right)

export const getErrorStatus = (error: unknown): number | null => {
  if (!error || typeof error !== "object") return null
  const maybeError = error as { status?: unknown; response?: { status?: unknown } }
  if (typeof maybeError.status === "number") return maybeError.status
  if (typeof maybeError.response?.status === "number") return maybeError.response.status
  return null
}

const parseLegacyRegexPattern = (pattern: string): boolean => {
  const trimmed = pattern.trim()
  return trimmed.startsWith("/") && /\/[gimsxy]*$/.test(trimmed)
}

const parseLegacyManagedLine = (line: string) => {
  const trimmed = line.trim()
  if (!trimmed) {
    return {
      rowKind: "empty" as const,
      pattern: "",
      action: "block" as const,
      replacement: null,
      categories: [] as string[]
    }
  }
  if (trimmed.startsWith("#")) {
    return {
      rowKind: "comment" as const,
      pattern: trimmed,
      action: "block" as const,
      replacement: null,
      categories: [] as string[]
    }
  }

  let ruleBody = trimmed
  let categories: string[] = []
  const categorySeparator = ruleBody.lastIndexOf(" #")
  if (categorySeparator >= 0) {
    const categoryText = ruleBody.slice(categorySeparator + 2)
    categories = normalizeCategories(categoryText)
    ruleBody = ruleBody.slice(0, categorySeparator).trim()
  }

  const actionMatch = ruleBody.match(/\s+->\s*(block|redact|warn)(?::(.+))?$/)
  const action = (actionMatch?.[1] ?? "block") as "block" | "redact" | "warn"
  const replacement = action === "redact" ? actionMatch?.[2]?.trim() || null : null
  const pattern = actionMatch
    ? ruleBody.slice(0, actionMatch.index).trim()
    : ruleBody.trim()
  const rowKind = parseLegacyRegexPattern(pattern) ? "regex" as const : "literal" as const

  return {
    rowKind,
    pattern,
    action,
    replacement,
    categories
  }
}

export const normalizeManagedBlocklistRows = (
  items: BlocklistManagedItem[]
): NormalizedManagedBlocklistRow[] =>
  (items || []).map((item) => {
    const parsed = parseLegacyManagedLine(item.line)
    const rowKind = item.pattern_type ?? parsed.rowKind
    const isValid = item.ok !== false
    const isActive = (rowKind === "literal" || rowKind === "regex") && isValid
    const statusLabel: NormalizedManagedBlocklistRow["statusLabel"] =
      !isValid
        ? "Invalid"
        : rowKind === "comment"
          ? "Comment"
          : rowKind === "empty"
            ? "Blank"
            : "Active"

    return {
      ...item,
      rowKind,
      pattern: parsed.pattern,
      action: item.action ?? parsed.action,
      replacement: item.replacement ?? parsed.replacement,
      categories: item.categories ?? parsed.categories,
      ok: item.ok ?? true,
      isValid,
      isActive,
      statusLabel
    }
  })

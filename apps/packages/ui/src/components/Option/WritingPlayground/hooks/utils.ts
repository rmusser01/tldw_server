/**
 * Standalone utility functions and types for the WritingPlayground component.
 * Extracted from index.tsx to reduce component file size.
 */

import type { JSONContent } from "@tiptap/react"
import type { ChatMessage } from "@/services/tldw/TldwApiClient"
import type {
  WritingTemplateResponse,
  WritingThemeResponse
} from "@/services/writing-playground"
import {
  parseWorldInfoKeysInput,
  type WritingAuthorNote,
  type WritingContextBlock,
  type WritingWorldInfoEntry,
  type WritingWorldInfoSettings
} from "../writing-context-utils"
import type {
  WritingRevisionAction,
  WritingRevisionOperation,
  WritingRevisionPayload,
  WritingRevisionPresetId,
  WritingRevisionProposal,
  WritingRevisionStatus
} from "../writing-revision-types"
import type { BasicStoppingModeType } from "../writing-stop-mode-utils"

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type SessionUsage = {
  name: string
  lastUsedAt: number
}

export type SessionUsageMap = Record<string, SessionUsage>

export type WritingSessionPayload = Record<string, unknown> & {
  prompt?: string
  prompt_rich?: JSONContent
  revisions?: WritingRevisionPayload
  revision_preset_id?: WritingRevisionPresetId | null
  settings?: WritingSessionSettings
  template_name?: string | null
  templateName?: string | null
  theme_name?: string | null
  themeName?: string | null
  chat_mode?: boolean
  chatMode?: boolean
}

export type PendingSave = {
  sessionId: string
  payload: WritingSessionPayload
}

export type WritingSessionSettings = {
  temperature: number
  top_p: number
  top_k: number
  token_streaming: boolean
  use_basic_stopping_mode: boolean
  basic_stopping_mode_type: BasicStoppingModeType
  logprobs: boolean
  top_logprobs: number | null
  max_tokens: number
  presence_penalty: number
  frequency_penalty: number
  seed: number | null
  stop: string[]
  advanced_extra_body: Record<string, unknown>
  memory_block: WritingContextBlock
  author_note: WritingAuthorNote
  world_info: WritingWorldInfoSettings
  context_order: string
  context_length: number
  author_note_depth_mode: "insertion" | "annotation"
}

export type EditorViewMode = "edit" | "preview" | "split"

export type GenerationMode = "append" | "predict" | "fill"

export type GenerationPlan = {
  mode: GenerationMode
  placeholder: "{predict}" | "{fill}" | null
  prefix: string
  suffix: string
}

export type GenerationHistoryEntry = {
  before: string
  after: string
}

export type LastGenerationContext = {
  prefix: string
  suffix: string
}

export type NormalizedTemplate = {
  name: string
  systemPrefix: string
  systemSuffix: string
  userPrefix: string
  userSuffix: string
  assistantPrefix: string
  assistantSuffix: string
  fimTemplate: string | null
}

export type TemplateFormState = {
  name: string
  systemPrefix: string
  systemSuffix: string
  userPrefix: string
  userSuffix: string
  assistantPrefix: string
  assistantSuffix: string
  fimTemplate: string
  isDefault: boolean
}

export type NormalizedTheme = {
  name: string
  className: string
  css: string
}

export type ThemeFormState = {
  name: string
  className: string
  css: string
  order: number
  isDefault: boolean
}

export type AdvancedNumberParamConfig = {
  key: string
  label: string
  min?: number
  max?: number
  step?: number
}

export type NonToolRole = Exclude<ChatMessage["role"], "tool">
export type NonToolMessage = Extract<ChatMessage, { role: NonToolRole }>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

export const WRITING_SPEECH_PREFS_STORAGE_KEY = "writing:speech-preferences"
export const WRITING_REVISION_PAYLOAD_SCHEMA_VERSION = 1
export const SAVE_DEBOUNCE_MS = 800
export const MAX_MATCHES = 500
export const MAX_CHUNKS = 80
export const MAX_WORDCLOUD_WORDS = 200
export const MAX_RESPONSE_LOGPROBS = 200
export const DEFAULT_TOP_LOGPROBS = 5
export const WORDCLOUD_POLL_ATTEMPTS = 20
export const WORDCLOUD_POLL_DELAY_MS = 600

export const PREDICT_PLACEHOLDER = "{predict}"
export const FILL_PLACEHOLDER = "{fill}"

const WRITING_REVISION_PRESET_IDS: readonly WritingRevisionPresetId[] = [
  "draft_freely",
  "polish_prose",
  "developmental_edit",
  "preserve_voice",
  "make_concise",
  "expand_sensory_detail"
]

const WRITING_REVISION_ACTIONS: readonly WritingRevisionAction[] = [
  "continue",
  "rewrite",
  "expand",
  "tighten",
  "tone",
  "outline",
  "custom"
]

const WRITING_REVISION_OPERATIONS: readonly WritingRevisionOperation[] = [
  "insert",
  "replace",
  "advisory"
]

const WRITING_REVISION_STATUSES: readonly WritingRevisionStatus[] = [
  "pending",
  "applied",
  "rejected",
  "conflict",
  "raw_suggestion",
  "advisory"
]

const WRITING_REVISION_TARGET_MODES = [
  "selection",
  "paragraph",
  "cursor",
  "document"
] as const

export const PREDICT_SYSTEM_PROMPT =
  "Continue the text from the prompt. Respond with only the continuation."
export const FILL_SYSTEM_PROMPT =
  "Fill in the missing text between the prefix and suffix. Respond with only the missing text."

export const DEFAULT_MEMORY_BLOCK: WritingContextBlock = {
  enabled: false,
  prefix: "Memory:\n",
  text: "",
  suffix: ""
}

export const DEFAULT_AUTHOR_NOTE: WritingAuthorNote = {
  enabled: false,
  prefix: "Author note:\n",
  text: "",
  suffix: "",
  insertion_depth: 1
}

export const DEFAULT_WORLD_INFO: WritingWorldInfoSettings = {
  enabled: false,
  search_range: 2000,
  prefix: "",
  suffix: "",
  entries: []
}

export const DEFAULT_CONTEXT_ORDER =
  "{memPrefix}{wiPrefix}{wiText}{wiSuffix}{memText}{memSuffix}{prompt}"

export const DEFAULT_SETTINGS: WritingSessionSettings = {
  temperature: 0.7,
  top_p: 0.9,
  top_k: 0,
  token_streaming: true,
  use_basic_stopping_mode: false,
  basic_stopping_mode_type: "max_tokens",
  logprobs: false,
  top_logprobs: null,
  max_tokens: 512,
  presence_penalty: 0,
  frequency_penalty: 0,
  seed: null,
  stop: [],
  advanced_extra_body: {},
  memory_block: DEFAULT_MEMORY_BLOCK,
  author_note: DEFAULT_AUTHOR_NOTE,
  world_info: DEFAULT_WORLD_INFO,
  context_order: DEFAULT_CONTEXT_ORDER,
  context_length: 8192,
  author_note_depth_mode: "insertion"
}

export const DEFAULT_TEMPLATE: NormalizedTemplate = {
  name: "default",
  systemPrefix: "",
  systemSuffix: "",
  userPrefix: "",
  userSuffix: "",
  assistantPrefix: "",
  assistantSuffix: "",
  fimTemplate: null
}

export const EMPTY_TEMPLATE_FORM: TemplateFormState = {
  name: "",
  systemPrefix: "",
  systemSuffix: "",
  userPrefix: "",
  userSuffix: "",
  assistantPrefix: "",
  assistantSuffix: "",
  fimTemplate: "",
  isDefault: false
}

export const DEFAULT_THEME: NormalizedTheme = {
  name: "default",
  className: "",
  css: ""
}

export const EMPTY_THEME_FORM: ThemeFormState = {
  name: "",
  className: "",
  css: "",
  order: 0,
  isDefault: false
}

export const ADVANCED_EXTRA_BODY_PARAM_KEYS = [
  "dynatemp_range",
  "dynatemp_exponent",
  "repeat_penalty",
  "repeat_last_n",
  "penalize_nl",
  "ignore_eos",
  "mirostat",
  "mirostat_tau",
  "mirostat_eta",
  "typical_p",
  "min_p",
  "tfs_z",
  "xtc_threshold",
  "xtc_probability",
  "dry_multiplier",
  "dry_base",
  "dry_allowed_length",
  "dry_penalty_last_n",
  "dry_sequence_breakers",
  "banned_tokens",
  "grammar",
  "logit_bias",
  "post_sampling_probs"
] as const

export const ADVANCED_NUMBER_PARAMS: AdvancedNumberParamConfig[] = [
  { key: "dynatemp_range", label: "Dynatemp range", min: 0, max: 10, step: 0.01 },
  { key: "dynatemp_exponent", label: "Dynatemp exponent", min: 0, max: 10, step: 0.01 },
  { key: "repeat_penalty", label: "Repeat penalty", min: 0, max: 3, step: 0.01 },
  { key: "repeat_last_n", label: "Repeat last N", min: -1, max: 8192, step: 1 },
  { key: "typical_p", label: "Typical P", min: 0, max: 1, step: 0.01 },
  { key: "min_p", label: "Min P", min: 0, max: 1, step: 0.01 },
  { key: "tfs_z", label: "TFS Z", min: 0, max: 2, step: 0.01 },
  { key: "mirostat", label: "Mirostat", min: 0, max: 2, step: 1 },
  { key: "mirostat_tau", label: "Mirostat tau", min: 0, max: 20, step: 0.1 },
  { key: "mirostat_eta", label: "Mirostat eta", min: 0, max: 5, step: 0.01 },
  { key: "xtc_threshold", label: "XTC threshold", min: 0, max: 1, step: 0.01 },
  { key: "xtc_probability", label: "XTC probability", min: 0, max: 1, step: 0.01 },
  { key: "dry_multiplier", label: "DRY multiplier", min: 0, max: 10, step: 0.01 },
  { key: "dry_base", label: "DRY base", min: 0, max: 10, step: 0.01 },
  { key: "dry_allowed_length", label: "DRY allowed length", min: 0, max: 4096, step: 1 },
  { key: "dry_penalty_last_n", label: "DRY penalty last N", min: 0, max: 8192, step: 1 }
]

import type { ResponseInspectorSort } from "../writing-response-inspector-utils"

export const RESPONSE_INSPECTOR_SORT_OPTIONS: Array<{
  value: ResponseInspectorSort
  label: string
}> = [
  { value: "sequence", label: "Sequence" },
  { value: "logprob_desc", label: "Logprob desc" },
  { value: "logprob_asc", label: "Logprob asc" },
  { value: "probability_desc", label: "Probability desc" },
  { value: "probability_asc", label: "Probability asc" }
]

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

export const cloneDefaultSettings = (): WritingSessionSettings => ({
  ...DEFAULT_SETTINGS,
  stop: [...DEFAULT_SETTINGS.stop],
  advanced_extra_body: {},
  memory_block: { ...DEFAULT_MEMORY_BLOCK },
  author_note: { ...DEFAULT_AUTHOR_NOTE },
  world_info: { ...DEFAULT_WORLD_INFO, entries: [] },
  context_order: DEFAULT_CONTEXT_ORDER,
  context_length: 8192,
  author_note_depth_mode: "insertion"
})

export const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value)

const isOneOf = <T extends string>(
  value: unknown,
  allowed: readonly T[]
): value is T => typeof value === "string" && allowed.includes(value as T)

const isOptionalString = (value: unknown): value is string | undefined =>
  value === undefined || typeof value === "string"

const isOptionalNullableString = (
  value: unknown
): value is string | null | undefined =>
  value == null || typeof value === "string"

const isStringArray = (value: unknown): value is string[] =>
  Array.isArray(value) && value.every((entry) => typeof entry === "string")

const isWritingRevisionPresetId = (
  value: unknown
): value is WritingRevisionPresetId =>
  isOneOf(value, WRITING_REVISION_PRESET_IDS)

const isWritingRevisionAnchor = (
  value: unknown
): value is WritingRevisionProposal["target"]["anchor"] =>
  isRecord(value) &&
  typeof value.documentFingerprint === "string" &&
  typeof value.prefix === "string" &&
  typeof value.suffix === "string"

const isWritingRevisionTarget = (
  value: unknown
): value is WritingRevisionProposal["target"] =>
  isRecord(value) &&
  isOneOf(value.mode, WRITING_REVISION_TARGET_MODES) &&
  typeof value.start === "number" &&
  Number.isFinite(value.start) &&
  Number.isInteger(value.start) &&
  value.start >= 0 &&
  typeof value.end === "number" &&
  Number.isFinite(value.end) &&
  Number.isInteger(value.end) &&
  value.end >= value.start &&
  typeof value.beforeText === "string" &&
  isWritingRevisionAnchor(value.anchor) &&
  typeof value.label === "string" &&
  typeof value.requiresConfirmation === "boolean" &&
  isOptionalString(value.confirmationReason)

const isWritingRevisionProposal = (
  value: unknown
): value is WritingRevisionProposal => {
  if (!isRecord(value)) return false
  if (typeof value.id !== "string") return false
  if (typeof value.sessionId !== "string") return false
  if (!isOneOf(value.action, WRITING_REVISION_ACTIONS)) return false
  if (!isOneOf(value.operation, WRITING_REVISION_OPERATIONS)) return false
  if (!isOptionalNullableString(value.presetInstruction)) return false
  if (value.presetId != null && !isWritingRevisionPresetId(value.presetId)) {
    return false
  }
  if (typeof value.instruction !== "string") return false
  if (!isWritingRevisionTarget(value.target)) return false
  if (!isOptionalString(value.replacementText)) return false
  if (!isOptionalString(value.rawText)) return false
  if (!isOptionalString(value.rationale)) return false
  if (!isOptionalString(value.title)) return false
  if (value.notes !== undefined && !isStringArray(value.notes)) return false
  if (!isOptionalString(value.regeneratedFromId)) return false
  if (typeof value.createdAt !== "string") return false
  return isOneOf(value.status, WRITING_REVISION_STATUSES)
}

export const buildRegex = (
  pattern: string,
  opts: { global: boolean; matchCase: boolean }
): RegExp | null => {
  try {
    const flags = `${opts.global ? "g" : ""}${opts.matchCase ? "" : "i"}`
    return new RegExp(pattern, flags)
  } catch {
    return null
  }
}

export const toNumber = (value: unknown, fallback: number): number => {
  if (typeof value === "number" && Number.isFinite(value)) return value
  if (typeof value === "string" && value.trim() !== "") {
    const parsed = Number(value)
    if (Number.isFinite(parsed)) return parsed
  }
  return fallback
}

export const toNullableNumber = (
  value: unknown,
  fallback: number | null
): number | null => {
  if (value == null || value === "") return fallback
  const parsed = toNumber(value, Number.NaN)
  return Number.isFinite(parsed) ? parsed : fallback
}

export const normalizeStopStrings = (value: unknown): string[] => {
  if (Array.isArray(value)) {
    return value.map((entry) => String(entry).trim()).filter(Boolean)
  }
  if (typeof value === "string") {
    return value
      .split(/\r?\n/)
      .map((entry) => entry.trim())
      .filter(Boolean)
  }
  return []
}

export const normalizeStringArrayValue = (value: unknown): string[] => {
  if (!Array.isArray(value)) return []
  return value.map((entry) => String(entry).trim()).filter(Boolean)
}

export const toBoolean = (value: unknown, fallback: boolean): boolean => {
  if (typeof value === "boolean") return value
  if (typeof value === "string") {
    const normalized = value.trim().toLowerCase()
    if (normalized === "true") return true
    if (normalized === "false") return false
  }
  return fallback
}

export const toStringValue = (value: unknown, fallback = ""): string =>
  typeof value === "string" ? value : fallback

export const createWorldInfoId = (): string =>
  `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`

export const normalizeWorldInfoEntries = (value: unknown): WritingWorldInfoEntry[] => {
  if (!Array.isArray(value)) return []
  return value
    .map((entry) => {
      if (!isRecord(entry)) return null
      const keys = Array.isArray(entry.keys)
        ? entry.keys.map((key) => String(key || "").trim()).filter(Boolean)
        : typeof entry.keys === "string"
          ? parseWorldInfoKeysInput(entry.keys)
          : []
      const content = toStringValue(entry.content).trim()
      if (!content) return null
      if (keys.length === 0) return null
      const entrySearchRange = toNullableNumber(
        entry.search_range ?? entry.searchRange,
        null
      )
      return {
        id: String(entry.id || createWorldInfoId()),
        display_name: toStringValue(
          entry.display_name ?? entry.displayName ?? entry.comment,
          ""
        ).trim(),
        enabled: toBoolean(entry.enabled, true),
        keys,
        content,
        use_regex: toBoolean(entry.use_regex ?? entry.useRegex, false),
        case_sensitive: toBoolean(
          entry.case_sensitive ?? entry.caseSensitive,
          false
        ),
        search_range:
          entrySearchRange == null
            ? undefined
            : Math.max(0, Math.floor(entrySearchRange))
      } as WritingWorldInfoEntry
    })
    .filter(Boolean) as WritingWorldInfoEntry[]
}

export const normalizeContextBlock = (
  raw: unknown,
  fallback: WritingContextBlock
): WritingContextBlock => {
  const value = isRecord(raw) ? raw : {}
  return {
    enabled: toBoolean(value.enabled, fallback.enabled),
    prefix: toStringValue(value.prefix, fallback.prefix),
    text: toStringValue(value.text, fallback.text),
    suffix: toStringValue(value.suffix, fallback.suffix)
  }
}

export const normalizeAuthorNote = (raw: unknown): WritingAuthorNote => {
  const value = isRecord(raw) ? raw : {}
  return {
    ...normalizeContextBlock(value, DEFAULT_AUTHOR_NOTE),
    insertion_depth: Math.max(
      1,
      Math.floor(
        toNumber(
          value.insertion_depth ?? value.insertionDepth,
          DEFAULT_AUTHOR_NOTE.insertion_depth
        )
      )
    )
  }
}

export const normalizeWorldInfoSettings = (raw: unknown): WritingWorldInfoSettings => {
  const value = isRecord(raw) ? raw : {}
  return {
    enabled: toBoolean(value.enabled, DEFAULT_WORLD_INFO.enabled),
    search_range: Math.max(
      0,
      Math.floor(toNumber(value.search_range ?? value.searchRange, 2000))
    ),
    prefix: toStringValue(value.prefix, DEFAULT_WORLD_INFO.prefix),
    suffix: toStringValue(value.suffix, DEFAULT_WORLD_INFO.suffix),
    entries: normalizeWorldInfoEntries(value.entries)
  }
}

export const pickAdvancedExtraBodyFromSettings = (
  settings: Record<string, unknown>
): Record<string, unknown> => {
  const direct = isRecord(settings.advanced_extra_body)
    ? { ...settings.advanced_extra_body }
    : isRecord(settings.extra_body)
      ? { ...settings.extra_body }
      : {}

  for (const key of ADVANCED_EXTRA_BODY_PARAM_KEYS) {
    if (!(key in direct) && key in settings) {
      direct[key] = settings[key]
    }
  }

  if (Array.isArray(direct.banned_tokens)) {
    direct.banned_tokens = normalizeStringArrayValue(direct.banned_tokens)
  }
  if (Array.isArray(direct.dry_sequence_breakers)) {
    direct.dry_sequence_breakers = normalizeStringArrayValue(
      direct.dry_sequence_breakers
    )
  }
  if (typeof direct.grammar === "string") {
    direct.grammar = direct.grammar.trim()
  }
  return direct
}

export const escapeRegex = (value: string): string =>
  value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")

export const getStringValue = (
  payload: Record<string, unknown>,
  keys: string[]
): string => {
  for (const key of keys) {
    const value = payload[key]
    if (typeof value === "string" && value.trim() !== "") {
      return value
    }
  }
  return ""
}

export const normalizeTemplatePayload = (
  template?: WritingTemplateResponse | null
): NormalizedTemplate => {
  if (!template || !isRecord(template.payload)) {
    return { ...DEFAULT_TEMPLATE }
  }
  const payload = template.payload
  const systemPrefix = getStringValue(payload, [
    "sys_pre", "sysPre", "sys_prefix", "system_prefix", "systemPrefix"
  ])
  const systemSuffix = getStringValue(payload, [
    "sys_suf", "sysSuf", "sys_suffix", "system_suffix", "systemSuffix"
  ])
  const userPrefix = getStringValue(payload, [
    "inst_pre", "instPre", "user_prefix", "userPrefix", "instruction_prefix", "instructionPrefix"
  ])
  const assistantPrefix = getStringValue(payload, [
    "inst_suf", "instSuf", "assistant_prefix", "assistantPrefix", "assistant_pre", "assistantPre"
  ])
  const userSuffix = getStringValue(payload, [
    "user_suffix", "userSuffix", "user_suf", "userSuf"
  ])
  const assistantSuffix = getStringValue(payload, [
    "assistant_suffix", "assistantSuffix", "assistant_suf", "assistantSuf"
  ])
  const fimTemplate = getStringValue(payload, [
    "fim_template", "fimTemplate", "fim"
  ])
  return {
    name: template.name,
    systemPrefix,
    systemSuffix,
    userPrefix,
    userSuffix,
    assistantPrefix,
    assistantSuffix,
    fimTemplate: fimTemplate || null
  }
}

export const applyFimTemplate = (
  template: NormalizedTemplate,
  prefix: string,
  suffix: string
): string | null => {
  if (!template.fimTemplate) return null
  return template.fimTemplate
    .replace(/\{\{?\s*prefix\s*\}?\}/gi, prefix)
    .replace(/\{\{?\s*suffix\s*\}?\}/gi, suffix)
}

export const resolveGenerationPlan = (text: string): GenerationPlan => {
  const predictIndex = text.indexOf(PREDICT_PLACEHOLDER)
  const fillIndex = text.indexOf(FILL_PLACEHOLDER)
  if (predictIndex === -1 && fillIndex === -1) {
    return {
      mode: "append",
      placeholder: null,
      prefix: text,
      suffix: ""
    }
  }
  const usePredict =
    predictIndex !== -1 && (fillIndex === -1 || predictIndex <= fillIndex)
  const placeholder = usePredict ? PREDICT_PLACEHOLDER : FILL_PLACEHOLDER
  const index = usePredict ? predictIndex : fillIndex
  const prefix = text.slice(0, index)
  const suffix = text.slice(index + placeholder.length)
  return {
    mode: usePredict ? "predict" : "fill",
    placeholder,
    prefix,
    suffix
  }
}

export const buildFillPrompt = (prefix: string, suffix: string): string => {
  return [
    "Fill in the missing text between the prefix and suffix.",
    "",
    "Prefix:",
    prefix,
    "",
    "Suffix:",
    suffix,
    "",
    "Return only the missing text."
  ].join("\n")
}

export const buildTemplateForm = (
  template?: WritingTemplateResponse | null
): TemplateFormState => {
  if (!template || !isRecord(template.payload)) {
    return { ...EMPTY_TEMPLATE_FORM }
  }
  const payload = template.payload
  return {
    name: template.name,
    systemPrefix: getStringValue(payload, [
      "sys_pre", "sysPre", "sys_prefix", "system_prefix", "systemPrefix"
    ]),
    systemSuffix: getStringValue(payload, [
      "sys_suf", "sysSuf", "sys_suffix", "system_suffix", "systemSuffix"
    ]),
    userPrefix: getStringValue(payload, [
      "inst_pre", "instPre", "user_prefix", "userPrefix", "instruction_prefix", "instructionPrefix"
    ]),
    userSuffix: getStringValue(payload, [
      "user_suffix", "userSuffix", "user_suf", "userSuf"
    ]),
    assistantPrefix: getStringValue(payload, [
      "inst_suf", "instSuf", "assistant_prefix", "assistantPrefix", "assistant_pre", "assistantPre"
    ]),
    assistantSuffix: getStringValue(payload, [
      "assistant_suffix", "assistantSuffix", "assistant_suf", "assistantSuf"
    ]),
    fimTemplate: getStringValue(payload, [
      "fim_template", "fimTemplate", "fim"
    ]),
    isDefault: template.is_default
  }
}

export const buildTemplatePayload = (
  form: TemplateFormState
): Record<string, unknown> => {
  const payload: Record<string, unknown> = {}
  if (form.systemPrefix.trim()) payload.sys_pre = form.systemPrefix
  if (form.systemSuffix.trim()) payload.sys_suf = form.systemSuffix
  if (form.userPrefix.trim()) payload.inst_pre = form.userPrefix
  if (form.userSuffix.trim()) payload.user_suffix = form.userSuffix
  if (form.assistantPrefix.trim()) payload.inst_suf = form.assistantPrefix
  if (form.assistantSuffix.trim()) payload.assistant_suf = form.assistantSuffix
  if (form.fimTemplate.trim()) payload.fim_template = form.fimTemplate
  return payload
}

export const normalizeThemeResponse = (
  theme?: WritingThemeResponse | null
): NormalizedTheme => {
  if (!theme) {
    return { ...DEFAULT_THEME }
  }
  return {
    name: theme.name,
    className: typeof theme.class_name === "string" ? theme.class_name : "",
    css: typeof theme.css === "string" ? theme.css : ""
  }
}

export const buildThemeForm = (theme?: WritingThemeResponse | null): ThemeFormState => {
  if (!theme) {
    return { ...EMPTY_THEME_FORM }
  }
  return {
    name: theme.name,
    className: typeof theme.class_name === "string" ? theme.class_name : "",
    css: typeof theme.css === "string" ? theme.css : "",
    order: Number.isFinite(theme.order) ? theme.order : 0,
    isDefault: theme.is_default
  }
}

export const buildThemePayload = (form: ThemeFormState): Record<string, unknown> => {
  const payload: Record<string, unknown> = {}
  if (form.className.trim()) payload.class_name = form.className
  if (form.css.trim()) payload.css = form.css
  if (Number.isFinite(form.order)) payload.order = form.order
  return payload
}

export const sanitizeThemeCss = (css: string): string => {
  if (!css.trim()) return ""
  let sanitized = css
  sanitized = sanitized.replace(/@import[^;]+;/gi, "")
  sanitized = sanitized.replace(/@font-face\s*{[^}]*}/gi, "")
  sanitized = sanitized.replace(/@keyframes\s+[^{]+{[\s\S]*?}\s*/gi, "")
  sanitized = sanitized.replace(/url\([^)]*\)/gi, "")
  sanitized = sanitized.replace(/(^|})\s*([^@}{][^{]*)\{/g, (match, close, selector) => {
    const scoped = selector
      .split(",")
      .map((part: string) => {
        const trimmed = part.trim()
        if (!trimmed) return trimmed
        if (trimmed.startsWith(".writing-playground")) {
          return trimmed
        }
        return `.writing-playground ${trimmed}`
      })
      .join(", ")
    return `${close}${scoped}{`
  })
  return sanitized.trim()
}

const findNextBoundary = (text: string, markers: string[]): number => {
  let earliest = -1
  for (const marker of markers) {
    if (!marker) continue
    const idx = text.indexOf(marker)
    if (idx === -1) continue
    if (earliest === -1 || idx < earliest) {
      earliest = idx
    }
  }
  return earliest
}

const buildNonToolMessage = (
  role: NonToolRole,
  content: string
): NonToolMessage => {
  if (role === "system") {
    return { role, content }
  }
  if (role === "assistant") {
    return { role, content }
  }
  return { role, content }
}

const extractMessage = (
  text: string,
  prefix: string,
  boundaries: string[],
  role: NonToolRole
): { message: NonToolMessage; remaining: string } | null => {
  if (!prefix || !text.startsWith(prefix)) return null
  const rest = text.slice(prefix.length)
  const endIndex = findNextBoundary(rest, boundaries)
  if (endIndex === -1) {
    return {
      message: buildNonToolMessage(role, rest.trim()),
      remaining: ""
    }
  }
  return {
    message: buildNonToolMessage(role, rest.slice(0, endIndex).trim()),
    remaining: rest.slice(endIndex)
  }
}

const skipToNextPrefix = (text: string, prefixes: string[]): string => {
  const nextIndex = findNextBoundary(text, prefixes)
  if (nextIndex <= 0) {
    return ""
  }
  return text.slice(nextIndex)
}

export const buildChatMessages = (
  text: string,
  template: NormalizedTemplate,
  chatMode: boolean
): ChatMessage[] => {
  const trimmed = text.trim()
  if (!trimmed) return []
  if (!chatMode) {
    return [{ role: "user", content: trimmed }]
  }
  const prefixes = [
    template.systemPrefix,
    template.userPrefix,
    template.assistantPrefix
  ].filter(Boolean)
  if (prefixes.length === 0 || !prefixes.some((p) => trimmed.includes(p))) {
    return [{ role: "user", content: trimmed }]
  }
  let remaining = trimmed
  const messages: ChatMessage[] = []
  while (remaining.length > 0) {
    const systemBoundaries = [
      template.systemSuffix,
      template.userPrefix,
      template.assistantPrefix
    ].filter(Boolean)
    const userBoundaries = [
      template.userSuffix || template.assistantPrefix,
      template.assistantPrefix,
      template.systemPrefix
    ].filter(Boolean)
    const assistantBoundaries = [
      template.assistantSuffix || template.userPrefix,
      template.userPrefix,
      template.systemPrefix
    ].filter(Boolean)

    let extracted =
      template.systemPrefix &&
      extractMessage(remaining, template.systemPrefix, systemBoundaries, "system")
    if (!extracted && template.userPrefix) {
      extracted = extractMessage(remaining, template.userPrefix, userBoundaries, "user")
    }
    if (!extracted && template.assistantPrefix) {
      extracted = extractMessage(
        remaining,
        template.assistantPrefix,
        assistantBoundaries,
        "assistant"
      )
    }
    if (!extracted) {
      remaining = skipToNextPrefix(remaining, prefixes)
      continue
    }
    if (extracted.message.content) {
      messages.push(extracted.message)
    }
    remaining = extracted.remaining.trimStart()
  }
  if (messages.length === 0) {
    return [{ role: "user", content: trimmed }]
  }
  const last = messages[messages.length - 1]
  if (last.role === "assistant" && !last.content.trim()) {
    messages.pop()
  }
  return messages
}

export const isAbortError = (error: unknown): boolean => {
  if (!error) return false
  if (error instanceof Error) {
    if (error.name === "AbortError") return true
    if (error.message.toLowerCase().includes("aborted")) return true
  }
  const cause = (error as { cause?: unknown } | null)?.cause
  if (cause instanceof Error) {
    return (
      cause.name === "AbortError" ||
      cause.message.toLowerCase().includes("aborted")
    )
  }
  return false
}

export const wait = (ms: number) =>
  new Promise<void>((resolve) => {
    window.setTimeout(resolve, ms)
  })

export const getSettingsFromPayload = (
  payload?: Record<string, unknown> | null
): WritingSessionSettings => {
  if (!isRecord(payload)) return cloneDefaultSettings()
  const raw = payload.settings
  const settings = isRecord(raw) ? raw : {}
  const advancedExtraBody = pickAdvancedExtraBodyFromSettings(settings)
  const memoryBlock = normalizeContextBlock(
    settings.memory_block ?? settings.memoryBlock,
    DEFAULT_MEMORY_BLOCK
  )
  const authorNote = normalizeAuthorNote(
    settings.author_note ?? settings.authorNote
  )
  const worldInfo = normalizeWorldInfoSettings(
    settings.world_info ?? settings.worldInfo
  )
  const rawTopLogprobs = toNullableNumber(
    settings.top_logprobs ?? settings.topLogprobs,
    DEFAULT_SETTINGS.top_logprobs
  )
  const rawBasicStoppingModeType = String(
    settings.basic_stopping_mode_type ?? settings.basicStoppingModeType ?? ""
  )
  const rawAuthorDepthMode = String(
    settings.author_note_depth_mode ?? settings.authorNoteDepthMode ?? ""
  )
  const supportedBasicStoppingModes: readonly BasicStoppingModeType[] = [
    "max_tokens",
    "new_line",
    "fill_suffix"
  ]
  return {
    temperature: toNumber(settings.temperature, DEFAULT_SETTINGS.temperature),
    top_p: toNumber(settings.top_p, DEFAULT_SETTINGS.top_p),
    top_k: toNumber(settings.top_k, DEFAULT_SETTINGS.top_k),
    token_streaming: toBoolean(
      settings.token_streaming ?? settings.tokenStreaming,
      DEFAULT_SETTINGS.token_streaming
    ),
    use_basic_stopping_mode: toBoolean(
      settings.use_basic_stopping_mode ?? settings.useBasicStoppingMode,
      DEFAULT_SETTINGS.use_basic_stopping_mode
    ),
    basic_stopping_mode_type: supportedBasicStoppingModes.includes(
      rawBasicStoppingModeType as BasicStoppingModeType
    )
      ? (rawBasicStoppingModeType as BasicStoppingModeType)
      : DEFAULT_SETTINGS.basic_stopping_mode_type,
    logprobs: toBoolean(settings.logprobs, DEFAULT_SETTINGS.logprobs),
    top_logprobs:
      rawTopLogprobs == null
        ? null
        : Math.max(1, Math.min(20, Math.floor(rawTopLogprobs))),
    max_tokens: Math.max(
      1,
      Math.round(toNumber(settings.max_tokens, DEFAULT_SETTINGS.max_tokens))
    ),
    presence_penalty: toNumber(
      settings.presence_penalty,
      DEFAULT_SETTINGS.presence_penalty
    ),
    frequency_penalty: toNumber(
      settings.frequency_penalty,
      DEFAULT_SETTINGS.frequency_penalty
    ),
    seed: toNullableNumber(settings.seed, DEFAULT_SETTINGS.seed),
    stop: normalizeStopStrings(settings.stop),
    advanced_extra_body: advancedExtraBody,
    memory_block: memoryBlock,
    author_note: authorNote,
    world_info: worldInfo,
    context_order:
      typeof (settings.context_order ?? settings.contextOrder) === "string" &&
      String(settings.context_order ?? settings.contextOrder).trim()
        ? String(settings.context_order ?? settings.contextOrder)
        : DEFAULT_CONTEXT_ORDER,
    context_length: Math.max(
      0,
      Math.floor(
        toNumber(
          settings.context_length ?? settings.contextLength,
          DEFAULT_SETTINGS.context_length
        )
      )
    ),
    author_note_depth_mode:
      rawAuthorDepthMode === "annotation" ? "annotation" : "insertion"
  }
}

export const getPromptFromPayload = (payload?: Record<string, unknown> | null): string => {
  if (!isRecord(payload)) return ""
  const prompt = payload.prompt
  return typeof prompt === "string" ? prompt : ""
}

export const getPromptRichFromPayload = (
  payload?: Record<string, unknown> | null
): JSONContent | null => {
  if (!isRecord(payload)) return null
  const promptRich = payload.prompt_rich
  return isRecord(promptRich) && promptRich.type === "doc"
    ? (promptRich as JSONContent)
    : null
}

export const getTemplateNameFromPayload = (
  payload?: Record<string, unknown> | null
): string | null => {
  if (!isRecord(payload)) return null
  const raw = payload.template_name ?? payload.templateName ?? payload.template
  if (typeof raw === "string" && raw.trim()) return raw.trim()
  return null
}

export const getThemeNameFromPayload = (
  payload?: Record<string, unknown> | null
): string | null => {
  if (!isRecord(payload)) return null
  const raw = payload.theme_name ?? payload.themeName ?? payload.theme
  if (typeof raw === "string" && raw.trim()) return raw.trim()
  return null
}

export const getChatModeFromPayload = (
  payload?: Record<string, unknown> | null
): boolean => {
  if (!isRecord(payload)) return false
  const raw = payload.chat_mode ?? payload.chatMode
  return Boolean(raw)
}

export const getRevisionsFromPayload = (
  payload?: Record<string, unknown> | null
): WritingRevisionProposal[] => {
  if (!isRecord(payload) || !isRecord(payload.revisions)) return []
  if (
    payload.revisions.schemaVersion !== WRITING_REVISION_PAYLOAD_SCHEMA_VERSION
  ) {
    return []
  }
  if (!Array.isArray(payload.revisions.items)) return []
  return payload.revisions.items.filter(isWritingRevisionProposal)
}

export const mergeRevisionsIntoPayload = (
  payload: Record<string, unknown> | null | undefined,
  revisions: WritingRevisionProposal[]
): WritingSessionPayload => {
  const base = isRecord(payload) ? payload : {}
  const next: WritingSessionPayload = { ...base }
  if (revisions.length > 0) {
    next.revisions = {
      schemaVersion: WRITING_REVISION_PAYLOAD_SCHEMA_VERSION,
      items: revisions
    }
  } else {
    delete next.revisions
  }
  return next
}

export const getRevisionPayloadSignature = (
  payload?: Record<string, unknown> | null
): string => JSON.stringify(getRevisionsFromPayload(payload))

export const getRevisionPresetIdFromPayload = (
  payload?: Record<string, unknown> | null
): WritingRevisionPresetId | null => {
  const value = isRecord(payload) ? payload.revision_preset_id : null
  return isWritingRevisionPresetId(value) ? value : null
}

export const mergePayloadIntoSession = (
  payload: Record<string, unknown> | null | undefined,
  prompt: string,
  settings: WritingSessionSettings,
  templateName: string | null,
  themeName: string | null,
  chatMode: boolean,
  options?: { promptRich?: JSONContent | null }
): WritingSessionPayload => {
  const base = isRecord(payload) ? payload : {}
  const next: WritingSessionPayload = {
    ...base,
    prompt,
    settings,
    template_name: templateName,
    theme_name: themeName,
    chat_mode: chatMode
  }
  if (options && Object.prototype.hasOwnProperty.call(options, "promptRich")) {
    if (options.promptRich) {
      next.prompt_rich = options.promptRich
    } else {
      delete next.prompt_rich
    }
  }
  return next
}

export const mergePendingPayloadIntoSession = (
  payload: Record<string, unknown> | null | undefined,
  pendingPayload: Record<string, unknown> | null | undefined,
  prompt: string,
  settings: WritingSessionSettings,
  templateName: string | null,
  themeName: string | null,
  chatMode: boolean,
  options?: { promptRich?: JSONContent | null }
): WritingSessionPayload =>
  mergePayloadIntoSession(
    isRecord(pendingPayload) ? pendingPayload : payload,
    prompt,
    settings,
    templateName,
    themeName,
    chatMode,
    options
  )

export const areSettingsEqual = (
  left: WritingSessionSettings,
  right: WritingSessionSettings
): boolean => {
  if (left.temperature !== right.temperature) return false
  if (left.top_p !== right.top_p) return false
  if (left.top_k !== right.top_k) return false
  if (left.token_streaming !== right.token_streaming) return false
  if (left.use_basic_stopping_mode !== right.use_basic_stopping_mode) return false
  if (left.basic_stopping_mode_type !== right.basic_stopping_mode_type) return false
  if (left.logprobs !== right.logprobs) return false
  if (left.top_logprobs !== right.top_logprobs) return false
  if (left.max_tokens !== right.max_tokens) return false
  if (left.presence_penalty !== right.presence_penalty) return false
  if (left.frequency_penalty !== right.frequency_penalty) return false
  if (left.seed !== right.seed) return false
  if (left.stop.length !== right.stop.length) return false
  if (!left.stop.every((value, index) => value === right.stop[index])) return false
  const leftAdvanced = JSON.stringify(left.advanced_extra_body || {})
  const rightAdvanced = JSON.stringify(right.advanced_extra_body || {})
  if (leftAdvanced !== rightAdvanced) return false
  const leftMemory = JSON.stringify(left.memory_block || {})
  const rightMemory = JSON.stringify(right.memory_block || {})
  if (leftMemory !== rightMemory) return false
  const leftAuthor = JSON.stringify(left.author_note || {})
  const rightAuthor = JSON.stringify(right.author_note || {})
  if (leftAuthor !== rightAuthor) return false
  const leftWorldInfo = JSON.stringify(left.world_info || {})
  const rightWorldInfo = JSON.stringify(right.world_info || {})
  if (leftWorldInfo !== rightWorldInfo) return false
  if (left.context_order !== right.context_order) return false
  if (left.context_length !== right.context_length) return false
  if (left.author_note_depth_mode !== right.author_note_depth_mode) return false
  return true
}

export const isVersionConflictError = (error: unknown) => {
  const status = (error as { status?: number } | null)?.status
  const msg = String((error as { message?: string } | null)?.message || "")
  const lower = msg.toLowerCase()
  return (
    status === 409 ||
    lower.includes("expected-version") ||
    lower.includes("expected_version") ||
    lower.includes("version mismatch")
  )
}

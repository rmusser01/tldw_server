import { apiSend, type ApiSendResponse } from "@/services/api-send"
import { toAllowedPath } from "@/services/tldw/path-utils"
import type { PromptImprovementLimits } from "@/services/prompts-api"

export type PromptImproveTarget = "system" | "user_message"

export type PromptImproveModelSelection = {
  selected_model: string
  provider_hint?: string | null
}

export type PromptProtectedToken = {
  kind: string
  value: string
  occurrences: number
}

export type RecognizedPromptToken = Omit<PromptProtectedToken, "occurrences">

export type PromptImproveRequest = {
  operation_id: string
  target: PromptImproveTarget
  text: string
  model_selection: PromptImproveModelSelection
  protected_tokens: PromptProtectedToken[]
}

export type PromptImproveFindingCategory =
  | "clarity"
  | "specificity"
  | "structure"
  | "constraints"
  | "output"
  | "consistency"
  | "concision"
  | "robustness"
  | "other"

export type PromptImproveFinding = {
  category: PromptImproveFindingCategory
  issue: string
  change: string
}

export type PromptResolvedModel = {
  provider: string
  model: string
  display_name: string
}

export type PromptImproveResponse = {
  schema_version: 1
  operation_id: string
  status: "improved" | "no_change"
  improved_text: string | null
  findings: PromptImproveFinding[]
  review_required: boolean
  warnings: string[]
  resolved_model: PromptResolvedModel
  meta_prompt_version: string
}

export type PromptImproveErrorCode =
  | "invalid_input"
  | "missing_model"
  | "unsupported_model"
  | "provider_not_configured"
  | "draft_too_large"
  | "provider_rate_limited"
  | "provider_timeout"
  | "provider_unavailable"
  | "model_refusal"
  | "invalid_model_output"
  | "preservation_failed"
  | "internal_error"

const ERROR_CODES = new Set<PromptImproveErrorCode>([
  "invalid_input",
  "missing_model",
  "unsupported_model",
  "provider_not_configured",
  "draft_too_large",
  "provider_rate_limited",
  "provider_timeout",
  "provider_unavailable",
  "model_refusal",
  "invalid_model_output",
  "preservation_failed",
  "internal_error"
])

const FINDING_CATEGORIES = new Set<PromptImproveFindingCategory>([
  "clarity",
  "specificity",
  "structure",
  "constraints",
  "output",
  "consistency",
  "concision",
  "robustness",
  "other"
])

const DEFAULT_LIMITS: PromptImprovementLimits = {
  max_request_bytes: 64_000,
  max_draft_chars: 24_000,
  max_candidate_chars: 24_000,
  max_raw_output_chars: 32_000,
  max_findings: 5,
  max_finding_text_chars: 500,
  max_provider_chars: 100,
  max_model_chars: 500,
  max_meta_prompt_version_chars: 100,
  max_warning_chars: 100,
  max_warnings: 16,
  max_protected_tokens: 64,
  max_protected_token_kind_chars: 50,
  max_protected_token_chars: 500,
  max_protected_token_occurrences: 100,
  max_protected_token_total_chars: 4_000
}

const TEMPLATE_VARIABLE_RE = /\{\{[A-Za-z_][A-Za-z0-9_.-]*\}\}/g
const MARKDOWN_FENCE_RE = /^ {0,3}(`{3,}|~{3,})(.*)$/
const SIMPLE_XML_TAG_RE = /<(\/)?([A-Za-z][A-Za-z0-9_.-]*)\s*>/g

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value)

const hasExactKeys = (value: Record<string, unknown>, keys: string[]): boolean => {
  const actual = Object.keys(value)
  return actual.length === keys.length && keys.every((key) => key in value)
}

const codePointLength = (value: string): number => Array.from(value).length

const countOccurrences = (text: string, value: string): number => {
  let count = 0
  let index = 0
  while ((index = text.indexOf(value, index)) !== -1) {
    count += 1
    index += value.length
  }
  return count
}

const multiset = (values: readonly string[]): Map<string, number> => {
  const counts = new Map<string, number>()
  for (const value of values) counts.set(value, (counts.get(value) ?? 0) + 1)
  return counts
}

const multisetsEqual = (
  left: Map<string, number>,
  right: Map<string, number>
): boolean =>
  left.size === right.size &&
  Array.from(left).every(([value, count]) => right.get(value) === count)

const scanMarkdownFences = (
  text: string
): { openers: string[]; balanced: boolean } => {
  const openers: string[] = []
  let openFence: { character: string; length: number } | null = null
  for (const line of text.replace(/\r\n?/g, "\n").split("\n")) {
    const match = line.match(MARKDOWN_FENCE_RE)
    if (!match) continue
    const marker = match[1]
    const rest = match[2]
    const character = marker[0]
    const length = marker.length
    if (!openFence) {
      if (character === "`" && rest.includes("`")) continue
      openFence = { character, length }
      openers.push(`${character}:${length}`)
    } else if (
      character === openFence.character &&
      length >= openFence.length &&
      !rest.trim()
    ) {
      openFence = null
    }
  }
  return { openers, balanced: openFence === null }
}

type ClassifiedTags = {
  matched: string[]
  unmatched: Map<string, number>
}

const classifySimpleTags = (text: string): ClassifiedTags => {
  const tags = Array.from(text.matchAll(SIMPLE_XML_TAG_RE), (match) => ({
    closing: Boolean(match[1]),
    name: match[2]
  }))
  const stack: Array<{ index: number; name: string }> = []
  const matchedIndices = new Set<number>()
  tags.forEach((tag, index) => {
    if (!tag.closing) {
      stack.push({ index, name: tag.name })
    } else if (stack[stack.length - 1]?.name === tag.name) {
      const opening = stack.pop() as { index: number; name: string }
      matchedIndices.add(opening.index)
      matchedIndices.add(index)
    }
  })
  const matched: string[] = []
  const unmatched = new Map<string, number>()
  tags.forEach((tag, index) => {
    const key = `${tag.closing ? "/" : ""}${tag.name}`
    if (matchedIndices.has(index)) matched.push(key)
    else unmatched.set(key, (unmatched.get(key) ?? 0) + 1)
  })
  return { matched, unmatched }
}

export const promptPreservationIsSafe = (
  original: string,
  candidate: string,
  protectedTokens: readonly PromptProtectedToken[]
): boolean => {
  if (
    !multisetsEqual(
      multiset(original.match(TEMPLATE_VARIABLE_RE) ?? []),
      multiset(candidate.match(TEMPLATE_VARIABLE_RE) ?? [])
    ) ||
    protectedTokens.some(
      (token) => countOccurrences(candidate, token.value) !== token.occurrences
    )
  ) {
    return false
  }
  const originalFences = scanMarkdownFences(original)
  const candidateFences = scanMarkdownFences(candidate)
  if (
    !candidateFences.balanced ||
    originalFences.openers.length !== candidateFences.openers.length ||
    originalFences.openers.some(
      (opener, index) => opener !== candidateFences.openers[index]
    )
  ) {
    return false
  }
  const originalTags = classifySimpleTags(original)
  const candidateTags = classifySimpleTags(candidate)
  return (
    originalTags.matched.length === candidateTags.matched.length &&
    originalTags.matched.every(
      (tag, index) => tag === candidateTags.matched[index]
    ) &&
    Array.from(candidateTags.unmatched).every(
      ([tag, count]) => count <= (originalTags.unmatched.get(tag) ?? 0)
    )
  )
}

export const collectProtectedTokens = (
  text: string,
  recognized: readonly RecognizedPromptToken[] = [],
  limits: Pick<
    PromptImprovementLimits,
    | "max_protected_tokens"
    | "max_protected_token_kind_chars"
    | "max_protected_token_chars"
    | "max_protected_token_occurrences"
    | "max_protected_token_total_chars"
  > = DEFAULT_LIMITS
): PromptProtectedToken[] => {
  const candidates: RecognizedPromptToken[] = [
    ...Array.from(text.matchAll(TEMPLATE_VARIABLE_RE), (match) => ({
      kind: "template_variable",
      value: match[0]
    })),
    ...recognized
  ]
  const result: PromptProtectedToken[] = []
  const seen = new Set<string>()
  let totalValueChars = 0

  for (const token of candidates) {
    if (result.length >= limits.max_protected_tokens) break
    if (
      typeof token?.kind !== "string" ||
      !token.kind ||
      codePointLength(token.kind) > limits.max_protected_token_kind_chars ||
      typeof token?.value !== "string" ||
      !token.value ||
      codePointLength(token.value) > limits.max_protected_token_chars
    ) {
      continue
    }
    const occurrences = countOccurrences(text, token.value)
    if (
      occurrences < 1 ||
      occurrences > limits.max_protected_token_occurrences
    ) {
      continue
    }
    const key = JSON.stringify([token.kind, token.value, occurrences])
    if (seen.has(key)) continue
    const valueChars = codePointLength(token.value)
    if (totalValueChars + valueChars > limits.max_protected_token_total_chars) {
      continue
    }
    seen.add(key)
    totalValueChars += valueChars
    result.push({ kind: token.kind, value: token.value, occurrences })
  }

  return result
}

export class PromptImprovementApiError extends Error {
  readonly code: PromptImproveErrorCode
  readonly retryable: boolean
  readonly retryAfterSeconds: number | null
  readonly requestId: string | null
  readonly status: number

  constructor(
    message: string,
    options: {
      code: PromptImproveErrorCode
      retryable: boolean
      retryAfterSeconds?: number | null
      requestId?: string | null
      status: number
    }
  ) {
    super(message)
    this.name = "PromptImprovementApiError"
    this.code = options.code
    this.retryable = options.retryable
    this.retryAfterSeconds = options.retryAfterSeconds ?? null
    this.requestId = options.requestId ?? null
    this.status = options.status
  }
}

const parseFinding = (
  value: unknown,
  limits: PromptImprovementLimits
): PromptImproveFinding | null => {
  if (!isRecord(value) || !hasExactKeys(value, ["category", "issue", "change"])) {
    return null
  }
  if (
    typeof value.category !== "string" ||
    !FINDING_CATEGORIES.has(value.category as PromptImproveFindingCategory) ||
    typeof value.issue !== "string" ||
    !value.issue ||
    codePointLength(value.issue) > limits.max_finding_text_chars ||
    typeof value.change !== "string" ||
    !value.change ||
    codePointLength(value.change) > limits.max_finding_text_chars
  ) {
    return null
  }
  return {
    category: value.category as PromptImproveFindingCategory,
    issue: value.issue,
    change: value.change
  }
}

const parseResolvedModel = (
  value: unknown,
  limits: PromptImprovementLimits
): PromptResolvedModel | null => {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, ["provider", "model", "display_name"]) ||
    typeof value.provider !== "string" ||
    !value.provider ||
    codePointLength(value.provider) > limits.max_provider_chars ||
    typeof value.model !== "string" ||
    !value.model ||
    codePointLength(value.model) > limits.max_model_chars ||
    typeof value.display_name !== "string" ||
    !value.display_name ||
    codePointLength(value.display_name) > limits.max_model_chars
  ) {
    return null
  }
  return {
    provider: value.provider,
    model: value.model,
    display_name: value.display_name
  }
}

const parsePromptImproveResponse = (
  value: unknown,
  operationId: string,
  limits: PromptImprovementLimits
): PromptImproveResponse | null => {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, [
      "schema_version",
      "operation_id",
      "status",
      "improved_text",
      "findings",
      "review_required",
      "warnings",
      "resolved_model",
      "meta_prompt_version"
    ]) ||
    value.schema_version !== 1 ||
    value.operation_id !== operationId ||
    (value.status !== "improved" && value.status !== "no_change") ||
    typeof value.review_required !== "boolean" ||
    !Array.isArray(value.findings) ||
    value.findings.length > limits.max_findings ||
    !Array.isArray(value.warnings) ||
    value.warnings.length > limits.max_warnings ||
    !value.warnings.every(
      (warning) =>
        typeof warning === "string" &&
        warning.length > 0 &&
        codePointLength(warning) <= limits.max_warning_chars
    ) ||
    typeof value.meta_prompt_version !== "string" ||
    !value.meta_prompt_version ||
    codePointLength(value.meta_prompt_version) >
      limits.max_meta_prompt_version_chars
  ) {
    return null
  }
  let improvedText: string | null
  if (value.status === "improved") {
    if (
      typeof value.improved_text !== "string" ||
      !value.improved_text.trim() ||
      codePointLength(value.improved_text) > limits.max_candidate_chars
    ) {
      return null
    }
    improvedText = value.improved_text
  } else {
    if (value.improved_text !== null) return null
    improvedText = null
  }
  const findings = value.findings.map((finding) => parseFinding(finding, limits))
  const resolvedModel = parseResolvedModel(value.resolved_model, limits)
  if (findings.some((finding) => !finding) || !resolvedModel) return null
  return {
    schema_version: 1,
    operation_id: operationId,
    status: value.status,
    improved_text: improvedText,
    findings: findings as PromptImproveFinding[],
    review_required: value.review_required,
    warnings: value.warnings.map((warning) => warning as string),
    resolved_model: resolvedModel,
    meta_prompt_version: value.meta_prompt_version
  }
}

const genericMessage = (code: PromptImproveErrorCode): string => {
  if (code === "provider_unavailable") return "The prompt improvement service is unavailable."
  if (code === "provider_timeout") return "The prompt improvement request timed out."
  if (code === "provider_rate_limited") return "The active provider is temporarily rate limited."
  if (code === "invalid_model_output") return "The model returned an invalid improvement result."
  return "Prompt improvement failed."
}

type ParsedErrorEnvelope = {
  code: PromptImproveErrorCode
  retryable: boolean
  retryAfterSeconds: number | null
  requestId: string
}

const parseErrorEnvelope = (value: unknown): ParsedErrorEnvelope | null => {
  if (!isRecord(value)) return null
  const keys = Object.keys(value)
  const hasRetryAfter = "retry_after_seconds" in value
  if (
    keys.length !== (hasRetryAfter ? 5 : 4) ||
    !["code", "message", "retryable", "request_id"].every((key) => key in value) ||
    typeof value.code !== "string" ||
    !ERROR_CODES.has(value.code as PromptImproveErrorCode) ||
    typeof value.message !== "string" ||
    !value.message ||
    codePointLength(value.message) > 300 ||
    typeof value.retryable !== "boolean" ||
    typeof value.request_id !== "string" ||
    !value.request_id ||
    codePointLength(value.request_id) > 128 ||
    (hasRetryAfter &&
      (!Number.isInteger(value.retry_after_seconds) ||
        Number(value.retry_after_seconds) < 0 ||
        Number(value.retry_after_seconds) > 86_400))
  ) {
    return null
  }
  return {
    code: value.code as PromptImproveErrorCode,
    retryable: value.retryable,
    retryAfterSeconds: hasRetryAfter ? Number(value.retry_after_seconds) : null,
    requestId: value.request_id
  }
}

const retryAfterFromResponse = (
  response: ApiSendResponse<unknown>
): number | null => {
  if (
    !Number.isInteger(response.retryAfterMs) ||
    Number(response.retryAfterMs) < 0 ||
    Number(response.retryAfterMs) > 86_400_000
  ) {
    return null
  }
  return Math.ceil(Number(response.retryAfterMs) / 1000)
}

const responseError = (response: ApiSendResponse<unknown>): PromptImprovementApiError => {
  const envelope = parseErrorEnvelope(response.data)
  const fallbackCode: PromptImproveErrorCode =
    response.status === 0
      ? "provider_unavailable"
      : response.status === 429
        ? "provider_rate_limited"
        : response.status === 408 || response.status === 504
          ? "provider_timeout"
          : response.status === 503
            ? "provider_unavailable"
            : "internal_error"
  const code = envelope?.code ?? fallbackCode

  return new PromptImprovementApiError(genericMessage(code), {
    code,
    retryable:
      envelope?.retryable ??
      (code === "provider_unavailable" ||
        code === "provider_timeout" ||
        code === "provider_rate_limited"),
    retryAfterSeconds:
      envelope?.retryAfterSeconds ?? retryAfterFromResponse(response),
    requestId: envelope?.requestId ?? null,
    status: response.status
  })
}

export async function improvePrompt(
  request: PromptImproveRequest,
  limits: PromptImprovementLimits = DEFAULT_LIMITS
): Promise<PromptImproveResponse> {
  let response: ApiSendResponse<unknown>
  try {
    response = await apiSend<unknown>({
      path: toAllowedPath("/api/v1/prompts/improve"),
      method: "POST",
      body: {
        operation_id: request.operation_id,
        target: request.target,
        text: request.text,
        model_selection:
          request.model_selection.provider_hint === undefined
            ? { selected_model: request.model_selection.selected_model }
            : {
                selected_model: request.model_selection.selected_model,
                provider_hint: request.model_selection.provider_hint
              },
        protected_tokens: request.protected_tokens.map((token) => ({
          kind: token.kind,
          value: token.value,
          occurrences: token.occurrences
        }))
      }
    })
  } catch {
    throw new PromptImprovementApiError(genericMessage("provider_unavailable"), {
      code: "provider_unavailable",
      retryable: true,
      status: 0
    })
  }
  if (!response.ok) throw responseError(response)
  const parsed = parsePromptImproveResponse(
    response.data,
    request.operation_id,
    limits
  )
  if (!parsed) {
    throw new PromptImprovementApiError(genericMessage("invalid_model_output"), {
      code: "invalid_model_output",
      retryable: false,
      status: response.status
    })
  }
  return parsed
}

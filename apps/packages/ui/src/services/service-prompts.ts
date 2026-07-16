import { createSafeStorage } from "@/utils/safe-storage"
import { buildChatSurfaceScopeKeyFromConfig } from "@/services/chat-surface-scope"
import { isHostedTldwDeployment } from "@/services/tldw/deployment-mode"
import { tldwAuth } from "@/services/tldw/TldwAuth"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { ServicePromptApiError } from "@/services/tldw/domains/service-prompts"
import type {
  KnownServicePromptId,
  ServicePromptCatalogItem,
  ServicePromptDetail,
  ServicePromptSource
} from "@/services/tldw/domains/service-prompts"
import {
  getWebSearchPrompt,
  LEGACY_SERVICE_PROMPT_DEFAULTS,
  promptForRag
} from "@/services/tldw-server"

const MAX_PART_CODE_POINTS = 20_000
const FIELD_NAME_PATTERN = /^[A-Za-z_][A-Za-z0-9_]*$/
const PYTHON_WHITESPACE_ONLY =
  /^[\u0009-\u000d\u001c-\u001f\u0020\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]*$/u

type TemplateToken =
  | { kind: "literal"; value: string }
  | { kind: "field"; name: string }

class TemplateSyntaxError extends Error {}

const tokenizeTemplate = (
  authoredText: string,
  requiredVariables: readonly string[]
): TemplateToken[] => {
  const tokens: TemplateToken[] = []
  let literal = ""
  let index = 0
  const fields: string[] = []

  const pushLiteral = () => {
    if (!literal) return
    tokens.push({ kind: "literal", value: literal })
    literal = ""
  }

  while (index < authoredText.length) {
    const character = authoredText[index]
    if (character === "{") {
      if (authoredText[index + 1] === "{") {
        literal += "{"
        index += 2
        continue
      }

      pushLiteral()
      const fieldStart = index + 1
      index = fieldStart
      while (index < authoredText.length && authoredText[index] !== "}") {
        if (authoredText[index] === "{") {
          throw new TemplateSyntaxError("Template has malformed braces.")
        }
        if (authoredText[index] === ":" || authoredText[index] === "!") {
          throw new TemplateSyntaxError(
            "Template fields cannot use conversions or format specifications."
          )
        }
        index += 1
      }
      if (index === authoredText.length) {
        throw new TemplateSyntaxError("Template has malformed braces.")
      }

      const field = authoredText.slice(fieldStart, index)
      if (!FIELD_NAME_PATTERN.test(field)) {
        throw new TemplateSyntaxError(
          "Template fields must be simple ASCII identifiers."
        )
      }
      fields.push(field)
      tokens.push({ kind: "field", name: field })
      index += 1
      continue
    }

    if (character === "}") {
      if (authoredText[index + 1] !== "}") {
        throw new TemplateSyntaxError("Template has malformed braces.")
      }
      literal += "}"
      index += 2
      continue
    }

    literal += character
    index += 1
  }
  pushLiteral()

  const actualCounts = new Map<string, number>()
  const requiredCounts = new Map<string, number>()
  for (const field of fields) {
    actualCounts.set(field, (actualCounts.get(field) ?? 0) + 1)
  }
  for (const field of requiredVariables) {
    requiredCounts.set(field, (requiredCounts.get(field) ?? 0) + 1)
  }
  if (
    actualCounts.size !== requiredCounts.size ||
    [...requiredCounts].some(([field, count]) => actualCounts.get(field) !== count)
  ) {
    throw new TemplateSyntaxError(
      "Template variables must match the registered variables exactly once."
    )
  }

  return tokens
}

export const validateServicePromptParts = (
  definition: ServicePromptCatalogItem,
  parts: Record<string, unknown>
): Record<string, string> => {
  if (!parts || typeof parts !== "object" || Array.isArray(parts)) {
    return { _parts: "Parts must be an object." }
  }

  const expectedKeys = new Set(definition.parts.map((part) => part.key))
  const providedKeys = new Set(Object.keys(parts))
  const fieldErrors: Record<string, string> = {}
  for (const part of definition.parts) {
    if (!providedKeys.has(part.key)) {
      fieldErrors[part.key] = "Part is required."
    }
  }
  if ([...providedKeys].some((key) => !expectedKeys.has(key))) {
    fieldErrors._parts = "Parts contain one or more unregistered keys."
  }
  if (Object.keys(fieldErrors).length > 0) return fieldErrors

  for (const part of definition.parts) {
    const value = parts[part.key]
    if (typeof value !== "string") {
      fieldErrors[part.key] = "Part must be a string."
      continue
    }
    if (PYTHON_WHITESPACE_ONLY.test(value)) {
      fieldErrors[part.key] = "Part must contain non-whitespace text."
      continue
    }
    if ([...value].length > MAX_PART_CODE_POINTS) {
      fieldErrors[part.key] =
        "Part must be at most 20000 Unicode code points."
      continue
    }
    if (part.mode === "template") {
      try {
        tokenizeTemplate(value, part.required_variables)
      } catch (error) {
        fieldErrors[part.key] = error instanceof TemplateSyntaxError
          ? error.message
          : "Template has malformed braces."
      }
    }
  }
  return fieldErrors
}

export const renderServicePromptPart = (
  definition: ServicePromptCatalogItem,
  partKey: string,
  authoredText: string,
  variables: Record<string, string>
): string => {
  const part = definition.parts.find((candidate) => candidate.key === partKey)
  if (!part) throw new Error("Part key is not registered.")
  if (part.mode === "literal") return authoredText

  let tokens: TemplateToken[]
  try {
    tokens = tokenizeTemplate(authoredText, part.required_variables)
  } catch (error) {
    throw new Error(
      error instanceof TemplateSyntaxError
        ? error.message
        : "Template has malformed braces."
    )
  }

  return tokens.map((token) => {
    if (token.kind === "literal") return token.value
    if (!Object.prototype.hasOwnProperty.call(variables, token.name)) {
      throw new Error("Render values are missing a required variable.")
    }
    const value = variables[token.name]
    if (typeof value !== "string") {
      throw new Error("Render values must be strings.")
    }
    return value
  }).join("")
}

type LegacyStorageKey =
  | "systemPromptForRag"
  | "questionPromptForRag"
  | "webSearchPrompt"

export type LegacyServicePromptCandidate = Readonly<{
  definitionId: Exclude<KnownServicePromptId, "media.text.translation">
  partKey: "template"
  storageKey: LegacyStorageKey
  value: string
}>

export type ServicePromptScope = Readonly<{
  config: NonNullable<Awaited<ReturnType<typeof tldwClient.getConfig>>>
  scopeKey: string
}>

export type ServicePromptSnapshot = Readonly<{
  scopeKey: string
  capability: "supported" | "legacy-404"
  definitions: Readonly<
    Partial<Record<KnownServicePromptId, Readonly<{
      parts: Readonly<Record<string, string>>
      source: ServicePromptSource
      revision: string | null
    }>>>
  >
}>

const legacyLocalStorage = createSafeStorage({ area: "local" })
const legacySyncStorage = createSafeStorage({ area: "sync" })

const LEGACY_MIGRATION_MAP = Object.freeze([
  Object.freeze({
    definitionId: "chat.rag.answer" as const,
    partKey: "template" as const,
    storageKey: "systemPromptForRag" as const,
    readSync: true
  }),
  Object.freeze({
    definitionId: "chat.rag.question_rewrite" as const,
    partKey: "template" as const,
    storageKey: "questionPromptForRag" as const,
    readSync: true
  }),
  Object.freeze({
    definitionId: "chat.web_search.answer" as const,
    partKey: "template" as const,
    storageKey: "webSearchPrompt" as const,
    readSync: false
  })
])

export const readLegacyServicePromptCandidates = async (): Promise<
  LegacyServicePromptCandidate[]
> => {
  const candidates: LegacyServicePromptCandidate[] = []
  for (const entry of LEGACY_MIGRATION_MAP) {
    const local = await legacyLocalStorage.get<unknown>(entry.storageKey)
    const raw = local === undefined && entry.readSync
      ? await legacySyncStorage.get<unknown>(entry.storageKey)
      : local
    if (typeof raw !== "string") continue
    candidates.push(Object.freeze({
      definitionId: entry.definitionId,
      partKey: entry.partKey,
      storageKey: entry.storageKey,
      value: raw
    }))
  }
  return candidates
}

export const clearLegacyServicePromptCandidate = async (
  id: KnownServicePromptId
): Promise<void> => {
  const entry = LEGACY_MIGRATION_MAP.find(
    (candidate) => candidate.definitionId === id
  )
  if (!entry) return
  await Promise.all([
    legacyLocalStorage.remove(entry.storageKey),
    legacySyncStorage.remove(entry.storageKey)
  ])
}

export const resolveServicePromptScope = async (): Promise<ServicePromptScope> => {
  const config = await tldwClient.getConfig()
  if (!config) {
    throw new Error("tldw server is not configured.")
  }

  let userId: string | number | null = null
  if (isHostedTldwDeployment() || config.authMode === "multi-user") {
    const user = await tldwAuth.getCurrentUser()
    if (user?.id === null || user?.id === undefined) {
      throw new Error("Authenticated Service Prompt scope is unresolved.")
    }
    userId = user.id
  }

  return Object.freeze({
    config,
    scopeKey: buildChatSurfaceScopeKeyFromConfig(config, { userId })
  })
}

const freezeSnapshot = (
  scopeKey: string,
  capability: ServicePromptSnapshot["capability"],
  definitions: Partial<Record<KnownServicePromptId, {
    parts: Record<string, string>
    source: ServicePromptSource
    revision: string | null
  }>>
): ServicePromptSnapshot => {
  const frozenDefinitions: Partial<Record<KnownServicePromptId, Readonly<{
    parts: Readonly<Record<string, string>>
    source: ServicePromptSource
    revision: string | null
  }>>> = {}
  for (const [id, definition] of Object.entries(definitions)) {
    frozenDefinitions[id as KnownServicePromptId] = Object.freeze({
      ...definition,
      parts: Object.freeze({ ...definition.parts })
    })
  }
  return Object.freeze({
    scopeKey,
    capability,
    definitions: Object.freeze(frozenDefinitions)
  })
}

const legacySnapshot = async (
  ids: readonly KnownServicePromptId[],
  scopeKey: string
): Promise<ServicePromptSnapshot> => {
  const requested = new Set(ids)
  const definitions: Partial<Record<KnownServicePromptId, {
    parts: Record<string, string>
    source: ServicePromptSource
    revision: null
  }>> = {}

  if (
    requested.has("chat.rag.answer") ||
    requested.has("chat.rag.question_rewrite")
  ) {
    const prompts = await promptForRag()
    if (requested.has("chat.rag.answer")) {
      definitions["chat.rag.answer"] = {
        parts: { template: prompts.ragPrompt },
        source: prompts.ragPrompt ===
          LEGACY_SERVICE_PROMPT_DEFAULTS["chat.rag.answer"].template
          ? "packaged"
          : "user",
        revision: null
      }
    }
    if (requested.has("chat.rag.question_rewrite")) {
      definitions["chat.rag.question_rewrite"] = {
        parts: { template: prompts.ragQuestionPrompt },
        source: prompts.ragQuestionPrompt ===
          LEGACY_SERVICE_PROMPT_DEFAULTS["chat.rag.question_rewrite"].template
          ? "packaged"
          : "user",
        revision: null
      }
    }
  }

  if (requested.has("chat.web_search.answer")) {
    const prompt = await getWebSearchPrompt()
    definitions["chat.web_search.answer"] = {
      parts: { template: prompt },
      source: prompt ===
        LEGACY_SERVICE_PROMPT_DEFAULTS["chat.web_search.answer"].template
        ? "packaged"
        : "user",
      revision: null
    }
  }

  return freezeSnapshot(scopeKey, "legacy-404", definitions)
}

const createInvocationSignal = (signal?: AbortSignal): {
  signal: AbortSignal
  cleanup: () => void
} => {
  const controller = new AbortController()
  const abort = () => controller.abort()
  if (signal?.aborted) {
    abort()
  } else {
    signal?.addEventListener("abort", abort, { once: true })
  }
  if (typeof window !== "undefined") {
    window.addEventListener("tldw:config-updated", abort)
  }
  return {
    signal: controller.signal,
    cleanup: () => {
      signal?.removeEventListener("abort", abort)
      if (typeof window !== "undefined") {
        window.removeEventListener("tldw:config-updated", abort)
      }
    }
  }
}

export const loadServicePromptSnapshot = async (
  ids: readonly KnownServicePromptId[],
  options: { signal?: AbortSignal } = {}
): Promise<ServicePromptSnapshot> => {
  const scope = await resolveServicePromptScope()
  const invocation = createInvocationSignal(options.signal)
  try {
    try {
      await tldwClient.listServicePrompts({ signal: invocation.signal })
    } catch (error) {
      if (error instanceof ServicePromptApiError && error.status === 404) {
        return await legacySnapshot(ids, scope.scopeKey)
      }
      throw error
    }

    const requested = [...new Set(ids)]
    const candidates = await readLegacyServicePromptCandidates()
    const unresolved = candidates.filter((candidate) =>
      requested.includes(candidate.definitionId)
    )
    if (unresolved.length > 0) {
      const error = new Error(
        "Review workflow prompts before continuing; browser-local values must be imported or discarded."
      ) as Error & { code: string; definitionIds: KnownServicePromptId[] }
      error.name = "ServicePromptMigrationRequiredError"
      error.code = "service_prompt_migration_required"
      error.definitionIds = unresolved.map((candidate) => candidate.definitionId)
      throw error
    }

    const details = await Promise.all(requested.map((id) =>
      tldwClient.getServicePrompt(id, { signal: invocation.signal })
    ))
    const definitions: Partial<Record<KnownServicePromptId, {
      parts: Record<string, string>
      source: ServicePromptSource
      revision: string | null
    }>> = {}
    for (const detail of details) {
      definitions[detail.id as KnownServicePromptId] = {
        parts: { ...detail.effective_parts },
        source: detail.source,
        revision: detail.revision
      }
    }
    return freezeSnapshot(scope.scopeKey, "supported", definitions)
  } finally {
    invocation.cleanup()
  }
}

export const importLegacyServicePromptCandidate = async (
  candidate: LegacyServicePromptCandidate,
  detail: ServicePromptDetail,
  options: { signal?: AbortSignal } = {}
): Promise<ServicePromptDetail> => {
  const saved = await tldwClient.saveServicePrompt(
    candidate.definitionId,
    {
      parts: {
        ...detail.effective_parts,
        [candidate.partKey]: candidate.value
      },
      expected_revision: detail.revision
    },
    { signal: options.signal }
  )
  await clearLegacyServicePromptCandidate(candidate.definitionId)
  return saved
}

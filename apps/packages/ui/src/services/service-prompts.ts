import { createSafeStorage } from "@/utils/safe-storage"
import { isHostedTldwDeployment } from "@/services/tldw/deployment-mode"
import { tldwAuth } from "@/services/tldw/TldwAuth"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { deriveScopedUserId } from "@/utils/media-navigation-scope"
import {
  buildChatSurfaceScopeKeyFromConfig,
  deriveSingleUserApiKeyCredentialScope
} from "@/services/chat-surface-scope"
import { ServicePromptApiError } from "@/services/tldw/domains/service-prompts"
import type {
  KnownServicePromptId,
  ServicePromptCatalogItem,
  ServicePromptDetail,
  ServicePromptRequestScope,
  ServicePromptSource
} from "@/services/tldw/domains/service-prompts"
import type { ServicePromptTargetConfig } from "@/services/tldw/TldwApiClient"
import {
  createServicePromptScopeChangedError,
  servicePromptPrincipalMatches,
  servicePromptTargetsMatch
} from "@/services/tldw/service-prompt-scope-error"
import {
  getWebSearchPrompt,
  LEGACY_SERVICE_PROMPT_DEFAULTS,
  promptForRag
} from "@/services/tldw-server"

const MAX_PART_CODE_POINTS = 20_000
const FIELD_NAME_PATTERN = /^[A-Za-z_][A-Za-z0-9_]*$/
const PYTHON_WHITESPACE_ONLY =
  // Keep browser validation aligned with Python's control-character whitespace.
  // eslint-disable-next-line no-control-regex
  /^[\u0009-\u000d\u001c-\u001f\u0020\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]*$/u

type TemplateToken =
  | { kind: "literal"; value: string }
  | { kind: "field"; name: string }

export type ServicePromptRenderDefinition = Readonly<{
  id: string
  parts: readonly Readonly<{
    key: string
    mode: "literal" | "template"
    required_variables: readonly string[]
  }>[]
}>

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
  definition: ServicePromptRenderDefinition,
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
  definition: ServicePromptRenderDefinition,
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
  config: ServicePromptTargetConfig
  scopeKey: string
  userId: string | number | null
  clientPrincipalVerified: boolean
}>

const SERVICE_PROMPT_SCOPE_UNRESOLVED = "service_prompt_scope_unresolved"

export const isServicePromptScopeUnresolvedError = (
  error: unknown
): boolean => Boolean(
  error &&
  typeof error === "object" &&
  (error as { code?: unknown }).code === SERVICE_PROMPT_SCOPE_UNRESOLVED
)

const unresolvedServicePromptScopeError = () => Object.assign(
  new Error("Authenticated Service Prompt scope is unresolved."),
  { code: SERVICE_PROMPT_SCOPE_UNRESOLVED }
)

export type ServicePromptSnapshot = Readonly<{
  scopeKey: string
  requestScope: ServicePromptRequestScope
  capability: "supported" | "legacy-404"
  definitions: Readonly<
    Partial<Record<KnownServicePromptId, Readonly<{
      definition: ServicePromptRenderDefinition
      parts: Readonly<Record<string, string>>
      source: ServicePromptSource
      revision: string | null
    }>>>
  >
  scopeSignal: AbortSignal
  scopeInvalidatedSignal: AbortSignal
  release: () => void
}>

type SnapshotDefinition = {
  definition: ServicePromptRenderDefinition
  parts: Record<string, string>
  source: ServicePromptSource
  revision: string | null
}

const freezeRenderDefinition = (
  definition: ServicePromptRenderDefinition
): ServicePromptRenderDefinition => Object.freeze({
  id: definition.id,
  parts: Object.freeze(definition.parts.map((part) => Object.freeze({
    key: part.key,
    mode: part.mode,
    required_variables: Object.freeze([...part.required_variables])
  })))
})

const LEGACY_RENDER_DEFINITIONS = Object.freeze({
  "chat.rag.answer": freezeRenderDefinition({
    id: "chat.rag.answer",
    parts: [{
      key: "template",
      mode: "template",
      required_variables: ["context", "question"]
    }]
  }),
  "chat.rag.question_rewrite": freezeRenderDefinition({
    id: "chat.rag.question_rewrite",
    parts: [{
      key: "template",
      mode: "template",
      required_variables: ["chat_history", "question"]
    }]
  }),
  "chat.web_search.answer": freezeRenderDefinition({
    id: "chat.web_search.answer",
    parts: [{
      key: "template",
      mode: "template",
      required_variables: ["current_date_time", "search_results"]
    }]
  }),
  "chat.title.generation": freezeRenderDefinition({
    id: "chat.title.generation",
    parts: [{
      key: "user_template",
      mode: "template",
      required_variables: ["query"]
    }]
  }),
  "image.prompt.refinement": freezeRenderDefinition({
    id: "image.prompt.refinement",
    parts: [
      {
        key: "system_semantics",
        mode: "literal",
        required_variables: []
      },
      {
        key: "rewrite_semantics",
        mode: "literal",
        required_variables: []
      }
    ]
  })
})

const legacyLocalStorage = createSafeStorage({ area: "local" })
const legacySyncStorage = createSafeStorage({ area: "sync" })

export const subscribeToServicePromptConfigChanges = (
  listener: () => void
): (() => void) => {
  const watchers = { tldwConfig: listener }
  legacyLocalStorage.watch(watchers)
  return () => legacyLocalStorage.unwatch(watchers)
}

type SignalOptions = { signal?: AbortSignal }

const throwAbortError = (): never => {
  const error = new Error("Service Prompt request was aborted.")
  error.name = "AbortError"
  throw error
}

const throwIfAborted = (signal?: AbortSignal): void => {
  if (signal?.aborted) throwAbortError()
}

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

export const readLegacyServicePromptCandidates = async (
  options: SignalOptions = {}
): Promise<
  LegacyServicePromptCandidate[]
> => {
  const candidates: LegacyServicePromptCandidate[] = []
  throwIfAborted(options.signal)
  for (const entry of LEGACY_MIGRATION_MAP) {
    let raw = await legacyLocalStorage.get<unknown>(entry.storageKey)
    throwIfAborted(options.signal)
    if (raw === undefined && entry.readSync) {
      raw = await legacySyncStorage.get<unknown>(entry.storageKey)
      throwIfAborted(options.signal)
    }
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

export const resolveServicePromptScope = async (
  options: SignalOptions = {}
): Promise<ServicePromptScope> => {
  throwIfAborted(options.signal)
  await tldwClient.initialize()
  throwIfAborted(options.signal)
  let initialConfig
  try {
    initialConfig = await tldwClient.ensureConfigForRequest(true)
  } catch (error) {
    throwIfAborted(options.signal)
    const storedConfig = await tldwClient
      .ensureConfigForRequest(false)
      .catch(() => null)
    throwIfAborted(options.signal)
    if (
      !isHostedTldwDeployment() &&
      storedConfig?.authMode === "multi-user" &&
      storedConfig.authSource !== "cookie-session" &&
      !String(storedConfig.accessToken || "").trim() &&
      !String(storedConfig.refreshToken || "").trim()
    ) {
      throw unresolvedServicePromptScopeError()
    }
    throw error
  }
  throwIfAborted(options.signal)
  if (!initialConfig) {
    throw new Error("tldw server is not configured.")
  }

  let userId: string | number | null = null
  let resolvedConfig = initialConfig
  if (isHostedTldwDeployment() || initialConfig.authMode === "multi-user") {
    const user = await tldwAuth.getCurrentUser().catch((error) => {
      if (error && typeof error === "object" &&
        (error as { status?: unknown }).status === 401) {
        throw unresolvedServicePromptScopeError()
      }
      throw error
    })
    throwIfAborted(options.signal)
    if (user?.id === null || user?.id === undefined) {
      throw unresolvedServicePromptScopeError()
    }
    await tldwClient.initialize()
    throwIfAborted(options.signal)
    const refreshedConfig = await tldwClient.ensureConfigForRequest(true)
    throwIfAborted(options.signal)
    if (!refreshedConfig) {
      throw new Error("tldw server is not configured.")
    }
    if (!servicePromptTargetsMatch(initialConfig, refreshedConfig)) {
      throw new Error("Authenticated Service Prompt scope changed while resolving.")
    }
    resolvedConfig = refreshedConfig
    userId = user.id
  }

  const singleUserApiKeyScope = deriveSingleUserApiKeyCredentialScope(
    resolvedConfig.authMode,
    resolvedConfig.apiKey
  )
  const config = Object.freeze({
    serverUrl: resolvedConfig.serverUrl,
    authMode: resolvedConfig.authMode,
    authSource: resolvedConfig.authSource,
    orgId: resolvedConfig.orgId,
    ...(singleUserApiKeyScope
      ? { expectedSingleUserApiKeyScope: singleUserApiKeyScope }
      : {})
  })

  return Object.freeze({
    config,
    scopeKey: buildChatSurfaceScopeKeyFromConfig(resolvedConfig, { userId }),
    userId,
    clientPrincipalVerified:
      userId === null || servicePromptPrincipalMatches(resolvedConfig, userId)
  })
}

type ServicePromptScopeLease = Readonly<{
  signal: AbortSignal
  scopeInvalidatedSignal: AbortSignal
  bind: (scope: ServicePromptScope) => void
  release: () => void
}>

const storedConfigMatchesScope = (
  value: unknown,
  scope: ServicePromptScope
): boolean => {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false
  const config = value as Record<string, unknown>
  if (!servicePromptTargetsMatch(config, scope.config)) {
    return false
  }
  if (scope.config.authMode === "single-user") {
    return deriveSingleUserApiKeyCredentialScope(
      typeof config.authMode === "string" ? config.authMode : null,
      typeof config.apiKey === "string" ? config.apiKey : null
    ) === scope.config.expectedSingleUserApiKeyScope
  }
  if (scope.userId === null) return true
  const accessToken = typeof config.accessToken === "string"
    ? config.accessToken
    : null
  const currentUser = deriveScopedUserId({
    userId: null,
    authMode: typeof config.authMode === "string" ? config.authMode : null,
    accessToken
  })
  const expectedUser = deriveScopedUserId({
    userId: scope.userId,
    authMode: scope.config.authMode,
    accessToken: null
  })
  return currentUser === expectedUser
}

const createServicePromptScopeLease = (
  parentSignal?: AbortSignal
): ServicePromptScopeLease => {
  const controller = new AbortController()
  const scopeController = new AbortController()
  let active = true
  let bound = false
  let scope: ServicePromptScope | null = null
  const storageWatch = {
    tldwConfig: (change: { newValue?: unknown }) => {
      if (scope && !storedConfigMatchesScope(change?.newValue, scope)) {
        invalidateScope()
      }
    }
  }
  const release = () => {
    if (!active) return
    active = false
    parentSignal?.removeEventListener("abort", abortRequest)
    if (bound) legacyLocalStorage.unwatch(storageWatch)
    if (typeof window !== "undefined") {
      window.removeEventListener(
        "tldw:auth-credentials-changed",
        invalidateScope
      )
    }
  }
  const abortRequest = () => {
    if (!controller.signal.aborted) controller.abort()
  }
  const invalidateScope = () => {
    if (!scopeController.signal.aborted) scopeController.abort()
    abortRequest()
  }

  if (parentSignal?.aborted) {
    abortRequest()
  } else {
    parentSignal?.addEventListener("abort", abortRequest, { once: true })
    if (typeof window !== "undefined") {
      window.addEventListener(
        "tldw:auth-credentials-changed",
        invalidateScope
      )
    }
  }

  return Object.freeze({
    signal: controller.signal,
    scopeInvalidatedSignal: scopeController.signal,
    bind: (resolvedScope: ServicePromptScope) => {
      if (!active || bound) return
      scope = resolvedScope
      bound = true
      legacyLocalStorage.watch(storageWatch)
    },
    release
  })
}

const freezeSnapshot = (
  scope: ServicePromptScope,
  capability: ServicePromptSnapshot["capability"],
  definitions: Partial<Record<KnownServicePromptId, SnapshotDefinition>>,
  lease: Pick<
    ServicePromptScopeLease,
    "signal" | "scopeInvalidatedSignal" | "release"
  >
): ServicePromptSnapshot => {
  const frozenDefinitions: Partial<Record<KnownServicePromptId, Readonly<{
    definition: ServicePromptRenderDefinition
    parts: Readonly<Record<string, string>>
    source: ServicePromptSource
    revision: string | null
  }>>> = {}
  for (const [id, definition] of Object.entries(definitions)) {
    frozenDefinitions[id as KnownServicePromptId] = Object.freeze({
      ...definition,
      definition: freezeRenderDefinition(definition.definition),
      parts: Object.freeze({ ...definition.parts })
    })
  }
  return Object.freeze({
    scopeKey: scope.scopeKey,
    requestScope: Object.freeze({
      config: scope.config,
      userId: scope.userId
    }),
    capability,
    definitions: Object.freeze(frozenDefinitions),
    scopeSignal: lease.signal,
    scopeInvalidatedSignal: lease.scopeInvalidatedSignal,
    release: lease.release
  })
}

const legacySnapshot = async (
  ids: readonly KnownServicePromptId[],
  scope: ServicePromptScope,
  lease: Pick<
    ServicePromptScopeLease,
    "signal" | "scopeInvalidatedSignal" | "release"
  >
): Promise<ServicePromptSnapshot> => {
  throwIfAborted(lease.signal)
  const requested = new Set(ids)
  const definitions: Partial<Record<KnownServicePromptId, SnapshotDefinition>> = {}

  if (
    requested.has("chat.rag.answer") ||
    requested.has("chat.rag.question_rewrite")
  ) {
    const prompts = await promptForRag()
    throwIfAborted(lease.signal)
    if (requested.has("chat.rag.answer")) {
      definitions["chat.rag.answer"] = {
        definition: LEGACY_RENDER_DEFINITIONS["chat.rag.answer"],
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
        definition: LEGACY_RENDER_DEFINITIONS["chat.rag.question_rewrite"],
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
    throwIfAborted(lease.signal)
    definitions["chat.web_search.answer"] = {
      definition: LEGACY_RENDER_DEFINITIONS["chat.web_search.answer"],
      parts: { template: prompt },
      source: prompt ===
        LEGACY_SERVICE_PROMPT_DEFAULTS["chat.web_search.answer"].template
        ? "packaged"
        : "user",
      revision: null
    }
  }

  if (requested.has("chat.title.generation")) {
    definitions["chat.title.generation"] = {
      definition: LEGACY_RENDER_DEFINITIONS["chat.title.generation"],
      parts: {
        user_template:
          LEGACY_SERVICE_PROMPT_DEFAULTS["chat.title.generation"].user_template
      },
      source: "packaged",
      revision: null
    }
  }

  if (requested.has("image.prompt.refinement")) {
    definitions["image.prompt.refinement"] = {
      definition: LEGACY_RENDER_DEFINITIONS["image.prompt.refinement"],
      parts: {
        system_semantics:
          LEGACY_SERVICE_PROMPT_DEFAULTS["image.prompt.refinement"]
            .system_semantics,
        rewrite_semantics:
          LEGACY_SERVICE_PROMPT_DEFAULTS["image.prompt.refinement"]
            .rewrite_semantics
      },
      source: "packaged",
      revision: null
    }
  }

  throwIfAborted(lease.signal)
  return freezeSnapshot(scope, "legacy-404", definitions, lease)
}

export const loadServicePromptSnapshot = async (
  ids: readonly KnownServicePromptId[],
  options: {
    signal?: AbortSignal
    requestScope?: ServicePromptRequestScope
  } = {}
): Promise<ServicePromptSnapshot> => {
  const lease = createServicePromptScopeLease(options.signal)
  try {
    const requested = [...new Set(ids)]
    throwIfAborted(lease.signal)
    const scope = await resolveServicePromptScope({ signal: lease.signal })
    const expectedRequestScope = options.requestScope
    if (expectedRequestScope) {
      const expectedMatchesResolved =
        servicePromptTargetsMatch(expectedRequestScope.config, scope.config) &&
        (expectedRequestScope.config.expectedSingleUserApiKeyScope ?? null) ===
          (scope.config.expectedSingleUserApiKeyScope ?? null) &&
        (expectedRequestScope.userId === null
          ? scope.userId === null
          : String(expectedRequestScope.userId) === String(scope.userId))

      if (!expectedMatchesResolved) {
        throw createServicePromptScopeChangedError()
      }
    }
    lease.bind(scope)
    throwIfAborted(lease.signal)
    let catalog: ServicePromptCatalogItem[]
    try {
      catalog = await tldwClient.listServicePrompts({
        signal: lease.signal,
        requestScope: scope
      })
      throwIfAborted(lease.signal)
    } catch (error) {
      if (error instanceof ServicePromptApiError && error.status === 404) {
        if (!scope.clientPrincipalVerified) {
          throw createServicePromptScopeChangedError()
        }
        return await legacySnapshot(
          ids,
          scope,
          lease
        )
      }
      throw error
    }

    const advertisedIds = new Set(catalog.map((definition) => definition.id))
    const catalogOmitsImageRefinement =
      requested.includes("image.prompt.refinement") &&
      !advertisedIds.has("image.prompt.refinement")
    const requestedFromServer = requested.filter(
      (id) => id !== "image.prompt.refinement" || !catalogOmitsImageRefinement
    )
    const candidates = requestedFromServer.length > 0
      ? await readLegacyServicePromptCandidates({ signal: lease.signal })
      : []
    throwIfAborted(lease.signal)
    const unresolved = candidates.filter((candidate) =>
      requestedFromServer.includes(candidate.definitionId)
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

    const details = await Promise.all(requestedFromServer.map(async (id) => {
      try {
        return await tldwClient.getServicePrompt(id, {
          signal: lease.signal,
          requestScope: scope
        })
      } catch (error) {
        if (
          id === "image.prompt.refinement" &&
          error instanceof ServicePromptApiError &&
          error.status === 404
        ) {
          return null
        }
        throw error
      }
    }))
    throwIfAborted(lease.signal)
    const usePackagedImageRefinement =
      catalogOmitsImageRefinement || details.some((detail) => detail === null)
    const definitions: Partial<Record<KnownServicePromptId, SnapshotDefinition>> = {}
    for (const detail of details) {
      if (!detail) continue
      definitions[detail.id as KnownServicePromptId] = {
        definition: detail,
        parts: { ...detail.effective_parts },
        source: detail.source,
        revision: detail.revision
      }
    }
    if (usePackagedImageRefinement) {
      definitions["image.prompt.refinement"] = {
        definition: LEGACY_RENDER_DEFINITIONS["image.prompt.refinement"],
        parts: {
          system_semantics:
            LEGACY_SERVICE_PROMPT_DEFAULTS["image.prompt.refinement"]
              .system_semantics,
          rewrite_semantics:
            LEGACY_SERVICE_PROMPT_DEFAULTS["image.prompt.refinement"]
              .rewrite_semantics
        },
        source: "packaged",
        revision: null
      }
    }
    return freezeSnapshot(scope, "supported", definitions, lease)
  } catch (error) {
    const scopeInvalidated = lease.scopeInvalidatedSignal.aborted
    lease.release()
    if (scopeInvalidated) {
      throw createServicePromptScopeChangedError()
    }
    throw error
  }
}

export const importLegacyServicePromptCandidate = async (
  candidate: LegacyServicePromptCandidate,
  detail: ServicePromptDetail,
  options: {
    signal?: AbortSignal
    requestScope?: ServicePromptRequestScope
  } = {}
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
    {
      signal: options.signal,
      requestScope: options.requestScope
    }
  )
  throwIfAborted(options.signal)
  await clearLegacyServicePromptCandidate(candidate.definitionId)
  return saved
}

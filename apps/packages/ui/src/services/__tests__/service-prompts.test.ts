import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  localGet: vi.fn(),
  syncGet: vi.fn(),
  localRemove: vi.fn(),
  syncRemove: vi.fn(),
  localWatch: vi.fn(),
  localUnwatch: vi.fn(),
  initialize: vi.fn(),
  ensureConfig: vi.fn(),
  getCurrentUser: vi.fn(),
  listServicePrompts: vi.fn(),
  getServicePrompt: vi.fn(),
  saveServicePrompt: vi.fn(),
  promptForRag: vi.fn(),
  getWebSearchPrompt: vi.fn(),
  isHosted: vi.fn(),
  bgRequest: vi.fn()
}))

vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: ({ area }: { area?: string } = {}) => ({
    get: (...args: unknown[]) =>
      (area === "sync" ? mocks.syncGet : mocks.localGet)(...args),
    remove: (...args: unknown[]) =>
      (area === "sync" ? mocks.syncRemove : mocks.localRemove)(...args),
    watch: (...args: unknown[]) => mocks.localWatch(...args),
    unwatch: (...args: unknown[]) => mocks.localUnwatch(...args)
  })
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.bgRequest(...args)
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: (...args: unknown[]) => mocks.initialize(...args),
    ensureConfigForRequest: (...args: unknown[]) => mocks.ensureConfig(...args),
    listServicePrompts: (...args: unknown[]) => mocks.listServicePrompts(...args),
    getServicePrompt: (...args: unknown[]) => mocks.getServicePrompt(...args),
    saveServicePrompt: (...args: unknown[]) => mocks.saveServicePrompt(...args)
  }
}))

vi.mock("@/services/tldw/TldwAuth", () => ({
  tldwAuth: {
    getCurrentUser: (...args: unknown[]) => mocks.getCurrentUser(...args)
  }
}))

vi.mock("@/services/tldw/deployment-mode", () => ({
  isHostedTldwDeployment: () => mocks.isHosted()
}))

vi.mock("@/services/tldw-server", async () => {
  const actual = await vi.importActual<typeof import("@/services/tldw-server")>(
    "@/services/tldw-server"
  )
  return {
    ...actual,
    promptForRag: (...args: unknown[]) => mocks.promptForRag(...args),
    getWebSearchPrompt: (...args: unknown[]) => mocks.getWebSearchPrompt(...args)
  }
})

import fixture from "@/utils/__fixtures__/service-prompt-rendering.json"
import {
  clearLegacyServicePromptCandidate,
  importLegacyServicePromptCandidate,
  loadServicePromptSnapshot,
  readLegacyServicePromptCandidates,
  renderServicePromptPart,
  resolveServicePromptScope,
  validateServicePromptParts
} from "@/services/service-prompts"
import { LEGACY_SERVICE_PROMPT_DEFAULTS } from "@/services/tldw-server"
import type {
  KnownServicePromptId,
  ServicePromptCatalogItem,
  ServicePromptDetail
} from "@/services/tldw/domains/service-prompts"
import { ServicePromptApiError } from "@/services/tldw/domains/service-prompts"
import { servicePromptMethods } from "@/services/tldw/domains/service-prompts"
import {
  buildChatSurfaceScopeKeyFromConfig,
  deriveSingleUserApiKeyCredentialScope
} from "@/services/chat-surface-scope"

const definition = (
  id: KnownServicePromptId,
  parts: ServicePromptCatalogItem["parts"]
): ServicePromptCatalogItem => ({
  id,
  label: id,
  description: id,
  parts,
  affected_workflows: []
})

const definitions: Partial<Record<KnownServicePromptId, ServicePromptCatalogItem>> = {
  "writing.agent.quick": definition("writing.agent.quick", [{ key: "system", label: "System instructions", mode: "literal", required_variables: [] }]),
  "writing.agent.planning": definition("writing.agent.planning", [{ key: "system", label: "System instructions", mode: "literal", required_variables: [] }]),
  "writing.agent.brainstorm": definition("writing.agent.brainstorm", [{ key: "system", label: "System instructions", mode: "literal", required_variables: [] }]),
  "chat.rag.answer": definition("chat.rag.answer", [
    {
      key: "template",
      label: "Template",
      mode: "template",
      required_variables: ["context", "question"]
    }
  ]),
  "chat.rag.question_rewrite": definition("chat.rag.question_rewrite", [
    {
      key: "template",
      label: "Template",
      mode: "template",
      required_variables: ["chat_history", "question"]
    }
  ]),
  "chat.web_search.answer": definition("chat.web_search.answer", [
    {
      key: "template",
      label: "Template",
      mode: "template",
      required_variables: ["current_date_time", "search_results"]
    }
  ]),
  "chat.title.generation": definition("chat.title.generation", [{
    key: "user_template",
    label: "User template",
    mode: "template",
    required_variables: ["query"]
  }]),
  "image.prompt.refinement": definition("image.prompt.refinement", [
    {
      key: "system_semantics",
      label: "Refinement guidance",
      mode: "literal",
      required_variables: []
    },
    {
      key: "rewrite_semantics",
      label: "Rewrite guidance",
      mode: "literal",
      required_variables: []
    }
  ]),
  "media.text.translation": definition("media.text.translation", [
    {
      key: "system",
      label: "System instructions",
      mode: "literal",
      required_variables: []
    },
    {
      key: "user_template",
      label: "User template",
      mode: "template",
      required_variables: ["target_language", "text"]
    }
  ]),
  "notes.title.generate": definition("notes.title.generate", [
    {
      key: "system",
      label: "System instructions",
      mode: "literal",
      required_variables: []
    },
    {
      key: "title_instruction",
      label: "Title instruction",
      mode: "literal",
      required_variables: []
    }
  ])
}

describe("Service Prompt validation and rendering", () => {
  it("keeps old-server defaults byte-equivalent to the shared fixture", () => {
    expect(LEGACY_SERVICE_PROMPT_DEFAULTS).toEqual({
      "writing.agent.quick": fixture.defaults["writing.agent.quick"],
      "writing.agent.planning": fixture.defaults["writing.agent.planning"],
      "writing.agent.brainstorm": fixture.defaults["writing.agent.brainstorm"],
      "chat.rag.answer": fixture.defaults["chat.rag.answer"],
      "chat.rag.question_rewrite": fixture.defaults["chat.rag.question_rewrite"],
      "chat.web_search.answer": fixture.defaults["chat.web_search.answer"],
      "chat.title.generation": fixture.defaults["chat.title.generation"],
      "image.prompt.refinement": fixture.defaults["image.prompt.refinement"]
    })
  })

  it("requires exactly the registered parts", () => {
    expect(validateServicePromptParts(definitions["chat.rag.answer"], {})).toEqual({
      template: "Part is required."
    })
    expect(
      validateServicePromptParts(definitions["chat.rag.answer"], {
        template: "{context} {question}",
        extra: "not registered"
      })
    ).toEqual({
      _parts: "Parts contain one or more unregistered keys."
    })
  })

  it.each([
    [null, "Part must be a string."],
    [42, "Part must be a string."],
    ["\t\n", "Part must contain non-whitespace text."]
  ])("rejects invalid part value %j", (value, message) => {
    expect(
      validateServicePromptParts(definitions["chat.rag.answer"], {
        template: value
      })
    ).toEqual({ template: message })
  })

  it("counts Unicode code points instead of UTF-16 code units", () => {
    const exactly = "{context}{question}" + "😀".repeat(19_981)
    const tooLong = exactly + "😀"

    expect([...exactly]).toHaveLength(20_000)
    expect(validateServicePromptParts(definitions["chat.rag.answer"], {
      template: exactly
    })).toEqual({})
    expect(validateServicePromptParts(definitions["chat.rag.answer"], {
      template: tooLong
    })).toEqual({
      template: "Part must be at most 20000 Unicode code points."
    })
  })

  it("matches Python strip whitespace semantics at the Unicode edges", () => {
    const translation = definitions["media.text.translation"]
    const userTemplate = "Translate to {target_language}: {text}"

    expect(validateServicePromptParts(translation, {
      system: "\u001c",
      user_template: userTemplate
    })).toEqual({ system: "Part must contain non-whitespace text." })
    expect(validateServicePromptParts(translation, {
      system: "\ufeff",
      user_template: userTemplate
    })).toEqual({})
  })

  it.each([
    ["{context.value} {question}", "Template fields must be simple ASCII identifiers."],
    ["{context[0]} {question}", "Template fields must be simple ASCII identifiers."],
    ["{0} {question}", "Template fields must be simple ASCII identifiers."],
    ["{context!r} {question}", "Template fields cannot use conversions or format specifications."],
    ["{context:>20} {question}", "Template fields cannot use conversions or format specifications."],
    ["{context} {question:}", "Template fields cannot use conversions or format specifications."],
    ["{context} {question", "Template has malformed braces."],
    ["{context} } {question}", "Template has malformed braces."],
    ["{context{value}} {question}", "Template has malformed braces."],
    ["{} {context} {question}", "Template fields must be simple ASCII identifiers."],
    ["{context} {other}", "Template variables must match the registered variables exactly once."],
    ["{context}", "Template variables must match the registered variables exactly once."],
    ["{context} {question} {question}", "Template variables must match the registered variables exactly once."]
  ])("rejects invalid template %s", (template, message) => {
    expect(validateServicePromptParts(definitions["chat.rag.answer"], {
      template
    })).toEqual({ template: message })
  })

  it("does not parse braces in literal parts", () => {
    const text = "Literal {unmatched and }} braces stay unchanged"
    const parts = {
      system: text,
      user_template: "Translate to {target_language}: {text}"
    }

    expect(validateServicePromptParts(definitions["media.text.translation"], parts))
      .toEqual({})
    expect(renderServicePromptPart(
      definitions["media.text.translation"],
      "system",
      text,
      {}
    )).toBe(text)
  })

  it.each(fixture.render_cases)("renders $name in one pass", (renderCase) => {
    const serviceDefinition = definitions[
      renderCase.definition_id as KnownServicePromptId
    ]

    expect(renderServicePromptPart(
      serviceDefinition,
      renderCase.part_key,
      renderCase.authored_text,
      renderCase.values
    )).toBe(renderCase.expected)
  })

  it("rejects unknown parts and missing or non-string render values", () => {
    const rag = definitions["chat.rag.answer"]

    expect(() => renderServicePromptPart(
      rag,
      "unknown",
      "secret authored text",
      {}
    )).toThrow("Part key is not registered.")
    expect(() => renderServicePromptPart(
      rag,
      "template",
      "{context} {question}",
      { context: "context" }
    )).toThrow("Render values are missing a required variable.")
    expect(() => renderServicePromptPart(
      rag,
      "template",
      "{context} {question}",
      { context: "context", question: 42 as unknown as string }
    )).toThrow("Render values must be strings.")
  })
})

const config = {
  serverUrl: "https://server.example",
  authMode: "single-user" as const,
  orgId: 7,
  apiKey: "test-key"
}

const jwtForUser = (userId: string | number): string =>
  `header.${btoa(JSON.stringify({ sub: String(userId) }))}.signature`

const targetConfig = (value: {
  serverUrl: string
  authMode: "single-user" | "multi-user"
  orgId?: number
  authSource?: "manual" | "cookie-session"
}) => ({
  serverUrl: value.serverUrl,
  authMode: value.authMode,
  authSource: value.authSource,
  orgId: value.orgId
})

const scopeKeyFor = (
  value: NonNullable<Parameters<typeof buildChatSurfaceScopeKeyFromConfig>[0]>,
  userId: string | number | null
) => buildChatSurfaceScopeKeyFromConfig(value, { userId })

const singleUserApiKeyScopeFor = (
  value: NonNullable<Parameters<typeof buildChatSurfaceScopeKeyFromConfig>[0]>
): string => {
  const scope = deriveSingleUserApiKeyCredentialScope(
    value.authMode,
    value.apiKey
  )
  if (!scope) throw new Error("Single-user config is missing its API-key scope.")
  return scope
}

const VALID_REVISION = "123e4567-e89b-42d3-a456-426614174000"

const detailFor = (
  id: KnownServicePromptId,
  overrides: Partial<ServicePromptDetail> = {}
): ServicePromptDetail => ({
  ...definitions[id],
  default_parts: fixture.defaults[id],
  saved_parts: null,
  effective_parts: fixture.defaults[id],
  source: "packaged",
  revision: null,
  ...overrides
})

const catalog = Object.values(definitions)

const renderDefinitionFor = (id: KnownServicePromptId) => ({
  id,
  parts: definitions[id].parts.map((part) => ({
    key: part.key,
    mode: part.mode,
    required_variables: [...part.required_variables]
  }))
})

describe("Service Prompt migration and runtime snapshots", () => {
  it.each(["quick", "planning", "brainstorm"] as const)("loads %s writing instructions with compatible defaults and no silent error fallback", async (mode) => {
    const id = `writing.agent.${mode}` as const
    for (const fallback of ["catalog-404", "omitted", "detail-404", "saved"]) {
      mocks.listServicePrompts.mockResolvedValue(fallback === "omitted" ? [] : catalog)
      if (fallback === "catalog-404") mocks.listServicePrompts.mockRejectedValueOnce(new ServicePromptApiError("Not found", { status: 404 }))
      mocks.getServicePrompt.mockResolvedValue(detailFor(id, { effective_parts: { system: "Custom {literal}" }, source: "user" }))
      if (fallback === "detail-404") mocks.getServicePrompt.mockRejectedValueOnce(new ServicePromptApiError("Not found", { status: 404 }))
      const snapshot = await loadServicePromptSnapshot([id])
      expect(snapshot.definitions[id]?.parts).toEqual(fallback === "saved" ? { system: "Custom {literal}" } : fixture.defaults[id])
      snapshot.release()
    }
    for (const status of [401, 403, 409, 422, 500]) {
      const error = new ServicePromptApiError("Failed", { status })
      mocks.getServicePrompt.mockRejectedValueOnce(error)
      await expect(loadServicePromptSnapshot([id])).rejects.toBe(error)
    }
  })

  beforeEach(() => {
    vi.clearAllMocks()
    mocks.localGet.mockResolvedValue(undefined)
    mocks.syncGet.mockResolvedValue(undefined)
    mocks.localRemove.mockResolvedValue(undefined)
    mocks.syncRemove.mockResolvedValue(undefined)
    mocks.initialize.mockResolvedValue(undefined)
    mocks.ensureConfig.mockResolvedValue(config)
    mocks.getCurrentUser.mockResolvedValue({ id: 42, username: "user" })
    mocks.isHosted.mockReturnValue(false)
    mocks.listServicePrompts.mockResolvedValue(catalog)
    mocks.getServicePrompt.mockImplementation(async (id: KnownServicePromptId) =>
      detailFor(id)
    )
    mocks.saveServicePrompt.mockImplementation(async (id: KnownServicePromptId) =>
      detailFor(id, { source: "user", revision: "revision-new" })
    )
    mocks.promptForRag.mockResolvedValue({
      ragPrompt: fixture.defaults["chat.rag.answer"].template,
      ragQuestionPrompt:
        fixture.defaults["chat.rag.question_rewrite"].template
    })
    mocks.getWebSearchPrompt.mockResolvedValue(
      fixture.defaults["chat.web_search.answer"].template
    )
    mocks.bgRequest.mockReset()
  })

  it("raw-probes fixed legacy keys with local RAG precedence and local-only web search", async () => {
    mocks.localGet.mockImplementation(async (key: string) => ({
      systemPromptForRag: "local rag {context} {question}",
      webSearchPrompt: "local web {current_date_time} {search_results}",
      webSearchFollowUpPrompt: "must remain untouched"
    } as Record<string, string>)[key])
    mocks.syncGet.mockImplementation(async (key: string) => ({
      systemPromptForRag: "sync rag {context} {question}",
      questionPromptForRag: "sync rewrite {chat_history} {question}",
      webSearchPrompt: "sync web must be ignored",
      webSearchFollowUpPrompt: "must remain untouched"
    } as Record<string, string>)[key])

    await expect(readLegacyServicePromptCandidates()).resolves.toEqual([
      {
        definitionId: "chat.rag.answer",
        partKey: "template",
        storageKey: "systemPromptForRag",
        value: "local rag {context} {question}"
      },
      {
        definitionId: "chat.rag.question_rewrite",
        partKey: "template",
        storageKey: "questionPromptForRag",
        value: "sync rewrite {chat_history} {question}"
      },
      {
        definitionId: "chat.web_search.answer",
        partKey: "template",
        storageKey: "webSearchPrompt",
        value: "local web {current_date_time} {search_results}"
      }
    ])

    expect(mocks.syncGet).not.toHaveBeenCalledWith("webSearchPrompt")
    expect(mocks.localGet).not.toHaveBeenCalledWith("webSearchFollowUpPrompt")
    expect(mocks.syncGet).not.toHaveBeenCalledWith("webSearchFollowUpPrompt")
    expect(mocks.promptForRag).not.toHaveBeenCalled()
    expect(mocks.getWebSearchPrompt).not.toHaveBeenCalled()
  })

  it("uses raw sync RAG only when local is absent and retains invalid text", async () => {
    mocks.syncGet.mockImplementation(async (key: string) =>
      key === "systemPromptForRag" ? "   " : undefined
    )

    await expect(readLegacyServicePromptCandidates()).resolves.toEqual([
      {
        definitionId: "chat.rag.answer",
        partKey: "template",
        storageKey: "systemPromptForRag",
        value: "   "
      }
    ])
  })

  it("resolves multi-user and hosted scope through the authenticated user, never anonymous", async () => {
    const multiUserConfig = {
      ...config,
      authMode: "multi-user" as const,
      accessToken: jwtForUser(84)
    }
    const multiUserTarget = targetConfig(multiUserConfig)
    mocks.ensureConfig.mockResolvedValue(multiUserConfig)
    mocks.getCurrentUser.mockResolvedValue({ id: 84, username: "resolved" })

    await expect(resolveServicePromptScope()).resolves.toEqual({
      config: multiUserTarget,
      scopeKey: scopeKeyFor(multiUserConfig, 84),
      userId: 84,
      clientPrincipalVerified: true
    })
    expect(mocks.getCurrentUser).toHaveBeenCalledTimes(1)
    expect(mocks.initialize).toHaveBeenCalledTimes(2)

    vi.clearAllMocks()
    mocks.isHosted.mockReturnValue(true)
    mocks.ensureConfig.mockResolvedValue(config)
    mocks.getCurrentUser.mockResolvedValue({ id: 85, username: "hosted" })

    const hosted = await resolveServicePromptScope()
    expect(mocks.getCurrentUser).toHaveBeenCalledTimes(1)
    expect(hosted.userId).toBe(85)
    expect(hosted.clientPrincipalVerified).toBe(false)
    expect(hosted.config).not.toHaveProperty("apiKey")
    expect(hosted.config).not.toHaveProperty("accessToken")
  })

  it("marks a missing authenticated user with a stable scope error code", async () => {
    mocks.ensureConfig.mockResolvedValue({
      ...config,
      authMode: "multi-user" as const,
      accessToken: "valid-token"
    })
    mocks.getCurrentUser.mockResolvedValue(null)

    await expect(resolveServicePromptScope()).rejects.toMatchObject({
      code: "service_prompt_scope_unresolved"
    })
  })

  it("marks an authenticated-user 401 as an unresolved scope", async () => {
    mocks.ensureConfig.mockResolvedValue({
      ...config,
      authMode: "multi-user" as const,
      accessToken: "rejected-token"
    })
    mocks.getCurrentUser.mockRejectedValue(Object.assign(
      new Error("redacted"),
      { status: 401 }
    ))

    await expect(resolveServicePromptScope()).rejects.toMatchObject({
      code: "service_prompt_scope_unresolved"
    })
  })

  it("preserves transient authenticated-user resolution failures", async () => {
    const transient = Object.assign(new Error("temporarily offline"), {
      status: 503
    })
    mocks.ensureConfig.mockResolvedValue({
      ...config,
      authMode: "multi-user" as const,
      accessToken: "valid-token"
    })
    mocks.getCurrentUser.mockRejectedValue(transient)

    await expect(resolveServicePromptScope()).rejects.toBe(transient)
  })

  it("marks a stored multi-user logout as an unresolved scope", async () => {
    const loggedOutConfig = {
      ...config,
      authMode: "multi-user" as const,
      accessToken: undefined,
      refreshToken: undefined
    }
    mocks.ensureConfig.mockImplementation(async (requireAuth: boolean) => {
      if (requireAuth) throw new Error("Not authenticated")
      return loggedOutConfig
    })

    await expect(resolveServicePromptScope()).rejects.toMatchObject({
      code: "service_prompt_scope_unresolved"
    })
    expect(mocks.getCurrentUser).not.toHaveBeenCalled()
  })

  it("captures credentials refreshed while resolving the authenticated user", async () => {
    const expiredConfig = {
      ...config,
      authMode: "multi-user" as const,
      accessToken: "expired-token"
    }
    const refreshedConfig = {
      ...expiredConfig,
      accessToken: "refreshed-token"
    }
    let clientRefreshed = false
    mocks.ensureConfig.mockImplementation(async () =>
      clientRefreshed ? refreshedConfig : expiredConfig
    )
    mocks.getCurrentUser.mockImplementation(async () => {
      clientRefreshed = true
      return { id: 84, username: "resolved" }
    })

    await expect(resolveServicePromptScope()).resolves.toMatchObject({
      config: targetConfig(refreshedConfig),
      userId: 84
    })
    expect(mocks.ensureConfig).toHaveBeenCalledTimes(2)
    expect(mocks.initialize).toHaveBeenCalledTimes(2)
  })

  it("rejects a server target change during authenticated-user resolution", async () => {
    const firstConfig = {
      ...config,
      authMode: "multi-user" as const,
      accessToken: "first-token"
    }
    const changedConfig = {
      ...firstConfig,
      serverUrl: "https://other-server.example",
      accessToken: "other-token"
    }
    let clientRefreshed = false
    mocks.ensureConfig.mockImplementation(async () =>
      clientRefreshed ? changedConfig : firstConfig
    )
    mocks.getCurrentUser.mockImplementation(async () => {
      clientRefreshed = true
      return { id: 84, username: "resolved" }
    })

    await expect(resolveServicePromptScope()).rejects.toThrow(
      "Authenticated Service Prompt scope changed while resolving."
    )
  })

  it("does not resolve a user for single-user scope", async () => {
    mocks.ensureConfig.mockResolvedValue({
      ...config,
      apiKey: "effective-runtime-key"
    })

    await expect(resolveServicePromptScope()).resolves.toMatchObject({
      config: targetConfig(config),
      userId: null,
      clientPrincipalVerified: true
    })

    expect(mocks.getCurrentUser).not.toHaveBeenCalled()
    expect(mocks.ensureConfig).toHaveBeenCalledWith(true)
  })

  it("uses compatibility getters only when the catalog alone returns 404", async () => {
    mocks.listServicePrompts.mockRejectedValue(
      new ServicePromptApiError("Not found", { status: 404 })
    )
    mocks.promptForRag.mockResolvedValue({
      ragPrompt: "legacy rag {context} {question}",
      ragQuestionPrompt: "legacy rewrite {chat_history} {question}"
    })
    mocks.getWebSearchPrompt.mockResolvedValue(
      "legacy web {current_date_time} {search_results}"
    )

    const snapshot = await loadServicePromptSnapshot([
      "chat.rag.answer",
      "chat.rag.question_rewrite",
      "chat.web_search.answer"
    ])

    expect(snapshot).toMatchObject({
      scopeKey: scopeKeyFor(config, null),
      capability: "legacy-404",
      definitions: {
        "chat.rag.answer": {
          definition: renderDefinitionFor("chat.rag.answer"),
          parts: { template: "legacy rag {context} {question}" },
          source: "user",
          revision: null
        },
        "chat.rag.question_rewrite": {
          definition: renderDefinitionFor("chat.rag.question_rewrite"),
          parts: { template: "legacy rewrite {chat_history} {question}" },
          source: "user",
          revision: null
        },
        "chat.web_search.answer": {
          definition: renderDefinitionFor("chat.web_search.answer"),
          parts: {
            template: "legacy web {current_date_time} {search_results}"
          },
          source: "user",
          revision: null
        }
      }
    })
    expect(snapshot.scopeSignal).toBeInstanceOf(AbortSignal)
    expect(snapshot.release).toBeTypeOf("function")
    expect(snapshot.requestScope).toEqual({
      config: {
        ...targetConfig(config),
        expectedSingleUserApiKeyScope: singleUserApiKeyScopeFor(config)
      },
      userId: null
    })
    expect(Object.isFrozen(snapshot.requestScope)).toBe(true)
    expect(Object.isFrozen(snapshot.requestScope.config)).toBe(true)
    snapshot.release()
    expect(mocks.getServicePrompt).not.toHaveBeenCalled()
  })

  it("refuses legacy compatibility when a cookie-session principal cannot be verified", async () => {
    const hostedConfig = {
      serverUrl: config.serverUrl,
      authMode: "multi-user" as const,
      authSource: "cookie-session" as const,
      orgId: config.orgId
    }
    mocks.isHosted.mockReturnValue(true)
    mocks.ensureConfig.mockResolvedValue(hostedConfig)
    mocks.getCurrentUser.mockResolvedValue({ id: 84, username: "hosted" })
    mocks.listServicePrompts.mockRejectedValue(
      new ServicePromptApiError("Not found", { status: 404 })
    )

    await expect(
      loadServicePromptSnapshot(["chat.rag.answer"])
    ).rejects.toMatchObject({
      status: 412,
      details: {
        detail: { code: "request_config_scope_changed" }
      }
    })

    expect(mocks.promptForRag).not.toHaveBeenCalled()
  })

  it("uses the packaged title template on old servers without reading legacy storage", async () => {
    mocks.listServicePrompts.mockRejectedValue(
      new ServicePromptApiError("Not found", { status: 404 })
    )

    const snapshot = await loadServicePromptSnapshot([
      "chat.title.generation"
    ])

    expect(snapshot).toMatchObject({
      capability: "legacy-404",
      definitions: {
        "chat.title.generation": {
          definition: renderDefinitionFor("chat.title.generation"),
          parts: {
            user_template: fixture.defaults["chat.title.generation"].user_template
          },
          source: "packaged",
          revision: null
        }
      }
    })
    expect(mocks.promptForRag).not.toHaveBeenCalled()
    expect(mocks.getWebSearchPrompt).not.toHaveBeenCalled()
    expect(mocks.localGet).not.toHaveBeenCalled()
    expect(mocks.syncGet).not.toHaveBeenCalled()
    snapshot.release()
  })

  it("uses packaged image-refinement semantics on old servers without reading legacy storage", async () => {
    mocks.listServicePrompts.mockRejectedValue(
      new ServicePromptApiError("Not found", { status: 404 })
    )

    const snapshot = await loadServicePromptSnapshot([
      "image.prompt.refinement"
    ])

    expect(snapshot).toMatchObject({
      capability: "legacy-404",
      definitions: {
        "image.prompt.refinement": {
          definition: {
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
          },
          parts: fixture.defaults["image.prompt.refinement"],
          source: "packaged",
          revision: null
        }
      }
    })
    expect(mocks.promptForRag).not.toHaveBeenCalled()
    expect(mocks.getWebSearchPrompt).not.toHaveBeenCalled()
    expect(mocks.localGet).not.toHaveBeenCalled()
    expect(mocks.syncGet).not.toHaveBeenCalled()
    snapshot.release()
  })

  it("uses packaged image-refinement semantics when a supported older catalog omits the definition", async () => {
    mocks.listServicePrompts.mockResolvedValue(
      catalog.filter((item) => item.id !== "image.prompt.refinement")
    )

    const snapshot = await loadServicePromptSnapshot([
      "image.prompt.refinement"
    ])

    expect(snapshot).toMatchObject({
      capability: "supported",
      definitions: {
        "image.prompt.refinement": {
          definition: renderDefinitionFor("image.prompt.refinement"),
          parts: fixture.defaults["image.prompt.refinement"],
          source: "packaged",
          revision: null
        }
      }
    })
    expect(mocks.getServicePrompt).not.toHaveBeenCalled()
    expect(mocks.localGet).not.toHaveBeenCalled()
    expect(mocks.syncGet).not.toHaveBeenCalled()
    snapshot.release()
  })

  it("uses packaged image-refinement semantics when an advertised detail returns 404", async () => {
    mocks.getServicePrompt.mockRejectedValueOnce(
      new ServicePromptApiError("Not found", { status: 404 })
    )

    const snapshot = await loadServicePromptSnapshot([
      "image.prompt.refinement"
    ])

    expect(snapshot).toMatchObject({
      capability: "supported",
      definitions: {
        "image.prompt.refinement": {
          definition: renderDefinitionFor("image.prompt.refinement"),
          parts: fixture.defaults["image.prompt.refinement"],
          source: "packaged",
          revision: null
        }
      }
    })
    expect(mocks.getServicePrompt).toHaveBeenCalledWith(
      "image.prompt.refinement",
      expect.objectContaining({ signal: expect.any(AbortSignal) })
    )
    expect(mocks.promptForRag).not.toHaveBeenCalled()
    expect(mocks.getWebSearchPrompt).not.toHaveBeenCalled()
    snapshot.release()
  })

  it.each([412, 500])(
    "does not fallback when an advertised image-refinement detail returns %s",
    async (status) => {
      const error = new ServicePromptApiError("Detail failed", { status })
      mocks.getServicePrompt.mockRejectedValueOnce(error)

      await expect(
        loadServicePromptSnapshot(["image.prompt.refinement"])
      ).rejects.toBe(error)
    }
  )

  it("rejects a mismatched authenticated user after resolving the matching multi-user target", async () => {
    const multiUserConfig = {
      ...config,
      authMode: "multi-user" as const,
      accessToken: jwtForUser(42)
    }
    mocks.ensureConfig.mockResolvedValue(multiUserConfig)
    mocks.getCurrentUser.mockResolvedValue({ id: 42, username: "resolved" })

    await expect(loadServicePromptSnapshot(
      ["chat.title.generation"],
      {
        requestScope: {
          config: targetConfig(multiUserConfig),
          userId: 999
        }
      }
    )).rejects.toMatchObject({
      status: 412,
      details: { detail: { code: "request_config_scope_changed" } }
    })
    expect(mocks.listServicePrompts).not.toHaveBeenCalled()
    expect(mocks.getServicePrompt).not.toHaveBeenCalled()
  })

  it.each([
    {
      name: "server",
      requestScope: {
        config: { ...config, serverUrl: "https://other-server.example" },
        userId: null
      }
    },
    {
      name: "single-user API-key scope",
      requestScope: {
        config: {
          ...config,
          expectedSingleUserApiKeyScope: "other-api-key-scope"
        },
        userId: null
      }
    }
  ])("rejects a mismatched expected request scope before catalog reads: $name", async ({ requestScope }) => {
    await expect(loadServicePromptSnapshot(
      ["chat.title.generation"],
      { requestScope }
    )).rejects.toMatchObject({
      status: 412,
      details: { detail: { code: "request_config_scope_changed" } }
    })
    expect(mocks.listServicePrompts).not.toHaveBeenCalled()
    expect(mocks.getServicePrompt).not.toHaveBeenCalled()
  })

  it("loads catalog and detail when expected server, account, and API-key scope match", async () => {
    const matchingRequestScope = {
      config: {
        ...targetConfig(config),
        expectedSingleUserApiKeyScope: singleUserApiKeyScopeFor(config)
      },
      userId: null
    }

    const snapshot = await loadServicePromptSnapshot(
      ["chat.title.generation"],
      { requestScope: matchingRequestScope }
    )

    expect(snapshot.definitions["chat.title.generation"]?.parts).toEqual(
      fixture.defaults["chat.title.generation"]
    )
    expect(mocks.listServicePrompts).toHaveBeenCalledOnce()
    expect(mocks.getServicePrompt).toHaveBeenCalledOnce()
    expect(mocks.listServicePrompts).toHaveBeenCalledWith(
      expect.objectContaining({
        requestScope: expect.objectContaining({
          config: matchingRequestScope.config,
          userId: null
        })
      })
    )
    expect(mocks.getServicePrompt).toHaveBeenCalledWith(
      "chat.title.generation",
      expect.objectContaining({
        requestScope: expect.objectContaining({
          config: matchingRequestScope.config,
          userId: null
        })
      })
    )
    snapshot.release()
  })

  it.each([401, 403, 500, 0])(
    "does not fall back on catalog status %i",
    async (status) => {
      const error = new ServicePromptApiError("Catalog failed", { status })
      mocks.listServicePrompts.mockRejectedValue(error)

      await expect(
        loadServicePromptSnapshot(["chat.rag.answer"])
      ).rejects.toBe(error)
      expect(mocks.promptForRag).not.toHaveBeenCalled()
      expect(mocks.getWebSearchPrompt).not.toHaveBeenCalled()
    }
  )

  it("does not fall back on non-protocol catalog failures", async () => {
    const error = new Error("network failed")
    mocks.listServicePrompts.mockRejectedValue(error)

    await expect(
      loadServicePromptSnapshot(["chat.rag.answer"])
    ).rejects.toBe(error)
    expect(mocks.promptForRag).not.toHaveBeenCalled()
  })

  it("blocks an affected supported workflow before detail reads when migration remains", async () => {
    mocks.localGet.mockImplementation(async (key: string) =>
      key === "systemPromptForRag"
        ? "legacy {context} {question}"
        : undefined
    )

    await expect(
      loadServicePromptSnapshot(["chat.rag.answer"])
    ).rejects.toThrow("Review workflow prompts")
    expect(mocks.getServicePrompt).not.toHaveBeenCalled()
  })

  it("does not block an unrelated requested workflow", async () => {
    mocks.localGet.mockImplementation(async (key: string) =>
      key === "systemPromptForRag"
        ? "legacy {context} {question}"
        : undefined
    )

    const snapshot = await loadServicePromptSnapshot(["chat.web_search.answer"])
    expect(snapshot).toMatchObject({ capability: "supported" })
    expect(mocks.getServicePrompt).toHaveBeenCalledWith(
      "chat.web_search.answer",
      expect.objectContaining({ signal: expect.any(AbortSignal) })
    )
    snapshot.release()
  })

  it("loads requested details concurrently and freshly for every invocation", async () => {
    let resolveAnswer!: (value: ServicePromptDetail) => void
    let resolveRewrite!: (value: ServicePromptDetail) => void
    mocks.getServicePrompt.mockImplementation((id: KnownServicePromptId) =>
      new Promise<ServicePromptDetail>((resolve) => {
        if (id === "chat.rag.answer") resolveAnswer = resolve
        if (id === "chat.rag.question_rewrite") resolveRewrite = resolve
      })
    )

    const pending = loadServicePromptSnapshot([
      "chat.rag.answer",
      "chat.rag.question_rewrite"
    ])
    await vi.waitFor(() => {
      expect(mocks.getServicePrompt).toHaveBeenCalledTimes(2)
    })
    resolveAnswer(detailFor("chat.rag.answer"))
    resolveRewrite(detailFor("chat.rag.question_rewrite"))
    const first = await pending

    mocks.getServicePrompt.mockImplementation(async (id: KnownServicePromptId) =>
      detailFor(id)
    )
    const second = await loadServicePromptSnapshot([
      "chat.rag.answer",
      "chat.rag.question_rewrite"
    ])
    expect(mocks.listServicePrompts).toHaveBeenCalledTimes(2)
    expect(mocks.getServicePrompt).toHaveBeenCalledTimes(4)
    first.release()
    second.release()
  })

  it("never converts a supported detail failure into legacy fallback", async () => {
    const error = new ServicePromptApiError("Detail failed", { status: 404 })
    mocks.getServicePrompt.mockRejectedValue(error)

    await expect(
      loadServicePromptSnapshot(["chat.rag.answer"])
    ).rejects.toBe(error)
    expect(mocks.promptForRag).not.toHaveBeenCalled()
  })

  it("deep-freezes snapshot definitions, render schemas, and nested variables", async () => {
    const snapshot = await loadServicePromptSnapshot(["chat.rag.answer"])
    const resolved = snapshot.definitions["chat.rag.answer"]!

    expect(Object.isFrozen(snapshot)).toBe(true)
    expect(Object.isFrozen(snapshot.definitions)).toBe(true)
    expect(Object.isFrozen(resolved)).toBe(true)
    expect(Object.isFrozen(resolved.parts)).toBe(true)
    expect(Object.isFrozen(resolved.definition)).toBe(true)
    expect(Object.isFrozen(resolved.definition.parts)).toBe(true)
    expect(Object.isFrozen(resolved.definition.parts[0])).toBe(true)
    expect(Object.isFrozen(
      resolved.definition.parts[0].required_variables
    )).toBe(true)
    expect(renderServicePromptPart(
      resolved.definition,
      "template",
      resolved.parts.template,
      { context: "Frozen context", question: "Frozen question" }
    )).toContain("Frozen context")
    snapshot.release()
  })

  it("keeps a parent caller abort as AbortError", async () => {
    let detailSignal: AbortSignal | undefined
    mocks.getServicePrompt.mockImplementation(
      async (_id: KnownServicePromptId, options: { signal?: AbortSignal }) => {
        detailSignal = options.signal
        return await new Promise<ServicePromptDetail>((_resolve, reject) => {
          options.signal?.addEventListener("abort", () =>
            reject(new DOMException("Aborted", "AbortError")))
        })
      }
    )
    const controller = new AbortController()
    const pending = loadServicePromptSnapshot(
      ["chat.rag.answer"],
      { signal: controller.signal }
    )
    await vi.waitFor(() => expect(detailSignal).toBeDefined())
    controller.abort()

    await expect(pending).rejects.toMatchObject({ name: "AbortError" })
    expect(detailSignal?.aborted).toBe(true)
  })

  it("normalizes hosted credential invalidation during principal resolution", async () => {
    let resolveUser!: (user: { id: number; username: string }) => void
    mocks.isHosted.mockReturnValue(true)
    mocks.getCurrentUser.mockImplementation(() =>
      new Promise((resolve) => {
        resolveUser = resolve
      })
    )

    const pending = loadServicePromptSnapshot(["chat.rag.answer"])
    await vi.waitFor(() => expect(mocks.getCurrentUser).toHaveBeenCalled())
    window.dispatchEvent(new Event("tldw:auth-credentials-changed"))
    resolveUser({ id: 99, username: "changed" })

    await expect(pending).rejects.toMatchObject({
      status: 412,
      details: { detail: { code: "request_config_scope_changed" } }
    })
    expect(mocks.listServicePrompts).not.toHaveBeenCalled()
  })

  it("normalizes invalidation during an unabortable catalog-404 compatibility getter", async () => {
    let resolvePrompts!: (value: {
      ragPrompt: string
      ragQuestionPrompt: string
    }) => void
    mocks.listServicePrompts.mockRejectedValue(
      new ServicePromptApiError("Not found", { status: 404 })
    )
    mocks.promptForRag.mockImplementation(() =>
      new Promise((resolve) => {
        resolvePrompts = resolve
      })
    )

    const pending = loadServicePromptSnapshot(["chat.rag.answer"])
    await vi.waitFor(() => expect(mocks.promptForRag).toHaveBeenCalled())
    const watched = mocks.localWatch.mock.calls.at(-1)?.[0] as
      | { tldwConfig?: (change: { newValue?: unknown }) => void }
      | undefined
    watched?.tldwConfig?.({
      newValue: { ...config, serverUrl: "https://other.example" }
    })
    resolvePrompts({
      ragPrompt: "legacy {context} {question}",
      ragQuestionPrompt: "legacy {chat_history} {question}"
    })

    await expect(pending).rejects.toMatchObject({
      status: 412,
      details: { detail: { code: "request_config_scope_changed" } }
    })
  })

  it("normalizes invalidation during the supported raw migration probe", async () => {
    let resolveRaw!: (value: undefined) => void
    mocks.localGet.mockImplementationOnce(() =>
      new Promise((resolve) => {
        resolveRaw = resolve
      })
    )

    const pending = loadServicePromptSnapshot(["chat.rag.answer"])
    await vi.waitFor(() => expect(mocks.localGet).toHaveBeenCalled())
    window.dispatchEvent(new Event("tldw:auth-credentials-changed"))
    resolveRaw(undefined)

    await expect(pending).rejects.toMatchObject({
      status: 412,
      details: { detail: { code: "request_config_scope_changed" } }
    })
    expect(mocks.getServicePrompt).not.toHaveBeenCalled()
  })

  it("normalizes cross-context scope invalidation during an active read", async () => {
    let detailSignal: AbortSignal | undefined
    mocks.getServicePrompt.mockImplementation(
      async (_id: KnownServicePromptId, options: { signal?: AbortSignal }) => {
        detailSignal = options.signal
        return await new Promise<ServicePromptDetail>((_resolve, reject) => {
          options.signal?.addEventListener("abort", () =>
            reject(new DOMException("Aborted", "AbortError")))
        })
      }
    )
    const external = new AbortController()
    const pending = loadServicePromptSnapshot(
      ["chat.rag.answer"],
      { signal: external.signal }
    )
    await vi.waitFor(() => expect(detailSignal).toBeDefined())

    const watched = mocks.localWatch.mock.calls.at(-1)?.[0] as
      | { tldwConfig?: (change: { newValue?: unknown }) => void }
      | undefined
    if (watched?.tldwConfig) {
      watched.tldwConfig({ newValue: { ...config, serverUrl: "https://new.example" } })
    } else {
      external.abort()
    }

    await expect(pending).rejects.toMatchObject({
      status: 412,
      details: { detail: { code: "request_config_scope_changed" } }
    })
    expect(watched?.tldwConfig).toBeTypeOf("function")
    expect(mocks.localUnwatch).toHaveBeenCalledWith(watched)
  })

  it("normalizes a single-user lease invalidation when the API-key account changes", async () => {
    let detailSignal: AbortSignal | undefined
    mocks.getServicePrompt.mockImplementation(
      async (_id: KnownServicePromptId, options: { signal?: AbortSignal }) => {
        detailSignal = options.signal
        return await new Promise<ServicePromptDetail>((_resolve, reject) => {
          options.signal?.addEventListener("abort", () =>
            reject(new DOMException("Aborted", "AbortError")))
        })
      }
    )

    const pending = loadServicePromptSnapshot(["chat.rag.answer"])
    await vi.waitFor(() => expect(detailSignal).toBeDefined())
    const watched = mocks.localWatch.mock.calls.at(-1)?.[0] as
      | { tldwConfig?: (change: { newValue?: unknown }) => void }
      | undefined
    watched?.tldwConfig?.({
      newValue: { ...config, apiKey: "different-account-key" }
    })

    await expect(pending).rejects.toMatchObject({
      status: 412,
      details: { detail: { code: "request_config_scope_changed" } }
    })
    expect(watched?.tldwConfig).toBeTypeOf("function")
  })

  it("aborts a single-user lease when changed API keys collide in the UI scope hash", async () => {
    const capturedKey = "key-s54895-4z7"
    const changedKey = "key-jiqole-3dcy"
    mocks.ensureConfig.mockResolvedValue({ ...config, apiKey: capturedKey })
    let detailSignal: AbortSignal | undefined
    mocks.getServicePrompt.mockImplementation(
      async (_id: KnownServicePromptId, options: { signal?: AbortSignal }) => {
        detailSignal = options.signal
        return await new Promise<ServicePromptDetail>((_resolve, reject) => {
          options.signal?.addEventListener("abort", () =>
            reject(new DOMException("Aborted", "AbortError")))
        })
      }
    )
    const external = new AbortController()
    const pending = loadServicePromptSnapshot(
      ["chat.rag.answer"],
      { signal: external.signal }
    )
    await vi.waitFor(() => expect(detailSignal).toBeDefined())
    const watched = mocks.localWatch.mock.calls.at(-1)?.[0] as
      | { tldwConfig?: (change: { newValue?: unknown }) => void }
      | undefined

    watched?.tldwConfig?.({
      newValue: { ...config, apiKey: changedKey }
    })

    try {
      await vi.waitFor(() => expect(detailSignal?.aborted).toBe(true))
    } finally {
      external.abort()
      await pending.catch(() => undefined)
    }
    expect(watched?.tldwConfig).toBeTypeOf("function")
  })

  it("keeps same-user token rotation alive and aborts the retained turn lease on account change", async () => {
    const multiUserConfig = {
      serverUrl: config.serverUrl,
      authMode: "multi-user" as const,
      orgId: config.orgId,
      accessToken: jwtForUser(42)
    }
    mocks.ensureConfig.mockResolvedValue(multiUserConfig)
    mocks.getCurrentUser.mockResolvedValue({ id: 42, username: "user" })
    let resolveDetail!: (value: ServicePromptDetail) => void
    mocks.getServicePrompt.mockImplementation(() =>
      new Promise<ServicePromptDetail>((resolve) => {
        resolveDetail = resolve
      })
    )

    const pending = loadServicePromptSnapshot(["chat.rag.answer"])
    await vi.waitFor(() => expect(mocks.getServicePrompt).toHaveBeenCalledOnce())
    const watched = mocks.localWatch.mock.calls.at(-1)?.[0] as
      | { tldwConfig?: (change: { newValue?: unknown }) => void }
      | undefined

    watched?.tldwConfig?.({
      newValue: { ...multiUserConfig, accessToken: jwtForUser(42) }
    })
    resolveDetail(detailFor("chat.rag.answer"))
    const snapshot = await pending

    expect(snapshot.scopeSignal).toBeInstanceOf(AbortSignal)
    expect(snapshot.scopeSignal?.aborted).toBe(false)
    expect(mocks.localUnwatch).not.toHaveBeenCalledWith(watched)

    watched?.tldwConfig?.({
      newValue: { ...multiUserConfig, accessToken: jwtForUser(84) }
    })

    expect(snapshot.scopeSignal?.aborted).toBe(true)
    expect(snapshot.scopeInvalidatedSignal?.aborted).toBe(true)
    expect(mocks.localUnwatch).not.toHaveBeenCalledWith(watched)
    snapshot.release()
    expect(mocks.localUnwatch).toHaveBeenCalledWith(watched)
  })

  it("keeps watching for scope invalidation after the caller aborts", async () => {
    const multiUserConfig = {
      serverUrl: config.serverUrl,
      authMode: "multi-user" as const,
      orgId: config.orgId,
      accessToken: jwtForUser(42)
    }
    mocks.ensureConfig.mockResolvedValue(multiUserConfig)
    mocks.getCurrentUser.mockResolvedValue({ id: 42, username: "user" })
    const parent = new AbortController()
    const snapshot = await loadServicePromptSnapshot(
      ["chat.rag.answer"],
      { signal: parent.signal }
    )
    const watched = mocks.localWatch.mock.calls.at(-1)?.[0] as
      | { tldwConfig?: (change: { newValue?: unknown }) => void }
      | undefined

    parent.abort()

    expect(snapshot.scopeSignal.aborted).toBe(true)
    expect(snapshot.scopeInvalidatedSignal?.aborted).toBe(false)
    expect(mocks.localUnwatch).not.toHaveBeenCalledWith(watched)

    watched?.tldwConfig?.({
      newValue: { ...multiUserConfig, accessToken: jwtForUser(84) }
    })

    expect(snapshot.scopeInvalidatedSignal?.aborted).toBe(true)
    expect(mocks.localUnwatch).not.toHaveBeenCalledWith(watched)
    snapshot.release()
    expect(mocks.localUnwatch).toHaveBeenCalledWith(watched)
  })

  it("rehydrates client config at the start of every invocation", async () => {
    const nextConfig = { ...config, serverUrl: "https://next.example" }
    let liveConfig = config
    mocks.ensureConfig.mockImplementation(async () => liveConfig)
    mocks.initialize.mockImplementation(async () => {
      if (mocks.initialize.mock.calls.length === 2) {
        liveConfig = nextConfig
      }
    })
    const first = await loadServicePromptSnapshot(["chat.rag.answer"])
    const second = await loadServicePromptSnapshot(["chat.rag.answer"])

    expect(mocks.initialize).toHaveBeenCalledTimes(2)
    expect(first.scopeKey).toBe(scopeKeyFor(config, null))
    expect(second.scopeKey).toBe(scopeKeyFor(nextConfig, null))
    first.release()
    second.release()
  })

  it("does not treat initialize-time normalization of the same target as cancellation", async () => {
    mocks.initialize.mockImplementation(async () => {
      const watched = mocks.localWatch.mock.calls.at(-1)?.[0] as
        | { tldwConfig?: (change: { newValue?: unknown }) => void }
        | undefined
      watched?.tldwConfig?.({ newValue: config })
    })

    const snapshot = await loadServicePromptSnapshot(["chat.rag.answer"])
    expect(snapshot).toMatchObject({ capability: "supported" })
    expect(mocks.ensureConfig).toHaveBeenCalled()
    snapshot.release()
  })

  it("retains the per-turn config watcher until the snapshot owner releases it", async () => {
    const snapshot = await loadServicePromptSnapshot(["chat.rag.answer"])

    const watched = mocks.localWatch.mock.calls[0]?.[0]
    expect(watched).toBeDefined()
    expect(mocks.localUnwatch).not.toHaveBeenCalledWith(watched)

    snapshot.release()

    expect(mocks.localUnwatch).toHaveBeenCalledWith(watched)
  })

  it("clears both raw areas only after each successful import", async () => {
    const candidates = [
      {
        definitionId: "chat.rag.answer" as const,
        partKey: "template" as const,
        storageKey: "systemPromptForRag" as const,
        value: "import {context} {question}"
      },
      {
        definitionId: "chat.rag.question_rewrite" as const,
        partKey: "template" as const,
        storageKey: "questionPromptForRag" as const,
        value: "import {chat_history} {question}"
      }
    ]
    mocks.saveServicePrompt
      .mockResolvedValueOnce(detailFor("chat.rag.answer", {
        source: "user",
        revision: "revision-new"
      }))
      .mockRejectedValueOnce(new Error("save failed"))

    await importLegacyServicePromptCandidate(
      candidates[0],
      detailFor("chat.rag.answer")
    )
    await expect(importLegacyServicePromptCandidate(
      candidates[1],
      detailFor("chat.rag.question_rewrite")
    )).rejects.toThrow("save failed")

    expect(mocks.saveServicePrompt).toHaveBeenNthCalledWith(1,
      "chat.rag.answer",
      {
        parts: { template: "import {context} {question}" },
        expected_revision: null
      },
      { signal: undefined }
    )
    expect(mocks.localRemove).toHaveBeenCalledTimes(1)
    expect(mocks.localRemove).toHaveBeenCalledWith("systemPromptForRag")
    expect(mocks.syncRemove).toHaveBeenCalledTimes(1)
    expect(mocks.syncRemove).toHaveBeenCalledWith("systemPromptForRag")
    expect(mocks.localRemove).not.toHaveBeenCalledWith("questionPromptForRag")
    expect(mocks.syncRemove).not.toHaveBeenCalledWith("questionPromptForRag")
  })

  it("clears only fixed mapped keys and never the follow-up key", async () => {
    await clearLegacyServicePromptCandidate("chat.web_search.answer")

    expect(mocks.localRemove).toHaveBeenCalledWith("webSearchPrompt")
    expect(mocks.syncRemove).toHaveBeenCalledWith("webSearchPrompt")
    expect(mocks.localRemove).not.toHaveBeenCalledWith("webSearchFollowUpPrompt")
    expect(mocks.syncRemove).not.toHaveBeenCalledWith("webSearchFollowUpPrompt")
  })

  it("does not clear legacy values when PUT returns a protocol error", async () => {
    mocks.saveServicePrompt.mockRejectedValueOnce(
      new ServicePromptApiError(
        "Service Prompt server response was invalid.",
        { status: 0, code: "service_prompt_protocol_error" }
      )
    )
    const candidate = {
      definitionId: "chat.rag.answer" as const,
      partKey: "template" as const,
      storageKey: "systemPromptForRag" as const,
      value: "import {context} {question}"
    }

    await expect(importLegacyServicePromptCandidate(
      candidate,
      detailFor("chat.rag.answer")
    )).rejects.toMatchObject({ code: "service_prompt_protocol_error" })
    expect(mocks.localRemove).not.toHaveBeenCalled()
    expect(mocks.syncRemove).not.toHaveBeenCalled()
  })

  it("does not clear a legacy value when its scope aborts after PUT", async () => {
    let resolveSave!: (value: ServicePromptDetail) => void
    mocks.saveServicePrompt.mockImplementationOnce(() =>
      new Promise<ServicePromptDetail>((resolve) => {
        resolveSave = resolve
      })
    )
    const controller = new AbortController()
    const candidate = {
      definitionId: "chat.rag.answer" as const,
      partKey: "template" as const,
      storageKey: "systemPromptForRag" as const,
      value: "import {context} {question}"
    }
    const pending = importLegacyServicePromptCandidate(
      candidate,
      detailFor("chat.rag.answer"),
      { signal: controller.signal }
    )
    await vi.waitFor(() => expect(mocks.saveServicePrompt).toHaveBeenCalled())

    controller.abort()
    resolveSave(detailFor("chat.rag.answer", {
      source: "user",
      revision: "revision-new"
    }))

    await expect(pending).rejects.toMatchObject({ name: "AbortError" })
    expect(mocks.localRemove).not.toHaveBeenCalled()
    expect(mocks.syncRemove).not.toHaveBeenCalled()
  })

  it.each([
    ["packaged PUT response", detailFor("chat.rag.answer")],
    ["mismatched user PUT response", detailFor("chat.rag.answer", {
      saved_parts: { template: "Different {context} {question}" },
      effective_parts: { template: "Different {context} {question}" },
      source: "user",
      revision: VALID_REVISION
    })]
  ])("does not clear raw values after a %s", async (_name, response) => {
    mocks.bgRequest.mockResolvedValueOnce(response)
    mocks.saveServicePrompt.mockImplementation((
      id: string,
      payload: { parts: Record<string, string>; expected_revision: string | null },
      options?: { signal?: AbortSignal }
    ) => servicePromptMethods.saveServicePrompt(id, payload, options))
    const candidate = {
      definitionId: "chat.rag.answer" as const,
      partKey: "template" as const,
      storageKey: "systemPromptForRag" as const,
      value: "Submitted {context} {question}"
    }

    await expect(importLegacyServicePromptCandidate(
      candidate,
      detailFor("chat.rag.answer")
    )).rejects.toMatchObject({ code: "service_prompt_protocol_error" })
    expect(mocks.localRemove).not.toHaveBeenCalled()
    expect(mocks.syncRemove).not.toHaveBeenCalled()
  })
})

import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  localGet: vi.fn(),
  syncGet: vi.fn(),
  localRemove: vi.fn(),
  syncRemove: vi.fn(),
  localWatch: vi.fn(),
  localUnwatch: vi.fn(),
  initialize: vi.fn(),
  getConfig: vi.fn(),
  getCurrentUser: vi.fn(),
  listServicePrompts: vi.fn(),
  getServicePrompt: vi.fn(),
  saveServicePrompt: vi.fn(),
  promptForRag: vi.fn(),
  getWebSearchPrompt: vi.fn(),
  buildScope: vi.fn(),
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
    getConfig: (...args: unknown[]) => mocks.getConfig(...args),
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

vi.mock("@/services/chat-surface-scope", () => ({
  buildChatSurfaceScopeKeyFromConfig: (...args: unknown[]) =>
    mocks.buildScope(...args)
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

const definitions: Record<KnownServicePromptId, ServicePromptCatalogItem> = {
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
  ])
}

describe("Service Prompt validation and rendering", () => {
  it("keeps the three old-server Chat defaults byte-equivalent to the shared fixture", () => {
    expect(LEGACY_SERVICE_PROMPT_DEFAULTS).toEqual({
      "chat.rag.answer": fixture.defaults["chat.rag.answer"],
      "chat.rag.question_rewrite": fixture.defaults["chat.rag.question_rewrite"],
      "chat.web_search.answer": fixture.defaults["chat.web_search.answer"]
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
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.localGet.mockResolvedValue(undefined)
    mocks.syncGet.mockResolvedValue(undefined)
    mocks.localRemove.mockResolvedValue(undefined)
    mocks.syncRemove.mockResolvedValue(undefined)
    mocks.initialize.mockResolvedValue(undefined)
    mocks.getConfig.mockResolvedValue(config)
    mocks.getCurrentUser.mockResolvedValue({ id: 42, username: "user" })
    mocks.buildScope.mockReturnValue("scope:server:user")
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
    mocks.getConfig.mockResolvedValue({ ...config, authMode: "multi-user" })
    mocks.getCurrentUser.mockResolvedValue({ id: 84, username: "resolved" })

    await expect(resolveServicePromptScope()).resolves.toEqual({
      config: { ...config, authMode: "multi-user" },
      scopeKey: "scope:server:user"
    })
    expect(mocks.getCurrentUser).toHaveBeenCalledTimes(1)
    expect(mocks.buildScope).toHaveBeenCalledWith(
      { ...config, authMode: "multi-user" },
      { userId: 84 }
    )
    expect(mocks.buildScope.mock.calls.flat().join(" ")).not.toContain(
      "user:anonymous"
    )

    vi.clearAllMocks()
    mocks.isHosted.mockReturnValue(true)
    mocks.getConfig.mockResolvedValue(config)
    mocks.getCurrentUser.mockResolvedValue({ id: 85, username: "hosted" })
    mocks.buildScope.mockReturnValue("scope:hosted:user")

    await resolveServicePromptScope()
    expect(mocks.getCurrentUser).toHaveBeenCalledTimes(1)
    expect(mocks.buildScope).toHaveBeenCalledWith(config, { userId: 85 })
  })

  it("does not resolve a user for single-user scope", async () => {
    await resolveServicePromptScope()

    expect(mocks.getCurrentUser).not.toHaveBeenCalled()
    expect(mocks.buildScope).toHaveBeenCalledWith(config, { userId: null })
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

    expect(snapshot).toEqual({
      scopeKey: "scope:server:user",
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
    expect(mocks.getServicePrompt).not.toHaveBeenCalled()
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

    await expect(
      loadServicePromptSnapshot(["chat.web_search.answer"])
    ).resolves.toMatchObject({ capability: "supported" })
    expect(mocks.getServicePrompt).toHaveBeenCalledWith(
      "chat.web_search.answer",
      expect.objectContaining({ signal: expect.any(AbortSignal) })
    )
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
    await pending

    mocks.getServicePrompt.mockImplementation(async (id: KnownServicePromptId) =>
      detailFor(id)
    )
    await loadServicePromptSnapshot([
      "chat.rag.answer",
      "chat.rag.question_rewrite"
    ])
    expect(mocks.listServicePrompts).toHaveBeenCalledTimes(2)
    expect(mocks.getServicePrompt).toHaveBeenCalledTimes(4)
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
  })

  it("propagates invocation aborts and aborts old reads on config changes", async () => {
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

    const second = loadServicePromptSnapshot(["chat.rag.answer"])
    await vi.waitFor(() => expect(detailSignal?.aborted).toBe(false))
    window.dispatchEvent(new Event("tldw:config-updated"))
    await expect(second).rejects.toMatchObject({ name: "AbortError" })
    expect(detailSignal?.aborted).toBe(true)
  })

  it("aborts when hosted credentials change during principal resolution", async () => {
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

    await expect(pending).rejects.toMatchObject({ name: "AbortError" })
    expect(mocks.listServicePrompts).not.toHaveBeenCalled()
  })

  it("aborts during an unabortable catalog-404 compatibility getter", async () => {
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
    window.dispatchEvent(new Event("tldw:config-updated"))
    resolvePrompts({
      ragPrompt: "legacy {context} {question}",
      ragQuestionPrompt: "legacy {chat_history} {question}"
    })

    await expect(pending).rejects.toMatchObject({ name: "AbortError" })
  })

  it("aborts during the supported raw migration probe", async () => {
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

    await expect(pending).rejects.toMatchObject({ name: "AbortError" })
    expect(mocks.getServicePrompt).not.toHaveBeenCalled()
  })

  it("aborts an active read when the cross-context config watcher fires", async () => {
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

    await expect(pending).rejects.toMatchObject({ name: "AbortError" })
    expect(watched?.tldwConfig).toBeTypeOf("function")
    expect(mocks.localUnwatch).toHaveBeenCalledWith(watched)
  })

  it("rehydrates client config at the start of every invocation", async () => {
    const nextConfig = { ...config, serverUrl: "https://next.example" }
    let liveConfig = config
    mocks.getConfig.mockImplementation(async () => liveConfig)
    mocks.initialize.mockImplementation(async () => {
      if (mocks.initialize.mock.calls.length === 2) {
        liveConfig = nextConfig
      }
    })
    mocks.buildScope.mockImplementation((cfg: typeof config) =>
      `scope:${cfg.serverUrl}`
    )

    const first = await loadServicePromptSnapshot(["chat.rag.answer"])
    const second = await loadServicePromptSnapshot(["chat.rag.answer"])

    expect(mocks.initialize).toHaveBeenCalledTimes(2)
    expect(first.scopeKey).toBe("scope:https://server.example")
    expect(second.scopeKey).toBe("scope:https://next.example")
  })

  it("treats an initialize-time config normalization write as scope cancellation", async () => {
    mocks.initialize.mockImplementation(async () => {
      const watched = mocks.localWatch.mock.calls.at(-1)?.[0] as
        | { tldwConfig?: (change: { newValue?: unknown }) => void }
        | undefined
      watched?.tldwConfig?.({ newValue: config })
    })

    await expect(
      loadServicePromptSnapshot(["chat.rag.answer"])
    ).rejects.toMatchObject({ name: "AbortError" })
    expect(mocks.getConfig).not.toHaveBeenCalled()
  })

  it("aborts when the hosted principal changes before a supported return", async () => {
    mocks.isHosted.mockReturnValue(true)
    mocks.getCurrentUser
      .mockResolvedValueOnce({ id: 42, username: "first" })
      .mockResolvedValueOnce({ id: 84, username: "second" })
    mocks.buildScope.mockImplementation(
      (_cfg: typeof config, options?: { userId?: number }) =>
        `scope:user:${options?.userId}`
    )

    await expect(
      loadServicePromptSnapshot(["chat.rag.answer"])
    ).rejects.toMatchObject({ name: "AbortError" })
    expect(mocks.getCurrentUser).toHaveBeenCalledTimes(2)
  })

  it("confirms hosted principal before legacy return and migration-required errors", async () => {
    mocks.isHosted.mockReturnValue(true)
    mocks.buildScope.mockImplementation(
      (_cfg: typeof config, options?: { userId?: number }) =>
        `scope:user:${options?.userId}`
    )
    mocks.listServicePrompts.mockRejectedValueOnce(
      new ServicePromptApiError("Not found", { status: 404 })
    )
    mocks.getCurrentUser
      .mockResolvedValueOnce({ id: 42, username: "first" })
      .mockResolvedValueOnce({ id: 84, username: "second" })

    await expect(
      loadServicePromptSnapshot(["chat.rag.answer"])
    ).rejects.toMatchObject({ name: "AbortError" })

    vi.clearAllMocks()
    mocks.initialize.mockResolvedValue(undefined)
    mocks.isHosted.mockReturnValue(true)
    mocks.getConfig.mockResolvedValue(config)
    mocks.buildScope.mockImplementation(
      (_cfg: typeof config, options?: { userId?: number }) =>
        `scope:user:${options?.userId}`
    )
    mocks.listServicePrompts.mockResolvedValue(catalog)
    mocks.localGet.mockImplementation(async (key: string) =>
      key === "systemPromptForRag"
        ? "legacy {context} {question}"
        : undefined
    )
    mocks.syncGet.mockResolvedValue(undefined)
    mocks.getCurrentUser
      .mockResolvedValueOnce({ id: 42, username: "first" })
      .mockResolvedValueOnce({ id: 84, username: "second" })

    await expect(
      loadServicePromptSnapshot(["chat.rag.answer"])
    ).rejects.toMatchObject({ name: "AbortError" })
  })

  it("always removes the per-invocation config watcher", async () => {
    await loadServicePromptSnapshot(["chat.rag.answer"])

    const watched = mocks.localWatch.mock.calls[0]?.[0]
    expect(watched).toBeDefined()
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

/**
 * Phase 7 Character Chat readiness and SEND gating signoff.
 *
 * This suite intentionally drives the real WebUI against a real FastAPI
 * backend. It may observe network traffic, but it must not mock or fulfill
 * successful Character Chat responses.
 */
import { type Page, type Request } from "@playwright/test"
import { test, expect, skipIfServerUnavailable } from "../../utils/fixtures"
import { captureAllApiCalls, expectNoApiCall } from "../../utils/api-assertions"
import { ChatPage } from "../../utils/page-objects"
import { fetchWithApiKey, TEST_CONFIG, waitForConnection } from "../../utils/helpers"
import { captureStreamedResponseBody, waitForStreamComplete } from "../../utils/journey-helpers"

import { readStore } from "../../../../extension/tests/e2e/utils/history-selection"
import type { HistoryBookmark, HistoryTurnRecovery } from "../../../../packages/ui/src/db/dexie/types"


type CanonicalTurnMessage = { id: string; conversation_id: string; content: string; sender: string; parent_message_id: string | null }

type ApiResult<T = unknown> = {
  ok: boolean
  status: number
  body: T | null
  path: string
}

type ModelDescriptor = Record<string, unknown>

type CharacterRecord = {
  id: number | null
  name: string
  version: number | null
}

type BlockedModelScenario = {
  label: string
  modelKey: string
  expectedReadiness: RegExp
  expectedStatus: RegExp
  expectedSelector: RegExp
}

const NATIVE_COMPLETION_PATH = /^\/api\/v1\/chat\/completions$/

const LOCAL_OR_SIMULATION_RISK_PROVIDERS = new Set([
  "local",
  "local-llm",
  "llamafile",
  "llama",
  "llamacpp",
  "llama.cpp",
  "lmstudio",
  "mlx",
  "ollama",
  "ollama2",
  "tabbyapi",
  "vllm",
  "custom",
  "custom_openai",
  "custom_openai_api",
  "custom-openai",
  "custom-openai-api",
  "customopenai",
  "tldw",
])

const normalizeServerUrl = (value: string): string => {
  const trimmed = value.trim().replace(/\/$/, "")
  if (/^https?:\/\//i.test(trimmed)) return trimmed
  return `http://${trimmed}`
}

const serverUrl = (): string =>
  normalizeServerUrl(
    process.env.TLDW_E2E_SERVER_URL ||
      process.env.TLDW_SERVER_URL ||
      TEST_CONFIG.serverUrl,
  )

const apiKey = (): string =>
  process.env.TLDW_E2E_API_KEY ||
  process.env.TLDW_API_KEY ||
  TEST_CONFIG.apiKey

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value)

const toArray = (value: unknown): unknown[] => {
  if (Array.isArray(value)) return value
  if (isRecord(value) && Array.isArray(value.models)) return value.models
  if (isRecord(value) && Array.isArray(value.providers)) return value.providers
  if (isRecord(value) && Array.isArray(value.items)) return value.items
  return []
}

const readNested = (value: unknown, key: string): unknown => {
  if (!isRecord(value)) return undefined
  return value[key] ?? (isRecord(value.details) ? value.details[key] : undefined) ??
    (isRecord(value.metadata) ? value.metadata[key] : undefined)
}

const readString = (value: unknown, keys: string[]): string | null => {
  for (const key of keys) {
    const field = readNested(value, key)
    if (typeof field === "string" || typeof field === "number") {
      const trimmed = String(field).trim()
      if (trimmed) return trimmed
    }
  }
  return null
}

const readBoolean = (value: unknown, keys: string[]): boolean | null => {
  for (const key of keys) {
    const field = readNested(value, key)
    if (typeof field === "boolean") return field
  }
  return null
}

const readStringList = (value: unknown, keys: string[]): string[] => {
  for (const key of keys) {
    const field = readNested(value, key)
    if (Array.isArray(field)) {
      return field
        .map((item) =>
          typeof item === "string" || typeof item === "number"
            ? String(item).trim().toLowerCase()
            : "",
        )
        .filter(Boolean)
    }
    if (typeof field === "string" || typeof field === "number") {
      const trimmed = String(field).trim().toLowerCase()
      if (trimmed) return [trimmed]
    }
  }
  return []
}

const normalizeProvider = (value: string | null): string | null => {
  if (!value) return null
  const normalized = value.trim().toLowerCase()
  if (!normalized) return null
  if (normalized === "llama.cpp") return "llamacpp"
  if (normalized === "local-llm") return "local"
  return normalized
}

const normalizeModelId = (value: string | null): string | null => {
  if (!value) return null
  const trimmed = value.trim().replace(/^tldw:/i, "")
  return trimmed.length > 0 ? trimmed : null
}

const descriptorProvider = (descriptor: ModelDescriptor): string | null =>
  normalizeProvider(
    readString(descriptor, [
      "provider",
      "provider_key",
      "providerKey",
      "api_provider",
      "apiProvider",
    ]),
  )

const descriptorModelId = (descriptor: ModelDescriptor): string | null =>
  normalizeModelId(
    readString(descriptor, ["model", "model_id", "id", "name"]),
  )

const formatModelKey = (descriptor: ModelDescriptor): string | null => {
  const modelId = descriptorModelId(descriptor)
  if (!modelId) return null
  const provider = descriptorProvider(descriptor)
  const serverModelKey = !provider
    ? modelId
    : modelId.toLowerCase().startsWith(`${provider}:`)
      ? modelId
      : `${provider}:${modelId}`
  return serverModelKey.toLowerCase().startsWith("tldw:")
    ? serverModelKey
    : `tldw:${serverModelKey}`
}

const isChatTextDescriptor = (descriptor: ModelDescriptor): boolean => {
  const types = readStringList(descriptor, ["type", "model_type", "modelType"])
  if (types.length > 0 && !types.includes("chat")) return false

  const outputModalities = readStringList(descriptor, [
    "output_modality",
    "outputModalities",
    "output_modalities",
    "modalities_output",
  ])
  if (outputModalities.length > 0 && !outputModalities.includes("text")) {
    return false
  }

  return Boolean(descriptorModelId(descriptor))
}

const statusField = (descriptor: ModelDescriptor): string | null =>
  readString(descriptor, ["status", "state"])?.toLowerCase() ?? null

const isProviderUnconfiguredDescriptor = (
  descriptor: ModelDescriptor,
): boolean => {
  const configured = readBoolean(descriptor, [
    "is_configured",
    "isConfigured",
    "configured",
  ])
  const providerConfigured = readBoolean(descriptor, [
    "provider_is_configured",
    "providerIsConfigured",
    "provider_configured",
    "providerConfigured",
  ])
  const apiKeyRequired = readBoolean(descriptor, [
    "api_key_required",
    "apiKeyRequired",
    "requires_api_key",
    "requiresApiKey",
  ])
  const apiKeyConfigured = readBoolean(descriptor, [
    "api_key_configured",
    "apiKeyConfigured",
    "has_api_key",
    "hasApiKey",
  ])
  const status = statusField(descriptor)

  return (
    configured === false ||
    providerConfigured === false ||
    (apiKeyRequired === true && apiKeyConfigured === false) ||
    status === "unconfigured" ||
    status === "not_configured"
  )
}

const isModelUnavailableDescriptor = (descriptor: ModelDescriptor): boolean => {
  const catalogOnly = readBoolean(descriptor, [
    "catalog_only",
    "catalogOnly",
    "is_catalog_only",
    "isCatalogOnly",
  ])
  const deprecated = readBoolean(descriptor, [
    "deprecated",
    "is_deprecated",
    "isDeprecated",
  ])
  const available = readBoolean(descriptor, [
    "available",
    "is_available",
    "isAvailable",
    "enabled",
    "active",
  ])
  const status = statusField(descriptor)

  return (
    catalogOnly === true ||
    deprecated === true ||
    available === false ||
    status === "catalog_only" ||
    status === "disabled" ||
    status === "inactive" ||
    status === "unavailable" ||
    status === "not_available" ||
    status === "deprecated"
  )
}

const isUsableChatDescriptor = (descriptor: ModelDescriptor): boolean =>
  isChatTextDescriptor(descriptor) &&
  !isProviderUnconfiguredDescriptor(descriptor) &&
  !isModelUnavailableDescriptor(descriptor)

async function fetchJson<T = unknown>(
  path: string,
  init: RequestInit = {},
): Promise<ApiResult<T>> {
  const headers: Record<string, string> = {
    "content-type": "application/json",
  }
  const response = await fetchWithApiKey(`${serverUrl()}${path}`, apiKey(), {
    ...init,
    headers,
  })
  const body = (await response.json().catch(() => null)) as T | null
  return {
    ok: response.ok,
    status: response.status,
    body,
    path,
  }
}

async function fetchModelDescriptors(): Promise<ModelDescriptor[]> {
  const metadata = await fetchJson(
    "/api/v1/llm/models/metadata?type=chat&output_modality=text",
  ).catch(() => null)
  const descriptors = toArray(metadata?.body)
    .filter(isRecord)
    .map((entry) => ({ ...entry }))

  const providers = await fetchJson("/api/v1/llm/providers").catch(() => null)
  const providerEntries = toArray(providers?.body).filter(isRecord)
  for (const provider of providerEntries) {
    const providerName =
      readString(provider, ["name", "id", "provider", "provider_key"]) ?? null
    const providerModels = Array.isArray(provider.models) ? provider.models : []
    for (const model of providerModels) {
      if (!isRecord(model)) continue
      descriptors.push({
        ...model,
        provider: readString(model, ["provider", "provider_key"]) ?? providerName,
        provider_is_configured:
          readBoolean(model, [
            "provider_is_configured",
            "providerIsConfigured",
            "provider_configured",
            "providerConfigured",
          ]) ??
          readBoolean(provider, [
            "provider_is_configured",
            "providerIsConfigured",
            "configured",
            "is_configured",
          ]) ??
          undefined,
      })
    }
  }

  const seen = new Set<string>()
  return descriptors.filter((descriptor) => {
    if (!isChatTextDescriptor(descriptor)) return false
    const key = `${descriptorProvider(descriptor) ?? "unknown"}:${descriptorModelId(descriptor)}`
    if (seen.has(key)) return false
    seen.add(key)
    return true
  })
}

async function findBlockedModelScenario(): Promise<BlockedModelScenario | null> {
  const descriptors = await fetchModelDescriptors()
  const providerBlocked = descriptors.find(isProviderUnconfiguredDescriptor)
  if (providerBlocked) {
    const modelKey = formatModelKey(providerBlocked)
    if (modelKey) {
      return {
        label: "provider-unconfigured model advertised by real backend",
        modelKey,
        expectedReadiness:
          /configure the selected model provider|provider setup|model setup|choose an available chat model|choose a model/i,
        expectedStatus:
          /model setup needed|provider setup|model setup|choose an available chat model|choose a model|model unavailable/i,
        expectedSelector:
          /configure the selected model provider|provider setup needed|model setup|not configured|choose an available chat model|choose a model|model unavailable/i,
      }
    }
  }

  const unavailable = descriptors.find(isModelUnavailableDescriptor)
  if (unavailable) {
    const modelKey = formatModelKey(unavailable)
    if (modelKey) {
      return {
        label: "not-callable model advertised by real backend",
        modelKey,
        expectedReadiness: /not callable|choose or configure a callable chat model/i,
        expectedStatus: /not callable|model unavailable/i,
        expectedSelector: /not callable|model unavailable/i,
      }
    }
  }

  const usableModels = descriptors.filter(isUsableChatDescriptor)
  if (descriptors.length === 0 || usableModels.length === 0) {
    return {
      label: "real backend has no usable chat models",
      modelKey: "openai:gpt-4o",
      expectedReadiness: /configure a chat model|choose a chat model|open model settings/i,
      expectedStatus: /configure a chat model|no chat models|model setup|no model/i,
      expectedSelector: /no chat models|model unavailable|openai:gpt-4o|api \/ model/i,
    }
  }

  return null
}

async function findCallableModelForSuccess(): Promise<string | null> {
  const explicit = process.env.TLDW_E2E_CHARACTER_CALLABLE_MODEL?.trim()
  if (explicit) return explicit

  const allowLocal = process.env.TLDW_E2E_ALLOW_LOCAL_PROVIDER_SUCCESS === "1"
  const descriptors = await fetchModelDescriptors()
  const usable = descriptors.filter(isUsableChatDescriptor)

  for (const descriptor of usable) {
    const provider = descriptorProvider(descriptor)
    const modelKey = formatModelKey(descriptor)
    if (!modelKey) continue
    if (!provider || !LOCAL_OR_SIMULATION_RISK_PROVIDERS.has(provider)) {
      return modelKey
    }
    if (allowLocal) {
      return modelKey
    }
  }

  return null
}

async function createCharacterViaApi(): Promise<CharacterRecord> {
  const name = `E2E Phase7 Roleplay ${Date.now()} ${Math.random()
    .toString(36)
    .slice(2, 7)}`
  const payload = {
    name,
    description: "Real-backend E2E character for Phase 7 readiness verification.",
    personality: "Precise and brief.",
    scenario: "The user is validating Character Chat readiness gating.",
    system_prompt:
      "You are a Phase 7 E2E role-play character. Keep answers short.",
    first_message: "Ready for the Phase 7 check.",
    tags: ["e2e", "phase7"],
  }

  const first = await fetchJson<Record<string, unknown>>("/api/v1/characters/", {
    method: "POST",
    body: JSON.stringify(payload),
  }).catch((error) => ({
    ok: false,
    status: 0,
    body: { detail: String(error) },
    path: "/api/v1/characters/",
  }))
  const result = first.ok
    ? first
    : await fetchJson<Record<string, unknown>>("/api/v1/characters", {
        method: "POST",
        body: JSON.stringify(payload),
      }).catch((error) => ({
        ok: false,
        status: 0,
        body: { detail: String(error) },
        path: "/api/v1/characters",
      }))

  test.skip(
    !result.ok,
    `Character create API unavailable at ${result.path}: status ${result.status} ${JSON.stringify(
      result.body,
    )}`,
  )

  const resultBody = isRecord(result.body)
    ? (result.body as Record<string, unknown>)
    : null
  const id = typeof resultBody?.id === "number" ? resultBody.id : null
  const version =
    typeof resultBody?.version === "number" ? resultBody.version : null
  return { id, name, version }
}

async function deleteCharacterViaApi(character: CharacterRecord): Promise<void> {
  if (character.id == null) return
  let expectedVersion = character.version
  if (expectedVersion == null) {
    const current = await fetchJson<Record<string, unknown>>(
      `/api/v1/characters/${character.id}`,
    ).catch(() => null)
    expectedVersion =
      isRecord(current?.body) && typeof current.body.version === "number"
        ? current.body.version
        : null
  }
  if (expectedVersion == null) return

  await fetchJson(
    `/api/v1/characters/${character.id}?expected_version=${expectedVersion}`,
    { method: "DELETE" },
  ).catch(() => null)
}

async function seedSelectedModel(page: Page, modelKey: string): Promise<void> {
  await page.addInitScript((selectedModel) => {
    try {
      localStorage.setItem("selectedModel", selectedModel)
      localStorage.setItem(
        "plasmo-storage-selectedModel",
        JSON.stringify(selectedModel),
      )
      localStorage.setItem(
        "chatModelUsageByProviderModel",
        JSON.stringify({
          [selectedModel]: {
            selectedCount: 1,
            lastSelectedAt: Date.now(),
          },
        }),
      )
    } catch {}
  }, modelKey)
}

async function openCharacterChatWithCharacter(
  page: Page,
  characterName: string,
  modelKey: string,
): Promise<ChatPage> {
  await seedSelectedModel(page, modelKey)
  await page.goto("/chat?mode=character", { waitUntil: "domcontentloaded" })
  await waitForConnection(page)

  await expect(page.getByTestId("playground-active-chat-mode")).toContainText(
    /Character Chat/i,
    { timeout: 30_000 },
  )

  const chatPage = new ChatPage(page)
  await chatPage.waitForReady()
  await chatPage.selectCharacter(characterName)
  return chatPage
}

async function expectSelectedCharacter(
  page: Page,
  characterName: string,
): Promise<void> {
  const selector = page.getByTestId("character-select").first()
  await expect
    .poll(
      async () => {
        const label =
          (await selector.getAttribute("aria-label").catch(() => null)) ||
          (await selector.getAttribute("title").catch(() => null)) ||
          (await selector.textContent().catch(() => null)) ||
          ""
        return label.includes(characterName)
      },
      {
        timeout: 10_000,
        message: `Expected selected character ${characterName} to remain active`,
      },
    )
    .toBe(true)
}

function nativeCompletionCallPredicate(url: string, method = "POST"): boolean {
  const parsed = new URL(url)
  return method === "POST" && NATIVE_COMPLETION_PATH.test(parsed.pathname)
}

async function clickPrimaryComposerAction(page: Page, name: RegExp): Promise<void> {
  const input = page.getByPlaceholder(/type a message/i).first()
  const composerForm = page.locator("form").filter({ has: input }).last()
  await composerForm.getByRole("button", { name }).first().click()
}

test.describe("Character Chat Phase 7 real-backend readiness", () => {
  test("blocks Character Chat SEND for a real unusable/no-provider model without calling complete-v2", async ({
    authedPage: page,
    serverInfo,
  }) => {
    skipIfServerUnavailable(serverInfo)

    const scenario = await findBlockedModelScenario()
    test.skip(
      !scenario,
      "Real backend exposes only usable chat models; no no-provider/unusable model state is available for send-gating verification.",
    )
    if (!scenario) return

    const character = await createCharacterViaApi()
    try {
      await test.step(`Open Character Chat with ${scenario.label}`, async () => {
        await openCharacterChatWithCharacter(page, character.name, scenario.modelKey)
      })

      await test.step("Verify all visible readiness surfaces agree on blocked setup state", async () => {
        const readiness = page.getByTestId("character-chat-readiness-panel")
        await expect(readiness).toBeVisible({ timeout: 30_000 })
        await expect(readiness).toContainText(scenario.expectedReadiness)

        await expect(readiness).toHaveAttribute("role", "status")
        await expect(readiness).toHaveAttribute(
          "aria-label",
          "Character Chat setup status",
        )
        await expect(readiness).toContainText(scenario.expectedStatus)

        const modelSelector = page.getByTestId("model-selector").first()
        await expect(modelSelector).toHaveAttribute(
          "aria-label",
          scenario.expectedSelector,
        )

        const input = page.getByPlaceholder(/type a message/i).first()
        const composerForm = page.locator("form").filter({ has: input }).last()
        await expect(
          composerForm.getByRole("button", { name: /open model settings/i }).first(),
        ).toBeVisible()
      })

      await test.step("Click blocked primary action and verify draft/character are preserved", async () => {
        const draft = "Phase 7 send gating should keep this draft."
        const input = page.getByPlaceholder(/type a message/i).first()
        await input.fill(draft)

        const noCompleteV2 = expectNoApiCall(
          page,
          {
            method: "POST",
            url: NATIVE_COMPLETION_PATH,
          },
          1_500,
        )

        await clickPrimaryComposerAction(page, /open model settings/i)
        await noCompleteV2

        await expect(input).toHaveValue(draft)
        await expectSelectedCharacter(page, character.name)
      })
    } finally {
      await deleteCharacterViaApi(character)
    }
  })

  test("shows model-settings recovery for a real backend provider/configuration failure", async ({ authedPage: page, serverInfo }, testInfo) => {
    skipIfServerUnavailable(serverInfo)
    const failureModel = process.env.TLDW_E2E_CHARACTER_PROVIDER_FAILURE_MODEL?.trim()
    test.skip(!failureModel, "Set TLDW_E2E_CHARACTER_PROVIDER_FAILURE_MODEL to exercise a real provider/configuration failure.")
    if (!failureModel) return
    const character = await createCharacterViaApi()
    const capture = captureAllApiCalls(page)
    try {
      await openCharacterChatWithCharacter(page, character.name, failureModel)
      const input = page.getByPlaceholder(/type a message/i).first()
      const question = "Trigger the real provider configuration failure."
      await input.fill(question)
      const completed = page.waitForResponse(response => nativeCompletionCallPredicate(response.url(), response.request().method()))
      await clickPrimaryComposerAction(page, /send/i)
      const response = await completed
      expect(response.status()).toBe(502)
      const sent = response.request().postDataJSON()
      const qualifiedModel = failureModel.replace(/^tldw:/, "")
      const separator = qualifiedModel.indexOf(":")
      expect(sent).toMatchObject({ model: separator >= 0 ? qualifiedModel.slice(separator + 1) : qualifiedModel,
        ...(separator >= 0 ? { api_provider: qualifiedModel.slice(0, separator) } : {}),
        stream: true, save_to_db: true, tldw_history_selection_v1: { version: 1 } })
      const chatId = sent.conversation_id
      expect(chatId).toEqual(expect.any(String))
      expect(sent.tldw_history_selection_v1.conversation_id).toBe(chatId)
      expect(sent.messages).toEqual([{ role: "user", content: question }])
      const safeMessage = "The selected provider credentials could not be authenticated."
      expect(await response.json()).toEqual({ detail: { error_code: "provider_authentication_failed", message: safeMessage } })
      await waitForStreamComplete(page, 90_000)
      const banner = page.getByTestId("playground-chat-error-banner")
      await expect(banner).toBeVisible()
      await expect(banner.getByRole("heading")).toHaveText("Character chat model setup needs attention.")
      const failedMessage = page.getByRole("log", { name: /chat messages/i }).getByRole("article").last()
      await failedMessage.getByRole("button", { name: "Show technical details", exact: true }).click()
      const technicalDetails = failedMessage.locator("pre")
      await expect(technicalDetails).toContainText("provider_authentication_failed")
      await expect(technicalDetails).toContainText(safeMessage)
      const pending = async () => ((await readStore(page, "historySelections")) as HistoryBookmark[])
        .filter(row => row.conversation_id === chatId).flatMap(row => Object.values(row.pending_turns ?? {})) as HistoryTurnRecovery[]
      await expect.poll(async () => (await pending()).length).toBe(1)
      const original = await pending()
      expect(original[0]).toMatchObject({ persistence: "server", state: "unknown", input_text: question, result_text: "", conversation_id: chatId })
      expect(original[0].admission).toBeUndefined()
      expect(original[0].input_id).toBeUndefined()
      expect(original[0].assistant_id).toBeUndefined()
      await expect(page.getByText("User admission outcome unknown", { exact: true })).toBeVisible()
      const canonical = await fetchJson<{ messages: CanonicalTurnMessage[] }>(`/api/v1/chats/${chatId}/messages?render_placeholders=false`)
      expect(canonical.ok).toBe(true)
      expect(canonical.body!.messages).toHaveLength(1)
      expect(canonical.body!.messages[0]).toMatchObject({ content: question, sender: "user", conversation_id: chatId, parent_message_id: null })
      const noRetry = expectNoApiCall(page, { method: "POST", url: NATIVE_COMPLETION_PATH }, 500)
      await banner.getByRole("button", { name: "Retry chat", exact: true }).click()
      await noRetry
      expect(await pending()).toEqual(original)
      const recoveryDraft = "Continue this character chat after configuring the provider."
      await input.fill(recoveryDraft)
      await banner.getByRole("button", { name: "Edit provider", exact: true }).click()
      const settings = page.getByRole("dialog", { name: "Current Chat Model Settings", exact: true })
      await expect(settings).toBeVisible()
      await settings.getByRole("button", { name: "Close", exact: true }).click()
      await expect(settings).toBeHidden()
      await expect(input).toHaveValue(recoveryDraft)
      await expectSelectedCharacter(page, character.name)
      expect(await pending()).toEqual(original)
      await page.reload({ waitUntil: "domcontentloaded" })
      await waitForConnection(page)
      await expect(page.getByText("User admission outcome unknown", { exact: true })).toBeVisible()
      expect(await pending()).toEqual(original)
      const calls = (await capture.stop()).filter(call => nativeCompletionCallPredicate(call.url, call.method))
      expect(calls).toHaveLength(1)
      await testInfo.attach("native-character-provider-recovery.json", { body: JSON.stringify({ request: sent, recovery: original, canonical: canonical.body!.messages }, null, 2), contentType: "application/json" })
    } finally { await capture.stop(); await deleteCharacterViaApi(character) }
  })

  test("saves a versioned native turn when a real callable character model is available", async ({ authedPage: page, serverInfo }, testInfo) => {
    skipIfServerUnavailable(serverInfo)
    const callableModel = await findCallableModelForSuccess()
    test.skip(!callableModel, "No trustworthy callable chat model is configured.")
    if (!callableModel) return
    const character = await createCharacterViaApi()
    const completionRequests: Request[] = []
    const captureCompletion = (request: Request) => {
      if (nativeCompletionCallPredicate(request.url(), request.method())) completionRequests.push(request)
    }
    page.on("request", captureCompletion)
    try {
      await openCharacterChatWithCharacter(page, character.name, callableModel)
      await expect(page.getByTestId("character-chat-readiness-panel")).toBeHidden()
      const question = "Reply with one short sentence for Phase 7."
      const answer = "onboarding UAT ready. The mock provider returned a deterministic success response."
      const chat = new ChatPage(page)
      const readCompletionBody = await captureStreamedResponseBody(page, "/api/v1/chat/completions")
      const completed = page.waitForResponse(response => nativeCompletionCallPredicate(response.url(), response.request().method()))
      await page.getByPlaceholder(/type a message/i).first().fill(question)
      await clickPrimaryComposerAction(page, /send/i)
      const completion = await completed
      expect(completion.status()).toBe(200)
      const sent = completion.request().postDataJSON()
      const qualifiedModel = callableModel.replace(/^tldw:/, "")
      const separator = qualifiedModel.indexOf(":")
      expect(sent).toMatchObject({ model: separator >= 0 ? qualifiedModel.slice(separator + 1) : qualifiedModel,
        ...(separator >= 0 ? { api_provider: qualifiedModel.slice(0, separator) } : {}),
        stream: true, save_to_db: true, tldw_history_selection_v1: { version: 1 } })
      const chatId = sent.conversation_id
      expect(chatId).toEqual(expect.any(String))
      expect(sent.tldw_history_selection_v1.conversation_id).toBe(chatId)
      expect(sent.messages).toEqual([{ role: "user", content: question }])
      await waitForStreamComplete(page, 90_000)
      await expect.poll(async () => (await chat.getMessages()).filter(row => row.role === "assistant").map(row => row.content)).toEqual([answer])

      const frames = (await readCompletionBody()).split("\n")
        .filter(line => line.startsWith("data: ") && line.slice(6) !== "[DONE]")
        .map(line => JSON.parse(line.slice(6)))
      const admission = frames.find(frame => frame.tldw_history_admission_v1)?.tldw_history_admission_v1
      const settlement = frames.find(frame => frame.tldw_message_id)
      expect(admission).toMatchObject({ version: 1, conversation_id: chatId,
        owner_key: sent.tldw_history_selection_v1.owner_key,
        selection_digest: sent.tldw_history_selection_v1.selection_digest,
        messages: sent.tldw_history_selection_v1.messages,
        originating_selection_revision: sent.tldw_history_selection_v1.selection_revision })
      expect(admission.owner_key).toMatch(/^native-history-v1:sha256:/)
      expect(admission.input_message_id).toEqual(expect.any(String))
      expect(admission.input_message_id).not.toBe("")
      expect(settlement).toMatchObject({ tldw_conversation_id: chatId,
        tldw_user_message_id: admission.input_message_id, tldw_message_id: expect.any(String) })
      expect(settlement.tldw_message_id).not.toBe("")
      expect(settlement.tldw_message_id).not.toBe(admission.input_message_id)
      expect(frames.map(frame => frame.choices?.[0]?.delta?.content ?? "").join("")).toBe(answer)
      let saved: CanonicalTurnMessage[] = []
      await expect.poll(async () => { saved = (await fetchJson<{ messages: CanonicalTurnMessage[] }>(`/api/v1/chats/${chatId}/messages?render_placeholders=false`)).body!.messages; return saved.length }).toBe(2)
      const user = saved.find(row => row.id === admission.input_message_id)!
      const assistant = saved.find(row => row.id === settlement.tldw_message_id)!
      expect(user).toMatchObject({ content: question, sender: "user", conversation_id: chatId, parent_message_id: null })
      expect(assistant).toMatchObject({ content: answer, conversation_id: chatId, parent_message_id: user.id })
      expect(["assistant", character.name.toLowerCase()]).toContain(assistant.sender.toLowerCase())
      expect(new Set(saved.map(row => row.id)).size).toBe(2)
      let bookmarks: HistoryBookmark[] = []
      await expect.poll(async () => {
        bookmarks = ((await readStore(page, "historySelections")) as HistoryBookmark[]).filter(row =>
          row.conversation_id === chatId && row.owner_key === admission.owner_key &&
          row.view.cursor.kind === "after_message" && row.view.cursor.message_id === assistant.id)
        return bookmarks.length
      }).toBe(1)
      const bookmark = bookmarks[0]
      expect(bookmark.view.interpretation).toEqual({ kind: "parent_graph_v1" })
      const reference = { profile_id: bookmark.profile_id, client_session_id: bookmark.client_session_id,
        owner_key: admission.owner_key, conversation_id: chatId, owner_kind: "native" }
      await page.goto("/chat?historySelection=" + encodeURIComponent(JSON.stringify(reference)), { waitUntil: "domcontentloaded" })
      await waitForConnection(page)
      await chat.waitForReady()
      await expect.poll(async () => (await chat.getMessages()).filter(row => row.role === "assistant").map(row => row.content)).toEqual([answer])
      await expect(page).toHaveURL(/\/chat$/)
      await page.reload({ waitUntil: "domcontentloaded" })
      await waitForConnection(page)
      await chat.waitForReady()
      await expect.poll(async () => (await chat.getMessages()).filter(row => row.role === "assistant").map(row => row.content)).toEqual([answer])
      expect((await fetchJson<{ messages: CanonicalTurnMessage[] }>(`/api/v1/chats/${chatId}/messages?render_placeholders=false`)).body!.messages).toEqual(saved)

      await expectSelectedCharacter(page, character.name)
      expect(completionRequests).toHaveLength(1)
      await testInfo.attach("native-character-canonical-turn.json", { body: JSON.stringify({ request: sent, admission, settlement, saved, selection: reference }, null, 2), contentType: "application/json" })
    } finally { page.removeListener("request", captureCompletion); await deleteCharacterViaApi(character) }
  })
})

/**
 * Phase 7 Character Chat readiness and SEND gating signoff.
 *
 * This suite intentionally drives the real WebUI against a real FastAPI
 * backend. It may observe network traffic, but it must not mock or fulfill
 * successful Character Chat responses.
 */
import { type Page, type Request } from "@playwright/test"
import { test, expect, skipIfServerUnavailable } from "../../utils/fixtures"
import { captureAllApiCalls } from "../../utils/api-assertions"
import { ChatPage } from "../../utils/page-objects"
import { fetchWithApiKey, TEST_CONFIG, waitForConnection } from "../../utils/helpers"
import { waitForStreamComplete } from "../../utils/journey-helpers"
import { parseRoleplayResumeAppendRequest, parseRoleplayResumeAppendReceipt, parseRoleplayResumeCompletionRequest, parseRoleplayResumeCompletionReceipt } from "../../../../packages/ui/src/services/tldw/roleplay-resume"

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

const ROLEPLAY_COMPLETION_PATH = /\/api\/v1\/chats\/[^/]+\/complete-v2$/

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

function roleplayCompletionCallPredicate(url: string, method = "POST"): boolean {
  const parsed = new URL(url)
  return method === "POST" && ROLEPLAY_COMPLETION_PATH.test(parsed.pathname)
}

async function clickPrimaryComposerAction(page: Page, name: RegExp): Promise<void> {
  const input = page.getByTestId("chat-input").first()
  const composerForm = page.locator("form").filter({ has: input }).last()
  await composerForm.getByRole("button", { name }).first().click()
}

test.describe("Character Chat Phase 7 real-backend readiness", () => {
  test("refuses tracked character setup for a real unusable model before native admission and keeps the draft", async ({ authedPage: page, serverInfo }) => {
    skipIfServerUnavailable(serverInfo)
    const scenario = await findBlockedModelScenario()
    test.skip(!scenario, "The real backend exposes no unusable model for setup-refusal verification.")
    if (!scenario) return
    const character = await createCharacterViaApi()
    const mutationAttempts: string[] = []
    const observeMutations = (request: Request) => {
      const path = new URL(request.url()).pathname
      if (request.method() === "POST" && (/\/api\/v1\/chats\/?$/.test(path) || /\/(?:roleplay\/messages|complete-v2|chat\/completions)$/.test(path))) mutationAttempts.push(path)
    }
    page.on("request", observeMutations)
    try {
      await seedSelectedModel(page, scenario.modelKey)
      await page.goto("/chat?mode=character", { waitUntil: "domcontentloaded" })
      await waitForConnection(page)
      const chat = new ChatPage(page)
      await chat.waitForReady()
      await expect(page.getByTestId("playground-active-chat-mode")).toContainText(/Character Chat/i)
      const input = await chat.getChatInput()
      const draft = "Phase 7 blocked setup should keep this draft."
      await input.fill(draft)
      await chat.openConversationControls()
      const selector = page.getByTestId("character-select").first()
      await expect(selector).toHaveAccessibleName("Select character or persona")
      await expect(page.getByTestId("model-selector").first()).toHaveAttribute("aria-label", scenario.expectedSelector)
      await selector.click()
      const picker = page.getByTestId("assistant-select-panel")
      const choice = picker.getByRole("button", { name: character.name, exact: true })
      await expect(choice).toBeVisible()
      await choice.click()
      await expect(page.getByText("Choose an available model with a configured provider before starting this character chat. Your current chat and draft are still here.", { exact: true })).toBeVisible()
      await expect(picker).toBeVisible()
      await expect(choice).toBeVisible()
      await expect(selector).toHaveAccessibleName("Select character or persona")
      await expect(input).toHaveValue(draft)
      expect(mutationAttempts).toEqual([])
      expect(await readStore(page, "historySelections")).toEqual([])
    } finally { page.off("request", observeMutations); await deleteCharacterViaApi(character) }
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
      const input = page.getByTestId("chat-input").first()
      const composer = page.locator("form").filter({ has: input }).last()
      const question = "Trigger the real provider configuration failure."
      await input.fill(question)
      const admitted = page.waitForResponse(response => response.request().method() === "POST" &&
        /\/api\/v1\/chats\/[^/]+\/roleplay\/messages$/.test(new URL(response.url()).pathname))
      const completed = page.waitForResponse(response => roleplayCompletionCallPredicate(response.url(), response.request().method()))
      await clickPrimaryComposerAction(page, /send/i)
      const appendResponse = await admitted
      expect(appendResponse.status()).toBe(201)
      const chatId = new URL(appendResponse.url()).pathname.split("/")[4]
      const originalInput = parseRoleplayResumeAppendRequest(appendResponse.request().postDataJSON())
      expect(originalInput.content).toBe(question)
      const admission = parseRoleplayResumeAppendReceipt(chatId, originalInput, await appendResponse.json())
      const response = await completed
      expect(response.status()).toBe(401)
      const sent = parseRoleplayResumeCompletionRequest(response.request().postDataJSON())
      expect(new URL(response.url()).pathname).toBe(`/api/v1/chats/${chatId}/complete-v2`)
      expect(sent).toMatchObject({ stream: false, save_to_db: true, include_character_context: true, append_user_message: false,
        tldw_roleplay_resume_v1: { input_message_id: admission.message_id, input_message_version: admission.message_version, expected: admission.fence } })
      const safeMessage = "The selected provider credentials could not be authenticated."
      expect(await response.json()).toEqual({ detail: { error: { code: "provider_authentication_failed", type: "provider_authentication_failed", message: safeMessage } } })
      await expect(page.getByText("Character chat model setup needs attention.", { exact: true })).toBeVisible()
      await expect(page.getByText("Reply save unconfirmed", { exact: true })).toBeVisible()
      await expect(page.getByRole("button", { name: "Retry original request", exact: true })).toBeEnabled()
      await expect(composer.getByRole("button", { name: "Send message", exact: true })).toBeEnabled()
      await expect(page.getByText("Reply save unconfirmed", { exact: true })).toBeVisible()
      const pending = async () => ((await readStore(page, "historySelections")) as HistoryBookmark[])
        .filter(row => row.conversation_id === chatId).flatMap(row => Object.values(row.pending_turns ?? {})) as HistoryTurnRecovery[]
      await expect.poll(async () => (await pending()).length).toBe(1)
      const original = (await pending())[0]
      expect(original).toMatchObject({ persistence: "server", state: "unknown", input_text: question, result_text: "", conversation_id: chatId,
        input_id: admission.message_id, roleplay_resume: { append: admission, intent: { assistant_message_id: sent.tldw_roleplay_resume_v1.assistant_message_id } } })
      expect(original.admission).toBeUndefined()
      expect(original.assistant_id).toBeUndefined()
      const canonical = await fetchJson<{ messages: CanonicalTurnMessage[] }>(`/api/v1/chats/${chatId}/messages?render_placeholders=false`)
      expect(canonical.ok).toBe(true)
      expect(canonical.body!.messages).toHaveLength(1)
      expect(canonical.body!.messages[0]).toMatchObject({ id: admission.message_id, content: question, sender: "user", conversation_id: chatId, parent_message_id: null })
      const retried = page.waitForResponse(reply => roleplayCompletionCallPredicate(reply.url(), reply.request().method()))
      await page.getByRole("button", { name: "Retry original request", exact: true }).click()
      const retry = await retried
      expect(retry.status()).toBe(401)
      expect(retry.request().postDataJSON()).toEqual(sent)
      await expect(page.getByText("Reply save unconfirmed", { exact: true })).toBeVisible()
      await expect(page.getByRole("button", { name: "Retry original request", exact: true })).toBeEnabled()
      await expect(composer.getByRole("button", { name: "Send message", exact: true })).toBeEnabled()
      await expect.poll(async () => (await pending())[0]?.state).toBe("unknown")
      expect((await pending())[0]).toMatchObject({ operation_id: original.operation_id, input_id: admission.message_id, roleplay_resume: original.roleplay_resume })
      const recoveryDraft = "Continue this character chat after configuring the provider."
      await input.fill(recoveryDraft)
      await new ChatPage(page).openConversationControls()
      await composer.getByRole("button", { name: "Chat Settings", exact: true }).click()
      const settings = page.getByRole("dialog", { name: "Current Chat Model Settings", exact: true })
      await expect(settings).toBeVisible()
      await settings.getByRole("button", { name: "Close", exact: true }).click()
      await expect(settings).toBeHidden()
      await expect(input).toHaveValue(recoveryDraft)
      await expectSelectedCharacter(page, character.name)
      expect((await pending())[0]).toMatchObject({ operation_id: original.operation_id, input_id: admission.message_id, roleplay_resume: original.roleplay_resume })
      const retained = (await pending())[0]
      await page.reload({ waitUntil: "domcontentloaded" })
      await waitForConnection(page)
      const retainedOutcomes = page.locator("summary").filter({ hasText: /^Earlier messages without a reply \(1\)$/ })
      const retainedPanel = page.locator("details").filter({ has: retainedOutcomes })
      await expect(retainedPanel).toHaveJSProperty("open", false)
      await retainedOutcomes.click()
      await expect(retainedPanel).toHaveJSProperty("open", true)
      await expect(page.getByText("Reply save unconfirmed", { exact: true })).toBeVisible()
      expect((await pending())[0]).toEqual(retained)
      const calls = await capture.stop()
      expect(calls.filter(call => roleplayCompletionCallPredicate(call.url, call.method))).toHaveLength(2)
      expect(calls.filter(call => call.method === "POST" && /\/roleplay\/messages$/.test(new URL(call.url).pathname))).toHaveLength(1)
      expect(calls.filter(call => call.method === "POST" && /\/api\/v1\/chat\/completions$/.test(new URL(call.url).pathname))).toHaveLength(0)
      expect((await fetchJson<{ messages: CanonicalTurnMessage[] }>(`/api/v1/chats/${chatId}/messages?render_placeholders=false`)).body!.messages).toEqual(canonical.body!.messages)
      await testInfo.attach("native-character-provider-recovery.json", { body: JSON.stringify({ request: sent, admission, recovery: retained, canonical: canonical.body!.messages }, null, 2), contentType: "application/json" })
    } finally {
      await capture.stop()
      await deleteCharacterViaApi(character)
    }
  })

  test("saves a versioned native turn when a real callable character model is available", async ({ authedPage: page, serverInfo }, testInfo) => {
    skipIfServerUnavailable(serverInfo)
    const callableModel = await findCallableModelForSuccess()
    test.skip(!callableModel, "No trustworthy callable chat model is configured.")
    if (!callableModel) return
    const character = await createCharacterViaApi()
    const completionRequests: Request[] = []
    const captureCompletion = (request: Request) => {
      if (roleplayCompletionCallPredicate(request.url(), request.method())) completionRequests.push(request)
    }
    page.on("request", captureCompletion)
    try {
      const capturedSelection = page.waitForResponse(response => response.request().method() === "POST" &&
        /\/api\/v1\/chat\/conversations\/[^/]+\/history\/selection$/.test(new URL(response.url()).pathname))
      await openCharacterChatWithCharacter(page, character.name, callableModel)
      await expect(page.getByTestId("character-chat-readiness-panel")).toBeHidden()
      const question = "Reply with one short sentence for Phase 7."
      const answer = "onboarding UAT ready. The mock provider returned a deterministic success response."
      const chat = new ChatPage(page)
      const admitted = page.waitForResponse(response => response.request().method() === "POST" &&
        /\/api\/v1\/chats\/[^/]+\/roleplay\/messages$/.test(new URL(response.url()).pathname))
      const completed = page.waitForResponse(response => roleplayCompletionCallPredicate(response.url(), response.request().method()))
      await page.getByTestId("chat-input").first().fill(question)
      await clickPrimaryComposerAction(page, /send/i)
      const completion = await completed
      expect(completion.status()).toBe(200)
      const appendResponse = await admitted
      expect(appendResponse.status()).toBe(201)
      const chatId = new URL(appendResponse.url()).pathname.split("/")[4]
      const originalSelection = await (await capturedSelection).json()
      expect(originalSelection.view).toMatchObject({ owner_key: expect.stringMatching(/^native-history-v1:sha256:/), conversation_id: chatId, cursor: { kind: "empty" } })
      expect(originalSelection.snapshot).toMatchObject({ owner_key: originalSelection.view.owner_key, conversation_id: chatId })
      expect(originalSelection.snapshot.source_digest).toMatch(/^sha256:[0-9a-f]{64}$/)
      const ownerKey = originalSelection.view.owner_key
      const originalInput = parseRoleplayResumeAppendRequest(appendResponse.request().postDataJSON())
      expect(originalInput.content).toBe(question)
      expect(originalInput.expected.tail).toEqual({ message_id: null, message_version: null })
      const admission = parseRoleplayResumeAppendReceipt(chatId, originalInput, await appendResponse.json())
      const sent = parseRoleplayResumeCompletionRequest(completion.request().postDataJSON())
      expect(sent).toMatchObject({ stream: false, save_to_db: true, include_character_context: true, append_user_message: false,
        tldw_roleplay_resume_v1: { input_message_id: admission.message_id, input_message_version: admission.message_version, expected: admission.fence } })
      const settlement = parseRoleplayResumeCompletionReceipt(chatId, sent, await completion.json())
      const qualifiedModel = callableModel.replace(/^tldw:/, "")
      const separator = qualifiedModel.indexOf(":")
      expect(settlement).toMatchObject({ character_id: character.id, saved: true, assistant_content: answer,
        ...(separator >= 0 ? { provider: qualifiedModel.slice(0, separator), model: qualifiedModel.slice(separator + 1) } : { model: qualifiedModel }) })
      await waitForStreamComplete(page, 90_000)
      await expect(page.getByTestId("character-chat-readiness-panel")).toBeHidden()
      expect((await chat.getMessages()).filter(row => row.role === "assistant").map(row => row.content)).toEqual([answer])
      let saved: CanonicalTurnMessage[] = []
      await expect.poll(async () => { saved = (await fetchJson<{ messages: CanonicalTurnMessage[] }>(`/api/v1/chats/${chatId}/messages?render_placeholders=false`)).body!.messages; return saved.length }).toBe(2)
      const user = saved.find(row => row.id === admission.message_id)!
      const assistant = saved.find(row => row.id === settlement.assistant_message_id)!
      expect(user).toMatchObject({ content: question, sender: "user", conversation_id: chatId, parent_message_id: null })
      expect(assistant).toMatchObject({ content: answer, conversation_id: chatId, parent_message_id: user.id })
      expect(["assistant", character.name.toLowerCase()]).toContain(assistant.sender.toLowerCase())
      expect(new Set(saved.map(row => row.id)).size).toBe(2)
      let bookmarks: HistoryBookmark[] = []
      await expect.poll(async () => {
        bookmarks = ((await readStore(page, "historySelections")) as HistoryBookmark[]).filter(row =>
          row.conversation_id === chatId && row.owner_key === ownerKey &&
          row.view.cursor.kind === "after_message" && row.view.cursor.message_id === assistant.id)
        return bookmarks.length
      }).toBe(1)
      const bookmark = bookmarks[0]
      expect(bookmark.view.interpretation).toEqual({ kind: "parent_graph_v1" })
      const reference = { profile_id: bookmark.profile_id, client_session_id: bookmark.client_session_id,
        owner_key: ownerKey, conversation_id: chatId, owner_kind: "native" }
      await page.goto("/chat?historySelection=" + encodeURIComponent(JSON.stringify(reference)), { waitUntil: "domcontentloaded" })
      await waitForConnection(page)
      await chat.waitForReady()
      await expect.poll(async () => (await chat.getMessages()).filter(row => row.role === "assistant").map(row => row.content)).toEqual([answer])
      // The handoff is consumed; a saved chat may now carry its own address (XP-05).
      await expect(page).toHaveURL(/\/chat(?:\?chat=[^&#]+)?$/)
      await expect(page).not.toHaveURL(/historySelection=/)
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

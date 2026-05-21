/**
 * Phase 7 Character Chat readiness and SEND gating signoff.
 *
 * This suite intentionally drives the real WebUI against a real FastAPI
 * backend. It may observe network traffic, but it must not mock or fulfill
 * successful Character Chat responses.
 */
import { type Page } from "@playwright/test"
import { test, expect, skipIfServerUnavailable } from "../../utils/fixtures"
import { captureAllApiCalls, expectNoApiCall } from "../../utils/api-assertions"
import { ChatPage } from "../../utils/page-objects"
import { fetchWithApiKey, TEST_CONFIG, waitForConnection } from "../../utils/helpers"
import { waitForStreamComplete } from "../../utils/journey-helpers"

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

const COMPLETE_V2_PATH = /^\/api\/v1\/chats\/[^/]+\/complete-v2$/

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
        expectedReadiness: /configure the selected model provider|provider setup|model setup/i,
        expectedStatus: /model setup needed|provider setup|model setup/i,
        expectedSelector:
          /configure the selected model provider|provider setup needed|model setup|not configured/i,
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

  const id =
    isRecord(result.body) && typeof result.body.id === "number"
      ? result.body.id
      : null
  const version =
    isRecord(result.body) && typeof result.body.version === "number"
      ? result.body.version
      : null
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

function completeV2CallPredicate(url: string, method = "POST"): boolean {
  const parsed = new URL(url)
  return method === "POST" && COMPLETE_V2_PATH.test(parsed.pathname)
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

        const status = page.getByRole("status", { name: "Chat status" })
        await expect(status).toContainText(scenario.expectedStatus)

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
            url: /\/api\/v1\/chats\/[^/]+\/complete-v2$/,
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

  test("shows model-settings recovery for a real backend provider/configuration failure", async ({
    authedPage: page,
    serverInfo,
  }) => {
    skipIfServerUnavailable(serverInfo)

    const failureModel =
      process.env.TLDW_E2E_CHARACTER_PROVIDER_FAILURE_MODEL?.trim()
    test.skip(
      !failureModel,
      "Set TLDW_E2E_CHARACTER_PROVIDER_FAILURE_MODEL to a real model that passes frontend readiness but returns a provider/configuration failure.",
    )
    if (!failureModel) return

    const character = await createCharacterViaApi()
    try {
      await openCharacterChatWithCharacter(page, character.name, failureModel)

      const input = page.getByPlaceholder(/type a message/i).first()
      await input.fill("Trigger the real provider configuration failure.")

      const completeResponse = page.waitForResponse(
        (response) =>
          completeV2CallPredicate(response.url(), response.request().method()),
        { timeout: 90_000 },
      )
      const capture = captureAllApiCalls(page)

      await clickPrimaryComposerAction(page, /send/i)

      const response = await completeResponse
      const calls = await capture.stop()
      const completeCall = calls.find((call) =>
        completeV2CallPredicate(call.url, call.method),
      )

      expect(response.status()).toBeGreaterThanOrEqual(400)
      expect(completeCall).toBeTruthy()
      expect(JSON.stringify(completeCall?.responseBody ?? {}).toLowerCase()).toMatch(
        /provider|credential|api key|configured|model/,
      )

      const banner = page.getByTestId("playground-chat-error-banner")
      await expect(banner).toBeVisible({ timeout: 30_000 })
      await expect(banner).toContainText(/model setup|not callable|provider/i)
      await expect(
        banner.getByRole("button", { name: /open model settings/i }),
      ).toBeVisible()
      await expectSelectedCharacter(page, character.name)
    } finally {
      await deleteCharacterViaApi(character)
    }
  })

  test("sends through complete-v2 only when a real callable character model is available", async ({
    authedPage: page,
    serverInfo,
  }) => {
    skipIfServerUnavailable(serverInfo)

    const callableModel = await findCallableModelForSuccess()
    test.skip(
      !callableModel,
      "No trustworthy real callable chat model is configured. Set TLDW_E2E_CHARACTER_CALLABLE_MODEL, or set TLDW_E2E_ALLOW_LOCAL_PROVIDER_SUCCESS=1 for a known real local provider.",
    )
    if (!callableModel) return

    const character = await createCharacterViaApi()
    try {
      const chatPage = await openCharacterChatWithCharacter(
        page,
        character.name,
        callableModel,
      )

      const capture = captureAllApiCalls(page)
      await chatPage.sendMessage("Reply with one short sentence for Phase 7.")
      await waitForStreamComplete(page)
      await chatPage.waitForResponse(90_000)

      const calls = await capture.stop()
      const completeCall = calls.find((call) =>
        completeV2CallPredicate(call.url, call.method),
      )

      expect(completeCall).toBeTruthy()
      expect(completeCall?.status).toBeGreaterThanOrEqual(200)
      expect(completeCall?.status).toBeLessThan(300)
      expect(completeCall?.requestBody).toEqual(
        expect.objectContaining({
          include_character_context: true,
          stream: true,
        }),
      )
      await expectSelectedCharacter(page, character.name)
    } finally {
      await deleteCharacterViaApi(character)
    }
  })
})

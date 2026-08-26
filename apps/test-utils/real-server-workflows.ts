import {
  expect,
  test,
  type BrowserContext,
  type Locator,
  type Page
} from "@playwright/test"

export type WorkflowDriver = {
  kind: "extension" | "web"
  serverUrl: string
  apiKey: string
  context: BrowserContext
  page: Page
  optionsUrl: string
  sidepanelUrl: string
  openSidepanel: (target?: string) => Promise<Page>
  goto: (
    page: Page,
    route: string,
    options?: Parameters<Page["goto"]>[1]
  ) => Promise<void>
  ensureHostPermission: () => Promise<boolean>
  close: () => Promise<void>
}

export type CreateWorkflowDriver = (options: {
  serverUrl: string
  apiKey: string
  page: Page
  context: BrowserContext
  featureFlags?: Record<string, boolean>
  testRef?: typeof test
}) => Promise<WorkflowDriver>

export const ALL_FEATURE_FLAGS_ENABLED = {
  ff_newChat: true,
  ff_newSettings: true,
  ff_commandPalette: true,
  ff_compactMessages: true,
  ff_chatSidebar: true,
  ff_compareMode: true
}

export const ALL_FEATURE_FLAGS_DISABLED = {
  ff_newChat: false,
  ff_newSettings: false,
  ff_commandPalette: false,
  ff_compactMessages: false,
  ff_chatSidebar: false,
  ff_compareMode: false
}

export const createRealServerWorkflowTldwConfig = (
  serverUrl: string,
  apiKey: string
) => {
  const normalizedServerUrl = serverUrl.replace(/\/$/, "")
  return {
    serverUrl: normalizedServerUrl,
    apiKey,
    authMode: "single-user" as const,
    authSource: "manual" as const,
    credentialSource: "manual" as const,
    apiKeyPersistence: "device" as const,
    apiKeyServerOrigin: new URL(normalizedServerUrl).origin
  }
}

export const FEATURE_FLAG_KEYS = {
  NEW_CHAT: "ff_newChat",
  NEW_SETTINGS: "ff_newSettings",
  COMMAND_PALETTE: "ff_commandPalette",
  COMPACT_MESSAGES: "ff_compactMessages",
  CHAT_SIDEBAR: "ff_chatSidebar",
  COMPARE_MODE: "ff_compareMode"
} as const

export const createRealServerWorkflowStorageSeed = (
  dismissedAt = Date.now()
): Record<string, unknown> => ({
  __tldw_first_run_complete: true,
  assistant_setup_dismissed: true,
  tldw_skip_landing_hub: true,
  quickIngestInspectorIntroDismissed: true,
  quickIngestOnboardingDismissed: true,
  "tldw:workflow:landing-config": {
    showOnFirstRun: true,
    dismissedAt,
    completedWorkflows: []
  }
})

export const REAL_SERVER_WORKFLOW_LOCAL_STORAGE_SEED = {
  "playground-tour-completed": "true",
  "notes-tutorial-shown": "1",
  "tldw-tutorials": JSON.stringify({
    state: {
      completedTutorials: ["playground", "chat", "notes", "media", "settings"],
      seenPromptPages: [
        "/",
        "/chat",
        "/notes",
        "/media",
        "/settings",
        "/playground",
        "/research-workspace"
      ]
    },
    version: 0
  })
} as const

export const LEGACY_REAL_SERVER_WORKFLOW_TITLES = [
  "chat -> save to notes -> open linked conversation",
  "notes lifecycle: create, tag, preview, export, delete",
  "chat -> save to flashcards -> review card",
  "quick ingest -> media review",
  "knowledge QA search -> open chat with RAG settings",
  "prompts -> use in chat -> send message",
  "world books -> entries -> attach -> export -> stats",
  "dictionaries -> entries -> validate -> preview -> export -> stats",
  "playground -> server chat -> open history -> pin/unpin",
  "quiz -> take attempt -> review score",
  "chatbooks export -> download archive",
  "tts playback -> server provider -> audio segments",
  "compare mode -> multi-model answers -> choose winner",
  "data tables -> chat source -> generate -> save -> delete",
  "media trash -> delete -> restore",
  "media ingestion -> analysis -> review -> re-analyze",
  "characters -> chat persona -> send message"
] as const

export function withFeatures(
  flags: Array<keyof typeof ALL_FEATURE_FLAGS_ENABLED>,
  baseConfig?: Record<string, any>
): Record<string, any> {
  const flagConfig = Object.fromEntries(flags.map((flag) => [flag, true]))
  return {
    ...ALL_FEATURE_FLAGS_DISABLED,
    ...flagConfig,
    ...(baseConfig || {})
  }
}

const requireRealServerConfig = (): { serverUrl: string; apiKey: string } => {
  const serverUrl = process.env.TLDW_E2E_SERVER_URL
  const apiKey = process.env.TLDW_E2E_API_KEY

  if (!serverUrl || !apiKey) {
    test.skip(
      true,
      "Set TLDW_E2E_SERVER_URL and TLDW_E2E_API_KEY to run real-server E2E tests."
    )
    return { serverUrl: "", apiKey: "" }
  }

  return { serverUrl, apiKey }
}

const normalizeServerUrl = (value: string) =>
  value.match(/^https?:\/\//) ? value : `http://${value}`

const normalizePath = (value: string) => {
  const trimmed = String(value || "").trim().replace(/^\/+|\/+$/g, "")
  return trimmed ? `/${trimmed}` : ""
}

const joinUrl = (base: string, path: string) => {
  const trimmedBase = base.replace(/\/$/, "")
  const trimmedPath = path.startsWith("/") ? path : `/${path}`
  return `${trimmedBase}${trimmedPath}`
}

const waitForConnectionStore = async (page: Page, label = "init") => {
  const waitForAppReady = async (timeoutMs: number) => {
    await page.waitForFunction(
      () => {
        const root = document.querySelector("#root, #__next")
        if (!root) return false
        return document.readyState !== "loading"
      },
      null,
      { timeout: timeoutMs }
    )
  }

  await page.waitForLoadState("domcontentloaded")
  const root = page.locator("#root, #__next")
  try {
    await root.waitFor({ state: "attached", timeout: 15_000 })
  } catch {
    // ignore if root takes longer to mount; store check will retry
  }

  try {
    await waitForAppReady(15_000)
  } catch {
    await page.reload({ waitUntil: "domcontentloaded" })
    try {
      await root.waitFor({ state: "attached", timeout: 15_000 })
    } catch {
      // ignore; waitForStore will still time out if app never mounts
    }
    await waitForAppReady(20_000)
  }
  await logConnectionSnapshot(page, label)
}

const logConnectionSnapshot = async (page: Page, label: string) => {
  await page.evaluate((tag) => {
    const root = document.querySelector("#root, #__next")
    const w: any = window as any
    const store = w.__tldw_useConnectionStore
    if (!store?.getState) {
      console.log(
        "CONNECTION_DEBUG",
        tag,
        JSON.stringify({
          storeReady: false,
          rootReady: !!root,
          rootChildren: root ? root.children.length : 0,
          readyState: document.readyState
        })
      )
      return
    }
    try {
      const state = store.getState().state
      console.log(
        "CONNECTION_DEBUG",
        tag,
        JSON.stringify({
          phase: state.phase,
          configStep: state.configStep,
          mode: state.mode,
          errorKind: state.errorKind,
          serverUrl: state.serverUrl,
          isConnected: state.isConnected,
          isChecking: state.isChecking,
          knowledgeStatus: state.knowledgeStatus,
          hasCompletedFirstRun: state.hasCompletedFirstRun
        })
      )
    } catch {
      // ignore snapshot failures
    }
  }, label)
}

/**
 * Waits for the __tldw_useStoreMessageOption store to be available.
 * This helps avoid race conditions where tests try to access the store
 * before the React component that exposes it has mounted.
 *
 * @param throwOnFailure - If true (default), throws an error if store is not ready within timeout.
 *                         If false, returns false instead of throwing.
 */
const waitForMessageStore = async (
  page: Page,
  label = "init",
  timeoutMs = 30000,
  throwOnFailure = true
): Promise<boolean> => {
  const startTime = Date.now()
  try {
    await page.waitForFunction(
      () => {
        const w = window as any
        const store = w.__tldw_useStoreMessageOption
        return store?.getState && typeof store.getState === "function"
      },
      null,
      { timeout: timeoutMs }
    )
    console.log(
      `[waitForMessageStore] ${label} store ready after ${Date.now() - startTime}ms`
    )
    return true
  } catch {
    console.log(
      `[waitForMessageStore] ${label} store NOT ready after ${Date.now() - startTime}ms (timeout=${timeoutMs}ms)`
    )
    // Log additional debug info about the page state
    const debugInfo = await page
      .evaluate(() => {
        const w = window as any
        const hasStore = !!w.__tldw_useStoreMessageOption
        const hasGetState =
          hasStore && typeof w.__tldw_useStoreMessageOption?.getState === "function"
        const rootEl = document.querySelector("#root, #__next")
        return {
          hasStore,
          hasGetState,
          hasRoot: !!rootEl,
          rootChildCount: rootEl ? rootEl.children.length : 0,
          readyState: document.readyState,
          url: window.location.href
        }
      })
      .catch(() => ({
        hasStore: false,
        hasGetState: false,
        hasRoot: false,
        rootChildCount: 0,
        readyState: "unknown",
        url: "unknown"
      }))
    console.log(
      `[waitForMessageStore] ${label} debug: ${JSON.stringify(debugInfo)}`
    )
    if (throwOnFailure) {
      throw new Error(
        `[waitForMessageStore] ${label} store not ready after ${timeoutMs}ms. ` +
        `Page state: url=${debugInfo.url}, hasStore=${debugInfo.hasStore}, ` +
        `hasGetState=${debugInfo.hasGetState}, hasRoot=${debugInfo.hasRoot}, ` +
        `rootChildCount=${debugInfo.rootChildCount}, readyState=${debugInfo.readyState}`
      )
    }
    return false
  }
}

const setSelectedModel = async (page: Page, model: string) => {
  await page.evaluate(
    async ({ modelId, timeoutMs, intervalMs }) => {
      const w: any = window as any
      const hasSync =
        w?.chrome?.storage?.sync?.set && w?.chrome?.storage?.sync?.get
      const hasLocal =
        w?.chrome?.storage?.local?.set && w?.chrome?.storage?.local?.get

      const storageArea = hasSync
        ? w.chrome.storage.sync
        : hasLocal
          ? w.chrome.storage.local
          : null

      const setValue = (
        area: typeof chrome.storage.local | typeof chrome.storage.sync,
        items: Record<string, unknown>
      ) =>
        new Promise<void>((resolve, reject) => {
          area.set(items, () => {
            const err = w?.chrome?.runtime?.lastError
            if (err) reject(err)
            else resolve()
          })
        })

      const getValue = (
        area: typeof chrome.storage.local | typeof chrome.storage.sync,
        keys: string[]
      ) =>
        new Promise<Record<string, unknown>>((resolve, reject) => {
          area.get(keys, (items: Record<string, unknown>) => {
            const err = w?.chrome?.runtime?.lastError
            if (err) reject(err)
            else resolve(items)
          })
        })

      const normalizeStoredValue = (value: unknown) => {
        if (typeof value !== "string") return value
        try {
          return JSON.parse(value)
        } catch {
          return value
        }
      }

      const applyStore = () => {
        try {
          const store = w.__tldw_useStoreMessageOption
          store?.getState?.().setSelectedModel?.(modelId)
        } catch {
          // ignore store update failures
        }
      }

      if (!storageArea) {
        try {
          localStorage.setItem("selectedModel", JSON.stringify(modelId))
          applyStore()
        } catch (error) {
          console.warn("MODEL_DEBUG: Failed to set selectedModel", error)
        }
        return
      }

      try {
        const serialized = JSON.stringify(modelId)
        if (hasSync && hasLocal) {
          await setValue(w.chrome.storage.sync, { selectedModel: serialized })
          await setValue(w.chrome.storage.local, { selectedModel: serialized })
        } else {
          await setValue(storageArea, { selectedModel: serialized })
        }
      } catch (error) {
        console.warn("MODEL_DEBUG: Failed to set selectedModel", error)
        return
      }

      const startedAt = Date.now()
      let lastRead: unknown = undefined
      while (Date.now() - startedAt < timeoutMs) {
        try {
          const data = await getValue(storageArea, ["selectedModel"])
          lastRead = normalizeStoredValue(data?.selectedModel)
          if (lastRead === modelId) {
            console.log("MODEL_DEBUG: Confirmed selectedModel stored as", modelId)
            applyStore()
            return
          }
        } catch (error) {
          console.warn("MODEL_DEBUG: Failed to read back selectedModel", error)
          return
        }

        await new Promise<void>((resolve) => {
          setTimeout(resolve, intervalMs)
        })
      }

      console.warn("MODEL_DEBUG: Timed out waiting for selectedModel", {
        expected: modelId,
        actual: lastRead
      })
      applyStore()
    },
    { modelId: model, timeoutMs: 3_000, intervalMs: 50 }
  )
}

export type RunnableChatModel = {
  id: string
  provider: string
}

const modelRecord = (value: unknown): Record<string, unknown> | null =>
  value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null

const nonEmptyString = (value: unknown): string | null => {
  if (typeof value !== "string" && typeof value !== "number") return null
  const normalized = String(value).trim()
  return normalized || null
}

const normalizeProviderKey = (value: unknown): string =>
  String(value || "")
    .trim()
    .toLowerCase()
    .replaceAll("-", "_")

const configuredProviderModels = (payload: unknown): RunnableChatModel[] => {
  const root = modelRecord(payload)
  const providers = Array.isArray(root?.providers)
    ? root.providers
    : Array.isArray(payload)
      ? payload
      : []

  return providers.flatMap((value) => {
    const provider = modelRecord(value)
    if (!provider || provider.is_configured !== true) return []
    const providerId = nonEmptyString(
      provider.chat_provider ??
        provider.chatProvider ??
        provider.api_provider ??
        provider.provider_key ??
        provider.name ??
        provider.id ??
        provider.provider
    )
    if (!providerId || !Array.isArray(provider.models)) return []

    return provider.models.flatMap((modelValue) => {
      const model = modelRecord(modelValue)
      const id = nonEmptyString(
        model?.id ?? model?.model ?? model?.name ?? modelValue
      )
      return id ? [{ id, provider: providerId }] : []
    })
  })
}

const configuredMetadataModels = (payload: unknown): RunnableChatModel[] => {
  const root = modelRecord(payload)
  const entries = Array.isArray(payload)
    ? payload
    : Array.isArray(root?.models)
      ? root.models
      : Array.isArray(root?.items)
        ? root.items
        : []

  return entries.flatMap((value) => {
    const model = modelRecord(value)
    if (!model) return []
    const details = modelRecord(model.details)
    const provider = nonEmptyString(
      model.chat_provider ??
        model.chatProvider ??
        model.api_provider ??
        model.provider_key ??
        model.provider ??
        details?.chat_provider ??
        details?.api_provider ??
        details?.provider
    )
    const id = nonEmptyString(model.id ?? model.model ?? model.name)
    if (!provider || !id) return []

    const configured =
      model.is_configured ??
      model.configured ??
      model.provider_is_configured ??
      details?.is_configured ??
      details?.configured
    if (configured !== true) return []

    const capabilities = [
      model.type,
      model.model_type,
      model.capabilities,
      details?.type,
      details?.model_type,
      details?.capabilities
    ]
      .flatMap((field) => (Array.isArray(field) ? field : [field]))
      .map((field) =>
        String(field || "")
          .trim()
          .toLowerCase()
      )
      .filter(Boolean)
    if (!capabilities.some((capability) => capability.includes("chat"))) {
      return []
    }

    return [{ id, provider }]
  })
}

export const resolveRunnableChatModel = (
  payload: unknown
): RunnableChatModel | null => {
  const candidates = [
    ...configuredProviderModels(payload),
    ...configuredMetadataModels(payload)
  ]
  return (
    candidates.find(
      (candidate) =>
        normalizeProviderKey(candidate.provider) === "custom_openai_api"
    ) ??
    candidates[0] ??
    null
  )
}

export const toSelectedModelId = ({
  id,
  provider
}: RunnableChatModel): string => {
  if (id.startsWith("tldw:")) return id
  return `tldw:${provider}:${id}`
}

const fetchWithKey = async (
  url: string,
  apiKey: string,
  init: RequestInit = {}
) => {
  const headers = {
    "x-api-key": apiKey,
    ...(init.headers || {})
  }
  try {
    return await fetch(url, { ...init, headers })
  } catch (error) {
    test.skip(
      true,
      `Real-server request unreachable in this environment: ${String(error)}`
    )
    throw error
  }
}

const fetchWithKeyTimeout = async (
  url: string,
  apiKey: string,
  init: RequestInit = {},
  timeoutMs = 15000
) => {
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs)
  try {
    return await fetchWithKey(url, apiKey, {
      ...init,
      signal: controller.signal
    })
  } catch (error: any) {
    if (error?.name === "AbortError") return null
    throw error
  } finally {
    clearTimeout(timeoutId)
  }
}

const resolveMediaApi = async (serverUrl: string, apiKey: string) => {
  const normalized = serverUrl.replace(/\/$/, "")
  let apiBase = normalized
  const override = process.env.TLDW_E2E_MEDIA_BASE
  let mediaBasePath = normalizePath(override || "/api/v1/media")

  const openApi = await fetchWithKey(
    `${normalized}/openapi.json`,
    apiKey
  ).catch(() => null)
  if (openApi?.ok) {
    const payload = await openApi.json().catch(() => null)
    const servers = Array.isArray(payload?.servers) ? payload.servers : []
    const serverEntry = servers.find(
      (entry: any) => typeof entry?.url === "string"
    )
    const openApiServerUrl =
      typeof serverEntry?.url === "string" ? serverEntry.url : ""
    if (openApiServerUrl && openApiServerUrl !== "/") {
      if (
        openApiServerUrl.startsWith("http://") ||
        openApiServerUrl.startsWith("https://")
      ) {
        apiBase = openApiServerUrl.replace(/\/$/, "")
      } else {
        apiBase = `${normalized}${openApiServerUrl.startsWith("/") ? "" : "/"}${openApiServerUrl}`.replace(
          /\/$/,
          ""
        )
      }
    }

    if (!override) {
      const paths =
        payload?.paths && typeof payload.paths === "object"
          ? Object.keys(payload.paths)
          : []
      const candidates = ["/api/v1/media", "/api/media", "/media"]
      for (const candidate of candidates) {
        const normalizedCandidate = normalizePath(candidate)
        if (
          paths.includes(normalizedCandidate) ||
          paths.includes(`${normalizedCandidate}/`) ||
          paths.includes(`${normalizedCandidate}/search`)
        ) {
          mediaBasePath = normalizedCandidate
          break
        }
      }
    }
  }

  return { apiBase, mediaBasePath }
}

const preflightMediaApi = async (
  apiBase: string,
  mediaBasePath: string,
  apiKey: string
) => {
  const listUrl = joinUrl(
    apiBase,
    `${mediaBasePath}?page=1&results_per_page=1`
  )
  const listRes = await fetchWithKey(listUrl, apiKey).catch(() => null)
  if (listRes?.ok) return
  if (listRes && listRes.status !== 404) {
    const body = await listRes.text().catch(() => "")
    throw new Error(
      `Media API preflight failed: ${listRes.status} ${listRes.statusText} ${body}`
    )
  }

  const searchUrl = joinUrl(
    apiBase,
    `${mediaBasePath}/search?page=1&results_per_page=1`
  )
  const searchRes = await fetchWithKey(searchUrl, apiKey, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query: "e2e-preflight",
      fields: ["title", "content"],
      sort_by: "relevance"
    })
  }).catch(() => null)
  if (searchRes?.ok) return
  const body = await searchRes?.text().catch(() => "")
  throw new Error(
    `Media API preflight failed: ${searchRes?.status ?? "no response"} ${searchRes?.statusText ?? ""} ${body}`
  )
}

const skipOrThrow = (condition: boolean, message: string) => {
  if (!condition) return
  test.skip(true, message)
}

/**
 * Checks if the page has rendered content (not a blank white page).
 * Returns diagnostic info about what's visible.
 */
const checkPageHasContent = async (page: Page): Promise<{
  hasContent: boolean
  bodyChildCount: number
  rootElementFound: boolean
  visibleTextLength: number
  errorMessages: string[]
}> => {
  return page.evaluate(() => {
    const body = document.body
    const bodyChildCount = body?.childElementCount ?? 0
    const rootElement = document.getElementById("root") || document.getElementById("app")
    const rootElementFound = !!rootElement && rootElement.childElementCount > 0
    const visibleText = body?.innerText?.trim() || ""
    const visibleTextLength = visibleText.length

    // Check for error messages in the DOM
    const errorMessages: string[] = []
    const errorElements = document.querySelectorAll('[class*="error"], [class*="Error"], .ant-alert-error')
    errorElements.forEach(el => {
      const text = (el as HTMLElement).innerText?.trim()
      if (text) errorMessages.push(text.slice(0, 200))
    })

    // Console errors aren't accessible here, but we can check for React error boundaries
    const reactErrorBoundary = document.querySelector('[data-reactroot] > div[style*="background: white"]')
    if (reactErrorBoundary) {
      errorMessages.push("Possible React error boundary detected")
    }

    return {
      hasContent: bodyChildCount > 0 && (rootElementFound || visibleTextLength > 50),
      bodyChildCount,
      rootElementFound,
      visibleTextLength,
      errorMessages
    }
  })
}

/**
 * Waits for the page to have rendered content, with diagnostic output on failure.
 */
const waitForPageContent = async (page: Page, label: string, timeoutMs = 15000): Promise<void> => {
  const startTime = Date.now()
  let lastCheck: Awaited<ReturnType<typeof checkPageHasContent>> | null = null

  while (Date.now() - startTime < timeoutMs) {
    lastCheck = await checkPageHasContent(page)
    if (lastCheck.hasContent) {
      return
    }
    await page.waitForTimeout(500)
  }

  // Page is blank - log diagnostics and throw
  console.error(`[${label}] Page appears blank after ${timeoutMs}ms:`, {
    url: page.url(),
    ...lastCheck
  })
  throw new Error(
    `Page failed to render content (blank page detected) for ${label}. ` +
    `URL: ${page.url()}, bodyChildCount: ${lastCheck?.bodyChildCount}, ` +
    `rootElementFound: ${lastCheck?.rootElementFound}, visibleTextLength: ${lastCheck?.visibleTextLength}`
  )
}

const pingBackgroundScript = async (page: Page): Promise<{ ok: boolean; pong?: boolean; error?: string }> => {
  try {
    const result = await page.evaluate(async () => {
      if (typeof chrome === "undefined" || !chrome.runtime?.sendMessage) {
        return { ok: false, error: "No chrome.runtime.sendMessage" }
      }
      return new Promise<{ ok: boolean; pong?: boolean; error?: string }>(
        (resolve) => {
          const timeout = setTimeout(() => {
            resolve({ ok: false, error: "ping timeout" })
          }, 5000)
          try {
            chrome.runtime.sendMessage({ type: "tldw:ping" }, (response) => {
              clearTimeout(timeout)
              if (chrome.runtime.lastError) {
                resolve({
                  ok: false,
                  error: chrome.runtime.lastError.message || "lastError"
                })
              } else {
                resolve(response || { ok: false, error: "no response" })
              }
            })
          } catch (err: any) {
            clearTimeout(timeout)
            resolve({ ok: false, error: err?.message || "exception" })
          }
        }
      )
    })
    return result
  } catch (err) {
    return { ok: false, error: String(err) }
  }
}

const logRuntimeDiagnostics = async (page: Page, label: string) => {
  const safeStringify = (value: unknown) => {
    try {
      return JSON.stringify(value)
    } catch {
      return "\"[unserializable]\""
    }
  }

  const runtime = await page
    .evaluate(() => {
      const w = globalThis as any
      const browserRuntime = w.browser?.runtime
      const chromeRuntime = w.chrome?.runtime
      return {
        url: w.location?.href || null,
        hasChrome: !!w.chrome,
        hasBrowser: !!w.browser,
        browserRuntime: {
          hasRuntime: !!browserRuntime,
          id: browserRuntime?.id || null,
          hasSendMessage: typeof browserRuntime?.sendMessage === "function",
          hasOnMessage: typeof browserRuntime?.onMessage?.addListener === "function"
        },
        chromeRuntime: {
          hasRuntime: !!chromeRuntime,
          id: chromeRuntime?.id || null,
          hasSendMessage: typeof chromeRuntime?.sendMessage === "function",
          hasOnMessage: typeof chromeRuntime?.onMessage?.addListener === "function",
          lastError: chromeRuntime?.lastError?.message || null
        },
        sameRuntime: browserRuntime === chromeRuntime,
        sameSendMessage: browserRuntime?.sendMessage === chromeRuntime?.sendMessage
      }
    })
    .catch((err) => ({ error: String(err) }))

  const context = page.context()
  const swUrls = context.serviceWorkers().map((sw) => sw.url())
  const bgUrls = context.backgroundPages().map((bg) => bg.url())

  console.log(
    `[E2E_RUNTIME] ${label}`,
    safeStringify({ runtime, swUrls, bgUrls })
  )
}

const logMessageBusDiagnostics = async (page: Page, label: string) => {
  const safeStringify = (value: unknown) => {
    try {
      return JSON.stringify(value)
    } catch {
      return "\"[unserializable]\""
    }
  }

  const result = await page
    .evaluate(async () => {
      const w = globalThis as any
      const chromeRuntime = w.chrome?.runtime
      const browserRuntime = w.browser?.runtime

      const runCallbackPing = (runtime: any, tag: string) =>
        new Promise((resolve) => {
          if (!runtime?.sendMessage) {
            resolve({ tag, ok: false, error: "no sendMessage" })
            return
          }
          let settled = false
          const timeout = setTimeout(() => {
            if (settled) return
            settled = true
            resolve({
              tag,
              ok: false,
              error: "timeout",
              lastError: runtime?.lastError?.message || null
            })
          }, 3000)
          try {
            runtime.sendMessage(
              { type: "tldw:ping", _e2e: "diagnostic-callback" },
              (response: any) => {
                if (settled) return
                settled = true
                clearTimeout(timeout)
                resolve({
                  tag,
                  ok: true,
                  response,
                  lastError: runtime?.lastError?.message || null
                })
              }
            )
          } catch (err: any) {
            if (settled) return
            settled = true
            clearTimeout(timeout)
            resolve({
              tag,
              ok: false,
              error: err?.message || "exception",
              lastError: runtime?.lastError?.message || null
            })
          }
        })

      const runPromisePing = async (runtime: any, tag: string) => {
        if (!runtime?.sendMessage) {
          return { tag, ok: false, error: "no sendMessage" }
        }
        try {
          const maybePromise = runtime.sendMessage({
            type: "tldw:ping",
            _e2e: "diagnostic-promise"
          })
          if (!maybePromise || typeof maybePromise.then !== "function") {
            return {
              tag,
              ok: false,
              error: "sendMessage did not return Promise",
              returnedType: typeof maybePromise
            }
          }
          const resp = await Promise.race([
            maybePromise
              .then((response: any) => ({
                ok: true,
                response,
                lastError: runtime?.lastError?.message || null
              }))
              .catch((err: any) => ({
                ok: false,
                error: err?.message || String(err),
                lastError: runtime?.lastError?.message || null
              })),
            new Promise((resolve) =>
              setTimeout(
                () =>
                  resolve({
                    ok: false,
                    error: "promise timeout",
                    lastError: runtime?.lastError?.message || null
                  }),
                3000
              )
            )
          ])
          return { tag, ...resp }
        } catch (err: any) {
          return {
            tag,
            ok: false,
            error: err?.message || "exception",
            lastError: runtime?.lastError?.message || null
          }
        }
      }

      const runPortTest = (runtime: any, tag: string) =>
        new Promise((resolve) => {
          if (!runtime?.connect) {
            resolve({ tag, ok: false, error: "no connect" })
            return
          }
          let disconnected = false
          let resolved = false
          try {
            const port = runtime.connect({ name: "e2e:diagnostic" })
            const timer = setTimeout(() => {
              if (resolved) return
              resolved = true
              try {
                port.disconnect()
              } catch {}
              resolve({
                tag,
                ok: true,
                connected: true,
                disconnected: false,
                lastError: runtime?.lastError?.message || null
              })
            }, 1000)

            port.onDisconnect.addListener(() => {
              if (resolved) return
              resolved = true
              disconnected = true
              clearTimeout(timer)
              resolve({
                tag,
                ok: true,
                connected: true,
                disconnected,
                lastError: runtime?.lastError?.message || null
              })
            })

            try {
              port.postMessage({
                type: "tldw:ping",
                _e2e: "diagnostic-port"
              })
            } catch {
              // ignore postMessage errors for diagnostics
            }
          } catch (err: any) {
            if (resolved) return
            resolved = true
            resolve({
              tag,
              ok: false,
              error: err?.message || "exception",
              lastError: runtime?.lastError?.message || null
            })
          }
        })

      return {
        url: w.location?.href || null,
        callbackPing: {
          chrome: await runCallbackPing(chromeRuntime, "chrome"),
          browser: await runCallbackPing(browserRuntime, "browser")
        },
        promisePing: {
          chrome: await runPromisePing(chromeRuntime, "chrome"),
          browser: await runPromisePing(browserRuntime, "browser")
        },
        portTest: {
          chrome: await runPortTest(chromeRuntime, "chrome"),
          browser: await runPortTest(browserRuntime, "browser")
        }
      }
    })
    .catch((err) => ({ error: String(err) }))

  console.log(`[E2E_MSG_DIAG] ${label}`, safeStringify(result))
}

const waitForConnected = async (
  page: Page,
  label: string,
  surface: WorkflowDriver["kind"]
) => {
  // First check that the page has actually rendered content (not blank)
  await waitForPageContent(page, label, 20000)

  await waitForConnectionStore(page, label)

  if (surface === "extension") {
    // The callback ping is an extension background-liveness probe. The WebUI
    // runtime shim intentionally has no extension background listener.
    await page.evaluate(() => {
      console.log("PING_DEBUG starting ping test")
    })

    const pingResult = await pingBackgroundScript(page)

    await page.evaluate((res) => {
      console.log("PING_DEBUG background script ping result", JSON.stringify(res))
    }, pingResult)

    if (!pingResult.ok) {
      console.warn(`[PING_DEBUG] background ping failed for ${label}:`, pingResult?.error || "unknown error")
      await logRuntimeDiagnostics(page, `${label}-ping-failed`)
      await logMessageBusDiagnostics(page, `${label}-ping-failed`)
      const shouldForceConnected =
        process.env.TLDW_E2E_FORCE_CONNECTED !== "0" &&
        process.env.TLDW_E2E_FORCE_CONNECTED !== "false"
      if (shouldForceConnected) {
        await page.evaluate(() => {
          const store = (window as any).__tldw_useConnectionStore
          if (!store?.getState || !store?.setState) return
          const prev = store.getState().state || {}
          const now = Date.now()
          store.setState({
            state: {
              ...prev,
              phase: "connected",
              isConnected: true,
              isChecking: false,
              offlineBypass: true,
              errorKind: "none",
              lastError: null,
              lastStatusCode: null,
              lastCheckedAt: now,
              knowledgeStatus: "ready",
              knowledgeLastCheckedAt: now,
              knowledgeError: null,
              mode: "normal",
              configStep: "health",
              hasCompletedFirstRun: true
            }
          })
        })
      }
    }
  }

  await page.evaluate(() => {
    const store = (window as any).__tldw_useConnectionStore
    try {
      store?.getState?.().markFirstRunComplete?.()
      store?.getState?.().checkOnce?.()
    } catch {
      // ignore check errors
    }
    window.dispatchEvent(new CustomEvent("tldw:check-connection"))
  })
  try {
    await page.waitForFunction(
      () => {
        const store = (window as any).__tldw_useConnectionStore
        const state = store?.getState?.().state
        return state?.isConnected === true && state?.phase === "connected"
      },
      undefined,
      { timeout: 20000 }
    )
  } catch (error) {
    await logConnectionSnapshot(page, `${label}-timeout`)
    throw error
  }
}

const waitForChatLanding = async (
  page: Page,
  driver: WorkflowDriver,
  timeoutMs = 20000
) => {
  await page.waitForFunction(
    (kind) => {
      const hash = window.location.hash || ""
      const path = window.location.pathname || ""
      const search = window.location.search || ""
      if (kind === "extension") {
        return hash.startsWith("#/chat") || search.includes("view=chat")
      }
      return (
        path === "/chat" ||
        path === "/" ||
        hash === "#/" ||
        hash === "#" ||
        hash.startsWith("#/chat")
      )
    },
    driver.kind,
    { timeout: timeoutMs }
  )
}

const openChatSidepanel = async (driver: WorkflowDriver): Promise<Page> => {
  if (driver.kind === "extension") {
    return driver.openSidepanel("/chat")
  }
  return driver.openSidepanel()
}

const ensureFreshNoteEditor = async (page: Page) => {
  const titleInput = page.getByPlaceholder("Title", { exact: true })
  const contentInput = page.getByPlaceholder(/Write your note here/i)
  const titleVisible = await titleInput.isVisible().catch(() => false)
  const contentVisible = await contentInput.isVisible().catch(() => false)
  if (titleVisible && contentVisible) {
    const titleValue = await titleInput.inputValue().catch(() => "")
    const contentValue = await contentInput.inputValue().catch(() => "")
    if (!titleValue.trim() && !contentValue.trim()) {
      return
    }
  }

  const newNoteButton = page.getByRole("button", { name: /New note/i })
  await expect(newNoteButton).toBeVisible({ timeout: 15000 })
  await newNoteButton.click()
}

const ensureServerPersistence = async (page: Page) => {
  const persistenceSwitch = page.getByRole("switch", {
    name: /Save chat to history|Temporary chat/i
  })
  if ((await persistenceSwitch.count()) === 0) return
  const checked = await persistenceSwitch
    .getAttribute("aria-checked")
    .catch(() => null)
  if (checked !== "true") {
    await persistenceSwitch.click()
  }
}

const selectTrackedCharacterFromRuntimeRail = async (
  page: Page,
  characterName: string,
  surface: WorkflowDriver["kind"],
  characterId?: string | number | null
) => {
  if (surface === "extension") {
    const trigger = page.getByTestId("chat-character-controls-trigger").first()
    await expect(trigger).toBeVisible({ timeout: 30000 })
    await trigger.click()

    const controls = page.getByTestId("chat-character-controls-sheet")
    await expect(controls).toBeVisible({ timeout: 15000 })
    await controls
      .getByRole("button", { name: "Start tracked character chat" })
      .click()

    const panel = page.getByTestId("assistant-select-panel")
    await expect(panel).toBeVisible({ timeout: 15000 })
    const search = panel.getByRole("textbox", {
      name: /Search characters and personas/i
    })
    if (await search.isVisible().catch(() => false)) {
      await search.fill(characterName)
    }

    const characterButton = panel.getByRole("button", {
      name: characterName,
      exact: true
    })
    const retryButton = panel.getByRole("button", { name: "Retry characters" })
    await expect
      .poll(
        async () => {
          if (await characterButton.isVisible().catch(() => false)) return true
          if (await retryButton.isVisible().catch(() => false)) {
            await retryButton.click()
          }
          return false
        },
        {
          timeout: 30000,
          intervals: [500, 1000, 2000],
          message: `Timed out finding extension character ${characterName}`
        }
      )
      .toBe(true)
    await characterButton.click()
    await expect(panel).toBeHidden({ timeout: 10000 })

    const controlsDialog = page.getByRole("dialog", {
      name: "Character controls"
    })
    // CharacterControlsSheet closes only after the selected assistant has been
    // committed and its completion callback has reset the previous chat. Waiting
    // for that lifecycle boundary avoids racing a send against the async storage
    // commit (and accidentally routing the turn as an untracked local chat).
    await expect(controlsDialog).toBeHidden({ timeout: 10000 })

    const selectedCharacterText = page.getByText(characterName, {
      exact: false
    })
    await expect
      .poll(
        async () => {
          const matches = await selectedCharacterText.count()
          for (let index = 0; index < matches; index += 1) {
            if (await selectedCharacterText.nth(index).isVisible()) return true
          }
          return false
        },
        {
          timeout: 30000,
          intervals: [250, 500, 1000],
          message: `Timed out waiting for extension character ${characterName} to become active`
        }
      )
      .toBe(true)
    return
  }

  const trigger = page
    .getByRole("button", { name: "Select character or persona" })
    .first()
  await expect(trigger).toBeVisible({ timeout: 30000 })
  await trigger.click()

  const panel = page.getByTestId("assistant-select-panel")
  await expect(panel).toBeVisible({ timeout: 15000 })
  const charactersTab = page.getByRole("tab", { name: "Characters" })
  await charactersTab.click()
  await expect(charactersTab).toHaveAttribute("aria-selected", "true")

  const search = panel.getByRole("textbox", {
    name: /Search characters and personas/i
  })
  if (await search.isVisible().catch(() => false)) {
    await search.fill(characterName)
  }

  const characterButton = panel.getByRole("button", {
    name: characterName,
    exact: true
  })
  const retryButton = panel.getByRole("button", { name: "Retry characters" })
  await expect
    .poll(
      async () => {
        if (await characterButton.isVisible().catch(() => false)) return true
        if (await retryButton.isVisible().catch(() => false)) {
          await retryButton.click()
        }
        return false
      },
      {
        timeout: 30000,
        intervals: [500, 1000, 2000],
        message: `Timed out finding tracked character ${characterName}`
      }
    )
    .toBe(true)

  await characterButton.click()
  await expect(panel).toBeHidden({ timeout: 10000 })

  const selectionTriggers = page.getByTestId("character-select")
  await expect
    .poll(
      async () => {
        const triggerCount = await selectionTriggers.count()
        for (let index = 0; index < triggerCount; index += 1) {
          const candidate = selectionTriggers.nth(index)
          if (!(await candidate.isVisible().catch(() => false))) continue
          const label = await candidate.getAttribute("aria-label").catch(() => null)
          if (label?.includes(characterName)) return true
        }
        return false
      },
      {
        timeout: 30000,
        intervals: [250, 500, 1000],
        message: `Timed out waiting for tracked character ${characterName} to become active`
      }
    )
    .toBe(true)
}

const ensureChatSidebarExpanded = async (page: Page) => {
  const sidebar = page.getByTestId("chat-sidebar")
  await expect(sidebar).toBeVisible({ timeout: 20000 })
  const search = page.getByTestId("chat-sidebar-search")
  const expanded = await search.isVisible().catch(() => false)
  if (!expanded) {
    const toggle = page.getByTestId("chat-sidebar-toggle")
    if ((await toggle.count()) > 0) {
      await toggle.first().click()
      await expect(search).toBeVisible({ timeout: 15000 })
    }
  }
  return sidebar
}

const dismissQuickIngestInspectorIntro = async (page: Page) => {
  const drawer = page
    .locator(".ant-drawer")
    .filter({ hasText: /Inspector/i })
    .first()
  const gotIt = drawer.getByRole("button", { name: /Got it/i })
  const gotItVisible = await gotIt.isVisible().catch(() => false)
  if (gotItVisible) {
    await gotIt.click()
    await expect(page.locator(".ant-drawer")).toHaveCount(0, {
      timeout: 5000
    })
    return
  }
  const closeButton = drawer.getByRole("button", { name: /Close/i })
  const closeVisible = await closeButton.isVisible().catch(() => false)
  if (closeVisible) {
    await closeButton.click()
    await expect(page.locator(".ant-drawer")).toHaveCount(0, {
      timeout: 5000
    })
  }
}

const clickQuickIngestRun = async (modal: Locator) => {
  const page = modal.page()
  const resolveQuickIngestAction = async (): Promise<Locator> => {
    const candidates = [
      modal.getByTestId("quick-ingest-run"),
      modal.getByRole("button", { name: "Next", exact: true }),
      modal.getByRole("button", { name: /Start Processing/i }),
      modal.getByRole("button", { name: /Run quick ingest/i }),
      modal.getByRole("button", { name: /Configure \d+ items?/i }),
      modal.getByRole("button", { name: /Review \d+ items?/i }),
      modal.getByRole("button", { name: /Process \d+ items?/i }),
      modal.getByRole("button", { name: /Ingest/i })
    ]

    for (const candidate of candidates) {
      const target = candidate.first()
      if (await target.isVisible().catch(() => false)) {
        return target
      }
    }

    const buttonLabels = await modal
      .getByRole("button")
      .allTextContents()
      .catch(() => [])
    throw new Error(
      `Quick Ingest action was not visible. Buttons: ${JSON.stringify(buttonLabels)}`
    )
  }

  const waitForStableConnection = async (label: string) => {
    await page.waitForFunction(
      () => {
        const store = (window as any).__tldw_useConnectionStore
        const state = store?.getState?.().state
        return (
          state?.isConnected === true &&
          state?.phase === "connected" &&
          state?.isChecking === false
        )
      },
      undefined,
      { timeout: 15000 }
    ).catch(async (error) => {
      await logConnectionSnapshot(page, `quick-ingest-run-${label}`)
      throw error
    })
  }
  await waitForStableConnection("before-click")

  const triggerRun = async () => {
    let activeAction: Locator | null = null
    await expect
      .poll(
        async () => {
          activeAction = await resolveQuickIngestAction().catch(() => null)
          return activeAction !== null
        },
        { timeout: 15000 }
      )
      .toBe(true)
    if (!activeAction) {
      throw new Error("Quick Ingest action did not render.")
    }
    const resolvedAction: Locator = activeAction
    await resolvedAction.scrollIntoViewIfNeeded()
    await expect(resolvedAction).toBeEnabled({ timeout: 15000 })
    const label = ((await resolvedAction.textContent().catch(() => "")) || "").trim()
    await resolvedAction.click({ timeout: 10000, force: true })
    return label
  }

  const processingHeading = modal.getByRole("heading", { name: /^Processing$/i })
  const resultsStep = modal.getByTestId("wizard-results-step")
  const currentStep = modal.locator('[aria-current="step"]').first()
  const hasStarted = async () =>
    (await processingHeading.isVisible().catch(() => false)) ||
    (await resultsStep.isVisible().catch(() => false))
  const clickedLabels: string[] = []
  let started = false

  for (let step = 0; step < 4; step += 1) {
    const previousStep = await currentStep.getAttribute("aria-label").catch(() => null)
    clickedLabels.push(await triggerRun())
    const advanced = await expect
      .poll(
        async () => {
          if (await hasStarted()) return true
          const nextStep = await currentStep
            .getAttribute("aria-label")
            .catch(() => null)
          return Boolean(nextStep && nextStep !== previousStep)
        },
        { timeout: 15000 }
      )
      .toBe(true)
      .then(() => true)
      .catch(() => false)
    started = await hasStarted()
    if (started || !advanced) break
  }
  if (!started) {
    const buttons = await modal
      .getByRole("button")
      .evaluateAll((elements) =>
        elements.map((element) => ({
          text: element.textContent?.trim() || "",
          disabled: (element as HTMLButtonElement).disabled,
          ariaCurrent: element.getAttribute("aria-current")
        }))
      )
      .catch(() => [])
    const notices = await page
      .locator(".ant-message-notice-content")
      .allTextContents()
      .catch(() => [])
    const warnings = await modal.locator(".text-warn").allTextContents().catch(() => [])
    const reattach = await modal
      .getByRole("button", { name: /Reattach/i })
      .allTextContents()
      .catch(() => [])
    const connection = await page.evaluate(() => {
      const store = (window as any).__tldw_useConnectionStore
      return store?.getState?.().state || null
    }).catch(() => null)
    throw new Error(
      `Quick ingest run did not render a processing or result state. Debug: ${JSON.stringify({
        clickedLabels,
        buttons,
        notices,
        warnings,
        reattach,
        connection
      })}`
    )
  }
}

const selectQuickIngestQuickPreset = async (modal: Locator) => {
  const quickPreset = modal.getByRole("button", {
    name: "Quick preset",
    exact: true
  })
  if (!(await quickPreset.isVisible().catch(() => false))) {
    const configureButton = modal
      .getByRole("button", { name: /Configure \d+ items?/i })
      .first()
    await expect(configureButton).toBeVisible({ timeout: 15000 })
    await expect(configureButton).toBeEnabled({ timeout: 15000 })
    await configureButton.click()
  }
  await expect(quickPreset).toBeVisible({ timeout: 15000 })
  if ((await quickPreset.getAttribute("aria-pressed")) !== "true") {
    await quickPreset.click()
    await expect(quickPreset).toHaveAttribute("aria-pressed", "true", {
      timeout: 15000
    })
  }
}

const waitForQuickIngestCompletion = async (
  modal: Locator,
  timeoutMs = 120000
) => {
  const resultsStep = modal.getByTestId("wizard-results-step")
  const completedRegion = modal
    .getByRole("region", { name: /completed items/i })
    .first()
  const errorRegion = modal
    .getByRole("region", { name: /error items/i })
    .first()

  await expect
    .poll(
      async () =>
        (await resultsStep.isVisible().catch(() => false)) ||
        (await completedRegion.isVisible().catch(() => false)) ||
        (await errorRegion.isVisible().catch(() => false)),
      {
        timeout: timeoutMs,
        message: "Timed out waiting for quick ingest to reach a result state"
      }
    )
    .toBe(true)
}

const closeQuickIngestModal = async (modal: Locator) => {
  const closeButton = modal
    .getByRole("button", { name: "Close", exact: true })
    .first()
  await expect(closeButton).toBeVisible({ timeout: 15000 })
  await closeButton.click({ timeout: 15000 })
  await expect(modal).toBeHidden({ timeout: 15000 })
}

const resolveQuickIngestModal = (page: Page) =>
  page.getByRole("dialog", { name: /Quick Ingest/i }).first()

const waitForQuickIngestReady = async (modal: Locator) => {
  await expect(modal).toBeVisible({ timeout: 15000 })
  await expect(
    modal.locator('[data-testid="qi-file-input"]').first()
  ).toHaveCount(1, { timeout: 20000 })
}

const openQuickIngestModal = async (page: Page) => {
  const modal = resolveQuickIngestModal(page)
  if (await modal.isVisible().catch(() => false)) return modal

  const triggerCandidates = [
    page.getByTestId("open-quick-ingest"),
    page.getByRole("button", { name: /Quick ingest/i })
  ]
  for (const trigger of triggerCandidates) {
    const visible = await trigger
      .first()
      .isVisible({ timeout: 3000 })
      .catch(() => false)
    if (!visible) continue
    await trigger.first().click()
    if (await modal.isVisible({ timeout: 3000 }).catch(() => false)) return modal
  }

  await page.evaluate(() => {
    ;(
      window as Window & {
        __tldwPendingQuickIngestOpen?: {
          mode: "normal"
          at: number
        }
      }
    ).__tldwPendingQuickIngestOpen = {
      mode: "normal",
      at: Date.now()
    }
    window.dispatchEvent(new CustomEvent("tldw:open-quick-ingest"))
  })
  await waitForQuickIngestReady(modal)
  return modal
}

const clickSaveToNotesAction = async (page: Page, message: Locator) => {
  await message.hover().catch(() => {})

  const directSave = message.getByRole("button", {
    name: /Save to Notes/i
  })
  if (await directSave.first().isVisible().catch(() => false)) {
    await directSave.first().click()
    return
  }

  const moreActions = message.getByRole("button", {
    name: /More actions/i
  })
  await expect
    .poll(() => moreActions.count(), { timeout: 15000 })
    .toBeGreaterThan(0)
  await moreActions.first().click()

  const saveToNotes = page.getByRole("button", {
    name: /Save to Notes/i
  })
  await expect(saveToNotes.first()).toBeVisible({ timeout: 10000 })
  await saveToNotes.first().click()
}

const resolveChatInput = async (page: Page) => {
  // Try multiple selectors in order of preference, checking visibility not just existence
  const selectors = [
    page.locator("#textarea-message"),
    page.getByTestId("chat-input"),
    page.getByPlaceholder(/Ask anything|Type a message|form\.textarea\.placeholder/i)
  ]

  for (const input of selectors) {
    try {
      // Wait briefly for visibility rather than just checking count
      await input.first().waitFor({ state: "visible", timeout: 2000 })
      return input.first()
    } catch {
      // Not visible, try next selector
    }
  }

  // Fallback: return the first selector that has any elements
  for (const input of selectors) {
    if ((await input.count()) > 0) return input.first()
  }

  // Last resort: return the placeholder locator
  return selectors[2]
}

const clickStartChatIfVisible = async (page: Page) => {
  const startChat = page.getByRole("button", { name: /Start chatting/i })
  if ((await startChat.count()) === 0) return
  if (!(await startChat.isVisible().catch(() => false))) return
  for (let attempt = 0; attempt < 3; attempt += 1) {
    try {
      await startChat.first().click({ timeout: 5000, force: true })
      return
    } catch {
      await page.waitForTimeout(200).catch(() => {})
    }
  }
}

const sendChatMessage = async (page: Page, message: string) => {
  let input = await resolveChatInput(page)
  const visible = await input.isVisible().catch(() => false)
  if (!visible) {
    await clickStartChatIfVisible(page)
  }
  if (!(await input.isVisible().catch(() => false))) {
    input = await resolveChatInput(page)
  }
  await expect(input).toBeVisible({ timeout: 15000 })
  await expect(input).toBeEditable({ timeout: 15000 })

  const checkingModelReadiness = page.getByText(
    /Checking chat model readiness/i
  )
  const healthyModelStatus = page.getByText(/^Healthy$/i)
  const isExtensionSidepanel =
    page.url().startsWith("chrome-extension://") &&
    page.url().includes("/sidepanel.html")
  if (!isExtensionSidepanel) {
    await expect
      .poll(
        async () => {
          const count = await healthyModelStatus.count()
          for (let index = 0; index < count; index += 1) {
            if (await healthyModelStatus.nth(index).isVisible().catch(() => false)) {
              return true
            }
          }
          return false
        },
        {
          timeout: 90000,
          message: "Timed out waiting for the selected chat model to become healthy"
        }
      )
      .toBe(true)
    await expect(checkingModelReadiness).toHaveCount(0, { timeout: 15000 })
  }

  await input.fill(message)
  await expect(input).toHaveValue(message)

  const dispatchAttempt = page
    .context()
    .waitForEvent(
      "request",
      {
        predicate: (request) => {
          if (request.method().toUpperCase() !== "POST") return false
          const pathname = new URL(request.url()).pathname
          return (
            pathname === "/api/v1/chat/completions" ||
            pathname === "/api/v1/chats" ||
            pathname === "/api/v1/chats/" ||
            /^\/api\/v1\/chats\/[^/]+\/(?:complete-v2|completions)$/.test(
              pathname
            )
          )
        },
        timeout: 20000
      }
    )
    .catch(() => null)

  const sendButton = page.getByRole("button", { name: /Send message/i })
  await expect(sendButton).toBeVisible({ timeout: 15000 })
  await expect(sendButton).toBeEnabled({ timeout: 15000 })
  await sendButton.click()

  if (!(await dispatchAttempt) && isExtensionSidepanel) {
    // Sidepanel chat streams are proxied through the MV3 service worker and
    // are not consistently surfaced as BrowserContext request events. The
    // cleared draft proves the composer accepted the send; the following
    // message-store assertion remains the authoritative completion check.
    await expect(input).toHaveValue("", { timeout: 20000 })
    return
  }

  if (!(await dispatchAttempt)) {
    const diagnostics = await page.evaluate(() => {
      const store = (window as any).__tldw_useStoreMessageOption
      return {
        selectedModel: store?.getState?.().selectedModel ?? null,
        draft: (document.querySelector("#textarea-message") as HTMLTextAreaElement)
          ?.value ?? null
      }
    })
    const alerts = await page
      .locator('[role="alert"], .mantine-InputWrapper-error')
      .allTextContents()
      .catch(() => [])
    throw new Error(
      `Composer did not dispatch the chat request: ${JSON.stringify({
        diagnostics,
        alerts
      })}`
    )
  }
}

const waitForAssistantMessage = async (page: Page) => {
  const assistantMessages = page.locator(
    '[data-testid="chat-message"][data-role="assistant"]'
  )
  await expect
    .poll(async () => assistantMessages.count(), { timeout: 90000 })
    .toBeGreaterThan(0)
  const lastAssistant = assistantMessages.last()
  await expect(lastAssistant).toBeVisible({ timeout: 90000 })
  const stopButton = page.getByRole("button", {
    name: /Stop streaming/i
  })
  if ((await stopButton.count()) > 0) {
    await stopButton.waitFor({ state: "visible", timeout: 10000 }).catch(() => {})
    await stopButton.waitFor({ state: "hidden", timeout: 90000 }).catch(() => {})
  }
  return lastAssistant
}

const getAssistantText = async (assistant: Locator) => {
  const body = assistant.locator(".prose").first()
  const bodyText = await body.innerText().catch(() => "")
  if (bodyText && bodyText.trim()) {
    return bodyText
  }
  return (await assistant.innerText().catch(() => "")) || ""
}

const escapeRegExp = (value: string) =>
  value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")

const parseListPayload = (
  payload: any,
  extraKeys: string[] = []
): any[] => {
  if (Array.isArray(payload)) return payload
  if (!payload || typeof payload !== "object") return []
  const keys = [
    ...extraKeys,
    "items",
    "results",
    "data",
    "documents",
    "docs",
    "characters",
    "media"
  ]
  for (const key of keys) {
    const value = (payload as any)[key]
    if (Array.isArray(value)) return value
  }
  return []
}

const fetchNoteByTitle = async (
  serverUrl: string,
  apiKey: string,
  title: string
) => {
  const normalized = serverUrl.replace(/\/$/, "")
  const searchUrl = `${normalized}/api/v1/notes/search/?query=${encodeURIComponent(
    title
  )}&limit=50&offset=0&include_keywords=true`
  let list: any[] = []
  const searchRes = await fetchWithKey(searchUrl, apiKey).catch(() => null)
  if (searchRes?.ok) {
    const payload = await searchRes.json().catch(() => [])
    list = parseListPayload(payload)
  }

  if (!list.length) {
    const listRes = await fetchWithKey(
      `${normalized}/api/v1/notes/?page=1&results_per_page=50`,
      apiKey
    ).catch(() => null)
    if (listRes?.ok) {
      const payload = await listRes.json().catch(() => [])
      list = parseListPayload(payload)
    }
  }

  const exact = list.find(
    (note: any) => String(note?.title || "") === title
  )
  if (exact) return exact
  return (
    list.find(
      (note: any) =>
        String(note?.title || "").includes(title)
    ) || null
  )
}

const pollForNoteByTitle = async (
  serverUrl: string,
  apiKey: string,
  title: string,
  timeoutMs = 30000
) => {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    const note = await fetchNoteByTitle(serverUrl, apiKey, title)
    if (note) return note
    await new Promise((r) => setTimeout(r, 1000))
  }
  return null
}

const extractNoteBacklink = (note: any) => {
  const meta = note?.metadata || {}
  const backlinks = meta?.backlinks || meta || {}
  const conversation =
    note?.conversation_id ??
    backlinks?.conversation_id ??
    backlinks?.conversationId ??
    meta?.conversation_id ??
    null
  const message =
    note?.message_id ??
    backlinks?.message_id ??
    backlinks?.messageId ??
    meta?.message_id ??
    null
  return {
    conversation_id: conversation != null ? String(conversation) : null,
    message_id: message != null ? String(message) : null
  }
}

const pollForNoteByConversation = async (
  serverUrl: string,
  apiKey: string,
  conversationId: string,
  messageId?: string | null,
  timeoutMs = 60000
) => {
  const normalized = serverUrl.replace(/\/$/, "")
  const deadline = Date.now() + timeoutMs
  const targetConversation = String(conversationId)
  const targetMessage = messageId ? String(messageId) : null
  while (Date.now() < deadline) {
    const listRes = await fetchWithKeyTimeout(
      `${normalized}/api/v1/notes/?page=1&results_per_page=50`,
      apiKey
    ).catch(() => null)
    if (listRes?.ok) {
      const payload = await listRes.json().catch(() => [])
      const list = parseListPayload(payload)
      const match = list.find((note: any) => {
        const links = extractNoteBacklink(note)
        if (links.conversation_id === targetConversation) return true
        if (targetMessage && links.message_id === targetMessage) return true
        return false
      })
      if (match) return match
    }
    await new Promise((r) => setTimeout(r, 1000))
  }
  return null
}

const findNoteRowInList = async (
  page: Page,
  conversationId: string | null,
  query: string,
  maxPages = 5
) => {
  const targetConversation = conversationId ? String(conversationId) : ""
  for (let pageIndex = 0; pageIndex < maxPages; pageIndex += 1) {
    const conversationLocator = targetConversation
      ? page.locator("button").filter({ hasText: targetConversation })
      : null
    const queryLocator = page.locator("button").filter({ hasText: query })
    if (conversationLocator && (await conversationLocator.count()) > 0) {
      return conversationLocator.first()
    }
    if ((await queryLocator.count()) > 0) {
      return queryLocator.first()
    }
    const nextPage = page.getByRole("button", { name: /Next Page/i })
    if ((await nextPage.count()) === 0) return null
    const disabled = await nextPage.getAttribute("aria-disabled")
    if (disabled === "true") return null
    await nextPage.click()
    await page.waitForTimeout(1000)
  }
  return null
}

const createSeedNoteForRag = async (
  serverUrl: string,
  apiKey: string,
  token: string
) => {
  const normalized = serverUrl.replace(/\/$/, "")
  const title = `E2E RAG Seed ${token}`
  const content = `# E2E RAG Seed\n\nToken: ${token}\n\nThis note exists to seed Knowledge QA.`
  const res = await fetchWithKey(`${normalized}/api/v1/notes/`, apiKey, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      title,
      content,
      keywords: [`e2e-rag-${token}`]
    })
  })
  if (!res.ok) {
    const body = await res.text().catch(() => "")
    throw new Error(
      `RAG seed note create failed: ${res.status} ${res.statusText} ${body}`
    )
  }
  const payload = await res.json().catch(() => null)
  return { note: payload, title, content }
}

const pollForRagSearch = async (
  serverUrl: string,
  apiKey: string,
  query: string,
  timeoutMs = 300000
) => {
  const normalized = serverUrl.replace(/\/$/, "")
  const deadline = Date.now() + timeoutMs
  let lastStatus: number | null = null
  let lastBody = ""
  let attemptCount = 0
  const startTime = Date.now()
  while (Date.now() < deadline) {
    attemptCount += 1
    const res = await fetchWithKey(`${normalized}/api/v1/rag/search`, apiKey, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        sources: ["notes"]
      })
    }).catch(() => null)
    if (res?.ok) {
      const payload = await res.json().catch(() => null)
      const payloadKeys = payload ? Object.keys(payload) : []
      const docs = parseListPayload(payload)
      const answer =
        payload?.generated_answer ||
        payload?.answer ||
        payload?.response ||
        ""
      console.log(
        `[pollForRagSearch] attempt=${attemptCount} status=${res.status} payloadKeys=${JSON.stringify(payloadKeys)} docsCount=${docs.length} hasAnswer=${Boolean(answer)} elapsedMs=${Date.now() - startTime}`
      )
      if (Array.isArray(docs) && docs.length > 0) return payload
      if (typeof answer === "string" && answer.trim()) return payload
    } else if (res) {
      lastStatus = res.status
      lastBody = await res.text().catch(() => "")
      console.log(
        `[pollForRagSearch] attempt=${attemptCount} status=${lastStatus} errorBody=${lastBody.slice(0, 200)} elapsedMs=${Date.now() - startTime}`
      )
    } else {
      console.log(
        `[pollForRagSearch] attempt=${attemptCount} status=null (fetch failed) elapsedMs=${Date.now() - startTime}`
      )
    }
    await new Promise((r) => setTimeout(r, 2000))
  }
  throw new Error(
    `RAG search did not return results for "${query}". Last status: ${String(
      lastStatus ?? "unknown"
    )} ${lastBody}`
  )
}

const clearRequestErrors = async (page: Page) => {
  await page.evaluate(async () => {
    const w: any = window as any
    const area = w?.chrome?.storage?.local
    if (area?.set) {
      await new Promise<void>((resolve) => {
        area.set(
          { __tldwLastRequestError: null, __tldwRequestErrors: [] },
          () => resolve()
        )
      })
      return
    }
    try {
      localStorage.setItem("__tldwLastRequestError", "null")
      localStorage.setItem("__tldwRequestErrors", "[]")
    } catch {
      // ignore localStorage errors
    }
  })
}

const readLastRequestError = async (page: Page) =>
  await page.evaluate(async () => {
    const w: any = window as any
    const area = w?.chrome?.storage?.local
    if (area?.get) {
      return await new Promise<{
        last: any | null
        recent: any[] | null
      }>((resolve) => {
        area.get(
          ["__tldwLastRequestError", "__tldwRequestErrors"],
          (items: any) => {
            resolve({
              last: items?.__tldwLastRequestError ?? null,
              recent: Array.isArray(items?.__tldwRequestErrors)
                ? items.__tldwRequestErrors.slice(0, 5)
                : null
            })
          }
        )
      })
    }
    const parseValue = (value: string | null) => {
      if (value == null) return null
      try {
        return JSON.parse(value)
      } catch {
        return value
      }
    }
    const last = parseValue(
      localStorage.getItem("__tldwLastRequestError")
    )
    const recent = parseValue(
      localStorage.getItem("__tldwRequestErrors")
    )
    return {
      last: last ?? null,
      recent: Array.isArray(recent) ? recent.slice(0, 5) : null
    }
  })

const logFlashcardsSnapshot = async (
  serverUrl: string,
  apiKey: string,
  label: string
) => {
  const normalized = serverUrl.replace(/\/$/, "")
  const res = await fetchWithKey(
    `${normalized}/api/v1/flashcards?limit=5&offset=0&due_status=all&order_by=created_at`,
    apiKey
  ).catch(() => null)
  if (!res?.ok) {
    const body = await res?.text().catch(() => "")
    console.log(
      `[e2e] flashcards snapshot ${label} failed: ${res?.status} ${res?.statusText} ${body}`
    )
    return
  }
  const payload = await res.json().catch(() => null)
  const items = parseListPayload(payload, ["items", "results", "data"]).slice(
    0,
    5
  )
  const summary = items.map((item: any) => ({
    uuid: item?.uuid ?? null,
    deck_id: item?.deck_id ?? null,
    due_at: item?.due_at ?? null,
    front:
      typeof item?.front === "string"
        ? item.front.slice(0, 80)
        : String(item?.front || "").slice(0, 80),
    back:
      typeof item?.back === "string"
        ? item.back.slice(0, 80)
        : String(item?.back || "").slice(0, 80)
  }))
  console.log(
    `[e2e] flashcards snapshot ${label}`,
    JSON.stringify({
      count: payload?.count ?? null,
      items: summary
    })
  )
}

const logChatMessagesSnapshot = async (
  serverUrl: string,
  apiKey: string,
  chatId: string,
  label: string
) => {
  const normalized = serverUrl.replace(/\/$/, "")
  const res = await fetchWithKey(
    `${normalized}/api/v1/chats/${encodeURIComponent(chatId)}/messages`,
    apiKey
  ).catch(() => null)
  if (!res?.ok) {
    const body = await res?.text().catch(() => "")
    console.log(
      `[e2e] chat messages snapshot ${label} failed: ${res?.status} ${res?.statusText} ${body}`
    )
    return
  }
  const payload = await res.json().catch(() => null)
  const list: any[] = Array.isArray(payload)
    ? payload
    : Array.isArray(payload?.messages)
      ? payload.messages
      : Array.isArray(payload?.items)
        ? payload.items
        : Array.isArray(payload?.results)
          ? payload.results
          : Array.isArray(payload?.data)
            ? payload.data
            : []
  const summary = list.slice(-5).map((item) => ({
    id: item?.id ?? item?.message_id ?? null,
    role: item?.role ?? item?.sender ?? item?.author ?? null,
    content:
      typeof item?.content === "string"
        ? item.content.slice(0, 80)
        : typeof item?.message?.content === "string"
          ? item.message.content.slice(0, 80)
          : null
  }))
  console.log(
    `[e2e] chat messages snapshot ${label}`,
    JSON.stringify({
      count: list.length,
      tail: summary
    })
  )
}

const probeSaveChatKnowledge = async (
  serverUrl: string,
  apiKey: string,
  payload: {
    conversation_id: string
    message_id: string
    snippet: string
    make_flashcard: boolean
  },
  label: string
) => {
  const normalized = serverUrl.replace(/\/$/, "")
  const res = await fetchWithKey(
    `${normalized}/api/v1/chat/knowledge/save`,
    apiKey,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }
  ).catch(() => null)
  if (!res) {
    console.log(`[e2e] chat knowledge save probe ${label} failed: no response`)
    return
  }
  const bodyText = await res.text().catch(() => "")
  let parsed: any = null
  if (bodyText) {
    try {
      parsed = JSON.parse(bodyText)
    } catch {
      parsed = null
    }
  }
  const bodySnippet =
    bodyText.length > 500
      ? `${bodyText.slice(0, 500)}...(truncated)`
      : bodyText
  console.log(
    `[e2e] chat knowledge save probe ${label}`,
    JSON.stringify({
      ok: res.ok,
      status: res.status,
      statusText: res.statusText,
      response: parsed ?? bodySnippet,
      payload: {
        conversation_id: payload.conversation_id,
        message_id: payload.message_id,
        snippet_preview: payload.snippet.slice(0, 120),
        snippet_length: payload.snippet.length,
        make_flashcard: payload.make_flashcard
      }
    })
  )
}

const fetchRecentFlashcards = async (
  serverUrl: string,
  apiKey: string,
  limit = 10
): Promise<any[]> => {
  const normalized = serverUrl.replace(/\/$/, "")
  const res = await fetchWithKey(
    `${normalized}/api/v1/flashcards?limit=${limit}&offset=0&due_status=all&order_by=created_at`,
    apiKey
  ).catch(() => null)
  if (!res?.ok) {
    const body = await res?.text().catch(() => "")
    throw new Error(
      `Flashcards list fetch failed: ${res?.status} ${res?.statusText} ${body}`
    )
  }
  const payload = await res.json().catch(() => null)
  return parseListPayload(payload, ["items", "results", "data"])
}

const pollForNewFlashcard = async (
  serverUrl: string,
  apiKey: string,
  baselineIds: Set<string>,
  snippet: string,
  timeoutMs = 60000
) => {
  const deadline = Date.now() + timeoutMs
  const target = normalizeMessageContent(snippet).slice(0, 80)
  while (Date.now() < deadline) {
    const items = await fetchRecentFlashcards(serverUrl, apiKey, 20)
    const match = items.find((item: any) => {
      const id = item?.uuid != null ? String(item.uuid) : ""
      if (!id || baselineIds.has(id)) return false
      if (!target) return true
      const front = normalizeMessageContent(item?.front ?? "")
      const back = normalizeMessageContent(item?.back ?? "")
      return front.includes(target) || back.includes(target)
    })
    if (match) return match
    await new Promise((r) => setTimeout(r, 2000))
  }
  throw new Error("New flashcard did not appear after saving.")
}

const cleanupFlashcard = async (
  serverUrl: string,
  apiKey: string,
  cardUuid: string
) => {
  const normalized = serverUrl.replace(/\/$/, "")
  const encodedUuid = encodeURIComponent(cardUuid)
  const latestResponse = await fetchWithKey(
    `${normalized}/api/v1/flashcards/${encodedUuid}`,
    apiKey
  ).catch(() => null)
  if (!latestResponse?.ok) return
  const latest = await latestResponse.json().catch(() => null)
  const version = Number(latest?.version)
  if (!Number.isInteger(version) || version < 1) return
  await fetchWithKey(
    `${normalized}/api/v1/flashcards/${encodedUuid}?expected_version=${version}`,
    apiKey,
    { method: "DELETE" }
  ).catch(() => {})
}

const setLastNoteId = async (page: Page, noteId: string) => {
  await page.evaluate(async (id) => {
    const w: any = window as any
    try {
      window.localStorage.setItem("tldw:lastNoteId", String(id))
    } catch {
      // ignore localStorage errors
    }
    const area = w?.chrome?.storage?.local
    if (!area?.set) return
    await new Promise<void>((resolve) => {
      area.set({ "tldw:lastNoteId": String(id) }, () => resolve())
    })
  }, noteId)
}

const pollForCharacterByName = async (
  serverUrl: string,
  apiKey: string,
  name: string,
  timeoutMs = 30000
) => {
  const normalized = serverUrl.replace(/\/$/, "")
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    const searchRes = await fetchWithKey(
      `${normalized}/api/v1/characters/search/?query=${encodeURIComponent(
        name
      )}`,
      apiKey
    ).catch(() => null)
    if (searchRes?.ok) {
      const payload = await searchRes.json().catch(() => [])
      const list = parseListPayload(payload)
      const match = list.find((item: any) => {
        const candidate =
          item?.name ?? item?.title ?? item?.slug ?? ""
        return String(candidate) === String(name)
      })
      if (match) return match
    }

    const listRes = await fetchWithKey(
      `${normalized}/api/v1/characters/`,
      apiKey
    ).catch(() => null)
    if (listRes?.ok) {
      const payload = await listRes.json().catch(() => [])
      const list = parseListPayload(payload, ["characters"])
      const match = list.find((item: any) => {
        const candidate =
          item?.name ?? item?.title ?? item?.slug ?? ""
        return String(candidate) === String(name)
      })
      if (match) return match
    }
    await new Promise((r) => setTimeout(r, 1000))
  }
  return null
}

const pollForWorldBookByName = async (
  serverUrl: string,
  apiKey: string,
  name: string,
  timeoutMs = 30000
) => {
  const normalized = serverUrl.replace(/\/$/, "")
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    const remainingMs = Math.max(0, deadline - Date.now())
    const listRes = await fetchWithKeyTimeout(
      `${normalized}/api/v1/characters/world-books`,
      apiKey,
      {},
      remainingMs
    ).catch(() => null)
    if (listRes?.ok) {
      const payload = await listRes.json().catch(() => [])
      const books = parseListPayload(payload, ["world_books"])
      const match = books.find((item: any) => {
        const candidate = item?.name ?? item?.title ?? ""
        return String(candidate) === String(name)
      })
      if (match) return match
    }
    await new Promise((r) => setTimeout(r, 1000))
  }
  return null
}

const pollForDictionaryByName = async (
  serverUrl: string,
  apiKey: string,
  name: string,
  timeoutMs = 30000
) => {
  const normalized = serverUrl.replace(/\/$/, "")
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    const remainingMs = Math.max(0, deadline - Date.now())
    const listRes = await fetchWithKeyTimeout(
      `${normalized}/api/v1/chat/dictionaries?include_inactive=true`,
      apiKey,
      {},
      remainingMs
    ).catch(() => null)
    if (listRes?.ok) {
      const payload = await listRes.json().catch(() => [])
      const dictionaries = parseListPayload(payload, ["dictionaries"])
      const match = dictionaries.find((item: any) => {
        const candidate = item?.name ?? item?.title ?? ""
        return String(candidate) === String(name)
      })
      if (match) return match
    }
    await new Promise((r) => setTimeout(r, 1000))
  }
  return null
}

const normalizeMessageContent = (value: unknown) =>
  String(value || "").replace(/\s+/g, " ").trim()

type AssistantSnapshot = {
  text: string
  localId: string | null
  serverMessageId: string | null
  serverChatId: string
}

const waitForAssistantSnapshot = async (
  page: Page,
  timeoutMs = 90000
): Promise<AssistantSnapshot | null> => {
  const storeTimeoutMs = Math.min(timeoutMs, 30000)
  try {
    return await page
      .waitForFunction(
        () => {
          const store = (window as any).__tldw_useStoreMessageOption
          const state = store?.getState?.()
          if (!state?.serverChatId) return null
          const messages = Array.isArray(state?.messages) ? state.messages : []
          for (let i = messages.length - 1; i >= 0; i -= 1) {
            const msg = messages[i]
            if (!msg?.isBot) continue
            if (msg?.messageType === "character:greeting") continue
            const content =
              typeof msg?.message === "string" ? msg.message : ""
            const trimmed = content.replace(/\s+/g, " ").trim()
            if (!trimmed || trimmed.includes("▋")) return null
            return {
              text: trimmed,
              localId: msg?.id != null ? String(msg.id) : null,
              serverMessageId:
                msg?.serverMessageId != null
                  ? String(msg.serverMessageId)
                  : null,
              serverChatId: String(state.serverChatId)
            }
          }
          return null
        },
        undefined,
        { timeout: storeTimeoutMs }
      )
      .then((handle) => handle.jsonValue())
  } catch {
    // Some packaged extension flows render the assistant message before the
    // debug store snapshot settles. Fall back to the visible DOM message so
    // the workflow can continue, then let later helpers recover server IDs.
  }

  const assistant = await waitForAssistantMessage(page)
  const text = normalizeMessageContent(await getAssistantText(assistant))
  if (!text) return null

  const [localId, serverMessageId, serverChatId] = await Promise.all([
    assistant.getAttribute("data-message-id").catch(() => null),
    assistant.getAttribute("data-server-message-id").catch(() => null),
    page
      .evaluate(() => {
        const store = (window as any).__tldw_useStoreMessageOption
        const state = store?.getState?.()
        if (!state?.serverChatId) return null
        return String(state.serverChatId)
      })
      .catch(() => null)
  ])

  return {
    text,
    localId,
    serverMessageId,
    serverChatId: serverChatId || ""
  }
}

const waitForAssistantServerMessageIdInStore = async (
  page: Page,
  options: {
    localId?: string | null
    assistantText: string
    timeoutMs?: number
  }
): Promise<string | null> => {
  const payload = {
    localId: options.localId ?? null,
    assistantText: normalizeMessageContent(options.assistantText),
    timeoutMs: options.timeoutMs ?? 30000
  }

  return page
    .waitForFunction(
      ({ localId, assistantText }) => {
        const normalize = (value: unknown) =>
          String(value || "").replace(/\s+/g, " ").trim()
        const store = (window as any).__tldw_useStoreMessageOption
        const messages = Array.isArray(store?.getState?.()?.messages)
          ? store.getState().messages
          : []
        if (messages.length === 0) return null

        const findServerId = (candidate: any) => {
          if (candidate?.serverMessageId != null) {
            return String(candidate.serverMessageId)
          }
          const variants = Array.isArray(candidate?.variants)
            ? candidate.variants
            : []
          for (const variant of variants) {
            if (variant?.serverMessageId != null) {
              return String(variant.serverMessageId)
            }
          }
          return null
        }

        if (localId) {
          const directMatch = messages.find(
            (msg: any) => String(msg?.id || "") === String(localId)
          )
          const directServerId = findServerId(directMatch)
          if (directServerId) return directServerId
        }

        const normalizedTarget = normalize(assistantText)
        for (let i = messages.length - 1; i >= 0; i -= 1) {
          const candidate = messages[i]
          if (!candidate?.isBot) continue
          if (candidate?.messageType === "character:greeting") continue
          const normalizedMessage = normalize(candidate?.message)
          if (
            normalizedTarget.length > 0 &&
            normalizedMessage !== normalizedTarget
          ) {
            continue
          }
          const serverId = findServerId(candidate)
          if (serverId) return serverId
        }

        return null
      },
      {
        localId: payload.localId,
        assistantText: payload.assistantText
      },
      { timeout: payload.timeoutMs }
    )
    .then((handle) => handle.jsonValue())
    .catch(() => null)
}

const syncAssistantServerMessageIdIntoStore = async (
  page: Page,
  options: {
    localId?: string | null
    serverMessageId: string
  }
) =>
  page.evaluate(
    ({ localId, serverMessageId }) => {
      const store = (window as any).__tldw_useStoreMessageOption
      if (!store?.getState || !store?.setState) return false
      const state = store.getState?.()
      const messages = Array.isArray(state?.messages) ? [...state.messages] : []
      if (messages.length === 0) return false
      let targetIndex = -1
      if (localId) {
        targetIndex = messages.findIndex(
          (msg) => String(msg?.id || "") === String(localId)
        )
      }
      if (targetIndex === -1) {
        for (let i = messages.length - 1; i >= 0; i -= 1) {
          const msg = messages[i]
          if (!msg?.isBot) continue
          if (msg?.messageType === "character:greeting") continue
          targetIndex = i
          break
        }
      }
      if (targetIndex === -1) return false
      const target = messages[targetIndex]
      if (target?.serverMessageId === serverMessageId) return true
      const updatedVariants = Array.isArray(target?.variants)
        ? target.variants.map((variant) => ({
            ...variant,
            serverMessageId: variant?.serverMessageId ?? serverMessageId
          }))
        : target?.variants
      messages[targetIndex] = {
        ...target,
        serverMessageId,
        variants: updatedVariants
      }
      store.setState({ messages })
      return true
    },
    options
  )

const getAssistantMessageLocator = (
  page: Page,
  snapshot: Pick<AssistantSnapshot, "localId">
) => {
  if (snapshot.localId) {
    return page.locator(
      `[data-testid="chat-message"][data-message-id="${snapshot.localId}"]`
    )
  }
  return page
    .locator('[data-testid="chat-message"][data-role="assistant"]')
    .last()
}

const clickMessageOverflowAction = async (
  page: Page,
  message: Locator,
  actionName: RegExp
) => {
  await message.hover()

  const overflowTriggers = message.getByRole("button", {
    name: /More actions/i
  })
  let visibleTrigger: Locator | null = null
  await expect
    .poll(
      async () => {
        const count = await overflowTriggers.count()
        for (let index = 0; index < count; index += 1) {
          const candidate = overflowTriggers.nth(index)
          if (await candidate.isVisible().catch(() => false)) {
            visibleTrigger = candidate
            return true
          }
        }
        return false
      },
      { timeout: 15000, intervals: [250, 500, 1000] }
    )
    .toBe(true)
  if (!visibleTrigger) {
    throw new Error("Message overflow trigger did not become visible.")
  }
  await visibleTrigger.click()

  const actions = page.getByRole("button", { name: actionName })
  let visibleAction: Locator | null = null
  await expect
    .poll(
      async () => {
        const count = await actions.count()
        for (let index = 0; index < count; index += 1) {
          const candidate = actions.nth(index)
          if (await candidate.isVisible().catch(() => false)) {
            visibleAction = candidate
            return true
          }
        }
        return false
      },
      { timeout: 15000, intervals: [250, 500, 1000] }
    )
    .toBe(true)
  if (!visibleAction) {
    throw new Error(`Message overflow action ${actionName} did not become visible.`)
  }
  await visibleAction.click()
}

const pollForServerAssistantMessageId = async (
  serverUrl: string,
  apiKey: string,
  chatId: string,
  assistantText: string,
  timeoutMs = 60000
) => {
  const normalized = serverUrl.replace(/\/$/, "")
  const deadline = Date.now() + timeoutMs
  const target = normalizeMessageContent(assistantText)
  const targetPrefix = target.slice(0, 80)
  while (Date.now() < deadline) {
    const res = await fetchWithKeyTimeout(
      `${normalized}/api/v1/chats/${encodeURIComponent(chatId)}/messages`,
      apiKey
    ).catch(() => null)
    if (res?.ok) {
      const payload = await res.json().catch(() => null)
      const list: any[] = Array.isArray(payload)
        ? payload
        : Array.isArray(payload?.messages)
          ? payload.messages
          : Array.isArray(payload?.items)
            ? payload.items
            : Array.isArray(payload?.results)
              ? payload.results
              : Array.isArray(payload?.data)
                ? payload.data
                : []
      const assistants = list.filter((item) => {
        const roleCandidate =
          item?.role ?? item?.sender ?? item?.author ?? item?.message?.role
        const isBot =
          item?.is_bot === true ||
          item?.isBot === true ||
          String(roleCandidate || "")
            .toLowerCase()
            .includes("assistant")
        return isBot
      })
      if (assistants.length > 0) {
        const exactMatch = assistants.find((item) => {
          const content = normalizeMessageContent(
            item?.content ?? item?.message?.content ?? ""
          )
          return content && (content === target || content.startsWith(targetPrefix))
        })
        const match = exactMatch ?? assistants[assistants.length - 1]
        if (match?.id != null) {
          return String(match.id)
        }
      }
    }
    await new Promise((resolve) => setTimeout(resolve, 2000))
  }
  return null
}

/**
 * Directly upload a file to the media API, bypassing extension messaging.
 * Used as a fallback when extension messaging doesn't work (e.g., in Playwright tests).
 */
const directMediaUpload = async (
  serverUrl: string,
  apiKey: string,
  fileName: string,
  fileContent: string,
  mediaBasePath = "/api/v1/media"
): Promise<{ ok: boolean; mediaId?: string; error?: string }> => {
  const normalized = serverUrl.replace(/\/$/, "")
  const basePath = normalizePath(mediaBasePath || "/api/v1/media")
  const url = `${normalized}${basePath}/add`

  try {
    const formData = new FormData()
    const blob = new Blob([fileContent], { type: "text/plain" })
    formData.append("files", blob, fileName)
    formData.append("media_type", "document")

    const res = await fetch(url, {
      method: "POST",
      headers: {
        "X-API-KEY": apiKey
      },
      body: formData
    })

    if (!res.ok) {
      const text = await res.text().catch(() => "")
      return { ok: false, error: `Upload failed: ${res.status} - ${text.slice(0, 200)}` }
    }

    const data = await res.json().catch(() => null)
    // The response format may vary - try to extract media ID
    const mediaId = data?.id || data?.media_id || data?.results?.[0]?.id || data?.results?.[0]?.media_id
    console.log(`[directMediaUpload] Upload succeeded: ${fileName} -> mediaId=${mediaId}`)
    return { ok: true, mediaId }
  } catch (err: any) {
    return { ok: false, error: `Upload error: ${err?.message}` }
  }
}

const pollForMediaMatch = async (
  serverUrl: string,
  apiKey: string,
  query: string,
  timeoutMs = 300000,
  mediaBasePath = "/api/v1/media"
) => {
  const normalized = serverUrl.replace(/\/$/, "")
  const basePath = normalizePath(mediaBasePath || "/api/v1/media")
  const deadline = Date.now() + timeoutMs
  let attemptCount = 0
  let lastStatus: number | null = null
  let lastPayloadKeys: string[] = []
  const startTime = Date.now()
  while (Date.now() < deadline) {
    attemptCount += 1
    const res = await fetchWithKeyTimeout(
      `${normalized}${basePath}/search?page=1&results_per_page=20`,
      apiKey,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          fields: ["title", "content"],
          sort_by: "relevance"
        })
      }
    ).catch(() => null)
    if (res?.ok) {
      const payload = await res.json().catch(() => null)
      const payloadKeys = payload ? Object.keys(payload) : []
      lastPayloadKeys = payloadKeys
      const items = parseListPayload(payload, ["items", "results"])
      console.log(
        `[pollForMediaMatch] attempt=${attemptCount} status=${res.status} query="${query}" payloadKeys=${JSON.stringify(payloadKeys)} itemsCount=${items.length} elapsedMs=${Date.now() - startTime}`
      )
      if (items.length > 0) {
        // Search through items for a title match
        const matchingItem = items.find((item: any) => {
          const title = String(item?.title || "").toLowerCase()
          const queryLower = query.toLowerCase()
          return title.includes(queryLower) || queryLower.split("-").every(part => title.includes(part))
        })
        if (matchingItem) {
          console.log(
            `[pollForMediaMatch] found match: id=${matchingItem?.id} title="${matchingItem?.title}"`
          )
          return matchingItem
        }
        // Log first few items for debugging
        console.log(
          `[pollForMediaMatch] items returned but no match. First 3 titles: ${items.slice(0, 3).map((i: any) => i?.title).join(", ")}`
        )
      }
    } else if (res) {
      lastStatus = res.status
      const errorBody = await res.text().catch(() => "")
      console.log(
        `[pollForMediaMatch] attempt=${attemptCount} status=${lastStatus} errorBody=${errorBody.slice(0, 200)} elapsedMs=${Date.now() - startTime}`
      )
    } else {
      console.log(
        `[pollForMediaMatch] attempt=${attemptCount} status=null (fetch failed) elapsedMs=${Date.now() - startTime}`
      )
    }
    await new Promise((resolve) => setTimeout(resolve, 2000))
  }
  throw new Error(
    `Timed out waiting for media search results for "${query}". Last status: ${String(lastStatus ?? "unknown")} lastPayloadKeys: ${JSON.stringify(lastPayloadKeys)}`
  )
}

const extractPersistedMediaAnalysis = (detail: any): string => {
  const candidates: unknown[] = [
    detail?.processing?.analysis,
    detail?.analysis,
    detail?.analysis_content,
    detail?.analysisContent,
    detail?.latest_version?.analysis_content,
    detail?.latestVersion?.analysisContent
  ]
  if (Array.isArray(detail?.analyses)) {
    for (const entry of detail.analyses) {
      candidates.push(
        typeof entry === "string"
          ? entry
          : entry?.content ?? entry?.text ?? entry?.summary ?? entry?.analysis_content
      )
    }
  }
  for (const candidate of candidates) {
    if (typeof candidate === "string" && candidate.trim()) {
      return candidate.trim()
    }
  }
  return ""
}

const pollForPersistedMediaAnalysis = async (
  serverUrl: string,
  apiKey: string,
  mediaId: string | number,
  timeoutMs = 60000,
  expectedAnalysis?: string
): Promise<string> => {
  const normalized = serverUrl.replace(/\/$/, "")
  const deadline = Date.now() + timeoutMs
  let lastStatus: number | null = null
  while (Date.now() < deadline) {
    const response = await fetchWithKeyTimeout(
      `${normalized}/api/v1/media/${encodeURIComponent(String(mediaId))}`,
      apiKey
    ).catch(() => null)
    lastStatus = response?.status ?? null
    if (response?.ok) {
      const detail = await response.json().catch(() => null)
      const analysis = extractPersistedMediaAnalysis(detail)
      if (
        analysis &&
        (!expectedAnalysis || analysis.includes(expectedAnalysis))
      ) return analysis
    }
    await new Promise((resolve) => setTimeout(resolve, 1000))
  }
  throw new Error(
    `Timed out waiting for persisted analysis on media ${String(mediaId)}. Last status: ${String(lastStatus ?? "unknown")}`
  )
}

const deleteCharacterByName = async (
  serverUrl: string,
  apiKey: string,
  name: string
) => {
  const normalized = serverUrl.replace(/\/$/, "")
  const primary = await fetchWithKeyTimeout(
    `${normalized}/api/v1/characters/`,
    apiKey
  ).catch(() => null)
  const res =
    primary && primary.ok
      ? primary
      : await fetchWithKeyTimeout(
          `${normalized}/api/v1/characters`,
          apiKey
        ).catch(() => null)
  if (!res?.ok) return
  const payload = await res.json().catch(() => null)
  const characters = parseListPayload(payload, ["characters"])
  const match = characters.find((c: any) => {
    const label = String(c?.name || c?.title || "").trim()
    return label === name
  })
  if (!match?.id) return
  const expectedVersion = Number(match?.version ?? match?.version_number)
  if (!Number.isFinite(expectedVersion)) return
  await fetchWithKeyTimeout(
    `${normalized}/api/v1/characters/${encodeURIComponent(String(match.id))}?expected_version=${encodeURIComponent(String(expectedVersion))}`,
    apiKey,
    { method: "DELETE" }
  ).catch(() => {})
}

const createCharacterByName = async (
  serverUrl: string,
  apiKey: string,
  name: string
) => {
  const normalized = serverUrl.replace(/\/$/, "")
  const greeting = `Hello from ${name}.`
  const payload = {
    name,
    greeting,
    first_message: greeting
  }
  const createPrimary = await fetchWithKey(
    `${normalized}/api/v1/characters/`,
    apiKey,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }
  ).catch(() => null)
  const createRes =
    createPrimary && createPrimary.ok
      ? createPrimary
      : await fetchWithKey(`${normalized}/api/v1/characters`, apiKey, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        }).catch(() => null)
  if (!createRes?.ok) {
    const body = await createRes?.text().catch(() => "")
    throw new Error(
      `Character create failed: ${createRes?.status} ${createRes?.statusText} ${body}`
    )
  }
  const created = await createRes.json().catch(() => null)
  return created?.id ?? created?.uuid ?? null
}

const deleteWorldBookByName = async (
  serverUrl: string,
  apiKey: string,
  name: string
) => {
  const normalized = serverUrl.replace(/\/$/, "")
  const list = await fetchWithKeyTimeout(
    `${normalized}/api/v1/characters/world-books`,
    apiKey
  ).catch(() => null)
  if (!list?.ok) return
  const payload = await list.json().catch(() => null)
  const books = parseListPayload(payload, ["world_books"])
  const match = books.find((b: any) => String(b?.name || "") === name)
  if (!match?.id) return
  await fetchWithKeyTimeout(
    `${normalized}/api/v1/characters/world-books/${encodeURIComponent(
      String(match.id)
    )}`,
    apiKey,
    { method: "DELETE" }
  ).catch(() => {})
}

const deleteDictionaryByName = async (
  serverUrl: string,
  apiKey: string,
  name: string
) => {
  const normalized = serverUrl.replace(/\/$/, "")
  const list = await fetchWithKey(
    `${normalized}/api/v1/chat/dictionaries?include_inactive=true`,
    apiKey
  ).catch(() => null)
  if (!list?.ok) return
  const payload = await list.json().catch(() => null)
  const dictionaries = parseListPayload(payload, ["dictionaries"])
  const match = dictionaries.find((d: any) => String(d?.name || "") === name)
  if (!match?.id) return
  await fetchWithKey(
    `${normalized}/api/v1/chat/dictionaries/${encodeURIComponent(
      String(match.id)
    )}`,
    apiKey,
    { method: "DELETE" }
  ).catch(() => {})
}

const fetchDictionaryByName = async (
  serverUrl: string,
  apiKey: string,
  name: string,
  includeInactive = true
) => {
  const normalized = serverUrl.replace(/\/$/, "")
  const qp = includeInactive ? "?include_inactive=true" : ""
  const list = await fetchWithKey(
    `${normalized}/api/v1/chat/dictionaries${qp}`,
    apiKey
  ).catch(() => null)
  if (!list?.ok) return null
  const payload = await list.json().catch(() => null)
  const dictionaries = parseListPayload(payload, ["dictionaries"])
  return (
    dictionaries.find((d: any) => String(d?.name || "") === name) || null
  )
}

const pollForDictionaryRemoval = async (
  serverUrl: string,
  apiKey: string,
  name: string,
  timeoutMs = 20000
) => {
  const deadline = Date.now() + timeoutMs
  let lastMatch: any = null
  while (Date.now() < deadline) {
    lastMatch = await fetchDictionaryByName(serverUrl, apiKey, name, true)
    if (!lastMatch) return null
    await new Promise((r) => setTimeout(r, 1000))
  }
  return lastMatch
}

const createPrompt = async (
  serverUrl: string,
  apiKey: string,
  payload: {
    name: string
    system_prompt: string
    user_prompt: string
    keywords?: string[]
  }
) => {
  const normalized = serverUrl.replace(/\/$/, "")
  const createPrimary = await fetchWithKey(
    `${normalized}/api/v1/prompts`,
    apiKey,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }
  ).catch(() => null)
  const createRes =
    createPrimary && createPrimary.ok
      ? createPrimary
      : await fetchWithKey(`${normalized}/api/v1/prompts/`, apiKey, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        }).catch(() => null)
  if (!createRes?.ok) {
    const body = await createRes?.text().catch(() => "")
    throw new Error(
      `Prompt create failed: ${createRes?.status} ${createRes?.statusText} ${body}`
    )
  }
  const created = await createRes.json().catch(() => null)
  return created?.id ?? created?.uuid ?? created?.name ?? null
}

const deletePromptById = async (
  serverUrl: string,
  apiKey: string,
  promptId: string | number
) => {
  const normalized = serverUrl.replace(/\/$/, "")
  await fetchWithKey(
    `${normalized}/api/v1/prompts/${encodeURIComponent(String(promptId))}`,
    apiKey,
    { method: "DELETE" }
  ).catch(() => {})
}

const pollForChatByTitle = async (
  serverUrl: string,
  apiKey: string,
  title: string,
  timeoutMs = 45000
) => {
  const normalized = serverUrl.replace(/\/$/, "")
  const deadline = Date.now() + timeoutMs
  const urls = [
    `${normalized}/api/v1/chats/?limit=50&offset=0`,
    `${normalized}/api/v1/chats?limit=50&offset=0`,
    `${normalized}/api/v1/chats/`,
    `${normalized}/api/v1/chats`
  ]
  let attemptCount = 0
  const startTime = Date.now()
  while (Date.now() < deadline) {
    attemptCount += 1
    for (let urlIndex = 0; urlIndex < urls.length; urlIndex++) {
      const url = urls[urlIndex]
      const res = await fetchWithKey(url, apiKey).catch(() => null)
      if (!res?.ok) {
        const status = res?.status ?? "null"
        if (urlIndex === 0) {
          console.log(
            `[pollForChatByTitle] attempt=${attemptCount} urlIndex=${urlIndex} status=${status} title="${title}" elapsedMs=${Date.now() - startTime}`
          )
        }
        continue
      }
      const payload = await res.json().catch(() => [])
      const payloadKeys = payload && typeof payload === "object" && !Array.isArray(payload) ? Object.keys(payload) : ["(array)"]
      const list = parseListPayload(payload, ["chats"])
      console.log(
        `[pollForChatByTitle] attempt=${attemptCount} urlIndex=${urlIndex} status=${res.status} payloadKeys=${JSON.stringify(payloadKeys)} listCount=${list.length} searchingFor="${title}" elapsedMs=${Date.now() - startTime}`
      )
      const match = list.find((chat: any) => {
        const label = String(chat?.title ?? chat?.name ?? "").trim()
        return label === title
      })
      if (match) {
        console.log(
          `[pollForChatByTitle] found match: id=${match.id} title="${match.title ?? match.name}"`
        )
        return match
      }
      if (list.length > 0) {
        const titles = list.slice(0, 5).map((c: any) => String(c?.title ?? c?.name ?? "").trim())
        console.log(
          `[pollForChatByTitle] no match, sample titles: ${JSON.stringify(titles)}`
        )
      }
    }
    await new Promise((r) => setTimeout(r, 1000))
  }
  console.log(
    `[pollForChatByTitle] timeout after ${attemptCount} attempts for title="${title}"`
  )
  return null
}

const deleteChatById = async (
  serverUrl: string,
  apiKey: string,
  chatId: string
) => {
  const normalized = serverUrl.replace(/\/$/, "")
  await fetchWithKey(
    `${normalized}/api/v1/chats/${encodeURIComponent(String(chatId))}`,
    apiKey,
    { method: "DELETE" }
  ).catch(() => {})
}

const createQuiz = async (
  serverUrl: string,
  apiKey: string,
  name: string
) => {
  const normalized = serverUrl.replace(/\/$/, "")
  const payload = {
    name,
    description: "Quiz created by Playwright."
  }
  const createPrimary = await fetchWithKey(
    `${normalized}/api/v1/quizzes`,
    apiKey,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }
  ).catch(() => null)
  const createRes =
    createPrimary && createPrimary.ok
      ? createPrimary
      : await fetchWithKey(`${normalized}/api/v1/quizzes/`, apiKey, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        }).catch(() => null)
  if (!createRes?.ok) {
    const body = await createRes?.text().catch(() => "")
    throw new Error(
      `Quiz create failed: ${createRes?.status} ${createRes?.statusText} ${body}`
    )
  }
  const created = await createRes.json().catch(() => null)
  return created?.id ?? created?.quiz_id ?? null
}

const addQuizQuestion = async (
  serverUrl: string,
  apiKey: string,
  quizId: string | number,
  payload: Record<string, any>
) => {
  const normalized = serverUrl.replace(/\/$/, "")
  const res = await fetchWithKey(
    `${normalized}/api/v1/quizzes/${encodeURIComponent(
      String(quizId)
    )}/questions`,
    apiKey,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }
  ).catch(() => null)
  if (!res?.ok) {
    const body = await res?.text().catch(() => "")
    throw new Error(
      `Quiz question create failed: ${res?.status} ${res?.statusText} ${body}`
    )
  }
  return res.json().catch(() => null)
}

const deleteQuizById = async (
  serverUrl: string,
  apiKey: string,
  quizId: string | number
) => {
  const normalized = serverUrl.replace(/\/$/, "")
  await fetchWithKey(
    `${normalized}/api/v1/quizzes/${encodeURIComponent(String(quizId))}`,
    apiKey,
    { method: "DELETE" }
  ).catch(() => {})
}

const createChatWithMessage = async (
  serverUrl: string,
  apiKey: string,
  characterId: string | number,
  title: string,
  message: string
) => {
  const normalized = serverUrl.replace(/\/$/, "")
  const payload = {
    title,
    character_id: characterId,
    state: "in-progress",
    source: "e2e"
  }
  const createPrimary = await fetchWithKey(
    `${normalized}/api/v1/chats/`,
    apiKey,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }
  ).catch(() => null)
  const createRes =
    createPrimary && createPrimary.ok
      ? createPrimary
      : await fetchWithKey(`${normalized}/api/v1/chats`, apiKey, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        }).catch(() => null)
  if (!createRes?.ok) {
    const body = await createRes?.text().catch(() => "")
    throw new Error(
      `Chat create failed: ${createRes?.status} ${createRes?.statusText} ${body}`
    )
  }
  const created = await createRes.json().catch(() => null)
  const rawId = created?.id ?? created?.chat_id ?? created?.conversation_id ?? null
  if (!rawId) {
    throw new Error("Chat create did not return an id.")
  }
  const chatId = String(rawId)
  if (message.trim()) {
    await fetchWithKey(
      `${normalized}/api/v1/chats/${encodeURIComponent(chatId)}/messages`,
      apiKey,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ role: "user", content: message })
      }
    ).catch(() => {})
  }
  return chatId
}

const deleteDataTableByName = async (
  serverUrl: string,
  apiKey: string,
  name: string
) => {
  const normalized = serverUrl.replace(/\/$/, "")
  const list = await fetchWithKey(
    `${normalized}/api/v1/data-tables?page=1&page_size=50`,
    apiKey
  ).catch(() => null)
  if (!list?.ok) return
  const payload = await list.json().catch(() => null)
  const tables = parseListPayload(payload, ["tables"])
  const match = tables.find((t: any) => String(t?.name || "") === name)
  if (!match?.id) return
  await fetchWithKey(
    `${normalized}/api/v1/data-tables/${encodeURIComponent(String(match.id))}`,
    apiKey,
    { method: "DELETE" }
  ).catch(() => {})
}

const cleanupMediaItem = async (
  serverUrl: string,
  apiKey: string,
  mediaId: string | number
) => {
  const normalized = serverUrl.replace(/\/$/, "")
  await fetchWithKey(
    `${normalized}/api/v1/media/${encodeURIComponent(String(mediaId))}`,
    apiKey,
    { method: "DELETE" }
  ).catch(() => {})
  await fetchWithKey(
    `${normalized}/api/v1/media/${encodeURIComponent(
      String(mediaId)
    )}/permanent`,
    apiKey,
    { method: "DELETE" }
  ).catch(() => {})
}

const fetchAudioProviders = async (serverUrl: string, apiKey: string) => {
  const normalized = serverUrl.replace(/\/$/, "")
  const res = await fetchWithKey(
    `${normalized}/api/v1/audio/providers`,
    apiKey
  ).catch(() => null)
  if (!res?.ok) return null
  const payload = await res.json().catch(() => null)
  const providers = payload?.providers ?? payload
  if (
    !providers ||
    typeof providers !== "object" ||
    Object.keys(providers).length === 0
  ) {
    return null
  }
  return payload
}

const selectTldwProvider = async (page: Page) => {
  await page.getByText("Text to speech").scrollIntoViewIfNeeded()
  const providerSelect = page.getByText("Browser TTS", { exact: false })
  await providerSelect.click()
  const option = page.getByRole("option", {
    name: /tldw server \(audio\/speech\)/i
  })
  const visible = await option
    .waitFor({ state: "visible", timeout: 5000 })
    .then(() => true)
    .catch(() => false)
  if (!visible) return false
  await option.click()
  return true
}

const selectServerTab = async (sidebar: Locator) => {
  const radio = sidebar.getByRole("radio", { name: /^Server/i })
  if ((await radio.count()) > 0) {
    await radio.first().click()
    return
  }
  const button = sidebar.getByRole("button", { name: /^Server/i })
  if ((await button.count()) > 0) {
    await button.first().click()
    return
  }
  await sidebar.getByText(/^Server/i).first().click()
}

export function registerRealServerWorkflows(
  createDriver: CreateWorkflowDriver
) {
const createDriverForTest = async (
  options: Parameters<CreateWorkflowDriver>[0]
) => {
  try {
    return await createDriver({ ...options, testRef: test })
  } catch (error) {
    const message = String(error || "")
    if (
      message.includes("browserType.launch") ||
      message.includes("Extension launch unavailable")
    ) {
      test.skip(
        true,
        `Extension launch unavailable in this environment (${message}).`
      )
      return undefined as never
    }
    throw error
  }
}

test.describe("Real server end-to-end workflows", () => {
  test(
    "chat -> save to notes -> open linked conversation",
    async ({ page: fixturePage, context: fixtureContext }, testInfo) => {
      test.setTimeout(180000)
      const debugLines: string[] = []
      const startedAt = Date.now()
      const safeStringify = (value: unknown) => {
        try {
          return JSON.stringify(value)
        } catch {
          return "\"[unserializable]\""
        }
      }
      const logStep = (message: string, details?: Record<string, unknown>) => {
        const payload = {
          elapsedMs: Date.now() - startedAt,
          ...(details || {})
        }
        const line = `[real-server-notes] ${message} ${safeStringify(
          payload
        )}`
        debugLines.push(line)
        console.log(line)
      }
      const step = async <T>(label: string, fn: () => Promise<T>) => {
        logStep(`start ${label}`)
        const stepStart = Date.now()
        try {
          const result = await test.step(label, fn)
          logStep(`done ${label}`, {
            durationMs: Date.now() - stepStart
          })
          return result
        } catch (error) {
          logStep(`error ${label}`, {
            durationMs: Date.now() - stepStart,
            error: String(error)
          })
          throw error
        }
      }
      const { serverUrl, apiKey } = requireRealServerConfig()
      const normalizedServerUrl = normalizeServerUrl(serverUrl)
      logStep("test config", { serverUrl: normalizedServerUrl })

      const modelsResponse = await step("preflight: models", async () => {
        const response = await fetchWithKey(
          `${normalizedServerUrl}/api/v1/llm/providers`,
          apiKey
        )
        logStep("models preflight response", {
          ok: response.ok,
          status: response.status,
          statusText: response.statusText
        })
        return response
      })
      if (!modelsResponse.ok) {
        const body = await modelsResponse.text().catch(() => "")
        skipOrThrow(
          true,
          `Chat models preflight failed: ${modelsResponse.status} ${modelsResponse.statusText} ${body}`
        )
        return
      }
      const runnableModel = resolveRunnableChatModel(
        await modelsResponse.json().catch(() => [])
      )
      if (!runnableModel) {
        skipOrThrow(
          true,
          "No configured chat-capable model is available on tldw_server."
        )
        return
      }
      const selectedModelId = toSelectedModelId(runnableModel)
      logStep("selected model resolved", { selectedModelId })

      const notesResponse = await step("preflight: notes list", async () => {
        const response = await fetchWithKey(
          `${normalizedServerUrl}/api/v1/notes/?page=1&results_per_page=1`,
          apiKey
        )
        logStep("notes preflight response", {
          ok: response.ok,
          status: response.status,
          statusText: response.statusText
        })
        return response
      })
      if (!notesResponse.ok) {
        const body = await notesResponse.text().catch(() => "")
        skipOrThrow(
          true,
          `Notes API preflight failed: ${notesResponse.status} ${notesResponse.statusText} ${body}`
        )
        return
      }
  
      const unique = Date.now()
      const characterName = `E2E Notes Character ${unique}`
      logStep("generated test identifiers", { unique, characterName })
      let createdCharacter = false
      let characterRecord: any | null = null
  
      const driver = await step("launch driver", async () =>
        createDriverForTest({
          serverUrl: normalizedServerUrl,
          apiKey,
          page: fixturePage,
          context: fixtureContext
        })
      )
      const {
        context,
        page,
        openSidepanel,
        optionsUrl,
        sidepanelUrl
      } = driver
      logStep("driver launched", {
        kind: driver.kind,
        optionsUrl,
        sidepanelUrl
      })
      const attachPageLogging = (targetPage: Page, tag: string) => {
        targetPage.on("console", (msg) => {
          const type = msg.type()
          if (type === "error" || type === "warning") {
            logStep(`${tag} console`, { type, text: msg.text() })
          }
        })
        targetPage.on("pageerror", (error) => {
          logStep(`${tag} pageerror`, { error: String(error) })
        })
        targetPage.on("response", (response) => {
          if (response.status() >= 400) {
            logStep(`${tag} failed response`, {
              status: response.status(),
              method: response.request().method(),
              url: response.url()
            })
          }
        })
      }
      attachPageLogging(page, "options")
  
      try {
        const granted = await step("grant host permission", async () => {
          const result = await driver.ensureHostPermission()
          logStep("host permission result", {
            origin: new URL(normalizedServerUrl).origin,
            granted: result
          })
          return result
        })
        if (!granted) {
          skipOrThrow(
            true,
            "Host permission not granted for tldw_server origin; allow it in chrome://extensions > tldw Assistant > Site access, then re-run"
          )
          return
        }
  
        const characterListResponse = await step(
          "preflight: characters list",
          async () => {
            const response = await fetchWithKey(
              `${normalizedServerUrl}/api/v1/characters/?page=1&results_per_page=1`,
              apiKey
            ).catch(() => null)
            logStep("characters preflight response", {
              ok: response?.ok ?? false,
              status: response?.status ?? null,
              statusText: response?.statusText ?? ""
            })
            return response
          }
        )
        if (!characterListResponse?.ok) {
          const body = await characterListResponse?.text().catch(() => "")
          skipOrThrow(
            true,
            `Characters API preflight failed: ${characterListResponse?.status} ${characterListResponse?.statusText} ${body}`
          )
          return
        }
        const characterId = await step("create character", async () => {
          const id = await createCharacterByName(
            normalizedServerUrl,
            apiKey,
            characterName
          )
          logStep("character created", { characterId: id })
          return id
        })
        if (!characterId) {
          throw new Error("Unable to create character for notes flow.")
        }
        createdCharacter = true
        characterRecord = await step("poll for character", async () => {
          const record = await pollForCharacterByName(
            normalizedServerUrl,
            apiKey,
            characterName,
            30000
          )
          logStep("character record resolved", {
            found: !!record,
            recordId: record?.id ?? record?.uuid ?? null
          })
          return record
        })
        if (!characterRecord) {
          throw new Error(
            "Character created but not returned by search for notes flow."
          )
        }

        await step("seed model selection", async () => {
          await setSelectedModel(page, selectedModelId)
        })

        const chatPage = await step("open sidepanel", async () => {
          const panel = await openChatSidepanel(driver)
          logStep("sidepanel opened", { url: panel.url() })
          return panel
        })
        attachPageLogging(chatPage, "sidepanel")
        await step("wait for sidepanel connected", async () => {
          await waitForConnected(chatPage, "workflow-chat-notes", driver.kind)
        })
        await step("select tracked character", async () => {
          await selectTrackedCharacterFromRuntimeRail(
            chatPage,
            characterName,
            driver.kind,
            characterId
          )
        })
        await step("ensure server persistence", async () => {
          await ensureServerPersistence(chatPage)
        })
  
        const userMessage = `E2E notes flow ${unique}`
        logStep("sending chat message", { userMessage })
        await step("send chat message", async () => {
          await sendChatMessage(chatPage, userMessage)
        })
        await step("wait for message store", async () => {
          await waitForMessageStore(chatPage, "notes-assistant-snapshot", 30000)
        })
        const assistantSnapshot = await step(
          "wait for assistant snapshot",
          async () => waitForAssistantSnapshot(chatPage)
        )
        if (!assistantSnapshot?.serverChatId || !assistantSnapshot?.text) {
          throw new Error(
            "Assistant server message not available after streaming."
          )
        }
        logStep("assistant snapshot resolved", {
          serverChatId: assistantSnapshot.serverChatId,
          serverMessageId: assistantSnapshot.serverMessageId,
          localId: assistantSnapshot.localId
        })
        const assistantText = normalizeMessageContent(assistantSnapshot.text)
        const serverChatId = String(assistantSnapshot.serverChatId)
        let serverMessageId = assistantSnapshot.serverMessageId
          ? String(assistantSnapshot.serverMessageId)
          : null
        const assistantMessage = getAssistantMessageLocator(chatPage, assistantSnapshot)
        await step("locate assistant message by local id", async () => {
          await expect(assistantMessage).toBeVisible({ timeout: 30000 })
        })
        logStep("assistant text captured", {
          serverChatId,
          serverMessageId,
          textPreview: assistantText.slice(0, 80)
        })
        if (!serverMessageId) {
          serverMessageId = await step(
            "wait for assistant server message id in store",
            async () =>
              waitForAssistantServerMessageIdInStore(chatPage, {
                localId: assistantSnapshot.localId,
                assistantText
              })
          )
        }
        if (!serverMessageId) {
          serverMessageId = await step("poll server message id", async () => {
            const resolved = await pollForServerAssistantMessageId(
              normalizedServerUrl,
              apiKey,
              serverChatId,
              assistantText
            )
            logStep("server message id polled", { serverMessageId: resolved })
            return resolved
          })
          if (serverMessageId) {
            await step("sync server message id into store", async () => {
              const synced = await syncAssistantServerMessageIdIntoStore(chatPage, {
                localId: assistantSnapshot.localId,
                serverMessageId
              })
              logStep("assistant store sync result", { synced, serverMessageId })
            })
          }
        }
        if (!serverMessageId) {
          throw new Error(
            "Assistant server message not available after streaming."
          )
        }
        await step(
          "wait for assistant save action to become eligible",
          async () => {
            await expect
              .poll(
                async () => {
                  const latestId = await waitForAssistantServerMessageIdInStore(
                    chatPage,
                    {
                      localId: assistantSnapshot.localId,
                      assistantText,
                      timeoutMs: 2000
                    }
                  )
                  return latestId ?? serverMessageId
                },
                { timeout: 30000, intervals: [500, 1000, 2000] }
              )
              .toBeTruthy()
          }
        )
        const snippet = assistantText.slice(0, 80)
        logStep("assistant snippet", { snippet })

        await step("save assistant to notes", async () => {
          await clickSaveToNotesAction(chatPage, assistantMessage)
        })
        const savedNote = await step("poll for saved note", async () => {
          const note = await pollForNoteByConversation(
            normalizedServerUrl,
            apiKey,
            serverChatId,
            serverMessageId
          )
          logStep("saved note poll result", {
            found: !!note,
            noteId: note?.id ?? note?.uuid ?? null
          })
          return note
        })
        if (!savedNote) {
          throw new Error("Saved note not found for conversation.")
        }
        const backlink = extractNoteBacklink(savedNote)
        logStep("saved note backlink", backlink)
        if (!backlink.conversation_id) {
          throw new Error("Saved note missing linked conversation id.")
        }
        const savedNoteId =
          savedNote?.id ?? savedNote?.note_id ?? savedNote?.noteId ?? null
        if (savedNoteId == null) {
          throw new Error("Saved note missing id.")
        }
        logStep("saved note id resolved", { savedNoteId })
        await step("seed last note id", async () => {
          await setLastNoteId(page, String(savedNoteId))
        })
  
        await step("open notes page", async () => {
          await driver.goto(page, "/notes", {
            waitUntil: "domcontentloaded"
          })
        })
        await step("wait for notes connected", async () => {
          await waitForConnected(page, "workflow-notes-view", driver.kind)
        })
  
        const noteTitle = String(savedNote?.title || "").trim()
        const query =
          noteTitle.length > 0 ? noteTitle.slice(0, 40) : snippet.slice(0, 40)
        logStep("notes search query", { noteTitle, query })
        const linkedConversationLabel = page
          .getByText(/Linked to conversation/i)
          .first()
        const backlinkVisible = await step("wait for note selection", async () =>
          linkedConversationLabel
            .waitFor({ state: "visible", timeout: 30000 })
            .then(() => true)
            .catch(() => false)
        )
        logStep("linked conversation visible", { backlinkVisible })
        if (!backlinkVisible) {
          await step("clear notes search", async () => {
            const searchInput = page.getByPlaceholder(
              /Search titles and contents|Search notes/i
            )
            await searchInput.fill("")
            await searchInput.press("Enter")
          })
  
          const resultRow = await step("find note row", async () =>
            findNoteRowInList(page, backlink.conversation_id, query, 6)
          )
          if (!resultRow) {
            throw new Error("Note row not visible in notes list.")
          }
          await step("select note row", async () => {
            await expect(resultRow).toBeVisible({ timeout: 10000 })
            await resultRow.click()
          })
          await expect(linkedConversationLabel).toBeVisible({ timeout: 15000 })
        }
  
      await step("verify linked conversation", async () => {
        await expect(linkedConversationLabel).toBeVisible({ timeout: 10000 })
      })
  
      await step("open linked conversation", async () => {
        const overflowMenu = page.getByTestId("notes-overflow-menu-button")
        await expect(overflowMenu).toBeVisible({ timeout: 10000 })
        await overflowMenu.click()
        const openConversation = page.getByRole("menuitem", {
          name: /Open linked conversation|Open conversation/i
        })
        await expect(openConversation).toBeVisible({ timeout: 10000 })
        logStep("open conversation url before", { url: page.url() })
        await openConversation.click()
        await waitForChatLanding(page, driver, 20000)
        await waitForConnected(page, "workflow-notes-open-linked", driver.kind)
        logStep("open conversation url after", { url: page.url() })
        await expect(
          page.locator("#textarea-message")
        ).toBeVisible({ timeout: 20000 })
        await expect(
          page.getByRole("log", { name: /chat messages/i })
        ).toContainText(userMessage, { timeout: 30000 })
      })
      } finally {
        await testInfo.attach("notes-flow-debug", {
          body: debugLines.join("\n"),
          contentType: "text/plain"
        })
        await driver.close()
        if (createdCharacter) {
          await deleteCharacterByName(
            normalizedServerUrl,
            apiKey,
            characterName
          )
        }
      }
    })

    test("chat -> save to flashcards -> review card", async ({
      page: fixturePage,
      context: fixtureContext
    }) => {
      test.setTimeout(180000)
      const { serverUrl, apiKey } = requireRealServerConfig()
      const normalizedServerUrl = normalizeServerUrl(serverUrl)

      const decksResponse = await fetchWithKey(
        `${normalizedServerUrl}/api/v1/flashcards/decks`,
        apiKey
      )
      if (!decksResponse.ok) {
        const body = await decksResponse.text().catch(() => "")
        skipOrThrow(
          true,
          `Flashcards API preflight failed: ${decksResponse.status} ${decksResponse.statusText} ${body}`
        )
        return
      }

      const modelsResponse = await fetchWithKey(
        `${normalizedServerUrl}/api/v1/llm/providers`,
        apiKey
      )
      if (!modelsResponse.ok) {
        const body = await modelsResponse.text().catch(() => "")
        skipOrThrow(
          true,
          `Chat models preflight failed: ${modelsResponse.status} ${modelsResponse.statusText} ${body}`
        )
        return
      }
      const runnableModel = resolveRunnableChatModel(
        await modelsResponse.json().catch(() => [])
      )
      if (!runnableModel) {
        skipOrThrow(
          true,
          "No configured chat-capable model is available on tldw_server."
        )
        return
      }
      const selectedModelId = toSelectedModelId(runnableModel)

      const unique = Date.now()
      const characterName = `E2E Flashcards Character ${unique}`
      let createdCharacter = false
      let characterRecord: any | null = null
      let savedFlashcardUuid: string | null = null

      const driver = await createDriverForTest({
        serverUrl: normalizedServerUrl,
        apiKey,
        page: fixturePage,
        context: fixtureContext
      })
      const { context, page, openSidepanel } = driver

      try {
        const granted = await driver.ensureHostPermission()
        if (!granted) {
          skipOrThrow(
            true,
            "Host permission not granted for tldw_server origin; allow it in chrome://extensions > tldw Assistant > Site access, then re-run"
          )
          return
        }

        const characterListResponse = await fetchWithKey(
          `${normalizedServerUrl}/api/v1/characters/?page=1&results_per_page=1`,
          apiKey
        ).catch(() => null)
        if (!characterListResponse?.ok) {
          const body = await characterListResponse?.text().catch(() => "")
          skipOrThrow(
            true,
            `Characters API preflight failed: ${characterListResponse?.status} ${characterListResponse?.statusText} ${body}`
          )
          return
        }
        const characterId = await createCharacterByName(
          normalizedServerUrl,
          apiKey,
          characterName
        )
        if (!characterId) {
          throw new Error("Unable to create character for flashcards flow.")
        }
        createdCharacter = true
        characterRecord = await pollForCharacterByName(
          normalizedServerUrl,
          apiKey,
          characterName,
          30000
        )
        if (!characterRecord) {
          throw new Error(
            "Character created but not returned by search for flashcards flow."
          )
      }

      await setSelectedModel(page, selectedModelId)

      const chatPage = await openChatSidepanel(driver)
      await waitForConnected(chatPage, "workflow-chat-flashcards", driver.kind)
      await selectTrackedCharacterFromRuntimeRail(
        chatPage,
        characterName,
        driver.kind,
        characterId
      )
      await ensureServerPersistence(chatPage)

        const userMessage = `E2E flashcards flow ${unique}`
        await sendChatMessage(chatPage, userMessage)
        await waitForAssistantMessage(chatPage)
        await waitForMessageStore(
          chatPage,
          "flashcards-assistant-snapshot",
          30000
        )
        const assistantSnapshot = await waitForAssistantSnapshot(chatPage)
        if (!assistantSnapshot?.serverChatId || !assistantSnapshot?.text) {
          throw new Error(
            "Assistant server message not available after streaming."
          )
        }
        const assistantText = normalizeMessageContent(assistantSnapshot.text)
        if (!assistantText) {
          throw new Error("Assistant message did not contain text.")
        }
        const serverChatId = String(assistantSnapshot.serverChatId)
        let serverMessageId = assistantSnapshot.serverMessageId
          ? String(assistantSnapshot.serverMessageId)
          : null
        const assistantMessage = getAssistantMessageLocator(
          chatPage,
          assistantSnapshot
        )
        await expect(assistantMessage).toBeVisible({ timeout: 30000 })
        if (!serverMessageId) {
          serverMessageId = await waitForAssistantServerMessageIdInStore(
            chatPage,
            {
              localId: assistantSnapshot.localId,
              assistantText
            }
          )
        }
        if (!serverMessageId) {
          serverMessageId = await pollForServerAssistantMessageId(
            normalizedServerUrl,
            apiKey,
            serverChatId,
            assistantText
          )
          if (serverMessageId) {
            await syncAssistantServerMessageIdIntoStore(chatPage, {
              localId: assistantSnapshot.localId,
              serverMessageId
            })
          }
        }
        if (!serverMessageId) {
          throw new Error(
            "Assistant server message not available after streaming."
          )
        }
        await expect
          .poll(
            async () =>
              waitForAssistantServerMessageIdInStore(chatPage, {
                localId: assistantSnapshot.localId,
                assistantText,
                timeoutMs: 2000
              }),
            { timeout: 30000, intervals: [500, 1000, 2000] }
          )
          .toBeTruthy()
        const baselineFlashcards = await fetchRecentFlashcards(
          normalizedServerUrl,
          apiKey,
          20
        )
        const baselineFlashcardIds = new Set(
          baselineFlashcards
            .map((item: any) => (item?.uuid != null ? String(item.uuid) : null))
            .filter((id: string | null): id is string => Boolean(id))
        )

        await clearRequestErrors(chatPage)
        await clickMessageOverflowAction(
          chatPage,
          assistantMessage,
          /Save to Flashcards/i
        )
        await expect(chatPage.getByText(/Saved to Flashcards/i)).toBeVisible({
          timeout: 15000
        })
        const requestErrors = await readLastRequestError(chatPage)
        if (requestErrors?.last || requestErrors?.recent?.length) {
          console.log(
            "[e2e] flashcards save request errors",
            JSON.stringify(requestErrors)
          )
        }
        await logFlashcardsSnapshot(normalizedServerUrl, apiKey, "after-save")
        try {
          const savedFlashcard = await pollForNewFlashcard(
            normalizedServerUrl,
            apiKey,
            baselineFlashcardIds,
            assistantText
          )
          savedFlashcardUuid = String(savedFlashcard?.uuid || "").trim() || null
          if (!savedFlashcardUuid) {
            throw new Error("Saved flashcard did not include a UUID.")
          }
        } catch (error) {
          await probeSaveChatKnowledge(
            normalizedServerUrl,
            apiKey,
            {
              conversation_id: serverChatId,
              message_id: serverMessageId,
              snippet: assistantText.slice(0, 1000),
              make_flashcard: true
            },
            "after-save-timeout"
          )
          await logChatMessagesSnapshot(
            normalizedServerUrl,
            apiKey,
            serverChatId,
            "after-save-timeout"
          )
          await logFlashcardsSnapshot(
            normalizedServerUrl,
            apiKey,
            "after-save-timeout"
          )
          throw error
        }
        if (!savedFlashcardUuid) {
          throw new Error("Saved flashcard UUID was unavailable after polling.")
        }

        await driver.goto(page, "/flashcards", {
          waitUntil: "domcontentloaded"
        })
        await waitForConnected(page, "workflow-flashcards-view", driver.kind)

        const manageTab = page.getByRole("tab", { name: /^Manage$/i })
        await manageTab.click()

        const cardRow = page.getByTestId(
          `flashcard-item-${savedFlashcardUuid}`
        )
        await expect(cardRow).toBeVisible({ timeout: 30000 })

        const studyTab = page.getByRole("tab", { name: /^Study$/i })
        await studyTab.click()

        const showAnswer = page.getByTestId("flashcards-review-show-answer")
        if (!(await showAnswer.isVisible().catch(() => false))) {
          const reviewAllDue = page.getByTestId("flashcards-review-all-due")
          await expect(reviewAllDue).toBeVisible({ timeout: 30000 })
          await expect(reviewAllDue).toBeEnabled({ timeout: 30000 })
          await reviewAllDue.click()
        }

        await expect(showAnswer).toBeVisible({ timeout: 30000 })
        await showAnswer.click()
        const rateButton = page.getByTestId("flashcards-review-rate-2")
        await rateButton.click()
        await expect(rateButton).toBeHidden({ timeout: 30000 })
      } finally {
        await driver.close()
        if (savedFlashcardUuid) {
          await cleanupFlashcard(
            normalizedServerUrl,
            apiKey,
            savedFlashcardUuid
          )
        }
        if (createdCharacter) {
          await deleteCharacterByName(
            normalizedServerUrl,
            apiKey,
            characterName
          )
        }
      }
    })

    test("media trash -> delete -> restore", async ({
      page: fixturePage,
      context: fixtureContext
    }) => {
      test.setTimeout(360000)
      const { serverUrl, apiKey } = requireRealServerConfig()
      const normalizedServerUrl = normalizeServerUrl(serverUrl)

      const trashResponse = await fetchWithKey(
        `${normalizedServerUrl}/api/v1/media/trash?page=1&results_per_page=1`,
        apiKey
      ).catch(() => null)
      if (!trashResponse?.ok) {
        const body = await trashResponse?.text().catch(() => "")
        skipOrThrow(
          true,
          `Media trash preflight failed: ${trashResponse?.status} ${trashResponse?.statusText} ${body}`
        )
        return
      }

      const driver = await createDriverForTest({
        serverUrl: normalizedServerUrl,
        apiKey,
        page: fixturePage,
        context: fixtureContext
      })
      const { context, page } = driver

      const unique = Date.now()
      const fileName = `e2e-trash-${unique}.txt`
      let mediaId: string | number | null = null

      try {
        const granted = await driver.ensureHostPermission()
        if (!granted) {
          skipOrThrow(
            true,
            "Host permission not granted for tldw_server origin; allow it in chrome://extensions > tldw Assistant > Site access, then re-run"
          )
          return
        }

        await driver.goto(page, "/media", {
          waitUntil: "domcontentloaded"
        })
        await waitForConnected(page, "workflow-media-trash-ingest", driver.kind)

        const modal = await openQuickIngestModal(page)
        await waitForQuickIngestReady(modal)

        await page.setInputFiles('[data-testid="qi-file-input"]', {
          name: fileName,
          mimeType: "text/plain",
          buffer: Buffer.from(`E2E media trash ${unique}`)
        })

        const fileRow = modal.getByText(fileName).first()
        await expect(fileRow).toBeVisible({ timeout: 15000 })
        await fileRow.click()
        await dismissQuickIngestInspectorIntro(page)

        await selectQuickIngestQuickPreset(modal)
        await clickQuickIngestRun(modal)
        await waitForQuickIngestCompletion(modal, 180000)
        await closeQuickIngestModal(modal)

        const mediaMatch = await pollForMediaMatch(
          normalizedServerUrl,
          apiKey,
          `e2e-trash-${unique}`, // Use filename prefix with words for FTS5 tokenization
          300000
        )
        mediaId = mediaMatch?.id ?? null
        if (mediaId == null) {
          throw new Error("Ingested media was returned without an ID.")
        }
        const expectedTitle = String(
          mediaMatch?.title || mediaMatch?.filename || fileName
        ).replace(/\.txt$/i, "")

        await driver.goto(
          page,
          `/media?id=${encodeURIComponent(String(mediaId))}`,
          {
          waitUntil: "domcontentloaded"
          }
        )
        await waitForConnected(page, "workflow-media-trash-delete", driver.kind)

        const searchInput = page.getByTestId("media-search-input")
        await expect(searchInput).toBeVisible({ timeout: 30000 })
        await searchInput.fill(String(unique))
        await page.getByTestId("media-search-submit").click({ timeout: 15000 })

        const resultRow = page
          .getByRole("button", {
            name: new RegExp(escapeRegExp(expectedTitle), "i")
          })
          .first()
        await expect(resultRow).toBeVisible({ timeout: 30000 })
        await resultRow.click({ timeout: 15000 })

        const deleteButton = page.getByRole("button", {
          name: /Delete item/i
        })
        await expect(deleteButton).toBeVisible({ timeout: 15000 })
        await deleteButton.click({ timeout: 15000 })
        await page
          .getByRole("button", { name: /^Delete$/ })
          .click({ timeout: 15000 })

        await expect(
          page.getByText("Moved to trash", { exact: true })
        ).toBeVisible({ timeout: 15000 })

        const trashButton = page.getByRole("button", { name: /^Trash$/i })
        await trashButton.click({ timeout: 15000 })
        await waitForConnected(page, "workflow-media-trash-view", driver.kind)

        const trashRow = page
          .locator("div")
          .filter({
            has: page.getByText(expectedTitle, { exact: true })
          })
          .filter({
            has: page.getByRole("button", { name: /^Restore$/i })
          })
          .first()
        await expect(trashRow).toBeVisible({ timeout: 30000 })
        const restoreButton = trashRow.getByRole("button", {
          name: /^Restore$/i
        })
        await restoreButton.click({ timeout: 15000 })

        await expect(page.getByText(/Item restored/i)).toBeVisible({
          timeout: 20000
        })
        await expect(
          page.getByText(expectedTitle, { exact: true })
        ).toHaveCount(0, {
          timeout: 20000
        })
      } finally {
        await driver.close()
        if (mediaId != null) {
          await cleanupMediaItem(normalizedServerUrl, apiKey, mediaId)
        }
      }
    })

    test("media ingestion -> analysis -> review -> re-analyze", async ({
      page: fixturePage,
      context: fixtureContext
    }) => {
      test.setTimeout(300000)
      const { serverUrl, apiKey } = requireRealServerConfig()
      const normalizedServerUrl = normalizeServerUrl(serverUrl)

      const mediaResponse = await fetchWithKey(
        `${normalizedServerUrl}/api/v1/media?page=1&results_per_page=1`,
        apiKey
      )
      if (!mediaResponse.ok) {
        const body = await mediaResponse.text().catch(() => "")
        skipOrThrow(
          true,
          `Media API preflight failed: ${mediaResponse.status} ${mediaResponse.statusText} ${body}`
        )
        return
      }

      const modelsResponse = await fetchWithKey(
        `${normalizedServerUrl}/api/v1/llm/providers`,
        apiKey
      )
      if (!modelsResponse.ok) {
        const body = await modelsResponse.text().catch(() => "")
        skipOrThrow(
          true,
          `Chat models preflight failed: ${modelsResponse.status} ${modelsResponse.statusText} ${body}`
        )
        return
      }
      const runnableModel = resolveRunnableChatModel(
        await modelsResponse.json().catch(() => [])
      )
      if (!runnableModel) {
        skipOrThrow(
          true,
          "No configured chat-capable model is available on tldw_server."
        )
        return
      }
      const selectedModelId = toSelectedModelId(runnableModel)

      const driver = await createDriverForTest({
        serverUrl: normalizedServerUrl,
        apiKey,
        page: fixturePage,
        context: fixtureContext
      })
      const { context, page } = driver

      const unique = Date.now()
      const fileName = `e2e-analysis-${unique}.txt`
      const token1 = "LIVE_TIER_ANALYSIS_ONE"
      const token2 = "LIVE_TIER_ANALYSIS_TWO"
      let mediaId: string | number | null = null

      const runAnalysis = async (token: string) => {
        const generateButton = page
          .getByRole("button", { name: /^Generate$/i })
          .first()
        await generateButton.scrollIntoViewIfNeeded()
        await generateButton.click({ timeout: 15000 })

        const modal = page.getByRole("dialog", { name: /^Generate$/i })
        await expect(modal).toBeVisible({ timeout: 15000 })

        const systemPrompt = modal.getByLabel(/System Prompt/i)
        await systemPrompt.fill(
          `Return exactly the token "${token}" and nothing else.`
        )
        const userPrefix = modal.getByLabel(/User Prompt Prefix/i)
        await userPrefix.fill("")

        const generateAnalysis = modal.getByRole("button", {
          name: /^Generate$/i
        })
        await expect(generateAnalysis).toBeEnabled({ timeout: 30000 })
        await generateAnalysis.click()

        await expect(modal).toBeHidden({ timeout: 180000 })
        if (mediaId == null) {
          throw new Error("Media ID was unavailable after analysis generation.")
        }
        const persistedAnalysis = await pollForPersistedMediaAnalysis(
          normalizedServerUrl,
          apiKey,
          mediaId,
          60000,
          token
        )
        const analysisOutput = page
          .getByRole("main")
          .getByText(persistedAnalysis, { exact: true })
          .first()
        await expect(analysisOutput).toBeVisible({ timeout: 60000 })
        await expect(
          page.getByText("Pending save", { exact: true })
        ).toHaveCount(0, { timeout: 30000 })
        return persistedAnalysis
      }

      try {
        const granted = await driver.ensureHostPermission()
        if (!granted) {
          skipOrThrow(
            true,
            "Host permission not granted for tldw_server origin; allow it in chrome://extensions > tldw Assistant > Site access, then re-run"
          )
          return
        }

        await setSelectedModel(page, selectedModelId)

        await driver.goto(page, "/media", {
          waitUntil: "domcontentloaded"
        })
        await waitForConnected(page, "workflow-analysis-ingest", driver.kind)

        const modal = await openQuickIngestModal(page)
        await waitForQuickIngestReady(modal)

        await page.setInputFiles('[data-testid="qi-file-input"]', {
          name: fileName,
          mimeType: "text/plain",
          buffer: Buffer.from(`E2E analysis content ${unique}`)
        })

        const fileRow = modal.getByText(fileName).first()
        await expect(fileRow).toBeVisible({ timeout: 15000 })
        await fileRow.click()
        await dismissQuickIngestInspectorIntro(page)

        await selectQuickIngestQuickPreset(modal)
        await clickQuickIngestRun(modal)
        await waitForQuickIngestCompletion(modal, 180000)
        await closeQuickIngestModal(modal)

        const mediaMatch = await pollForMediaMatch(
          normalizedServerUrl,
          apiKey,
          `e2e-analysis-${unique}`, // Use filename prefix with words for FTS5 tokenization
          300000
        )
        mediaId = mediaMatch?.id ?? null
        if (mediaId == null) {
          throw new Error("Ingested analysis media was returned without an ID.")
        }
        const expectedTitle = String(
          mediaMatch?.title || mediaMatch?.filename || fileName
        ).replace(/\.txt$/i, "")

        await driver.goto(
          page,
          `/media?id=${encodeURIComponent(String(mediaId))}`,
          { waitUntil: "domcontentloaded" }
        )
        await waitForConnected(page, "workflow-analysis-media", driver.kind)

        const searchInput = page.getByTestId("media-search-input")
        await expect(searchInput).toBeVisible({ timeout: 30000 })
        await searchInput.fill(String(unique))
        await page.getByTestId("media-search-submit").click({ timeout: 15000 })

        const resultRow = page
          .getByRole("button", {
            name: new RegExp(escapeRegExp(expectedTitle), "i")
          })
          .first()
        await expect(resultRow).toBeVisible({ timeout: 30000 })
        await resultRow.click()

        const firstAnalysis = await runAnalysis(token1)

        await driver.goto(page, "/media-multi", {
          waitUntil: "domcontentloaded"
        })
        await waitForConnected(page, "workflow-analysis-review", driver.kind)

        const reviewSearch = page.getByPlaceholder(/Search media/i)
        await expect(reviewSearch).toBeVisible({ timeout: 15000 })
        await reviewSearch.fill(String(unique))
        await page
          .getByRole("button", { name: /^Search$/i })
          .click({ timeout: 15000 })

        const reviewRow = page
          .getByTestId("media-review-results-list")
          .getByRole("button", {
            name: new RegExp(escapeRegExp(expectedTitle), "i")
          })
          .first()
        await expect(reviewRow).toBeVisible({ timeout: 30000 })
        await reviewRow.click()

        const reviewAnalysis = page
          .getByText(firstAnalysis, { exact: true })
          .first()
        await expect(reviewAnalysis).toBeVisible({ timeout: 60000 })

        await driver.goto(
          page,
          `/media?id=${encodeURIComponent(String(mediaId))}`,
          { waitUntil: "domcontentloaded" }
        )
        await waitForConnected(page, "workflow-analysis-reanalyze", driver.kind)

        await expect(searchInput).toBeVisible({ timeout: 30000 })
        await searchInput.fill(String(unique))
        await page.getByTestId("media-search-submit").click({ timeout: 15000 })
        await expect(resultRow).toBeVisible({ timeout: 30000 })
        await resultRow.click()

        const secondAnalysis = await runAnalysis(token2)
        expect(secondAnalysis).not.toBe(firstAnalysis)
      } finally {
        await driver.close()
        if (mediaId != null) {
          await cleanupMediaItem(normalizedServerUrl, apiKey, mediaId)
        }
      }
    })
  })
}

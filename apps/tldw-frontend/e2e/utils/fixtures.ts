/**
 * Extended Playwright test fixtures for workflow tests
 */
import { test as base, expect, type Page } from "@playwright/test"
import { readFileSync } from "node:fs"
import path from "node:path"
import {
  seedAuth,
  TEST_CONFIG,
  isBenign,
  fetchWithApiKey
} from "./helpers"
import { startApiCapture, getCapturedApiCalls } from "./api-assertions"

/**
 * Diagnostics data collected during page visits
 */
export interface DiagnosticsData {
  console: Array<{ type: string; text: string; location?: { url: string; lineNumber: number } }>
  pageErrors: Array<{ message: string; stack: string }>
  requestFailures: Array<{ url: string; errorText: string }>
}

/**
 * Server info for preflight checks
 */
export interface ServerInfo {
  available: boolean
  version?: string
  models?: string[]
  modelSource?: "metadata" | "providers"
}

/**
 * Extended test fixtures
 */
export interface WorkflowFixtures {
  /** Diagnostics data collected during test */
  diagnostics: DiagnosticsData
  /** Page pre-seeded with auth config */
  authedPage: Page
  /** Server availability info */
  serverInfo: ServerInfo
}

/**
 * Extended test with workflow fixtures
 */
export const test = base.extend<WorkflowFixtures>({
  // Collect diagnostics automatically
  diagnostics: async ({ page }, use) => {
    const data: DiagnosticsData = {
      console: [],
      pageErrors: [],
      requestFailures: []
    }

    page.on("console", (msg) => {
      const location = msg.location()
      data.console.push({
        type: msg.type(),
        text: msg.text(),
        location: location.url ? { url: location.url, lineNumber: location.lineNumber } : undefined
      })
    })

    page.on("pageerror", (err) => {
      data.pageErrors.push({
        message: err.message,
        stack: err.stack || ""
      })
    })

    page.on("requestfailed", (req) => {
      data.requestFailures.push({
        url: req.url(),
        errorText: req.failure()?.errorText || ""
      })
    })

    await use(data)
  },

  // Pre-seeded authenticated page
  authedPage: async ({ page }, use, testInfo) => {
    const appOrigin = new URL(TEST_CONFIG.webUrl).origin
    await page.context().grantPermissions(["clipboard-read", "clipboard-write"], {
      origin: appOrigin,
    })
    await seedAuth(page)
    startApiCapture(page)
    await use(page)
    // Teardown: attach API call log on test failure for debugging
    if (testInfo.status !== "passed") {
      const apiLog = getCapturedApiCalls(page)
      if (apiLog.length > 0) {
        await testInfo.attach("api-calls.json", {
          body: JSON.stringify(apiLog, null, 2),
          contentType: "application/json",
        })
      }
    }
  },

  // Server availability check
  serverInfo: async ({}, use) => {
    const info: ServerInfo = { available: false }

    try {
      // Check server health
      const healthUrl = `${TEST_CONFIG.serverUrl}/api/v1/health`
      const healthRes = await fetchWithApiKey(healthUrl).catch(() => null)

      if (healthRes?.ok) {
        info.available = true
        const healthData = await healthRes.json().catch(() => ({}))
        info.version = healthData.version
      } else {
        // Try alternative health check
        const rootRes = await fetchWithApiKey(`${TEST_CONFIG.serverUrl}/`).catch(() => null)
        info.available = rootRes?.ok ?? false
      }

      // Check available models. Prefer the richer metadata endpoint because the
      // providers endpoint includes catalog-only models that may not be runnable.
      if (info.available) {
        const metadataUrl = `${TEST_CONFIG.serverUrl}/api/v1/llm/models/metadata`
        const metadataRes = await fetchWithApiKey(metadataUrl).catch(() => null)
        if (metadataRes?.ok) {
          const metadataData = await metadataRes.json().catch(() => ({}))
          info.models = extractUsableModelIds(metadataData)
          info.modelSource = "metadata"
        }

        if (!info.models || info.models.length === 0) {
          const modelsUrl = `${TEST_CONFIG.serverUrl}/api/v1/llm/providers`
          const modelsRes = await fetchWithApiKey(modelsUrl).catch(() => null)
          if (modelsRes?.ok) {
            const modelsData = await modelsRes.json().catch(() => ({}))
            info.models = extractModelIds(modelsData)
            info.modelSource = "providers"
          }
        }
      }
    } catch {
      info.available = false
    }

    await use(info)
  }
})

export { expect }

export interface ModerationReviewItemsFixture {
  populated: any[]
  empty: any[]
  permissionDenied: { status: number; body: Record<string, unknown> }
  backendError: { status: number; body: Record<string, unknown> }
  partialData: any[]
  expiredUndo: any[]
  redactedContent: any[]
}

export function loadModerationReviewItemsFixture(): ModerationReviewItemsFixture {
  return JSON.parse(
    readFileSync(path.resolve(__dirname, "../fixtures/moderation-review-items.json"), "utf8")
  ) as ModerationReviewItemsFixture
}

/**
 * Extract configured model IDs from the metadata response.
 */
export function extractUsableModelIds(payload: any): string[] {
  const models: string[] = []
  const entries = Array.isArray(payload?.models)
    ? payload.models
    : Array.isArray(payload)
      ? payload
      : []

  for (const model of entries) {
    if (!isRunnableModelDescriptor(model)) continue
    const provider = normalizeModelField(
      model?.provider ?? model?.provider_key ?? model?.api_provider
    )
    const id = normalizeModelField(model?.id ?? model?.model ?? model?.name)
    if (!id) continue
    models.push(provider ? `${provider}:${id}` : id)
  }

  return [...new Set(models)]
}

/**
 * Extract model IDs from provider response.
 */
export function extractModelIds(payload: any): string[] {
  const models: string[] = []

  // Handle { providers: [{ name, models: [...] }, ...] } shape (actual API response)
  const providers = Array.isArray(payload?.providers)
    ? payload.providers
    : Array.isArray(payload)
      ? payload
      : []

  for (const provider of providers) {
    if (Array.isArray(provider?.models)) {
      const providerUsable = isRunnableModelDescriptor(provider)
      for (const model of provider.models) {
        if (!providerUsable || !isRunnableModelDescriptor(model)) continue
        if (typeof model === "string") {
          models.push(model)
        } else {
          const id = model?.id || model?.model || model?.name
          if (id) models.push(String(id))
        }
      }
    }
  }

  // Fallback: payload.models direct array
  if (models.length === 0 && Array.isArray(payload?.models)) {
    for (const model of payload.models) {
      if (!isRunnableModelDescriptor(model)) continue
      if (typeof model === "string") {
        models.push(model)
      } else {
        const id = model?.id || model?.model || model?.name
        if (id) models.push(String(id))
      }
    }
  }

  return models
}

function isRunnableModelDescriptor(value: any): boolean {
  if (!value || typeof value === "string") return true
  const catalogOnly = firstBoolean(value, ["catalog_only", "catalogOnly", "is_catalog_only"])
  if (catalogOnly === true) return false
  const configured = firstBoolean(value, ["is_configured", "isConfigured", "configured"])
  if (configured === false) return false
  const providerConfigured = firstBoolean(value, [
    "provider_is_configured",
    "providerIsConfigured",
    "provider_configured",
    "providerConfigured"
  ])
  if (providerConfigured === false) return false
  const deprecated = firstBoolean(value, ["deprecated", "is_deprecated", "isDeprecated"])
  if (deprecated === true) return false
  return true
}

function firstBoolean(value: any, keys: string[]): boolean | null {
  for (const key of keys) {
    const field = value?.[key] ?? value?.details?.[key] ?? value?.metadata?.[key]
    if (typeof field === "boolean") return field
  }
  return null
}

function normalizeModelField(value: unknown): string | null {
  if (typeof value !== "string" && typeof value !== "number") return null
  const trimmed = String(value).trim()
  return trimmed.length > 0 ? trimmed : null
}

/**
 * Get first available model ID
 */
export function getFirstModelId(serverInfo: ServerInfo): string | null {
  return serverInfo.models?.[0] ?? null
}

/**
 * Filter diagnostics to only critical issues
 */
export function getCriticalIssues(diagnostics: DiagnosticsData): {
  pageErrors: Array<{ message: string; stack: string }>
  consoleErrors: Array<{ type: string; text: string }>
  requestFailures: Array<{ url: string; errorText: string }>
} {
  return {
    pageErrors: diagnostics.pageErrors.filter((e) => !isBenign(e.message)),
    consoleErrors: diagnostics.console.filter(
      (c) => c.type === "error" && !isBenign(c.text)
    ),
    requestFailures: diagnostics.requestFailures.filter(
      (r) => !isBenign(r.url) && !isBenign(r.errorText)
    )
  }
}

/**
 * Assert no critical page errors occurred
 */
export async function assertNoCriticalErrors(diagnostics: DiagnosticsData): Promise<void> {
  const critical = getCriticalIssues(diagnostics)

  if (critical.pageErrors.length > 0) {
    const messages = critical.pageErrors.map((e) => e.message).join("\n")
    throw new Error(`Uncaught page errors:\n${messages}`)
  }
}

/**
 * Skip test if server is not available
 */
export function skipIfServerUnavailable(serverInfo: ServerInfo): void {
  if (!serverInfo.available) {
    test.skip(true, "Server is not available")
  }
}

/**
 * Skip test if no models are available
 */
export function skipIfNoModels(serverInfo: ServerInfo): void {
  if (!serverInfo.models || serverInfo.models.length === 0) {
    test.skip(true, "No LLM models available")
  }
}

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

      // Check available models
      if (info.available) {
        const modelsUrl = `${TEST_CONFIG.serverUrl}/api/v1/llm/providers`
        const modelsRes = await fetchWithApiKey(modelsUrl).catch(() => null)
        if (modelsRes?.ok) {
          const modelsData = await modelsRes.json().catch(() => ({}))
          info.models = extractModelIds(modelsData)
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
 * Extract model IDs from provider response
 */
function extractModelIds(payload: any): string[] {
  const models: string[] = []

  // Handle { providers: [{ name, models: [...] }, ...] } shape (actual API response)
  const providers = Array.isArray(payload?.providers)
    ? payload.providers
    : Array.isArray(payload)
      ? payload
      : []

  for (const provider of providers) {
    if (Array.isArray(provider?.models)) {
      for (const model of provider.models) {
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

import { expect, test, type Page } from "@playwright/test"
import { launchWithBuiltExtensionOrSkip, requireRealServerConfig } from "./utils/real-server"
import { logConnectionSnapshot, waitForConnectionStore } from "./utils/connection"
import { grantHostPermission } from "./utils/permissions"
import { runResearchWorkspaceParityContract } from "../../../test-utils/research-workspace"

const shouldSkipHostPermission =
  process.env.TLDW_E2E_SKIP_HOST_PERMISSION !== "0" &&
  process.env.TLDW_E2E_SKIP_HOST_PERMISSION !== "false"

const API_FETCH_TIMEOUT_MS = 30_000

const BENIGN_PAGE_ERROR_PATTERNS = [
  /AbortError/i
]

const BENIGN_CONSOLE_ERROR_PATTERNS = [
  /Failed to fetch models from tldw:\s+AbortError/i,
  /Failed to load resource: net::ERR_FILE_NOT_FOUND/i
]

const seedLocalStorage = {
  "playground-tour-completed": "true",
  "tldw-tutorials": JSON.stringify({
    state: {
      completedTutorials: ["playground", "chat", "notes", "media", "settings"],
      seenPromptPages: ["/", "/chat", "/notes", "/media", "/settings", "/playground", "/research-workspace"]
    },
    version: 0
  })
}

const isBenignByPattern = (value: string, patterns: RegExp[]): boolean =>
  patterns.some((pattern) => pattern.test(value))

const isBenignRequestFailure = (url: string, errorText: string): boolean => {
  if (
    url.startsWith("chrome-extension://") &&
    /\/fonts\/.+\.(ttf|woff|woff2)$/i.test(url) &&
    /ERR_FILE_NOT_FOUND/i.test(errorText)
  ) {
    return true
  }

  return false
}

const normalizeServerUrl = (value: string) =>
  value.match(/^https?:\/\//) ? value.replace(/\/$/, "") : `http://${value}`

interface ConnectionStoreSnapshot {
  state?: {
    isConnected?: boolean
    phase?: string
  }
}

interface ConnectionStoreHook {
  getState?: () => ConnectionStoreSnapshot
}

type ConnectionStoreWindow = Window & {
  __tldw_useConnectionStore?: ConnectionStoreHook
}

const apiFetch = async (
  serverUrl: string,
  apiKey: string,
  path: string,
  init: RequestInit = {}
) => {
  const headers = new Headers(init.headers)
  headers.set("X-API-Key", apiKey)
  if (init.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json")
  }
  const timeoutController = new AbortController()
  const timeoutId = setTimeout(
    () => timeoutController.abort(),
    API_FETCH_TIMEOUT_MS
  )
  let removeAbortListener: (() => void) | undefined

  if (init.signal) {
    const abortFromCaller = () => timeoutController.abort()
    if (init.signal.aborted) {
      abortFromCaller()
    } else {
      init.signal.addEventListener("abort", abortFromCaller, { once: true })
      removeAbortListener = () =>
        init.signal?.removeEventListener("abort", abortFromCaller)
    }
  }

  try {
    const response = await fetch(`${serverUrl}${path}`, {
      ...init,
      headers,
      signal: timeoutController.signal
    })
    if (!response.ok) {
      throw new Error(
        `${init.method || "GET"} ${path} returned ${response.status}: ${await response.text()}`
      )
    }
    return response
  } finally {
    clearTimeout(timeoutId)
    removeAbortListener?.()
  }
}

const recoverOptionsErrorStateIfNeeded = async (page: Page): Promise<void> => {
  const reloadButton = page.getByRole("button", { name: /Reload Options/i }).first()
  if (await reloadButton.isVisible().catch(() => false)) {
    await reloadButton.click()
    await page.waitForLoadState("networkidle")
  }
}

const waitForConnected = async (page: Page, label: string): Promise<void> => {
  try {
    await page.waitForFunction(
      () => {
        const store = (window as ConnectionStoreWindow).__tldw_useConnectionStore
        const state = store?.getState?.().state
        return state?.isConnected === true && state?.phase === "connected"
      },
      undefined,
      { timeout: 30_000 }
    )
  } catch (error) {
    await logConnectionSnapshot(page, `${label}-connection-timeout`)
    throw error
  }
}

test.describe("Research Workspace parity (extension real backend)", () => {
  test("passes baseline + deterministic studio parity contract against a real server", async () => {
    const { serverUrl, apiKey } = requireRealServerConfig(test)
    const normalizedServerUrl = normalizeServerUrl(serverUrl)

    const pageErrors: string[] = []
    const consoleErrors: string[] = []
    const requestFailures: string[] = []

    const { context, page, optionsUrl, extensionId } = await launchWithBuiltExtensionOrSkip(
      test,
      {
        seedConfig: {
          __tldw_first_run_complete: true,
          tldwConfig: {
            serverUrl: normalizedServerUrl,
            authMode: "single-user",
            apiKey
          }
        },
        seedLocalStorage
      }
    )

    if (!shouldSkipHostPermission) {
      const origin = new URL(normalizedServerUrl).origin + "/*"
      const granted = await grantHostPermission(context, extensionId, origin)
      if (!granted) {
        await context.close()
        test.skip(
          true,
          "Host permission not granted for real-server origin; allow it in chrome://extensions and re-run."
        )
      }
    }

    page.on("pageerror", (error) => {
      if (isBenignByPattern(error.message, BENIGN_PAGE_ERROR_PATTERNS)) return
      pageErrors.push(error.message)
    })
    page.on("console", (message) => {
      if (message.type() !== "error") return
      const text = message.text()
      if (isBenignByPattern(text, BENIGN_CONSOLE_ERROR_PATTERNS)) return
      consoleErrors.push(text)
    })
    page.on("requestfailed", (request) => {
      const url = request.url()
      const errorText = request.failure()?.errorText || "request failed"
      if (isBenignRequestFailure(url, errorText)) {
        return
      }
      requestFailures.push(`${errorText} :: ${url}`)
    })

    try {
      await waitForConnectionStore(page, "workspace-parity-extension-real")
      await recoverOptionsErrorStateIfNeeded(page)
      await waitForConnected(page, "workspace-parity-extension-real")

      await runResearchWorkspaceParityContract({
        platform: "extension",
        page,
        optionsUrl
      })

      expect(pageErrors).toEqual([])
      expect(consoleErrors).toEqual([])
      expect(requestFailures).toEqual([])
    } finally {
      await context.close()
    }
  })

  test("saves a Web Clipper workspace clip and opens canonical Research Workspace", async () => {
    const { serverUrl, apiKey } = requireRealServerConfig(test)
    const normalizedServerUrl = normalizeServerUrl(serverUrl)
    const suffix = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
    const workspaceId = `task-47812-webclip-${suffix}`
    const clipId = `clip-${suffix}`
    const uniqueBody = `TASK-47812-WEBCLIP-HANDOFF-${suffix}`

    await apiFetch(
      normalizedServerUrl,
      apiKey,
      `/api/v1/workspaces/${encodeURIComponent(workspaceId)}`,
      {
        method: "PUT",
        body: JSON.stringify({
          name: "TASK-478.12 Web Clipper Handoff",
          study_materials_policy: "workspace"
        })
      }
    )

    const { context, openSidepanel } = await launchWithBuiltExtensionOrSkip(
      test,
      {
        seedConfig: {
          __tldw_first_run_complete: true,
          tldwConfig: {
            serverUrl: normalizedServerUrl,
            authMode: "single-user",
            apiKey
          }
        },
        seedLocalStorage
      }
    )

    try {
      const sidepanel = await openSidepanel("/clipper")
      await waitForConnectionStore(sidepanel, "workspace-clipper-extension-real")
      await waitForConnected(sidepanel, "workspace-clipper-extension-real")
      await sidepanel.evaluate(
        ({ draft }) => {
          window.sessionStorage.setItem(
            "tldw:web-clipper:pendingDraft",
            JSON.stringify(draft)
          )
          window.dispatchEvent(
            new CustomEvent("tldw:web-clipper-pending-draft", {
              detail: draft
            })
          )
        },
        {
          draft: {
            clipId,
            requestedType: "article",
            clipType: "article",
            pageUrl: `https://example.com/task-47812/${suffix}`,
            pageTitle: "TASK-478.12 Web Clipper Handoff",
            visibleBody: uniqueBody,
            fullExtract: `${uniqueBody}\n\nFull extracted article body.`,
            selectionText: uniqueBody,
            captureMetadata: {
              clipType: "article",
              actualType: "article",
              fallbackPath: ["article"]
            },
            capturedAt: new Date().toISOString()
          }
        }
      )

      await expect(
        sidepanel.getByRole("radio", { name: "Workspace" })
      ).toBeVisible()
      await sidepanel.getByRole("radio", { name: "Workspace" }).check({ force: true })
      await expect(sidepanel.getByLabel("Workspace ID")).toBeVisible()
      await sidepanel.getByLabel("Workspace ID").fill(workspaceId)

      const openedPagePromise = context.waitForEvent("page")
      await sidepanel.getByRole("button", { name: "Save and open" }).click()
      const openedPage = await openedPagePromise
      await openedPage.waitForLoadState("domcontentloaded")

      expect(openedPage.url()).toContain("#/research-workspace")
      expect(openedPage.url()).not.toContain("document-workspace")
      await expect(sidepanel.getByText("Clip saved")).toBeVisible()

      const clipStatusResponse = await apiFetch(
        normalizedServerUrl,
        apiKey,
        `/api/v1/web-clipper/${encodeURIComponent(clipId)}`
      )
      const clipStatus = await clipStatusResponse.json()
      expect(clipStatus.workspace_placements).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            workspace_id: workspaceId,
            source_note_id: clipId
          })
        ])
      )

      const notesResponse = await apiFetch(
        normalizedServerUrl,
        apiKey,
        `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/notes`
      )
      const notes = await notesResponse.json()
      expect(notes).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            title: "TASK-478.12 Web Clipper Handoff",
            content: expect.stringContaining(uniqueBody)
          })
        ])
      )

      const sourceStatusResponse = await apiFetch(
        normalizedServerUrl,
        apiKey,
        `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/sources/status`
      )
      const sourceStatus = await sourceStatusResponse.json()
      expect(sourceStatus.workspace_id).toBe(workspaceId)
      expect(Array.isArray(sourceStatus.sources)).toBe(true)
      const expectedSourceId = `web-clipper:${clipId}`
      const promotedSource = sourceStatus.sources.find(
        (source: { id?: string; media_id?: number }) => source.id === expectedSourceId
      ) as { media_id?: number } | undefined
      expect(promotedSource).toEqual(
        expect.objectContaining({
          id: expectedSourceId,
          workspace_id: workspaceId,
          title: "TASK-478.12 Web Clipper Handoff",
          source_type: "web_clip",
          url: `https://example.com/task-47812/${suffix}`,
          state: expect.stringMatching(/^(partially_queryable|queryable)$/),
          readiness: expect.objectContaining({
            metadata_ready: true,
            text_extracted: true,
            fts_ready: true,
            citation_ready: true
          })
        })
      )
      expect(promotedSource?.media_id).toEqual(expect.any(Number))
      expect(promotedSource?.media_id).toBeGreaterThan(0)
    } finally {
      await context.close()
    }
  })
})

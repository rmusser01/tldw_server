import { chromium, expect, test, type BrowserContext, type Page } from "@playwright/test"
import { spawn } from "node:child_process"
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs"
import { createServer, type Server } from "node:http"
import { tmpdir } from "node:os"
import path from "node:path"

import {
  MANUAL_API_KEY,
  startManualApiKeyFixture,
} from "./helpers/manual-api-key-fixture"

const INITIAL_API_KEY =
  process.env.TLDW_COOKIE_LIFECYCLE_API_KEY ||
  "THIS-IS-A-SECURE-KEY-123-FAKE-KEY"
const ROTATED_API_KEY =
  process.env.TLDW_COOKIE_LIFECYCLE_ROTATED_API_KEY ||
  "THIS-IS-A-ROTATED-KEY-456-FAKE-KEY"
const API_PORT = Number(process.env.TLDW_COOKIE_LIFECYCLE_API_PORT || "18001")
const HOSTILE_PORT = Number(
  process.env.TLDW_COOKIE_LIFECYCLE_HOSTILE_PORT || "18002"
)
const WEB_URL = (process.env.TLDW_WEB_URL || "http://localhost:8080").replace(
  /\/$/,
  ""
)
const API_URL = `http://127.0.0.1:${API_PORT}`
const SESSION_COOKIE_NAME = "tldw_single_user_session"
const PRESERVED_REMOTE_URL = `http://127.0.0.1:${HOSTILE_PORT}`
const PRESERVED_REMOTE_API_KEY = MANUAL_API_KEY

const repoRoot = path.resolve(__dirname, "..", "..", "..")
const lifecycleScript = path.join(
  repoRoot,
  "tldw_Server_API",
  "scripts",
  "server_lifecycle.py"
)
const python =
  process.env.TLDW_E2E_PYTHON ||
  (existsSync(path.join(repoRoot, ".venv", "bin", "python"))
    ? path.join(repoRoot, ".venv", "bin", "python")
    : process.env.PYTHON || "python3")
const runtimeRoot = mkdtempSync(
  path.join(tmpdir(), "tldw-single-user-cookie-lifecycle-")
)
const serverLabel = `single-user-cookie-lifecycle-${process.pid}`
const serverLog = path.join(repoRoot, `server-${serverLabel}.log`)
const serverPid = path.join(repoRoot, `server-${serverLabel}.pid`)

let backendKey = INITIAL_API_KEY
let backendRunning = false
let infrastructureSkipReason: string | null = null

const backendEnvironment = (apiKey: string): NodeJS.ProcessEnv => ({
  ...process.env,
  SERVER_LABEL: serverLabel,
  SERVER_PORT: String(API_PORT),
  E2E_TEST_BASE_URL: API_URL,
  STARTUP_TIMEOUT_SECONDS: "90",
  AUTH_MODE: "single_user",
  SINGLE_USER_API_KEY: apiKey,
  DATABASE_URL: `sqlite:///${path.join(runtimeRoot, "authnz.db")}`,
  JOBS_DB_PATH: path.join(runtimeRoot, "jobs.db"),
  USER_DB_BASE_DIR: path.join(runtimeRoot, "user-databases"),
  SESSION_ENCRYPTION_KEY: "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
  SINGLE_USER_SESSION_COOKIE_NAME: SESSION_COOKIE_NAME,
  SESSION_COOKIE_SECURE: "0",
  CSRF_ENABLED: "1",
  TEST_MODE: "true",
  PYTEST_CURRENT_TEST: "single-user-cookie-lifecycle",
  DEFER_HEAVY_STARTUP: "true",
  AUTHNZ_FORCE_REAL_SESSION_MANAGER: "1",
  MINIMAL_TEST_INCLUDE_AUDIO: "1",
  EPHEMERAL_CLEANUP_ENABLED: "false",
  CLAIMS_REBUILD_ENABLED: "false",
  ALLOWED_ORIGINS: WEB_URL,
})

const runLifecycle = async (
  command: "start" | "health-check" | "stop",
  apiKey: string
): Promise<void> => {
  await new Promise<void>((resolve, reject) => {
    const child = spawn(python, [lifecycleScript, command], {
      cwd: repoRoot,
      env: backendEnvironment(apiKey),
      stdio: ["ignore", "pipe", "pipe"],
    })
    let output = ""
    child.stdout?.on("data", (chunk) => {
      output += String(chunk)
    })
    child.stderr?.on("data", (chunk) => {
      output += String(chunk)
    })
    const timeout = setTimeout(() => {
      child.kill("SIGTERM")
      reject(new Error(`Backend lifecycle ${command} timed out`))
    }, command === "health-check" ? 100_000 : 45_000)

    child.once("error", (error) => {
      clearTimeout(timeout)
      reject(error)
    })
    child.once("exit", (code) => {
      clearTimeout(timeout)
      if (code === 0) {
        resolve()
        return
      }
      reject(
        new Error(
          `Backend lifecycle ${command} exited ${code ?? "without a code"}: ${output
            .trim()
            .slice(-2_000)}`
        )
      )
    })
  })
}

const startBackend = async (apiKey: string): Promise<void> => {
  await runLifecycle("start", apiKey)
  try {
    await runLifecycle("health-check", apiKey)
  } catch (error) {
    await runLifecycle("stop", apiKey).catch(() => undefined)
    throw error
  }
  backendKey = apiKey
  backendRunning = true
}

const stopBackend = async (): Promise<void> => {
  if (!backendRunning && !existsSync(serverPid)) return
  await runLifecycle("stop", backendKey)
  backendRunning = false
}

const readBackendLog = (): string => {
  try {
    return readFileSync(serverLog, "utf8")
  } catch {
    return ""
  }
}

const isListenerInfrastructureFailure = (error: unknown): boolean => {
  const details = `${error instanceof Error ? error.message : String(error)}\n${readBackendLog()}`
  return /operation not permitted|permission denied|EACCES|EPERM/i.test(
    details
  )
}

const withPersistentBrowser = async <T>(
  userDataDir: string,
  callback: (context: BrowserContext, page: Page) => Promise<T>
): Promise<T> => {
  const context = await chromium.launchPersistentContext(userDataDir, {
    headless: true,
    baseURL: WEB_URL,
    locale: "en-US",
  })
  await context.addInitScript(() => {
    localStorage.setItem("__tldw_first_run_complete", "true")
    localStorage.setItem("assistant_setup_dismissed", "true")
  })
  const page = context.pages()[0] || (await context.newPage())
  try {
    return await callback(context, page)
  } finally {
    await context.close()
  }
}

const expectCookieAuthentication = async (page: Page): Promise<void> => {
  try {
    await expect
      .poll(async () =>
        (await page.context().cookies(`${WEB_URL}/api/`)).some(
          (cookie) => cookie.name === SESSION_COOKIE_NAME
        ),
        { timeout: 60_000 }
      )
      .toBe(true)
  } catch (error) {
    const diagnostics = await page.evaluate(async () => {
      const runtimeConfig = await fetch("/api/_tldw-webui/runtime-config", {
        credentials: "same-origin",
        cache: "no-store",
      })
      const session = await fetch("/api/_tldw-webui/session", {
        method: "POST",
        credentials: "include",
        cache: "no-store",
      })
      const profile = await fetch("/api/v1/users/me/profile", {
        credentials: "same-origin",
        cache: "no-store",
      })
      return {
        location: window.location.href,
        readyState: document.readyState,
        runtimeConfigStatus: runtimeConfig.status,
        runtimeConfigBody: await runtimeConfig.text(),
        sessionStatus: session.status,
        profileStatus: profile.status,
      }
    })
    throw new Error(
      `Cookie-session bootstrap failed: ${JSON.stringify(diagnostics)}`,
      { cause: error }
    )
  }

  await expect
    .poll(
      async () => {
        try {
          return await page.evaluate(async () =>
            fetch("/api/v1/auth/sessions", {
              credentials: "same-origin",
              cache: "no-store",
            }).then((response) => response.status)
          )
        } catch {
          return 0
        }
      },
      { timeout: 60_000 }
    )
    .toBe(200)
}

const browserSecretInventory = async (page: Page): Promise<string[]> =>
  page.evaluate(
    async ({ fixtureKeys }) => {
      const findings: string[] = []
      const seen = new WeakSet<object>()
      const inspect = (value: unknown, location: string, keyName = "") => {
        if (typeof value === "string") {
          if (
            fixtureKeys.includes(value) ||
            (/api.?key/i.test(keyName) && value.trim().length > 0)
          ) {
            findings.push(location)
          }
          try {
            inspect(JSON.parse(value), `${location}:json`)
          } catch {
            // Plain strings are expected in browser storage.
          }
          return
        }
        if (!value || typeof value !== "object" || seen.has(value)) return
        seen.add(value)
        for (const [key, child] of Object.entries(value as Record<string, unknown>)) {
          inspect(child, `${location}.${key}`, key)
        }
      }

      for (const [name, storage] of [
        ["localStorage", localStorage],
        ["sessionStorage", sessionStorage],
      ] as const) {
        for (let index = 0; index < storage.length; index += 1) {
          const key = storage.key(index)
          if (key) inspect(storage.getItem(key), `${name}.${key}`, key)
        }
      }

      const indexedDbWithCatalog = indexedDB as IDBFactory & {
        databases?: () => Promise<Array<{ name?: string; version?: number }>>
      }
      for (const database of (await indexedDbWithCatalog.databases?.()) || []) {
        if (!database.name) continue
        const opened = await new Promise<IDBDatabase | null>((resolve) => {
          const request = indexedDB.open(database.name as string)
          request.onsuccess = () => resolve(request.result)
          request.onerror = () => resolve(null)
        })
        if (!opened) continue
        for (const storeName of Array.from(opened.objectStoreNames)) {
          const values = await new Promise<unknown[]>((resolve) => {
            const request = opened
              .transaction(storeName, "readonly")
              .objectStore(storeName)
              .getAll()
            request.onsuccess = () => resolve(request.result)
            request.onerror = () => resolve([])
          })
          inspect(values, `indexedDB.${database.name}.${storeName}`)
        }
        opened.close()
      }

      inspect(
        (window as typeof window & { __NEXT_DATA__?: unknown }).__NEXT_DATA__,
        "window.__NEXT_DATA__"
      )
      inspect(document.documentElement.outerHTML, "document")
      return findings
    },
    { fixtureKeys: [INITIAL_API_KEY, ROTATED_API_KEY] }
  )

type WebSocketResult = {
  opened: boolean
  closeCode: number
  url: string
}

const inspectWebSocket = async (page: Page, url: string): Promise<WebSocketResult> =>
  page.evaluate(
    ({ target }) =>
      new Promise<WebSocketResult>((resolve) => {
        const socket = new WebSocket(target)
        let settled = false
        const finish = (result: WebSocketResult) => {
          if (settled) return
          settled = true
          resolve(result)
        }
        const timeout = window.setTimeout(() => {
          socket.close()
          finish({ opened: false, closeCode: -1, url: socket.url })
        }, 15_000)
        socket.onopen = () => {
          window.clearTimeout(timeout)
          const openedUrl = socket.url
          socket.close(1000)
          finish({ opened: true, closeCode: 1000, url: openedUrl })
        }
        socket.onclose = (event) => {
          window.clearTimeout(timeout)
          finish({ opened: false, closeCode: event.code, url: socket.url })
        }
        socket.onerror = () => undefined
      }),
    { target: url }
  )

const startHostileOrigin = async (): Promise<Server> => {
  const server = createServer((_request, response) => {
    response.writeHead(200, { "Content-Type": "text/html; charset=utf-8" })
    response.end("<!doctype html><title>hostile origin fixture</title>")
  })
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject)
    server.listen(HOSTILE_PORT, "127.0.0.1", () => resolve())
  })
  return server
}

test.describe.serial("single-user HttpOnly cookie lifecycle", () => {
  test.describe.configure({ timeout: 240_000 })

  test.beforeAll(async () => {
    test.setTimeout(240_000)
    try {
      await startBackend(INITIAL_API_KEY)
    } catch (error) {
      if (!isListenerInfrastructureFailure(error)) throw error
      infrastructureSkipReason =
        "Live lifecycle infrastructure cannot bind the isolated backend listener in this environment"
    }
  })

  test.afterAll(async () => {
    test.setTimeout(120_000)
    await stopBackend().catch(() => undefined)
    rmSync(serverPid, { force: true })
    rmSync(serverLog, { force: true })
    rmSync(runtimeRoot, { recursive: true, force: true })
  })

  test("persists, revokes, rotates, and authenticates secret-free WebSockets", async ({}, testInfo) => {
    test.skip(Boolean(infrastructureSkipReason), infrastructureSkipReason || undefined)

    const profile = testInfo.outputPath("persistent-profile")
    const observedBrowserUrls: string[] = []
    let originalCookieValue = ""

    await withPersistentBrowser(profile, async (context, page) => {
      page.on("request", (request) => observedBrowserUrls.push(request.url()))
      page.on("websocket", (socket) => observedBrowserUrls.push(socket.url()))
      await page.goto(`${WEB_URL}/settings/chat`, {
        waitUntil: "domcontentloaded",
      })
      await expectCookieAuthentication(page)
      await expect(
        page.getByRole("radio", { name: /split brief/i })
      ).toBeVisible({ timeout: 60_000 })

      const runtimeConfig = await page.evaluate(async () =>
        fetch("/api/_tldw-webui/runtime-config", {
          credentials: "same-origin",
          cache: "no-store",
        }).then((response) => response.json())
      )
      expect(runtimeConfig).toEqual({
        runtimeAuth: {
          available: true,
          authMode: "single-user",
          transport: "cookie-session",
        },
        networking: { deploymentMode: "quickstart", serverUrl: "" },
      })
      expect(JSON.stringify(runtimeConfig)).not.toContain(INITIAL_API_KEY)
      expect(await browserSecretInventory(page)).toEqual([])
      expect(await page.evaluate(() => document.cookie)).not.toContain(
        `${SESSION_COOKIE_NAME}=`
      )

      const sessionCookie = (await context.cookies()).find(
        (cookie) => cookie.name === SESSION_COOKIE_NAME
      )
      expect(sessionCookie).toBeDefined()
      expect(sessionCookie).toMatchObject({
        httpOnly: true,
        secure: false,
        sameSite: "Lax",
        path: "/api",
        domain: "localhost",
      })
      expect(sessionCookie!.expires - Date.now() / 1000).toBeGreaterThan(
        29 * 24 * 60 * 60
      )
      expect(sessionCookie!.expires - Date.now() / 1000).toBeLessThanOrEqual(
        30 * 24 * 60 * 60
      )
      originalCookieValue = sessionCookie!.value

      await page.reload({ waitUntil: "domcontentloaded" })
      await expectCookieAuthentication(page)
      expect(
        (await context.cookies()).find(
          (cookie) => cookie.name === SESSION_COOKIE_NAME
        )?.value
      ).toBe(originalCookieValue)
      expect(await browserSecretInventory(page)).toEqual([])

      await page.evaluate(
        ({ apiKey, serverUrl }) => {
          localStorage.setItem(
            "tldwConfig",
            JSON.stringify({
              serverUrl,
              authMode: "single-user",
              authSource: "manual",
              credentialSource: "manual",
              apiKeyPersistence: "device",
              apiKeyServerOrigin: serverUrl,
              apiKey,
            })
          )
        },
        {
          apiKey: PRESERVED_REMOTE_API_KEY,
          serverUrl: PRESERVED_REMOTE_URL,
        }
      )
    })

    await withPersistentBrowser(profile, async (context, page) => {
      page.on("request", (request) => observedBrowserUrls.push(request.url()))
      page.on("websocket", (socket) => observedBrowserUrls.push(socket.url()))
      await page.goto(`${WEB_URL}/settings/chat`, {
        waitUntil: "domcontentloaded",
      })
      await expectCookieAuthentication(page)
      await expect(
        page.getByRole("radio", { name: /split brief/i })
      ).toBeVisible({ timeout: 60_000 })
      expect(
        (await context.cookies()).find(
          (cookie) => cookie.name === SESSION_COOKIE_NAME
        )?.value
      ).toBe(originalCookieValue)

      expect(
        await page.evaluate(() =>
          JSON.parse(localStorage.getItem("tldwConfig") || "null")
        )
      ).toEqual({
        serverUrl: PRESERVED_REMOTE_URL,
        authMode: "single-user",
        authSource: "manual",
        credentialSource: "manual",
        apiKeyPersistence: "device",
        apiKeyServerOrigin: PRESERVED_REMOTE_URL,
        apiKey: PRESERVED_REMOTE_API_KEY,
      })

      const profilePatch = page.waitForResponse(
        (response) =>
          response.request().method() === "PATCH" &&
          new URL(response.url()).pathname === "/api/v1/users/me/profile"
      )
      await page.getByRole("radio", { name: /split brief/i }).click()
      const profilePatchResponse = await profilePatch
      expect(profilePatchResponse.status()).toBe(200)
      const profilePatchHeaders = await profilePatchResponse
        .request()
        .allHeaders()
      expect(profilePatchHeaders["x-csrf-token"]).toBeTruthy()
      expect(profilePatchHeaders["x-api-key"]).toBeUndefined()
      expect(profilePatchHeaders.authorization).toBeUndefined()

      await page.goto(`${WEB_URL}/api/_tldw-webui/runtime-config`, {
        waitUntil: "load",
      })

      const wsBase = WEB_URL.replace(/^http/, "ws")
      const representativeSockets = [
        `${wsBase}/api/v1/persona/stream`,
        `${wsBase}/api/v1/acp/multiplex`,
        `${wsBase}/api/v1/audio/stream/transcribe`,
      ]
      for (const socketUrl of representativeSockets) {
        const result = await inspectWebSocket(page, socketUrl)
        expect(result.opened, `${socketUrl} did not authenticate through the cookie`).toBe(
          true
        )
        expect(new URL(result.url).searchParams.has("api_key")).toBe(false)
        expect(new URL(result.url).searchParams.has("token")).toBe(false)
        expect(result.url).not.toContain(INITIAL_API_KEY)
      }

      const hostileServer = await startHostileOrigin()
      try {
        const hostilePage = await context.newPage()
        await hostilePage.goto(`http://localhost:${HOSTILE_PORT}`, {
          waitUntil: "domcontentloaded",
        })
        const rejected = await inspectWebSocket(
          hostilePage,
          `${wsBase}/api/v1/persona/stream`
        )
        expect(rejected.opened).toBe(false)
        expect([1006, 4403]).toContain(rejected.closeCode)
      } finally {
        await new Promise<void>((resolve) => hostileServer.close(() => resolve()))
      }

      const csrfToken = await page.evaluate(() =>
        document.cookie
          .split("; ")
          .find((entry) => entry.startsWith("csrf_token="))
          ?.slice("csrf_token=".length)
      )
      expect(csrfToken).toBeTruthy()
      await page.goto(`${WEB_URL}/settings/tldw`, {
        waitUntil: "domcontentloaded",
      })
      await expect(
        page.getByText("Connected securely through this WebUI.")
      ).toBeVisible({ timeout: 60_000 })
      expect(
        await page.evaluate(() =>
          localStorage.getItem("tldwCookieSessionConfig")
        )
      ).not.toBeNull()
      const preLogoutCheck = page
        .getByRole("button", { name: /^(Recheck|Test Connection)$/ })
        .first()
      await preLogoutCheck.click()
      await expect(page.getByText("Core: reachable")).toBeVisible()
      await expect(page.getByText("RAG: healthy")).toBeVisible()
      await expect(preLogoutCheck).not.toHaveAttribute("aria-busy", "true")

      const preservedRemote = await startManualApiKeyFixture(HOSTILE_PORT)
      try {
        const logoutResponsePromise = page.waitForResponse(
          (response) =>
            response.request().method() === "DELETE" &&
            new URL(response.url()).pathname ===
              "/api/v1/auth/single-user/session"
        )
        await page.getByRole("button", { name: "Logout" }).click()
        const logoutResponse = await logoutResponsePromise
        expect(logoutResponse.status()).toBe(200)
        expect(logoutResponse.headers()["cache-control"]).toBe("no-store")
        expect(
          (await context.cookies()).some(
            (cookie) => cookie.name === SESSION_COOKIE_NAME
          )
        ).toBe(false)
        expect(
          await page.evaluate(() =>
            localStorage.getItem("tldwCookieSessionConfig")
          )
        ).toBeNull()
        expect(
          await page.evaluate(() =>
            JSON.parse(localStorage.getItem("tldwConfig") || "null")
          )
        ).toEqual({
          serverUrl: PRESERVED_REMOTE_URL,
          authMode: "single-user",
          authSource: "manual",
          credentialSource: "manual",
          apiKeyPersistence: "device",
          apiKeyServerOrigin: PRESERVED_REMOTE_URL,
          apiKey: PRESERVED_REMOTE_API_KEY,
        })
        await expect(page.getByText("Core: not checked yet")).toBeVisible()
        await expect(page.getByText("RAG: not checked yet")).toBeVisible()

        const remoteRequestOffset = preservedRemote.requests().length
        await page
          .getByRole("button", { name: /^(Recheck|Test Connection)$/ })
          .first()
          .click()
        await expect
          .poll(() =>
            preservedRemote
              .requests()
              .slice(remoteRequestOffset)
              .some((request) => request.authenticated)
          )
          .toBe(true)
      } finally {
        await preservedRemote.close()
      }

      await context.addCookies([
        {
          name: SESSION_COOKIE_NAME,
          value: originalCookieValue,
          domain: "localhost",
          path: "/api",
          httpOnly: true,
          secure: false,
          sameSite: "Lax",
          expires: Date.now() / 1000 + 30 * 24 * 60 * 60,
        },
      ])
      const staleSession = await page.evaluate(async (token) => {
        const protectedResponse = await fetch("/api/v1/auth/sessions", {
            credentials: "same-origin",
            cache: "no-store",
          })
        const logoutResponse = await fetch(
          "/api/v1/auth/single-user/session",
          {
            method: "DELETE",
            credentials: "same-origin",
            headers: { "X-CSRF-Token": token || "" },
          }
        )
        return {
          protectedStatus: protectedResponse.status,
          logoutStatus: logoutResponse.status,
          cacheControl: logoutResponse.headers.get("cache-control"),
        }
      }, csrfToken)
      expect(staleSession).toEqual({
        protectedStatus: 401,
        logoutStatus: 200,
        cacheControl: "no-store",
      })
      expect(
        (await context.cookies()).some(
          (cookie) => cookie.name === SESSION_COOKIE_NAME
        )
      ).toBe(false)

      const repeatedLogout = await page.evaluate(async () => {
        const response = await fetch("/api/v1/auth/single-user/session", {
          method: "DELETE",
          credentials: "same-origin",
        })
        return {
          status: response.status,
          cacheControl: response.headers.get("cache-control"),
        }
      })
      expect(repeatedLogout).toEqual({
        status: 200,
        cacheControl: "no-store",
      })
    })

    let reprovisionedCookieValue = ""
    await withPersistentBrowser(profile, async (context, page) => {
      await page.goto(`${WEB_URL}/chat`, { waitUntil: "domcontentloaded" })
      await expectCookieAuthentication(page)
      reprovisionedCookieValue =
        (await context.cookies()).find(
          (cookie) => cookie.name === SESSION_COOKIE_NAME
        )?.value || ""
      expect(reprovisionedCookieValue).toBeTruthy()
      expect(reprovisionedCookieValue).not.toBe(originalCookieValue)

      await stopBackend()
      await startBackend(ROTATED_API_KEY)
      expect(
        await page.evaluate(async () =>
          fetch("/api/v1/auth/sessions", {
            credentials: "same-origin",
            cache: "no-store",
          }).then((response) => response.status)
        )
      ).toBe(401)
      await page.evaluate(() => localStorage.removeItem("tldwConfig"))
      expect(await browserSecretInventory(page)).toEqual([])
    })

    expect(reprovisionedCookieValue).not.toBe(originalCookieValue)
    expect(observedBrowserUrls).not.toEqual(
      expect.arrayContaining([
        expect.stringMatching(/[?&](?:api_key|token)=/i),
        expect.stringContaining(INITIAL_API_KEY),
        expect.stringContaining(ROTATED_API_KEY),
      ])
    )
  })
})

import {
  test,
  expect,
  seedAuth,
  getCriticalIssues,
  classifySmokeIssues
} from "./smoke.setup"
import { waitForAppShell } from "../utils/helpers"
import type { DiagnosticsData } from "./smoke.setup"
import type { Page } from "@playwright/test"

const LOAD_TIMEOUT = 60_000
const NAVIGATION_MAX_ATTEMPTS = 3

const LISTED_ANTD_DEPRECATION_PATTERNS = [
  /\[antd:\s*Drawer\].*`width`\s+is\s+deprecated/i,
  /\[antd:\s*Space\].*`direction`\s+is\s+deprecated/i,
  /\[antd:\s*Alert\].*`message`\s+is\s+deprecated/i,
  /\[antd:\s*List\].*deprecated/i
]
const MAX_UPDATE_DEPTH_PATTERN = /Maximum update depth exceeded/i
const UNRESOLVED_TEMPLATE_PATTERN = /\{\{[^{}\n]{1,120}\}\}/g
const NAVIGATION_RETRY_WAIT_MS = 1_500

const clearDiagnostics = (diagnostics: DiagnosticsData) => {
  diagnostics.console.length = 0
  diagnostics.pageErrors.length = 0
  diagnostics.requestFailures.length = 0
}

const isTransientNavigationError = (error: unknown): boolean => {
  const message = error instanceof Error ? error.message : String(error ?? "")
  return (
    /ERR_CONNECTION_REFUSED/i.test(message) ||
    /ERR_EMPTY_RESPONSE/i.test(message) ||
    /Timeout .* exceeded/i.test(message)
  )
}

const gotoCriticalRoute = async (
  page: Page,
  diagnostics: DiagnosticsData,
  path: string,
  timeoutMs: number
) => {
  let lastError: unknown

  for (let attempt = 1; attempt <= NAVIGATION_MAX_ATTEMPTS; attempt += 1) {
    try {
      const response = await page.goto(path, {
        waitUntil: "domcontentloaded",
        timeout: timeoutMs
      })
      return response
    } catch (error) {
      lastError = error
      if (!isTransientNavigationError(error)) {
        throw error
      }
      if (attempt === NAVIGATION_MAX_ATTEMPTS) {
        const message = error instanceof Error ? error.message : String(error ?? "")
        throw new Error(
          `Transient navigation failure after ${NAVIGATION_MAX_ATTEMPTS} attempts for ${path}: ${message}`
        )
      }
      clearDiagnostics(diagnostics)
      await new Promise((resolve) => setTimeout(resolve, NAVIGATION_RETRY_WAIT_MS))
    }
  }

  if (isTransientNavigationError(lastError)) {
    return null
  }

  throw lastError instanceof Error ? lastError : new Error(`Navigation failed for ${path}`)
}

type CriticalRoute = {
  path: string
  name: string
  expectedPath?: string
  allowRedirectPanel?: boolean
  loadTimeoutMs?: number
}

const CRITICAL_ROUTES: CriticalRoute[] = [
  { path: "/chat", name: "Chat" },
  { path: "/chat-workspace", name: "Chat Workspace" },
  { path: "/settings", name: "Settings" },
  { path: "/chat/settings", name: "Chat Settings", expectedPath: "/settings/chat" },
  { path: "/settings/chatbooks", name: "Chatbooks Settings" },
  { path: "/chatbooks", name: "Chatbooks" },
  { path: "/flashcards", name: "Flashcards" },
  { path: "/admin/llamacpp", name: "LlamaCpp Admin" },
  { path: "/content-review", name: "Content Review" },
  {
    path: "/claims-review",
    name: "Claims Review",
    expectedPath: "/content-review",
    allowRedirectPanel: true
  },
  { path: "/research-workspace", name: "Research Workspace" },
  { path: "/writing-playground", name: "Writing Playground" },
  { path: "/stt", name: "STT" },
  { path: "/speech", name: "Speech" }
]

test.describe("Stage 5 release gate", () => {
  test.describe.configure({ mode: "parallel" })

  for (const route of CRITICAL_ROUTES) {
    test(`enforces UX console/error budget on ${route.name} (${route.path})`, async ({
      page,
      diagnostics
    }) => {
      const routeLoadTimeout = route.loadTimeoutMs || LOAD_TIMEOUT
      test.setTimeout(
        Math.max(180_000, routeLoadTimeout * NAVIGATION_MAX_ATTEMPTS + 45_000)
      )
      await seedAuth(page)

      const response = await gotoCriticalRoute(
        page,
        diagnostics,
        route.path,
        routeLoadTimeout
      )
      expect(response, `Route navigation did not produce a response for ${route.path}`).not.toBeNull()
      if (!response) return

      const status = response.status() ?? 0
      expect(status, `Route returned non-success status for ${route.path}`).toBeGreaterThanOrEqual(
        200
      )
      expect(status, `Route returned non-success status for ${route.path}`).toBeLessThan(400)

      const expectedPath = route.expectedPath || route.path
      let resolvedViaRedirectPanel = false
      if (route.allowRedirectPanel) {
        await page.waitForFunction(
          ({ targetPath, allowRedirectPanel }) => {
            return (
              window.location.pathname === targetPath ||
              (allowRedirectPanel &&
                Boolean(document.querySelector('[data-testid="route-redirect-panel"]')))
            )
          },
          {
            targetPath: expectedPath,
            allowRedirectPanel: route.allowRedirectPanel
          },
          {
            timeout: routeLoadTimeout
          }
        )
        const redirectPanel = page.getByTestId("route-redirect-panel")
        resolvedViaRedirectPanel =
          route.allowRedirectPanel &&
          (await redirectPanel.isVisible().catch(() => false))
      } else {
        await page.waitForURL((url) => url.pathname === expectedPath, {
          timeout: routeLoadTimeout
        })
      }
      await waitForAppShell(page, routeLoadTimeout)

      if (!resolvedViaRedirectPanel) {
        await page.waitForURL((url) => url.pathname === expectedPath, {
          timeout: routeLoadTimeout
        })
      }

      const issues = getCriticalIssues(diagnostics)
      const classified = classifySmokeIssues(route.path, issues)

      expect(
        issues.pageErrors,
        `Uncaught page errors on ${route.path}: ${issues.pageErrors
          .map((entry) => entry.message)
          .join(" | ")}`
      ).toHaveLength(0)

      expect(
        classified.unexpectedConsoleErrors,
        `Unexpected console errors on ${route.path}: ${classified.unexpectedConsoleErrors
          .map((entry) => entry.text)
          .join(" | ")}`
      ).toHaveLength(0)

      expect(
        classified.unexpectedRequestFailures,
        `Unexpected request failures on ${route.path}: ${classified.unexpectedRequestFailures
          .map((entry) => `${entry.url} (${entry.errorText})`)
          .join(" | ")}`
      ).toHaveLength(0)

      const listedAntdDeprecations = diagnostics.console.filter((entry) =>
        LISTED_ANTD_DEPRECATION_PATTERNS.some((pattern) => pattern.test(entry.text))
      )
      expect(
        listedAntdDeprecations,
        `Listed AntD deprecations on ${route.path}: ${listedAntdDeprecations
          .map((entry) => entry.text)
          .join(" | ")}`
      ).toHaveLength(0)

      const maxDepthConsole = diagnostics.console.filter((entry) =>
        MAX_UPDATE_DEPTH_PATTERN.test(entry.text)
      )
      const maxDepthPage = diagnostics.pageErrors.filter((entry) =>
        MAX_UPDATE_DEPTH_PATTERN.test(entry.message)
      )
      expect(
        maxDepthConsole,
        `Maximum update depth console warnings on ${route.path}: ${maxDepthConsole
          .map((entry) => entry.text)
          .join(" | ")}`
      ).toHaveLength(0)
      expect(
        maxDepthPage,
        `Maximum update depth page errors on ${route.path}: ${maxDepthPage
          .map((entry) => entry.message)
          .join(" | ")}`
      ).toHaveLength(0)

      const bodyText = await page.evaluate(() => document.body?.innerText || "")
      const unresolvedTemplates = Array.from(bodyText.matchAll(UNRESOLVED_TEMPLATE_PATTERN)).map(
        (match) => match[0]
      )
      const uniqueUnresolvedTemplates = Array.from(new Set(unresolvedTemplates))

      expect(
        uniqueUnresolvedTemplates,
        `Unresolved template placeholders on ${route.path}: ${uniqueUnresolvedTemplates.join(
          " | "
        )}`
      ).toHaveLength(0)
    })
  }
})

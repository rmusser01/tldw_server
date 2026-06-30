import type { Page, Route } from "@playwright/test"
import {
  test,
  expect,
  seedAuth,
  getCriticalIssues,
  classifySmokeIssues,
  SMOKE_LOAD_TIMEOUT
} from "./smoke.setup"
import { stubNotificationsApi, waitForAppShell } from "../utils/helpers"

type RouteFixture = {
  name: string
  path: string
  openApiPaths?: string[]
  failingApiPaths?: string[]
  diagnosis: RegExp
  action: RegExp
  rawPrimaryPattern: RegExp
  expectsRawDetails?: boolean
}

const LOAD_TIMEOUT = SMOKE_LOAD_TIMEOUT

const json = async (route: Route, status: number, body: unknown) => {
  await route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(body)
  })
}

const missingEndpointBody = (pathname: string) => ({
  detail: `Not Found (GET ${pathname})`
})

const toOpenApiSpec = (paths: string[] = []) => ({
  openapi: "3.0.0",
  info: { title: "tldw capability governance fixture", version: "e2e" },
  paths: Object.fromEntries(paths.map((path) => [path, {}]))
})

const installGovernanceBackend = async (
  page: Page,
  fixture: RouteFixture
): Promise<void> => {
  const failingPaths = new Set(fixture.failingApiPaths ?? [])

  await page.addInitScript(() => {
    localStorage.removeItem("__tldwServerCapabilitiesCacheV5")
    sessionStorage.removeItem("__tldwServerCapabilitiesCacheV5")
  })

  await stubNotificationsApi(page)

  await page.route("**/*", async (route) => {
    const request = route.request()
    const url = new URL(request.url())
    const { pathname } = url

    if (pathname === "/openapi.json") {
      await json(route, 200, toOpenApiSpec(fixture.openApiPaths))
      return
    }

    if (pathname === "/api/v1/config/docs-info") {
      await json(route, 200, {
        info: { version: "e2e" },
        capabilities: {
          hasAudio: false,
          hasStt: false,
          hasTts: false
        }
      })
      return
    }

    if (pathname === "/api/v1/health") {
      await json(route, 200, { status: "ok" })
      return
    }

    if (failingPaths.has(pathname)) {
      await json(route, 404, missingEndpointBody(pathname))
      return
    }

    if (pathname === "/api/v1/users/keys/openai/oauth/status") {
      await json(route, 200, { connected: false })
      return
    }

    if (pathname === "/api/v1/users/keys") {
      await json(route, 200, { keys: [] })
      return
    }

    if (pathname === "/api/v1/llm/models/metadata") {
      await json(route, 200, { models: [], total: 0 })
      return
    }

    if (pathname === "/api/v1/evaluations/recipes") {
      await json(route, 200, [])
      return
    }

    if (pathname === "/api/v1/evaluations") {
      await json(route, 200, { items: [], total: 0 })
      return
    }

    if (pathname === "/api/v1/mcp/hub/tool-registry/summary") {
      await json(route, 200, { entries: [], modules: [] })
      return
    }

    if (pathname === "/api/v1/mcp/hub/external-servers") {
      await json(route, 200, [])
      return
    }

    if (pathname === "/api/v1/mcp/hub/external-servers/refresh-discovery") {
      await json(route, 200, { ok: true, message: null, errors: {} })
      return
    }

    if (pathname === "/api/v1/skills") {
      await json(route, 200, { skills: [], total: 0 })
      return
    }

    if (pathname === "/api/v1/data-tables") {
      await json(route, 200, { tables: [], total: 0, page: 1, page_size: 10 })
      return
    }

    if (pathname === "/api/v1/audio/providers") {
      await json(route, 200, { providers: {}, voices: {} })
      return
    }

    if (pathname === "/api/v1/audio/voices/catalog") {
      await json(route, 200, { voices: [] })
      return
    }

    if (pathname.startsWith("/api/v1/")) {
      await json(route, 200, {})
      return
    }

    await route.continue()
  })
}

const visibleBodyText = async (page: Page): Promise<string> =>
  page.evaluate(() => {
    const hiddenAncestorSelector = [
      "details:not([open]) > :not(summary)",
      "[hidden]",
      "[aria-hidden='true']",
      "script",
      "style"
    ].join(",")
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT)
    const parts: string[] = []
    let node = walker.nextNode()
    while (node) {
      const text = node.textContent?.replace(/\s+/g, " ").trim()
      const parent = node.parentElement
      if (text && parent && !parent.closest(hiddenAncestorSelector)) {
        const style = window.getComputedStyle(parent)
        if (
          style.display !== "none" &&
          style.visibility !== "hidden" &&
          Number(style.opacity) !== 0
        ) {
          parts.push(text)
        }
      }
      node = walker.nextNode()
    }
    return parts.join(" ")
  })

const expectActionVisible = async (page: Page, action: RegExp) => {
  await expect(
    page
      .getByRole("button", { name: action })
      .or(page.getByRole("link", { name: action }))
      .first()
  ).toBeVisible({ timeout: LOAD_TIMEOUT })
}

const expectRawDetailsGoverned = async (page: Page, fixture: RouteFixture) => {
  const beforeDisclosure = await visibleBodyText(page)
  expect(
    beforeDisclosure,
    `${fixture.name} exposed raw endpoint or Not Found text as primary UI`
  ).not.toMatch(fixture.rawPrimaryPattern)

  if (!fixture.expectsRawDetails) {
    return
  }

  const disclosure = page
    .locator("summary")
    .filter({ hasText: /diagnostics|technical details|request details/i })
    .first()
  await expect(disclosure).toBeVisible({ timeout: LOAD_TIMEOUT })
  await disclosure.evaluate((element) => {
    const details = element.closest("details")
    if (details) {
      details.setAttribute("open", "")
      return
    }
    ;(element as HTMLElement).click()
  })

  const afterDisclosure = await visibleBodyText(page)
  expect(
    afterDisclosure,
    `${fixture.name} did not expose endpoint details after opening diagnostics`
  ).toMatch(fixture.rawPrimaryPattern)
}

type ConsoleIssueWithLocation = {
  type: string
  text: string
  location?: { url: string; lineNumber: number }
}

const isExpectedInjectedFailure = (
  entry: ConsoleIssueWithLocation,
  fixture: RouteFixture
): boolean => {
  if (!fixture.failingApiPaths?.length) return false
  const matchesFailingPath = fixture.failingApiPaths.some((path) => entry.text.includes(path))
  if (
    matchesFailingPath &&
    /Failed to fetch|Failed to load resource: the server responded with a status of 404|Not Found \(GET/i.test(
      entry.text
    )
  ) {
    return true
  }

  if (!/Failed to load resource: the server responded with a status of 404/i.test(entry.text)) {
    return false
  }
  const url = entry.location?.url
  if (!url) return false

  try {
    const pathname = new URL(url).pathname
    return fixture.failingApiPaths.some((path) => pathname === path)
  } catch {
    return false
  }
}

const removeExpectedInjectedFailures = (
  issues: ReturnType<typeof getCriticalIssues>,
  fixture: RouteFixture
): ReturnType<typeof getCriticalIssues> => ({
  ...issues,
  consoleErrors: issues.consoleErrors.filter(
    (entry) => !isExpectedInjectedFailure(entry as ConsoleIssueWithLocation, fixture)
  )
})

const fixtures: RouteFixture[] = [
  {
    name: "Sources",
    path: "/sources",
    diagnosis: /sources are unavailable|does not expose the ingestion sources capability/i,
    action: /check server setup/i,
    rawPrimaryPattern: /Not Found \(GET|\/api\/v1\/ingestion-sources/i
  },
  {
    name: "Scheduled Tasks",
    path: "/scheduled-tasks",
    diagnosis: /scheduled tasks are unavailable|does not expose the scheduled tasks capability/i,
    action: /check server setup/i,
    rawPrimaryPattern: /Not Found \(GET|\/api\/v1\/scheduled-tasks/i,
    expectsRawDetails: true
  },
  {
    name: "Integrations",
    path: "/integrations",
    diagnosis: /personal integrations are unavailable|does not expose the personal integrations capability/i,
    action: /check server setup/i,
    rawPrimaryPattern: /Not Found \(GET|\/api\/v1\/integrations\/personal/i,
    expectsRawDetails: true
  },
  {
    name: "Model Settings",
    path: "/settings/model",
    failingApiPaths: ["/api/v1/llm/models/metadata"],
    diagnosis: /unable to load models from server|no providers available/i,
    action: /retry|configure server/i,
    rawPrimaryPattern: /Not Found \(GET|\/api\/v1\/llm\/models\/metadata/i,
    expectsRawDetails: true
  },
  {
    name: "Evaluations",
    path: "/evaluations",
    failingApiPaths: ["/api/v1/evaluations/recipes"],
    diagnosis: /unable to load recipes|check server connection and try again/i,
    action: /try again|health.*diagnostics/i,
    rawPrimaryPattern: /Not Found \(GET|\/api\/v1\/evaluations\/recipes/i,
    expectsRawDetails: true
  },
  {
    name: "MCP Hub",
    path: "/mcp-hub?workflow=setup&view=tool-catalogs",
    failingApiPaths: [
      "/api/v1/mcp/hub/tool-registry/summary",
      "/api/v1/mcp/hub/external-servers"
    ],
    diagnosis: /failed to load tool registry metadata|server inventory unavailable/i,
    action: /refresh tools|retry server inventory/i,
    rawPrimaryPattern: /Not Found \(GET|\/api\/v1\/mcp\/hub\/(tool-registry\/summary|external-servers)/i,
    expectsRawDetails: true
  },
  {
    name: "Skills",
    path: "/skills",
    diagnosis: /skills not available|does not support the skills api/i,
    action: /check server setup|health.*diagnostics/i,
    rawPrimaryPattern: /Not Found \(GET|\/api\/v1\/skills/i
  },
  {
    name: "TTS",
    path: "/tts",
    diagnosis: /tldw audio\/speech api not detected|server tts: blocked/i,
    action: /open speech settings|settings/i,
    rawPrimaryPattern: /Not Found \(GET|\/api\/v1\/audio\/speech/i
  },
  {
    name: "Speech",
    path: "/speech",
    diagnosis: /tldw audio\/speech api not detected|server tts: blocked/i,
    action: /open speech settings|settings/i,
    rawPrimaryPattern: /Not Found \(GET|\/api\/v1\/audio\/speech/i
  },
  {
    name: "Data Tables",
    path: "/data-tables",
    failingApiPaths: ["/api/v1/data-tables"],
    diagnosis: /data tables could not load|unable to load tables|check diagnostics or try again/i,
    action: /refresh|try again/i,
    rawPrimaryPattern: /Not Found \(GET|\/api\/v1\/data-tables/i,
    expectsRawDetails: true
  }
]

test.describe("route capability state governance", () => {
  test.beforeEach(async ({ page }) => {
    await seedAuth(page)
  })

  for (const fixture of fixtures) {
    test(`${fixture.name} governs unavailable capability and raw error state`, async ({
      page,
      diagnostics
    }) => {
      test.setTimeout(90_000)
      await installGovernanceBackend(page, fixture)

      await page.goto(fixture.path, {
        waitUntil: "domcontentloaded",
        timeout: LOAD_TIMEOUT
      })
      await waitForAppShell(page, LOAD_TIMEOUT)

      await expect(page.getByText(fixture.diagnosis).first()).toBeVisible({
        timeout: LOAD_TIMEOUT
      })
      await expectActionVisible(page, fixture.action)
      await expectRawDetailsGoverned(page, fixture)

      const issues = removeExpectedInjectedFailures(getCriticalIssues(diagnostics), fixture)
      const classified = classifySmokeIssues(fixture.path, issues)
      expect(issues.pageErrors).toHaveLength(0)
      expect(classified.unexpectedConsoleErrors).toHaveLength(0)
      expect(classified.unexpectedRequestFailures).toHaveLength(0)
    })
  }
})

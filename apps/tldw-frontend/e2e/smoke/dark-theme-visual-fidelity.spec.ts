import { expect, test, seedAuth, SMOKE_LOAD_TIMEOUT } from "./smoke.setup"
import type { Page, Route, TestInfo } from "@playwright/test"
import { waitForAppShell } from "../utils/helpers"

type LightSurfaceLeak = {
  tag: string
  className: string
  testId: string
  role: string
  aria: string
  text: string
  backgroundColor: string
  color: string
  box: [number, number, number, number]
}

type LowContrastTextLeak = LightSurfaceLeak & {
  effectiveBackgroundColor: string
  contrast: number
}

const NOTE_FIXTURE = {
  id: "note-a",
  title: "Dark mode audit note",
  content: "A note used to check menus and editor surfaces.",
  version: 1,
  keywords: [],
  created_at: "2026-07-05T12:00:00Z",
  updated_at: "2026-07-05T12:00:00Z",
}

const CHARACTER_FIXTURE = {
  id: 7,
  name: "Demo Archivist",
  description: "A test character for visual QA.",
  personality: "Concise",
  version: 1,
}

const fulfillJson = async (route: Route, status: number, body: unknown) =>
  route.fulfill({
    status,
    contentType: "application/json",
    headers: {
      "access-control-allow-origin": "*",
      "access-control-allow-headers": "*",
      "access-control-allow-methods": "GET,POST,PUT,PATCH,DELETE,OPTIONS",
    },
    body: JSON.stringify(body),
  })

const emptyChatbookJobsResponse = (url: URL) => {
  const limit = Number(url.searchParams.get("limit") ?? 100)
  const offset = Number(url.searchParams.get("offset") ?? 0)

  return {
    jobs: [],
    total: 0,
    has_more: false,
    next_offset: null,
    pagination: {
      mode: "offset",
      limit,
      offset,
      total: 0,
      has_more: false,
      next_offset: null,
    },
  }
}

const installApiMocks = async (page: Page) => {
  await page.route("**/*", async (route) => {
    const request = route.request()
    const url = new URL(request.url())
    const path = url.pathname
    const method = request.method().toUpperCase()

    if (method === "OPTIONS") {
      await route.fulfill({
        status: 204,
        headers: {
          "access-control-allow-origin": "*",
          "access-control-allow-headers": "*",
          "access-control-allow-methods": "GET,POST,PUT,PATCH,DELETE,OPTIONS",
        },
      })
      return
    }

    if (path === "/api/v1/health" || path === "/api/v1/health/live") {
      await fulfillJson(route, 200, { status: "ok", checks: {} })
      return
    }

    if (path === "/openapi.json") {
      await fulfillJson(route, 200, {
        openapi: "3.0.0",
        info: { version: "visual-fidelity" },
        paths: {
          "/api/v1/chat/completions": {},
          "/api/v1/characters": {},
          "/api/v1/characters/query": {},
          "/api/v1/characters/world-books": {},
          "/api/v1/chatbooks/export": {},
          "/api/v1/chatbooks/export/jobs": {},
          "/api/v1/chatbooks/health": {},
          "/api/v1/chatbooks/import/jobs": {},
          "/api/v1/notes/": {},
        },
      })
      return
    }

    if (path === "/api/v1/config/docs-info") {
      await fulfillJson(route, 200, {
        info: { version: "visual-fidelity" },
        capabilities: { hasAudio: true, hasChatbooks: true, hasStt: true, hasTts: true },
      })
      return
    }

    if (path.startsWith("/api/v1/notifications")) {
      await fulfillJson(
        route,
        200,
        path.endsWith("/unread-count")
          ? { unread_count: 0 }
          : path.endsWith("/preferences")
            ? {
                reminder_enabled: false,
                job_completed_enabled: false,
                job_failed_enabled: false,
              }
            : { items: [], total: 0 }
      )
      return
    }

    if (path === "/api/v1/llm/providers") {
      await fulfillJson(route, 200, {
        providers: [{ id: "mock", name: "Mock Provider", status: "configured" }],
      })
      return
    }

    if (path === "/api/v1/llm/models") {
      await fulfillJson(route, 200, ["mock:model"])
      return
    }

    if (path === "/api/v1/llm/models/metadata") {
      await fulfillJson(route, 200, {
        models: [
          {
            id: "mock:model",
            name: "Mock Model",
            provider: "mock",
            type: "chat",
            output_modality: "text",
            context_length: 8192,
            capabilities: ["chat"],
          },
        ],
        total: 1,
      })
      return
    }

    if (path.startsWith("/api/v1/characters/search")) {
      await fulfillJson(route, 200, [CHARACTER_FIXTURE])
      return
    }

    if (path === "/api/v1/characters" || path === "/api/v1/characters/") {
      await fulfillJson(route, 200, [CHARACTER_FIXTURE])
      return
    }

    if (path.startsWith("/api/v1/chat/conversations")) {
      await fulfillJson(route, 200, method === "GET" ? { items: [] } : { success: true })
      return
    }

    if (path === "/api/v1/chats" || path === "/api/v1/chats/") {
      await fulfillJson(
        route,
        200,
        method === "GET"
          ? { chats: [], total: 0 }
          : {
              id: "audit-thread",
              title: "Audit thread",
              state: "active",
              version: 1,
              created_at: "2026-07-05T12:00:00Z",
              updated_at: "2026-07-05T12:00:00Z",
            }
      )
      return
    }

    if (/\/api\/v1\/chats\/[^/]+$/.test(path)) {
      await fulfillJson(route, 200, {
        id: "audit-thread",
        title: "Audit thread",
        state: "active",
        version: 1,
        created_at: "2026-07-05T12:00:00Z",
        updated_at: "2026-07-05T12:00:00Z",
      })
      return
    }

    if (/\/api\/v1\/chats\/[^/]+\/messages$/.test(path)) {
      await fulfillJson(route, 200, {
        id: "audit-message",
        role: "assistant",
        content: "Mock assistant response for visual QA.",
        created_at: "2026-07-05T12:00:00Z",
        version: 1,
      })
      return
    }

    if (path === "/api/v1/chat/completions") {
      await fulfillJson(route, 200, {
        choices: [
          {
            index: 0,
            finish_reason: "stop",
            message: {
              role: "assistant",
              content: "Mock assistant response for visual QA.",
            },
          },
        ],
      })
      return
    }

    if (path.startsWith("/api/v1/notes/title-settings")) {
      await fulfillJson(route, 200, {
        llm_enabled: false,
        default_strategy: "heuristic",
      })
      return
    }

    if (path.startsWith("/api/v1/notes/search")) {
      await fulfillJson(route, 200, { notes: [NOTE_FIXTURE], total: 1 })
      return
    }

    if (path === "/api/v1/notes/" || path === "/api/v1/notes") {
      await fulfillJson(
        route,
        200,
        method === "GET" ? { notes: [NOTE_FIXTURE], total: 1 } : NOTE_FIXTURE
      )
      return
    }

    if (path === "/api/v1/notes/note-a") {
      await fulfillJson(route, 200, { ...NOTE_FIXTURE, links: [] })
      return
    }

    if (path.startsWith("/api/v1/notes/note-a/neighbors")) {
      await fulfillJson(route, 200, { nodes: [], edges: [] })
      return
    }

    if (path.startsWith("/api/v1/notes/keywords")) {
      await fulfillJson(route, 200, { keywords: [], total: 0 })
      return
    }

    if (path.startsWith("/api/v1/notes/collections")) {
      await fulfillJson(route, 200, { collections: [], total: 0 })
      return
    }

    if (path.startsWith("/api/v1/notes/moodboards")) {
      await fulfillJson(route, 200, { moodboards: [], total: 0 })
      return
    }

    if (path.startsWith("/api/v1/notes/trash")) {
      await fulfillJson(route, 200, { notes: [], total: 0 })
      return
    }

    if (path === "/api/v1/media") {
      await fulfillJson(route, 200, {
        items: [],
        pagination: { total_items: 0 },
      })
      return
    }

    if (path === "/api/v1/prompts") {
      await fulfillJson(route, 200, [])
      return
    }

    if (path === "/api/v1/evaluations" || path === "/api/v1/evaluations/") {
      await fulfillJson(route, 200, { data: [], total: 0 })
      return
    }

    if (path === "/api/v1/characters/world-books") {
      await fulfillJson(route, 200, [])
      return
    }

    if (path === "/api/v1/chat/dictionaries") {
      await fulfillJson(route, 200, [])
      return
    }

    if (path === "/api/v1/chat/documents") {
      await fulfillJson(route, 200, { documents: [], total: 0 })
      return
    }

    if (path === "/api/v1/chatbooks/export/jobs") {
      await fulfillJson(route, 200, emptyChatbookJobsResponse(url))
      return
    }

    if (path === "/api/v1/chatbooks/import/jobs") {
      await fulfillJson(route, 200, emptyChatbookJobsResponse(url))
      return
    }

    if (path === "/api/v1/chatbooks/health") {
      await fulfillJson(route, 200, {
        service: "chatbooks",
        status: "healthy",
        timestamp: "2026-07-05T12:00:00.000Z",
        components: {
          storage_base: {
            path: "/tmp/tldw-chatbooks",
            exists: true,
            writable: true,
          },
        },
      })
      return
    }

    if (path === "/api/v1/chatbooks/cleanup") {
      await fulfillJson(route, 200, { deleted_count: 0 })
      return
    }

    if (path.startsWith("/api/v1/")) {
      await fulfillJson(route, 200, {})
      return
    }

    await route.continue()
  })
}

const forceDarkMode = async (page: Page) => {
  await page.addInitScript(() => {
    localStorage.setItem("theme", "dark")
    localStorage.setItem("tldw:themePreset", "default")
    localStorage.setItem("playgroundComposerOptionsExpanded", "true")
    localStorage.removeItem("__tldwServerCapabilitiesCacheV5")
    sessionStorage.removeItem("__tldwServerCapabilitiesCacheV5")
  })
}

const scanDarkVisualLeaks = async (
  page: Page
): Promise<{
  lightSurfaces: LightSurfaceLeak[]
  lowContrastText: LowContrastTextLeak[]
}> =>
  page.evaluate(() => {
    const parseRgb = (value: string) => {
      const match = value.match(/rgba?\(([^)]+)\)/)
      if (!match) return null
      const parts = match[1].split(",").map((part) => Number.parseFloat(part.trim()))
      if (parts.length < 3) return null
      return { r: parts[0], g: parts[1], b: parts[2], a: parts[3] ?? 1 }
    }

    const luminance = ({ r, g, b }: { r: number; g: number; b: number }) => {
      const channels = [r, g, b].map((value) => {
        const normalized = value / 255
        return normalized <= 0.03928
          ? normalized / 12.92
          : Math.pow((normalized + 0.055) / 1.055, 2.4)
      })
      return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]
    }

    const directText = (element: Element) =>
      Array.from(element.childNodes)
        .filter((node) => node.nodeType === Node.TEXT_NODE)
        .map((node) => node.textContent || "")
        .join("")
        .trim()
        .replace(/\s+/g, " ")

    const effectiveBackground = (element: Element) => {
      let current: Element | null = element
      while (current) {
        const background = parseRgb(getComputedStyle(current).backgroundColor)
        if (background && background.a >= 0.85) {
          return {
            value: getComputedStyle(current).backgroundColor,
            rgb: background,
          }
        }
        current = current.parentElement
      }
      return null
    }

    const isVisibleInViewport = (
      rect: DOMRect,
      style: CSSStyleDeclaration,
      minWidth: number,
      minHeight: number,
      minArea = 0
    ) =>
      rect.width > minWidth &&
      rect.height > minHeight &&
      rect.width * rect.height >= minArea &&
      style.display !== "none" &&
      style.visibility !== "hidden" &&
      Number.parseFloat(style.opacity) >= 0.1 &&
      rect.bottom >= 0 &&
      rect.right >= 0 &&
      rect.left <= window.innerWidth &&
      rect.top <= window.innerHeight

    const lightSurfaces: LightSurfaceLeak[] = []
    const lowContrastText: LowContrastTextLeak[] = []
    for (const element of Array.from(document.querySelectorAll("body *"))) {
      if (element.closest("[hidden], [aria-hidden='true'], .sr-only")) {
        continue
      }

      const rect = element.getBoundingClientRect()
      const style = getComputedStyle(element)

      if (isVisibleInViewport(rect, style, 36, 18, 1_200)) {
        const background = parseRgb(style.backgroundColor)
        if (background && background.a >= 0.85 && luminance(background) >= 0.72) {
          lightSurfaces.push({
            tag: element.tagName.toLowerCase(),
            className:
              typeof element.className === "string" ? element.className.slice(0, 180) : "",
            testId: element.getAttribute("data-testid") || "",
            role: element.getAttribute("role") || "",
            aria: element.getAttribute("aria-label") || "",
            text: (element.textContent || "").trim().replace(/\s+/g, " ").slice(0, 100),
            backgroundColor: style.backgroundColor,
            color: style.color,
            box: [
              Math.round(rect.x),
              Math.round(rect.y),
              Math.round(rect.width),
              Math.round(rect.height),
            ],
          })
        }
      }

      const text = directText(element)
      if (!text || !isVisibleInViewport(rect, style, 8, 8)) {
        continue
      }

      const color = parseRgb(style.color)
      const background = effectiveBackground(element)
      if (!color || color.a < 0.85 || !background) continue

      const foregroundLuminance = luminance(color)
      const backgroundLuminance = luminance(background.rgb)
      if (backgroundLuminance > 0.24) continue

      const contrast =
        (Math.max(foregroundLuminance, backgroundLuminance) + 0.05) /
        (Math.min(foregroundLuminance, backgroundLuminance) + 0.05)

      if (foregroundLuminance < 0.18 && contrast < 3) {
        lowContrastText.push({
          tag: element.tagName.toLowerCase(),
          className:
            typeof element.className === "string" ? element.className.slice(0, 180) : "",
          testId: element.getAttribute("data-testid") || "",
          role: element.getAttribute("role") || "",
          aria: element.getAttribute("aria-label") || "",
          text: text.slice(0, 100),
          backgroundColor: style.backgroundColor,
          effectiveBackgroundColor: background.value,
          color: style.color,
          contrast: Number(contrast.toFixed(2)),
          box: [
            Math.round(rect.x),
            Math.round(rect.y),
            Math.round(rect.width),
            Math.round(rect.height),
          ],
        })
      }
    }
    return { lightSurfaces, lowContrastText }
  })

const expectNoDarkVisualLeaks = async (page: Page, label: string) => {
  const { lightSurfaces, lowContrastText } = await scanDarkVisualLeaks(page)
  expect(lightSurfaces, `${label} has light surfaces in dark mode`).toEqual([])
  expect(lowContrastText, `${label} has low-contrast dark text in dark mode`).toEqual([])
}

const captureDarkScreenshot = async (
  page: Page,
  testInfo: TestInfo,
  name: string
) => {
  const path = testInfo.outputPath(`${name}.png`)
  await page.screenshot({ path, fullPage: false })
  await testInfo.attach(name, { path, contentType: "image/png" })
}

const preparePage = async (page: Page) => {
  await seedAuth(page)
  await forceDarkMode(page)
  await installApiMocks(page)
}

test("chat, character chat, extension sidepanel chat, characters, notes, and chatbooks stay visually dark", async ({
  page,
}, testInfo) => {
  test.setTimeout(150_000)

  await preparePage(page)
  await page.setViewportSize({ width: 1440, height: 960 })

  await page.goto("/chat", { waitUntil: "domcontentloaded", timeout: SMOKE_LOAD_TIMEOUT })
  await waitForAppShell(page, SMOKE_LOAD_TIMEOUT)
  await expect(page.getByTestId("chat-input")).toBeVisible({
    timeout: SMOKE_LOAD_TIMEOUT,
  })
  await captureDarkScreenshot(page, testInfo, "chat-dark")
  await expectNoDarkVisualLeaks(page, "Normal chat")

  const moreToolsButton = page.getByRole("button", { name: "More tools" })
  if ((await moreToolsButton.count()) > 0 && (await moreToolsButton.first().isVisible())) {
    await moreToolsButton.first().click()
    await expectNoDarkVisualLeaks(page, "Normal chat model menu")
    await page.keyboard.press("Escape")
  }

  await page.goto("/chat?mode=character", {
    waitUntil: "domcontentloaded",
    timeout: SMOKE_LOAD_TIMEOUT,
  })
  await waitForAppShell(page, SMOKE_LOAD_TIMEOUT)
  await captureDarkScreenshot(page, testInfo, "character-chat-dark")
  await expectNoDarkVisualLeaks(page, "Character chat")

  await page.goto("/__debug__/sidepanel-chat", {
    waitUntil: "domcontentloaded",
    timeout: SMOKE_LOAD_TIMEOUT,
  })
  await waitForAppShell(page, SMOKE_LOAD_TIMEOUT)
  await captureDarkScreenshot(page, testInfo, "sidepanel-chat-dark")
  await expectNoDarkVisualLeaks(page, "Extension sidepanel chat")

  await page.goto("/characters", { waitUntil: "domcontentloaded", timeout: SMOKE_LOAD_TIMEOUT })
  await waitForAppShell(page, SMOKE_LOAD_TIMEOUT)
  await expect(page.getByTestId("characters-page")).toBeVisible({
    timeout: SMOKE_LOAD_TIMEOUT,
  })
  await captureDarkScreenshot(page, testInfo, "characters-dark")
  await expectNoDarkVisualLeaks(page, "Characters")

  await page.getByRole("button", { name: /filters/i }).click()
  await expect(page.locator("#characters-advanced-filters-panel")).toBeVisible({
    timeout: SMOKE_LOAD_TIMEOUT,
  })
  await captureDarkScreenshot(page, testInfo, "characters-filters-dark")
  await expectNoDarkVisualLeaks(page, "Characters filters")

  await page.getByRole("button", { name: /display/i }).click()
  await expect(page.locator(".ant-dropdown").first()).toBeVisible({
    timeout: SMOKE_LOAD_TIMEOUT,
  })
  await captureDarkScreenshot(page, testInfo, "characters-display-menu-dark")
  await expectNoDarkVisualLeaks(page, "Characters display menu")
  await page.keyboard.press("Escape")
  await expect(page.locator(".ant-dropdown").first()).toBeHidden({
    timeout: SMOKE_LOAD_TIMEOUT,
  })

  await page.getByTestId("characters-new-button").click()
  await expect(page.locator(".ant-drawer")).toBeVisible({
    timeout: SMOKE_LOAD_TIMEOUT,
  })
  await captureDarkScreenshot(page, testInfo, "characters-create-drawer-dark")
  await expectNoDarkVisualLeaks(page, "Characters create drawer")
  await page.keyboard.press("Escape")

  await page.goto("/notes", { waitUntil: "domcontentloaded", timeout: SMOKE_LOAD_TIMEOUT })
  await waitForAppShell(page, SMOKE_LOAD_TIMEOUT)
  await expect(page.getByTestId("notes-list-region")).toBeVisible({
    timeout: SMOKE_LOAD_TIMEOUT,
  })
  await captureDarkScreenshot(page, testInfo, "notes-dark")
  await expectNoDarkVisualLeaks(page, "Notes")

  const overflow = page.getByTestId("notes-overflow-menu-button")
  if ((await overflow.count()) > 0 && (await overflow.first().isVisible())) {
    await overflow.first().click()
    await expectNoDarkVisualLeaks(page, "Notes overflow menu")
  }

  await page.goto("/chatbooks", { waitUntil: "domcontentloaded", timeout: SMOKE_LOAD_TIMEOUT })
  await waitForAppShell(page, SMOKE_LOAD_TIMEOUT)
  await expect(page.getByRole("heading", { name: /chatbooks playground/i })).toBeVisible({
    timeout: SMOKE_LOAD_TIMEOUT,
  })
  await captureDarkScreenshot(page, testInfo, "chatbooks-export-dark")
  await expectNoDarkVisualLeaks(page, "Chatbooks export")

  const mediaQualitySelect = page
    .locator(".ant-select")
    .filter({ has: page.getByRole("combobox", { name: /media quality/i }) })
    .first()
  await mediaQualitySelect.click()
  const mediaQualityDropdown = page.locator(".ant-select-dropdown:visible").first()
  await expect(mediaQualityDropdown).toBeVisible({
    timeout: SMOKE_LOAD_TIMEOUT,
  })
  await captureDarkScreenshot(page, testInfo, "chatbooks-export-media-menu-dark")
  await expectNoDarkVisualLeaks(page, "Chatbooks export media menu")
  await page.keyboard.press("Escape")
  await expect(mediaQualityDropdown).toBeHidden()

  await page.getByText(/^Evaluations$/).first().scrollIntoViewIfNeeded()
  await captureDarkScreenshot(page, testInfo, "chatbooks-export-lower-dark")
  await expectNoDarkVisualLeaks(page, "Chatbooks lower export pickers")
  await page.getByRole("heading", { name: /chatbooks playground/i }).scrollIntoViewIfNeeded()

  await page.getByRole("tab", { name: /^Import$/ }).click()
  await expect(page.getByText(/Preview before import|Drop a \.zip/i).first()).toBeVisible({
    timeout: SMOKE_LOAD_TIMEOUT,
  })
  await captureDarkScreenshot(page, testInfo, "chatbooks-import-dark")
  await expectNoDarkVisualLeaks(page, "Chatbooks import")

  const sourceSelect = page
    .locator(".ant-select")
    .filter({ has: page.getByRole("combobox", { name: /import source/i }) })
    .first()
  await sourceSelect.click()
  const sourceDropdown = page.locator(".ant-select-dropdown:visible").first()
  await expect(sourceDropdown).toBeVisible({
    timeout: SMOKE_LOAD_TIMEOUT,
  })
  await captureDarkScreenshot(page, testInfo, "chatbooks-import-source-menu-dark")
  await expectNoDarkVisualLeaks(page, "Chatbooks import source menu")
  await page.locator(".ant-select-item-option").filter({ hasText: "OpenWebUI JSON" }).first().click()
  await expect(sourceDropdown).toBeHidden()
  await expect(page.getByText(/OpenWebUI attachment hydration/i)).toBeVisible({
    timeout: SMOKE_LOAD_TIMEOUT,
  })
  await captureDarkScreenshot(page, testInfo, "chatbooks-import-openwebui-dark")
  await expectNoDarkVisualLeaks(page, "Chatbooks OpenWebUI import")

  const conflictSelect = page
    .locator(".ant-select")
    .filter({ has: page.getByRole("combobox", { name: /conflict resolution/i }) })
    .first()
  await conflictSelect.click()
  const conflictDropdown = page.locator(".ant-select-dropdown:visible").first()
  await expect(conflictDropdown).toBeVisible({
    timeout: SMOKE_LOAD_TIMEOUT,
  })
  await captureDarkScreenshot(page, testInfo, "chatbooks-import-conflict-menu-dark")
  await expectNoDarkVisualLeaks(page, "Chatbooks import conflict menu")
  await page.keyboard.press("Escape")
  await expect(conflictDropdown).toBeHidden()

  await page.getByRole("tab", { name: /^Jobs$/ }).click()
  await expect(page.getByText(/Job status/i)).toBeVisible({
    timeout: SMOKE_LOAD_TIMEOUT,
  })
  await captureDarkScreenshot(page, testInfo, "chatbooks-jobs-dark")
  await expectNoDarkVisualLeaks(page, "Chatbooks jobs")
})

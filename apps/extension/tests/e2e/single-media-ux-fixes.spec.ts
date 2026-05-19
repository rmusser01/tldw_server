import { type Locator, type Page, expect, test } from "@playwright/test"
import http from "node:http"
import { AddressInfo } from "node:net"

import { forceConnected, waitForConnectionStore } from "./utils/connection"
import { launchWithBuiltExtension } from "./utils/extension-build"

type MockServerOptions = {
  delayMs?: number
  failSearch?: boolean
}

type MockServerHandle = {
  server: http.Server
}

const MEDIA_ITEMS = [
  {
    id: 1,
    title: "First Video",
    snippet: "Video transcript",
    type: "video",
    keywords: ["demo", "video"]
  },
  {
    id: 2,
    title: "Second Document",
    snippet: "PDF content",
    type: "document",
    keywords: ["pdf", "reference"]
  },
  {
    id: 3,
    title: "Third Audio",
    snippet: "Audio transcript",
    type: "audio",
    keywords: ["audio", "podcast"]
  }
]

const MEDIA_DETAILS: Record<number, Record<string, unknown>> = {
  1: {
    id: 1,
    title: "First Video",
    type: "video",
    content: { text: "Full video transcript content here" },
    keywords: ["demo", "video"]
  },
  2: {
    id: 2,
    title: "Second Document",
    type: "document",
    content: { text: "Full document content here" },
    keywords: ["pdf", "reference"]
  },
  3: {
    id: 3,
    title: "Third Audio",
    type: "audio",
    content: { text: "Full audio transcript here" },
    keywords: ["audio", "podcast"]
  }
}

function createMockServer(options: MockServerOptions = {}): MockServerHandle {
  const filterItems = (body: Record<string, unknown> | null) => {
    const query =
      typeof body?.query === "string" ? body.query.trim().toLowerCase() : ""
    const mediaTypes = Array.isArray(body?.media_types)
      ? body.media_types
          .filter((value): value is string => typeof value === "string")
          .map((value) => value.toLowerCase())
      : []

    return MEDIA_ITEMS.filter((item) => {
      const matchesQuery =
        query.length === 0 ||
        item.title.toLowerCase().includes(query) ||
        item.snippet.toLowerCase().includes(query) ||
        item.keywords.some((keyword) => keyword.toLowerCase().includes(query))

      const matchesMediaType =
        mediaTypes.length === 0 || mediaTypes.includes(item.type.toLowerCase())

      return matchesQuery && matchesMediaType
    })
  }

  const server = http.createServer((req, res) => {
    const url = req.url || ""
    const method = (req.method || "GET").toUpperCase()

    const writeJson = (code: number, body: unknown) => {
      res.writeHead(code, {
        "content-type": "application/json",
        "access-control-allow-origin": "*",
        "access-control-allow-credentials": "true"
      })
      res.end(JSON.stringify(body))
    }

    const respond = (code: number, body: unknown) => {
      if (options.delayMs) {
        setTimeout(() => writeJson(code, body), options.delayMs)
      } else {
        writeJson(code, body)
      }
    }

    const readJsonBody = async (): Promise<Record<string, unknown> | null> => {
      const chunks: Buffer[] = []
      for await (const chunk of req) {
        chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk)))
      }

      const raw = Buffer.concat(chunks).toString("utf8").trim()
      if (!raw) return null

      try {
        return JSON.parse(raw) as Record<string, unknown>
      } catch {
        return null
      }
    }

    if (method === "OPTIONS") {
      res.writeHead(204, {
        "access-control-allow-origin": "*",
        "access-control-allow-credentials": "true",
        "access-control-allow-headers": "content-type, x-api-key, authorization"
      })
      res.end()
      return
    }

    if (url === "/api/v1/health" && method === "GET") {
      respond(200, { status: "ok" })
      return
    }

    if (url === "/api/v1/llm/models" && method === "GET") {
      respond(200, ["mock/model"])
      return
    }

    if (/^\/api\/v1\/media\/\d+$/.test(url) && method === "GET") {
      const id = Number(url.split("/").pop())
      const detail = MEDIA_DETAILS[id]
      respond(detail ? 200 : 404, detail ?? { detail: "not found" })
      return
    }

    if (url.includes("/api/v1/media/search") && method === "POST") {
      void (async () => {
        const body = await readJsonBody()

        if (options.failSearch) {
          respond(500, { detail: "Search failed" })
          return
        }

        const items = filterItems(body)
        respond(200, {
          items,
          pagination: {
            total_items: items.length,
            total_pages: 1
          }
        })
      })()
      return
    }

    if (url.startsWith("/api/v1/media") && method === "GET") {
      respond(200, {
        items: MEDIA_ITEMS,
        pagination: {
          total_items: MEDIA_ITEMS.length,
          total_pages: 1
        }
      })
      return
    }

    if (url === "/openapi.json" && method === "GET") {
      respond(200, {
        openapi: "3.0.0",
        paths: {
          "/api/v1/media/": {},
          "/api/v1/media/search": {},
          "/api/v1/health": {},
          "/api/v1/llm/models": {}
        }
      })
      return
    }

    respond(404, { detail: "not found" })
  })

  return { server }
}

async function closeMockServer(server: http.Server) {
  await new Promise<void>((resolve) => server.close(() => resolve()))
}

async function setupConnectedMediaPage(
  handle: MockServerHandle,
  options: { waitForResults?: boolean } = {}
) {
  await new Promise<void>((resolve) =>
    handle.server.listen(0, "127.0.0.1", resolve)
  )
  const address = handle.server.address() as AddressInfo
  const baseUrl = `http://127.0.0.1:${address.port}`

  const { context, page, optionsUrl } = await launchWithBuiltExtension({
    allowOffline: true,
    seedConfig: {
      serverUrl: baseUrl,
      authMode: "single-user",
      apiKey: "test"
    }
  })

  await page.goto(optionsUrl, { waitUntil: "domcontentloaded" })
  await waitForConnectionStore(page, "single-media-ux-fixes:init")
  await forceConnected(
    page,
    { serverUrl: baseUrl },
    "single-media-ux-fixes:connected"
  )

  await page.goto(`${optionsUrl}#/media`, { waitUntil: "domcontentloaded" })
  await expect(page.getByTestId("media-search-input")).toBeVisible()
  if (options.waitForResults !== false) {
    await expect
      .poll(async () => getResultRows(page).count(), {
        timeout: 10_000,
        message: "Expected seeded media results to render in the sidebar"
      })
      .toBe(MEDIA_ITEMS.length)
  }

  return { context, page, baseUrl }
}

const getResultRows = (page: Page): Locator =>
  page.locator(
    "[data-testid='media-results-list'] [role='button'][aria-selected]"
  )

const getResultRowByTitle = (page: Page, title: string): Locator =>
  getResultRows(page).filter({ hasText: title }).first()

test.describe("Single Media Page UX", () => {
  test("shows empty-state guidance before a user selects media", async () => {
    const handle = createMockServer()
    const { context, page } = await setupConnectedMediaPage(handle)

    try {
      await expect(
        page.getByRole("heading", { name: /No media item selected/i })
      ).toBeVisible()
      await expect(
        page.getByText(/Tip: Use j\/k to navigate items/i)
      ).toBeVisible()
      await expect(
        page.getByRole("button", { name: /Open Quick Ingest/i })
      ).toBeVisible()
    } finally {
      await context.close()
      await closeMockServer(handle.server)
    }
  })

  test("selecting media reveals chat actions and marks the active row", async () => {
    const handle = createMockServer()
    const { context, page } = await setupConnectedMediaPage(handle)

    try {
      const firstRow = getResultRowByTitle(page, "First Video")
      await firstRow.click()

      await expect(firstRow).toHaveAttribute("aria-selected", "true")
      await expect(
        page.getByRole("button", { name: /Chat with this media/i })
      ).toBeVisible()
      await expect(page.getByRole("button", { name: /Actions/i })).toBeVisible()
      await expect(
        page.getByRole("heading", { name: /First Video/i }).last()
      ).toBeVisible()
    } finally {
      await context.close()
      await closeMockServer(handle.server)
    }
  })

  test("keyboard shortcuts keep result selection and preview state in sync", async () => {
    const handle = createMockServer()
    const { context, page } = await setupConnectedMediaPage(handle)

    try {
      const firstRow = getResultRowByTitle(page, "First Video")
      const secondRow = getResultRowByTitle(page, "Second Document")

      await firstRow.click()
      await expect(firstRow).toHaveAttribute("aria-selected", "true")
      await expect(
        page.getByRole("heading", { name: /First Video/i }).last()
      ).toBeVisible()

      await page.keyboard.press("j")

      await expect(secondRow).toHaveAttribute("aria-selected", "true")
      await expect(firstRow).toHaveAttribute("aria-selected", "false")
      await expect(
        page.getByRole("heading", { name: /Second Document/i }).last()
      ).toBeVisible()

      await page.keyboard.press("k")

      await expect(firstRow).toHaveAttribute("aria-selected", "true")
      await expect(secondRow).toHaveAttribute("aria-selected", "false")
      await expect(
        page.getByRole("heading", { name: /First Video/i }).last()
      ).toBeVisible()
    } finally {
      await context.close()
      await closeMockServer(handle.server)
    }
  })

  test("metadata mode updates the filter clear-all affordance", async () => {
    const handle = createMockServer()
    const { context, page } = await setupConnectedMediaPage(handle)

    try {
      const metadataModeButton = page.getByRole("button", {
        name: /metadata search/i
      })
      const clearAll = page.getByRole("button", { name: /^Clear all/ })

      await expect(clearAll).toHaveText("Clear all")

      await metadataModeButton.click()

      await expect(metadataModeButton).toHaveAttribute("aria-pressed", "true")
      await expect(clearAll).toHaveText(/Clear all \(1\)/)

      await clearAll.click()

      await expect(metadataModeButton).toHaveAttribute("aria-pressed", "false")
      await expect(clearAll).toHaveText("Clear all")
    } finally {
      await context.close()
      await closeMockServer(handle.server)
    }
  })

  test("slash shortcut focuses the search field", async () => {
    const handle = createMockServer()
    const { context, page } = await setupConnectedMediaPage(handle)

    try {
      const searchInput = page.getByTestId("media-search-input")

      await page.locator("body").click()
      await page.keyboard.press("/")

      await expect(searchInput).toBeFocused()
    } finally {
      await context.close()
      await closeMockServer(handle.server)
    }
  })

  test("search input exposes a clear affordance and restores the default list when cleared", async () => {
    const handle = createMockServer()
    const { context, page } = await setupConnectedMediaPage(handle)

    try {
      const searchInput = page.getByTestId("media-search-input")
      await searchInput.fill("audio")

      const clearSearch = page.getByTestId("media-search-clear")
      await expect(clearSearch).toBeVisible()

      await clearSearch.click()

      await expect(searchInput).toHaveValue("")
      await expect(clearSearch).toHaveCount(0)
      await expect
        .poll(async () => getResultRows(page).count(), {
          timeout: 10_000,
          message:
            "Expected clearing the query to restore the default result list"
        })
        .toBe(MEDIA_ITEMS.length)
    } finally {
      await context.close()
      await closeMockServer(handle.server)
    }
  })
})

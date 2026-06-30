import { test, expect, type Page, type Route } from "@playwright/test"

import { seedAuth, TEST_CONFIG } from "../utils/helpers"

type KnowledgeRouteState = "ready" | "results" | "noResults"

const fulfillJson = async (route: Route, payload: unknown, status = 200) => {
  await route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(payload),
  })
}

async function mockKnowledgeQaApi(page: Page, state: KnowledgeRouteState = "ready") {
  await page.route(/\/api\/v1\/health(?:\?.*)?$/, async (route) => {
    await fulfillJson(route, { status: "healthy" })
  })

  await page.route(/\/openapi\.json(?:\?.*)?$/, async (route) => {
    await fulfillJson(route, {
      openapi: "3.0.0",
      paths: {
        "/api/v1/rag/search": { post: {} },
        "/api/v1/rag/health": { get: {} },
        "/api/v1/research/websearch": { post: {} },
      },
    })
  })

  await page.route(/\/api\/v1\/llm\/providers(?:\?.*)?$/, async (route) => {
    await fulfillJson(route, {
      providers: [
        {
          name: "server-default",
          models: ["server-default-model"],
        },
      ],
    })
  })

  await page.route(/\/api\/v1\/characters\/search(?:\?.*)?$/, async (route) => {
    await fulfillJson(route, [
      {
        id: 1,
        name: "Helpful AI Assistant",
      },
    ])
  })

  await page.route(/\/api\/v1\/characters(?:\/)?(?:\?.*)?$/, async (route) => {
    await fulfillJson(route, [
      {
        id: 1,
        name: "Helpful AI Assistant",
      },
    ])
  })

  await page.route(/\/api\/v1\/chats(?:\/)?(?:\?.*)?$/, async (route) => {
    if (route.request().method() === "POST") {
      await fulfillJson(route, {
        id: "knowledge-chat-1",
        title: "Grounded QA fixture",
        state: "in-progress",
        source: "knowledge_qa",
        version: 1,
      })
      return
    }

    await fulfillJson(route, {
      chats: [],
      total: 0,
    })
  })

  await page.route(/\/api\/v1\/chats\/knowledge-chat-1\/messages(?:\?.*)?$/, async (route) => {
    let payload: Record<string, unknown> = {}
    try {
      payload = route.request().postDataJSON() as Record<string, unknown>
    } catch {
      payload = {}
    }
    const role = typeof payload?.role === "string" ? payload.role : "user"
    await fulfillJson(route, {
      id: `knowledge-message-${role}`,
      role,
      content: typeof payload?.content === "string" ? payload.content : "",
      created_at: "2026-06-07T00:00:00.000Z",
    })
  })

  await page.route(/\/api\/v1\/chat\/conversations(?:\?.*)?$/, async (route) => {
    await fulfillJson(route, {
      conversations: [],
      total: 0,
    })
  })

  await page.route(/\/api\/v1\/chat\/conversations\/knowledge-chat-1(?:\?.*)?$/, async (route) => {
    await fulfillJson(route, {
      id: "knowledge-chat-1",
      title: "Grounded QA fixture",
      state: "in-progress",
      source: "knowledge_qa",
      keywords: ["knowledge_qa"],
      version: 1,
    })
  })

  await page.route(/\/api\/v1\/chat\/messages\/[^/]+\/rag-context(?:\?.*)?$/, async (route) => {
    await fulfillJson(route, {
      success: true,
    })
  })

  await page.route(/\/api\/v1\/media(?:\/)?(?:\?.*)?$/, async (route) => {
    await fulfillJson(route, {
      items: [
        {
          id: 101,
          title: "Grounded QA Notes",
          media_type: "document",
        },
      ],
      total: 1,
    })
  })

  await page.route(/\/api\/v1\/notes(?:\/)?(?:\?.*)?$/, async (route) => {
    await fulfillJson(route, {
      items: [
        {
          id: "note-grounded-qa",
          title: "Grounded QA checklist",
        },
      ],
      total: 1,
    })
  })

  await page.route(/\/api\/v1\/rag\/search\/stream(?:\?.*)?$/, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "text/plain",
      body: "",
    })
  })

  await page.route(/\/api\/v1\/rag\/search(?:\?.*)?$/, async (route) => {
    if (state === "noResults") {
      await fulfillJson(route, {
        results: [],
        generated_answer: null,
      })
      return
    }

    await fulfillJson(route, {
      results: [
        {
          id: "knowledge-source-1",
          content: "Grounded answers should cite visible evidence.",
          excerpt: "Grounded answers should cite visible evidence.",
          score: 0.93,
          metadata: {
            title: "Grounded QA Notes",
            source_type: "media_db",
          },
        },
      ],
      generated_answer: "Grounded answers should cite visible evidence [1].",
      metadata: {
        retrieval_metrics: {
          documents_considered: 1,
          chunks_considered: 1,
        },
      },
    })
  })
}

test.describe("Knowledge QA deterministic route states", () => {
  test("WebUI renders ready search state without a live backend", async ({ page }) => {
    await mockKnowledgeQaApi(page, "ready")
    await seedAuth(page, {
      serverUrl: TEST_CONFIG.serverUrl,
      allowOffline: true,
    })

    await page.goto("/knowledge", { waitUntil: "domcontentloaded" })

    await expect(page.getByRole("heading", { name: /Ask Your Library/i })).toBeVisible()
    await expect(page.getByLabel(/Search your knowledge base/i)).toBeVisible()
  })

  test("WebUI can exercise a cited result state with mocked RAG search", async ({ page }) => {
    await mockKnowledgeQaApi(page, "results")
    await seedAuth(page, {
      serverUrl: TEST_CONFIG.serverUrl,
      allowOffline: true,
    })

    await page.goto("/knowledge", { waitUntil: "domcontentloaded" })

    const searchBox = page.getByLabel(/Search your knowledge base/i)
    await expect(searchBox).toBeVisible()
    await searchBox.fill("What does my library say about grounded answers?")
    await page.getByRole("button", { name: /^Ask$/i }).click()

    await expect(page.getByText(/Grounded answers should cite visible evidence/i)).toBeVisible()
    await expect(page.getByRole("button", { name: /Export/i })).toBeVisible()
  })
})

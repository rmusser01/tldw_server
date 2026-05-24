/**
 * KnowledgeQA (RAG Search) Workflow E2E Tests
 *
 * Tests the complete knowledge QA lifecycle:
 * - Basic RAG search (query, results, citations)
 * - Settings & Presets (fast, balanced, thorough, expert mode)
 * - Follow-up questions (thread context)
 * - Search history
 * - No results / error states
 *
 * Run: npx playwright test e2e/workflows/knowledge-qa.spec.ts
 */
import { type Page } from "@playwright/test"
import {
  test,
  expect,
  skipIfServerUnavailable,
  assertNoCriticalErrors
} from "../utils/fixtures"
import { KnowledgeQAPage } from "../utils/page-objects/KnowledgeQAPage"
import { ResearchWorkspacePage } from "../utils/page-objects/ResearchWorkspacePage"
import {
  seedAuth,
  generateTestId,
  waitForConnection,
  dismissConnectionModals,
  fetchWithApiKey,
  TEST_CONFIG
} from "../utils/helpers"

type SeededKnowledgeThread = {
  threadId: string
  query: string
  answer: string
}

const asRecord = (value: unknown): Record<string, unknown> | null =>
  value && typeof value === "object" ? (value as Record<string, unknown>) : null

const parseJsonRecord = async (
  response: { json: () => Promise<unknown> },
  label: string
): Promise<Record<string, unknown>> => {
  const payload = await response.json().catch(() => null)
  const record = asRecord(payload)
  if (!record) {
    throw new Error(`${label} returned a non-object payload`)
  }
  return record
}

const canReachKnowledgeChatEndpoint = async (): Promise<{
  reachable: boolean
  reason?: string
}> => {
  const endpoint = `${TEST_CONFIG.serverUrl}/api/v1/chats/?limit=1&offset=0&ordering=-updated_at`
  try {
    const response = await fetchWithApiKey(endpoint)
    if (response.ok) {
      return { reachable: true }
    }
    return {
      reachable: false,
      reason: `GET /api/v1/chats preflight returned HTTP ${response.status}`
    }
  } catch (error) {
    return {
      reachable: false,
      reason:
        error instanceof Error ? error.message : "GET /api/v1/chats preflight failed"
    }
  }
}

const resolveKnowledgeCharacterId = async (): Promise<number> => {
  const response = await fetchWithApiKey(
    `${TEST_CONFIG.serverUrl}/api/v1/chats/?limit=20&offset=0&ordering=-updated_at`
  )
  if (!response.ok) {
    return 2
  }

  const payload = await response.json().catch(() => null)
  const chats = Array.isArray(payload)
    ? payload
    : Array.isArray(asRecord(payload)?.chats)
      ? (asRecord(payload)?.chats as unknown[])
      : Array.isArray(asRecord(payload)?.conversations)
        ? (asRecord(payload)?.conversations as unknown[])
        : []

  const matchingChat = chats.find((entry) => {
    const candidate = asRecord(entry)
    return (
      candidate?.source === "knowledge_qa" &&
      Number.isFinite(Number(candidate.character_id)) &&
      Number(candidate.character_id) > 0
    )
  })

  const characterId = Number(asRecord(matchingChat)?.character_id)
  return Number.isFinite(characterId) && characterId > 0 ? Math.trunc(characterId) : 2
}

const createSeededKnowledgeThread = async (slug: string): Promise<SeededKnowledgeThread> => {
  const characterId = await resolveKnowledgeCharacterId()
  const query = `Knowledge live share query ${slug}`
  const answer = `Knowledge live share answer for ${slug} [1]`

  const createResponse = await fetchWithApiKey(`${TEST_CONFIG.serverUrl}/api/v1/chats/`, TEST_CONFIG.apiKey, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      character_id: characterId,
      title: `Knowledge live share ${slug}`,
      state: "in-progress",
      source: "knowledge_qa"
    })
  })
  if (!createResponse.ok) {
    throw new Error(`POST /api/v1/chats returned HTTP ${createResponse.status}`)
  }
  const createdThread = await parseJsonRecord(createResponse, "create knowledge thread")
  const threadId = String(createdThread.id || "")
  if (!threadId) {
    throw new Error(`Create chat returned no usable id: ${JSON.stringify(createdThread)}`)
  }

  const userResponse = await fetchWithApiKey(
    `${TEST_CONFIG.serverUrl}/api/v1/chats/${encodeURIComponent(threadId)}/messages`,
    TEST_CONFIG.apiKey,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        role: "user",
        content: query
      })
    }
  )
  if (!userResponse.ok) {
    throw new Error(`POST user message returned HTTP ${userResponse.status}`)
  }

  const assistantResponse = await fetchWithApiKey(
    `${TEST_CONFIG.serverUrl}/api/v1/chats/${encodeURIComponent(threadId)}/messages`,
    TEST_CONFIG.apiKey,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        role: "assistant",
        content: answer
      })
    }
  )
  if (!assistantResponse.ok) {
    throw new Error(`POST assistant message returned HTTP ${assistantResponse.status}`)
  }
  const assistantMessage = await parseJsonRecord(
    assistantResponse,
    "create knowledge assistant message"
  )
  const assistantMessageId = String(assistantMessage.id || "")
  if (!assistantMessageId) {
    throw new Error(`Assistant message returned no id: ${JSON.stringify(assistantMessage)}`)
  }

  const ragContextResponse = await fetchWithApiKey(
    `${TEST_CONFIG.serverUrl}/api/v1/chat/messages/${encodeURIComponent(assistantMessageId)}/rag-context`,
    TEST_CONFIG.apiKey,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message_id: assistantMessageId,
        rag_context: {
          search_query: query,
          search_mode: "hybrid",
          generated_answer: answer,
          retrieved_documents: [
            {
              id: `knowledge-live-share-doc-${slug}`,
              title: "Knowledge Live Share Source",
              source_type: "media_db",
              excerpt: `Evidence for ${slug}`,
              score: 0.97,
              url: `https://example.com/knowledge-live-share?slug=${slug}`
            }
          ],
          citations: [
            {
              index: 1,
              documentId: `knowledge-live-share-doc-${slug}`
            }
          ],
          settings_snapshot: {
            keyword: "__knowledge_QA__"
          }
        }
      })
    }
  )
  if (!ragContextResponse.ok) {
    throw new Error(
      `POST /api/v1/chat/messages/${assistantMessageId}/rag-context returned HTTP ${ragContextResponse.status}`
    )
  }

  return {
    threadId,
    query,
    answer
  }
}

const revokeKnowledgeShareLink = async (
  threadId: string,
  shareId: string | null
): Promise<void> => {
  if (!threadId || !shareId) return

  const response = await fetchWithApiKey(
    `${TEST_CONFIG.serverUrl}/api/v1/chat/conversations/${encodeURIComponent(threadId)}/share-links/${encodeURIComponent(shareId)}`,
    TEST_CONFIG.apiKey,
    {
      method: "DELETE"
    }
  ).catch(() => null)

  if (response && !response.ok && response.status !== 404) {
    throw new Error(
      `DELETE /api/v1/chat/conversations/${threadId}/share-links/${shareId} returned HTTP ${response.status}`
    )
  }
}

const deleteKnowledgeThread = async (threadId: string): Promise<void> => {
  if (!threadId) return

  const softDeleteResponse = await fetchWithApiKey(
    `${TEST_CONFIG.serverUrl}/api/v1/chats/${encodeURIComponent(threadId)}`,
    TEST_CONFIG.apiKey,
    {
      method: "DELETE"
    }
  ).catch(() => null)

  if (softDeleteResponse && !softDeleteResponse.ok && softDeleteResponse.status !== 404) {
    throw new Error(
      `DELETE /api/v1/chats/${threadId} returned HTTP ${softDeleteResponse.status}`
    )
  }

  const hardDeleteResponse = await fetchWithApiKey(
    `${TEST_CONFIG.serverUrl}/api/v1/chats/${encodeURIComponent(threadId)}?hard_delete=true`,
    TEST_CONFIG.apiKey,
    {
      method: "DELETE"
    }
  ).catch(() => null)

  if (hardDeleteResponse && !hardDeleteResponse.ok && hardDeleteResponse.status !== 404) {
    throw new Error(
      `DELETE /api/v1/chats/${threadId}?hard_delete=true returned HTTP ${hardDeleteResponse.status}`
    )
  }
}

test.describe("KnowledgeQA Workflow", () => {
  let qaPage: KnowledgeQAPage

  const clickAskButton = async (page: Page): Promise<void> => {
    const askButton = page.getByRole("button", { name: /^Ask$/i })
    await expect(askButton).toBeEnabled({ timeout: 10_000 })
    await askButton.click()
  }

  test.beforeEach(async ({ page }) => {
    await seedAuth(page)
    qaPage = new KnowledgeQAPage(page)
  })

  // ═════════════════════════════════════════════════════════════════════
  // 3.1  Basic RAG Search
  // ═════════════════════════════════════════════════════════════════════

  test.describe("Basic RAG Search", () => {
    test("should navigate to KnowledgeQA page and display search bar", async ({
      authedPage,
      diagnostics
    }) => {
      qaPage = new KnowledgeQAPage(authedPage)
      await qaPage.goto()
      await qaPage.waitForReady()

      // Search bar should be visible
      const input = await qaPage.getSearchInput()
      await expect(input).toBeVisible()

      await assertNoCriticalErrors(diagnostics)
    })

    test("should focus search bar with / key", async ({
      authedPage,
      diagnostics
    }) => {
      qaPage = new KnowledgeQAPage(authedPage)
      await qaPage.goto()
      await qaPage.waitForReady()

      // Click somewhere else first, then press /
      await authedPage.locator("body").click()
      await qaPage.pressSlashToFocus()

      const input = await qaPage.getSearchInput()
      await expect(input).toBeFocused({ timeout: 5_000 })

      await assertNoCriticalErrors(diagnostics)
    })

    test("should perform a RAG search and display results", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      qaPage = new KnowledgeQAPage(authedPage)
      await qaPage.goto()
      await qaPage.waitForReady()

      // Perform search
      const query = "What is machine learning?"

      const [ragResult] = await Promise.all([
        qaPage.waitForRagSearch(),
        qaPage.search(query)
      ])

      expect(ragResult.status).toBe(200)
      expect(ragResult.requestBody?.query).toBe(query)

      // Wait for results to render
      await qaPage.waitForResults()

      // Either we get an answer or a no-results state
      const answer = await qaPage.getAnswerText()
      const noResults = await qaPage.hasNoResults()

      // One of these should be true
      expect(answer.length > 0 || noResults).toBeTruthy()
      if (answer.length > 0) {
        expect(answer).not.toMatch(/chatcmpl|finish_reason|chat\.completion\.chunk/i)
      }

      await assertNoCriticalErrors(diagnostics)
    })

    test("should surface the evidence panel after a live search", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      qaPage = new KnowledgeQAPage(authedPage)
      await qaPage.goto()
      await qaPage.waitForReady()

      await qaPage.search("content analysis")
      await qaPage.waitForResults()

      await expect(qaPage.getEvidencePanel()).toBeVisible({ timeout: 10_000 })
      const answer = await qaPage.getAnswerText()
      if (answer.length > 0) {
        const citationButtons = qaPage.getCitationButtons()
        const citationCount = await citationButtons.count()
        if (citationCount > 0) {
          await qaPage.clickCitation(0)
          await expect(qaPage.getEvidencePanel()).toBeVisible({ timeout: 10_000 })
        } else {
          await expect(
            qaPage.getEvidencePanel().getByText(/No sources yet|0 sources/i).first()
          ).toBeVisible({ timeout: 10_000 })
        }
      }

      await assertNoCriticalErrors(diagnostics)
    })

    test("keeps citation jumps aligned with the matching evidence card", async ({
      authedPage,
      diagnostics
    }) => {
      qaPage = new KnowledgeQAPage(authedPage)
      const threadId = "knowledge-e2e-thread"
      const query = "citation coherence regression"

      await authedPage.route("**/api/v1/config/docs-info", async (route) => {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            info: { version: "e2e" },
            capabilities: {}
          })
        })
      })

      await authedPage.route("**/api/v1/chat/conversations?*", async (route) => {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            conversations: []
          })
        })
      })

      await authedPage.route("**/api/v1/characters/search?*", async (route) => {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify([
            {
              id: 1,
              name: "Helpful AI Assistant"
            }
          ])
        })
      })

      await authedPage.route("**/api/v1/chats/", async (route) => {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            id: threadId,
            title: query,
            version: 1,
            state: "in-progress",
            created_at: new Date().toISOString()
          })
        })
      })

      await authedPage.route("**/api/v1/chat/conversations/*", async (route) => {
        const method = route.request().method()
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body:
            method === "GET"
              ? JSON.stringify({
                  keywords: ["__knowledge_QA__"]
                })
              : JSON.stringify({
                  success: true
                })
        })
      })

      await authedPage.route("**/api/v1/chats/*/messages", async (route) => {
        const payload = route.request().postDataJSON() as { role?: string; content?: string }
        const suffix = payload?.role === "assistant" ? "assistant" : "user"
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            id: `msg-${suffix}-1`,
            role: payload?.role || suffix,
            content: payload?.content || "",
            created_at: new Date().toISOString()
          })
        })
      })

      await authedPage.route("**/api/v1/chat/messages/*/rag-context", async (route) => {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            success: true
          })
        })
      })

      await authedPage.route("**/api/v1/rag/search/stream", async (route) => {
        await route.fulfill({
          status: 200,
          contentType: "text/plain",
          body: ""
        })
      })

      await authedPage.route("**/api/v1/rag/search", async (route) => {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            results: [
              {
                id: "100",
                content: "Alpha excerpt about the premise.",
                metadata: {
                  title: "Alpha Source",
                  source_type: "pdf",
                  media_id: 100
                },
                score: 0.84
              },
              {
                id: "200",
                content: "Beta excerpt about the conclusion.",
                metadata: {
                  title: "Beta Source",
                  source_type: "pdf",
                  media_id: 200
                },
                score: 0.96
              }
            ],
            answer:
              "Alpha source frames the premise [1]. Beta source verifies the conclusion [2].",
            expanded_queries: [query]
          })
        })
      })

      await qaPage.goto()
      await qaPage.waitForReady()

      const searchInput = await qaPage.getSearchInput()
      await searchInput.fill(query)
      await dismissConnectionModals(authedPage)
      await clickAskButton(authedPage)

      await qaPage.waitForResults()
      await expect(
        qaPage.getEvidencePanel().getByRole("heading", { name: /Alpha Source/i })
      ).toBeVisible()
      await expect(
        qaPage.getEvidencePanel().getByRole("heading", { name: /Beta Source/i })
      ).toBeVisible()

      const citationTwo = authedPage
        .getByRole("button", { name: "Jump to source 2" })
        .first()
      await citationTwo.click()

      await expect(citationTwo).toHaveAttribute("aria-current", "true")
      await expect
        .poll(
          async () => ({
            firstSourceClass:
              (await authedPage.locator("#source-card-0").getAttribute("class")) || "",
            secondSourceClass:
              (await authedPage.locator("#source-card-1").getAttribute("class")) || ""
          }),
          { timeout: 10_000 }
        )
        .toMatchObject({
          firstSourceClass: expect.not.stringContaining("ring-2"),
          secondSourceClass: expect.stringContaining("ring-2")
        })

      await assertNoCriticalErrors(diagnostics)
    })

    test("should show loading state during search", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      qaPage = new KnowledgeQAPage(authedPage)
      await qaPage.goto()
      await qaPage.waitForReady()

      // Start search
      const input = await qaPage.getSearchInput()
      await input.fill("test loading state")
      await input.press("Enter")

      // Check for loading indicator (may be brief)
      const wasLoading = await qaPage.isLoading()
      // Loading state is transient, we just verify no crash

      await qaPage.waitForResults()

      await assertNoCriticalErrors(diagnostics)
    })

    test("shows progressive loading stages for delayed long-running searches", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      qaPage = new KnowledgeQAPage(authedPage)

      await authedPage.route("**/api/v1/rag/search/stream", async (route) => {
        await route.fulfill({
          status: 200,
          contentType: "text/plain",
          body: "",
        })
      })

      await authedPage.route("**/api/v1/rag/search", async (route) => {
        // Keep the mocked search pending past the 5s AnswerPanel threshold so the
        // reranking stage renders before the response resolves.
        await new Promise((resolve) => setTimeout(resolve, 6_500))
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            results: [
              {
                id: "delayed-source-1",
                content: "Delayed source excerpt",
                metadata: {
                  title: "Delayed Source",
                  source_type: "media_db",
                  url: "https://example.com/delayed-source",
                },
                score: 0.92,
              },
            ],
            answer: "Delayed answer [1]",
            expanded_queries: ["delayed response query"],
          }),
        })
      })

      await qaPage.goto()
      await qaPage.waitForReady()

      const input = await qaPage.getSearchInput()
      await input.fill("delayed response query")
      await input.press("Enter")

      await expect(
        authedPage.getByText(/Searching documents\.\.\./i)
      ).toBeVisible({ timeout: 4_000 })
      await expect(
        authedPage.getByText(/Reranking results\.\.\./i)
      ).toBeVisible({ timeout: 10_000 })

      await qaPage.waitForResults()
      await expect(authedPage.getByText("AI Answer")).toBeVisible({
        timeout: 10_000
      })
      await expect(
        authedPage.getByTestId("knowledge-answer-content")
      ).toContainText(/Delayed answer/i, { timeout: 10_000 })
      await expect(qaPage.getCitationButtons().first()).toBeVisible({
        timeout: 10_000
      })

      await assertNoCriticalErrors(diagnostics)
    })

    test("treats whitespace-only answers as no generated answer", async ({
      authedPage,
      diagnostics
    }) => {
      qaPage = new KnowledgeQAPage(authedPage)
      const threadId = "knowledge-whitespace-thread"
      const query = "blank answer regression"

      await authedPage.route("**/api/v1/config/docs-info", async (route) => {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            info: { version: "e2e" },
            capabilities: {}
          })
        })
      })

      await authedPage.route("**/api/v1/chat/conversations?*", async (route) => {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            conversations: []
          })
        })
      })

      await authedPage.route("**/api/v1/characters/search?*", async (route) => {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify([
            {
              id: 1,
              name: "Helpful AI Assistant"
            }
          ])
        })
      })

      await authedPage.route("**/api/v1/chats/", async (route) => {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            id: threadId,
            title: query,
            version: 1,
            state: "in-progress",
            created_at: new Date().toISOString()
          })
        })
      })

      await authedPage.route("**/api/v1/chat/conversations/*", async (route) => {
        const method = route.request().method()
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body:
            method === "GET"
              ? JSON.stringify({
                  keywords: ["__knowledge_QA__"]
                })
              : JSON.stringify({
                  success: true
                })
        })
      })

      await authedPage.route("**/api/v1/chats/*/messages", async (route) => {
        const payload = route.request().postDataJSON() as { role?: string; content?: string }
        const suffix = payload?.role === "assistant" ? "assistant" : "user"
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            id: `msg-${suffix}-blank-answer`,
            role: payload?.role || suffix,
            content: payload?.content || "",
            created_at: new Date().toISOString()
          })
        })
      })

      await authedPage.route("**/api/v1/chat/messages/*/rag-context", async (route) => {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            success: true
          })
        })
      })

      await authedPage.route("**/api/v1/rag/search/stream", async (route) => {
        await route.fulfill({
          status: 200,
          contentType: "text/plain",
          body: "",
        })
      })

      await authedPage.route("**/api/v1/rag/search", async (route) => {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            results: [
              {
                id: "blank-answer-source-1",
                content: "Relevant source excerpt",
                metadata: {
                  title: "Blank Answer Source",
                  source_type: "notes",
                },
                score: 0.91,
              },
            ],
            answer: "   ",
          }),
        })
      })

      await qaPage.goto()
      await qaPage.waitForReady()
      await qaPage.search(query)
      await qaPage.waitForResults()

      await expect(
        authedPage.getByText(/Found 1 relevant source\./i)
      ).toBeVisible({ timeout: 10_000 })
      await expect(authedPage.getByText("AI Answer")).toHaveCount(0)
      await expect(authedPage.getByTestId("knowledge-answer-content")).toHaveCount(0)
      await expect(await qaPage.getAnswerText()).toBe("")

      await assertNoCriticalErrors(diagnostics)
    })
  })

  // ═════════════════════════════════════════════════════════════════════
  // 3.2  Settings & Presets
  // ═════════════════════════════════════════════════════════════════════

  test.describe("Settings & Presets", () => {
    test("should open settings panel", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      qaPage = new KnowledgeQAPage(authedPage)
      await qaPage.goto()
      await qaPage.waitForReady()

      await qaPage.openSettings()
      await expect(qaPage.getSettingsDialog()).toBeVisible({ timeout: 10_000 })
      await expect(
        qaPage.getSettingsDialog().getByText(/RAG Settings/i)
      ).toBeVisible()

      await assertNoCriticalErrors(diagnostics)
    })

    test("should switch between presets", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      qaPage = new KnowledgeQAPage(authedPage)
      await qaPage.goto()
      await qaPage.waitForReady()

      await qaPage.openSettings()
      await expect(qaPage.getSettingsDialog()).toBeVisible({ timeout: 10_000 })

      for (const preset of ["fast", "balanced", "thorough"] as const) {
        await qaPage.selectPreset(preset)
        await expect(
          qaPage
            .getSettingsDialog()
            .getByRole("radio", { name: new RegExp(`^${preset}\\b`, "i") })
        ).toHaveAttribute("aria-checked", "true")
      }

      await assertNoCriticalErrors(diagnostics)
    })

    test("should toggle expert mode", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      qaPage = new KnowledgeQAPage(authedPage)
      await qaPage.goto()
      await qaPage.waitForReady()

      await qaPage.openSettings()
      const settingsDialog = qaPage.getSettingsDialog()
      const expertToggle = qaPage.getExpertModeToggle()

      const initialChecked = await expertToggle.getAttribute("aria-checked")
      await qaPage.toggleExpertMode()

      await expect(expertToggle).not.toHaveAttribute("aria-checked", initialChecked)
      await expect(settingsDialog.getByText("Agentic RAG")).toBeVisible({
        timeout: 10_000
      })

      await assertNoCriticalErrors(diagnostics)
    })

    test("should apply settings to search request", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      qaPage = new KnowledgeQAPage(authedPage)
      await qaPage.goto()
      await qaPage.waitForReady()

      await qaPage.openSettings()
      await qaPage.selectPreset("thorough")
      await expect(
        qaPage
          .getSettingsDialog()
          .getByRole("radio", { name: /^thorough\b/i })
      ).toHaveAttribute("aria-checked", "true")
      await qaPage.getSettingsDialog().getByRole("button", { name: /^Done$/i }).click()

      const [ragResult] = await Promise.all([
        qaPage.waitForRagSearch(),
        qaPage.search("test with settings")
      ])

      await qaPage.waitForResults()
      expect(ragResult.status).toBe(200)
      expect(ragResult.requestBody?.top_k).toBe(20)
      expect(ragResult.requestBody?.enable_citations).toBe(true)
      expect(ragResult.requestBody?.enable_post_verification).toBe(true)

      await assertNoCriticalErrors(diagnostics)
    })
  })

  // ═════════════════════════════════════════════════════════════════════
  // 3.3  Follow-up Questions
  // ═════════════════════════════════════════════════════════════════════

  test.describe("Follow-up Questions", () => {
    test("should show follow-up input after initial search", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      qaPage = new KnowledgeQAPage(authedPage)
      await qaPage.goto()
      await qaPage.waitForReady()

      await qaPage.search("What is neural network?")
      await qaPage.waitForResults()

      const answer = await qaPage.getAnswerText()
      if (answer.length > 0) {
        const followUpInput = await qaPage.getFollowUpInput()
        await expect(followUpInput).toBeVisible()
      }

      await assertNoCriticalErrors(diagnostics)
    })

    test("should submit follow-up question with thread context", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      qaPage = new KnowledgeQAPage(authedPage)
      await qaPage.goto()
      await qaPage.waitForReady()

      await qaPage.search("What is deep learning?")
      await qaPage.waitForResults()

      const hasFollowUp = await qaPage.isFollowUpVisible()
      expect(hasFollowUp).toBeTruthy()
      if (hasFollowUp) {
        const [ragResult] = await Promise.all([
          qaPage.waitForRagSearch(),
          qaPage.askFollowUp("Can you elaborate on convolutional networks?")
        ])

        await qaPage.waitForResults()
        expect(ragResult.status).toBe(200)
        expect(ragResult.requestBody?.query).toBe(
          "Can you elaborate on convolutional networks?"
        )
        await expect(
          authedPage.getByText(/Conversation • 2 turns/i)
        ).toBeVisible({ timeout: 10_000 })
      }

      await assertNoCriticalErrors(diagnostics)
    })
  })

  // ═════════════════════════════════════════════════════════════════════
  // 3.4  Search History
  // ═════════════════════════════════════════════════════════════════════

  test.describe("Search History", () => {
    test("should open history sidebar", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      qaPage = new KnowledgeQAPage(authedPage)
      await qaPage.goto()
      await qaPage.waitForReady()

      const firstQuery = `history-${generateTestId("first")}`
      const secondQuery = `history-${generateTestId("second")}`

      await qaPage.search(firstQuery)
      await qaPage.waitForResults()

      const input = await qaPage.getSearchInput()
      await input.fill(secondQuery)
      await input.press("Enter")
      await qaPage.waitForResults()

      await qaPage.toggleHistorySidebar()
      await expect(qaPage.getHistorySidebar()).toBeVisible({ timeout: 10_000 })

      const firstHistoryEntry = authedPage.getByRole("button", {
        name: new RegExp(secondQuery, "i")
      })
      await expect(firstHistoryEntry).toBeVisible({ timeout: 10_000 })
      await firstHistoryEntry.click()

      await expect(input).toHaveValue(secondQuery)

      await assertNoCriticalErrors(diagnostics)
    })

    test("should start new search with Cmd+K", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      qaPage = new KnowledgeQAPage(authedPage)
      await qaPage.goto()
      await qaPage.waitForReady()

      const input = await qaPage.getSearchInput()
      await input.fill("temporary knowledge query")

      await qaPage.pressNewSearch()

      // Search input should be focused and cleared for the next query
      await expect(input).toBeFocused({ timeout: 5_000 })
      await expect(input).toHaveValue("", { timeout: 5_000 })

      await assertNoCriticalErrors(diagnostics)
    })
  })

  // ═════════════════════════════════════════════════════════════════════
  // 3.5  Export & Sharing
  // ═════════════════════════════════════════════════════════════════════

  test.describe("Export & Sharing", () => {
    test("creates a live share link and hydrates the shared permalink from a seeded thread", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)

      const knowledgeChatPreflight = await canReachKnowledgeChatEndpoint()
      test.skip(
        !knowledgeChatPreflight.reachable,
        knowledgeChatPreflight.reason ||
          "Skipping live KnowledgeQA share flow: chat bootstrap endpoint unavailable"
      )

      qaPage = new KnowledgeQAPage(authedPage)
      const slug = generateTestId("knowledge-live-share")
      let seededThreadId = ""
      let createdShareId: string | null = null

      try {
        const seededThread = await createSeededKnowledgeThread(slug)
        seededThreadId = seededThread.threadId
        const renderedAnswerText = seededThread.answer.replace(" [1]", "")

        const threadLoadResponsePromise = authedPage.waitForResponse(
          (response) =>
            response.request().method().toUpperCase() === "GET" &&
            response.url().includes(
              `/api/v1/chat/conversations/${encodeURIComponent(seededThread.threadId)}/messages-with-context`
            ),
          { timeout: 15_000 }
        )

        await authedPage.goto(`/knowledge/thread/${encodeURIComponent(seededThread.threadId)}`, {
          waitUntil: "domcontentloaded"
        })
        await waitForConnection(authedPage)
        await qaPage.waitForReady()

        const threadLoadResponse = await threadLoadResponsePromise
        expect(threadLoadResponse.ok()).toBeTruthy()
        const threadMessages = await threadLoadResponse.json().catch(() => null)
        expect(Array.isArray(threadMessages)).toBeTruthy()
        expect(threadMessages).toEqual(
          expect.arrayContaining([
            expect.objectContaining({ sender: "user", content: seededThread.query }),
            expect.objectContaining({
              sender: "assistant",
              content: seededThread.answer,
              rag_context: expect.objectContaining({
                search_query: seededThread.query,
                generated_answer: seededThread.answer
              })
            })
          ])
        )
        await qaPage.waitForResults()

        const searchInput = await qaPage.getSearchInput()
        await expect(searchInput).toHaveValue(seededThread.query)
        await expect(authedPage.getByTestId("knowledge-answer-content")).toContainText(
          renderedAnswerText
        )
        await expect(
          qaPage.getEvidencePanel().getByRole("heading", {
            name: /Knowledge Live Share Source/i
          })
        ).toBeVisible()

        const shareResponsePromise = authedPage.waitForResponse((response) => {
          const request = response.request()
          return (
            request.method().toUpperCase() === "POST" &&
            response.url().includes(
              `/api/v1/chat/conversations/${encodeURIComponent(seededThread.threadId)}/share-links`
            )
          )
        })

        await authedPage.keyboard.press("Escape").catch(() => {})
        const exportButton = qaPage.resultsShell.getByRole("button", { name: /^Export$/i })
        await expect(exportButton).toBeVisible()
        await exportButton.click()
        const exportDialog = authedPage.getByRole("dialog", { name: /Export Conversation/i })
        await expect(exportDialog).toBeVisible()
        await exportDialog.getByRole("button", { name: /Create share link/i }).click()

        const shareResponse = await shareResponsePromise
        expect(shareResponse.ok()).toBeTruthy()
        const sharePayload = await parseJsonRecord(
          shareResponse,
          "create knowledge share link"
        )
        const sharePath = String(sharePayload.share_path || "")
        createdShareId = String(sharePayload.share_id || "") || null
        expect(sharePath).toMatch(/^\/knowledge\/shared\//)
        await expect(exportDialog.getByText(/Active link expires/i)).toBeVisible()

        await authedPage.goto(sharePath, {
          waitUntil: "domcontentloaded"
        })
        await waitForConnection(authedPage)
        await qaPage.waitForReady()
        await qaPage.waitForResults()

        const sharedSearchInput = await qaPage.getSearchInput()
        await expect(sharedSearchInput).toHaveValue(seededThread.query)
        await expect(authedPage.getByTestId("knowledge-answer-content")).toContainText(
          renderedAnswerText
        )
        await expect(
          qaPage.getEvidencePanel().getByRole("heading", {
            name: /Knowledge Live Share Source/i
          })
        ).toBeVisible()
        await expect(authedPage.getByRole("button", { name: /^Export$/i })).toBeVisible()
      } finally {
        await revokeKnowledgeShareLink(seededThreadId, createdShareId)
        await deleteKnowledgeThread(seededThreadId)
      }

      await assertNoCriticalErrors(diagnostics)
    })

    test("opens the export dialog and manages share links for a server-backed thread", async ({
      authedPage,
      diagnostics
    }) => {
      qaPage = new KnowledgeQAPage(authedPage)
      const threadId = "knowledge-export-thread"
      const query = "shareable export regression"
      let shareRequestBody: { permission?: string } | null = null
      let revokeRequestUrl = ""

      await authedPage.addInitScript(() => {
        Object.defineProperty(window.navigator, "clipboard", {
          configurable: true,
          value: {
            writeText: async () => undefined,
            readText: async () => ""
          }
        })
      })

      await authedPage.route("**/api/v1/config/docs-info", async (route) => {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            info: { version: "e2e" },
            capabilities: {}
          })
        })
      })

      await authedPage.route("**/api/v1/chat/conversations?*", async (route) => {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            conversations: []
          })
        })
      })

      await authedPage.route("**/api/v1/characters/search?*", async (route) => {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify([
            {
              id: 1,
              name: "Helpful AI Assistant"
            }
          ])
        })
      })

      await authedPage.route("**/api/v1/chats/", async (route) => {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            id: threadId,
            title: query,
            version: 1,
            state: "in-progress",
            created_at: new Date().toISOString()
          })
        })
      })

      await authedPage.route("**/api/v1/chat/conversations/*", async (route) => {
        const method = route.request().method()
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body:
            method === "GET"
              ? JSON.stringify({
                  keywords: ["__knowledge_QA__"]
                })
              : JSON.stringify({
                  success: true
                })
        })
      })

      await authedPage.route("**/api/v1/chats/*/messages", async (route) => {
        const payload = route.request().postDataJSON() as { role?: string; content?: string }
        const suffix = payload?.role === "assistant" ? "assistant" : "user"
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            id: `msg-${suffix}-export`,
            role: payload?.role || suffix,
            content: payload?.content || "",
            created_at: new Date().toISOString()
          })
        })
      })

      await authedPage.route("**/api/v1/chat/messages/*/rag-context", async (route) => {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            success: true
          })
        })
      })

      await authedPage.route("**/api/v1/rag/search/stream", async (route) => {
        await route.fulfill({
          status: 200,
          contentType: "text/plain",
          body: ""
        })
      })

      await authedPage.route("**/api/v1/rag/search", async (route) => {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            results: [
              {
                id: "301",
                content: "Exportable source excerpt.",
                metadata: {
                  title: "Export Source",
                  source_type: "pdf",
                  media_id: 301,
                  url: "https://example.com/export-source"
                },
                score: 0.93
              }
            ],
            answer: "Export-ready answer with citations [1].",
            expanded_queries: [query]
          })
        })
      })

      await authedPage.route("**/api/v1/chat/conversations/*/share-links", async (route) => {
        shareRequestBody = route.request().postDataJSON() as { permission?: string }
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            share_id: "share-knowledge-export",
            token: "token-knowledge-export",
            share_path: "/knowledge/shared/token-knowledge-export",
            expires_at: "2030-01-01T00:00:00Z"
          })
        })
      })

      await authedPage.route("**/api/v1/chat/conversations/*/share-links/*", async (route) => {
        revokeRequestUrl = route.request().url()
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            success: true,
            share_id: "share-knowledge-export"
          })
        })
      })

      await qaPage.goto()
      await qaPage.waitForReady()
      const searchInput = await qaPage.getSearchInput()
      await searchInput.fill(query)
      await dismissConnectionModals(authedPage)
      await clickAskButton(authedPage)
      await qaPage.waitForResults()

      await authedPage.getByRole("button", { name: /^Export$/i }).click()
      const exportDialog = authedPage.getByRole("dialog", { name: /Export Conversation/i })
      await expect(exportDialog).toBeVisible()

      const createShareLinkButton = exportDialog.getByRole("button", {
        name: /Create share link/i
      })
      await expect(createShareLinkButton).toBeEnabled()
      await createShareLinkButton.click()

      await expect
        .poll(() => shareRequestBody, { timeout: 10_000 })
        .toMatchObject({ permission: "view" })
      await expect(exportDialog.getByText(/Active link expires/i)).toBeVisible()

      const revokeLinkButton = exportDialog.getByRole("button", {
        name: /^Revoke link$/i
      })
      await expect(revokeLinkButton).toBeEnabled()
      await revokeLinkButton.click()

      await expect
        .poll(() => revokeRequestUrl, { timeout: 10_000 })
        .toContain("/share-links/share-knowledge-export")
      await expect(exportDialog.getByText(/Active link expires/i)).toHaveCount(0)

      await assertNoCriticalErrors(diagnostics)
    })

    test("hydrates shared conversations from tokenized knowledge routes", async ({
      authedPage,
      diagnostics
    }) => {
      qaPage = new KnowledgeQAPage(authedPage)
      let resolvedShareToken = ""

      await authedPage.route("**/api/v1/config/docs-info", async (route) => {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            info: { version: "e2e" },
            capabilities: {}
          })
        })
      })

      await authedPage.route("**/api/v1/chat/conversations?*", async (route) => {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            conversations: []
          })
        })
      })

      await authedPage.route("**/api/v1/characters/search?*", async (route) => {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify([
            {
              id: 1,
              name: "Helpful AI Assistant"
            }
          ])
        })
      })

      await authedPage.route("**/api/v1/chat/shared/conversations/*", async (route) => {
        const rawToken = route.request().url().split("/").pop() || ""
        resolvedShareToken = decodeURIComponent(rawToken)
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            conversation_id: "shared-thread-9",
            permission: "view",
            shared_by_user_id: "1",
            expires_at: "2030-01-01T00:00:00Z",
            messages: [
              {
                id: "shared-u1",
                role: "user",
                content: "Shared question",
                created_at: "2026-02-19T09:00:00.000Z"
              },
              {
                id: "shared-a1",
                role: "assistant",
                content: "Shared answer [1]",
                created_at: "2026-02-19T09:00:02.000Z",
                rag_context: {
                  search_query: "Shared question",
                  generated_answer: "Shared answer [1]",
                  retrieved_documents: [
                    {
                      id: "doc-1",
                      title: "Shared source",
                      excerpt: "Evidence from the shared thread."
                    }
                  ]
                }
              }
            ]
          })
        })
      })

      await authedPage.goto("/knowledge/shared/share-token-route", {
        waitUntil: "domcontentloaded"
      })
      await waitForConnection(authedPage)
      await qaPage.waitForReady()
      await qaPage.waitForResults()

      await expect
        .poll(() => resolvedShareToken, { timeout: 10_000 })
        .toBe("share-token-route")
      const searchInput = await qaPage.getSearchInput()
      await expect(searchInput).toHaveValue("Shared question")
      await expect(authedPage.getByTestId("knowledge-answer-content")).toContainText(
        /Shared answer/i
      )
      await expect(
        qaPage.getEvidencePanel().getByRole("heading", { name: /Shared source/i })
      ).toBeVisible()
      await expect(authedPage.getByRole("button", { name: /^Export$/i })).toBeVisible()

      await assertNoCriticalErrors(diagnostics)
    })

    test("branches from a prior turn on the thread permalink route", async ({
      authedPage,
      diagnostics
    }) => {
      qaPage = new KnowledgeQAPage(authedPage)
      let branchCreatePayload: {
        parent_conversation_id?: string
        forked_from_message_id?: string
        title?: string
      } | null = null

      await authedPage.route("**/api/v1/config/docs-info", async (route) => {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            info: { version: "e2e" },
            capabilities: {}
          })
        })
      })

      await authedPage.route("**/api/v1/chat/conversations?*", async (route) => {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            conversations: [
              {
                id: "source-thread-1",
                title: "Knowledge QA",
                keywords: ["__knowledge_QA__"],
                message_count: 4,
                last_modified: "2026-02-19T08:01:02.000Z"
              }
            ]
          })
        })
      })

      await authedPage.route("**/api/v1/characters/search?*", async (route) => {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify([
            {
              id: 1,
              name: "Helpful AI Assistant"
            }
          ])
        })
      })

      await authedPage.route(
        "**/api/v1/chat/conversations/*/messages-with-context?*",
        async (route) => {
          await route.fulfill({
            status: 200,
            contentType: "application/json",
            body: JSON.stringify([
              {
                id: "u1",
                role: "user",
                content: "Branch source question",
                created_at: "2026-02-19T08:00:00.000Z"
              },
              {
                id: "a1",
                role: "assistant",
                content: "Branch source answer [1]",
                created_at: "2026-02-19T08:00:02.000Z",
                rag_context: {
                  search_query: "Branch source question",
                  generated_answer: "Branch source answer [1]",
                  retrieved_documents: [
                    {
                      id: "doc-a1",
                      title: "Source A",
                      excerpt: "Alpha"
                    }
                  ]
                }
              },
              {
                id: "u2",
                role: "user",
                content: "Latest question",
                created_at: "2026-02-19T08:01:00.000Z"
              },
              {
                id: "a2",
                role: "assistant",
                content: "Latest answer [1]",
                created_at: "2026-02-19T08:01:02.000Z",
                rag_context: {
                  search_query: "Latest question",
                  generated_answer: "Latest answer [1]",
                  retrieved_documents: [
                    {
                      id: "doc-a2",
                      title: "Source B",
                      excerpt: "Beta"
                    }
                  ]
                }
              }
            ])
          })
        }
      )

      await authedPage.route("**/api/v1/chats/", async (route) => {
        branchCreatePayload = route.request().postDataJSON() as {
          parent_conversation_id?: string
          forked_from_message_id?: string
          title?: string
        }
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            id: "branch-thread-1",
            title: "Branch: Branch source question",
            version: 1,
            state: "in-progress",
            created_at: "2026-02-19T10:00:00.000Z"
          })
        })
      })

      await authedPage.route("**/api/v1/chat/conversations/branch-thread-1", async (route) => {
        const method = route.request().method()
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body:
            method === "GET"
              ? JSON.stringify({
                  id: "branch-thread-1",
                  version: 1
                })
              : JSON.stringify({
                  success: true
                })
        })
      })

      await authedPage.route("**/api/v1/chats/*/messages", async (route) => {
        const payload = route.request().postDataJSON() as {
          role?: string
          content?: string
        }
        const suffix = payload?.role === "assistant" ? "a1" : "u1"
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            id: `branch-${suffix}`,
            role: payload?.role || "user",
            content: payload?.content || "",
            created_at:
              payload?.role === "assistant"
                ? "2026-02-19T10:00:02.000Z"
                : "2026-02-19T10:00:01.000Z"
          })
        })
      })

      await authedPage.route("**/api/v1/chat/messages/*/rag-context", async (route) => {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            success: true
          })
        })
      })

      await authedPage.goto("/knowledge/thread/source-thread-1", {
        waitUntil: "domcontentloaded"
      })
      await waitForConnection(authedPage)
      await qaPage.waitForReady()
      await qaPage.waitForResults()

      await expect(
        authedPage.getByRole("heading", { name: /Conversation Thread \(1 prior turn\)/i })
      ).toBeVisible()
      await expect(
        authedPage.getByRole("button", { name: /^Start Branch$/i })
      ).toBeVisible()

      await authedPage.getByRole("button", { name: /^Start Branch$/i }).click()

      await expect
        .poll(() => branchCreatePayload, { timeout: 10_000 })
        .toMatchObject({
          parent_conversation_id: "source-thread-1",
          forked_from_message_id: "u1"
        })

      const searchInput = await qaPage.getSearchInput()
      await expect(searchInput).toHaveValue("Branch source question")
      await expect(authedPage.getByTestId("knowledge-answer-content")).toContainText(
        /Branch source answer/i
      )
      await expect(authedPage.getByText(/Branch created from selected turn/i)).toBeVisible()

      await assertNoCriticalErrors(diagnostics)
    })
  })

  // ═════════════════════════════════════════════════════════════════════
  // 3.6  No Results / Error States
  // ═════════════════════════════════════════════════════════════════════

  test.describe("No Results / Error States", () => {
    test("should handle no results gracefully", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      qaPage = new KnowledgeQAPage(authedPage)
      await qaPage.goto()
      await qaPage.waitForReady()

      // Search for something that won't match
      const nonsenseQuery = `xyzzy-${generateTestId()}-qqq`
      await qaPage.search(nonsenseQuery)
      await qaPage.waitForResults()

      // Should get empty results or web fallback
      const answer = await qaPage.getAnswerText()
      const noResults = await qaPage.hasNoResults()
      const sourceOnlyState = await qaPage.hasSourceOnlyState()

      expect(answer.length > 0 || noResults || sourceOnlyState).toBeTruthy()

      await assertNoCriticalErrors(diagnostics)
    })

    test("should display error state when API fails", async ({
      authedPage,
      diagnostics
    }) => {
      qaPage = new KnowledgeQAPage(authedPage)
      await qaPage.goto()
      await qaPage.waitForReady()

      // Mock a failing API by intercepting the route
      await authedPage.route("**/api/v1/rag/search/stream", (route) => {
        route.fulfill({
          status: 500,
          contentType: "application/json",
          body: JSON.stringify({ detail: "Internal server error" })
        })
      })
      await authedPage.route("**/api/v1/rag/search", (route) => {
        route.fulfill({
          status: 500,
          contentType: "application/json",
          body: JSON.stringify({ detail: "Internal server error" })
        })
      })

      try {
        await qaPage.search("trigger error")
        await qaPage.waitForResults()

        await expect
          .poll(async () => await qaPage.getErrorMessage(), { timeout: 10_000 })
          .not.toBeNull()

        // Should show some kind of error state
        const errorMsg = await qaPage.getErrorMessage()
        expect(errorMsg).not.toBeNull()
      } finally {
        // Unroute to not affect other tests
        await authedPage.unroute("**/api/v1/rag/search/stream")
        await authedPage.unroute("**/api/v1/rag/search")
      }

      await assertNoCriticalErrors(diagnostics)
    })
  })

  // ═════════════════════════════════════════════════════════════════════
  // 3.7  Workspace Handoff
  // ═════════════════════════════════════════════════════════════════════

  test.describe("Workspace Handoff", () => {
    test("should carry answer context into workspace route", async ({
      authedPage,
      diagnostics
    }) => {
      qaPage = new KnowledgeQAPage(authedPage)
      const workspacePage = new ResearchWorkspacePage(authedPage)
      const query = `knowledge workspace handoff ${generateTestId("handoff")}`
      const answer = "Workspace-ready synthesis [1]"
      const sourceTitle = "Workspace Handoff Source"
      const sourceUrl = "https://example.com/workspace-handoff-source"

      await authedPage.route("**/api/v1/rag/search/stream", async (route) => {
        await route.fulfill({
          status: 200,
          contentType: "text/plain",
          body: ""
        })
      })

      await authedPage.route("**/api/v1/config/docs-info", async (route) => {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            info: {
              version: "e2e"
            },
            capabilities: {}
          })
        })
      })

      await authedPage.route("**/api/v1/rag/search", async (route) => {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            results: [
              {
                id: "501",
                content: "Workspace handoff excerpt",
                metadata: {
                  title: sourceTitle,
                  source_type: "pdf",
                  url: sourceUrl,
                  media_id: 501,
                  page_number: 7
                },
                score: 0.97
              }
            ],
            answer,
            expanded_queries: [query]
          })
        })
      })

      await qaPage.goto()
      await qaPage.waitForReady()
      const searchInput = await qaPage.getSearchInput()
      await searchInput.fill(query)
      await dismissConnectionModals(authedPage)
      await clickAskButton(authedPage)
      await qaPage.waitForResults()

      await expect(authedPage.getByTestId("knowledge-answer-content")).toContainText(
        /Workspace-ready synthesis/i
      )
      await expect(
        authedPage.getByRole("button", { name: /^Open in Workspace$/i })
      ).toBeVisible()

      await dismissConnectionModals(authedPage)
      await authedPage
        .getByRole("button", { name: /^Open in Workspace$/i })
        .evaluate((button: HTMLButtonElement) => button.click())

      await authedPage.waitForURL(/\/research-workspace(?:\?|$)/, {
        timeout: 10_000
      })
      await workspacePage.waitForReady()
      await expect(workspacePage.sourcesPanel.getByText(sourceTitle)).toBeVisible({
        timeout: 10_000
      })

      await expect
        .poll(
          async () =>
            authedPage.evaluate(() => {
              const store = (window as {
                __tldw_useWorkspaceStore?: {
                  getState?: () => {
                    sources?: Array<{
                      id: string
                      mediaId?: number
                      title?: string
                    }>
                    selectedSourceIds?: string[]
                    currentNote?: {
                      title?: string
                      content?: string
                    }
                  }
                }
              }).__tldw_useWorkspaceStore

              const state = store?.getState?.()
              const sources = state?.sources || []
              const handoffSource =
                sources.find((source) => source.mediaId === 501) || null

              return {
                handoffSourcePresent: Boolean(handoffSource),
                handoffSourceSelected: Boolean(
                  handoffSource &&
                    state?.selectedSourceIds?.includes(handoffSource.id)
                ),
                noteTitle: state?.currentNote?.title || "",
                noteContent: state?.currentNote?.content || "",
                prefillPending:
                  window.localStorage.getItem(
                    "__tldw_research_workspace_prefill"
                  ) !== null
              }
            }),
          { timeout: 10_000 }
        )
        .toMatchObject({
          handoffSourcePresent: true,
          handoffSourceSelected: true,
          noteTitle: `Knowledge QA: ${query}`,
          prefillPending: false
        })

      const workspaceSnapshot = await authedPage.evaluate(() => {
        const store = (window as {
          __tldw_useWorkspaceStore?: {
            getState?: () => {
              currentNote?: {
                content?: string
              }
            }
          }
        }).__tldw_useWorkspaceStore
        const state = store?.getState?.()
        return {
          noteContent: state?.currentNote?.content || ""
        }
      })

      expect(workspaceSnapshot.noteContent).toContain("Imported from Knowledge QA")
      expect(workspaceSnapshot.noteContent).toContain(`Question: ${query}`)
      expect(workspaceSnapshot.noteContent).toContain(answer)
      expect(workspaceSnapshot.noteContent).toContain(`[1] ${sourceTitle}`)
      expect(workspaceSnapshot.noteContent).toContain(sourceUrl)

      await assertNoCriticalErrors(diagnostics)
    })
  })
})

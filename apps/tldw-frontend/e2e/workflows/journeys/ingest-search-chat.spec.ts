/**
 * Journey: Ingest -> Search -> Chat
 *
 * End-to-end workflow that ingests content via URL, searches for it,
 * then chats about it with RAG context.
 */
import { test, expect, skipIfServerUnavailable, skipIfNoModels } from "../../utils/fixtures"
import path from "node:path"
import { readFile } from "node:fs/promises"
import { ChatPage, SearchPage } from "../../utils/page-objects"
import { waitForStreamComplete } from "../../utils/journey-helpers"
import { fetchWithApiKey, generateTestId, TEST_CONFIG } from "../../utils/helpers"

const PLAYWRIGHT_DOCUMENT = "Playwright is an open-source framework for reliable end-to-end browser testing."

type IngestedMedia = {
  id: string
  title: string
}

const ingestLocalDocument = async (title: string, content: string): Promise<IngestedMedia> => {
  const body = new FormData()
  body.append("media_type", "document")
  body.append("title", title)
  body.append("perform_analysis", "false")
  body.append("perform_chunking", "true")
  body.append("chunk_method", "sentences")
  body.append(
    "files",
    new Blob([content], { type: "text/plain" }),
    `${title}.txt`
  )

  const response = await fetchWithApiKey(
    `${TEST_CONFIG.serverUrl}/api/v1/media/add`,
    TEST_CONFIG.apiKey,
    { method: "POST", body }
  )
  if (!response.ok) {
    throw new Error(`Failed to ingest local fixture: ${response.status} ${await response.text()}`)
  }

  const payload = await response.json() as {
    results?: Array<{ status?: string; db_id?: number }>
  }
  const result = payload.results?.[0]
  if (result?.status !== "Success" || !result.db_id) {
    throw new Error("Local fixture was not persisted by /media/add")
  }
  return { id: String(result.db_id), title }
}

const deleteOwnedMedia = async (mediaId: string): Promise<void> => {
  const trashResponse = await fetchWithApiKey(
    `${TEST_CONFIG.serverUrl}/api/v1/media/${mediaId}`,
    TEST_CONFIG.apiKey,
    { method: "DELETE" }
  )
  if (!trashResponse.ok) {
    throw new Error(`Failed to trash seeded media ${mediaId}: ${trashResponse.status} ${await trashResponse.text()}`)
  }
  const permanentResponse = await fetchWithApiKey(
    `${TEST_CONFIG.serverUrl}/api/v1/media/${mediaId}/permanent`,
    TEST_CONFIG.apiKey,
    { method: "DELETE" }
  )
  if (!permanentResponse.ok) {
    throw new Error(`Failed to permanently delete seeded media ${mediaId}: ${permanentResponse.status} ${await permanentResponse.text()}`)
  }
}

const assertTokenScopedRagContext = async (fixture: IngestedMedia): Promise<void> => {
  const response = await fetchWithApiKey(
    `${TEST_CONFIG.serverUrl}/api/v1/rag/search`,
    TEST_CONFIG.apiKey,
    {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        query: `Playwright ${fixture.title}`,
        sources: ["media_db"],
        search_mode: "fts",
        fts_level: "media",
        top_k: 1,
        enable_generation: true,
        api_name: "openai",
        model: "gpt-4o-mini",
      }),
    }
  )
  if (!response.ok) {
    throw new Error(`Failed to retrieve seeded Playwright fixture: ${response.status} ${await response.text()}`)
  }
  const payload = await response.json() as {
    documents?: Array<{ content?: string; metadata?: { title?: string } }>
    generated_answer?: string
  }
  expect(payload.documents).toHaveLength(1)
  expect(payload.documents?.[0]?.metadata?.title).toBe(fixture.title)
  expect(payload.documents?.[0]?.content).toContain(`Fixture token: ${fixture.title}`)
  expect(payload.generated_answer).toBe(PLAYWRIGHT_DOCUMENT)
}

const ingestLocalPlaywrightFixture = async (fixtureTitle: string): Promise<IngestedMedia> => {
  const fixturePath = path.resolve(
    process.cwd(),
    "e2e/fixtures/media/playwright-grounded.txt"
  )
  return await ingestLocalDocument(
    fixtureTitle,
    `${await readFile(fixturePath, "utf8")}\nFixture token: ${fixtureTitle}`
  )
}

test.describe("Ingest -> Search -> Chat journey", () => {
  test("ingest content, search for it, then chat with RAG context", async ({
    authedPage: page,
    serverInfo,
  }) => {
    skipIfServerUnavailable(serverInfo)
    skipIfNoModels(serverInfo)

    const fixtureTitle = generateTestId("task2c-playwright")
    const ownedMedia: IngestedMedia[] = []

    try {
      await test.step("Seed a pre-existing Playwright-like document", async () => {
        ownedMedia.push(await ingestLocalDocument(
          generateTestId("preexisting-playwright"),
          "Playwright is a pre-existing unrelated document."
        ))
      })

      const fixture = await test.step("Ingest deterministic local Playwright content", async () => {
        const seededFixture = await ingestLocalPlaywrightFixture(fixtureTitle)
        ownedMedia.push(seededFixture)
        expect(seededFixture.id).toBeTruthy()
        return seededFixture
      })

      await test.step("Retrieve only the generated-token fixture", async () => {
        await assertTokenScopedRagContext(fixture)
      })

      await test.step("Search for the ingested content", async () => {
        const searchPage = new SearchPage(page)
        await searchPage.goto()
        await searchPage.waitForReady()

        await searchPage.search(`Playwright ${fixture.title}`)
        await searchPage.waitForResults()

        const results = await searchPage.getResults()
        expect(results.length).toBeGreaterThan(0)
      })

      await test.step("Chat about ingested content with RAG context", async () => {
        const chatPage = new ChatPage(page)
        await chatPage.goto()
        await chatPage.waitForReady()

        await chatPage.sendMessage("What is Playwright? Use the ingested content to answer.")
        await waitForStreamComplete(page)
        await chatPage.waitForResponse()

        const messages = await chatPage.getMessages()
        const assistantMessages = messages.filter((message) => message.role === "assistant")
        expect(assistantMessages.length).toBeGreaterThan(0)
        expect(assistantMessages.at(-1)?.content ?? "").toMatch(/playwright/i)
      })
    } finally {
      await Promise.all(ownedMedia.map(async ({ id }) => await deleteOwnedMedia(id)))
    }
  })
})

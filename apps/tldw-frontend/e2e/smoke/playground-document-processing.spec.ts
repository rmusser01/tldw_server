import { expect, seedAuth, test } from "./smoke.setup"

const MODEL_ID = "openai:gpt-4.1-mini"
const DOCUMENT_TEXT =
  "Parsed chat-scoped notes about document upload processing."
const ASSISTANT_TEXT = "I can use the chat-scoped notes."

const streamChunk = (text: string) =>
  `data: ${JSON.stringify({ choices: [{ delta: { content: text } }] })}\n\n`

const seedModelSelection = async (page: import("@playwright/test").Page) => {
  await page.addInitScript((modelId: string) => {
    window.localStorage.setItem("selectedModel", JSON.stringify(modelId))
  }, MODEL_ID)
}

const stubChatBootstrap = async (page: import("@playwright/test").Page) => {
  await page.route("**/api/v1/health", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ status: "ok", checks: {} })
    })
  })

  await page.route("**/api/v1/llm/providers**", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        providers: [
          {
            id: "openai",
            name: "OpenAI",
            apiProvider: "openai",
            models: ["gpt-4.1-mini"],
            configured: true,
            available: true
          }
        ]
      })
    })
  })

  await page.route("**/api/v1/llm/models/metadata**", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        models: [
          {
            id: MODEL_ID,
            model: "gpt-4.1-mini",
            name: "GPT-4.1 Mini",
            provider: "openai",
            apiProvider: "openai",
            capabilities: ["chat"]
          }
        ]
      })
    })
  })
}

const stubDocumentPreflight = async (page: import("@playwright/test").Page) => {
  await page.route("**/api/v1/media/document-upload/preflight", async (route) => {
    const body = route.request().postDataJSON() as {
      files?: Array<{ client_id?: string; filename?: string }>
    }
    const file = body.files?.[0] ?? {}
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        files: [
          {
            client_id: file.client_id || "document-1",
            filename: file.filename || "upload.md",
            media_type: "document",
            default_mode: "add_to_chat",
            modes: {
              add_to_chat: { available: true, status: "available" },
              ocr_pages: {
                available: false,
                status: "unavailable",
                reason: "OCR unavailable: server cannot render .MD pages"
              },
              ingest_to_library: { available: true, status: "available" }
            },
            max_size_bytes: 20 * 1024 * 1024,
            max_pages: 200,
            max_chat_tokens: 24000,
            estimated_pages: null,
            estimated_tokens: null,
            requires_send_time_estimate: true
          }
        ]
      })
    })
  })
}

test("chat document upload can stay chat-scoped after send-time processing", async ({
  page
}) => {
  test.setTimeout(90_000)

  let releaseDocumentProcessing: (() => void) | null = null
  let markDocumentProcessingStarted: () => void = () => undefined
  const documentProcessingStarted = new Promise<void>((resolve) => {
    markDocumentProcessingStarted = resolve
  })
  await page.route("**/api/v1/media/process-documents", async (route) => {
    markDocumentProcessingStarted()
    await new Promise<void>((release) => {
      releaseDocumentProcessing = release
    })
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        results: [{ content: DOCUMENT_TEXT, source: "upload.md" }]
      })
    })
  })

  const chatRequests: Array<Record<string, unknown>> = []
  const fulfillChatStream = async (route: import("@playwright/test").Route) => {
    chatRequests.push(
      (route.request().postDataJSON() as Record<string, unknown>) || {}
    )
    await route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      headers: {
        "Cache-Control": "no-cache",
        Connection: "keep-alive"
      },
      body: streamChunk(ASSISTANT_TEXT) + "data: [DONE]\n\n"
    })
  }
  await page.route("**/api/v1/chat/completions", fulfillChatStream)
  await page.route("**/api/proxy/chat/completions", fulfillChatStream)
  await page.route(/\/api\/v1\/chats\/[^/]+\/complete(?:\?.*)?$/, fulfillChatStream)

  await seedAuth(page)
  await seedModelSelection(page)
  await stubChatBootstrap(page)
  await stubDocumentPreflight(page)

  await page.goto("/chat", { waitUntil: "domcontentloaded" })
  const input = page.getByTestId("chat-input").first()
  await input.waitFor({ state: "visible", timeout: 30_000 })

  await page.locator("#document-upload").setInputFiles({
    name: "upload.md",
    mimeType: "text/markdown",
    buffer: Buffer.from("# Upload notes\n\nUse these notes only in chat.")
  })

  await expect(page.getByRole("button", { name: /Attachments \(1\)/ })).toBeVisible()

  await input.fill("Summarize the uploaded notes")
  await page.getByTestId("composer-inline-send-control").click()

  await expect(page.getByText("Processing documents")).toBeVisible({
    timeout: 10_000
  })
  await expect(
    page.getByTestId("chat-message").getByText("upload.md")
  ).toBeVisible()
  await expect(
    page.getByTestId("chat-message").getByText("Add to chat")
  ).toBeVisible()

  await documentProcessingStarted
  releaseDocumentProcessing?.()

  await expect(page.getByText("Sending prompt")).toBeVisible({
    timeout: 10_000
  })
  await expect(page.getByText(ASSISTANT_TEXT)).toBeVisible({ timeout: 10_000 })
  await expect(page.getByText("Processing documents")).toHaveCount(0)
  await expect(page.getByText("Ingest to library")).toHaveCount(0)

  await expect.poll(() => chatRequests.length, { timeout: 10_000 }).toBe(1)
  const requestText = JSON.stringify(chatRequests[0])
  expect(requestText).toContain(DOCUMENT_TEXT)
  expect(requestText).not.toContain("ragMediaIds")
})

import AxeBuilder from "@axe-core/playwright"
import {
  expect,
  test,
  type Locator,
  type Page,
  type Route,
} from "@playwright/test"

import { expectNoHorizontalOverflow, seedAuth } from "../utils/helpers"

const API_ORIGIN = "http://127.0.0.1:18092"
const MODEL = "openai/gpt-e2e"
const SYSTEM_TEMPLATE_ID = "e2e-system-template"
const SYSTEM_TEMPLATE_TITLE = "E2E System Template"
const SYSTEM_DRAFT = "System draft for {{topic}}. Keep responses concise."
const USER_DRAFT = "Explain {{topic}} for a new reader."
const SYSTEM_CANDIDATE = "System draft for {{topic}}. Keep every response concise and direct."
const USER_CANDIDATE = "Explain {{topic}} clearly for a new reader and include one example."

const SENTINELS = {
  systemCounterpart: "SYSTEM_COUNTERPART_SENTINEL",
  userCounterpart: "USER_COUNTERPART_SENTINEL",
  history: "HISTORY_SENTINEL",
  pageContext: "PAGE_CONTEXT_SENTINEL",
  rag: "RAG_SENTINEL",
  tools: "TOOLS_SENTINEL",
} as const

const LIMITS = {
  max_request_bytes: 64_000,
  max_draft_chars: 24_000,
  max_candidate_chars: 24_000,
  max_raw_output_chars: 32_000,
  max_findings: 5,
  max_finding_text_chars: 500,
  max_provider_chars: 100,
  max_model_chars: 500,
  max_meta_prompt_version_chars: 100,
  max_warning_chars: 100,
  max_warnings: 16,
  max_protected_tokens: 64,
  max_protected_token_kind_chars: 50,
  max_protected_token_chars: 500,
  max_protected_token_occurrences: 100,
  max_protected_token_total_chars: 4_000,
}

type CapabilityMode = "supported" | "false" | "404" | "offline"
type ImproveRequest = {
  operation_id: string
  target: "system" | "user_message"
  text: string
  model_selection: { selected_model: string; provider_hint?: string }
  protected_tokens: Array<{
    kind: string
    value: string
    occurrences: number
  }>
}

type ApiMockOptions = {
  capability?: CapabilityMode
  modelAvailable?: boolean
  onImprove?: (
    route: Route,
    request: ImproveRequest,
    attempt: number,
  ) => Promise<void>
}

const json = async (route: Route, status: number, body: unknown) => {
  await route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(body),
  })
}

const capabilityResponse = (supported: boolean) => ({
  prompt_improvement_v1: { supported, limits: LIMITS },
  single_text_recipe_v2: { supported: false, limits: LIMITS },
})

const improvementResponse = (request: ImproveRequest) => ({
  schema_version: 1,
  operation_id: request.operation_id,
  status: "improved",
  improved_text:
    request.target === "system" ? SYSTEM_CANDIDATE : USER_CANDIDATE,
  findings: [
    {
      category: "clarity",
      issue: "The requested result could be more explicit.",
      change: "Clarified the expected result.",
    },
  ],
  review_required: false,
  warnings: [],
  resolved_model: {
    provider: "openai",
    model: "gpt-e2e",
    display_name: "GPT E2E",
  },
  meta_prompt_version: "prompt-improvement-v1",
})

async function mockChatApi(page: Page, options: ApiMockOptions = {}) {
  const requests: ImproveRequest[] = []
  let attempts = 0

  await page.route("**/api/v1/**", async (route) => {
    const request = route.request()
    const path = new URL(request.url()).pathname

    if (request.method() === "OPTIONS") {
      await route.fulfill({ status: 204 })
      return
    }

    if (path === "/api/v1/prompts/capabilities") {
      const mode = options.capability ?? "supported"
      if (mode === "offline") {
        await route.abort("failed")
      } else if (mode === "404") {
        await json(route, 404, { detail: "Not found" })
      } else {
        await json(route, 200, capabilityResponse(mode === "supported"))
      }
      return
    }

    if (path === "/api/v1/prompts/improve") {
      const body = request.postDataJSON() as ImproveRequest
      requests.push(body)
      attempts += 1
      if (options.onImprove) {
        await options.onImprove(route, body, attempts)
      } else {
        await json(route, 200, improvementResponse(body))
      }
      return
    }

    if (path === "/api/v1/health") {
      await json(route, 200, { status: "ok" })
      return
    }

    if (path === "/api/v1/llm/models/metadata") {
      await json(route, 200, {
        models: options.modelAvailable === false ? [] : [
          {
            id: MODEL,
            name: "gpt-e2e",
            provider: "openai",
            type: "chat",
            is_configured: true,
            provider_enabled: true,
            availability: "available",
          },
        ],
      })
      return
    }

    if (path === "/api/v1/llm/providers") {
      await json(route, 200, {
        providers: options.modelAvailable === false ? [] : [
          {
            name: "openai",
            display_name: "OpenAI",
            is_configured: true,
            enabled: true,
            models: ["gpt-e2e"],
          },
        ],
      })
      return
    }

    if (path.startsWith("/api/v1/characters")) {
      await json(route, 200, [])
      return
    }

    await route.abort("failed")
  })

  return requests
}

async function prepareChat(
  page: Page,
  options: ApiMockOptions & { selectedModel?: boolean } = {},
) {
  await seedAuth(page, {
    serverUrl: API_ORIGIN,
    allowOffline: true,
  })
  await page.addInitScript(
    ({ model, selectedModel }: { model: string; selectedModel: boolean }) => {
      localStorage.removeItem("tldw:nextgenComposerEnabled")
      localStorage.removeItem("tldw:composerVariant")
      localStorage.setItem("playgroundComposerOptionsExpanded", "true")
      if (selectedModel) {
        localStorage.setItem("selectedModel", JSON.stringify(model))
        localStorage.setItem(
          "plasmo-storage-selectedModel",
          JSON.stringify(model),
        )
      } else {
        localStorage.removeItem("selectedModel")
        localStorage.removeItem("plasmo-storage-selectedModel")
      }
    },
    { model: MODEL, selectedModel: options.selectedModel !== false },
  )
  const requests = await mockChatApi(page, options)
  const runtimeError = new Promise<never>((_resolve, reject) => {
    page.once("pageerror", reject)
  })
  await page.goto("/chat", { waitUntil: "domcontentloaded" })
  await Promise.race([
    page.getByTestId("chat-input").first().waitFor({
      state: "visible",
      timeout: 30_000,
    }),
    runtimeError,
  ])
  await expect(page.getByRole("button", { name: "Improve prompt" }).first()).toBeVisible()
  return requests
}

async function seedTemplate(page: Page, content = SYSTEM_DRAFT) {
  await page.evaluate(
    ({ id, title, content }) =>
      new Promise<void>((resolve, reject) => {
        const open = indexedDB.open("PageAssistDatabase")
        open.onerror = () => reject(open.error)
        open.onsuccess = () => {
          const database = open.result
          const transaction = database.transaction("prompts", "readwrite")
          transaction.objectStore("prompts").put({
            id,
            title,
            name: title,
            content,
            is_system: true,
            createdAt: Date.now(),
            updatedAt: Date.now(),
            deletedAt: null,
          })
          transaction.oncomplete = () => {
            database.close()
            resolve()
          }
          transaction.onerror = () => reject(transaction.error)
        }
      }),
    {
      id: SYSTEM_TEMPLATE_ID,
      title: SYSTEM_TEMPLATE_TITLE,
      content,
    },
  )
  await page.evaluate((id) => {
    localStorage.setItem("selectedSystemPrompt", JSON.stringify(id))
  }, SYSTEM_TEMPLATE_ID)
  await page.reload({ waitUntil: "domcontentloaded" })
  await expect(page.getByTestId("chat-input").first()).toBeVisible({
    timeout: 30_000,
  })
}

async function seedExcludedContext(page: Page) {
  await page.waitForFunction(() =>
    Boolean(
      (window as unknown as { __tldw_useStoreMessageOption?: unknown })
        .__tldw_useStoreMessageOption,
    ),
  )
  await page.evaluate((sentinels) => {
    const store = (
      window as unknown as {
        __tldw_useStoreMessageOption: {
          setState: (state: Record<string, unknown>) => void
        }
      }
    ).__tldw_useStoreMessageOption
    store.setState({
      messages: [
        {
          id: "history-sentinel",
          isBot: false,
          name: "User",
          role: "user",
          message: sentinels.history,
          sources: [],
          toolCalls: [
            {
              id: "tool-sentinel",
              type: "function",
              function: {
                name: sentinels.tools,
                arguments: "{}",
              },
            },
          ],
          createdAt: Date.now(),
        },
      ],
      history: [{ role: "user", content: sentinels.history }],
      documentContext: [
        {
          type: "tab",
          title: sentinels.pageContext,
          url: "https://context.invalid/e2e",
        },
      ],
      ragPinnedResults: [
        {
          id: "rag-sentinel",
          title: "Pinned result",
          snippet: sentinels.rag,
        },
      ],
      actionInfo: sentinels.tools,
    })
  }, SENTINELS)
}

async function openPromptActions(page: Page, scope: Page | Locator = page) {
  const trigger = scope.getByRole("button", { name: "Improve prompt" }).first()
  await trigger.click()
  await expect(
    page.getByRole("group", { name: "Prompt improvement actions" }).first(),
  ).toBeVisible()
  return trigger
}

function expectIsolatedRequest(
  request: ImproveRequest,
  expected: { target: ImproveRequest["target"]; text: string },
) {
  expect(Object.keys(request).sort()).toEqual([
    "model_selection",
    "operation_id",
    "protected_tokens",
    "target",
    "text",
  ])
  expect(request.target).toBe(expected.target)
  expect(request.text).toBe(expected.text)
  expect(request.model_selection).toEqual({ selected_model: MODEL })
  expect(request.protected_tokens).toEqual([
    { kind: "template_variable", value: "{{topic}}", occurrences: 1 },
  ])
  const serialized = JSON.stringify(request)
  for (const sentinel of Object.values(SENTINELS)) {
    expect(serialized).not.toContain(sentinel)
  }
}

test.describe("WebUI prompt improvement parity", () => {
  test("system Improve now preserves template identity and supports exact Undo", async ({ page }) => {
    const requests = await prepareChat(page)
    await seedTemplate(page)
    await page.getByTestId("chat-input").first().fill(SENTINELS.userCounterpart)
    await seedExcludedContext(page)

    const promptTrigger = page.getByTestId("chat-prompt-select").first()
    await expect(promptTrigger).toContainText(SYSTEM_TEMPLATE_TITLE)
    await promptTrigger.click()
    await page.getByRole("menuitem", { name: /Edit system prompt/i }).last().click()

    const editor = page.getByPlaceholder("Enter system prompt")
    await expect(editor).toHaveValue(SYSTEM_DRAFT)
    const editorDialog = page.getByRole("dialog", { name: "Edit system prompt" })
    await openPromptActions(page, editorDialog)
    await expect(page.getByText("Build from recipe", { exact: true })).toHaveCount(0)
    await page.getByRole("button", { name: /Improve now/ }).click()

    await expect(editor).toHaveValue(SYSTEM_CANDIDATE)
    await expect(page.getByText(/Override active:/)).toBeVisible()
    await expect(promptTrigger).toContainText(SYSTEM_TEMPLATE_TITLE)
    await page.getByRole("button", { name: "Undo improvement" }).click()
    await expect(editor).toHaveValue(SYSTEM_DRAFT)
    await expect(page.getByText(/Override active:/)).toHaveCount(0)
    await expect(promptTrigger).toContainText(SYSTEM_TEMPLATE_TITLE)

    expect(requests).toHaveLength(1)
    expectIsolatedRequest(requests[0], { target: "system", text: SYSTEM_DRAFT })
  })

  test("user Review changes remains editable and applies only the reviewed candidate", async ({ page }) => {
    const requests = await prepareChat(page)
    const composer = page.getByTestId("chat-input").first()
    await composer.fill(USER_DRAFT)
    await seedTemplate(page, SENTINELS.systemCounterpart)
    await composer.fill(USER_DRAFT)
    await seedExcludedContext(page)

    await openPromptActions(page)
    await page.getByRole("button", { name: /Review changes/ }).click()
    const candidate = page.getByRole("textbox", {
      name: "Improved prompt candidate",
    })
    await expect(candidate).toHaveValue(USER_CANDIDATE)
    await candidate.fill("Reviewed {{topic}} candidate with a concrete example.")
    await page.getByRole("button", { name: "Apply to draft" }).click()

    await expect(composer).toHaveValue(
      "Reviewed {{topic}} candidate with a concrete example.",
    )
    await expect(
      page.getByRole("button", { name: "Undo improvement" }).first(),
    ).toBeVisible()
    expect(requests).toHaveLength(1)
    expectIsolatedRequest(requests[0], {
      target: "user_message",
      text: USER_DRAFT,
    })
  })

  test("typing while Improve now is pending makes the result stale and requires confirmed replacement", async ({ page }) => {
    let release: (() => Promise<void>) | undefined
    const requests = await prepareChat(page, {
      onImprove: async (route, request) => {
        await new Promise<void>((resolve) => {
          release = async () => {
            await json(route, 200, improvementResponse(request))
            resolve()
          }
        })
      },
    })
    const composer = page.getByTestId("chat-input").first()
    await composer.fill(USER_DRAFT)
    await openPromptActions(page)
    await page.getByRole("button", { name: /Improve now/ }).click()
    await expect(page.getByTestId("prompt-assist-spinner")).toBeVisible()

    await composer.fill("Newer live draft typed while pending.")
    await expect.poll(() => Boolean(release)).toBe(true)
    await release?.()

    await expect(composer).toHaveValue("Newer live draft typed while pending.")
    await expect(
      page.getByRole("alert").filter({ hasText: "draft changed" }),
    ).toBeVisible()
    await page.getByRole("button", { name: "Replace current draft" }).click()
    await expect(page.getByRole("button", { name: "Confirm replace" })).toBeVisible()
    await page.getByRole("button", { name: "Confirm replace" }).click()
    await expect(composer).toHaveValue(USER_CANDIDATE)
    expect(requests).toHaveLength(1)
  })

  test("missing model recovery opens the existing model selector", async ({ page }) => {
    await prepareChat(page, {
      selectedModel: false,
      modelAvailable: false,
    })
    const composer = page.getByTestId("chat-input").first()
    await composer.fill(USER_DRAFT)
    await openPromptActions(page)

    await expect(page.getByText("Select a chat model to improve this draft.")).toBeVisible()
    await expect(page.getByRole("button", { name: /Improve now/ })).toBeDisabled()
    await page.getByRole("button", { name: "Select model" }).click()
    await expect(page.getByRole("textbox", { name: "Search models" })).toBeVisible()
    await expect(page.getByRole("menu")).toContainText("No models available")
  })

  test("structured provider failure is sanitized and explicit Retry starts a new operation", async ({ page }) => {
    const requests = await prepareChat(page, {
      onImprove: async (route, request, attempt) => {
        if (attempt === 1) {
          await json(route, 503, {
            code: "provider_unavailable",
            message: `raw provider detail ${USER_DRAFT}`,
            retryable: true,
            request_id: "request-e2e-1",
          })
          return
        }
        await json(route, 200, improvementResponse(request))
      },
    })
    await page.getByTestId("chat-input").first().fill(USER_DRAFT)
    await openPromptActions(page)
    await page.getByRole("button", { name: /Review changes/ }).click()

    const failureAlert = page
      .getByRole("alert")
      .filter({ hasText: "prompt improvement service" })
    await expect(failureAlert).toHaveText(
      "The prompt improvement service is unavailable.",
    )
    await expect(failureAlert).not.toContainText("raw provider detail")
    await page.getByRole("button", { name: "Retry" }).click()
    await expect(
      page.getByRole("textbox", { name: "Improved prompt candidate" }),
    ).toHaveValue(USER_CANDIDATE)
    expect(requests).toHaveLength(2)
    expect(requests[1].operation_id).not.toBe(requests[0].operation_id)
  })

  for (const capability of ["false", "404", "offline"] as const) {
    test(`capability ${capability} fails closed without exposing Track B`, async ({ page }) => {
      const requests = await prepareChat(page, { capability })
      await page.getByTestId("chat-input").first().fill(USER_DRAFT)
      await openPromptActions(page)

      await expect(page.getByRole("button", { name: /Improve now/ })).toBeDisabled()
      await expect(page.getByRole("button", { name: /Review changes/ })).toBeDisabled()
      await expect(page.getByText("Build from recipe", { exact: true })).toHaveCount(0)
      await expect(
        page.getByText(
          capability === "false"
            ? "Prompt improvement requires a newer server version."
            : "Prompt improvement requires a newer server version.",
        ),
      ).toBeVisible()
      expect(requests).toHaveLength(0)
    })
  }

  test("Escape closes prompt surfaces and restores each owning focus target", async ({ page }) => {
    await prepareChat(page)
    await page.getByTestId("chat-input").first().fill(USER_DRAFT)
    const trigger = await openPromptActions(page)
    await page.keyboard.press("Escape")
    await expect(
      page.getByRole("group", { name: "Prompt improvement actions" }),
    ).toBeHidden()
    await expect(trigger).toBeFocused()

    await trigger.press("Enter")
    await page.getByRole("button", { name: /Review changes/ }).press("Enter")
    await expect(page.getByRole("dialog", { name: "Prompt improvement" })).toBeVisible()
    await page.keyboard.press("Escape")
    await expect(page.getByRole("dialog", { name: "Prompt improvement" })).toBeHidden()
    await expect(page.getByTestId("chat-input").first()).toBeFocused()
  })

  test("mobile review is full width, overflow-free, and passes a scoped axe scan", async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 })
    await prepareChat(page)
    await page.getByTestId("chat-input").first().fill(USER_DRAFT)
    await openPromptActions(page)
    await page.getByRole("button", { name: /Review changes/ }).click()

    const dialog = page.getByRole("dialog", { name: "Prompt improvement" })
    await expect(dialog).toBeVisible()
    await page.waitForFunction(() => {
      const wrapper = document.querySelector<HTMLElement>(
        ".ant-drawer-content-wrapper",
      )
      if (!wrapper) return false
      return (
        !wrapper.classList.contains(
          "ant-drawer-panel-motion-right-appear-active",
        ) && window.getComputedStyle(wrapper).transform === "none"
      )
    })
    const box = await dialog.boundingBox()
    expect(box).not.toBeNull()
    expect(box?.width).toBeGreaterThanOrEqual(385)
    await expectNoHorizontalOverflow(page, "mobile prompt review")

    const results = await new AxeBuilder({ page })
      .include("section[aria-label='Prompt improvement']")
      .analyze()
    await test.info().attach("prompt-review-axe-results", {
      body: JSON.stringify(results, null, 2),
      contentType: "application/json",
    })
    expect(results.violations).toEqual([])
  })
})

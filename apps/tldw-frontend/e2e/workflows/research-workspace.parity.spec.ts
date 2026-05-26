import { test, assertNoCriticalErrors } from "../utils/fixtures"
import { runResearchWorkspaceParityContract } from "../../../test-utils/research-workspace"

const DESKTOP_VIEWPORT = { width: 1440, height: 900 }

const EMPTY_LIST_RESPONSE = {
  status: 200,
  contentType: "application/json",
  body: JSON.stringify({ items: [], total: 0 })
}

test.describe("Research Workspace parity (WebUI)", () => {
  test.beforeEach(async ({ authedPage }) => {
    await authedPage.setViewportSize(DESKTOP_VIEWPORT)

    await authedPage.route("**/api/v1/chats**", async (route) => {
      await route.fulfill(EMPTY_LIST_RESPONSE)
    })

    await authedPage.route("**/api/v1/chats/**", async (route) => {
      await route.fulfill(EMPTY_LIST_RESPONSE)
    })

    await authedPage.route("**/api/v1/chat/conversations**", async (route) => {
      await route.fulfill(EMPTY_LIST_RESPONSE)
    })

    await authedPage.route("**/api/v1/chat/conversations/**", async (route) => {
      await route.fulfill(EMPTY_LIST_RESPONSE)
    })
  })

  test("passes baseline + deterministic studio parity contract", async ({
    authedPage,
    diagnostics
  }) => {
    await runResearchWorkspaceParityContract({
      platform: "web",
      page: authedPage
    })

    await assertNoCriticalErrors(diagnostics)
  })
})

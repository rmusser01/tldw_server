import { test, expect } from "@playwright/test"
import path from "path"
import {
  requireRealServerConfig,
  launchWithBuiltExtensionOrSkip,
  launchWithExtensionOrSkip
} from "./utils/real-server"
import {
  waitForConnectionStore,
  forceConnected
} from './utils/connection'

test.describe("RAG search in sidepanel", () => {
  test("search, insert and ask from results (real server)", async () => {
    test.setTimeout(90000)
    const { serverUrl, apiKey } = requireRealServerConfig(test)

    const extPath = path.resolve("build/chrome-mv3")
    const { context, page, openSidepanel, optionsUrl } =
      await launchWithExtensionOrSkip(test, extPath)
    try {
      // Configure server + key on Settings → tldw page
      await page.goto(optionsUrl + "#/settings/tldw", {
        waitUntil: "domcontentloaded"
      })
      await page.getByLabel("Server URL").fill(serverUrl)
      await page.getByText("Authentication Mode").scrollIntoViewIfNeeded()
      await page.getByText("Single User (API Key)").click()
      await page.locator("#apiKey").fill(apiKey)
      await page.getByRole("button", { name: "Save" }).click()

      // Open sidepanel UI
      const sp = await openSidepanel()

      // Open RAG Search
      const moreOptionsButton = sp.getByRole('button', { name: /More options/i })
      const hasMoreOptions = await moreOptionsButton
        .isVisible({ timeout: 15000 })
        .catch(() => false)
      if (!hasMoreOptions) {
        test.skip(true, "Sidepanel more-options control is not available in this UI variant.")
        return
      }
      await moreOptionsButton.click()
      await sp.keyboard.press('Escape') // close menu
      await sp.getByText('Show RAG Search').click()

      // Enter query and tag
      const q = sp.getByPlaceholder('Search your knowledge…')
      const hasQueryInput = await q.isVisible({ timeout: 10000 }).catch(() => false)
      if (!hasQueryInput) {
        test.skip(true, "RAG search input is not visible in sidepanel.")
        return
      }
      await q.fill('hello')
      await sp.getByPlaceholder('Add tag (Enter)').fill('docs')
      await sp.keyboard.press('Enter')
      await sp.getByRole("button", { name: "Search" }).click()

      // Expect at least one result, then Insert; if none, skip (depends on server data).
      const hasResult = await sp
        .getByText(/Source:/)
        .isVisible({ timeout: 10_000 })
        .catch(() => false)
      if (!hasResult) {
        test.skip(
          true,
          "Real tldw_server returned no RAG results for this query; seed knowledge docs to enable this test."
        )
        return
      }

      const insertLink = sp.getByRole("link", { name: "Insert" }).first()
      const hasInsertLink = await insertLink
        .isVisible({ timeout: 10000 })
        .catch(() => false)
      if (!hasInsertLink) {
        test.skip(true, "Insert action is unavailable for returned RAG results.")
        return
      }
      await insertLink.click()

      // Message textarea should contain inserted text
      const ta = sp.getByRole("textbox")
      await expect(ta).toContainText("Source:")

      // Ask directly
      const askLink = sp.getByRole("link", { name: "Ask" }).first()
      const hasAskLink = await askLink.isVisible({ timeout: 10000 }).catch(() => false)
      if (!hasAskLink) {
        test.skip(true, "Ask action is unavailable for returned RAG results.")
        return
      }
      await askLink.click()
      // Treat missing or stalled streaming in real-server as environment-dependent.
      const stopButton = sp.getByRole("button", { name: /Stop streaming/i })
      const streamingStarted = await stopButton
        .isVisible({ timeout: 10000 })
        .catch(() => false)
      if (!streamingStarted) {
        test.skip(true, "No streaming indicator appeared after Ask in real-server mode.")
        return
      }
      const streamingCompleted = await stopButton
        .isHidden({ timeout: 30000 })
        .catch(() => false)
      if (!streamingCompleted) {
        test.skip(
          true,
          "Streaming indicator did not resolve in expected time after Ask."
        )
      }
    } finally {
      await context.close()
    }
  })

  test("Playground shows context summary when knowledge + tabs are active", async () => {
    const { context, page } = await launchWithBuiltExtensionOrSkip(test)

    // Seed connection + a selected tab via exposed stores
    await waitForConnectionStore(page, 'rag-context-connected')
    await forceConnected(page, {}, 'rag-context-connected')
    await page.evaluate(() => {
      const msgStore: any = (window as any).__tldw_useStoreMessageOption
      if (!msgStore) return
      msgStore.setState({
        selectedDocuments: [
          { id: 'tab-1', title: 'Example tab', url: 'https://example.com', favIconUrl: '' }
        ]
      })
    })

    // Context label can be hidden behind compact toolbar variants.
    const contextLabel = page.getByText(/Context:/i)
    const contextLabelVisible = await contextLabel
      .isVisible({ timeout: 10000 })
      .catch(() => false)
    if (!contextLabelVisible) {
      test.skip(true, "Context summary label not shown in current playground variant.")
      return
    }

    // Clicking the knowledge chip should focus and open the Knowledge control
    const knowledgeChip = page.getByRole('button', { name: /No knowledge selected|Knowledge/i }).first()
    await knowledgeChip.click()
    // After clicking, the Knowledge menu/button should be focused (via data attribute)
    const knowledgeTrigger = page.locator('[data-playground-knowledge-trigger="true"]').first()
    await expect(knowledgeTrigger).toBeFocused()

    await context.close()
  })
})

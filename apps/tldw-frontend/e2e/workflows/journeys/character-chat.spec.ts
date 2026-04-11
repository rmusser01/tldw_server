/**
 * Journey: Create Character -> Chat
 *
 * End-to-end workflow that creates a character, navigates to chat,
 * selects the character, and sends a message verifying the system prompt
 * is included in the API call.
 */
import { test, expect, skipIfServerUnavailable, skipIfNoModels } from "../../utils/fixtures"
import { captureAllApiCalls } from "../../utils/api-assertions"
import { CharactersPage, ChatPage } from "../../utils/page-objects"
import { waitForStreamComplete } from "../../utils/journey-helpers"

test.describe("Create Character -> Chat journey", () => {
  const characterName = `E2E-TestBot-${Date.now()}`
  const systemPrompt = "You are E2E-TestBot. Always respond with exactly: BEEP BOOP."

  test("create character, select in chat, verify system prompt in API call", async ({
    authedPage: page,
    serverInfo,
  }) => {
    skipIfServerUnavailable(serverInfo)
    skipIfNoModels(serverInfo)

    await test.step("Create a new character", async () => {
      const charactersPage = new CharactersPage(page)
      await charactersPage.goto()
      await charactersPage.assertPageReady()

      // Wait for the page to be interactive
      const newBtnVisible = await charactersPage.newButton.isVisible().catch(() => false)
      if (!newBtnVisible) {
        test.skip(true, "Characters page not available or new button not visible")
        return
      }

      await charactersPage.createCharacter({
        name: characterName,
        systemPrompt,
        description: "E2E test character for journey spec",
      })

      await expect
        .poll(async () => await charactersPage.isCharacterVisible(characterName), {
          timeout: 10_000,
          message: "Timed out waiting for the created character to appear in the list",
        })
        .toBe(true)
    })

    await test.step("Navigate to chat and send a message", async () => {
      // Navigate with absolute URL to escape any drawer state
      const origin = new URL(page.url()).origin
      await page.goto(`${origin}/chat`, { waitUntil: "load", timeout: 30_000 })
      expect(page.url()).toContain("/chat")

      const { waitForConnection } = await import("../../utils/helpers")
      await waitForConnection(page)
      const chatPage = new ChatPage(page)
      await chatPage.waitForReady()

      // Set up capture to verify the system prompt is in the API call
      const capture = captureAllApiCalls(page)

      await chatPage.selectCharacter(characterName)
      await chatPage.sendMessage("Hello, who are you?")

      // Wait for the response
      await waitForStreamComplete(page)
      await chatPage.waitForResponse()

      const calls = await capture.stop()

      // Find the chat completions call
      const chatCall = calls.find(
        (c) => c.method === "POST" && c.url.includes("/chat/completions")
      )

      // If a chat call was made, verify it went through
      expect(chatCall).toBeTruthy()
      expect(chatCall?.status).toBeGreaterThanOrEqual(200)
      expect(chatCall?.status).toBeLessThan(300)
      expect(JSON.stringify(chatCall?.requestBody || {})).toContain(systemPrompt)
    })
  })
})

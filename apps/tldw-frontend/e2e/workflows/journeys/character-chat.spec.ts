/**
 * Journey: Create Character -> Chat
 *
 * End-to-end workflow that creates a character, navigates to chat,
 * selects the character, and sends a message through the real character
 * chat complete-v2 backend path.
 */
import { test, expect, skipIfServerUnavailable, skipIfNoModels } from "../../utils/fixtures"
import { captureAllApiCalls } from "../../utils/api-assertions"
import { CharactersPage, ChatPage } from "../../utils/page-objects"
import { waitForStreamComplete } from "../../utils/journey-helpers"

test.describe("Create Character -> Chat journey", () => {
  const characterName = `E2E-TestBot-${Date.now()}`
  const systemPrompt = "You are E2E-TestBot. Always respond with exactly: BEEP BOOP."

  test("create character, select in chat, verify complete-v2 character stream path", async ({
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

      // Set up capture to verify the character chat backend path and payload.
      const capture = captureAllApiCalls(page)

      await chatPage.selectCharacter(characterName)
      await chatPage.sendMessage("Hello, who are you?")

      // Wait for the response
      await waitForStreamComplete(page)
      await chatPage.waitForResponse()

      const calls = await capture.stop()

      const chatCreateCall = calls.find((c) => {
        const url = new URL(c.url)
        return c.method === "POST" && url.pathname === "/api/v1/chats/"
      })
      const characterCompleteCall = calls.find((c) => {
        const url = new URL(c.url)
        return (
          c.method === "POST" &&
          /^\/api\/v1\/chats\/[^/]+\/complete-v2$/.test(url.pathname)
        )
      })

      expect(chatCreateCall).toBeTruthy()
      expect(JSON.stringify(chatCreateCall?.requestBody || {})).toContain("character_id")

      expect(characterCompleteCall).toBeTruthy()
      expect(characterCompleteCall?.status).toBeGreaterThanOrEqual(200)
      expect(characterCompleteCall?.status).toBeLessThan(300)
      expect(characterCompleteCall?.requestBody).toMatchObject(
        expect.objectContaining({
          include_character_context: true,
          stream: true,
        })
      )
    })
  })
})

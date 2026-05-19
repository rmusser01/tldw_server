/**
 * Journey: Create Character -> Chat
 *
 * End-to-end workflow that creates a character, navigates to chat,
 * selects the character, and sends a message through the real character
 * chat complete-v2 backend path. When the real backend lacks provider
 * credentials, the journey verifies the visible recovery state instead of
 * treating catalog-only model inventory as a successful stream.
 */
import { test, expect, skipIfServerUnavailable } from "../../utils/fixtures"
import { captureAllApiCalls } from "../../utils/api-assertions"
import { CharactersPage, ChatPage } from "../../utils/page-objects"
import { waitForStreamComplete } from "../../utils/journey-helpers"

const getErrorCode = (body: unknown): string | null => {
  if (!body || typeof body !== "object") return null
  const record = body as Record<string, any>
  return (
    record.error_code ??
    record.code ??
    record.detail?.error_code ??
    record.detail?.code ??
    record.detail?.error?.code ??
    null
  )
}

test.describe("Create Character -> Chat journey", () => {
  const characterName = `E2E-TestBot-${Date.now()}`
  const systemPrompt = "You are E2E-TestBot. Always respond with exactly: BEEP BOOP."

  test("create character, select in chat, verify complete-v2 character stream path", async ({
    authedPage: page,
    serverInfo,
  }) => {
    skipIfServerUnavailable(serverInfo)

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
      expect(chatCreateCall?.status).toBeGreaterThanOrEqual(200)
      expect(chatCreateCall?.status).toBeLessThan(300)
      expect(chatCreateCall?.requestBody).toEqual(
        expect.objectContaining({
          character_id: expect.anything(),
        })
      )

      expect(characterCompleteCall).toBeTruthy()
      expect(characterCompleteCall?.requestBody).toMatchObject(
        expect.objectContaining({
          include_character_context: true,
          stream: true,
        })
      )

      const completeStatus = characterCompleteCall?.status ?? 0
      expect(completeStatus).toBeGreaterThanOrEqual(200)

      if (completeStatus >= 300) {
        expect(getErrorCode(characterCompleteCall?.responseBody)).toBe(
          "missing_provider_credentials"
        )
        await expect(
          page.getByText(/something went wrong while talking to your tldw server/i)
        ).toBeVisible()
        await expect(page.getByRole("button", { name: /retry same model/i })).toBeVisible()
        await expect(page.getByRole("button", { name: /switch model/i })).toBeVisible()
      } else {
        expect(completeStatus).toBeLessThan(300)
      }
    })
  })
})

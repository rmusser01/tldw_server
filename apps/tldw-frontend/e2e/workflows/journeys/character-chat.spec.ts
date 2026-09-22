/**
 * Seeded engineering regression: create TestBot -> complete-v2 -> persisted reload.
 * Controlled downstream inference does not certify real-model or fresh-install UAT.
 * Provider/configuration recovery is exercised separately in Phase 7.
 */
import { randomUUID } from "node:crypto"
import { test, expect } from "../../utils/fixtures"
import { CharactersPage, ChatPage } from "../../utils/page-objects"
import { waitForStreamComplete } from "../../utils/journey-helpers"
import { TEST_CONFIG, fetchWithApiKey, waitForConnection } from "../../utils/helpers"

const SYSTEM = "You are E2E-TestBot. Always respond with exactly: BEEP BOOP."
const QUESTION = "Hello, who are you?"
const ANSWER = "BEEP BOOP."

test.describe("Create Character -> Chat journey", () => {
  test("saves the selected character and its successful complete-v2 turn across reload", async ({
    authedPage: page, serverInfo,
  }, testInfo) => {
    test.setTimeout(180_000)
    expect(serverInfo.available, "The real application backend is required").toBe(true)
    expect(process.env.UAT_CHARACTER_MODE).toBe("deterministic")
    const provider = process.env.UAT_CHARACTER_PROVIDER
    const model = process.env.UAT_CHARACTER_MODEL
    expect(provider).toBeTruthy()
    expect(model).toBeTruthy()
    await page.unrouteAll({ behavior: "wait" })
    const characterName = `E2E-TestBot-${randomUUID()}`
    const evidence: Record<string, unknown> = { characterName, mode: "deterministic", auth: "seeded" }
    const apiGet = async (path: string) => {
      const response = await fetchWithApiKey(`${TEST_CONFIG.serverUrl}${path}`)
      expect(response.ok, `Canonical GET ${path}: HTTP ${response.status}`).toBe(true)
      return response.json()
    }
    try {
      const characters = new CharactersPage(page)
      await characters.goto()
      await characters.assertPageReady()
      await expect(characters.newButton).toBeVisible()
      const createdCharacter = page.waitForResponse(response =>
        response.request().method() === "POST" &&
        /\/api\/v1\/characters\/?$/.test(response.url()) &&
        response.request().postDataJSON().name === characterName
      )
      const [creation] = await Promise.all([
        createdCharacter,
        characters.createCharacter({ name: characterName, systemPrompt: SYSTEM, description: "E2E test character for journey spec" }),
      ])
      expect(creation.ok()).toBe(true)
      const character = await creation.json()
      expect(character).toMatchObject({ name: characterName, system_prompt: SYSTEM })
      expect(Number.isInteger(character.id) && character.id > 0).toBe(true)
      evidence.characterId = character.id
      await expect.poll(() => characters.isCharacterVisible(characterName)).toBe(true)
      await page.goto("/chat", { waitUntil: "domcontentloaded" })
      await waitForConnection(page)
      const chat = new ChatPage(page)
      await chat.waitForReady()
      const createdChat = page.waitForResponse(response =>
        response.request().method() === "POST" && /\/api\/v1\/chats\/?$/.test(response.url())
      )
      await chat.selectCharacter(characterName)
      await chat.selectModel(model!)
      const completed = page.waitForResponse(response =>
        response.request().method() === "POST" && /\/api\/v1\/chats\/[^/]+\/complete-v2$/.test(new URL(response.url()).pathname)
      )
      const [[chatCreation, completion]] = await Promise.all([
        Promise.all([createdChat, completed]), chat.sendMessage(QUESTION),
      ])
      expect(chatCreation.ok()).toBe(true)
      expect(String(chatCreation.request().postDataJSON().character_id)).toBe(String(character.id))
      const conversation = await chatCreation.json()
      const chatId = conversation.id
      expect(typeof chatId).toBe("string")
      expect(chatId).not.toMatch(/^(?:local[-_]|$)/)
      expect(conversation.character_id).toBe(character.id)
      expect(completion.status()).toBe(200)
      const sent = completion.request().postDataJSON()
      expect(sent).toMatchObject({
        include_character_context: true, stream: true, save_to_db: true,
        model: `${provider}:${model}`, provider,
      })
      evidence.chatId = chatId
      evidence.completionRequest = sent
      await waitForStreamComplete(page, 90_000)
      await chat.waitForResponse()
      expect((await chat.getMessages()).filter(message => message.role === "assistant").at(-1)?.content).toBe(ANSWER)
      const saved = await apiGet(`/api/v1/chats/${chatId}/messages?render_placeholders=false`)
      expect(saved.messages).toHaveLength(2)
      expect(saved.messages.map((message: { content: string }) => message.content.trim())).toEqual([QUESTION, ANSWER])
      expect(saved.messages[0].sender.toLowerCase()).toBe("user")
      expect(["assistant", characterName.toLowerCase()]).toContain(saved.messages[1].sender.toLowerCase())
      expect(saved.messages.every((message: { id: string; conversation_id: string }) =>
        typeof message.id === "string" && message.id.length > 0 && message.conversation_id === chatId
      )).toBe(true)
      expect(new Set(saved.messages.map((message: { id: string }) => message.id)).size).toBe(2)
      evidence.messages = saved.messages
      await page.goto(`/chat?settingsServerChatId=${encodeURIComponent(chatId)}`, { waitUntil: "domcontentloaded" })
      await waitForConnection(page)
      await chat.waitForReady()
      await expect.poll(async () => (await chat.getMessages())
        .filter(message => message.role === "assistant").at(-1)?.content).toBe(ANSWER)
      expect((await apiGet(`/api/v1/chats/${chatId}`)).character_id).toBe(character.id)
      expect((await apiGet(`/api/v1/chats/${chatId}/messages?render_placeholders=false`)).messages).toEqual(saved.messages)
      expect(await apiGet(`/api/v1/characters/${character.id}`)).toMatchObject({
        id: character.id, name: characterName, system_prompt: SYSTEM,
      })
    } finally {
      await testInfo.attach("character-chat-evidence.json", {
        body: JSON.stringify(evidence, null, 2), contentType: "application/json",
      })
    }
  })
})

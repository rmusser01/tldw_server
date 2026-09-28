/**
 * Seeded engineering regression: create TestBot -> versioned native completion -> persisted reload.
 * Controlled downstream inference does not certify real-model or fresh-install UAT.
 * Provider/configuration recovery is exercised separately in Phase 7.
 */
import { randomUUID } from "node:crypto"
import { readStore } from "../../../../extension/tests/e2e/utils/history-selection"
import type { HistoryBookmark } from "../../../../packages/ui/src/db/dexie/types"

import { test, expect } from "../../utils/fixtures"
import { CharactersPage, ChatPage } from "../../utils/page-objects"
import { waitForStreamComplete } from "../../utils/journey-helpers"
import { TEST_CONFIG, fetchWithApiKey, waitForConnection } from "../../utils/helpers"

const SYSTEM = "You are E2E-TestBot. Always respond with exactly: BEEP BOOP."
const QUESTION = "Hello, who are you?"
const ANSWER = "BEEP BOOP."

type CanonicalTurnMessage = { id: string; conversation_id: string; content: string; sender: string; parent_message_id: string | null }

test.describe("Create Character -> Chat journey", () => {
  test("saves the selected character and its successful native turn across reload", async ({
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
        response.request().method() === "POST" && /^\/api\/v1\/chat\/completions$/.test(new URL(response.url()).pathname)
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
      expect(sent).toMatchObject({ stream: true, save_to_db: true, conversation_id: chatId,
        model, api_provider: provider, tldw_history_selection_v1: { version: 1, conversation_id: chatId } })
      expect(sent.messages).toEqual([{ role: "user", content: QUESTION }])
      evidence.chatId = chatId
      evidence.completionRequest = sent
      await waitForStreamComplete(page, 90_000)
      await chat.waitForResponse()
      const question = QUESTION
      const answer = ANSWER
      expect((await chat.getMessages()).filter(row => row.role === "assistant").map(row => row.content)).toEqual([answer])

      const frames = (await completion.text()).split("\n")
        .filter(line => line.startsWith("data: ") && line.slice(6) !== "[DONE]")
        .map(line => JSON.parse(line.slice(6)))
      const admission = frames.find(frame => frame.tldw_history_admission_v1)?.tldw_history_admission_v1
      const settlement = frames.find(frame => frame.tldw_message_id)
      expect(admission).toMatchObject({ version: 1, conversation_id: chatId,
        owner_key: sent.tldw_history_selection_v1.owner_key,
        selection_digest: sent.tldw_history_selection_v1.selection_digest,
        messages: sent.tldw_history_selection_v1.messages,
        originating_selection_revision: sent.tldw_history_selection_v1.selection_revision })
      expect(admission.owner_key).toMatch(/^native-history-v1:sha256:/)
      expect(admission.input_message_id).toEqual(expect.any(String))
      expect(admission.input_message_id).not.toBe("")
      expect(settlement).toMatchObject({ tldw_conversation_id: chatId,
        tldw_user_message_id: admission.input_message_id, tldw_message_id: expect.any(String) })
      expect(settlement.tldw_message_id).not.toBe("")
      expect(settlement.tldw_message_id).not.toBe(admission.input_message_id)
      expect(frames.map(frame => frame.choices?.[0]?.delta?.content ?? "").join("")).toBe(answer)
      let saved: CanonicalTurnMessage[] = []
      await expect.poll(async () => { saved = (await apiGet(`/api/v1/chats/${chatId}/messages?render_placeholders=false`)).messages; return saved.length }).toBe(2)
      const user = saved.find(row => row.id === admission.input_message_id)!
      const assistant = saved.find(row => row.id === settlement.tldw_message_id)!
      expect(user).toMatchObject({ content: question, sender: "user", conversation_id: chatId, parent_message_id: null })
      expect(assistant).toMatchObject({ content: answer, conversation_id: chatId, parent_message_id: user.id })
      expect(["assistant", character.name.toLowerCase()]).toContain(assistant.sender.toLowerCase())
      expect(new Set(saved.map(row => row.id)).size).toBe(2)
      let bookmarks: HistoryBookmark[] = []
      await expect.poll(async () => {
        bookmarks = ((await readStore(page, "historySelections")) as HistoryBookmark[]).filter(row =>
          row.conversation_id === chatId && row.owner_key === admission.owner_key &&
          row.view.cursor.kind === "after_message" && row.view.cursor.message_id === assistant.id)
        return bookmarks.length
      }).toBe(1)
      const bookmark = bookmarks[0]
      expect(bookmark.view.interpretation).toEqual({ kind: "parent_graph_v1" })
      const reference = { profile_id: bookmark.profile_id, client_session_id: bookmark.client_session_id,
        owner_key: admission.owner_key, conversation_id: chatId, owner_kind: "native" }
      await page.goto("/chat?historySelection=" + encodeURIComponent(JSON.stringify(reference)), { waitUntil: "domcontentloaded" })
      await waitForConnection(page)
      await chat.waitForReady()
      await expect.poll(async () => (await chat.getMessages()).filter(row => row.role === "assistant").map(row => row.content)).toEqual([answer])
      await expect(page).toHaveURL(/\/chat$/)
      await page.reload({ waitUntil: "domcontentloaded" })
      await waitForConnection(page)
      await chat.waitForReady()
      await expect.poll(async () => (await chat.getMessages()).filter(row => row.role === "assistant").map(row => row.content)).toEqual([answer])
      expect((await apiGet(`/api/v1/chats/${chatId}/messages?render_placeholders=false`)).messages).toEqual(saved)
      evidence.messages = saved
      evidence.admission = admission
      evidence.settlement = settlement
      evidence.selection = reference
      expect((await apiGet(`/api/v1/chats/${chatId}`)).character_id).toBe(character.id)
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

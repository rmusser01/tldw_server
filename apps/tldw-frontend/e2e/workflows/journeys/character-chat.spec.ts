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
import { parseRoleplayResumeAppendRequest, parseRoleplayResumeAppendReceipt, parseRoleplayResumeCompletionRequest, parseRoleplayResumeCompletionReceipt } from "../../../../packages/ui/src/services/tldw/roleplay-resume"
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
      const capturedSelection = page.waitForResponse(response =>
        response.request().method() === "POST" && /\/api\/v1\/chat\/conversations\/[^/]+\/history\/selection$/.test(new URL(response.url()).pathname)
      )
      await chat.selectModel(model!)
      await chat.selectCharacter(characterName)
      const admitted = page.waitForResponse(response =>
        response.request().method() === "POST" && /\/api\/v1\/chats\/[^/]+\/roleplay\/messages$/.test(new URL(response.url()).pathname)
      )
      const completed = page.waitForResponse(response =>
        response.request().method() === "POST" && /\/api\/v1\/chats\/[^/]+\/complete-v2$/.test(new URL(response.url()).pathname)
      )
      const [[chatCreation, capture, appendResponse, completion]] = await Promise.all([
        Promise.all([createdChat, capturedSelection, admitted, completed]), chat.sendMessage(QUESTION),
      ])
      expect(chatCreation.ok()).toBe(true)
      expect(String(chatCreation.request().postDataJSON().character_id)).toBe(String(character.id))
      const conversation = await chatCreation.json()
      const chatId = conversation.id
      expect(typeof chatId).toBe("string")
      expect(chatId).not.toMatch(/^(?:local[-_]|$)/)
      expect(conversation.character_id).toBe(character.id)
      expect(completion.status()).toBe(200)
      const selected = await capture.json()
      expect(selected.view).toMatchObject({ owner_key: expect.stringMatching(/^native-history-v1:sha256:/), conversation_id: chatId, cursor: { kind: "empty" } })
      expect(selected.snapshot).toMatchObject({ owner_key: selected.view.owner_key, conversation_id: chatId })
      expect(selected.snapshot.source_digest).toMatch(/^sha256:[0-9a-f]{64}$/)
      const ownerKey = selected.view.owner_key
      expect(appendResponse.status()).toBe(201)
      expect(new URL(appendResponse.url()).pathname).toBe(`/api/v1/chats/${chatId}/roleplay/messages`)
      const input = parseRoleplayResumeAppendRequest(appendResponse.request().postDataJSON())
      expect(input.content).toBe(QUESTION)
      expect(input.expected.tail).toEqual({ message_id: null, message_version: null })
      const admission = parseRoleplayResumeAppendReceipt(chatId, input, await appendResponse.json())
      expect(new URL(completion.url()).pathname).toBe(`/api/v1/chats/${chatId}/complete-v2`)
      const sent = parseRoleplayResumeCompletionRequest(completion.request().postDataJSON())
      expect(sent).toMatchObject({ stream: false, save_to_db: true, include_character_context: true, append_user_message: false,
        tldw_roleplay_resume_v1: { input_message_id: admission.message_id, input_message_version: admission.message_version, expected: admission.fence } })
      const settlement = parseRoleplayResumeCompletionReceipt(chatId, sent, await completion.json())
      expect(settlement).toMatchObject({ character_id: character.id, provider, model, saved: true, assistant_content: ANSWER })
      evidence.chatId = chatId
      evidence.completionRequest = sent
      evidence.originalSelection = selected.view
      await waitForStreamComplete(page, 90_000)
      await chat.waitForResponse()
      const question = QUESTION
      const answer = ANSWER
      expect((await chat.getMessages()).filter(row => row.role === "assistant").map(row => row.content)).toEqual([answer])
      let saved: CanonicalTurnMessage[] = []
      await expect.poll(async () => { saved = (await apiGet(`/api/v1/chats/${chatId}/messages?render_placeholders=false`)).messages; return saved.length }).toBe(2)
      const user = saved.find(row => row.id === admission.message_id)!
      const assistant = saved.find(row => row.id === settlement.assistant_message_id)!
      expect(user).toMatchObject({ content: question, sender: "user", conversation_id: chatId, parent_message_id: null })
      expect(assistant).toMatchObject({ content: answer, conversation_id: chatId, parent_message_id: user.id })
      expect(["assistant", character.name.toLowerCase()]).toContain(assistant.sender.toLowerCase())
      expect(new Set(saved.map(row => row.id)).size).toBe(2)
      let bookmarks: HistoryBookmark[] = []
      await expect.poll(async () => {
        bookmarks = ((await readStore(page, "historySelections")) as HistoryBookmark[]).filter(row =>
          row.conversation_id === chatId && row.owner_key === ownerKey &&
          row.view.cursor.kind === "after_message" && row.view.cursor.message_id === assistant.id)
        return bookmarks.length
      }).toBe(1)
      const bookmark = bookmarks[0]
      expect(bookmark.view.interpretation).toEqual({ kind: "parent_graph_v1" })
      const reference = { profile_id: bookmark.profile_id, client_session_id: bookmark.client_session_id,
        owner_key: ownerKey, conversation_id: chatId, owner_kind: "native" }
      await page.goto("/chat?historySelection=" + encodeURIComponent(JSON.stringify(reference)), { waitUntil: "domcontentloaded" })
      await waitForConnection(page)
      await chat.waitForReady()
      await expect.poll(async () => (await chat.getMessages()).filter(row => row.role === "assistant").map(row => row.content)).toEqual([answer])
      // The handoff is consumed; a saved chat may now carry its own address (XP-05).
      await expect(page).toHaveURL(/\/chat(?:\?chat=[^&#]+)?$/)
      await expect(page).not.toHaveURL(/historySelection=/)
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

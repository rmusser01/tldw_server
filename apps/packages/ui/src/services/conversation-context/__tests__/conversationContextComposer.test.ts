import { describe, expect, it, vi } from "vitest"

import { composeConversationContext } from "../conversationContextComposer"
import type { ConversationContextPrimitiveClient } from "../conversationContextComposer"

const buildPrimitives = (
  overrides: Partial<ConversationContextPrimitiveClient> = {}
): ConversationContextPrimitiveClient => ({
  processDictionary: vi.fn(async ({ text }) => ({
    original_text: text,
    processed_text: text,
    replacements: 0,
    iterations: 0,
    entries_used: []
  })),
  processWorldBookContext: vi.fn(async () => ({
    injected_content: "",
    entries_matched: 0,
    tokens_used: 0,
    books_used: 0,
    entry_ids: [],
    diagnostics: []
  })),
  ...overrides
})

describe("conversation context composer", () => {
  it("composes blank chat with dictionary before worldbook matching", async () => {
    const calls: string[] = []
    const primitives = buildPrimitives({
      processDictionary: vi.fn(async ({ text, dictionary_ids }) => {
        calls.push(`dictionary:${text}:${dictionary_ids?.join(",")}`)
        return {
          original_text: text,
          processed_text: "Echo Vault",
          replacements: 1,
          iterations: 1,
          entries_used: [7]
        }
      }),
      processWorldBookContext: vi.fn(async ({ text, world_book_ids }) => {
        calls.push(`worldbook:${text}:${world_book_ids?.join(",")}`)
        return {
          injected_content: "Echo Vault sits below the old rail station.",
          entries_matched: 1,
          tokens_used: 9,
          books_used: 1,
          entry_ids: [11],
          diagnostics: [
            {
              entry_id: 11,
              world_book_id: 3,
              activation_reason: "keyword_match",
              keyword: "Echo Vault"
            }
          ]
        }
      })
    })

    const composition = await composeConversationContext({
      inputText: "EV",
      selection: {
        characterId: null,
        worldBookIds: [3],
        dictionaryIds: [7]
      },
      primitives
    })

    expect(calls).toEqual(["dictionary:EV:7", "worldbook:Echo Vault:3"])
    expect(composition.transformedInputText).toBe("Echo Vault")
    expect(composition.pieces).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          kind: "dictionary",
          id: 7,
          source: "explicit_chat",
          status: "active"
        }),
        expect.objectContaining({
          kind: "worldbook",
          id: 3,
          source: "explicit_chat",
          status: "matched"
        })
      ])
    )
  })

  it("keeps worldbooks and dictionaries conversation-scoped without a character", async () => {
    const primitives = buildPrimitives({
      processDictionary: vi.fn(async ({ text }) => ({
        original_text: text,
        processed_text: "normalized lore",
        replacements: 1,
        iterations: 1,
        entries_used: [2]
      })),
      processWorldBookContext: vi.fn(async () => ({
        injected_content: "Conversation lore",
        entries_matched: 1,
        tokens_used: 2,
        books_used: 1,
        entry_ids: [5],
        diagnostics: [{ entry_id: 5, world_book_id: 1 }]
      }))
    })

    const composition = await composeConversationContext({
      inputText: "draft",
      selection: {
        characterId: null,
        worldBookIds: [1],
        dictionaryIds: [2]
      },
      primitives
    })

    expect(composition.selection.characterId).toBeNull()
    expect(composition.pieces.every((piece) => piece.kind !== "character")).toBe(
      true
    )
    expect(composition.pieces).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ kind: "worldbook", source: "explicit_chat" }),
        expect.objectContaining({ kind: "dictionary", source: "explicit_chat" })
      ])
    )
  })

  it("uses the same composed content for preview sections and provider messages", async () => {
    const primitives = buildPrimitives({
      processWorldBookContext: vi.fn(async () => ({
        injected_content: "Shared lore for preview and send.",
        entries_matched: 1,
        tokens_used: 6,
        books_used: 1,
        entry_ids: [9],
        diagnostics: [{ entry_id: 9, world_book_id: 4 }]
      }))
    })

    const composition = await composeConversationContext({
      inputText: "Tell me about the vault.",
      selection: {
        worldBookIds: [4],
        dictionaryIds: []
      },
      primitives
    })

    expect(composition.previewSections).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          name: "Worldbooks",
          content: "Shared lore for preview and send."
        })
      ])
    )
    expect(composition.providerMessages).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          role: "system",
          content: expect.stringContaining("Shared lore for preview and send.")
        })
      ])
    )
  })

  it("labels explicit chat and character-inherited worldbooks separately", async () => {
    const primitives = buildPrimitives({
      processWorldBookContext: vi.fn(async () => ({
        injected_content: "Explicit and inherited lore.",
        entries_matched: 2,
        tokens_used: 7,
        books_used: 2,
        entry_ids: [12, 13],
        diagnostics: [
          { entry_id: 12, world_book_id: 1 },
          { entry_id: 13, world_book_id: 2 }
        ]
      }))
    })

    const composition = await composeConversationContext({
      inputText: "Lore",
      selection: {
        characterId: 10,
        worldBookIds: [1],
        dictionaryIds: []
      },
      inheritedWorldBookIds: [2],
      primitives
    })

    expect(composition.pieces).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          kind: "worldbook",
          id: 1,
          source: "explicit_chat"
        }),
        expect.objectContaining({
          kind: "worldbook",
          id: 2,
          source: "character_inherited"
        })
      ])
    )
  })
})

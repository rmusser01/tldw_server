import { describe, expect, it } from "vitest"

import {
  buildConversationContextSettingsPatch,
  resolveConversationContextSelection
} from "../conversationContextSettings"

describe("conversation context settings", () => {
  it("reads nested context settings before legacy dictionary aliases", () => {
    const selection = resolveConversationContextSelection({
      settings: {
        conversationContext: {
          world_book_ids: [1, "2", 0, "bad", 2],
          chat_dictionary_ids: [7, "8"]
        },
        chat_dictionary_ids: [99]
      }
    })

    expect(selection.worldBookIds).toEqual([1, 2])
    expect(selection.dictionaryIds).toEqual([7, 8])
  })

  it("falls back to legacy chat_dictionary_ids when nested dictionaries are absent", () => {
    const selection = resolveConversationContextSelection({
      settings: {
        chat_dictionary_ids: [3, "4", 3]
      }
    })

    expect(selection.worldBookIds).toEqual([])
    expect(selection.dictionaryIds).toEqual([3, 4])
  })

  it("merges route seed ids before persisted explicit-chat settings", () => {
    const selection = resolveConversationContextSelection({
      seed: {
        worldBookIds: [9, 1],
        dictionaryIds: [8]
      },
      settings: {
        conversationContext: {
          world_book_ids: [1, 2],
          chat_dictionary_ids: [8, 7]
        }
      }
    })

    expect(selection.worldBookIds).toEqual([9, 1, 2])
    expect(selection.dictionaryIds).toEqual([8, 7])
  })

  it("writes nested canonical settings and the dictionary compatibility mirror", () => {
    expect(
      buildConversationContextSettingsPatch({
        worldBookIds: [2, 1],
        dictionaryIds: [7, 42]
      })
    ).toEqual({
      conversationContext: {
        world_book_ids: [2, 1],
        chat_dictionary_ids: [7, 42]
      },
      chat_dictionary_ids: [7, 42]
    })
  })
})

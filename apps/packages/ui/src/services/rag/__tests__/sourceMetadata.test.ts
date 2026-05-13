import { describe, expect, it } from "vitest"

import {
  ALL_RAG_SOURCES,
  getRagSourceDescription,
  getRagSourceLabel,
  getRagSourceOptions,
  getRagSourceTranslationKey,
  isRagSource,
} from "../sourceMetadata"

describe("sourceMetadata", () => {
  it("exposes the canonical Knowledge QA source set in product order", () => {
    expect(ALL_RAG_SOURCES).toEqual([
      "media_db",
      "notes",
      "chats",
      "characters",
      "kanban",
      "prompts",
      "world_books",
      "dictionaries",
    ])
  })

  it("provides translation keys alongside translated source options", () => {
    const translate = (key: string, fallback: string) => `${key}:${fallback}`

    const options = getRagSourceOptions(translate)

    expect(options).toHaveLength(ALL_RAG_SOURCES.length)
    expect(options[0]).toMatchObject({
      value: ALL_RAG_SOURCES[0],
      translationKey: getRagSourceTranslationKey(ALL_RAG_SOURCES[0]),
      label: `${getRagSourceTranslationKey(ALL_RAG_SOURCES[0])}:Documents & Media`
    })
  })

  it("falls back to English labels when no translator is provided", () => {
    const options = getRagSourceOptions()

    expect(options[0]).toMatchObject({
      value: "media_db",
      label: "Documents & Media",
      translationKey: "sidepanel:rag.sources.media"
    })
  })

  it("provides readable labels and descriptions for every canonical source", () => {
    expect(ALL_RAG_SOURCES.map(getRagSourceLabel)).toEqual([
      "Documents & Media",
      "Notes",
      "Chats",
      "Characters",
      "Task Boards",
      "Prompts",
      "World Books",
      "Dictionaries",
    ])

    for (const source of ALL_RAG_SOURCES) {
      expect(getRagSourceDescription(source).length).toBeGreaterThan(12)
      expect(isRagSource(source)).toBe(true)
    }

    expect(isRagSource("character_cards")).toBe(false)
    expect(isRagSource("chat_history")).toBe(false)
  })
})

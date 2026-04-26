import { describe, expect, it } from "vitest"

import {
  ALL_RAG_SOURCES,
  getRagSourceOptions,
  getRagSourceTranslationKey,
} from "../sourceMetadata"

describe("sourceMetadata", () => {
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
})

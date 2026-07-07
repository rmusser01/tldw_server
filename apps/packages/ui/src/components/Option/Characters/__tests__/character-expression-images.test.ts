import { describe, expect, it } from "vitest"
import { DEFAULT_CHARACTER_PROMPT_PRESET } from "@/data/character-prompt-presets"
import {
  EXPRESSION_IMAGE_STARTER_STATES,
  expressionRowsFromExtensions,
  expressionRowsToMoodImages,
  normalizeExpressionImageRows
} from "../character-expression-images"
import { applyCharacterMetadataToExtensions } from "../utils"

describe("character expression image rows", () => {
  it("creates starter rows plus configured custom rows", () => {
    const rows = expressionRowsFromExtensions({
      tldw: {
        mood_images: {
          happy: "https://example.test/happy.png",
          smirk: "https://example.test/smirk.png"
        }
      }
    })

    expect(rows.map((row) => row.state)).toEqual([
      ...EXPRESSION_IMAGE_STARTER_STATES,
      "smirk"
    ])
    expect(rows.find((row) => row.state === "happy")?.starter).toBe(true)
    expect(rows.find((row) => row.state === "happy")?.image.url).toBe(
      "https://example.test/happy.png"
    )
    expect(rows.find((row) => row.state === "smirk")?.image.url).toBe(
      "https://example.test/smirk.png"
    )
  })

  it("loads legacy mood image metadata", () => {
    const rows = expressionRowsFromExtensions({
      tldw: {
        moodImages: {
          thinking: "https://example.test/thinking.png",
          "soft smile": "https://example.test/soft-smile.png"
        }
      }
    })

    expect(rows.find((row) => row.state === "thinking")?.image.url).toBe(
      "https://example.test/thinking.png"
    )
    expect(rows.at(-1)).toMatchObject({
      state: "soft-smile",
      starter: false
    })
  })

  it("blocks duplicate and incomplete custom rows", () => {
    const result = normalizeExpressionImageRows([
      {
        id: "1",
        state: "happy",
        image: {
          mode: "url",
          url: "https://example.test/happy.png",
          base64: ""
        },
        starter: true
      },
      {
        id: "2",
        state: "happy",
        image: {
          mode: "url",
          url: "https://example.test/other.png",
          base64: ""
        },
        starter: false
      },
      {
        id: "3",
        state: "smirk",
        image: { mode: "url", url: "", base64: "" },
        starter: false
      }
    ])

    expect(result.errors).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ id: "2", reason: "duplicate" }),
        expect.objectContaining({ id: "3", reason: "missing-image" })
      ])
    )
  })

  it("blocks invalid states and empty custom rows", () => {
    const result = normalizeExpressionImageRows([
      {
        id: "bad-state",
        state: "../../bad",
        image: {
          mode: "url",
          url: "https://example.test/bad.png",
          base64: ""
        },
        starter: false
      },
      {
        id: "empty",
        state: "",
        image: { mode: "url", url: "", base64: "" },
        starter: false
      }
    ])

    expect(result.errors).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ id: "bad-state", reason: "invalid-state" }),
        expect.objectContaining({ id: "empty", reason: "missing-state" }),
        expect.objectContaining({ id: "empty", reason: "missing-image" })
      ])
    )
  })

  it("blocks image sources that would be dropped by metadata persistence", () => {
    const result = normalizeExpressionImageRows([
      {
        id: "relative",
        state: "smirk",
        image: {
          mode: "url",
          url: "/foo.png",
          base64: ""
        },
        starter: false
      }
    ])

    expect(result.rows).toEqual([])
    expect(result.errors).toEqual([
      expect.objectContaining({ id: "relative", reason: "invalid-image" })
    ])
  })

  it("drops empty starter rows and returns mood image map", () => {
    expect(
      expressionRowsToMoodImages([
        {
          id: "neutral",
          state: "neutral",
          image: { mode: "url", url: "", base64: "" },
          starter: true
        },
        {
          id: "thinking",
          state: "thinking",
          image: {
            mode: "url",
            url: "https://example.test/thinking.png",
            base64: ""
          },
          starter: true
        }
      ])
    ).toEqual({ thinking: "https://example.test/thinking.png" })
  })

  it("preserves invalid raw extensions when only empty starter rows exist", () => {
    const rawExtensions = "{not valid json"
    const result = applyCharacterMetadataToExtensions(rawExtensions, {
      preset: DEFAULT_CHARACTER_PROMPT_PRESET,
      expressionRows: expressionRowsFromExtensions({})
    })

    expect(result).toBe(rawExtensions)
  })

  it("blocks invalid raw extensions when expression rows need a metadata write", () => {
    const result = applyCharacterMetadataToExtensions("{not valid json", {
      preset: DEFAULT_CHARACTER_PROMPT_PRESET,
      expressionRows: [
        {
          id: "thinking",
          state: "thinking",
          image: {
            mode: "url",
            url: "https://example.test/thinking.png",
            base64: ""
          },
          starter: true
        }
      ]
    })

    expect(result).toBeNull()
  })

  it("writes canonical mood image metadata", () => {
    const result = applyCharacterMetadataToExtensions(
      { tldw: { moodImages: { sad: "https://example.test/sad.png" } } },
      {
        preset: DEFAULT_CHARACTER_PROMPT_PRESET,
        expressionRows: [
          {
            id: "smirk",
            state: "smirk",
            image: {
              mode: "url",
              url: "https://example.test/smirk.png",
              base64: ""
            },
            starter: false
          }
        ]
      }
    )

    expect(result).toEqual({
      tldw: {
        mood_images: {
          smirk: "https://example.test/smirk.png"
        }
      }
    })
  })
})

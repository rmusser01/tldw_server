import { describe, expect, it } from "vitest"

import { parseRecipeCardToolResult } from "../recipe-card-ui"

const validPayload = {
  tldw_ui: {
    kind: "recipe_card",
    version: 1,
    recipe: {
      title: "Cajun Alfredo Sauce",
      servings: { value: 2, label: "2 servings" },
      ingredients: [
        {
          display: "3 tbsp butter",
          name: "butter",
          quantity: 3,
          unit: "tbsp",
          scalable: true
        },
        { display: "salt to taste", scalable: false }
      ],
      steps: [{ display: "Melt butter.", timer_seconds: null }],
      summary: "Rich sauce for pasta.",
      notes: ["Add pasta water slowly."]
    }
  }
}

const resultWith = (content: unknown) => ({
  content: typeof content === "string" ? content : JSON.stringify(content)
})

describe("parseRecipeCardToolResult", () => {
  it("parses a valid payload and preserves title and quantity", () => {
    const parsed = parseRecipeCardToolResult(resultWith(validPayload))

    expect(parsed?.recipe.title).toBe("Cajun Alfredo Sauce")
    expect(parsed?.recipe.ingredients[0].quantity).toBe(3)
  })

  it("returns null for error results", () => {
    expect(
      parseRecipeCardToolResult({
        content: JSON.stringify(validPayload),
        error: true
      })
    ).toBeNull()
  })

  it("returns null for malformed JSON", () => {
    expect(parseRecipeCardToolResult({ content: "{" })).toBeNull()
  })

  it("returns null for unsupported version", () => {
    expect(
      parseRecipeCardToolResult(
        resultWith({ ...validPayload, tldw_ui: { ...validPayload.tldw_ui, version: 2 } })
      )
    ).toBeNull()
  })

  it("returns null for missing tldw_ui", () => {
    expect(parseRecipeCardToolResult(resultWith({ recipe: validPayload.tldw_ui.recipe }))).toBeNull()
  })

  it("returns null for wrong kind", () => {
    expect(
      parseRecipeCardToolResult(
        resultWith({
          ...validPayload,
          tldw_ui: { ...validPayload.tldw_ui, kind: "shopping_list" }
        })
      )
    ).toBeNull()
  })

  it("returns null for empty ingredients", () => {
    expect(
      parseRecipeCardToolResult(
        resultWith({
          ...validPayload,
          tldw_ui: {
            ...validPayload.tldw_ui,
            recipe: { ...validPayload.tldw_ui.recipe, ingredients: [] }
          }
        })
      )
    ).toBeNull()
  })

  it("returns null for too many ingredients", () => {
    expect(
      parseRecipeCardToolResult(
        resultWith({
          ...validPayload,
          tldw_ui: {
            ...validPayload.tldw_ui,
            recipe: {
              ...validPayload.tldw_ui.recipe,
              ingredients: Array.from({ length: 61 }, (_, index) => ({
                display: `ingredient ${index}`
              }))
            }
          }
        })
      )
    ).toBeNull()
  })

  it("returns null for too many steps", () => {
    expect(
      parseRecipeCardToolResult(
        resultWith({
          ...validPayload,
          tldw_ui: {
            ...validPayload.tldw_ui,
            recipe: {
              ...validPayload.tldw_ui.recipe,
              steps: Array.from({ length: 41 }, (_, index) => ({
                display: `step ${index}`
              }))
            }
          }
        })
      )
    ).toBeNull()
  })

  it("returns null for non-string ingredient display", () => {
    expect(
      parseRecipeCardToolResult(
        resultWith({
          ...validPayload,
          tldw_ui: {
            ...validPayload.tldw_ui,
            recipe: {
              ...validPayload.tldw_ui.recipe,
              ingredients: [{ display: 3 }]
            }
          }
        })
      )
    ).toBeNull()
  })

  it("returns null for invalid servings value", () => {
    expect(
      parseRecipeCardToolResult(
        resultWith({
          ...validPayload,
          tldw_ui: {
            ...validPayload.tldw_ui,
            recipe: {
              ...validPayload.tldw_ui.recipe,
              servings: { value: 0 }
            }
          }
        })
      )
    ).toBeNull()
  })

  it("returns null for invalid ingredient quantities", () => {
    expect(
      parseRecipeCardToolResult(
        resultWith({
          ...validPayload,
          tldw_ui: {
            ...validPayload.tldw_ui,
            recipe: {
              ...validPayload.tldw_ui.recipe,
              ingredients: [{ display: "bad butter", quantity: -1 }]
            }
          }
        })
      )
    ).toBeNull()
  })
})

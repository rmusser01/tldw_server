import type { RecipeCardPayload } from "@/types/recipe-card"
import type { ToolCallResult } from "@/types/tool-calls"

const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value)

const isString = (value: unknown, max: number, allowEmpty = false) =>
  typeof value === "string" &&
  value.length <= max &&
  (allowEmpty || value.trim().length > 0)

const isOptionalString = (value: unknown, max: number) =>
  value === undefined || isString(value, max, true)

const isOptionalStringOrNull = (value: unknown, max: number) =>
  value === undefined || value === null || isString(value, max, true)

const isFiniteNumber = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value)

const isIngredientQuantity = (value: unknown): value is number =>
  isFiniteNumber(value) && value > 0 && value <= 100000

const isValidIngredient = (ingredient: unknown) => {
  if (!isObject(ingredient) || !isString(ingredient.display, 180)) return false
  if (!isOptionalString(ingredient.name, 120)) return false
  if (ingredient.quantity !== undefined && !isIngredientQuantity(ingredient.quantity)) {
    return false
  }
  if (!isOptionalString(ingredient.unit, 32)) return false
  if (!isOptionalStringOrNull(ingredient.note, 160)) return false
  return ingredient.scalable === undefined || typeof ingredient.scalable === "boolean"
}

const isValidStep = (step: unknown) => {
  if (!isObject(step) || !isString(step.display, 600)) return false
  const timer = step.timer_seconds
  return (
    timer === undefined ||
    timer === null ||
    (typeof timer === "number" &&
      Number.isInteger(timer) &&
      timer >= 1 &&
      timer <= 86400)
  )
}

const isValidRecipe = (recipe: unknown) => {
  if (!isObject(recipe) || !isString(recipe.title, 120)) return false
  if (!isObject(recipe.servings)) return false
  if (
    !isFiniteNumber(recipe.servings.value) ||
    recipe.servings.value < 1 ||
    recipe.servings.value > 50 ||
    !isOptionalString(recipe.servings.label, 80)
  ) {
    return false
  }
  if (
    !Array.isArray(recipe.ingredients) ||
    recipe.ingredients.length < 1 ||
    recipe.ingredients.length > 60 ||
    !recipe.ingredients.every(isValidIngredient)
  ) {
    return false
  }
  if (
    !Array.isArray(recipe.steps) ||
    recipe.steps.length < 1 ||
    recipe.steps.length > 40 ||
    !recipe.steps.every(isValidStep)
  ) {
    return false
  }
  if (!isOptionalStringOrNull(recipe.summary, 300)) return false
  return (
    recipe.notes === undefined ||
    (Array.isArray(recipe.notes) &&
      recipe.notes.length <= 8 &&
      recipe.notes.every((note) => isString(note, 300, true)))
  )
}

export const parseRecipeCardToolResult = (
  result?: Pick<ToolCallResult, "content" | "error">
): RecipeCardPayload | null => {
  if (!result || result.error === true) return null

  try {
    const parsed = JSON.parse(result.content) as unknown
    if (!isObject(parsed) || !isObject(parsed.tldw_ui)) return null

    const payload = parsed.tldw_ui
    if (
      payload.kind !== "recipe_card" ||
      payload.version !== 1 ||
      !isValidRecipe(payload.recipe)
    ) {
      return null
    }

    return payload as RecipeCardPayload
  } catch {
    return null
  }
}

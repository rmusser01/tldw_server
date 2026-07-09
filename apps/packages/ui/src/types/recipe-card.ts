export type RecipeCardIngredient = {
  display: string
  name?: string | null
  quantity?: number | null
  unit?: string | null
  note?: string | null
  scalable?: boolean
}

export type RecipeCardStep = {
  display: string
  timer_seconds?: number | null
}

export type RecipeCardPayload = {
  kind: "recipe_card"
  version: 1
  recipe: {
    title: string
    servings: { value: number; label?: string }
    ingredients: RecipeCardIngredient[]
    steps: RecipeCardStep[]
    summary?: string | null
    notes?: string[]
  }
}

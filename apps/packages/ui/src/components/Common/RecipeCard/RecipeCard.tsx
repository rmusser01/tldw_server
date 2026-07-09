import React from "react"
import { ChefHat, ListChecks, Minus, Plus } from "lucide-react"
import { useTranslation } from "react-i18next"

import { classNames } from "@/libs/class-name"
import type { RecipeCardIngredient, RecipeCardPayload } from "@/types/recipe-card"

type RecipeCardProps = {
  payload: RecipeCardPayload
  className?: string
}

const formatQuantity = (value: number) =>
  Number.isInteger(value) ? String(value) : String(Math.round(value * 100) / 100)

const MIN_SERVINGS = 1
const MAX_SERVINGS = 50

const ingredientText = (
  ingredient: RecipeCardIngredient,
  factor: number
) => {
  if (
    ingredient.scalable === true &&
    Number.isFinite(ingredient.quantity) &&
    ingredient.unit &&
    ingredient.name
  ) {
    const note = ingredient.note ? ` (${ingredient.note})` : ""
    return `${formatQuantity((ingredient.quantity as number) * factor)} ${ingredient.unit} ${ingredient.name}${note}`
  }
  return ingredient.display
}

export function RecipeCard({ payload, className = "" }: RecipeCardProps) {
  const { t } = useTranslation("common")
  const { recipe } = payload
  const baseServings =
    Number.isFinite(recipe.servings.value) && recipe.servings.value > 0
      ? recipe.servings.value
      : MIN_SERVINGS
  const [servings, setServings] = React.useState(baseServings)
  const [isCookingMode, setIsCookingMode] = React.useState(false)
  const factor = baseServings > 0 ? servings / baseServings : 1
  const servingLabel =
    servings === 1
      ? t("recipeCard.serving", "{{count}} serving", { count: servings })
      : t("recipeCard.servings", "{{count}} servings", { count: servings })
  const durationText = React.useCallback(
    (seconds: number) =>
      seconds >= 60 && seconds % 60 === 0
        ? t("recipeCard.durationMinutes", {
            defaultValue: "{{count}} min",
            count: seconds / 60
          })
        : t("recipeCard.durationSeconds", {
            defaultValue: "{{count}} sec",
            count: seconds
          }),
    [t]
  )

  React.useEffect(() => {
    setServings(baseServings)
  }, [baseServings])

  return (
    <section
      className={classNames(
        "w-full max-w-full rounded-lg border border-border bg-surface px-3 py-3 text-sm text-text shadow-sm",
        className
      )}
      aria-label={recipe.title}
    >
      <header className="flex items-start gap-2">
        <ChefHat className="mt-0.5 h-4 w-4 shrink-0 text-primary" aria-hidden="true" />
        <div className="min-w-0 flex-1">
          <h3 className="break-words text-sm font-semibold leading-5 text-text">
            {recipe.title}
          </h3>
          <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-xs text-text-muted">
            <span>
              {t("recipeCard.ingredientsCount", {
                defaultValue: "{{count}} ingredients",
                count: recipe.ingredients.length
              })}
            </span>
            <span>
              {t("recipeCard.stepsCount", {
                defaultValue: "{{count}} steps",
                count: recipe.steps.length
              })}
            </span>
          </div>
        </div>
      </header>

      {recipe.summary ? (
        <p className="mt-2 break-words text-xs leading-5 text-text-muted">
          {recipe.summary}
        </p>
      ) : null}

      <div className="mt-3 flex flex-wrap items-center gap-2">
        <span className="text-xs font-medium text-text-muted">{servingLabel}</span>
        <div className="flex items-center rounded-md border border-border">
          <button
            type="button"
            aria-label={t("recipeCard.decreaseServings", "Decrease servings")}
            className="flex h-7 w-8 items-center justify-center text-text-muted hover:bg-surface2 hover:text-text disabled:opacity-40"
            disabled={servings <= MIN_SERVINGS}
            onClick={() => setServings((value) => Math.max(MIN_SERVINGS, value - 1))}
          >
            <Minus className="h-3.5 w-3.5" aria-hidden="true" />
          </button>
          <button
            type="button"
            aria-label={t("recipeCard.increaseServings", "Increase servings")}
            className="flex h-7 w-8 items-center justify-center border-l border-border text-text-muted hover:bg-surface2 hover:text-text disabled:opacity-40"
            disabled={servings >= MAX_SERVINGS}
            onClick={() => setServings((value) => Math.min(MAX_SERVINGS, value + 1))}
          >
            <Plus className="h-3.5 w-3.5" aria-hidden="true" />
          </button>
        </div>
        <button
          type="button"
          className={classNames(
            "ml-auto inline-flex min-h-[28px] items-center gap-1.5 rounded-md border px-2.5 py-1 text-xs font-medium transition-colors",
            isCookingMode
              ? "border-primary/50 bg-primary/10 text-primary"
              : "border-border text-text hover:bg-surface2"
          )}
          aria-pressed={isCookingMode}
          onClick={() => setIsCookingMode((value) => !value)}
        >
          <ListChecks className="h-3.5 w-3.5" aria-hidden="true" />
          {t("recipeCard.cookingMode", "Cooking mode")}
        </button>
      </div>

      <ul className="mt-3 space-y-1.5">
        {recipe.ingredients.map((ingredient, index) => (
          <li key={`${ingredient.display}-${index}`} className="break-words leading-5">
            {ingredientText(ingredient, factor)}
          </li>
        ))}
      </ul>

      {recipe.notes?.length ? (
        <ul className="mt-3 space-y-1 border-t border-border pt-2 text-xs text-text-muted">
          {recipe.notes.map((note, index) => (
            <li key={`${note}-${index}`} className="break-words">
              {note}
            </li>
          ))}
        </ul>
      ) : null}

      {isCookingMode ? (
        <ol className="mt-3 space-y-2 border-t border-border pt-3">
          {recipe.steps.map((step, index) => (
            <li key={`${step.display}-${index}`} className="flex gap-2 leading-5">
              <span className="text-xs font-semibold text-text-muted">{index + 1}.</span>
              <span className="min-w-0 flex-1 break-words">{step.display}</span>
              {step.timer_seconds ? (
                <span className="shrink-0 text-xs text-text-muted">
                  {durationText(step.timer_seconds)}
                </span>
              ) : null}
            </li>
          ))}
        </ol>
      ) : null}
    </section>
  )
}

export default RecipeCard

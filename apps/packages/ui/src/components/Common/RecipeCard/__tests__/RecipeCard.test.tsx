// @vitest-environment jsdom

import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it } from "vitest"

import { RecipeCard } from "../RecipeCard"
import type { RecipeCardPayload } from "@/types/recipe-card"

const payload: RecipeCardPayload = {
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
    steps: [
      { display: "Melt butter.", timer_seconds: 300 },
      { display: "Season to taste.", timer_seconds: 90 }
    ],
    summary: "Rich sauce for pasta.",
    notes: ["Add pasta water slowly."]
  }
}

describe("RecipeCard", () => {
  it("renders title, counts, and ingredients", () => {
    render(<RecipeCard payload={payload} />)

    expect(screen.getByText("Cajun Alfredo Sauce")).toBeInTheDocument()
    expect(screen.getByText("2 ingredients")).toBeInTheDocument()
    expect(screen.getByText("2 steps")).toBeInTheDocument()
    expect(screen.getByText("3 tbsp butter")).toBeInTheDocument()
    expect(screen.getByText("salt to taste")).toBeInTheDocument()
  })

  it("increases servings and scales only structured scalable ingredients", async () => {
    render(<RecipeCard payload={payload} />)

    await userEvent.click(screen.getByRole("button", { name: "Increase servings" }))

    expect(screen.getByText("3 servings")).toBeInTheDocument()
    expect(screen.getByText("4.5 tbsp butter")).toBeInTheDocument()
    expect(screen.getByText("salt to taste")).toBeInTheDocument()
  })

  it("does not decrease below 1 serving", async () => {
    render(<RecipeCard payload={payload} />)
    const decrease = screen.getByRole("button", { name: "Decrease servings" })

    await userEvent.click(decrease)
    await userEvent.click(decrease)

    expect(screen.getByText("1 serving")).toBeInTheDocument()
  })

  it("resyncs servings when a new recipe payload changes the base serving count", async () => {
    const { rerender } = render(<RecipeCard payload={payload} />)

    await userEvent.click(screen.getByRole("button", { name: "Increase servings" }))
    expect(screen.getByText("3 servings")).toBeInTheDocument()

    rerender(
      <RecipeCard
        payload={{
          ...payload,
          recipe: {
            ...payload.recipe,
            title: "Garlic Alfredo Sauce",
            servings: { value: 4, label: "4 servings" }
          }
        }}
      />
    )

    expect(await screen.findByText("4 servings")).toBeInTheDocument()
  })

  it("toggles cooking mode steps and duration text", async () => {
    render(<RecipeCard payload={payload} />)

    expect(screen.queryByText("Melt butter.")).not.toBeInTheDocument()

    await userEvent.click(screen.getByRole("button", { name: "Cooking mode" }))

    expect(screen.getByText("Melt butter.")).toBeInTheDocument()
    expect(screen.getByText("5 min")).toBeInTheDocument()
    expect(screen.getByText("90 sec")).toBeInTheDocument()
  })
})

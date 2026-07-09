// @vitest-environment jsdom

import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"

import { ToolCallBlock } from "../ToolCallBlock"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallback?: string | {
        defaultValue?: string
      }
    ) => {
      if (typeof fallback === "string") return fallback
      return fallback?.defaultValue || key
    }
  })
}))

const recipeToolCall = {
  id: "call_recipe",
  type: "function" as const,
  function: {
    name: "cooking.recipe_card.render",
    arguments: "{}"
  }
}

const recipePayload = {
  tldw_ui: {
    kind: "recipe_card",
    version: 1,
    recipe: {
      title: "Cajun Alfredo Sauce",
      servings: { value: 2, label: "2 servings" },
      ingredients: [
        {
          display: "3 tbsp butter",
          quantity: 3,
          unit: "tbsp",
          name: "butter",
          scalable: true
        }
      ],
      steps: [{ display: "Melt butter.", timer_seconds: null }],
      summary: null,
      notes: []
    }
  }
}

describe("ToolCallBlock recipe card rendering", () => {
  it("renders a valid recipe card tool result instead of raw JSON", async () => {
    render(
      <ToolCallBlock
        toolCalls={[recipeToolCall]}
        results={[
          {
            tool_call_id: "call_recipe",
            content: JSON.stringify(recipePayload)
          }
        ]}
      />
    )

    await userEvent.click(screen.getByRole("button", { name: /Recipe Card/i }))

    expect(screen.getByText("Recipe Card")).toBeInTheDocument()
    expect(screen.getByText("Cajun Alfredo Sauce")).toBeInTheDocument()
    expect(screen.getByText("3 tbsp butter")).toBeInTheDocument()
    expect(screen.queryByText(/"tldw_ui"/)).not.toBeInTheDocument()
  })

  it("falls back to generic error output for failed recipe tool results", async () => {
    render(
      <ToolCallBlock
        toolCalls={[recipeToolCall]}
        results={[
          {
            tool_call_id: "call_recipe",
            content: JSON.stringify({
              ok: false,
              error: "invalid recipe"
            }),
            error: true
          }
        ]}
      />
    )

    await userEvent.click(screen.getByRole("button", { name: /Recipe Card/i }))

    expect(screen.queryByText("Cajun Alfredo Sauce")).not.toBeInTheDocument()
    expect(screen.getByText("Error: invalid recipe")).toBeInTheDocument()
  })

  it("falls back to generic output for malformed recipe JSON", async () => {
    render(
      <ToolCallBlock
        toolCalls={[recipeToolCall]}
        results={[
          {
            tool_call_id: "call_recipe",
            content: "{"
          }
        ]}
      />
    )

    await userEvent.click(screen.getByRole("button", { name: /Recipe Card/i }))

    expect(screen.queryByText("Cajun Alfredo Sauce")).not.toBeInTheDocument()
    expect(screen.getByText("{")).toBeInTheDocument()
  })

  it("keeps valid recipe-shaped JSON generic for unknown tools", async () => {
    render(
      <ToolCallBlock
        toolCalls={[
          {
            ...recipeToolCall,
            function: {
              name: "other_tool",
              arguments: "{}"
            }
          }
        ]}
        results={[
          {
            tool_call_id: "call_recipe",
            content: JSON.stringify(recipePayload)
          }
        ]}
      />
    )

    await userEvent.click(screen.getByRole("button", { name: /other tool/i }))

    expect(screen.queryByRole("region", { name: "Cajun Alfredo Sauce" })).not.toBeInTheDocument()
    expect(screen.getByText(/"tldw_ui"/)).toBeInTheDocument()
  })
})

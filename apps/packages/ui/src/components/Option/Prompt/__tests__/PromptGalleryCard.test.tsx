import React from "react"
import { describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import {
  PromptGalleryCard,
  getAvatarFallbackTokens,
  hashNameToHue
} from "../PromptGalleryCard"
import type { PromptRowVM } from "../prompt-workspace-types"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      options?: { defaultValue?: string; [name: string]: unknown }
    ) => {
      if (!options?.defaultValue) return key
      return Object.entries(options).reduce(
        (value, [name, replacement]) =>
          name === "defaultValue"
            ? value
            : value.replace(
                new RegExp(`{{${name}}}`, "g"),
                String(replacement)
              ),
        options.defaultValue
      )
    }
  })
}))

const makePrompt = (overrides: Partial<PromptRowVM> = {}): PromptRowVM => ({
  id: "prompt-1",
  title: "My Prompt",
  keywords: [],
  favorite: false,
  syncStatus: "local",
  sourceSystem: "workspace",
  createdAt: Date.now(),
  usageCount: 0,
  ...overrides
})

describe("PromptGalleryCard", () => {
  it("renders title text", () => {
    render(
      <PromptGalleryCard prompt={makePrompt()} onClick={vi.fn()} />
    )
    expect(screen.getByText("My Prompt")).toBeInTheDocument()
  })

  it("labels recipes separately from ordinary prompts", () => {
    const { rerender } = render(
      <PromptGalleryCard
        prompt={makePrompt({ kind: "recipe", recipeTarget: "system" })}
        onClick={vi.fn()}
      />
    )
    expect(screen.getByText("Recipe")).toBeInTheDocument()
    expect(screen.getByText("System")).toBeInTheDocument()

    rerender(<PromptGalleryCard prompt={makePrompt()} onClick={vi.fn()} />)
    expect(screen.queryByText("Recipe")).not.toBeInTheDocument()
  })

  it("names a recipe card action for opening the recipe editor", () => {
    render(
      <PromptGalleryCard
        prompt={makePrompt({ kind: "recipe", recipeTarget: "system" })}
        onClick={vi.fn()}
      />
    )

    expect(
      screen.getByRole("button", {
        name: "Open recipe editor for My Prompt"
      })
    ).toBeInTheDocument()
  })

  it("shows colored fallback avatar with initial letter", () => {
    const tokens = getAvatarFallbackTokens("My Prompt")
    render(
      <PromptGalleryCard prompt={makePrompt()} onClick={vi.fn()} />
    )
    expect(screen.getByText("M")).toBeInTheDocument()
    expect(screen.getByTestId("prompt-gallery-fallback-avatar")).toHaveStyle({
      backgroundColor: tokens.backgroundColor,
      color: tokens.color
    })
  })

  it("produces deterministic hue values", () => {
    expect(hashNameToHue("test")).toBe(hashNameToHue("test"))
    expect(hashNameToHue("alpha")).not.toBe(hashNameToHue("beta"))
  })

  it("shows usage badge when usageCount > 0", () => {
    render(
      <PromptGalleryCard
        prompt={makePrompt({ id: "p-usage", usageCount: 5 })}
        onClick={vi.fn()}
      />
    )
    expect(screen.getByTestId("prompt-gallery-usage-p-usage")).toBeInTheDocument()
    expect(screen.getByText("5")).toBeInTheDocument()
  })

  it("hides usage badge when usageCount is 0", () => {
    render(
      <PromptGalleryCard
        prompt={makePrompt({ id: "p-no-usage", usageCount: 0 })}
        onClick={vi.fn()}
      />
    )
    expect(screen.queryByTestId("prompt-gallery-usage-p-no-usage")).not.toBeInTheDocument()
  })

  it("caps usage badge display at 99+", () => {
    render(
      <PromptGalleryCard
        prompt={makePrompt({ usageCount: 150 })}
        onClick={vi.fn()}
      />
    )
    expect(screen.getByText("99+")).toBeInTheDocument()
  })

  it("calls onClick on click", async () => {
    const user = userEvent.setup()
    const onClick = vi.fn()
    render(
      <PromptGalleryCard prompt={makePrompt()} onClick={onClick} />
    )
    await user.click(screen.getByRole("button", { name: /Click to preview/i }))
    expect(onClick).toHaveBeenCalledTimes(1)
  })

  it("calls onClick on Enter key", async () => {
    const user = userEvent.setup()
    const onClick = vi.fn()
    render(
      <PromptGalleryCard prompt={makePrompt()} onClick={onClick} />
    )
    const card = screen.getByRole("button", { name: /Click to preview/i })
    card.focus()
    await user.keyboard("{Enter}")
    expect(onClick).toHaveBeenCalledTimes(1)
  })

  it("calls onToggleFavorite when star clicked", async () => {
    const user = userEvent.setup()
    const onToggle = vi.fn()
    render(
      <PromptGalleryCard
        prompt={makePrompt({ favorite: false })}
        onClick={vi.fn()}
        onToggleFavorite={onToggle}
      />
    )
    await user.click(screen.getByTestId("prompt-gallery-favorite-prompt-1"))
    expect(onToggle).toHaveBeenCalledWith(true)
  })

  it("shows preview and keywords in rich density", () => {
    render(
      <PromptGalleryCard
        prompt={makePrompt({
          previewSystem: "You are a helpful assistant.",
          keywords: ["ai", "helper", "chat", "extra"]
        })}
        density="rich"
        onClick={vi.fn()}
      />
    )
    expect(screen.getByText("You are a helpful assistant.")).toBeInTheDocument()
    expect(screen.getByText("ai")).toBeInTheDocument()
    expect(screen.getByText("helper")).toBeInTheDocument()
    expect(screen.getByText("chat")).toBeInTheDocument()
    expect(screen.queryByText("extra")).not.toBeInTheDocument()
  })

  it("hides preview and keywords in compact density", () => {
    render(
      <PromptGalleryCard
        prompt={makePrompt({
          previewSystem: "Hidden text",
          keywords: ["hidden-kw"]
        })}
        density="compact"
        onClick={vi.fn()}
      />
    )
    expect(screen.getByText("My Prompt")).toBeInTheDocument()
    expect(screen.queryByText("Hidden text")).not.toBeInTheDocument()
    expect(screen.queryByText("hidden-kw")).not.toBeInTheDocument()
  })
})

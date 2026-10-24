// @vitest-environment jsdom
/**
 * CC-09 (#3101): in the Pro advanced row, "System prompts" used a larger
 * label and padding than the "Formatting guide" control beside it.
 */
import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback ?? _key
  })
}))

import { SystemPromptTemplatesButton } from "../SystemPromptTemplates"

describe("SystemPromptTemplatesButton size", () => {
  it("matches the compact toolbar controls when asked", () => {
    render(<SystemPromptTemplatesButton onSelect={vi.fn()} size="compact" />)

    const button = screen.getByRole("button", { name: "System prompts" })
    expect(button).toHaveClass("text-xs", "rounded-md")
    expect(button).not.toHaveClass("text-sm")
  })

  it("keeps the default size elsewhere", () => {
    render(<SystemPromptTemplatesButton onSelect={vi.fn()} />)

    expect(screen.getByRole("button", { name: "System prompts" })).toHaveClass(
      "text-sm"
    )
  })
})

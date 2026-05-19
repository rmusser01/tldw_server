import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { FlipCardDemo } from "../FlipCardDemo"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, options?: { defaultValue?: string }) =>
      options?.defaultValue ?? _key
  })
}))

describe("FlipCardDemo", () => {
  it("supports flipping cards and exposes labeled navigation controls", () => {
    render(<FlipCardDemo />)

    expect(screen.getByText("What is spaced repetition?")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Previous card" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Next card" })).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Show answer" }))
    expect(
      screen.getByText(/A learning technique that reviews material at increasing intervals/i)
    ).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Next card" }))
    expect(screen.getByText("What does RAG stand for in AI?")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Previous card" }))
    expect(screen.getByText("What is spaced repetition?")).toBeInTheDocument()
  })
})

import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { TokenProgressBar } from "../TokenProgressBar"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback || _key
  })
}))

vi.mock("antd", () => ({
  Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>
}))

describe("TokenProgressBar", () => {
  it("invokes onClick when interactive", () => {
    const onClick = vi.fn()

    render(
      <TokenProgressBar
        conversationTokens={1200}
        draftTokens={80}
        maxTokens={4096}
        modelLabel="Test model"
        onClick={onClick}
      />
    )

    fireEvent.click(
      screen.getByRole("button", { name: "Configure context window size" })
    )

    expect(onClick).toHaveBeenCalledTimes(1)
  })

  // AX-08 (#3124): the focus ring was 50% primary, not the theme focus colour.
  it.each([
    ["full", { maxTokens: 4096 }],
    ["compact", { maxTokens: 4096, compact: true }],
    ["unknown window", { maxTokens: 0 }]
  ] as const)("shows a solid focus-colour ring on the %s control", (_name, props) => {
    render(
      <TokenProgressBar
        conversationTokens={1200}
        draftTokens={80}
        modelLabel="Test model"
        onClick={vi.fn()}
        {...props}
      />
    )

    const control = screen.getByRole("button", { name: "Configure context window size" })
    expect(control).toHaveClass("focus-visible:ring-2", "focus-visible:ring-focus")
    expect(control.className).not.toMatch(/ring-primary\//)
  })

  it("renders a non-interactive indicator when no click handler is provided", () => {
    render(
      <TokenProgressBar
        conversationTokens={1200}
        draftTokens={80}
        maxTokens={4096}
        modelLabel="Test model"
      />
    )

    expect(
      screen.queryByRole("button", { name: "Configure context window size" })
    ).toBeNull()
    expect(screen.getByText(/used/i)).toBeInTheDocument()
  })

  it("shows compact numeric conversation + draft tokens when context is known", () => {
    render(
      <TokenProgressBar
        conversationTokens={500}
        draftTokens={100}
        maxTokens={1000}
        compact
      />
    )

    expect(screen.getByText("500 + ~100 = 600 tokens")).toBeInTheDocument()
  })

  it("shows conversation token count when context window is unavailable", () => {
    render(
      <TokenProgressBar
        conversationTokens={1280}
        draftTokens={0}
        maxTokens={null}
        compact
      />
    )

    expect(screen.getByText("1,280 + ~0 = 1,280 tokens")).toBeInTheDocument()
  })
})

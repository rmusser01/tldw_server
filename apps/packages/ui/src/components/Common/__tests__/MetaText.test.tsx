import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { MetaText } from "../MetaText"

// XP-15 (#3124): antd's hashed font reset (.css-<hash>.ant-typography) outranks
// Tailwind size utilities, so Typography.Text type="secondary" with
// text-[11px] rendered at 14px. MetaText is a plain element instead.
describe("MetaText", () => {
  it("renders small muted text outside antd Typography", () => {
    render(<MetaText>Markdown supported</MetaText>)

    const text = screen.getByText("Markdown supported")
    expect(text.tagName).toBe("SPAN")
    expect(text).toHaveClass("text-[11px]", "text-text-muted")
    expect(text.className).not.toMatch(/ant-typography/)
  })

  it("adds caller classes and passes attributes through", () => {
    render(
      <MetaText id="hint" className="block uppercase">
        Formatting
      </MetaText>
    )

    const text = screen.getByText("Formatting")
    expect(text).toHaveAttribute("id", "hint")
    expect(text).toHaveClass("block", "uppercase", "text-[11px]", "text-text-muted")
  })
})

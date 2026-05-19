import React from "react"
import { render } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { TemplateCodeEditor } from "../TemplateCodeEditor"

vi.mock("@monaco-editor/react", () => {
  const PendingMonaco = () => {
    throw new Promise(() => {})
  }

  return { default: PendingMonaco }
})

describe("TemplateCodeEditor loading state", () => {
  it("renders the Suspense fallback through the canonical LoadingState primitive", () => {
    const { container } = render(
      <TemplateCodeEditor
        value="# Template"
        onChange={vi.fn()}
        format="md"
        height={240}
      />
    )

    const loadingState = container.querySelector(
      '[data-ds-component="LoadingState"]'
    )
    expect(loadingState).toBeInTheDocument()
    expect(loadingState).toHaveStyle({ height: "240px" })
  })
})

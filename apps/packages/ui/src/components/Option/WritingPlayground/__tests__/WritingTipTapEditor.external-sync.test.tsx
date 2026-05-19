import React from "react"
import type { JSONContent } from "@tiptap/react"
import { render, waitFor } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { WritingTipTapEditor } from "../WritingTipTapEditor"

const FIRST_DOC: JSONContent = {
  type: "doc",
  content: [
    {
      type: "paragraph",
      content: [{ type: "text", text: "First draft" }]
    }
  ]
}

const SECOND_DOC: JSONContent = {
  type: "doc",
  content: [
    {
      type: "paragraph",
      content: [{ type: "text", text: "Second draft" }]
    }
  ]
}

describe("WritingTipTapEditor external sync", () => {
  it("accepts external content updates without a component-level focus gate", async () => {
    const onContentChange = vi.fn()
    const { container, rerender } = render(
      <WritingTipTapEditor content={FIRST_DOC} onContentChange={onContentChange} />
    )

    rerender(
      <WritingTipTapEditor content={SECOND_DOC} onContentChange={onContentChange} />
    )

    await waitFor(() => {
      expect(container.textContent).toContain("Second draft")
    })
  })
})

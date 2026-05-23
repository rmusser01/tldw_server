import React from "react"
import type { JSONContent } from "@tiptap/react"
import { act, render, waitFor } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { WritingTipTapEditor } from "../WritingTipTapEditor"
import type { WritingEditorAdapter } from "../writing-editor-adapter"

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

  it("maps selections to plain-text offsets after paragraph boundaries", async () => {
    const adapterRef: { current: WritingEditorAdapter | null } = {
      current: null
    }
    const selectionChanges: Array<{ start: number; end: number }> = []

    render(
      <WritingTipTapEditor
        content={{
          type: "doc",
          content: [
            {
              type: "paragraph",
              content: [{ type: "text", text: "Alpha" }]
            },
            {
              type: "paragraph",
              content: [{ type: "text", text: "Beta" }]
            }
          ]
        }}
        onContentChange={vi.fn()}
        onAdapterReady={(adapter) => {
          adapterRef.current = adapter
        }}
        onSelectionChange={(selection) => {
          selectionChanges.push(selection)
        }}
      />
    )

    await waitFor(() => {
      expect(adapterRef.current).not.toBeNull()
    })

    act(() => {
      adapterRef.current?.setSelection({ start: 6, end: 10 })
    })

    await waitFor(() => {
      expect(adapterRef.current?.getSelectedText("Alpha\nBeta")).toBe("Beta")
    })
    await waitFor(() => {
      expect(selectionChanges.at(-1)).toEqual({ start: 6, end: 10 })
    })
  })
})

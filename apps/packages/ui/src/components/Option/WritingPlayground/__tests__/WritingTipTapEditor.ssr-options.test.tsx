import type { JSONContent } from "@tiptap/react"
import { render } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { WritingTipTapEditor } from "../WritingTipTapEditor"

const tiptapMock = vi.hoisted(() => {
  const state = {
    options: null as Record<string, unknown> | null,
  }
  const mockEditor = {
    isDestroyed: false,
    commands: {
      focus: vi.fn(),
      setContent: vi.fn(),
      setTextSelection: vi.fn(),
    },
    getJSON: vi.fn(() => ({ type: "doc", content: [] })),
    setEditable: vi.fn(),
    state: {
      selection: { from: 0, to: 0 },
      doc: {
        content: { size: 0 },
        forEach: vi.fn(),
      },
    },
    view: {
      coordsAtPos: vi.fn(() => ({ top: 0, bottom: 0 })),
      dom: {
        getBoundingClientRect: vi.fn(() => ({ top: 0 })),
      },
    },
  }

  return {
    state,
    mockEditor,
    useEditor: vi.fn((options: Record<string, unknown>) => {
      state.options = options
      return mockEditor
    }),
  }
})

vi.mock("@tiptap/react", () => ({
  EditorContent: () => null,
  useEditor: tiptapMock.useEditor,
}))

const DOC: JSONContent = {
  type: "doc",
  content: [
    {
      type: "paragraph",
      content: [{ type: "text", text: "SSR-safe editor" }],
    },
  ],
}

describe("WritingTipTapEditor SSR options", () => {
  it("opts out of immediate TipTap rendering for Next WebUI hydration", () => {
    render(<WritingTipTapEditor content={DOC} onContentChange={vi.fn()} />)

    expect(tiptapMock.useEditor).toHaveBeenCalled()
    expect(tiptapMock.state.options?.immediatelyRender).toBe(false)
  })
})

import { Editor } from "@tiptap/core"
import StarterKit from "@tiptap/starter-kit"
import { describe, expect, it, vi } from "vitest"
import { SceneBreakExtension } from "../extensions/SceneBreakExtension"
import {
  createTextareaEditorAdapter,
  createTipTapEditorAdapter
} from "../writing-editor-adapter"
import { tipTapJsonToPlainText } from "../writing-tiptap-utils"

describe("writing editor adapter", () => {
  it("sets TipTap selections using plain-text offsets after paragraph breaks", () => {
    const editor = new Editor({
      extensions: [StarterKit],
      content: {
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
      }
    })
    const adapter = createTipTapEditorAdapter(editor)
    const plainText = tipTapJsonToPlainText(editor.getJSON())
    const start = plainText.indexOf("Beta")

    expect(plainText).toBe("Alpha\n\nBeta")

    adapter?.setSelection({ start, end: start + "Beta".length })

    const { from, to } = editor.state.selection
    expect(editor.state.doc.textBetween(from, to, "\n", "\n")).toBe("Beta")
    expect(adapter?.getSelection()).toEqual({ start, end: start + "Beta".length })

    editor.destroy()
  })

  it("sets TipTap selections using offsets after empty paragraphs", () => {
    const editor = new Editor({
      extensions: [StarterKit],
      content: {
        type: "doc",
        content: [
          {
            type: "paragraph",
            content: [{ type: "text", text: "Alpha" }]
          },
          {
            type: "paragraph"
          },
          {
            type: "paragraph",
            content: [{ type: "text", text: "Beta" }]
          }
        ]
      }
    })
    const adapter = createTipTapEditorAdapter(editor)
    const plainText = tipTapJsonToPlainText(editor.getJSON())
    const start = plainText.indexOf("Beta")

    expect(plainText).toBe("Alpha\n\n\n\nBeta")

    adapter?.setSelection({ start, end: start + "Beta".length })

    const { from, to } = editor.state.selection
    expect(editor.state.doc.textBetween(from, to, "\n", "\n")).toBe("Beta")
    expect(adapter?.getSelection()).toEqual({ start, end: start + "Beta".length })

    editor.destroy()
  })

  it("sets TipTap selections using offsets inside nested list items", () => {
    const editor = new Editor({
      extensions: [StarterKit],
      content: {
        type: "doc",
        content: [
          {
            type: "bulletList",
            content: [
              {
                type: "listItem",
                content: [
                  {
                    type: "paragraph",
                    content: [{ type: "text", text: "First item" }]
                  }
                ]
              },
              {
                type: "listItem",
                content: [
                  {
                    type: "paragraph",
                    content: [{ type: "text", text: "Second item" }]
                  }
                ]
              }
            ]
          }
        ]
      }
    })
    const adapter = createTipTapEditorAdapter(editor)
    const plainText = tipTapJsonToPlainText(editor.getJSON())
    const start = plainText.indexOf("Second item")

    expect(plainText).toBe("First item\n\nSecond item")

    adapter?.setSelection({ start, end: start + "Second item".length })

    const { from, to } = editor.state.selection
    expect(editor.state.doc.textBetween(from, to, "\n", "\n")).toBe("Second item")
    expect(adapter?.getSelection()).toEqual({
      start,
      end: start + "Second item".length
    })

    editor.destroy()
  })

  it("maps TipTap selections against serialized scene breaks", () => {
    const editor = new Editor({
      extensions: [StarterKit, SceneBreakExtension],
      content: {
        type: "doc",
        content: [
          {
            type: "paragraph",
            content: [{ type: "text", text: "Alpha" }]
          },
          { type: "sceneBreak" },
          {
            type: "paragraph",
            content: [{ type: "text", text: "Beta" }]
          }
        ]
      }
    })
    const adapter = createTipTapEditorAdapter(editor)
    const plainText = tipTapJsonToPlainText(editor.getJSON())
    const start = plainText.indexOf("Beta")

    expect(plainText).toBe("Alpha\n\n***\n\nBeta")

    adapter?.setSelection({ start, end: start + "Beta".length })

    const { from, to } = editor.state.selection
    expect(editor.state.doc.textBetween(from, to, "\n", "\n")).toBe("Beta")
    expect(adapter?.getSelection()).toEqual({ start, end: start + 4 })

    editor.destroy()
  })

  it("exposes TipTap range measurement using mapped plain-text offsets", () => {
    const editor = new Editor({
      extensions: [StarterKit],
      content: {
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
      }
    })
    const adapter = createTipTapEditorAdapter(editor)
    const plainText = tipTapJsonToPlainText(editor.getJSON())
    const start = plainText.indexOf("Beta")

    expect(plainText).toBe("Alpha\n\nBeta")

    adapter?.setSelection({ start, end: start + "Beta".length })
    const { from } = editor.state.selection
    const coordsAtPos = vi
      .spyOn(editor.view, "coordsAtPos")
      .mockImplementationOnce((position) => ({
        top: position === from ? 24 : 64,
        bottom: 40,
        left: 0,
        right: 0
      }))
      .mockImplementationOnce(() => ({
        top: 64,
        bottom: 82,
        left: 0,
        right: 0
      }))

    expect(adapter?.measureRange).toBeTypeOf("function")
    expect(adapter?.measureRange?.({ start, end: start + "Beta".length })).toEqual({
      top: 24,
      bottom: 82,
      height: 58
    })
    expect(coordsAtPos).toHaveBeenCalledWith(from)
    expect(coordsAtPos).toHaveBeenCalledTimes(2)

    coordsAtPos.mockRestore()
    editor.destroy()
  })

  it("returns null for invalid or stale TipTap measurement ranges", () => {
    const editor = new Editor({
      extensions: [StarterKit],
      content: {
        type: "doc",
        content: [
          {
            type: "paragraph",
            content: [{ type: "text", text: "Alpha" }]
          }
        ]
      }
    })
    const adapter = createTipTapEditorAdapter(editor)

    expect(adapter?.measureRange?.({ start: 99, end: 104 })).toBeNull()
    expect(adapter?.measureRange?.({ start: 2, end: 2 })).toBeNull()

    editor.destroy()
  })

  it("does not expose textarea range measurement support", () => {
    const textareaRef = { current: null }
    const adapter = createTextareaEditorAdapter(textareaRef)

    expect("measureRange" in adapter).toBe(false)
    expect(adapter.measureRange).toBeUndefined()
  })
})

import { Editor } from "@tiptap/core"
import StarterKit from "@tiptap/starter-kit"
import { describe, expect, it } from "vitest"
import { SceneBreakExtension } from "../extensions/SceneBreakExtension"
import { createTipTapEditorAdapter } from "../writing-editor-adapter"
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

    adapter?.setSelection({ start: 6, end: 10 })

    const { from, to } = editor.state.selection
    expect(editor.state.doc.textBetween(from, to, "\n", "\n")).toBe("Beta")
    expect(adapter?.getSelection()).toEqual({ start: 6, end: 10 })

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

    expect(plainText).toBe("Alpha\n\n***\nBeta")

    adapter?.setSelection({ start, end: start + "Beta".length })

    const { from, to } = editor.state.selection
    expect(editor.state.doc.textBetween(from, to, "\n", "\n")).toBe("Beta")
    expect(adapter?.getSelection()).toEqual({ start, end: start + 4 })

    editor.destroy()
  })
})

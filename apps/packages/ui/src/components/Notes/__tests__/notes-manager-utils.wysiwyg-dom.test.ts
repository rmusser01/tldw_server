import { afterEach, describe, expect, it } from "vitest"
import {
  getEditableCaretOffset,
  getEditableSelectionBlock,
  replaceEditableHtml,
  setEditableCaretOffset,
  unwrapListsFromBlocks,
  wysiwygHtmlToMarkdown
} from "../notes-manager-utils"

const mountEditor = (html: string): HTMLDivElement => {
  const editor = document.createElement("div")
  editor.contentEditable = "true"
  editor.tabIndex = 0
  editor.innerHTML = html
  document.body.appendChild(editor)
  return editor
}

const placeCaret = (node: Node, offset: number) => {
  const range = document.createRange()
  range.setStart(node, offset)
  range.collapse(true)
  const selection = window.getSelection()
  selection?.removeAllRanges()
  selection?.addRange(range)
}

const firstText = (element: Element | null): Text => {
  const walker = document.createTreeWalker(element as Node, NodeFilter.SHOW_TEXT)
  return walker.nextNode() as Text
}

afterEach(() => {
  window.getSelection()?.removeAllRanges()
  document.body.innerHTML = ""
})

describe("WYSIWYG caret helpers", () => {
  it("reads and restores the caret as a text offset across blocks", () => {
    const editor = mountEditor("<p>Hello</p><p>wide world</p>")
    placeCaret(firstText(editor.querySelectorAll("p")[1]), 4)

    expect(getEditableCaretOffset(editor)).toBe("Hello".length + 4)

    setEditableCaretOffset(editor, 2)
    expect(getEditableCaretOffset(editor)).toBe(2)
    expect(window.getSelection()?.anchorNode).toBe(firstText(editor.querySelector("p")))
  })

  it("clamps an offset past the end to the end of the text", () => {
    const editor = mountEditor("<p>Short</p>")
    setEditableCaretOffset(editor, 99)
    expect(getEditableCaretOffset(editor)).toBe("Short".length)
  })

  it("returns null when the selection is outside the editor", () => {
    const editor = mountEditor("<p>Inside</p>")
    const outside = document.createElement("p")
    outside.textContent = "Outside"
    document.body.appendChild(outside)
    placeCaret(outside.firstChild as Text, 3)
    expect(getEditableCaretOffset(editor)).toBeNull()
  })

  it("keeps the caret offset when replacing the HTML of the focused editor", () => {
    const editor = mountEditor("<p>Hello</p>")
    editor.focus()
    placeCaret(firstText(editor), 5)

    replaceEditableHtml(editor, "<p>Hello</p>")

    expect(getEditableCaretOffset(editor)).toBe(5)
  })

  it("leaves the selection alone when the editor does not have focus", () => {
    const editor = mountEditor("<p>Hello</p>")
    const outside = document.createElement("input")
    document.body.appendChild(outside)
    outside.focus()

    replaceEditableHtml(editor, "<p>Replaced</p>")

    expect(editor.innerHTML).toBe("<p>Replaced</p>")
    expect(getEditableCaretOffset(editor)).toBeNull()
  })
})

describe("getEditableSelectionBlock", () => {
  it("finds the paragraph, heading or list item that holds the caret", () => {
    const editor = mountEditor("<p>Para</p><h2>Head</h2><ul><li><strong>Item</strong></li></ul>")

    placeCaret(firstText(editor.querySelector("p")), 1)
    expect(getEditableSelectionBlock(editor)?.tagName).toBe("P")

    placeCaret(firstText(editor.querySelector("h2")), 1)
    expect(getEditableSelectionBlock(editor)?.tagName).toBe("H2")

    placeCaret(firstText(editor.querySelector("li")), 1)
    expect(getEditableSelectionBlock(editor)?.tagName).toBe("LI")
  })

  it("returns null for text directly under the editor or outside it", () => {
    const editor = mountEditor("Loose text")
    placeCaret(editor.firstChild as Text, 2)
    expect(getEditableSelectionBlock(editor)).toBeNull()

    window.getSelection()?.removeAllRanges()
    expect(getEditableSelectionBlock(editor)).toBeNull()
  })
})

describe("unwrapListsFromBlocks", () => {
  it("moves a list Chrome built inside a paragraph to the top level", () => {
    // The HTML parser would close the <p> before the <ul>, so build the nested
    // shape Chrome's insertUnorderedList produces with DOM calls.
    const editor = mountEditor("")
    const paragraph = document.createElement("p")
    paragraph.innerHTML = "<ul><li>First line</li></ul>"
    const second = document.createElement("p")
    second.textContent = "Second"
    editor.append(paragraph, second)
    const itemText = firstText(editor.querySelector("li"))
    placeCaret(itemText, 5)

    unwrapListsFromBlocks(editor)

    expect(editor.innerHTML).toBe("<ul><li>First line</li></ul><p>Second</p>")
    expect(window.getSelection()?.anchorNode).toBe(itemText)
    expect(window.getSelection()?.anchorOffset).toBe(5)
    expect(wysiwygHtmlToMarkdown(editor.innerHTML)).toBe("- First line\n\nSecond")
  })

  it("moves a list out of a heading", () => {
    const editor = mountEditor("")
    const heading = document.createElement("h2")
    heading.innerHTML = "<ul><li>Heading</li></ul>"
    editor.append(heading)

    unwrapListsFromBlocks(editor)

    expect(editor.innerHTML).toBe("<ul><li>Heading</li></ul>")
    expect(wysiwygHtmlToMarkdown(editor.innerHTML)).toBe("- Heading")
  })

  it("splits a paragraph around a list in its middle", () => {
    const editor = mountEditor("")
    const paragraph = document.createElement("p")
    paragraph.innerHTML = "before<ul><li>item</li></ul>after"
    editor.append(paragraph)

    unwrapListsFromBlocks(editor)

    expect(editor.innerHTML).toBe("<p>before</p><ul><li>item</li></ul><p>after</p>")
  })

  it("leaves top-level and nested sub-lists alone", () => {
    const html = "<ul><li>one<ul><li>nested</li></ul></li></ul><p>text</p>"
    const editor = mountEditor(html)

    unwrapListsFromBlocks(editor)

    expect(editor.innerHTML).toBe(html)
  })
})

import { afterEach, describe, expect, it } from "vitest"
import {
  getEditableCaretOffset,
  getEditableSelectionBlock,
  syncEditableCheckboxes,
  makeEditableChecklist,
  removeEditableTaskControls,
  continueEditableChecklist,
  markdownToWysiwygHtml,
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

describe("edited list roundtrip", () => {
  it("changes only a selected nested task while preserving parent and siblings", () => {
    const editor = mountEditor(
      '<ul><li><input type="checkbox" checked>parent<ul><li><input type="checkbox" checked>child</li><li><input type="checkbox">sibling</li></ul></li><li>outside</li></ul>'
    )
    const parent = editor.querySelector("li")!
    const nested = parent.querySelectorAll("li")
    const parentBox = parent.querySelector("input")!
    const sibling = nested[1]
    const outside = editor.querySelectorAll(":scope > ul > li")[1]
    const text = nested[0].lastChild!
    const range = document.createRange()
    range.setStart(text, 0)
    range.setEnd(text, text.textContent!.length)
    window.getSelection()?.removeAllRanges()
    window.getSelection()?.addRange(range)
    removeEditableTaskControls(editor)
    expect(nested[0].querySelector("input")).toBeNull()
    expect(parent.querySelector("input")).toBe(parentBox)
    expect(parentBox.checked).toBe(true)
    expect(parent.querySelectorAll("li")[1]).toBe(sibling)
    expect(sibling.querySelector("input")?.checked).toBe(false)
    makeEditableChecklist(editor)
    expect(nested[0].querySelector("input")?.checked).toBe(false)
    expect(editor.querySelectorAll(":scope > ul > li")[1]).toBe(outside)
    expect(wysiwygHtmlToMarkdown(editor.innerHTML)).toBe(
      "- [x] parent\n    - [ ] child\n    - [ ] sibling\n- outside"
    )
  })

  it("adds direct controls to selected parent and nested child bodies", () => {
    const editor = mountEditor(
      '<ul><li>parent<ul><li>child</li><li><input type="checkbox" checked>sibling</li></ul></li></ul>'
    )
    const parent = editor.querySelector("li")!
    const child = parent.querySelector("li")!
    const sibling = parent.querySelectorAll("li")[1]
    const siblingBox = sibling.querySelector("input")!
    const parentText = parent.firstChild!
    const childText = child.firstChild!
    const range = document.createRange()
    range.setStart(parentText, 0)
    range.setEnd(childText, childText.textContent!.length)
    window.getSelection()?.removeAllRanges()
    window.getSelection()?.addRange(range)
    expect(range.startContainer).toBe(parentText)
    expect(range.endContainer).toBe(childText)
    expect(range.toString()).toBe("parentchild")
    expect(siblingBox.checked).toBe(true)
    makeEditableChecklist(editor)
    expect(parent.querySelectorAll(":scope > input[type=checkbox]")).toHaveLength(
      1
    )
    expect(child.querySelectorAll(":scope > input[type=checkbox]")).toHaveLength(
      1
    )
    expect(parent.firstChild?.nextSibling).toBe(parentText)
    expect(child.firstChild?.nextSibling).toBe(childText)
    expect(parent.querySelectorAll("li")[1]).toBe(sibling)
    expect(sibling.querySelector("input")).toBe(siblingBox)
    expect(siblingBox.checked).toBe(true)
    expect(wysiwygHtmlToMarkdown(editor.innerHTML)).toBe(
      "- [ ] parent\n    - [ ] child\n    - [x] sibling"
    )
  })

  it("removes direct controls from selected parent and nested child bodies", () => {
    const editor = mountEditor(
      '<ul><li><input type="checkbox" checked>parent<ul><li><input type="checkbox" checked>child</li><li><input type="checkbox" checked>sibling</li></ul></li></ul>'
    )
    const parent = editor.querySelector("li")!
    const child = parent.querySelector("li")!
    const sibling = parent.querySelectorAll("li")[1]
    const parentBox = parent.querySelector(":scope > input") as HTMLInputElement
    const childBox = child.querySelector(":scope > input") as HTMLInputElement
    const siblingBox = sibling.querySelector("input")!
    const parentText = parentBox.nextSibling!
    const childText = childBox.nextSibling!
    const range = document.createRange()
    range.setStart(parentText, 0)
    range.setEnd(childText, childText.textContent!.length)
    window.getSelection()?.removeAllRanges()
    window.getSelection()?.addRange(range)
    expect(range.startContainer).toBe(parentText)
    expect(range.endContainer).toBe(childText)
    expect(range.toString()).toBe("parentchild")
    expect([parentBox.checked, childBox.checked, siblingBox.checked]).toEqual([
      true,
      true,
      true
    ])
    removeEditableTaskControls(editor)
    expect(parent.querySelectorAll(":scope > input[type=checkbox]")).toHaveLength(
      0
    )
    expect(child.querySelectorAll(":scope > input[type=checkbox]")).toHaveLength(
      0
    )
    expect(parent.firstChild).toBe(parentText)
    expect(child.firstChild).toBe(childText)
    expect(parent.querySelectorAll("li")[1]).toBe(sibling)
    expect(sibling.querySelector("input")).toBe(siblingBox)
    expect(siblingBox.checked).toBe(true)
    expect(wysiwygHtmlToMarkdown(editor.innerHTML)).toBe(
      "- parent\n    - child\n    - [x] sibling"
    )
  })

  it("retains ordinal starts, nested list structure and checked tasks after a body edit", () => {
    const editor = mountEditor(
      markdownToWysiwygHtml(
        "3. first\n4. second\n    - [x] child\n\n- [ ] next"
      )
    )
    const item = editor.querySelector("li")
    expect(item).not.toBeNull()
    item!.append(document.createTextNode(" edited"))
    expect(wysiwygHtmlToMarkdown(editor.innerHTML)).toBe(
      "3. first edited\n4. second\n    - [x] child\n\n- [ ] next"
    )
  })

  it("keeps a nonsequential explicit ordinal value", () => {
    const editor = mountEditor(markdownToWysiwygHtml("3. first\n8. next"))
    expect(wysiwygHtmlToMarkdown(editor.innerHTML)).toBe("3. first\n8. next")
  })
  it("records live checkbox state rather than its stale checked attribute", () => {
    const editor = mountEditor(markdownToWysiwygHtml("- [x] done\n- [ ] open"))
    const boxes = editor.querySelectorAll<HTMLInputElement>("input")
    boxes[0].checked = false
    boxes[1].checked = true
    syncEditableCheckboxes(editor)
    const paragraph = document.createElement("p")
    paragraph.textContent = "Edited"
    editor.append(paragraph)
    expect(wysiwygHtmlToMarkdown(editor.innerHTML)).toBe(
      "- [ ] done\n- [x] open\n\nEdited"
    )
  })

  it("adds a checkbox only to the selected list item without replacing text nodes", () => {
    const editor = mountEditor("<ul><li>first</li><li>second</li></ul>")
    const text = firstText(editor.querySelector("li"))
    placeCaret(text, 3)
    makeEditableChecklist(editor)
    expect(editor.querySelectorAll("input")).toHaveLength(1)
    expect(window.getSelection()?.anchorNode).toBe(text)
    expect(window.getSelection()?.anchorOffset).toBe(3)
  })

  it("continues a checked rich task unchecked and moves its inline suffix intact", () => {
    const editor = mountEditor(
      '<ul><li><input type="checkbox" checked contenteditable="false">hello <a href="https://example.com">suffix</a></li><li>next</li></ul>'
    )
    const link = editor.querySelector("a")
    const neighbour = editor.querySelectorAll("li")[1]
    placeCaret(firstText(editor.querySelector("li")), 6)
    expect(continueEditableChecklist(editor)).toBe(true)
    syncEditableCheckboxes(editor)
    expect(wysiwygHtmlToMarkdown(editor.innerHTML)).toBe(
      "- [x] hello\n- [ ] [suffix](https://example.com)\n- next"
    )
    expect(editor.querySelector("a")).toBe(link)
    expect(editor.querySelectorAll("li")[2]).toBe(neighbour)
    expect(window.getSelection()?.anchorNode).toBe(
      editor.querySelectorAll("li")[1]
    )
  })

  it("exits an empty task between two tasks without losing either neighbour", () => {
    const editor = mountEditor(
      '<ul><li>before</li><li><input type="checkbox"><br></li><li>after</li></ul>'
    )
    const empty = editor.querySelectorAll("li")[1]
    placeCaret(empty, 1)
    expect(continueEditableChecklist(editor)).toBe(true)
    expect(wysiwygHtmlToMarkdown(editor.innerHTML)).toBe("- before\n\n- after")
    expect(editor.children[1].tagName).toBe("P")
  })
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

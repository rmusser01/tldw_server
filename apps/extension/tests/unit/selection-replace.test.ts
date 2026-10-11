import { describe, expect, test } from "bun:test"
import {
  detectSelectionTarget,
  isSelectionTargetValid,
  replaceSelectionTarget
} from "@tldw/ui/utils/selection-replace"

const maybeTest = typeof document === "undefined" ? test.skip : test

describe("selection replace helpers", () => {
  test("replaces selection inside input target with stubs", () => {
    const element = {
      tagName: "TEXTAREA",
      value: "Hello world",
      isConnected: true,
      readOnly: false,
      disabled: false,
      focus: () => {},
      setRangeText: (replacement: string, start: number, end: number) => {
        element.value =
          element.value.slice(0, start) +
          replacement +
          element.value.slice(end)
      }
    } as HTMLTextAreaElement

    const target = {
      kind: "input" as const,
      element,
      originalValue: "Hello world",
      start: 6,
      end: 11
    }

    expect(isSelectionTargetValid(target)).toBe(true)
    const ok = replaceSelectionTarget(target, "universe")
    expect(ok).toBe(true)
    expect(element.value).toBe("Hello universe")
  })

  test("replaces selection inside contenteditable with stubs", () => {
    const originalDocument = globalThis.document
    const insertState = { inserted: "", deleted: false, collapsed: false }
    const textNode = {} as Node
    const range = {
      startContainer: textNode,
      endContainer: textNode,
      collapsed: false,
      toString: () => "world",
      deleteContents: () => {
        insertState.deleted = true
      },
      insertNode: (node: any) => {
        insertState.inserted = node?.textContent ?? ""
      },
      collapse: () => {
        insertState.collapsed = true
      }
    } as Range

    const root = {
      isConnected: true,
      isContentEditable: true,
      contains: (node: Node) => node === textNode,
      focus: () => {}
    } as HTMLElement

    ;(globalThis as any).document = {
      createTextNode: (text: string) => ({ textContent: text })
    } as Document

    const target = {
      kind: "contenteditable" as const,
      range,
      root,
      selectedText: "world"
    }

    try {
      expect(isSelectionTargetValid(target)).toBe(true)
      const ok = replaceSelectionTarget(target, "universe")
      expect(ok).toBe(true)
      expect(insertState.deleted).toBe(true)
      expect(insertState.inserted).toBe("universe")
      expect(insertState.collapsed).toBe(true)
    } finally {
      ;(globalThis as any).document = originalDocument
    }
  })

  maybeTest("replaces selection inside textarea", () => {
    const textarea = document.createElement("textarea")
    textarea.value = "Hello world"
    document.body.appendChild(textarea)
    textarea.focus()
    textarea.selectionStart = 6
    textarea.selectionEnd = 11

    const target = detectSelectionTarget(window.getSelection())
    expect(target?.kind).toBe("input")
    expect(isSelectionTargetValid(target)).toBe(true)

    const ok = replaceSelectionTarget(target!, "universe")
    expect(ok).toBe(true)
    expect(textarea.value).toBe("Hello universe")

    textarea.remove()
  })

  maybeTest("replaces selection inside contenteditable", () => {
    const root = document.createElement("div")
    root.setAttribute("contenteditable", "true")
    root.textContent = "Hello world"
    document.body.appendChild(root)

    const textNode = root.firstChild as Text
    const range = document.createRange()
    range.setStart(textNode, 6)
    range.setEnd(textNode, 11)

    const selection = window.getSelection()
    selection?.removeAllRanges()
    selection?.addRange(range)

    const target = detectSelectionTarget(selection)
    expect(target?.kind).toBe("contenteditable")
    expect(isSelectionTargetValid(target)).toBe(true)

    const ok = replaceSelectionTarget(target!, "universe")
    expect(ok).toBe(true)
    expect(root.textContent).toBe("Hello universe")

    root.remove()
    selection?.removeAllRanges()
  })
})

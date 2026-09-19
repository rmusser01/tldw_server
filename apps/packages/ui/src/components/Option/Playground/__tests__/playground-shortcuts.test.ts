// @vitest-environment jsdom
import { describe, expect, it } from "vitest"
import {
  isEditableTarget,
  resolvePlaygroundShortcutAction,
  shouldOpenShortcutsHelp
} from "../playground-shortcuts"

describe("playground-shortcuts", () => {
  it("maps Alt+Shift shortcuts to actions", () => {
    expect(
      resolvePlaygroundShortcutAction({
        altKey: true,
        shiftKey: true,
        key: "a"
      })
    ).toBe("toggle_artifacts")
    expect(
      resolvePlaygroundShortcutAction({
        altKey: true,
        shiftKey: true,
        key: "c"
      })
    ).toBe("toggle_compare")
    expect(
      resolvePlaygroundShortcutAction({
        altKey: true,
        shiftKey: true,
        key: "m"
      })
    ).toBe("toggle_modes")
  })

  it("ignores shortcuts while typing in editable controls", () => {
    const input = document.createElement("input")
    expect(
      resolvePlaygroundShortcutAction({
        altKey: true,
        shiftKey: true,
        key: "a",
        target: input
      })
    ).toBeNull()
  })

  it("rejects conflicting modifier combinations and repeats", () => {
    expect(
      resolvePlaygroundShortcutAction({
        altKey: true,
        shiftKey: true,
        ctrlKey: true,
        key: "a"
      })
    ).toBeNull()
    expect(
      resolvePlaygroundShortcutAction({
        altKey: true,
        shiftKey: true,
        key: "a",
        repeat: true
      })
    ).toBeNull()
  })
})

describe("shouldOpenShortcutsHelp", () => {
  // Regression: the "?" branch used to run before the editable-target guard,
  // so every question mark typed into the composer was swallowed by
  // preventDefault() and replaced by the shortcuts panel.
  it("does not fire for a real \"?\" keystroke inside the composer", () => {
    const composer = document.createElement("textarea")
    expect(
      shouldOpenShortcutsHelp({ shiftKey: true, key: "?", target: composer })
    ).toBe(false)
  })

  it("does not fire inside inputs, selects or contenteditable regions", () => {
    const input = document.createElement("input")
    const select = document.createElement("select")
    const rich = document.createElement("div")
    rich.contentEditable = "true"
    Object.defineProperty(rich, "isContentEditable", { value: true })
    for (const target of [input, select, rich]) {
      expect(
        shouldOpenShortcutsHelp({ shiftKey: true, key: "?", target })
      ).toBe(false)
    }
  })

  it("still fires for \"?\" outside any editable target", () => {
    expect(
      shouldOpenShortcutsHelp({
        shiftKey: true,
        key: "?",
        target: document.createElement("div")
      })
    ).toBe(true)
    expect(shouldOpenShortcutsHelp({ shiftKey: true, key: "?" })).toBe(true)
  })

  it("ignores \"?\" combined with a modifier chord", () => {
    expect(shouldOpenShortcutsHelp({ shiftKey: true, key: "?", metaKey: true })).toBe(false)
    expect(shouldOpenShortcutsHelp({ shiftKey: true, key: "?", ctrlKey: true })).toBe(false)
    expect(shouldOpenShortcutsHelp({ shiftKey: true, key: "?", altKey: true })).toBe(false)
  })

  it("requires the shift flag a physical \"?\" keystroke carries", () => {
    // A synthetic event without shiftKey is exactly why this shipped untested.
    expect(shouldOpenShortcutsHelp({ key: "?" })).toBe(false)
  })
})

describe("isEditableTarget", () => {
  it("treats inputs, textareas, selects and contenteditable as editable", () => {
    expect(isEditableTarget(document.createElement("input"))).toBe(true)
    expect(isEditableTarget(document.createElement("textarea"))).toBe(true)
    expect(isEditableTarget(document.createElement("select"))).toBe(true)
    expect(isEditableTarget(document.createElement("div"))).toBe(false)
    expect(isEditableTarget(null)).toBe(false)
    expect(isEditableTarget(undefined)).toBe(false)
  })
})

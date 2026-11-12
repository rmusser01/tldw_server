import { describe, expect, it } from "vitest"
import { isEditableTarget } from "../editable-target"

const el = (tag: string, attrs: Record<string, string> = {}) => {
  const node = document.createElement(tag)
  for (const [name, value] of Object.entries(attrs)) node.setAttribute(name, value)
  return node
}

const childOf = (parent: HTMLElement) => {
  const child = document.createElement("span")
  parent.appendChild(child)
  return child
}

describe("isEditableTarget", () => {
  it.each([
    ["input", el("input")],
    ["checkbox input", el("input", { type: "checkbox" })],
    ["textarea", el("textarea")],
    ["select", el("select")],
    ["contenteditable=true", el("div", { contenteditable: "true" })],
    ["contenteditable (empty value)", el("div", { contenteditable: "" })],
    ["contenteditable=plaintext-only", el("div", { contenteditable: "plaintext-only" })],
    ["child of contenteditable", childOf(el("div", { contenteditable: "true" }))],
    ["role=textbox", el("div", { role: "textbox" })],
    ["role=searchbox", el("div", { role: "searchbox" })],
    ["child of role=combobox", childOf(el("div", { role: "combobox" }))]
  ])("treats %s as editable", (_label, target) => {
    expect(isEditableTarget(target)).toBe(true)
  })

  it.each([
    ["plain div", el("div")],
    ["button", el("button")],
    ["contenteditable=false", el("div", { contenteditable: "false" })],
    ["null", null],
    ["undefined", undefined],
    ["window", window],
    ["document", document]
  ])("treats %s as not editable", (_label, target) => {
    expect(isEditableTarget(target as EventTarget | null | undefined)).toBe(false)
  })
})

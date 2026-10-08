// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from "vitest"
import { readFileSync } from "node:fs"
import path from "node:path"
import ts from "typescript"

const source = readFileSync(
  path.resolve(__dirname, "../PlaygroundForm.tsx"),
  "utf8"
)
const ast = ts.createSourceFile(
  "PlaygroundForm.tsx",
  source,
  ts.ScriptTarget.Latest,
  true,
  ts.ScriptKind.TSX
)
let declaration: ts.VariableDeclaration | undefined
function visit(node: ts.Node) {
  if (
    ts.isVariableDeclaration(node) &&
    /^keepComposer(?:Bottom|Focus)InView$/.test(node.name.getText(ast))
  )
    declaration = node
  ts.forEachChild(node, visit)
}
visit(ast)
if (!declaration) throw new Error("Composer visibility callback missing")
const compiled = ts.transpileModule(
  `const reveal = ${declaration.initializer!.getText(ast)};`,
  {
    compilerOptions: { target: ts.ScriptTarget.ES2022 }
  }
).outputText
const bind = new Function(
  "React",
  "stickyDockEnabled",
  "composerShellRef",
  "textareaRef",
  compiled + "return reveal;"
)

// This exercises the real callback against declared DOM/scroll platform doubles.
// Native screenshots and rectangles establish actual browser clipping separately.
function fixture(sticky = false) {
  const dock = document.createElement("div")
  dock.style.overflowY = "auto"
  dock.getBoundingClientRect = () => ({ top: 200, bottom: 500 }) as DOMRect
  Object.defineProperties(dock, {
    scrollHeight: { value: 1200 },
    clientHeight: { value: 300 }
  })
  const form = document.createElement("div")
  form.getBoundingClientRect = () => ({ top: 200, bottom: 1200 }) as DOMRect
  const input = document.createElement("textarea")
  const button = document.createElement("button")
  form.append(input, button)
  dock.append(form)
  const outside = document.createElement("button")
  document.body.append(dock, outside)
  const inputScroll = vi.fn(),
    buttonScroll = vi.fn()
  input.scrollIntoView = inputScroll
  button.scrollIntoView = buttonScroll
  const reveal = bind(
    { useCallback: (callback: () => void) => callback },
    sticky,
    { current: form },
    { current: input }
  )
  return { dock, input, button, outside, inputScroll, buttonScroll, reveal }
}

afterEach(() => {
  document.body.replaceChildren()
  vi.restoreAllMocks()
})

describe("bounded composer focus visibility", () => {
  it.each(["input", "button"] as const)(
    "reveals the currently focused %s instead of the full tall Form",
    (kind) => {
      const view = fixture()
      view[kind].focus()
      view.reveal()
      expect(
        kind === "input" ? view.inputScroll : view.buttonScroll
      ).toHaveBeenCalledWith({ block: "nearest", inline: "nearest" })
    }
  )

  it("does not scroll the composer after focus moves to an external dialog or header", () => {
    const view = fixture()
    view.input.focus()
    view.outside.focus()
    view.reveal()
    expect(view.dock.scrollTop).toBe(0)
    expect(view.inputScroll).not.toHaveBeenCalled()
  })

  it("leaves the existing sticky dock's keyboard positioning in charge", () => {
    const view = fixture(true)
    view.button.focus()
    view.reveal()
    expect(view.buttonScroll).not.toHaveBeenCalled()
  })
})

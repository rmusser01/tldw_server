import { readFileSync } from "node:fs"
import path from "node:path"
import ts from "typescript"
import { describe, expect, it, vi } from "vitest"

// Execute the real callbacks with their I/O boundaries, without mounting the legacy shell.
const source = readFileSync(path.resolve(__dirname, "../Sidebar.tsx"), "utf8")
const ast = ts.createSourceFile(
  "Sidebar.tsx",
  source,
  ts.ScriptTarget.Latest,
  true,
  ts.ScriptKind.TSX
)
const callbacks = new Map<string, ts.Expression>()
const visit = (node: ts.Node) => {
  if (
    ts.isVariableDeclaration(node) &&
    node.name.getText(ast) === "loadLocalConversation" &&
    node.initializer
  )
    callbacks.set("load", node.initializer)
  if (
    ts.isCallExpression(node) &&
    node.expression.getText(ast) === "useMutation"
  ) {
    const options = node.arguments[0]
    if (options && ts.isObjectLiteralExpression(options)) {
      const key = options.properties.find(
        (property) =>
          ts.isPropertyAssignment(property) &&
          property.name.getText(ast) === "mutationKey"
      )
      const success = options.properties.find(
        (property) =>
          ts.isPropertyAssignment(property) &&
          property.name.getText(ast) === "onSuccess"
      )
      if (
        key &&
        ts.isPropertyAssignment(key) &&
        key.initializer.getText(ast) === '["deleteHistory"]' &&
        success &&
        ts.isPropertyAssignment(success)
      )
        callbacks.set("delete", success.initializer)
    }
  }
  ts.forEachChild(node, visit)
}
visit(ast)
const execute = (name: string, values: Record<string, unknown>) => {
  const expression = callbacks.get(name)
  if (!expression) throw new Error(`Missing production ${name} callback`)
  const compiled = ts.transpileModule(`return ${expression.getText(ast)};`, {
    compilerOptions: { target: ts.ScriptTarget.ES2022 }
  }).outputText
  return new Function(...Object.keys(values), compiled)(
    ...Object.values(values)
  )
}

describe("legacy sidebar source admission", () => {
  it.each([false, true])(
    "loads on-device history only after Settings navigation accepts (%s)",
    async (accepted) => {
      const load = vi.fn(async () => true)
      const navigate = vi.fn()
      const close = vi.fn()
      const run = execute("load", {
        React: { useCallback: (callback: unknown) => callback },
        requestSettingsNavigation: () => accepted,
        openLocalConversation: load,
        navigate,
        onClose: close
      })
      expect(await run("device-chat")).toBe(accepted)
      expect(load).toHaveBeenCalledTimes(Number(accepted))
      expect(navigate).toHaveBeenCalledTimes(Number(accepted))
      expect(close).toHaveBeenCalledTimes(Number(accepted))
    }
  )
  it("does not clear a newer Temporary conversation after an older local deletion resolves", () => {
    const clear = vi.fn(() => true)
    const title = vi.fn()
    const run = execute("delete", {
      historyId: "deleted",
      useStoreMessageOption: { getState: () => ({ historyId: "temp" }) },
      clearChat: clear,
      updatePageTitle: title,
      client: { invalidateQueries: vi.fn() }
    })
    run("deleted")
    expect(clear).not.toHaveBeenCalled()
    expect(title).not.toHaveBeenCalled()
  })
  it("stops title changes when clearing the live deleted conversation is declined", () => {
    const title = vi.fn()
    const run = execute("delete", {
      historyId: "deleted",
      useStoreMessageOption: { getState: () => ({ historyId: "deleted" }) },
      clearChat: () => false,
      updatePageTitle: title,
      client: { invalidateQueries: vi.fn() }
    })
    run("deleted")
    expect(title).not.toHaveBeenCalled()
  })
})

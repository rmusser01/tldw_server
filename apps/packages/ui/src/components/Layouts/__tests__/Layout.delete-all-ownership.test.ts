import { readFileSync } from "node:fs"
import path from "node:path"
import ts from "typescript"
import { describe, expect, it, vi } from "vitest"
import { captureChatDraftSource } from "@/hooks/chat/chat-draft-transition"

const files = [
  path.resolve(__dirname, "../Layout.tsx"),
  path.resolve(
    __dirname,
    "../../../../../../tldw-frontend/components/layout/WebLayout.tsx"
  )
]
const callbackFor = (file: string) => {
  const source = readFileSync(file, "utf8")
  const ast = ts.createSourceFile(
    file,
    source,
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TSX
  )
  let callback: ts.Expression | undefined
  const visit = (node: ts.Node) => {
    if (
      ts.isJsxAttribute(node) &&
      node.name.getText(ast) === "onClick" &&
      node.initializer &&
      ts.isJsxExpression(node.initializer) &&
      node.initializer.expression?.getText(ast).includes("deleteAllChatHistory")
    )
      callback = node.initializer.expression
    ts.forEachChild(node, visit)
  }
  visit(ast)
  if (!callback)
    throw new Error(`Missing production delete-all callback: ${file}`)
  return ts.transpileModule(`return ${callback.getText(ast)};`, {
    compilerOptions: { target: ts.ScriptTarget.ES2022 }
  }).outputText
}

describe.each(files)("held delete-all ownership (%s)", (file) => {
  it.each([false, true])(
    "uses the original H1 selection after confirmation (retired=%s)",
    async (retired) => {
      let current = true
      let confirm!: (accepted: boolean) => void
      const clear = vi.fn(() => true)
      const remove = vi.fn(async () => undefined)
      const invalidate = vi.fn(async () => undefined)
      const run = new Function(
        "captureChatDraftSource",
        "captureSelectionFence",
        "confirmDanger",
        "clearChat",
        "PageAssistDatabase",
        "queryClient",
        "t",
        callbackFor(file)
      )(
        captureChatDraftSource,
        () => () => current,
        () =>
          new Promise<boolean>((resolve) => {
            confirm = resolve
          }),
        clear,
        class {
          deleteAllChatHistory = remove
        },
        { invalidateQueries: invalidate },
        (key: string) => key
      )
      const result = run()
      if (retired) current = false
      confirm(true)
      await result
      expect(clear).toHaveBeenCalledTimes(Number(!retired))
      expect(remove).toHaveBeenCalledTimes(Number(!retired))
      expect(invalidate).toHaveBeenCalledTimes(Number(!retired))
      if (!retired)
        expect(clear).toHaveBeenCalledWith({ discardTemporaryConfirmed: true })
    }
  )
})

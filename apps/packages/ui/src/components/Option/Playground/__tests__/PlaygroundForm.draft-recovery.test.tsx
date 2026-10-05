import type { RecoveredChatDraft } from "@/components/Chat/composer/hooks/useComposerText"
import { useStoreMessageOption } from "@/store/option"
import { readFileSync } from "node:fs"
import path from "node:path"
import ts from "typescript"
import { beforeEach, describe, expect, it, vi } from "vitest"

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
let callback: ts.Expression | undefined
const visit = (node: ts.Node) => {
  if (
    ts.isVariableDeclaration(node) &&
    node.name.getText(ast) === "restoreRetainedDraft"
  )
    callback = node.initializer
  ts.forEachChild(node, visit)
}
visit(ast)
if (!callback)
  throw new Error("Missing production WebUI retained-draft recovery callback")
const compiled = ts.transpileModule(
  `const restore = ${callback.getText(ast)};`,
  {
    compilerOptions: { target: ts.ScriptTarget.ES2022 }
  }
).outputText
const navigation = vi.fn(() => true)
const clearChat = vi.fn(() => {
  if (!navigation()) return false
  useStoreMessageOption.getState().startNewChatDraft()
  useStoreMessageOption.setState({
    history: [],
    messages: [],
    historyId: null,
    serverChatId: null,
    queuedMessages: []
  })
  return true
})
const restore = new Function(
  "useStoreMessageOption",
  "requestSettingsNavigation",
  "clearChat",
  `${compiled}; return restore;`
)(useStoreMessageOption, navigation, clearChat) as (
  draft: RecoveredChatDraft,
  copy: (surface: string) => Promise<boolean>
) => Promise<boolean>
const draft: RecoveredChatDraft = {
  key: "owned-record",
  surfaceKey: "tldw:playgroundChatDraft",
  content: "RETAINED TEXT",
  updatedAt: 1,
  handle: {
    ownerHash: "a".repeat(64),
    address: {
      kind: "fresh",
      id: "retained",
      temporary: true,
      scope: { type: "global" }
    }
  }
}

describe("production WebUI retained-draft recovery boundary", () => {
  beforeEach(() => {
    navigation.mockReset().mockReturnValue(true)
    clearChat.mockClear()
    useStoreMessageOption.getState().startNewChatDraft(false)
  })
  it("copies before accepting New, adopts the retained mode/address and starts without history IDs", async () => {
    const copy = vi.fn(async (surface) => {
      expect(clearChat).not.toHaveBeenCalled()
      return surface === draft.surfaceKey
    })
    expect(await restore(draft, copy)).toBe(true)
    expect(useStoreMessageOption.getState().chatDraftAddress).toEqual(
      draft.handle.address
    )
    expect(useStoreMessageOption.getState().temporaryChat).toBe(true)
    expect(useStoreMessageOption.getState().historyId).toBeNull()
    expect(useStoreMessageOption.getState().serverChatId).toBeNull()
  })
  it.each(["navigation", "copy"])(
    "keeps the source when %s is declined",
    async (boundary) => {
      const source = useStoreMessageOption.getState().chatDraftAddress
      const copy = vi.fn().mockResolvedValue(boundary !== "copy")
      if (boundary === "navigation") navigation.mockReturnValue(false)
      expect(await restore(draft, copy)).toBe(false)
      expect(clearChat).not.toHaveBeenCalled()
      expect(useStoreMessageOption.getState().chatDraftAddress).toBe(source)
    }
  )
  it.each(["New", "account"])(
    "rejects a held copy after %s changes the source",
    async (boundary) => {
      let finish!: (value: boolean) => void
      const copy = vi.fn(
        () =>
          new Promise<boolean>((resolve) => {
            finish = resolve
          })
      )
      const result = restore(draft, copy)
      if (boundary === "New")
        useStoreMessageOption.getState().startNewChatDraft()
      else window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed"))
      const accepted = useStoreMessageOption.getState().chatDraftAddress
      finish(true)
      expect(await result).toBe(false)
      expect(clearChat).not.toHaveBeenCalled()
      expect(useStoreMessageOption.getState().chatDraftAddress).toBe(accepted)
    }
  )
  it("honors navigation declined again after a checked copy", async () => {
    const source = useStoreMessageOption.getState().chatDraftAddress
    navigation.mockReturnValueOnce(true).mockReturnValueOnce(false)
    expect(await restore(draft, vi.fn().mockResolvedValue(true))).toBe(false)
    expect(useStoreMessageOption.getState().chatDraftAddress).toBe(source)
  })
})

import { readFileSync } from "node:fs"
import path from "node:path"
import ts from "typescript"
import { act, renderHook } from "@testing-library/react"
import { useHistorySelection } from "@/hooks/chat/useHistorySelection"
import { beforeEach, describe, expect, it } from "vitest"
import { useStoreMessageOption } from "@/store/option"

// Execute the production capture with the real chat store. The large Form is
// not mounted here; composer account restoration is covered in its own suite.
const ast = ts.createSourceFile("form.tsx", readFileSync(path.resolve(__dirname, "../form.tsx"), "utf8"), ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX)
let callback: ts.Expression | undefined
let selectionCapture: ts.Expression | undefined
const visit = (node: ts.Node) => {
  if (ts.isVariableDeclaration(node) && node.name.getText(ast) === "captureHandoffGuard") callback = node.initializer
  if (ts.isVariableDeclaration(node) && node.name.getText(ast) === "handoffSelectionIsCurrent") selectionCapture = node.initializer
  ts.forEachChild(node, visit)
}
visit(ast)
const renderedCapture = (owner: () => boolean, input: () => number, selection: () => boolean, fenceFactory = () => selection) => {
  const rendered = useStoreMessageOption.getState()
  const inputRevision = input()
  expect(callback, "Missing production source ownership capture").toBeDefined()
  const code = ts.transpileModule(`${selectionCapture ? `const handoffSelectionIsCurrent = ${selectionCapture.getText(ast)};` : ""} const captureHandoffGuard = ${callback!.getText(ast)}; return captureHandoffGuard;`, { compilerOptions: { target: ts.ScriptTarget.ES2022 } }).outputText
  return new Function("React", "useStoreMessageOption", "isDraftOwnerCurrent", "getHandoffInputRevision", "greetingHistorySelection", "draftAddress", "sourceDraftRevision", "historyId", "serverChatId", "handoffInputRevision", "handoffSourceKey", code)(
    { useCallback: (fn: unknown) => fn, useMemo: (fn: () => unknown) => fn() }, useStoreMessageOption, owner, input, { fence: fenceFactory }, rendered.chatDraftAddress, rendered.chatDraftRevision, rendered.historyId, rendered.serverChatId, inputRevision, "rendered-source"
  ) as () => () => boolean
}

const capture = (owner: () => boolean, input: () => number, selection: () => boolean) => renderedCapture(owner, input, selection)()

describe("production sidepanel copy source fence", () => {
  beforeEach(() => useStoreMessageOption.getState().startNewChatDraft(false))
  it.each(["New", "ABA", "account", "input", "history"])("retires a held source copy after %s changes", boundary => {
    const address = useStoreMessageOption.getState().chatDraftAddress!
    let owned = true
    let revision = 0
    let selected = true
    const guard = capture(() => owned, () => revision, () => selected)
    expect(guard()).toBe(true)
    if (boundary === "account") owned = false
    else if (boundary === "input") revision++
    else if (boundary === "history") selected = false
    else {
      useStoreMessageOption.getState().startNewChatDraft(false)
      if (boundary === "ABA") useStoreMessageOption.getState().setChatDraftAddress(address)
    }
    expect(guard()).toBe(false)
  })
  it.each(["New", "ABA", "input"])("refuses a stale rendered action invoked after synchronous %s before rerender", boundary => {
    const address = useStoreMessageOption.getState().chatDraftAddress!
    let revision = 0
    const renderedAction = renderedCapture(() => true, () => revision, () => true)
    if (boundary === "input") revision++
    else {
      useStoreMessageOption.getState().startNewChatDraft(false)
      if (boundary === "ABA") useStoreMessageOption.getState().setChatDraftAddress(address)
    }
    expect(renderedAction()()).toBe(false)
  })
  it("refuses a stale rendered action after H1 starts another load before IDs or React commit change", () => {
    const hook = renderHook(() => useHistorySelection())
    const renderedController = hook.result.current
    const renderedAction = renderedCapture(() => true, () => 0, () => true, renderedController.fence)
    expect(renderedAction()()).toBe(true)
    act(() => {
      renderedController.beginLoad()
      // Assert before React publishes the next controller value.
      expect(renderedAction()()).toBe(false)
    })
    hook.unmount()
  })
  it("does not retire a copy for an unrelated retrieval toggle", () => {
    const guard = capture(() => true, () => 0, () => true)
    useStoreMessageOption.setState({ fileRetrievalEnabled: !useStoreMessageOption.getState().fileRetrievalEnabled })
    expect(guard()).toBe(true)
  })
})

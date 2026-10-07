import React from "react"
import { render, screen } from "@testing-library/react"
import { usePlaygroundSessionStore } from "@/store/playground-session"
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
const handoffCallbacks = new Map<string, ts.Expression>()
let handoffReview: ts.ConditionalExpression | undefined
const visit = (node: ts.Node) => {
  if (
    ts.isVariableDeclaration(node) &&
    node.name.getText(ast) === "restoreRetainedDraft"
  )
    callback = node.initializer
  if (ts.isVariableDeclaration(node) && node.initializer && ["captureAssistantActionGuard", "captureHandoffGuard"].includes(node.name.getText(ast)))
    handoffCallbacks.set(node.name.getText(ast), node.initializer)
  if (ts.isConditionalExpression(node) && node.condition.getText(ast) === "sidepanelHandoffConflict") handoffReview = node
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


describe("production Form copied-draft ownership and review boundaries", () => {
  const guardCode = ts.transpileModule(Array.from(handoffCallbacks, ([name, value]) => `const ${name} = ${value.getText(ast)};`).join("\n"), {
    compilerOptions: { target: ts.ScriptTarget.ES2022 }
  }).outputText
  const capture = (ownerCurrent: () => boolean, actionRevision = { current: 0 }, historySelection: { fence: () => () => boolean } | null = null) => new Function("React", "captureSelectionFence", "assistantOwnerIsCurrent", "assistantActionRevisionRef", "assistantActionMountedRef", "usePlaygroundSessionStore", "useStoreMessageOption", "isDraftOwnerCurrent", "historySelection",
    `${guardCode}; return captureHandoffGuard();`)(
    { useCallback: (callback: unknown) => callback }, undefined, () => true, actionRevision, { current: true }, usePlaygroundSessionStore, useStoreMessageOption, ownerCurrent, historySelection) as () => boolean

  it.each(["New", "ABA", "account"])("retires a captured production handoff guard after %s", boundary => {
    useStoreMessageOption.getState().startNewChatDraft(false)
    const address = useStoreMessageOption.getState().chatDraftAddress!
    let ownerCurrent = true
    const guard = capture(() => ownerCurrent)
    expect(guard()).toBe(true)
    if (boundary === "account") ownerCurrent = false
    else {
      useStoreMessageOption.getState().startNewChatDraft(false)
      if (boundary === "ABA") useStoreMessageOption.getState().setChatDraftAddress(address)
    }
    expect(guard()).toBe(false)
  })

  it("retires a held destination guard when its selected history fence changes", () => {
    useStoreMessageOption.getState().startNewChatDraft(false)
    let selectedHistoryCurrent = true
    const guard = capture(() => true, { current: 0 }, { fence: () => () => selectedHistoryCurrent })
    expect(guard()).toBe(true)
    selectedHistoryCurrent = false
    expect(guard()).toBe(false)
  })

  it("keeps the destination guard valid when only file retrieval or assistant preparation changes", () => {
    useStoreMessageOption.getState().startNewChatDraft(false)
    const actionRevision = { current: 0 }
    const guard = capture(() => true, actionRevision)
    usePlaygroundSessionStore.getState().markSourceSelectionIntent()
    actionRevision.current++
    expect(guard()).toBe(true)
  })

  it("renders the exact copied text as plain text before an explicit choice", () => {
    if (!handoffReview) throw new Error("Missing production Form handoff review")
    const compiledReview = ts.transpileModule(`const review = ${handoffReview.getText(ast)};`, {
      compilerOptions: { target: ts.ScriptTarget.ES2022, jsx: ts.JsxEmit.React }
    }).outputText
    const copy = "  <script>untrusted</script>\n\nCopied draft  "
    const review = new Function("React", "t", "sidepanelHandoffConflict", "sidepanelImportPending", "sidepanelImportFeedback", "insertHandoffDraft", "replaceWithHandoffDraft", "cancelHandoffImport", `${compiledReview}; return review;`)(
      React, (_key: string, fallback: string) => fallback, { handoff: { draft: { text: copy } } }, false, null, vi.fn(), vi.fn(), vi.fn())
    const view = render(review)
    expect(view.container.querySelector("pre")?.textContent).toBe(copy)
    expect(view.container.querySelector("script")).toBeNull()
    expect(screen.getByText("Copied draft")).toBeVisible()
    expect(screen.getByRole("button", { name: "Insert handoff draft" })).toBeEnabled()
  })
})

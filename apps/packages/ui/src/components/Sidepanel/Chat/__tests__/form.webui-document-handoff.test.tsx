import { readFileSync } from "node:fs"
import path from "node:path"
import React from "react"
import { act, renderHook, waitFor } from "@testing-library/react"
import ts from "typescript"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { useStoreMessageOption } from "@/store/option"

const api = { createDocumentUploadDraft: vi.fn() }
const notification = { error: vi.fn() }
const t = (key: string, fallback?: string) => fallback ?? key
const source = readFileSync(path.resolve(__dirname, "../form.tsx"), "utf8")
const ast = ts.createSourceFile("form.tsx", source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX)
const nodes: ts.Node[] = []
const visit = (node: ts.Node) => { nodes.push(node); ts.forEachChild(node, visit) }
visit(ast)
const declaration = (name: string) => nodes.find((node): node is ts.VariableDeclaration =>
  ts.isVariableDeclaration(node) && node.name.getText(ast) === name)
const callback = declaration("openChatInWebUiWithDocumentHandoff")?.initializer
if (!callback) throw new Error("Missing production WebUI document handoff callback")
const controller = declaration("documentWebUiHandoffRef")
const cleanup = nodes.find(node => ts.isExpressionStatement(node) &&
  node.getText(ast).includes("React.useEffect(() => () => documentWebUiHandoffRef.current?.abort()"))
// Execute the production callback and its lifecycle cleanup. Keep unrelated
// speech, model and ingestion hooks outside this focused boundary regression.
const compiled = ts.transpileModule(`function useHandoff(contextFiles, onOpenChatInWebUi, currentDocumentSelectionKeyRef) {
  ${controller ? `const ${controller.getText(ast)};` : ""}
  ${cleanup?.getText(ast) ?? ""}
  const open = ${callback.getText(ast)};
  return open;
}`, { compilerOptions: { target: ts.ScriptTarget.ES2022 } }).outputText
const useHandoff = new Function("React", "tldwClient", "useStoreMessageOption", "notification", "t",
  `${compiled}; return useHandoff;`)(React, api, useStoreMessageOption, notification, t) as (
    files: string[], onOpen: (overrides?: { chatDocumentDraftId: string }) => Promise<void>,
    selection: React.MutableRefObject<string>
  ) => () => Promise<void>

const mount = (files = ["document"]) => {
  const onOpen = vi.fn(async () => {})
  const selection = { current: "current-document-selection" }
  const view = renderHook(() => useHandoff(files, onOpen, selection))
  return { ...view, onOpen, selection }
}

describe("production compact WebUI document handoff", () => {
  beforeEach(() => {
    api.createDocumentUploadDraft.mockReset()
    notification.error.mockReset()
    useStoreMessageOption.getState().startNewChatDraft(false)
  })

  it("opens the current document draft once after its server acknowledgement", async () => {
    api.createDocumentUploadDraft.mockResolvedValue({ draft_id: "document-draft" })
    const view = mount()
    await act(async () => { await view.result.current() })
    expect(api.createDocumentUploadDraft).toHaveBeenCalledWith({ payload: {
      source: "sidepanel-chat-document-upload", files: ["document"] } })
    expect(view.onOpen).toHaveBeenCalledExactlyOnceWith({ chatDocumentDraftId: "document-draft" })
  })

  it("opens directly when there are no attached documents", async () => {
    const view = mount([])
    await act(async () => { await view.result.current() })
    expect(api.createDocumentUploadDraft).not.toHaveBeenCalled()
    expect(view.onOpen).toHaveBeenCalledExactlyOnceWith()
  })

  it.each(["New", "Temporary", "account", "unmount", "documents"])(
    "does not open a delayed document handoff after %s", async boundary => {
      let finish!: (value: { draft_id: string }) => void
      api.createDocumentUploadDraft.mockImplementation(() => new Promise(resolve => { finish = resolve }))
      const view = mount()
      act(() => { void view.result.current() })
      await waitFor(() => expect(finish).toBeTypeOf("function"))
      act(() => {
        if (boundary === "New") useStoreMessageOption.getState().startNewChatDraft(false)
        if (boundary === "Temporary") useStoreMessageOption.getState().setTemporaryChat(true)
        if (boundary === "account") window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed"))
        if (boundary === "unmount") view.unmount()
        if (boundary === "documents") view.selection.current = "replacement-document-selection"
      })
      await act(async () => { finish({ draft_id: "old-document-draft" }) })
      expect(view.onOpen).not.toHaveBeenCalled()
      expect(notification.error).not.toHaveBeenCalled()
    })

  it("reports a current preparation failure without opening a partial handoff", async () => {
    api.createDocumentUploadDraft.mockRejectedValue(new Error("Document preparation failed"))
    const view = mount()
    await act(async () => { await view.result.current() })
    expect(view.onOpen).not.toHaveBeenCalled()
    expect(notification.error).toHaveBeenCalledOnce()
  })
})

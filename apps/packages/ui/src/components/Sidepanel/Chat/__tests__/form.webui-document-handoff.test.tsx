import { readFileSync } from "node:fs"
import path from "node:path"
import React from "react"
import { act, render, renderHook, waitFor } from "@testing-library/react"
import ts from "typescript"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { useSimpleForm } from "@/hooks/useSimpleForm"
import { useStoreMessageOption } from "@/store/option"
import type { SidepanelWebUiOpenSource } from "../form"

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
  /React\.use(?:Layout)?Effect\(\(\) => \(\) => documentWebUiHandoffRef\.current\?\.abort\(\)/.test(node.getText(ast)))
// Execute the production callback and its lifecycle cleanup. Keep unrelated
// speech, model and ingestion hooks outside this focused boundary regression.
const compiled = ts.transpileModule(`function useHandoff(contextFiles, onOpenChatInWebUi, currentDocumentSelectionKeyRef, text, ready, selectionIsCurrent) {
  const draftReady = ready;
  const form = text.form;
  const currentDocumentSelectionKey = currentDocumentSelectionKeyRef.current;
  const revision = form.getRevision();
  const rendered = useStoreMessageOption.getState();
  const captureHandoffGuard = () => () => form.getRevision() === revision && selectionIsCurrent() &&
    useStoreMessageOption.getState().chatDraftRevision === rendered.chatDraftRevision &&
    useStoreMessageOption.getState().chatDraftAddress === rendered.chatDraftAddress;
  ${controller ? `const ${controller.getText(ast)};` : ""}
  ${cleanup?.getText(ast) ?? ""}
  const open = ${callback.getText(ast)};
  return open;
}`, { compilerOptions: { target: ts.ScriptTarget.ES2022 } }).outputText
const useHandoff = new Function("React", "tldwClient", "useStoreMessageOption", "notification", "t",
  `${compiled}; return useHandoff;`)(React, api, useStoreMessageOption, notification, t) as (
    files: string[], onOpen: (overrides: { chatDocumentDraftId: string } | undefined, source: SidepanelWebUiOpenSource) => Promise<void>,
    selection: React.MutableRefObject<string>, text: { form: ReturnType<typeof useSimpleForm> }, ready: boolean, selectionIsCurrent: () => boolean
  ) => () => Promise<boolean>

const mount = (files = ["document"], ready = true, selected = () => true) => {
  const onOpen = vi.fn(async (_overrides: { chatDocumentDraftId: string } | undefined, _source: SidepanelWebUiOpenSource) => {})
  const selection = { current: "current-document-selection" }
  let form!: ReturnType<typeof useSimpleForm<{ message: string }>>
  const view = renderHook(() => {
    form = useSimpleForm({ initialValues: { message: "  EXACT DRAFT  " } })
    return useHandoff(files, onOpen, selection, { form }, ready, selected)
  })
  return { ...view, onOpen, selection, edit: (value: string) => form.setFieldValue("message", value) }
}

beforeEach(() => {
  api.createDocumentUploadDraft.mockReset()
  notification.error.mockReset()
  useStoreMessageOption.getState().startNewChatDraft(false)
})

describe("production compact WebUI document handoff", () => {

  it("opens the current document draft once after its server acknowledgement", async () => {
    api.createDocumentUploadDraft.mockResolvedValue({ draft_id: "document-draft" })
    const view = mount()
    await act(async () => { await view.result.current() })
    expect(api.createDocumentUploadDraft).toHaveBeenCalledWith({ payload: {
      source: "sidepanel-chat-document-upload", files: ["document"] } })
    expect(view.onOpen).toHaveBeenCalledExactlyOnceWith({ chatDocumentDraftId: "document-draft" }, expect.objectContaining({ draft: "  EXACT DRAFT  ", isCurrent: expect.any(Function) }))
  })

  it("opens directly when there are no attached documents", async () => {
    const view = mount([])
    await act(async () => { await view.result.current() })
    expect(api.createDocumentUploadDraft).not.toHaveBeenCalled()
    expect(view.onOpen).toHaveBeenCalledExactlyOnceWith(undefined, expect.objectContaining({ draft: "  EXACT DRAFT  ", isCurrent: expect.any(Function) }))
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

  it.each([{ files: [] }, { files: ["document"] }])("waits for actual draft readiness before opening files $files", async ({ files }) => {
    const view = mount(files, false)
    await act(async () => { await view.result.current() })
    expect(api.createDocumentUploadDraft).not.toHaveBeenCalled()
    expect(view.onOpen).not.toHaveBeenCalled()
  })

  it.each(["input", "edit undo", "H1"])("retires document preparation after %s", async boundary => {
    let finish!: (value: { draft_id: string }) => void
    let selected = true
    api.createDocumentUploadDraft.mockImplementation(() => new Promise(resolve => { finish = resolve }))
    const view = mount(["document"], true, () => selected)
    act(() => { void view.result.current() })
    if (boundary === "H1") selected = false
    else act(() => { view.edit("NEW TEXT"); if (boundary === "edit undo") view.edit("  EXACT DRAFT  ") })
    await act(async () => { finish({ draft_id: "old-document" }) })
    expect(view.onOpen).not.toHaveBeenCalled()
    expect(notification.error).not.toHaveBeenCalled()
  })

  it("rejects old rendered files after their selection changes before React commits", async () => {
    api.createDocumentUploadDraft.mockResolvedValue({ draft_id: "wrong-files" })
    const view = mount()
    const open = view.result.current
    view.selection.current = "new-document-selection"
    await act(async () => { await open() })
    expect(api.createDocumentUploadDraft).not.toHaveBeenCalled()
  })

  it("reports a current preparation failure without opening a partial handoff", async () => {
    api.createDocumentUploadDraft.mockRejectedValue(new Error("Document preparation failed"))
    const view = mount()
    await act(async () => { await view.result.current() })
    expect(view.onOpen).not.toHaveBeenCalled()
    expect(notification.error).toHaveBeenCalledOnce()
  })
})


it("invalidates the captured opener during replacement layout work", async () => {
  let current!: () => boolean
  let currentDuringReplacement: boolean | undefined
  let finish!: () => void
  const onOpen = vi.fn((_overrides, source) => {
    current = source.isCurrent
    return new Promise<void>(resolve => { finish = resolve })
  })
  const Child = () => {
    const form = useSimpleForm({ initialValues: { message: "SOURCE" } })
    const selection = React.useRef("files")
    const open = useHandoff([], onOpen, selection, { form }, true, () => true)
    return <button onClick={() => { void open() }}>Open</button>
  }
  const Parent = ({ visible }: { visible: boolean }) => {
    React.useLayoutEffect(() => {
      if (!visible) currentDuringReplacement = current()
    }, [visible])
    return visible ? <Child /> : null
  }
  const view = render(<Parent visible />)
  act(() => { view.getByRole("button", { name: "Open" }).click() })
  view.rerender(<Parent visible={false} />)
  expect(currentDuringReplacement).toBe(false)
  await act(async () => { finish() })
})


it("reports retirement when another entry point supersedes the same source opening", async () => {
  const view = mount([])
  const finishes: Array<() => void> = []
  view.onOpen.mockImplementation(() => new Promise<void>(resolve => { finishes.push(resolve) }))
  let first!: Promise<unknown>
  let second!: Promise<unknown>
  act(() => { first = view.result.current(); second = view.result.current() })
  let firstResult: unknown
  let secondResult: unknown
  await act(async () => { finishes[0](); firstResult = await first })
  expect(firstResult).toBe(false)
  await act(async () => { finishes[1](); secondResult = await second })
  expect(secondResult).toBe(true)
})


it.each(["New", "Temporary ABA", "edit undo", "H1"])("refuses a stale rendered opening before I/O after %s", async boundary => {
  let selected = true
  const view = mount(["document"], true, () => selected)
  const open = view.result.current
  act(() => {
    if (boundary === "New") useStoreMessageOption.getState().startNewChatDraft(false)
    if (boundary === "Temporary ABA") { useStoreMessageOption.getState().setTemporaryChat(true); useStoreMessageOption.getState().setTemporaryChat(false) }
    if (boundary === "edit undo") { view.edit("EDIT"); view.edit("  EXACT DRAFT  ") }
    if (boundary === "H1") selected = false
  })
  let opened: boolean | undefined
  await act(async () => { opened = await open() })
  expect(opened).toBe(false)
  expect(api.createDocumentUploadDraft).not.toHaveBeenCalled()
  expect(view.onOpen).not.toHaveBeenCalled()
})

it("keeps the exact initial draft and setup through document preparation", async () => {
  useStoreMessageOption.getState().setSelectedModel("initial-model")
  let finish!: (value: { draft_id: string }) => void
  api.createDocumentUploadDraft.mockImplementation(() => new Promise(resolve => { finish = resolve }))
  const view = mount()
  act(() => { void view.result.current() })
  act(() => { useStoreMessageOption.getState().setSelectedModel("new-model") })
  await act(async () => { finish({ draft_id: "prepared" }) })
  expect(view.onOpen.mock.calls[0][1]).toMatchObject({ draft: "  EXACT DRAFT  ", state: { selectedModel: "initial-model" } })
})

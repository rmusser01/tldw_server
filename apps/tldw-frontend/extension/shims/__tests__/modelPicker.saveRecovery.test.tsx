import React from "react"
import { readFileSync } from "node:fs"
import path from "node:path"
import ts from "typescript"
import { findUnavailableChatModel, normalizeChatModelId } from "@/utils/chat-model-availability"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { ModelSelect } from "@/components/Common/ModelSelect"
import { SelectedModelSaveFeedback } from "@/components/Common/SelectedModelSaveFeedback"
import { useSelectedModel } from "@/hooks/chat/useSelectedModel"
import { useModelSelector } from "@/hooks/playground/useModelSelector"
import { useStoreMessageOption } from "@/store/option"
import { useStoreChatModelSettings } from "@/store/model"

const catalog = vi.hoisted(() => ({ isLoading: false, isError: false }))
const models = [{ model: "new-model", provider: "openai", nickname: "New model" }]
vi.mock("@/hooks/useChatModelCatalog", () => ({
  useChatModelCatalog: () => ({ data: catalog.isLoading ? undefined : models, ...catalog, refetch: vi.fn() })
}))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string, fallback?: unknown) => typeof fallback === "string" ? fallback : key })
}))
vi.mock("@/components/Common/ProviderIcon", () => ({ ProviderIcons: () => null }))
vi.mock("antd", () => ({
  Avatar: () => null, Input: () => null, Select: () => null,
  Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  Dropdown: ({ children, menu }: { children: React.ReactNode; menu: { items: any[] } }) => (
    <div>{children}{menu.items.flatMap(item => item.children ?? [item]).filter(item => item.onClick).map(item =>
      <div role="menuitem" key={item.key} onClick={item.onClick}>{item.label}</div>
    )}</div>
  )
}))

const FullPicker = () => {
  const preference = useSelectedModel()
  const picker = useModelSelector({
    composerModels: models,
    selectedModel: preference.selectedModel,
    setSelectedModel: preference.setSelectedModel,
    navigate: () => undefined
  })
  return <>
    <SelectedModelSaveFeedback failed={preference.selectedModelSaveFailed} onRetry={preference.retrySelectedModelSave} />
    {picker.modelDropdownMenuItems.flatMap(item => item.children ?? [item]).filter(item => item.onClick).map(item =>
      <div role="menuitem" key={item.key} onClick={() => item.onClick?.()}>{item.label}</div>
    )}
  </>
}
// Execute the actual inline composer JSX so its conditional lifetime is covered.
const formSource = readFileSync(path.resolve(__dirname, "../../../../packages/ui/src/components/Sidepanel/Chat/form.tsx"), "utf8")
const formAst = ts.createSourceFile("form.tsx", formSource, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX)
let inlineSource: string | undefined
const visit = (node: ts.Node) => {
  if (ts.isVariableDeclaration(node) && node.name.getText(formAst) === "composerInlineMessagesNode") {
    inlineSource = node.initializer?.getText(formAst)
  }
  ts.forEachChild(node, visit)
}
visit(formAst)
if (!inlineSource) throw new Error("Missing Compact inline composer")
const compactInline = new Function("React", "ModelSelect", "normalizeChatModelId", "findUnavailableChatModel",
  ts.transpileModule(`return function(selectedModel) {
    const isConnectionReady = true, isProMode = false, streaming = false;
    const personaSubmitDisabledReason = null;
    const availableChatModelIds = new Set(["openai:new-model"]);
    const form = { errors: {} };
    return ${inlineSource};
  }`, { compilerOptions: { jsx: ts.JsxEmit.React, target: ts.ScriptTarget.ES2022 } }).outputText
)(React, ModelSelect, normalizeChatModelId, findUnavailableChatModel) as (model: string | null) => React.ReactNode
const CompactHost = () => {
  const { selectedModel } = useSelectedModel()
  return <>{compactInline(selectedModel)}</>
}
beforeEach(() => {
  catalog.isLoading = false
  catalog.isError = false
  window.localStorage.clear()
  window.localStorage.setItem("selectedModel", JSON.stringify("openai:old-model"))
  useStoreMessageOption.getState().setSelectedModel("openai:old-model")
  useStoreChatModelSettings.getState().reset()
})
afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  window.localStorage.clear()
})
describe.each(["full", "compact"] as const)("actual %s picker save recovery", surface => {
  it("shows failed physical persistence and retries without a new selection", async () => {
    render(surface === "full" ? <FullPicker /> : <CompactHost />)
    const choice = await screen.findByRole("menuitem", { name: /New model/i })
    await waitFor(() => expect(useStoreMessageOption.getState().selectedModelStorageHandled).toBe(false))
    const prototype = Object.getPrototypeOf(window.localStorage) as globalThis.Storage
    const write = prototype.setItem
    let denied = true
    vi.spyOn(prototype, "setItem").mockImplementation(function(this: globalThis.Storage, key, value) {
      if (key === "selectedModel" && denied) throw new Error("Physical preference storage denied")
      return write.call(this, key, value)
    })
    fireEvent.click(choice)
    if (surface === "compact") expect(screen.queryByRole("menuitem", { name: /New model/i })).not.toBeInTheDocument()
    expect(await screen.findByRole("alert")).toHaveTextContent("Your preference could not be saved")
    expect(window.localStorage.getItem("selectedModel")).toBe(JSON.stringify("openai:old-model"))
    const revision = useStoreMessageOption.getState().selectedModelRevision
    denied = false
    fireEvent.click(screen.getByRole("button", { name: "Retry saving model" }))
    await waitFor(() => expect(screen.queryByRole("alert")).not.toBeInTheDocument())
    expect(useStoreMessageOption.getState().selectedModelRevision).toBe(revision)
    expect(window.localStorage.getItem("selectedModel")).toBe(JSON.stringify("openai:new-model"))
  })
  it("retires failed feedback after a same-value selection by another owner", async () => {
    render(surface === "full" ? <FullPicker /> : <CompactHost />)
    const choice = await screen.findByRole("menuitem", { name: /New model/i })
    const prototype = Object.getPrototypeOf(window.localStorage) as globalThis.Storage
    const write = prototype.setItem
    vi.spyOn(prototype, "setItem").mockImplementation(function(this: globalThis.Storage, key, value) {
      if (key === "selectedModel") throw new Error("Physical preference storage denied")
      return write.call(this, key, value)
    })
    fireEvent.click(choice)
    await screen.findByRole("alert")
    act(() => useStoreMessageOption.getState().setSelectedModel("openai:new-model", { storageHandled: true }))
    await waitFor(() => expect(screen.queryByRole("alert")).not.toBeInTheDocument())
    expect(window.localStorage.getItem("selectedModel")).toBe(JSON.stringify("openai:old-model"))
  })
})

for (const state of ["isLoading", "isError"] as const) {
  it(`keeps failed model save recovery visible during catalog ${state}`, async () => {
    const view = render(<ModelSelect />)
    const choice = await screen.findByRole("menuitem", { name: /New model/i })
    const prototype = Object.getPrototypeOf(window.localStorage) as globalThis.Storage
    vi.spyOn(prototype, "setItem").mockImplementation(() => { throw new Error("Physical preference storage denied") })
    fireEvent.click(choice)
    await screen.findByRole("alert")
    catalog[state] = true
    view.rerender(<ModelSelect />)
    expect(screen.getByRole("alert")).toHaveTextContent("Your preference could not be saved")
    expect(screen.getByRole("button", { name: "Retry saving model" })).toBeEnabled()
  })
}

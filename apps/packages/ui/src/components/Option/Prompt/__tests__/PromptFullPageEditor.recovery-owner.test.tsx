import React from "react"
import { afterEach, beforeEach, expect, it, vi } from "vitest"
import { fireEvent, render, screen } from "@testing-library/react"
import { PromptFullPageEditor } from "../PromptFullPageEditor"

vi.mock("react-i18next", () => ({ useTranslation: () => ({ t: (key: string, options?: { defaultValue?: string }) => options?.defaultValue ?? key }) }))
vi.mock("../PromptEditorPreview", () => ({ PromptEditorPreview: () => null }))
vi.mock("@/utils/storage-guard", () => ({ checkStorageBeforeWrite: () => ({ recommendation: null }), notifyStorageWrite: vi.fn() }))
vi.mock("@/services/prompts-api", () => ({ previewStructuredPromptServer: vi.fn() }))

const props = { open: true, mode: "create" as const, onClose: vi.fn(), onSubmit: vi.fn(), isLoading: false, allTags: [] }
beforeEach(() => { localStorage.clear() })
afterEach(() => { vi.restoreAllMocks() })
const savedDrafts = () => Array.from({ length: localStorage.length }, (_, index) => localStorage.key(index)!).filter(key => key.startsWith("tldw-prompt-fullpage-draft-")).map(key => JSON.parse(localStorage.getItem(key)!))
const observePrivateReads = (secret: string) => {
  const read = Storage.prototype.getItem
  const privateReads: string[] = []
  vi.spyOn(Storage.prototype, "getItem").mockImplementation(function (this: Storage, key: string) {
    const value = read.call(this, key)
    if (value?.includes(secret)) privateReads.push(value)
    return value
  })
  return privateReads
}

it("C01 does not recover Alice's unsaved prompt in Bob's editor", () => {
  const alice = render(<PromptFullPageEditor {...props} {...{ draftScopeKey: "alice-owner" }} />)
  fireEvent.change(screen.getByTestId("full-editor-system-prompt"), { target: { value: "Alice private instruction" } })
  expect(savedDrafts().some(draft => draft.formData.system_prompt === "Alice private instruction"), JSON.stringify(savedDrafts())).toBe(true)
  alice.unmount()
  const reads = observePrivateReads("Alice private instruction")
  render(<PromptFullPageEditor {...props} {...{ draftScopeKey: "bob-owner" }} />)
  expect(reads).toEqual([])
  expect(screen.getByTestId("full-editor-system-prompt")).toHaveValue("")
})

it("C01 recovers the latest text typed after the form was already dirty", () => {
  const first = render(<PromptFullPageEditor {...props} {...{ draftScopeKey: "alice-owner" }} />)
  fireEvent.change(screen.getByTestId("full-editor-system-prompt"), { target: { value: "First instruction" } })
  fireEvent.change(screen.getByTestId("full-editor-system-prompt"), { target: { value: "Latest instruction" } })
  expect(savedDrafts().some(draft => draft.formData.system_prompt === "Latest instruction")).toBe(true)
  first.unmount()
  render(<PromptFullPageEditor {...props} {...{ draftScopeKey: "alice-owner" }} />)
  expect(screen.getByTestId("full-editor-system-prompt")).toHaveValue("Latest instruction")
})

it("C01 leaves unassigned historical recovery data untouched and does not adopt it", () => {
  const legacy = JSON.stringify({ formType: "create", formData: { system_prompt: "Unassigned private instruction" }, savedAt: Date.now() })
  localStorage.setItem("tldw-prompt-fullpage-draft-new", legacy)
  const reads = observePrivateReads("Unassigned private instruction")
  render(<PromptFullPageEditor {...props} {...{ draftScopeKey: "alice-owner" }} />)
  expect(reads).toEqual([])
  expect(screen.getByTestId("full-editor-system-prompt")).toHaveValue("")
  expect(localStorage.getItem("tldw-prompt-fullpage-draft-new")).toBe(legacy)
})

it("C01 keeps a same-owner recovered instruction visible after recovery settles", () => {
  const first = render(<PromptFullPageEditor {...props} {...{ draftScopeKey: "alice-owner" }} />)
  fireEvent.change(screen.getByTestId("full-editor-system-prompt"), { target: { value: "Owned recovered instruction" } })
  expect(savedDrafts().some(draft => draft.formData.system_prompt === "Owned recovered instruction")).toBe(true)
  first.unmount()
  render(<PromptFullPageEditor {...props} {...{ draftScopeKey: "alice-owner" }} />)
  expect(screen.getByTestId("full-editor-system-prompt")).toHaveValue("Owned recovered instruction")
})

it("recovers structured edits after a later title edit", () => {
  const first = render(<PromptFullPageEditor {...props} draftScopeKey="alice-owner" />)
  fireEvent.change(screen.getByTestId("full-editor-user-prompt"), { target: { value: "Summarize {{topic}}" } })
  fireEvent.click(screen.getByRole("button", { name: "Convert to structured" }))
  fireEvent.click(screen.getByRole("button", { name: "managePrompts.structured.blockList.edit" }))
  fireEvent.change(screen.getByTestId("structured-block-content"), { target: { value: "Summarize {{topic}} carefully" } })
  fireEvent.change(screen.getByTestId("full-editor-name"), { target: { value: "Recovered structured prompt" } })
  first.unmount()
  render(<PromptFullPageEditor {...props} draftScopeKey="alice-owner" />)
  expect(screen.getByTestId("full-editor-name")).toHaveValue("Recovered structured prompt")
  expect(screen.getByTestId("structured-block-list")).toBeInTheDocument()
  fireEvent.click(screen.getByRole("button", { name: "managePrompts.structured.blockList.edit" }))
  expect(screen.getByTestId("structured-block-content")).toHaveValue("Summarize {{topic}} carefully")
})

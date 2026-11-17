import React from "react"
import { notification, ConfigProvider, App } from "antd"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterAll, afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { apiSend } from "@/services/api-send"
import { PromptEditorDrawer } from "../PromptEditorDrawer"
import { StudioPromptsTab } from "../StudioPromptsTab"
import { usePromptStudioStore } from "@/store/prompt-studio"

vi.mock("@/services/api-send", () => ({ apiSend: vi.fn() }))
vi.mock("react-i18next", () => ({ useTranslation: () => ({
  t: (key: string, value?: string | { defaultValue?: string }) =>
    typeof value === "string" ? value : value?.defaultValue ?? key
}) }))
let client: QueryClient | undefined
const prompt = {
  id: 71, project_id: 9, name: "Retained draft", version_number: 1,
  prompt_format: "structured", prompt_schema_version: 1,
  prompt_definition: { schema_version: 1, format: "structured", variables: [], blocks: [
    { id: "task", name: "Task", role: "user", content: "{{missing}}", enabled: true, order: 10, is_template: true }
  ] }
}
function mount(previewResponse: unknown, promptId: number | null = 71, onClose = vi.fn(), duplicate = false) {
  if (duplicate) usePromptStudioStore.getState().setSelectedProjectId(9)
  vi.mocked(apiSend).mockImplementation(async request => {
    if (request.method === "GET" && request.path.startsWith("/api/v1/prompt-studio/prompts/list/9?")) return { ok: true, status: 200, data: { success: true, data: [prompt] } } as never
    if (request.method === "GET" && request.path === "/api/v1/prompt-studio/prompts/get/71") {
      return { ok: true, status: 200, data: { success: true, data: prompt } } as never
    }
    if ((request.method === "POST" && ["/api/v1/prompt-studio/prompts/preview", "/api/v1/prompt-studio/prompts/create"].includes(request.path)) || (request.method === "PUT" && request.path === "/api/v1/prompt-studio/prompts/update/71")) return previewResponse as never
    throw new Error(`Unexpected request ${request.method} ${request.path}`)
  })
  client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  render(<ConfigProvider theme={{ token: { motion: false } }}><QueryClientProvider client={client}>
    <App>{duplicate ? <StudioPromptsTab /> : <PromptEditorDrawer open promptId={promptId} projectId={9} onClose={onClose} />}</App>
  </QueryClientProvider></ConfigProvider>)
  return (next: unknown) => { previewResponse = next }
}
beforeEach(() => {
  usePromptStudioStore.getState().resetStore()
  ConfigProvider.config({ holderRender: children => <ConfigProvider theme={{ token: { motion: false } }}>{children}</ConfigProvider> })
})
afterAll(() => { ConfigProvider.config({ holderRender: children => children }) })
afterEach(async () => {
  await act(async () => {
    cleanup()
    client?.clear()
    notification.destroy()
  })
  await waitFor(() => expect(document.querySelector(".ant-notification-notice-description")).toBeNull())
  vi.mocked(apiSend).mockReset()
})

describe("C02 preview refusal with the real service", () => {
  it.each(["http", "envelope"])("shows a %s refusal and preserves the unsaved template", async kind => {
    const error = "Unknown variable reference: missing"
    mount(kind === "http" ? { ok: false, status: 400, error }
      : { ok: true, status: 200, data: { success: false, error } })
    await screen.findByTestId("structured-preview-button")
    fireEvent.change(screen.getByLabelText(/Prompt Name/), { target: { value: "Unsaved correction" } })
    fireEvent.click(screen.getByTestId("structured-preview-button"))
    await waitFor(() => expect(screen.getByText(error)).toBeVisible())
    expect(screen.getByLabelText(/Prompt Name/)).toHaveValue("Unsaved correction")
    expect(screen.getByRole("textbox", { name: "Block content", exact: true })).toHaveValue("{{missing}}")
    await waitFor(() => expect(screen.getByTestId("structured-preview-button")).toBeEnabled())
    expect(vi.mocked(apiSend).mock.calls.map(([request]) => [request.method, request.path])).toEqual([
      ["GET", "/api/v1/prompt-studio/prompts/get/71"],
      ["POST", "/api/v1/prompt-studio/prompts/preview"]
    ])
  })
  it("renders a successful preview through the same service", async () => {
    mount({ ok: true, status: 200, data: { success: true, data: {
      prompt_format: "structured", prompt_schema_version: 1,
      assembled_messages: [{ role: "user", content: "Successful server preview" }],
      legacy_system_prompt: "", legacy_user_prompt: "Successful server preview"
    } } })
    await screen.findByTestId("structured-preview-button")
    fireEvent.click(screen.getByTestId("structured-preview-button"))
    expect((await screen.findAllByText("Successful server preview")).length).toBeGreaterThan(0)
    expect(screen.queryByText("Unknown variable reference: missing")).toBeNull()
  })

  it("clears an earlier successful preview after a refused preview in the same draft", async () => {
    const replaceResponse = mount({ ok: true, status: 200, data: { success: true, data: {
      prompt_format: "structured", prompt_schema_version: 1,
      assembled_messages: [{ role: "user", content: "Old server preview" }],
      legacy_system_prompt: "", legacy_user_prompt: "Old server preview"
    } } })
    await screen.findByTestId("structured-preview-button")
    fireEvent.change(screen.getByLabelText(/Prompt Name/), { target: { value: "Retained after refusal" } })
    fireEvent.click(screen.getByTestId("structured-preview-button"))
    expect((await screen.findAllByText("Old server preview")).length).toBeGreaterThan(0)
    await waitFor(() => expect(screen.getByTestId("structured-preview-button")).toBeEnabled())
    replaceResponse({ ok: false, status: 400, error: "Unknown variable reference: missing" })
    fireEvent.click(screen.getByTestId("structured-preview-button"))
    await waitFor(() => expect(screen.getByText("Unknown variable reference: missing")).toBeVisible())
    expect(screen.queryAllByText("Old server preview")).toHaveLength(0)
    expect(screen.getByLabelText(/Prompt Name/)).toHaveValue("Retained after refusal")
    expect(screen.getByRole("textbox", { name: "Block content", exact: true })).toHaveValue("{{missing}}")
  })

})


describe("C02 Save refusal with the real service", () => {
  it.each([
    ["create", "http"], ["create", "envelope"],
    ["update", "http"], ["update", "envelope"]
  ])("preserves the %s draft after a %s refusal", async (operation, kind) => {
    const error = "Prompt cannot be saved: invalid template"
    const onClose = vi.fn()
    mount(kind === "http" ? { ok: false, status: 400, error }
      : { ok: true, status: 200, data: { success: false, error } }, operation === "create" ? null : 71, onClose)
    const name = await screen.findByLabelText(/Prompt Name/)
    if (operation === "update") {
      await screen.findByTestId("structured-preview-button")
      fireEvent.change(screen.getByLabelText("Change Description"), { target: { value: "Retain this version description" } })
    }
    fireEvent.change(name, { target: { value: "Unsaved rejected Save" } })
    if (operation === "create") {
      fireEvent.change(screen.getByLabelText(/User Prompt/), { target: { value: "Retain this unsaved text" } })
    }
    fireEvent.click(screen.getByRole("button", { name: operation === "create" ? "Create Prompt" : "Save", exact: true }))
    await waitFor(() => expect(vi.mocked(apiSend).mock.calls.filter(([request]) => request.method === (operation === "create" ? "POST" : "PUT"))).toHaveLength(1))
    await waitFor(() => expect(screen.getByText(error)).toBeVisible())
    expect(onClose).not.toHaveBeenCalled()
    expect(name).toHaveValue("Unsaved rejected Save")
    if (operation === "create") expect(screen.getByLabelText(/User Prompt/)).toHaveValue("Retain this unsaved text")
    else expect(screen.getByRole("textbox", { name: "Block content", exact: true })).toHaveValue("{{missing}}")
    expect(screen.queryByText(operation === "create" ? "Prompt created" : "Prompt updated")).toBeNull()
    expect(vi.mocked(apiSend).mock.calls.map(([request]) => [request.method, request.path])).toEqual(operation === "create"
      ? [["POST", "/api/v1/prompt-studio/prompts/create"]]
      : [["GET", "/api/v1/prompt-studio/prompts/get/71"], ["PUT", "/api/v1/prompt-studio/prompts/update/71"]])
  })
})


describe("C02 Duplicate refusal with the real service", () => {
  it.each(["http", "envelope"])("does not announce success after a %s refusal", async kind => {
    const error = "Prompt duplication refused"
    mount(kind === "http" ? { ok: false, status: 403, error }
      : { ok: true, status: 200, data: { success: false, error } }, null, vi.fn(), true)
    const row = (await screen.findByText("Retained draft", { exact: true })).closest("tr")!
    fireEvent.click(within(row.lastElementChild as HTMLElement).getByRole("button"))
    fireEvent.click(await screen.findByRole("menuitem", { name: /Duplicate/ }))
    await waitFor(() => expect(vi.mocked(apiSend).mock.calls.filter(([request]) => request.method === "POST")).toHaveLength(1))
    await waitFor(() => expect(screen.getByText(error)).toBeVisible())
    expect(screen.queryByText("Prompt duplicated")).toBeNull()
    expect(screen.getByText("Retained draft", { exact: true })).toBeVisible()
    expect(vi.mocked(apiSend).mock.calls.filter(([request]) => request.method !== "GET").map(([request]) => [request.method, request.path, request.body])).toEqual([
      ["POST", "/api/v1/prompt-studio/prompts/create", expect.objectContaining({ name: "Retained draft (Copy)", project_id: 9, prompt_definition: prompt.prompt_definition })]
    ])
  })
})

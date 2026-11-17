import React from "react"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { ConfigProvider } from "antd"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { EvaluationsTab } from "../EvaluationsTab"
import { DatasetsTab } from "../DatasetsTab"
import { useEvaluationsStore } from "@/store/evaluations"

const mocks = vi.hoisted(() => ({
  create: vi.fn(), update: vi.fn(),
  detail: null as null | Record<string, unknown>,
}))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string, options?: { defaultValue?: string }) => options?.defaultValue || key }),
}))
vi.mock("@tanstack/react-query", () => ({ useQueryClient: () => ({ invalidateQueries: vi.fn() }) }))
vi.mock("../../hooks/useEvaluations", async (importOriginal) => ({
  ...await importOriginal<typeof import("../../hooks/useEvaluations")>(),
  useEvaluationsList: () => ({ data: { data: { data: [] } }, isLoading: false, isError: false }),
  useEvaluationDetail: () => ({ data: mocks.detail ? { data: mocks.detail } : null }),
  useEvaluationDefaults: () => ({ data: undefined }),
  useCreateEvaluation: () => ({ mutateAsync: mocks.create, isPending: false }),
  useUpdateEvaluation: () => ({ mutateAsync: mocks.update, isPending: false }),
  useDeleteEvaluation: () => ({ mutateAsync: vi.fn(), isPending: false }),
}))
vi.mock("../../hooks/useDatasets", () => ({
  useDatasetsList: () => ({ data: { data: { data: [] } }, isLoading: false }),
  useCreateDataset: () => ({ mutateAsync: vi.fn(), isPending: false }),
  useDeleteDataset: () => ({ mutateAsync: vi.fn(), isPending: false }),
  useLoadDatasetSamples: () => ({ mutateAsync: vi.fn(), isPending: false }),
  useCloseDatasetViewer: () => vi.fn(),
  parseSamplesJson: vi.fn(),
}))

beforeEach(() => {
  vi.clearAllMocks()
  useEvaluationsStore.getState().resetStore()
  mocks.detail = null
  mocks.create.mockResolvedValue({})
  mocks.update.mockResolvedValue({})
  vi.stubGlobal("ResizeObserver", class { observe() {} unobserve() {} disconnect() {} })
  vi.stubGlobal("matchMedia", vi.fn(() => ({ matches: false, addListener() {}, removeListener() {} })))
})

afterEach(() => vi.unstubAllGlobals())

function checkedControl<T extends HTMLElement>(element: T, role: string, name: string | RegExp): T {
  expect(element).toHaveRole(role)
  expect(element).toHaveAccessibleName(name)
  expect(element).toBeVisible()
  expect(element).toBeEnabled()
  expect(element.closest('[aria-hidden="true"]')).toBeNull()
  return element
}

function button(scope: Pick<typeof screen, "getByText">, name: string): HTMLButtonElement {
  const element = scope.getByText(name, { exact: true }).closest("button")
  if (!(element instanceof HTMLButtonElement)) throw new Error(`Missing button: ${name}`)
  return checkedControl(element, "button", name)
}

async function openExactMatch() {
  const user = userEvent.setup()
  render(
    <ConfigProvider theme={{ token: { motion: false } }}>
      <EvaluationsTab />
    </ConfigProvider>
  )
  fireEvent.click(button(screen, "New evaluation"))
  const wizard = within(screen.getByRole("dialog", { name: "New evaluation" }))
  const nameInput = await waitFor(() => checkedControl(wizard.getByLabelText(/Name/, { selector: "input" }), "textbox", /Name/))
  fireEvent.change(nameInput, { target: { value: "exact_run" } })
  fireEvent.change(checkedControl(wizard.getByLabelText("Description", { selector: "textarea" }), "textbox", "Description"), { target: { value: "Two known outcomes" } })
  fireEvent.mouseDown(checkedControl(wizard.getByLabelText(/Evaluation type/, { selector: "input" }), "combobox", /Evaluation type/))
  const option = screen.getByTitle("exact_match")
  await waitFor(() => expect(option).toBeVisible())
  fireEvent.click(option)
  await user.click(button(wizard, "Next"))
  return { user, wizard }
}

describe("evaluation wizard preserved form state", () => {
  it("binds the visible create key through regeneration and manual submission", async () => {
    const { user, wizard } = await openExactMatch()
    await user.click(wizard.getByRole("button", { name: /Advanced options/ }))
    const input = wizard.getByPlaceholderText("Prevents duplicate create on retry")
    const first = useEvaluationsStore.getState().evalIdempotencyKey
    expect(first).not.toBe("")
    expect(input).toHaveValue(first)
    expect(wizard.getByLabelText("Idempotency key")).toBe(input)
    await user.click(button(wizard, "Regenerate"))
    const generated = useEvaluationsStore.getState().evalIdempotencyKey
    expect(generated).not.toBe(first)
    await waitFor(() => expect(input).toHaveValue(generated))
    fireEvent.change(input, { target: { value: "manual-create-key" } })
    await user.click(button(wizard, "Next"))
    await user.click(button(wizard, "Create"))
    await waitFor(() => expect(mocks.create).toHaveBeenCalledWith(expect.objectContaining({
      idempotencyKey: "manual-create-key",
    })))
  })

  it("retains exact-match configuration when fields unmount and steps are revisited", async () => {
    const { user, wizard } = await openExactMatch()
    const control = checkedControl(await wizard.findByLabelText("Case sensitive", { selector: "button" }), "switch", "Case sensitive")
    await user.click(control)
    await user.click(button(wizard, "Next"))
    await user.click(button(wizard, "Back"))
    expect(checkedControl(await wizard.findByLabelText("Case sensitive", { selector: "button" }), "switch", "Case sensitive")).toBeChecked()
    await user.click(button(wizard, "Back"))
    expect(checkedControl(wizard.getByLabelText(/Name/, { selector: "input" }), "textbox", /Name/)).toHaveValue("exact_run")
    await user.click(button(wizard, "Next"))
    expect(checkedControl(await wizard.findByLabelText("Case sensitive", { selector: "button" }), "switch", "Case sensitive")).toBeChecked()
  })

  it("submits earlier-step identity together with the inline dataset", async () => {
    const { user, wizard } = await openExactMatch()
    await user.click(button(wizard, "Next"))
    const samples = [
      { input: { output: "ORBIT-742" }, expected: { output: "ORBIT-742" } },
      { input: { output: "ORBIT-999" }, expected: { output: "ORBIT-742" } },
    ]
    await user.click(wizard.getByRole("checkbox", { name: "Attach inline dataset instead of referencing dataset_id" }))
    fireEvent.change(wizard.getByRole("textbox"), { target: { value: JSON.stringify(samples) } })
    await user.click(wizard.getByRole("button", { name: "Create" }))
    await waitFor(() => expect(mocks.create).toHaveBeenCalledWith(expect.objectContaining({
      payload: expect.objectContaining({ name: "exact_run", description: "Two known outcomes", eval_type: "exact_match", dataset: samples }),
      idempotencyKey: expect.any(String),
    })))
  })

  it("retains saved metadata and identity when editing through all steps", async () => {
    mocks.detail = { id: "eval_owned", name: "owned", description: "Keep me", eval_type: "exact_match", eval_spec: { case_sensitive: true }, metadata: { custom: { source: "owned" } } }
    useEvaluationsStore.setState({ selectedEvalId: "eval_owned" })
    const user = userEvent.setup()
    render(
      <ConfigProvider theme={{ token: { motion: false } }}>
        <EvaluationsTab />
      </ConfigProvider>
    )
    await user.click(within(screen.getByTestId("evaluations-list-card")).getByRole("button", { name: "Edit" }))
    const wizard = within(screen.getByRole("dialog", { name: "Edit evaluation" }))
    await user.click(wizard.getByRole("button", { name: "Next" }))
    await user.click(wizard.getByRole("button", { name: "Next" }))
    await user.click(wizard.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(mocks.update).toHaveBeenCalledWith({
      evalId: "eval_owned", payload: expect.objectContaining({
        name: "owned", description: "Keep me", eval_type: "exact_match", metadata: { custom: { source: "owned" } }, eval_spec: { case_sensitive: true },
      }),
    }))
  })
})


describe("retained evaluation workspace form labels", () => {
  it.each([
    ["New dataset", "New evaluation", "my_eval_run"],
    ["New evaluation", "New dataset", "my_dataset"],
  ])("labels the visible form after canceling %s then opening %s", async (firstTitle, secondTitle, placeholder) => {
    const user = userEvent.setup()
    render(
      <ConfigProvider theme={{ token: { motion: false } }}>
        <EvaluationsTab />
        <DatasetsTab />
      </ConfigProvider>
    )
    await user.click(button(screen, firstTitle))
    const firstDialog = screen.getByRole("dialog", { name: firstTitle })
    await user.click(within(firstDialog).getByRole("button", { name: "Cancel" }))
    await waitFor(() => expect(firstDialog).not.toBeVisible())
    await user.click(button(screen, secondTitle))
    const secondDialog = screen.getByRole("dialog")
    const scope = within(secondDialog)
    await waitFor(() => expect(scope.getByText(secondTitle, { selector: ".ant-modal-title" })).toBeVisible())
    expect(scope.getByPlaceholderText(placeholder)).toHaveAccessibleName(/Name/)
    expect(scope.getByLabelText("Description", { selector: "textarea" })).toHaveAccessibleName("Description")
    expect(mocks.create).not.toHaveBeenCalled()
  })
})

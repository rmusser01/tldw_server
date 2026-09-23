import React from "react"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { EvaluationsTab } from "../EvaluationsTab"
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

async function openExactMatch() {
  const user = userEvent.setup()
  render(<EvaluationsTab />)
  await user.click(screen.getByRole("button", { name: "New evaluation", exact: true }))
  const wizard = within(screen.getByRole("dialog", { name: "New evaluation" }))
  fireEvent.change(wizard.getByRole("textbox", { name: /Name/ }), { target: { value: "exact_run" } })
  fireEvent.change(wizard.getByRole("textbox", { name: "Description" }), { target: { value: "Two known outcomes" } })
  await user.click(wizard.getByRole("combobox", { name: /Evaluation type/ }))
  await user.click(screen.getByTitle("exact_match"))
  await user.click(wizard.getByRole("button", { name: "Next" }))
  return { user, wizard }
}

describe("evaluation wizard preserved form state", () => {
  it("retains exact-match configuration when fields unmount and steps are revisited", async () => {
    const { user, wizard } = await openExactMatch()
    const control = await wizard.findByRole("switch", { name: "Case sensitive" })
    await user.click(control)
    await user.click(wizard.getByRole("button", { name: "Next" }))
    await user.click(wizard.getByRole("button", { name: "Back" }))
    expect(await wizard.findByRole("switch", { name: "Case sensitive" })).toBeChecked()
    await user.click(wizard.getByRole("button", { name: "Back" }))
    expect(wizard.getByRole("textbox", { name: /Name/ })).toHaveValue("exact_run")
    await user.click(wizard.getByRole("button", { name: "Next" }))
    expect(await wizard.findByRole("switch", { name: "Case sensitive" })).toBeChecked()
  })

  it("submits earlier-step identity together with the inline dataset", async () => {
    const { user, wizard } = await openExactMatch()
    await user.click(wizard.getByRole("button", { name: "Next" }))
    const samples = [
      { input: { output: "ORBIT-742" }, expected: { output: "ORBIT-742" } },
      { input: { output: "ORBIT-999" }, expected: { output: "ORBIT-742" } },
    ]
    await user.click(wizard.getByRole("checkbox", { name: "Attach inline dataset instead of referencing dataset_id" }))
    fireEvent.change(wizard.getByRole("textbox"), { target: { value: JSON.stringify(samples) } })
    await user.click(wizard.getByRole("button", { name: "Create", exact: true }))
    await waitFor(() => expect(mocks.create).toHaveBeenCalledWith(expect.objectContaining({
      payload: expect.objectContaining({ name: "exact_run", description: "Two known outcomes", eval_type: "exact_match", dataset: samples }),
      idempotencyKey: expect.any(String),
    })))
  })

  it("retains saved metadata and identity when editing through all steps", async () => {
    mocks.detail = { id: "eval_owned", name: "owned", description: "Keep me", eval_type: "exact_match", eval_spec: { case_sensitive: true }, metadata: { custom: { source: "owned" } } }
    useEvaluationsStore.setState({ selectedEvalId: "eval_owned" })
    const user = userEvent.setup()
    render(<EvaluationsTab />)
    await user.click(within(screen.getByTestId("evaluations-list-card")).getByRole("button", { name: "Edit", exact: true }))
    const wizard = within(screen.getByRole("dialog", { name: "Edit evaluation" }))
    await user.click(wizard.getByRole("button", { name: "Next" }))
    await user.click(wizard.getByRole("button", { name: "Next" }))
    await user.click(wizard.getByRole("button", { name: "Save", exact: true }))
    await waitFor(() => expect(mocks.update).toHaveBeenCalledWith({
      evalId: "eval_owned", payload: expect.objectContaining({
        name: "owned", description: "Keep me", eval_type: "exact_match", metadata: { custom: { source: "owned" } }, eval_spec: { case_sensitive: true },
      }),
    }))
  })
})

import React from "react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { Form, Modal } from "antd"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { DictionaryFormModal } from "../components/DictionaryFormModal"

if (typeof window.ResizeObserver === "undefined") {
  class ResizeObserverMock {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
  ;(window as any).ResizeObserver = ResizeObserverMock
  ;(globalThis as any).ResizeObserver = ResizeObserverMock
}

if (!window.matchMedia) {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    value: (query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: () => undefined,
      removeListener: () => undefined,
      addEventListener: () => undefined,
      removeEventListener: () => undefined,
      dispatchEvent: () => false
    })
  })
}

vi.mock("@/components/Common/LabelWithHelp", () => ({
  LabelWithHelp: ({ label }: { label: React.ReactNode }) => <span>{label}</span>
}))

type DictionaryFormModalHarnessProps = {
  title: string
  submitLabel: string
  onFinish: (values: any) => void
  includeActiveField?: boolean
}

const DictionaryFormModalHarness: React.FC<DictionaryFormModalHarnessProps> = ({
  title,
  submitLabel,
  onFinish,
  includeActiveField = false
}) => {
  const [form] = Form.useForm()
  return (
    <DictionaryFormModal
      title={title}
      open
      onCancel={() => undefined}
      form={form}
      onFinish={onFinish}
      submitLabel={submitLabel}
      submitLoading={false}
      includeActiveField={includeActiveField}
    />
  )
}

describe("DictionaryFormModal", () => {
  afterEach(() => {
    Modal.destroyAll()
  })

  it("submits create dictionary values with default token budget", async () => {
    const user = userEvent.setup()
    const onFinish = vi.fn()
    render(
      <DictionaryFormModalHarness
        title="Create Dictionary"
        submitLabel="Create"
        onFinish={onFinish}
      />
    )

    await user.type(screen.getByRole("textbox", { name: "Name" }), "Clinical Terms")
    await user.type(screen.getByRole("textbox", { name: "Category" }), "Medical")
    await user.click(screen.getByRole("combobox", { name: "Starter Template" }))
    await user.click(
      await screen.findByText(/Medical Abbreviations/i, {
        selector: ".ant-select-item-option-content"
      })
    )
    await user.click(screen.getByRole("combobox", { name: "Tags" }))
    await user.keyboard("clinical{Enter}abbr{Enter}")
    await user.type(
      screen.getByRole("spinbutton", { name: "Processing limit" }),
      "450"
    )
    await user.click(screen.getByRole("button", { name: "Create" }))

    await waitFor(() => {
      expect(onFinish).toHaveBeenCalledWith(
        expect.objectContaining({
          name: "Clinical Terms",
          category: "Medical",
          tags: ["clinical", "abbr"],
          starter_template: "medical_abbreviations",
          default_token_budget: 450
        })
      )
    })
    expect(screen.queryByText("Active")).not.toBeInTheDocument()
  }, 20000)

  it("renders active switch in edit mode", () => {
    render(
      <DictionaryFormModalHarness
        title="Edit Dictionary"
        submitLabel="Save"
        onFinish={vi.fn()}
        includeActiveField
      />
    )

    expect(screen.getByText("Active")).toBeInTheDocument()
    expect(screen.getByRole("switch")).toBeInTheDocument()
    expect(
      screen.queryByRole("combobox", { name: "Starter Template" })
    ).not.toBeInTheDocument()
  })
})

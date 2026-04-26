import React from "react"
import { Form } from "antd"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { describe, expect, it } from "vitest"

import { DictionaryEntryEditForm } from "../DictionaryEntryEditForm"

function DictionaryEntryEditFormHarness() {
  const [form] = Form.useForm()

  return (
    <div>
      <button
        type="button"
        onClick={() => {
          form.setFieldsValue({ probability: 0.5 })
        }}
      >
        Load advanced values
      </button>
      <DictionaryEntryEditForm
        form={form}
        updatingEntry={false}
        onSubmit={() => undefined}
        entryGroupOptions={[]}
        normalizeProbabilityValue={(value, fallback = 1) =>
          typeof value === "number" ? value : fallback
        }
        formatProbabilityFrequencyHint={(value) => `Frequency ${String(value)}`}
      />
    </div>
  )
}

describe("DictionaryEntryEditForm", () => {
  it("auto-expands advanced options when unmounted advanced fields receive values", async () => {
    render(<DictionaryEntryEditFormHarness />)

    expect(
      screen.getByRole("button", { name: "Advanced options" })
    ).toHaveAttribute("aria-expanded", "false")

    fireEvent.click(screen.getByRole("button", { name: "Load advanced values" }))

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Simple mode" })).toBeInTheDocument()
    })
    expect(screen.getByText("Frequency 0.5")).toBeInTheDocument()
  })
})

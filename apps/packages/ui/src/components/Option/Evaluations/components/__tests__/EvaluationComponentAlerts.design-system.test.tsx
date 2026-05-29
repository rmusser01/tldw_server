// @vitest-environment jsdom

import { describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import React from "react"
import { Form } from "antd"
import { CreateEvaluationWizard } from "../CreateEvaluationWizard"
import { DatasetUpload } from "../DatasetUpload"
import { RateLimitsWidget } from "../RateLimitsWidget"
import { VisualSpecBuilder } from "../VisualSpecBuilder"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      _key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string
            [key: string]: unknown
          }
    ) => {
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      if (defaultValueOrOptions?.defaultValue) {
        return defaultValueOrOptions.defaultValue.replace(
          /\{\{(\w+)\}\}/g,
          (_match, key) => String(defaultValueOrOptions[key] ?? "")
        )
      }
      return _key
    }
  })
}))

if (!(globalThis as any).ResizeObserver) {
  ;(globalThis as any).ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}

const expectDesignSystemAlert = (text: string | RegExp) => {
  const node = screen.getByText(text)
  expect(node.closest('[data-ds-component="Alert"]')).toBeInTheDocument()
}

const CreateEvaluationWizardHarness = () => {
  const [form] = Form.useForm()

  return (
    <CreateEvaluationWizard
      form={form}
      datasets={[]}
      evalDefaults={null}
      evalSpecText="{}"
      evalSpecError={null}
      inlineDatasetEnabled={false}
      inlineDatasetText=""
      evalIdempotencyKey="eval-key-1"
      onSpecChange={vi.fn()}
      onSpecError={vi.fn()}
      onInlineDatasetEnabled={vi.fn()}
      onInlineDatasetText={vi.fn()}
      onRegenerateIdempotencyKey={() => "eval-key-2"}
      onCancel={vi.fn()}
      onSubmit={vi.fn()}
    />
  )
}

describe("Evaluation component product-state alerts", () => {
  it("renders the evaluation type hint with the design-system Alert", () => {
    render(<CreateEvaluationWizardHarness />)

    expectDesignSystemAlert(/Supported: model_graded/)
  })

  it("renders dataset upload parse failures with the design-system Alert", async () => {
    const { container } = render(<DatasetUpload onSamplesLoaded={vi.fn()} />)
    const input = container.querySelector('input[type="file"]')

    expect(input).not.toBeNull()
    fireEvent.change(input!, {
      target: {
        files: [new File(["not-json"], "dataset.json", { type: "application/json" })]
      }
    })

    await waitFor(() => expect(screen.getByText("Upload failed")).toBeInTheDocument())
    expectDesignSystemAlert("Upload failed")
  })

  it("renders rate-limit error and quota snapshots with the design-system Alert", () => {
    const { rerender } = render(<RateLimitsWidget isError />)

    expectDesignSystemAlert("Unable to fetch rate limits")

    rerender(
      <RateLimitsWidget
        quotaSnapshot={{
          limitDay: 100,
          remainingDay: 75,
          limitMinute: 10,
          remainingMinute: 8,
          reset: "2026-05-29T23:00:00Z"
        }}
      />
    )

    expectDesignSystemAlert("Evaluation limits")
  })

  it("renders visual spec builder guidance and parse warnings with the design-system Alert", () => {
    const { rerender } = render(
      <VisualSpecBuilder
        evalType="ocr"
        specText="{}"
        onSpecChange={vi.fn()}
        onValidationError={vi.fn()}
      />
    )

    expectDesignSystemAlert(/This evaluation type uses JSON configuration/)

    rerender(
      <VisualSpecBuilder
        evalType="response_quality"
        specText="{"
        onSpecChange={vi.fn()}
        onValidationError={vi.fn()}
      />
    )

    expectDesignSystemAlert(/Spec JSON is invalid/)
  })
})

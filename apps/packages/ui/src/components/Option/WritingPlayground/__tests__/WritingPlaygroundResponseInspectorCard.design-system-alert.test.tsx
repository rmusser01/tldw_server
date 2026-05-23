import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { WritingPlaygroundResponseInspectorCard } from "../WritingPlaygroundResponseInspectorCard"
import type { ResponseInspectorCardProps } from "../WritingPlaygroundDiagnostics.types"

const t: ResponseInspectorCardProps["t"] = (_key, defaultValue, options) =>
  defaultValue.replace(/\{\{(\w+)\}\}/g, (_, key) => String(options?.[key] ?? ""))

const makeProps = (
  overrides: Partial<ResponseInspectorCardProps> = {}
): ResponseInspectorCardProps => ({
  t,
  responseInspectorRowsCount: 0,
  responseLogprobsCount: 0,
  settingsLogprobsEnabled: false,
  settingsDisabled: false,
  responseLogprobRowsCount: 0,
  responseLogprobTruncated: false,
  onCopyResponseInspectorJson: vi.fn(),
  onExportResponseInspectorCsv: vi.fn(),
  onClearResponseInspector: vi.fn(),
  ...overrides
})

describe("WritingPlaygroundResponseInspectorCard product-state alerts", () => {
  it("renders response inspector guidance through the design-system Alert", () => {
    render(<WritingPlaygroundResponseInspectorCard {...makeProps()} />)

    const guidance = screen.getByText(
      "Enable logprobs in generation settings to capture response token scores."
    )
    expect(guidance.closest('[data-ds-component="Alert"]')).toBeInTheDocument()
  })
})

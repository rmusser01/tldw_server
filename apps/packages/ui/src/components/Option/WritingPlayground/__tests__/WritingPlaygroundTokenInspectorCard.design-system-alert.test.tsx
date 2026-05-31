import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { WritingPlaygroundTokenInspectorCard } from "../WritingPlaygroundTokenInspectorCard"
import type { TokenInspectorCardProps } from "../WritingPlaygroundDiagnostics.types"

const t: TokenInspectorCardProps["t"] = (_key, defaultValue, options) =>
  defaultValue.replace(/\{\{(\w+)\}\}/g, (_, key) => String(options?.[key] ?? ""))

const makeProps = (
  overrides: Partial<TokenInspectorCardProps> = {}
): TokenInspectorCardProps => ({
  t,
  tokenizerName: null,
  serverSupportsTokenCount: false,
  canCountTokens: false,
  isCountingTokens: false,
  onCountTokens: vi.fn(),
  serverSupportsTokenize: false,
  canTokenizePreview: false,
  isTokenizingText: false,
  onTokenizePreview: vi.fn(),
  hasTokenCountResult: false,
  tokenCountValue: null,
  hasTokenizeResult: false,
  tokenInspectorError: null,
  tokenInspectorBusy: false,
  tokenInspectorUnavailableReason: null,
  onClearTokenInspector: vi.fn(),
  tokenPreviewRowsCount: 0,
  tokenPreviewTotal: 0,
  ...overrides
})

describe("WritingPlaygroundTokenInspectorCard product-state alerts", () => {
  it("renders unavailable reasons through the design-system Alert", () => {
    render(
      <WritingPlaygroundTokenInspectorCard
        {...makeProps({
          tokenInspectorUnavailableReason: "Token counting requires a connected server."
        })}
      />
    )

    const unavailableReason = screen.getByText("Token counting requires a connected server.")
    expect(
      unavailableReason.closest('[data-ds-component="Alert"]')
    ).toBeInTheDocument()
  })

  it("renders token inspector errors through the design-system Alert", () => {
    render(
      <WritingPlaygroundTokenInspectorCard
        {...makeProps({
          tokenInspectorError: "Unable to tokenize this draft."
        })}
      />
    )

    const errorMessage = screen.getByText("Unable to tokenize this draft.")
    expect(errorMessage.closest('[data-ds-component="Alert"]')).toBeInTheDocument()
  })
})

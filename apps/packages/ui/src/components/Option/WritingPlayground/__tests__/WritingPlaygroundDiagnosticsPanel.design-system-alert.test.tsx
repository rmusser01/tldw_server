import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { WritingPlaygroundDiagnosticsPanel } from "../WritingPlaygroundDiagnosticsPanel"
import type { WritingPlaygroundDiagnosticsPanelProps } from "../WritingPlaygroundDiagnostics.types"

const t: WritingPlaygroundDiagnosticsPanelProps["t"] = (_key, defaultValue) =>
  defaultValue

const makeProps = (
  overrides: Partial<WritingPlaygroundDiagnosticsPanelProps> = {}
): WritingPlaygroundDiagnosticsPanelProps => ({
  title: "Diagnostics",
  t,
  status: "ready",
  showOffline: false,
  showUnsupported: false,
  hasActiveSession: false,
  response: {
    enabled: false,
    responseInspectorRowsCount: 0,
    responseLogprobsCount: 0,
    settingsLogprobsEnabled: false,
    settingsDisabled: false,
    responseLogprobRowsCount: 0,
    responseLogprobTruncated: false,
    onCopyResponseInspectorJson: vi.fn(),
    onExportResponseInspectorCsv: vi.fn(),
    onClearResponseInspector: vi.fn()
  },
  token: {
    enabled: false,
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
    tokenPreviewTotal: 0
  },
  wordcloud: {
    enabled: false,
    wordcloudStatus: null,
    wordcloudStatusColor: "default",
    canGenerateWordcloud: false,
    isGeneratingWordcloud: false,
    onGenerateWordcloud: vi.fn(),
    wordcloudError: null,
    onClearWordcloud: vi.fn(),
    wordcloudWords: []
  },
  ...overrides
})

describe("WritingPlaygroundDiagnosticsPanel product-state alerts", () => {
  it("renders the offline diagnostics state through the design-system Alert", () => {
    render(
      <WritingPlaygroundDiagnosticsPanel {...makeProps({ showOffline: true })} />
    )

    const offlineTitle = screen.getByText("Server required")
    expect(offlineTitle.closest('[data-ds-component="Alert"]')).toBeInTheDocument()
  })

  it("renders the unsupported diagnostics state through the design-system Alert", () => {
    render(
      <WritingPlaygroundDiagnosticsPanel
        {...makeProps({ showUnsupported: true })}
      />
    )

    const unsupportedTitle = screen.getByText("Playground unavailable")
    expect(
      unsupportedTitle.closest('[data-ds-component="Alert"]')
    ).toBeInTheDocument()
  })
})

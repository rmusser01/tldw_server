import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { getDesignSystemState } from "@/design-system"
import { WritingPlaygroundDiagnosticsPanel } from "../WritingPlaygroundDiagnosticsPanel"
import type { WritingPlaygroundDiagnosticsPanelProps } from "../WritingPlaygroundDiagnostics.types"

const { registryReadyLabel } = vi.hoisted(() => ({
  registryReadyLabel: "Registry Ready"
}))

vi.mock("@/design-system", async (importActual) => {
  const actual = await importActual<typeof import("@/design-system")>()
  return {
    ...actual,
    getDesignSystemState: vi.fn(
      (key: Parameters<typeof actual.getDesignSystemState>[0]) => {
        const state = actual.getDesignSystemState(key)
        return key === "ready" ? { ...state, label: registryReadyLabel } : state
      }
    )
  }
})

const t: WritingPlaygroundDiagnosticsPanelProps["t"] = (_key, defaultValue) =>
  defaultValue

const makeProps = (): WritingPlaygroundDiagnosticsPanelProps => ({
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
  }
})

describe("WritingPlaygroundDiagnosticsPanel design-system state labels", () => {
  it("renders the ready diagnostics label from the design-system registry", () => {
    render(<WritingPlaygroundDiagnosticsPanel {...makeProps()} />)

    expect(screen.getByText(registryReadyLabel)).toBeInTheDocument()
    expect(getDesignSystemState).toHaveBeenCalledWith("ready")
  })
})

// @vitest-environment jsdom
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

const antdMocks = vi.hoisted(() => ({
  confirm: vi.fn()
}))

vi.mock("antd", () => ({
  Modal: { confirm: antdMocks.confirm },
  Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>
}))

vi.mock("@/services/moderation", () => ({
  setUserOverride: vi.fn().mockResolvedValue({})
}))

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeSettings(policyData: Record<string, any> | null = null) {
  return {
    draft: { piiEnabled: false, categoriesEnabled: [], persist: false },
    setDraft: vi.fn(),
    baseline: null,
    isDirty: false,
    save: vi.fn().mockResolvedValue(undefined),
    reset: vi.fn(),
    reload: vi.fn().mockResolvedValue(undefined),
    settingsQuery: { data: undefined, isLoading: false, refetch: vi.fn() },
    policyQuery: { data: policyData, isLoading: false, refetch: vi.fn() }
  }
}

function makeBlocklist() {
  return {
    rawText: "",
    setRawText: vi.fn(),
    rawLint: null,
    managedItems: [],
    managedVersion: "",
    managedLine: "",
    setManagedLine: vi.fn(),
    managedLint: null,
    loading: false,
    loadRaw: vi.fn().mockResolvedValue(undefined),
    saveRaw: vi.fn().mockResolvedValue(undefined),
    saveRawText: vi.fn().mockResolvedValue(undefined),
    previewRawReplace: vi.fn().mockResolvedValue({
      previousText: "",
      nextText: "",
      addedCount: 0,
      removedCount: 0,
      changedCount: 0,
      lint: { items: [], valid_count: 0, invalid_count: 0 }
    }),
    confirmRawReplace: vi.fn().mockResolvedValue(undefined),
    cancelRawReplace: vi.fn(),
    undoRawReplace: vi.fn().mockResolvedValue(undefined),
    lintRaw: vi.fn().mockResolvedValue(undefined),
    loadManaged: vi.fn().mockResolvedValue(undefined),
    appendManaged: vi.fn().mockResolvedValue(undefined),
    deleteManaged: vi.fn().mockResolvedValue(undefined),
    lintManagedLine: vi.fn().mockResolvedValue(undefined)
  }
}

function makeOverrides() {
  return {
    draft: {},
    setDraft: vi.fn(),
    baseline: null,
    loaded: false,
    loading: false,
    userIdError: null,
    isDirty: false,
    rules: [],
    bannedRules: [],
    notifyRules: [],
    overridesQuery: { data: { overrides: [] }, isLoading: false, refetch: vi.fn() },
    updateDraft: vi.fn(),
    reset: vi.fn(),
    save: vi.fn().mockResolvedValue(undefined),
    remove: vi.fn().mockResolvedValue(undefined),
    bulkDelete: vi.fn().mockResolvedValue([]),
    addRule: vi.fn().mockReturnValue(true),
    removeRule: vi.fn(),
    applyPreset: vi.fn().mockResolvedValue(undefined)
  }
}

function makeMessageApi() {
  return { success: vi.fn(), error: vi.fn(), warning: vi.fn() }
}

// ---------------------------------------------------------------------------
// Import component under test (after mocks)
// ---------------------------------------------------------------------------

import AdvancedPanel from "../AdvancedPanel"

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("AdvancedPanel", () => {
  let messageApi: ReturnType<typeof makeMessageApi>

  beforeEach(() => {
    vi.clearAllMocks()
    messageApi = makeMessageApi()
  })

  it("renders performance tuning section with field labels", () => {
    const settings = makeSettings({ max_scan_chars: 150000, max_replacements_per_pattern: 500 })
    render(
      <AdvancedPanel
        settings={settings as any}
        blocklist={makeBlocklist() as any}
        overrides={makeOverrides() as any}
        messageApi={messageApi}
      />
    )
    expect(screen.getByText("Performance Tuning")).toBeInTheDocument()
    expect(screen.getByLabelText("max_scan_chars")).toHaveValue("150000")
    expect(screen.getByLabelText("max_replacements_per_pattern")).toHaveValue("500")
    expect(screen.getByLabelText("match_window_chars")).toBeInTheDocument()
    expect(screen.getByLabelText("blocklist_write_debounce_ms")).toBeInTheDocument()
  })

  it("renders export buttons for blocklist and overrides", () => {
    render(
      <AdvancedPanel
        settings={makeSettings() as any}
        blocklist={makeBlocklist() as any}
        overrides={makeOverrides() as any}
        messageApi={messageApi}
      />
    )
    expect(screen.getByRole("button", { name: /download blocklist/i })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /download overrides/i })).toBeInTheDocument()
  })

  it("renders reload button", () => {
    render(
      <AdvancedPanel
        settings={makeSettings() as any}
        blocklist={makeBlocklist() as any}
        overrides={makeOverrides() as any}
        messageApi={messageApi}
      />
    )
    expect(screen.getByRole("button", { name: /reload from disk/i })).toBeInTheDocument()
  })

  it("renders config viewer collapsible", () => {
    render(
      <AdvancedPanel
        settings={makeSettings({ enabled: true }) as any}
        blocklist={makeBlocklist() as any}
        overrides={makeOverrides() as any}
        messageApi={messageApi}
      />
    )
    expect(screen.getByText("View current configuration")).toBeInTheDocument()
  })

  it("shows default values when policy data is null", () => {
    render(
      <AdvancedPanel
        settings={makeSettings(null) as any}
        blocklist={makeBlocklist() as any}
        overrides={makeOverrides() as any}
        messageApi={messageApi}
      />
    )
    expect(screen.getByLabelText("max_scan_chars")).toHaveValue("200000")
    expect(screen.getByLabelText("max_replacements_per_pattern")).toHaveValue("1000")
    expect(screen.getByLabelText("match_window_chars")).toHaveValue("4096")
    expect(screen.getByLabelText("blocklist_write_debounce_ms")).toHaveValue("0")
  })

  it("previews blocklist uploads and only replaces after confirmation", async () => {
    const blocklist = makeBlocklist()
    blocklist.previewRawReplace.mockResolvedValue({
      previousText: "old",
      nextText: "new rule\n# note",
      addedCount: 2,
      removedCount: 1,
      changedCount: 3,
      lint: { items: [{ index: 0, line: "new rule", ok: true }], valid_count: 2, invalid_count: 0 }
    })

    render(
      <AdvancedPanel
        settings={makeSettings() as any}
        blocklist={blocklist as any}
        overrides={makeOverrides() as any}
        messageApi={messageApi}
      />
    )

    const file = new File(["new rule\n# note"], "blocklist.txt", { type: "text/plain" })
    await userEvent.upload(screen.getByTestId("blocklist-file-input"), file)

    await waitFor(() => {
      expect(blocklist.previewRawReplace).toHaveBeenCalledWith("new rule\n# note")
    })
    expect(blocklist.saveRawText).not.toHaveBeenCalled()
    expect(blocklist.confirmRawReplace).not.toHaveBeenCalled()

    const onOk = antdMocks.confirm.mock.calls.at(-1)?.[0]?.onOk
    await onOk()

    expect(blocklist.confirmRawReplace).toHaveBeenCalledTimes(1)
  })

  it("blocks blocklist upload confirmation when lint preview is invalid", async () => {
    const blocklist = makeBlocklist()
    blocklist.previewRawReplace.mockResolvedValue({
      previousText: "",
      nextText: "/[bad/",
      addedCount: 1,
      removedCount: 0,
      changedCount: 1,
      lint: {
        items: [{ index: 0, line: "/[bad/", ok: false, pattern_type: "regex", error: "invalid regex" }],
        valid_count: 0,
        invalid_count: 1
      }
    })

    render(
      <AdvancedPanel
        settings={makeSettings() as any}
        blocklist={blocklist as any}
        overrides={makeOverrides() as any}
        messageApi={messageApi}
      />
    )

    const file = new File(["/[bad/"], "blocklist.txt", { type: "text/plain" })
    await userEvent.upload(screen.getByTestId("blocklist-file-input"), file)

    await waitFor(() => {
      expect(antdMocks.confirm).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Confirm blocklist upload",
          okButtonProps: expect.objectContaining({ disabled: true })
        })
      )
    })
    expect(blocklist.confirmRawReplace).not.toHaveBeenCalled()
  })
})

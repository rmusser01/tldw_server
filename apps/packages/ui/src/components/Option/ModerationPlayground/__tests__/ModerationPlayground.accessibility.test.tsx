// @vitest-environment jsdom
import { fireEvent, render, screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"

const antdMocks = vi.hoisted(() => ({
  confirm: vi.fn()
}))

vi.mock("antd", () => ({
  Select: ({
    "aria-label": ariaLabel,
    placeholder,
    mode,
    value,
    onChange,
    options = [],
    ...props
  }: any) => (
    <select
      aria-label={ariaLabel || placeholder}
      data-testid={props["data-testid"]}
      multiple={mode === "tags"}
      value={mode === "tags" ? value ?? [] : value ?? ""}
      onChange={(event) => {
        if (mode === "tags") {
          onChange?.(
            Array.from(event.currentTarget.selectedOptions).map((option) => option.value)
          )
        } else {
          onChange?.(event.currentTarget.value)
        }
      }}
    >
      {options.map((option: { value: string; label: string }) => (
        <option key={option.value} value={option.value}>
          {option.label}
        </option>
      ))}
    </select>
  ),
  Modal: { confirm: antdMocks.confirm },
  Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>
}))

vi.mock("lucide-react", () => ({
  Download: () => <span aria-hidden="true" />,
  Play: () => <span aria-hidden="true" />,
  RefreshCw: () => <span aria-hidden="true" />,
  RotateCcw: () => <span aria-hidden="true" />,
  Search: () => <span aria-hidden="true" />,
  Trash2: () => <span aria-hidden="true" />,
  Upload: () => <span aria-hidden="true" />,
  X: () => <span aria-hidden="true" />,
  Zap: () => <span aria-hidden="true" />
}))

vi.mock("@/services/moderation", () => ({
  setUserOverride: vi.fn().mockResolvedValue({})
}))

vi.mock("../components/BlocklistSyntaxRef", () => ({
  BlocklistSyntaxRef: () => <div data-testid="syntax-ref">Blocklist Syntax Reference</div>
}))

import AdvancedPanel from "../AdvancedPanel"
import BlocklistStudioPanel from "../BlocklistStudioPanel"
import { ModerationContextBar } from "../ModerationContextBar"
import TestSandboxPanel from "../TestSandboxPanel"
import UserOverridesPanel from "../UserOverridesPanel"

function makeMessageApi() {
  return { success: vi.fn(), error: vi.fn(), warning: vi.fn() }
}

function makeBlocklist(overrides: Record<string, any> = {}) {
  return {
    rawText: "",
    setRawText: vi.fn(),
    rawLint: null,
    isDirtyRaw: false,
    pendingRawPreview: null,
    rawReplaceUndo: null,
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
    managedItems: [],
    managedVersion: "",
    managedLine: "",
    setManagedLine: vi.fn(),
    managedLint: null,
    loadManaged: vi.fn().mockResolvedValue(undefined),
    appendManaged: vi.fn().mockResolvedValue(undefined),
    appendLine: vi.fn().mockResolvedValue(undefined),
    deleteManaged: vi.fn().mockResolvedValue(undefined),
    lintManagedLine: vi.fn().mockResolvedValue(undefined),
    lintLine: vi.fn().mockResolvedValue({ items: [], valid_count: 0, invalid_count: 0 }),
    ...overrides
  }
}

function makeTester(overrides: Record<string, any> = {}) {
  return {
    phase: "input" as const,
    setPhase: vi.fn(),
    text: "",
    setText: vi.fn(),
    userId: "",
    setUserId: vi.fn(),
    result: null,
    history: [],
    running: false,
    runTest: vi.fn().mockResolvedValue(undefined),
    runTestWith: vi.fn().mockResolvedValue(undefined),
    clearHistory: vi.fn(),
    loadFromHistory: vi.fn(),
    ...overrides
  }
}

function makeCtx(overrides: Record<string, any> = {}) {
  return {
    scope: "user" as const,
    setScope: vi.fn(),
    userIdDraft: "",
    setUserIdDraft: vi.fn(),
    activeUserId: null,
    setActiveUserId: vi.fn(),
    loadUser: vi.fn(),
    clearUser: vi.fn(),
    ...overrides
  }
}

function makeOverrides(overrides: Record<string, any> = {}) {
  return {
    draft: {
      enabled: true,
      input_enabled: true,
      output_enabled: true,
      input_action: "block",
      output_action: "redact",
      redact_replacement: "[REMOVED]",
      categories_enabled: [],
      rules: []
    },
    updateDraft: vi.fn(),
    isDirty: false,
    loaded: true,
    loading: false,
    userIdError: null,
    rules: [],
    bannedRules: [],
    notifyRules: [],
    reset: vi.fn(),
    save: vi.fn().mockResolvedValue(undefined),
    remove: vi.fn().mockResolvedValue(undefined),
    bulkDelete: vi.fn().mockResolvedValue([]),
    addRule: vi.fn().mockReturnValue(true),
    removeRule: vi.fn(),
    applyPreset: vi.fn().mockResolvedValue(undefined),
    setDraft: vi.fn(),
    baseline: null,
    overridesQuery: {
      data: { overrides: { alice: { enabled: true } } },
      isLoading: false,
      refetch: vi.fn()
    },
    ...overrides
  }
}

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

describe("ModerationPlayground accessibility", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("gives the sticky context controls stable accessible names", async () => {
    const user = userEvent.setup()
    render(
      <ModerationContextBar
        scope="user"
        onScopeChange={vi.fn()}
        userIdDraft=""
        onUserIdDraftChange={vi.fn()}
        onLoadUser={vi.fn()}
        activeUserId={null}
        onClearUser={vi.fn()}
        userLoading={false}
        policy={{ enabled: true, input_action: "block", output_action: "redact", blocklist_count: 4 }}
        hasUnsavedChanges={false}
        onReload={vi.fn()}
        onRunQuickTest={vi.fn().mockResolvedValue({ flagged: false, action: "pass", effective: {} })}
        onOpenTestTab={vi.fn()}
      />
    )

    expect(screen.getByRole("combobox", { name: /moderation scope/i })).toBeInTheDocument()
    expect(screen.getByLabelText(/user id/i)).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: /quick test/i }))
    expect(screen.getByLabelText(/quick test text/i)).toBeInTheDocument()
    expect(screen.getByRole("combobox", { name: /quick test phase/i })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /close quick test/i })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /reload moderation config/i })).toBeInTheDocument()
  })

  it("labels the active-user clear action by its outcome", () => {
    render(
      <ModerationContextBar
        scope="user"
        onScopeChange={vi.fn()}
        userIdDraft=""
        onUserIdDraftChange={vi.fn()}
        onLoadUser={vi.fn()}
        activeUserId="alice"
        onClearUser={vi.fn()}
        userLoading={false}
        policy={{ enabled: true }}
        hasUnsavedChanges={false}
        onReload={vi.fn()}
        onRunQuickTest={vi.fn()}
        onOpenTestTab={vi.fn()}
      />
    )

    expect(screen.getByRole("button", { name: /clear active user alice/i })).toBeInTheDocument()
  })

  it("exposes sandbox phase selection as a radiogroup and labels editable fields", () => {
    render(<TestSandboxPanel tester={makeTester() as any} messageApi={makeMessageApi()} />)

    const phaseGroup = screen.getByRole("radiogroup", { name: /phase/i })
    expect(within(phaseGroup).getByRole("radio", { name: /user message/i })).toHaveAttribute("aria-checked", "true")
    expect(within(phaseGroup).getByRole("radio", { name: /ai response/i })).toHaveAttribute("aria-checked", "false")
    expect(screen.getByLabelText(/user id/i)).toBeInTheDocument()
    expect(screen.getByLabelText(/sample text/i)).toBeInTheDocument()
  })

  it("supports arrow-key navigation in blocklist tabs and labels rule editors", async () => {
    render(<BlocklistStudioPanel blocklist={makeBlocklist() as any} messageApi={makeMessageApi()} />)

    const managedTab = screen.getByRole("tab", { name: /managed rules/i })
    const rawTab = screen.getByRole("tab", { name: /raw editor/i })
    expect(screen.getByLabelText(/^pattern$/i)).toBeInTheDocument()
    expect(screen.getByLabelText(/^action$/i)).toBeInTheDocument()
    expect(screen.getByLabelText(/rule categories/i)).toBeInTheDocument()

    managedTab.focus()
    fireEvent.keyDown(managedTab, { key: "ArrowRight" })

    expect(rawTab).toHaveAttribute("aria-selected", "true")
    expect(rawTab).toHaveFocus()
    expect(screen.getByLabelText(/raw blocklist editor/i)).toBeInTheDocument()
  })

  it("labels user override picker, phrase rule fields, and table filter", () => {
    const { rerender } = render(
      <UserOverridesPanel
        ctx={makeCtx({ activeUserId: null }) as any}
        overrides={makeOverrides() as any}
        messageApi={makeMessageApi()}
      />
    )

    expect(screen.getByLabelText(/search or enter user id/i)).toBeInTheDocument()
    expect(screen.getByLabelText(/filter user overrides/i)).toBeInTheDocument()

    rerender(
      <UserOverridesPanel
        ctx={makeCtx({ activeUserId: "alice" }) as any}
        overrides={makeOverrides() as any}
        messageApi={makeMessageApi()}
      />
    )

    expect(screen.getByLabelText(/phrase pattern/i)).toBeInTheDocument()
    expect(screen.getByRole("radiogroup", { name: /phrase action/i })).toBeInTheDocument()
    expect(screen.getByRole("combobox", { name: /phrase phase/i })).toBeInTheDocument()
  })

  it("labels hidden import inputs through their visible upload actions", () => {
    render(
      <AdvancedPanel
        settings={makeSettings() as any}
        blocklist={makeBlocklist() as any}
        overrides={makeOverrides() as any}
        messageApi={makeMessageApi()}
      />
    )

    expect(screen.getByLabelText(/upload blocklist file/i)).toBeInTheDocument()
    expect(screen.getByLabelText(/upload user overrides file/i)).toBeInTheDocument()
  })
})

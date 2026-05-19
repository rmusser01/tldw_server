// @vitest-environment jsdom
import { render, screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

const antdMocks = vi.hoisted(() => ({
  confirm: vi.fn()
}))

vi.mock("antd", () => ({
  Select: ({ placeholder, ...props }: any) => (
    <div data-testid="categories-select">{placeholder}</div>
  ),
  Modal: { confirm: antdMocks.confirm },
  Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>
}))

vi.mock("lucide-react", () => ({
  Trash2: ({ size }: any) => <span data-testid="trash-icon">trash</span>
}))

vi.mock("../components/BlocklistSyntaxRef", () => ({
  BlocklistSyntaxRef: () => <div data-testid="syntax-ref">Blocklist Syntax Reference</div>
}))

// ---------------------------------------------------------------------------
// Helper — mock blocklist object
// ---------------------------------------------------------------------------

function makeBlocklist(overrides: Partial<ReturnType<typeof import("../hooks/useBlocklist").useBlocklist>> = {}) {
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

function makeMessageApi() {
  return {
    success: vi.fn(),
    error: vi.fn(),
    warning: vi.fn()
  }
}

// ---------------------------------------------------------------------------
// Import component under test (after mocks)
// ---------------------------------------------------------------------------

import BlocklistStudioPanel from "../BlocklistStudioPanel"

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("BlocklistStudioPanel", () => {
  let messageApi: ReturnType<typeof makeMessageApi>

  beforeEach(() => {
    vi.clearAllMocks()
    messageApi = makeMessageApi()
  })

  it("renders Managed Rules tab by default", () => {
    const blocklist = makeBlocklist()
    render(<BlocklistStudioPanel blocklist={blocklist as any} messageApi={messageApi} />)

    const managedTab = screen.getByRole("tab", { name: /managed rules/i })
    expect(managedTab).toHaveAttribute("aria-selected", "true")
  })

  it("renders Raw Editor tab and can switch to it", async () => {
    const blocklist = makeBlocklist()
    render(<BlocklistStudioPanel blocklist={blocklist as any} messageApi={messageApi} />)

    const rawTab = screen.getByRole("tab", { name: /raw editor/i })
    expect(rawTab).toHaveAttribute("aria-selected", "false")

    await userEvent.click(rawTab)
    expect(rawTab).toHaveAttribute("aria-selected", "true")
  })

  it("renders add rule form with pattern input and action selector", () => {
    const blocklist = makeBlocklist()
    render(<BlocklistStudioPanel blocklist={blocklist as any} messageApi={messageApi} />)

    expect(screen.getByTestId("pattern-input")).toBeInTheDocument()
    expect(screen.getByTestId("action-select")).toBeInTheDocument()
    expect(screen.getByText("Add Rule")).toBeInTheDocument()
  })

  it("renders syntax reference", () => {
    const blocklist = makeBlocklist()
    render(<BlocklistStudioPanel blocklist={blocklist as any} messageApi={messageApi} />)

    expect(screen.getByTestId("syntax-ref")).toBeInTheDocument()
  })

  it("shows empty state message when no rules loaded", () => {
    const blocklist = makeBlocklist({ managedItems: [] })
    render(<BlocklistStudioPanel blocklist={blocklist as any} messageApi={messageApi} />)

    expect(screen.getByTestId("empty-rules")).toBeInTheDocument()
    expect(screen.getByText(/no rules loaded/i)).toBeInTheDocument()
  })

  it("auto-loads managed rules on mount", () => {
    const blocklist = makeBlocklist()
    render(<BlocklistStudioPanel blocklist={blocklist as any} messageApi={messageApi} />)

    expect(blocklist.loadManaged).toHaveBeenCalledTimes(1)
  })

  it("renders rules table when managedItems are present", () => {
    const blocklist = makeBlocklist({
      managedItems: [
        { id: 1, line: "badword -> block #violence" },
        { id: 2, line: "/nsfw/i -> redact" }
      ]
    })
    render(<BlocklistStudioPanel blocklist={blocklist as any} messageApi={messageApi} />)

    expect(screen.getByTestId("rules-table")).toBeInTheDocument()
    expect(screen.getByText("badword")).toBeInTheDocument()
  })

  it("defaults to active valid rules and excludes comments, blanks, and invalid rows from counts", () => {
    const blocklist = makeBlocklist({
      managedItems: [
        { id: 1, line: "badword -> block #violence", pattern_type: "literal", action: "block", categories: ["violence"], ok: true },
        { id: 2, line: "# internal note", pattern_type: "comment", ok: true, warning: "comment (ignored)" },
        { id: 3, line: "", pattern_type: "empty", ok: true, warning: "blank line (ignored)" },
        { id: 4, line: "/[broken/ -> block", pattern_type: "regex", action: "block", ok: false, error: "invalid regex" }
      ]
    })
    render(<BlocklistStudioPanel blocklist={blocklist as any} messageApi={messageApi} />)

    expect(screen.getByText(/1 active rule/i)).toBeInTheDocument()
    expect(screen.getByText(/3 non-active rows hidden/i)).toBeInTheDocument()
    expect(screen.getByText("badword")).toBeInTheDocument()
    expect(screen.queryByText("# internal note")).not.toBeInTheDocument()
    expect(screen.queryByText("/[broken/")).not.toBeInTheDocument()
  })

  it("can reveal comments and blanks, filter by category, and sort rules", async () => {
    const blocklist = makeBlocklist({
      managedItems: [
        { id: 1, line: "zeta -> warn #pii", pattern_type: "literal", action: "warn", categories: ["pii"], sample: "zeta", ok: true },
        { id: 2, line: "alpha -> block #violence", pattern_type: "literal", action: "block", categories: ["violence"], sample: "alpha", ok: true },
        { id: 3, line: "# note", pattern_type: "comment", ok: true }
      ]
    })
    render(<BlocklistStudioPanel blocklist={blocklist as any} messageApi={messageApi} />)

    await userEvent.click(screen.getByRole("checkbox", { name: /show comments and blanks/i }))
    expect(screen.getByText("# note")).toBeInTheDocument()

    await userEvent.type(screen.getByPlaceholderText(/search rules/i), "alpha")
    expect(screen.getByText("alpha")).toBeInTheDocument()
    expect(screen.queryByText("zeta")).not.toBeInTheDocument()
  })

  it("previews raw replace and only confirms through the preview modal", async () => {
    const blocklist = makeBlocklist({
      rawText: "draft rule",
      previewRawReplace: vi.fn().mockResolvedValue({
        previousText: "old rule",
        nextText: "draft rule",
        addedCount: 1,
        removedCount: 1,
        changedCount: 1,
        lint: { items: [{ index: 0, line: "draft rule", ok: true }], valid_count: 1, invalid_count: 0 }
      })
    })
    render(<BlocklistStudioPanel blocklist={blocklist as any} messageApi={messageApi} />)

    await userEvent.click(screen.getByRole("tab", { name: /raw editor/i }))
    await userEvent.click(screen.getByRole("button", { name: /save \/ replace/i }))

    expect(blocklist.previewRawReplace).toHaveBeenCalledWith("draft rule")
    expect(blocklist.saveRaw).not.toHaveBeenCalled()
    expect(blocklist.confirmRawReplace).not.toHaveBeenCalled()
    expect(antdMocks.confirm).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Confirm blocklist replacement",
        okButtonProps: expect.objectContaining({ danger: true, disabled: false })
      })
    )

    const onOk = antdMocks.confirm.mock.calls.at(-1)?.[0]?.onOk
    await onOk()
    expect(blocklist.confirmRawReplace).toHaveBeenCalledTimes(1)
  })

  it("disables raw replace confirmation when preview lint has invalid rows", async () => {
    const blocklist = makeBlocklist({
      rawText: "/[bad/",
      previewRawReplace: vi.fn().mockResolvedValue({
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
    })
    render(<BlocklistStudioPanel blocklist={blocklist as any} messageApi={messageApi} />)

    await userEvent.click(screen.getByRole("tab", { name: /raw editor/i }))
    await userEvent.click(screen.getByRole("button", { name: /save \/ replace/i }))

    expect(antdMocks.confirm).toHaveBeenCalledWith(
      expect.objectContaining({
        okButtonProps: expect.objectContaining({ disabled: true })
      })
    )
    expect(blocklist.confirmRawReplace).not.toHaveBeenCalled()
  })

  it("offers a session undo after deleting a managed rule", async () => {
    const blocklist = makeBlocklist({
      managedItems: [
        { id: 10, line: "badword -> block", pattern_type: "literal", action: "block", ok: true }
      ]
    })
    render(<BlocklistStudioPanel blocklist={blocklist as any} messageApi={messageApi} />)

    await userEvent.click(screen.getByRole("button", { name: /delete rule 10/i }))
    const onOk = antdMocks.confirm.mock.calls.at(-1)?.[0]?.onOk
    await onOk()

    expect(blocklist.deleteManaged).toHaveBeenCalledWith(10)
    expect(screen.getByRole("button", { name: /undo deleted rule/i })).toBeInTheDocument()

    await userEvent.click(screen.getByRole("button", { name: /undo deleted rule/i }))
    expect(blocklist.appendLine).toHaveBeenCalledWith("badword -> block")
  })

  it("renders raw editor warning banner when in raw tab", async () => {
    const blocklist = makeBlocklist()
    render(<BlocklistStudioPanel blocklist={blocklist as any} messageApi={messageApi} />)

    await userEvent.click(screen.getByRole("tab", { name: /raw editor/i }))

    expect(screen.getByText(/raw file editing replaces all existing rules/i)).toBeInTheDocument()
    expect(screen.getByTestId("raw-editor")).toBeInTheDocument()
  })

  it("renders validate and add rule buttons", () => {
    const blocklist = makeBlocklist()
    render(<BlocklistStudioPanel blocklist={blocklist as any} messageApi={messageApi} />)

    expect(screen.getByRole("button", { name: /validate/i })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /add rule/i })).toBeInTheDocument()
  })

  it("does not render unsupported phase controls in the managed rule form", () => {
    const blocklist = makeBlocklist()
    render(<BlocklistStudioPanel blocklist={blocklist as any} messageApi={messageApi} />)

    expect(screen.queryByText(/^Phase$/)).not.toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Input" })).not.toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Output" })).not.toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Both" })).not.toBeInTheDocument()
  })

  it("renders syntax reference in raw editor tab too", async () => {
    const blocklist = makeBlocklist()
    render(<BlocklistStudioPanel blocklist={blocklist as any} messageApi={messageApi} />)

    await userEvent.click(screen.getByRole("tab", { name: /raw editor/i }))

    expect(screen.getByTestId("syntax-ref")).toBeInTheDocument()
  })
})

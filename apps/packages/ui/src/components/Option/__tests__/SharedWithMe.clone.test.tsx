import "./shared-with-me-i18n"
import { fireEvent, render, screen, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import axe from "axe-core"
import { SharedWithMe } from "../SharedWithMe"
import type { CloneRowState } from "@/services/shared-clone-manager"
import { TldwApiError } from "@/services/tldw/api-error"
import { sharedCloneOperationSchema } from "@/types/shared-workspace-clone"
import {
  clonePayload,
  workspaceId
} from "@/services/tldw/domains/__tests__/shared-workspace-clone.fixture"

const mocks = vi.hoisted(() => ({
  shares: vi.fn(),
  clones: vi.fn(),
  navigate: vi.fn(),
  begin: vi.fn(),
  refresh: vi.fn()
}))
vi.mock("@/hooks/useSharing", () => ({
  useSharedWithMe: mocks.shares,
  useCloneWorkspace: () => ({ mutate: vi.fn() })
}))
vi.mock("@/hooks/useSharedWorkspaceClones", () => ({
  useSharedWorkspaceClones: mocks.clones
}))
vi.mock("react-router-dom", () => ({ useNavigate: () => mocks.navigate }))

const share = {
  share_id: 42,
  workspace_id: "original",
  workspace_name: "Research",
  owner_user_id: 10,
  owner_username: "Ada",
  access_level: "view_chat",
  allow_clone: true
}
const row = (status = "queued"): CloneRowState => ({
  entry: { share_id: 42, expires_at: Date.now() + 10000 },
  operation: sharedCloneOperationSchema.parse(clonePayload(status)),
  pending: false,
  recoveryAvailable: true
})
const setRows = (rows: CloneRowState[]) =>
  mocks.clones.mockReturnValue({
    rows,
    scope: "scope-42",
    status: "ready",
    begin: mocks.begin,
    refresh: mocks.refresh
  })

describe("Shared With Me durable copy UI", () => {
  beforeEach(() => {
    mocks.navigate.mockReset()
    mocks.begin.mockReset()
    mocks.refresh.mockReset()
    mocks.shares.mockReturnValue({
      data: { items: [share], total: 1 },
      isLoading: false,
      error: null
    })
    setRows([])
  })
  it("starts a command without announcing success", () => {
    render(<SharedWithMe />)
    fireEvent.click(screen.getByRole("button", { name: "Clone Research" }))
    expect(mocks.begin).toHaveBeenCalledWith(42, "Research")
    expect(screen.queryByText("Copy ready.")).not.toBeInTheDocument()
    expect(screen.getByText("Shared by Ada")).toBeInTheDocument()
    expect(screen.queryByText(/account 10/)).not.toBeInTheDocument()
  })
  it.each(["queued", "running"])(
    "renders accessible %s progress independently from other rows",
    (status) => {
      mocks.shares.mockReturnValue({
        data: {
          items: [share, { ...share, share_id: 43, workspace_name: "Other" }]
        }
      })
      setRows([row(status)])
      render(<SharedWithMe />)
      const progress = screen.getByRole("progressbar", {
        name: "Copy progress for Research"
      })
      expect(progress).toHaveAttribute("aria-valuenow", "40")
      expect(screen.getByRole("status")).toHaveAttribute("aria-live", "polite")
      expect(
        screen.getByRole("button", { name: "Clone Research" })
      ).toBeDisabled()
      expect(screen.getByRole("button", { name: "Clone Other" })).toBeEnabled()
      expect(
        screen.queryByRole("button", { name: /Open copy/ })
      ).not.toBeInTheDocument()
    }
  )
  it("opens only the confirmed recipient-owned target and exposes readiness", () => {
    setRows([row("succeeded")])
    render(<SharedWithMe />)
    expect(screen.getByText("Copy ready.")).toBeInTheDocument()
    expect(
      screen.getByText("Vector search needs indexing.")
    ).toBeInTheDocument()
    fireEvent.click(
      screen.getByRole("button", { name: "Open copy of Research" })
    )
    expect(mocks.navigate).toHaveBeenCalledWith(
      `/research-workspace?workspace=${workspaceId}`
    )
  })
  it("shows partial results and bounded warnings without raw server copy", () => {
    const state = row("succeeded")
    if (state.operation?.status === "succeeded") {
      state.operation.result.outcome = "partial"
      state.operation.result.counts.sources_failed = 1
      state.operation.result.counts.sources_copied = 1
      state.operation.result.warnings = [
        { code: "unknown_future_warning", count: 1 }
      ]
    }
    setRows([state])
    render(<SharedWithMe />)
    expect(screen.getByText("Copy ready with omissions.")).toBeInTheDocument()
    expect(screen.getByText(/1 of 2 sources copied/)).toBeInTheDocument()
    expect(
      screen.getByText("Some items need attention (1).")
    ).toBeInTheDocument()
  })
  it.each([true, false])(
    "offers a new attempt only when retryable=%s",
    (retryable) => {
      const state = row("failed")
      state.operation!.retryable = retryable
      setRows([state])
      render(<SharedWithMe />)
      expect(
        screen.getByText("The copy could not be completed.")
      ).toBeInTheDocument()
      expect(
        Boolean(
          screen.queryByRole("button", { name: "Retry copy of Research" })
        )
      ).toBe(retryable)
    }
  )
  it("keeps terminal feedback when the share disappears without granting source access", () => {
    mocks.shares.mockReturnValue({
      data: { items: [] },
      isLoading: false,
      error: null
    })
    setRows([row("succeeded")])
    render(<SharedWithMe />)
    expect(
      screen.getByText("Shared workspace no longer available")
    ).toBeInTheDocument()
    expect(screen.getByText("Copy ready.")).toBeInTheDocument()
    const item = screen.getByRole("listitem")
    expect(
      within(item).queryByRole("button", { name: /^Open Shared/ })
    ).not.toBeInTheDocument()
    expect(
      within(item).getByRole("button", { name: /^Open copy/ })
    ).toBeEnabled()
  })
  it("preserves access to a completed owned copy after sharing permission is lost", () => {
    mocks.shares.mockReturnValue({
      data: { items: [share] },
      error: new TldwApiError("Permission required", 403, {
        code: "sharing_permission_required"
      })
    })
    setRows([row("succeeded")])
    render(<SharedWithMe />)
    expect(
      screen.queryByRole("button", { name: "Open Research" })
    ).not.toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Clone Research" })
    ).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: /^Open copy/ }))
    expect(mocks.navigate).toHaveBeenCalledWith(
      `/research-workspace?workspace=${workspaceId}`
    )
  })
  it("shows safe recovery feedback for transport and local storage failures", () => {
    setRows([{ ...row(), issue: "uncertain", recoveryAvailable: false }])
    render(<SharedWithMe />)
    expect(
      screen.getByText(/Copy status could not be confirmed/)
    ).toBeInTheDocument()
    expect(
      screen.getByText(/Reload recovery is unavailable/)
    ).toBeInTheDocument()
    fireEvent.click(
      screen.getByRole("button", { name: "Check copy status for Research" })
    )
    expect(mocks.refresh).toHaveBeenCalledWith(42)
  })
  it("does not render cached shares while authentication scope is unresolved", () => {
    mocks.clones.mockReturnValue({
      rows: [],
      scope: null,
      status: "loading",
      begin: mocks.begin
    })
    render(<SharedWithMe />)
    expect(screen.queryByText("Research")).not.toBeInTheDocument()
  })

  it.each([false, true])(
    "shows recovery conflict instead of clone controls while shares loading=%s",
    (isLoading) => {
      mocks.shares.mockReturnValue({ data: { items: [share] }, isLoading })
      mocks.clones.mockReturnValue({
        rows: [row("failed")],
        scope: null,
        status: "recovery_conflict",
        begin: mocks.begin,
        refresh: mocks.refresh
      })
      render(<SharedWithMe />)
      expect(screen.getByRole("alert")).toHaveTextContent(
        "Copy recovery belongs to another account or server. Switch back to resume. Copies have not been canceled."
      )
      expect(
        screen.queryByRole("button", { name: /Clone|Retry copy|Check copy/ })
      ).not.toBeInTheDocument()
      expect(screen.queryByText("Research")).not.toBeInTheDocument()
      fireEvent.click(screen.getByRole("button", { name: /Try again/ }))
      expect(mocks.refresh).toHaveBeenCalledWith()
      expect(mocks.begin).not.toHaveBeenCalled()
    }
  )

  it("has no detectable structural accessibility violations in the operation surface", async () => {
    setRows([row("succeeded")])
    const { container } = render(
      <main>
        <SharedWithMe />
      </main>
    )
    const result = await axe.run(container, {
      rules: { "color-contrast": { enabled: false } }
    })
    expect(result.violations).toEqual([])
  })
})

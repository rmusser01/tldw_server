import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { ACPPermissionModal } from "../ACPPermissionModal"
import type { ACPPendingPermission } from "@/services/acp/types"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?: string | { defaultValue?: string },
      maybeOptions?: Record<string, unknown>
    ) => {
      if (typeof fallbackOrOptions === "string") {
        return fallbackOrOptions
      }
      if (
        fallbackOrOptions &&
        typeof fallbackOrOptions === "object" &&
        typeof fallbackOrOptions.defaultValue === "string"
      ) {
        return fallbackOrOptions.defaultValue
      }
      return maybeOptions?.defaultValue || key
    },
  }),
}))

const pendingPermissions: ACPPendingPermission[] = [
  {
    request_id: "req-1",
    tool_name: "fs.writeTextFile",
    tool_arguments: { path: "/tmp/demo.txt" },
    tier: "batch",
    approval_requirement: "approval_required",
    governance_reason: "policy_approval_required",
    provenance_summary: { source_kinds: ["capability_mapping"] },
    runtime_narrowing_reason: "workspace_trust_required",
    policy_snapshot_fingerprint: "snapshot-1234567890",
    timeout_seconds: 30,
    requestedAt: new Date(),
  },
  {
    request_id: "req-2",
    tool_name: "exec.bash",
    tool_arguments: { command: "echo ok" },
    tier: "individual",
    timeout_seconds: 30,
    requestedAt: new Date(),
  },
]

describe("ACPPermissionModal", () => {
  it("renders current permission and queue indicator", () => {
    render(
      <ACPPermissionModal
        pendingPermissions={pendingPermissions}
        approvePermission={vi.fn()}
        denyPermission={vi.fn()}
      />
    )

    expect(screen.getByText("Permission Required")).toBeInTheDocument()
    expect(screen.getByText("fs.writeTextFile")).toBeInTheDocument()
    expect(screen.getByText(/permission requests queued/i)).toBeInTheDocument()
  })

  it("sends approve and deny actions for the current permission", () => {
    const approvePermission = vi.fn()
    const denyPermission = vi.fn()

    render(
      <ACPPermissionModal
        pendingPermissions={pendingPermissions}
        approvePermission={approvePermission}
        denyPermission={denyPermission}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Approve" }))
    expect(approvePermission).toHaveBeenCalledWith("req-1", undefined)

    fireEvent.click(screen.getByRole("checkbox"))
    fireEvent.click(screen.getByRole("button", { name: "Approve" }))
    expect(approvePermission).toHaveBeenLastCalledWith("req-1", "batch")

    fireEvent.click(screen.getByRole("button", { name: "Deny" }))
    expect(denyPermission).toHaveBeenCalledWith("req-1")
  })

  it("shows compact policy metadata and expandable provenance details", () => {
    render(
      <ACPPermissionModal
        pendingPermissions={pendingPermissions}
        approvePermission={vi.fn()}
        denyPermission={vi.fn()}
      />
    )

    expect(screen.getByText("Policy Snapshot")).toBeInTheDocument()
    expect(screen.getByText("approval_required")).toBeInTheDocument()
    expect(screen.getByText("policy_approval_required")).toBeInTheDocument()
    expect(screen.getByText("workspace_trust_required")).toBeInTheDocument()
    expect(screen.getByText("snapshot-123")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Show details" }))

    expect(screen.getByText(/capability_mapping/)).toBeInTheDocument()
  })

  it("does not violate hook ordering when the queue becomes empty", () => {
    const { rerender } = render(
      <ACPPermissionModal
        pendingPermissions={pendingPermissions}
        approvePermission={vi.fn()}
        denyPermission={vi.fn()}
      />
    )

    expect(() =>
      rerender(
        <ACPPermissionModal
          pendingPermissions={[]}
          approvePermission={vi.fn()}
          denyPermission={vi.fn()}
        />
      )
    ).not.toThrow()
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
  })

  it("resets batch approval and policy details when the current request changes", () => {
    const nextPermission: ACPPendingPermission = {
      ...pendingPermissions[0],
      request_id: "req-3",
      tool_name: "fs.appendFile",
      provenance_summary: { source_kinds: ["policy_cache"] },
    }
    const { rerender } = render(
      <ACPPermissionModal
        pendingPermissions={[pendingPermissions[0]]}
        approvePermission={vi.fn()}
        denyPermission={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("checkbox"))
    fireEvent.click(screen.getByRole("button", { name: "Show details" }))

    expect(screen.getByRole("checkbox")).toBeChecked()
    expect(screen.getByText(/capability_mapping/)).toBeInTheDocument()

    rerender(
      <ACPPermissionModal
        pendingPermissions={[nextPermission]}
        approvePermission={vi.fn()}
        denyPermission={vi.fn()}
      />
    )

    expect(screen.getByRole("checkbox")).not.toBeChecked()
    expect(screen.getByRole("button", { name: "Show details" })).toBeInTheDocument()
    expect(screen.queryByText(/policy_cache/)).not.toBeInTheDocument()
  })
})

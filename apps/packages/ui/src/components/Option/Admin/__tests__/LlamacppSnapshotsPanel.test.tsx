import React from "react"
import { describe, expect, it, vi } from "vitest"
import { render, screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import i18next from "i18next"
import { initReactI18next } from "react-i18next"
import ICU from "@/i18n/icu-format"
import { LlamacppSnapshotsPanel } from "../LlamacppSnapshotsPanel"

void i18next
  .use(ICU)
  .use(initReactI18next)
  .init({ lng: "en", resources: {}, interpolation: { escapeValue: false } })

const snapshot = {
  snapshot_id: "snapshot-one",
  source_slot: 0,
  created_at: "2026-09-04T14:32:00Z",
  commit_sequence: 1,
  byte_count: 600000000,
  token_count: 8192,
  compatibility: "compatible" as const,
  reasons: []
}
const slots = {
  capability: "ready" as const,
  reason: null,
  launch_generation: "launch-one",
  request_id: "signed-token",
  latest_operation_id: null,
  slots: [{ slot_id: 0, busy: false, token_count: 8192 }]
}
const fixture = () => ({
  enabled: true,
  retention: 10,
  slots,
  catalog: {
    snapshots: [snapshot],
    total: 1,
    total_bytes: 600000000,
    offset: 0,
    limit: 50,
    retention: 10
  },
  onEnable: vi.fn(),
  onRetention: vi.fn(),
  onRefresh: vi.fn(),
  onPage: vi.fn(),
  onSave: vi.fn(),
  onRestore: vi.fn(),
  onDelete: vi.fn(),
  onStop: vi.fn()
})

describe("LlamacppSnapshotsPanel", () => {
  it.each(["Delete", "Stop recovery"])(
    "focuses the %s confirmation so immediate Escape closes it",
    async (name) => {
      const user = userEvent.setup()
      const props = fixture()
      render(
        <LlamacppSnapshotsPanel
          {...props}
          operation={
            name === "Stop recovery"
              ? {
                  profile_id: "test",
                  operation_id: "op",
                  launch_generation: "launch-one",
                  kind: "restore",
                  state: "outcome_unknown",
                  recovery_action: "stop_runtime"
                }
              : null
          }
        />
      )
      const trigger = screen.getByRole("button", { name })
      await user.click(trigger)
      expect(screen.getByRole("button", { name: "Cancel" })).toHaveFocus()
      await user.keyboard("{Escape}")
      expect(screen.queryByRole("button", { name: "Cancel" })).toBeNull()
      expect(trigger).toHaveFocus()
      expect(props.onDelete).not.toHaveBeenCalled()
      expect(props.onStop).not.toHaveBeenCalled()
    }
  )
  it("allows deleting saved copies after an unknown launch is confirmed stopped", () => {
    render(
      <LlamacppSnapshotsPanel
        {...fixture()}
        slots={{ ...slots, capability: "stopped" }}
        operation={{
          profile_id: "test",
          operation_id: "old-op",
          launch_generation: "launch-one",
          kind: "restore",
          state: "outcome_unknown",
          recovery_action: "stop_runtime"
        }}
      />
    )
    expect(screen.getByRole("button", { name: "Delete" })).toBeEnabled()
    expect(screen.queryByRole("button", { name: "Stop recovery" })).toBeNull()
  })
  it("requires explicit destination confirmation before restoring", async () => {
    const props = fixture()
    const user = userEvent.setup()
    render(<LlamacppSnapshotsPanel {...props} />)
    await user.click(screen.getByRole("button", { name: "Restore" }))
    expect(props.onRestore).not.toHaveBeenCalled()
    expect(screen.getByText(/Failure may also clear it/)).toBeVisible()
    await user.click(
      screen.getByRole("button", { name: "Restore into slot 0" })
    )
    expect(props.onRestore).toHaveBeenCalledExactlyOnceWith("snapshot-one", 0)
  })

  it("returns keyboard focus to Restore after cancelling or Escape", async () => {
    const user = userEvent.setup()
    render(<LlamacppSnapshotsPanel {...fixture()} />)
    const trigger = screen.getByRole("button", { name: "Restore" })
    await user.click(trigger)
    expect(
      screen.getByRole("combobox", { name: "Destination slot" })
    ).toHaveFocus()
    await user.keyboard("{Escape}")
    expect(trigger).toHaveFocus()
    await user.click(trigger)
    await user.click(screen.getByRole("button", { name: "Cancel" }))
    expect(trigger).toHaveFocus()
  })

  it("names permanent deletion and never calls a slot mutation", async () => {
    const props = fixture()
    const user = userEvent.setup()
    render(<LlamacppSnapshotsPanel {...props} />)
    await user.click(screen.getByRole("button", { name: "Delete" }))
    expect(props.onDelete).not.toHaveBeenCalled()
    expect(screen.getByText(/does not erase an active slot/)).toBeVisible()
    await user.click(
      screen.getByRole("button", { name: "Permanently delete snapshot-one" })
    )
    expect(props.onDelete).toHaveBeenCalledExactlyOnceWith("snapshot-one")
    expect(props.onRestore).not.toHaveBeenCalled()
    expect(props.onSave).not.toHaveBeenCalled()
  })

  it.each([
    "stopped",
    "unsupported",
    "busy",
    "restart_required",
    "disabled",
    "unavailable"
  ] as const)("explains %s without enabling save or restore", (capability) => {
    render(
      <LlamacppSnapshotsPanel
        {...fixture()}
        slots={{ ...slots, capability, reason: "unsupported_build" }}
      />
    )
    expect(screen.getByRole("button", { name: "Save snapshot" })).toBeDisabled()
    expect(screen.getByRole("button", { name: "Restore" })).toBeDisabled()
    expect(screen.getByText(/unsupported_build/)).toBeVisible()
    expect(screen.getByRole("button", { name: "Delete" })).toBeEnabled()
  })

  it("keeps busy and incompatible reasons visible", () => {
    const props = fixture()
    render(
      <LlamacppSnapshotsPanel
        {...props}
        slots={{
          ...slots,
          slots: [{ slot_id: 0, busy: true, token_count: 2 }]
        }}
        catalog={{
          ...props.catalog,
          snapshots: [
            {
              ...snapshot,
              compatibility: "incompatible",
              reasons: ["model_changed"]
            }
          ]
        }}
      />
    )
    expect(screen.getByText(/Save unavailable: busy/)).toBeVisible()
    expect(screen.getByText(/model_changed/)).toBeVisible()
    expect(screen.getByRole("button", { name: "Restore" })).toBeDisabled()
  })

  it("announces unknown outcome and requires explicit Stop recovery", async () => {
    const props = fixture()
    const user = userEvent.setup()
    render(
      <LlamacppSnapshotsPanel
        {...props}
        operation={{
          profile_id: "test",
          operation_id: "op-one",
          launch_generation: "launch-one",
          kind: "restore",
          state: "outcome_unknown",
          recovery_action: "stop_runtime"
        }}
      />
    )
    expect(
      within(screen.getByRole("status")).getByText(/Outcome unknown/)
    ).toBeVisible()
    expect(screen.queryByRole("button", { name: /Retry Restore/i })).toBeNull()
    expect(screen.getByRole("button", { name: "Restore" })).toBeDisabled()
    await user.click(screen.getByRole("button", { name: "Stop recovery" }))
    expect(props.onStop).not.toHaveBeenCalled()
    await user.click(
      screen.getByRole("button", { name: "Stop runtime and inference" })
    )
    expect(props.onStop).toHaveBeenCalledOnce()
  })

  it("distinguishes a failed catalog read from an empty catalog", () => {
    render(
      <LlamacppSnapshotsPanel
        {...fixture()}
        catalog={null}
        error="Snapshot catalog could not be read."
      />
    )
    expect(
      screen.getByText("Snapshot catalog could not be read.")
    ).toBeVisible()
    expect(screen.queryByText(/No saved snapshots/)).toBeNull()
    expect(screen.getByRole("button", { name: "Save snapshot" })).toBeDisabled()
  })

  it("teaches first use and keeps enablement separate from restart", async () => {
    const props = fixture()
    const user = userEvent.setup()
    render(
      <LlamacppSnapshotsPanel
        {...props}
        enabled={false}
        slots={{ ...slots, capability: "disabled", slots: [] }}
        catalog={{ ...props.catalog, snapshots: [], total: 0, total_bytes: 0 }}
      />
    )
    expect(screen.getByText(/No saved snapshots/)).toBeVisible()
    expect(screen.getByText(/does not restart/)).toBeVisible()
    await user.click(screen.getByRole("button", { name: "Enable snapshots" }))
    expect(props.onEnable).toHaveBeenCalledExactlyOnceWith(true)
    expect(props.onStop).not.toHaveBeenCalled()
  })
})

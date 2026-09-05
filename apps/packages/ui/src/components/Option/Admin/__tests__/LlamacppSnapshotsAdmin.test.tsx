import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import i18next from "i18next"
import { initReactI18next } from "react-i18next"
import ICU from "@/i18n/icu-format"
import { LlamacppSnapshotsAdmin } from "../LlamacppAdminPage"

void i18next.use(ICU).use(initReactI18next).init({ lng: "en", resources: {} })
const api = vi.hoisted(() => ({
  getLlamacppSnapshotSlots: vi.fn(),
  listLlamacppSnapshots: vi.fn(),
  getLlamacppSnapshotOperation: vi.fn(),
  saveLlamacppSnapshot: vi.fn(),
  restoreLlamacppSnapshot: vi.fn(),
  deleteLlamacppSnapshot: vi.fn(),
  updateLlamacppProfile: vi.fn(),
  stopLlamacppProfile: vi.fn()
}))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: api }))
const profile = {
  profile_id: "test-profile",
  name: "Disposable",
  enabled: true,
  mode: "chat" as const,
  host: "127.0.0.1",
  port: 8181,
  port_policy: "explicit" as const,
  server_args: {},
  autostart: false,
  restart_policy: {},
  tags: [],
  snapshots_enabled: true,
  snapshot_retention: 10
}
const slots = {
  capability: "ready",
  launch_generation: "generation-one",
  request_id: "fresh-token",
  latest_operation_id: null,
  slots: [{ slot_id: 0, busy: false, token_count: 8192 }]
}
const receipt = {
  profile_id: "test-profile",
  operation_id: "operation-one",
  launch_generation: "generation-one",
  kind: "save",
  state: "complete",
  token_count: 8192,
  recovery_action: "none"
}
const catalog = {
  snapshots: [],
  total: 0,
  total_bytes: 0,
  retention: 10,
  offset: 0,
  limit: 50
}
beforeEach(() => {
  vi.clearAllMocks()
  api.getLlamacppSnapshotSlots.mockResolvedValue(slots)
  api.listLlamacppSnapshots.mockResolvedValue(catalog)
  api.getLlamacppSnapshotOperation.mockResolvedValue(receipt)
  api.saveLlamacppSnapshot.mockResolvedValue(receipt)
})

describe("snapshot Admin coordination", () => {
  it.each([
    [422, "snapshot_incompatible"],
    [409, "stale_launch_generation"],
    [503, "runtime_owner_unavailable"]
  ] as const)(
    "allows refresh and a new manual action after definitive %s admission rejection with no receipt",
    async (status, detail) => {
      api.saveLlamacppSnapshot.mockRejectedValueOnce(
        Object.assign(new Error(detail), {
          status,
          details: { detail }
        })
      )
      render(
        <LlamacppSnapshotsAdmin profile={profile} onProfileChanged={vi.fn()} />
      )
      const save = await screen.findByRole("button", { name: "Save snapshot" })
      await waitFor(() => expect(save).toBeEnabled())
      fireEvent.click(save)
      await screen.findByText(detail)
      await waitFor(() =>
        expect(screen.getByRole("button", { name: "Refresh" })).toBeEnabled()
      )
      expect(screen.queryByRole("button", { name: "Stop recovery" })).toBeNull()
      expect(api.saveLlamacppSnapshot).toHaveBeenCalledOnce()
      fireEvent.click(screen.getByRole("button", { name: "Refresh" }))
      await waitFor(() => expect(save).toBeEnabled())
      fireEvent.click(save)
      await waitFor(() =>
        expect(api.saveLlamacppSnapshot).toHaveBeenCalledTimes(2)
      )
      expect(api.stopLlamacppProfile).not.toHaveBeenCalled()
    }
  )

  it.each(["complete", "failed", "saving", "outcome_unknown"])(
    "preserves historical %s receipts without blocking the new launch",
    async (state) => {
      api.getLlamacppSnapshotSlots.mockResolvedValue({
        ...slots,
        launch_generation: "new-launch",
        latest_operation_id: "operation-one"
      })
      api.getLlamacppSnapshotOperation.mockResolvedValue({ ...receipt, state })
      render(
        <LlamacppSnapshotsAdmin profile={profile} onProfileChanged={vi.fn()} />
      )
      expect(await screen.findByText(/Latest operation:/)).toBeVisible()
      expect(screen.getByText("operation-one")).toBeInTheDocument()
      await waitFor(() =>
        expect(
          screen.getByRole("button", { name: "Save snapshot" })
        ).toBeEnabled()
      )
      expect(screen.queryByRole("button", { name: "Stop recovery" })).toBeNull()
      expect(api.saveLlamacppSnapshot).not.toHaveBeenCalled()
    }
  )

  it("recovers a historical receipt when there is no runner generation", async () => {
    api.getLlamacppSnapshotSlots.mockResolvedValue({
      ...slots,
      capability: "stopped",
      launch_generation: null,
      latest_operation_id: "operation-one",
      slots: []
    })
    render(
      <LlamacppSnapshotsAdmin profile={profile} onProfileChanged={vi.fn()} />
    )
    expect(await screen.findByText(/Latest operation: Complete/)).toBeVisible()
  })
  it("recovers the latest receipt on reload without resubmitting", async () => {
    api.getLlamacppSnapshotSlots.mockResolvedValue({
      ...slots,
      latest_operation_id: "operation-one"
    })
    render(
      <LlamacppSnapshotsAdmin profile={profile} onProfileChanged={vi.fn()} />
    )
    expect(await screen.findByText(/Latest operation: Complete/)).toBeVisible()
    expect(api.getLlamacppSnapshotOperation).toHaveBeenCalledWith(
      "test-profile",
      "operation-one",
      expect.any(AbortSignal)
    )
    expect(api.saveLlamacppSnapshot).not.toHaveBeenCalled()
    expect(api.restoreLlamacppSnapshot).not.toHaveBeenCalled()
  })

  it("gets a new signed token for every new manual mutation", async () => {
    api.getLlamacppSnapshotSlots.mockResolvedValueOnce({
      ...slots,
      request_id: "old-token"
    })
    render(
      <LlamacppSnapshotsAdmin profile={profile} onProfileChanged={vi.fn()} />
    )
    const save = await screen.findByRole("button", { name: "Save snapshot" })
    await waitFor(() => expect(save).toBeEnabled())
    fireEvent.click(save)
    await waitFor(() =>
      expect(api.saveLlamacppSnapshot).toHaveBeenCalledExactlyOnceWith(
        "test-profile",
        {
          slot_id: 0,
          expected_launch_generation: "generation-one",
          request_id: "fresh-token"
        }
      )
    )
  })

  it("rejects a changed generation before dispatch", async () => {
    api.getLlamacppSnapshotSlots
      .mockResolvedValueOnce(slots)
      .mockResolvedValueOnce(slots)
      .mockResolvedValue({ ...slots, launch_generation: "generation-two" })
    render(
      <LlamacppSnapshotsAdmin profile={profile} onProfileChanged={vi.fn()} />
    )
    const save = await screen.findByRole("button", { name: "Save snapshot" })
    await waitFor(() => expect(save).toBeEnabled())
    fireEvent.click(save)
    expect(await screen.findByText(/Runtime changed/)).toBeVisible()
    expect(api.saveLlamacppSnapshot).not.toHaveBeenCalled()
  })

  it("aborts reads on profile change and discards a late old catalog", async () => {
    let resolveOld!: (value: unknown) => void
    api.listLlamacppSnapshots.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveOld = resolve
        })
    )
    const { rerender } = render(
      <LlamacppSnapshotsAdmin profile={profile} onProfileChanged={vi.fn()} />
    )
    await waitFor(() =>
      expect(api.listLlamacppSnapshots).toHaveBeenCalledOnce()
    )
    const signal = api.getLlamacppSnapshotSlots.mock.calls[0][1]
    rerender(
      <LlamacppSnapshotsAdmin
        profile={{ ...profile, profile_id: "other-profile" }}
        onProfileChanged={vi.fn()}
      />
    )
    await waitFor(() =>
      expect(api.listLlamacppSnapshots).toHaveBeenCalledTimes(2)
    )
    expect(signal.aborted).toBe(true)
    await act(async () => {
      resolveOld({ ...catalog, total: 999 })
    })
    expect(screen.queryByText(/999 saved copies/)).toBeNull()
  })

  it("rejects catalog and receipt reads crossing a launch generation", async () => {
    api.getLlamacppSnapshotSlots
      .mockResolvedValueOnce(slots)
      .mockResolvedValue({ ...slots, launch_generation: "generation-two" })
    render(
      <LlamacppSnapshotsAdmin profile={profile} onProfileChanged={vi.fn()} />
    )
    expect(await screen.findByText(/Runtime changed/)).toBeVisible()
    expect(screen.queryByRole("button", { name: "Save snapshot" })).toBeNull()
    expect(api.saveLlamacppSnapshot).not.toHaveBeenCalled()
  })

  it.each(["transport", "server", "malformed", "unrecognized-rejection"])(
    "shows uncertain %s outcomes without retrying and allows explicit Stop",
    async (outcome) => {
      if (outcome === "malformed")
        api.saveLlamacppSnapshot.mockResolvedValue({})
      else if (outcome === "unrecognized-rejection")
        api.saveLlamacppSnapshot.mockRejectedValue(
          Object.assign(new Error("unrecognized response"), {
            status: 422,
            details: { detail: "unrecognized_response" }
          })
        )
      else
        api.saveLlamacppSnapshot.mockRejectedValue(
          outcome === "server"
            ? Object.assign(new Error("snapshot_storage_unavailable"), {
                status: 503,
                details: { detail: "snapshot_storage_unavailable" }
              })
            : new Error("connection closed")
        )
      render(
        <LlamacppSnapshotsAdmin profile={profile} onProfileChanged={vi.fn()} />
      )
      const save = await screen.findByRole("button", { name: "Save snapshot" })
      await waitFor(() => expect(save).toBeEnabled())
      fireEvent.click(save)
      expect(
        await screen.findByRole("button", { name: "Stop recovery" })
      ).toBeVisible()
      expect(api.saveLlamacppSnapshot).toHaveBeenCalledOnce()
      expect(save).toBeDisabled()
      fireEvent.click(screen.getByRole("button", { name: "Stop recovery" }))
      fireEvent.click(
        screen.getByRole("button", { name: "Stop runtime and inference" })
      )
      await waitFor(() =>
        expect(api.stopLlamacppProfile).toHaveBeenCalledExactlyOnceWith(
          "test-profile"
        )
      )
    }
  )

  it("polls active receipts only while visible and aborts on unmount", async () => {
    api.getLlamacppSnapshotSlots.mockResolvedValue({
      ...slots,
      capability: "busy",
      latest_operation_id: "operation-one"
    })
    api.getLlamacppSnapshotOperation.mockResolvedValue({
      ...receipt,
      state: "saving"
    })
    const { unmount } = render(
      <LlamacppSnapshotsAdmin profile={profile} onProfileChanged={vi.fn()} />
    )
    await screen.findByText(/Latest operation: Saving/)
    Object.defineProperty(document, "visibilityState", {
      configurable: true,
      value: "hidden"
    })
    fireEvent(document, new Event("visibilitychange"))
    const calls = api.getLlamacppSnapshotOperation.mock.calls.length
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 1700))
    })
    expect(api.getLlamacppSnapshotOperation).toHaveBeenCalledTimes(calls)
    Object.defineProperty(document, "visibilityState", {
      configurable: true,
      value: "visible"
    })
    fireEvent(document, new Event("visibilitychange"))
    await waitFor(() =>
      expect(api.getLlamacppSnapshotOperation).toHaveBeenCalledTimes(calls + 1)
    )
    const signal = api.getLlamacppSnapshotSlots.mock.calls.at(-1)![1]
    unmount()
    expect(signal.aborted).toBe(true)
  })

  it("discards a late mutation response after changing profile", async () => {
    let finish!: (value: unknown) => void
    api.saveLlamacppSnapshot.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          finish = resolve
        })
    )
    const { rerender } = render(
      <LlamacppSnapshotsAdmin profile={profile} onProfileChanged={vi.fn()} />
    )
    const save = await screen.findByRole("button", { name: "Save snapshot" })
    await waitFor(() => expect(save).toBeEnabled())
    fireEvent.click(save)
    await waitFor(() => expect(api.saveLlamacppSnapshot).toHaveBeenCalledOnce())
    rerender(
      <LlamacppSnapshotsAdmin
        profile={{ ...profile, profile_id: "other-profile" }}
        onProfileChanged={vi.fn()}
      />
    )
    await waitFor(() =>
      expect(api.listLlamacppSnapshots).toHaveBeenCalledTimes(2)
    )
    await act(async () => {
      finish({ ...receipt, state: "outcome_unknown" })
    })
    expect(screen.queryByRole("button", { name: "Stop recovery" })).toBeNull()
    expect(screen.queryByText(/Latest operation/)).toBeNull()
  })

  it("updates enablement without starting, stopping or mutating a slot", async () => {
    render(
      <LlamacppSnapshotsAdmin profile={profile} onProfileChanged={vi.fn()} />
    )
    const disable = screen.getByRole("button", { name: "Disable snapshots" })
    await waitFor(() => expect(disable).toBeEnabled())
    fireEvent.click(disable)
    await waitFor(() =>
      expect(api.updateLlamacppProfile).toHaveBeenCalledExactlyOnceWith(
        "test-profile",
        { snapshots_enabled: false }
      )
    )
    expect(api.stopLlamacppProfile).not.toHaveBeenCalled()
    expect(api.saveLlamacppSnapshot).not.toHaveBeenCalled()
    expect(api.restoreLlamacppSnapshot).not.toHaveBeenCalled()
  })
})

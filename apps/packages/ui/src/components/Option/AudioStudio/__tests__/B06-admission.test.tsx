import React from "react"
import { act, renderHook, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { beforeEach, describe, expect, it, vi } from "vitest"
const service = vi.hoisted(() => ({ create: vi.fn(), list: vi.fn() }))
vi.mock("@/services/audio-studio", () => ({ createAudioStudioGeneration: (...args: unknown[]) => service.create(...args), listAudioStudioJobs: (...args: unknown[]) => service.list(...args) }))
vi.mock("../AudioStudioSession", () => ({ useAudioStudioSession: () => ({ ownerKey: "owner-seven", options: { assertCurrent: () => {} } }) }))
import { useCreateAudioStudioGeneration } from "@/hooks/useAudioStudioGeneration"
import { restoreAudioStudioIntents, admitAudioStudioJob } from "../audioStudioAdmission"
import { useAudioStudioStore } from "@/store/audio-studio"
const payload = (key: string) => ({ kind: "speech", provider: "tts", idempotency_key: key, target_resource_kind: "section", target_resource_id: "s", target_revision_id: "r", options: { model: "pocket_tts", voice: "reference" } })
const wrapper = ({ children }: React.PropsWithChildren) => <QueryClientProvider client={new QueryClient()}>{children}</QueryClientProvider>
beforeEach(() => { vi.resetAllMocks(); sessionStorage.clear(); useAudioStudioStore.getState().resetAudioStudio(); service.list.mockResolvedValue([]) })
describe("B06 shared admission intent", () => {
  it("locks two entry points before either request completes", async () => {
    let finish!: (value: unknown) => void
    service.create.mockReturnValueOnce(new Promise((resolve) => { finish = resolve })).mockResolvedValue({ job_id: "duplicate", status: "queued" })
    const { result } = renderHook(() => [useCreateAudioStudioGeneration("p"), useCreateAudioStudioGeneration("p")], { wrapper })
    let first!: Promise<unknown>
    await act(async () => {
      first = result.current[0].mutateAsync(payload("first-operation-key"))
      await waitFor(() => expect(service.create).toHaveBeenCalledTimes(1))
      try { await expect(result.current[1].mutateAsync(payload("second-operation-key"))).rejects.toThrow(/pending|progress|queued/i) }
      finally { finish({ job_id: "j", status: "queued" }); await first }
    })
    expect(service.create).toHaveBeenCalledTimes(1)
  })
  it("reuses the retained key after a lost admission response across entry points", async () => {
    service.create.mockRejectedValueOnce(new Error("response timed out")).mockResolvedValueOnce({ job_id: "j", status: "queued" })
    const { result } = renderHook(() => [useCreateAudioStudioGeneration("p"), useCreateAudioStudioGeneration("p")], { wrapper })
    await act(async () => { await expect(result.current[0].mutateAsync(payload("retained-operation-key"))).rejects.toThrow("response timed out") })
    await act(async () => { await result.current[1].mutateAsync(payload("new-random-operation-key")) })
    expect(service.create.mock.calls.map((call) => call[1].idempotency_key)).toEqual(["retained-operation-key", "retained-operation-key"])
  })
  it("restores an uncertain identity after the page state is reset", async () => {
    service.create.mockRejectedValueOnce(new Error("response timed out")).mockResolvedValueOnce({ job_id: "j", status: "queued" })
    const first = renderHook(() => useCreateAudioStudioGeneration("p"), { wrapper })
    await act(async () => { await expect(first.result.current.mutateAsync(payload("reload-retained-key"))).rejects.toThrow("response timed out") })
    first.unmount(); useAudioStudioStore.getState().resetAudioStudio()
    restoreAudioStudioIntents("owner-seven")
    const second = renderHook(() => useCreateAudioStudioGeneration("p"), { wrapper })
    await act(async () => { await second.result.current.mutateAsync(payload("new-key-after-reload")) })
    expect(service.create.mock.calls.map((call) => call[1].idempotency_key)).toEqual(["reload-retained-key", "reload-retained-key"])
  })
  it("refuses changed inputs until the uncertain admission is reconciled", async () => {
    service.create.mockRejectedValueOnce(new Error("response timed out"))
    const { result } = renderHook(() => useCreateAudioStudioGeneration("p"), { wrapper })
    await act(async () => { await expect(result.current.mutateAsync(payload("original-uncertain-key"))).rejects.toThrow() })
    await act(async () => { await expect(result.current.mutateAsync({ ...payload("new-operation-key"), options: { voice: "different" } })).rejects.toThrow(/pending admission|previous admission/i) })
    expect(service.create).toHaveBeenCalledTimes(1)
  })
  it("creates a fresh intent only after the previous job is known terminal", async () => {
    service.create.mockResolvedValueOnce({ job_id: "j", status: "queued" }).mockResolvedValueOnce({ job_id: "retry", status: "queued" })
    const { result } = renderHook(() => useCreateAudioStudioGeneration("p"), { wrapper })
    await act(async () => { await result.current.mutateAsync(payload("first-admission-key")) })
    service.list.mockResolvedValue([{ job_id: "j", status: "failed" }])
    await act(async () => { await result.current.mutateAsync(payload("legitimate-retry-key")) })
    expect(service.create.mock.calls.map((call) => call[1].idempotency_key)).toEqual(["first-admission-key", "legitimate-retry-key"])
  })
  it("never sends a job when its durable identity cannot be retained", async () => {
    const write = vi.spyOn(Storage.prototype, "setItem").mockImplementationOnce(() => { throw new Error("browser storage full") })
    const { result } = renderHook(() => useCreateAudioStudioGeneration("p"), { wrapper })
    await act(async () => { await expect(result.current.mutateAsync(payload("retain-before-send-key"))).rejects.toThrow("browser storage full") })
    expect(service.create).not.toHaveBeenCalled()
    write.mockRestore()
  })
  it("does not restore another owner's pending identity", async () => {
    service.create.mockRejectedValueOnce(new Error("response timed out"))
    const { result } = renderHook(() => useCreateAudioStudioGeneration("p"), { wrapper })
    await act(async () => { await expect(result.current.mutateAsync(payload("private-pending-key"))).rejects.toThrow() })
    useAudioStudioStore.getState().resetAudioStudio(); restoreAudioStudioIntents("owner-eight")
    expect(useAudioStudioStore.getState().jobIntents).toEqual({})
    restoreAudioStudioIntents("owner-seven")
    expect(useAudioStudioStore.getState().jobIntents.p.body.idempotency_key).toBe("private-pending-key")
  })

  it("does not overwrite uncertain identity in read-only recovery", async () => {
    sessionStorage.setItem("tldw:audio-studio:admissions:owner-seven", "invalid-json")
    await expect(admitAudioStudioJob("owner-seven", "p", "generation", payload("read-only-operation-key"), { assertCanWrite: () => { throw new Error("Read only recovery") } })).rejects.toThrow("Read only recovery")
    expect(service.list).not.toHaveBeenCalled()
    expect(service.create).not.toHaveBeenCalled()
    expect(sessionStorage.getItem("tldw:audio-studio:admissions:owner-seven")).toBe("invalid-json")
  })

})

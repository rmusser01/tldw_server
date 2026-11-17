import React from "react"
import { act, renderHook, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { beforeEach, describe, expect, it, vi } from "vitest"
const service = vi.hoisted(() => ({ jobs: vi.fn(), artifacts: vi.fn() }))
vi.mock("@/services/audio-studio", () => ({ listAudioStudioJobs: (...args: unknown[]) => service.jobs(...args), listAudioStudioArtifacts: (...args: unknown[]) => service.artifacts(...args) }))
const session = { ownerKey: "lifecycle-owner", options: { assertCurrent: () => {} } }
vi.mock("../AudioStudioSession", () => ({ useAudioStudioSession: () => session }))
import { useAudioStudioJobs } from "../useAudioStudioJobs"
import { saveAudioStudioIntent } from "../audioStudioAdmission"
import { useAudioStudioArtifacts } from "@/hooks/useAudioStudioProjects"
import { useAudioStudioStore } from "@/store/audio-studio"
beforeEach(() => { vi.resetAllMocks(); sessionStorage.clear(); useAudioStudioStore.getState().resetAudioStudio() })
describe("B06 completed generation reconciliation", () => {
  it("keeps admission locked until newly completed artifacts are loaded", async () => {
    let finish!: (value: unknown) => void
    service.jobs.mockResolvedValue([{ job_id: "j", status: "completed" }])
    service.artifacts.mockResolvedValueOnce([]).mockReturnValueOnce(new Promise((resolve) => { finish = resolve }))
    saveAudioStudioIntent(session.ownerKey, "p", { kind: "generation", job_id: "j", body: { kind: "speech", provider: "tts", idempotency_key: "retained-operation-key", target_resource_kind: "section", target_resource_id: "s", target_revision_id: "r" } })
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const wrapper = ({children}: React.PropsWithChildren) => <QueryClientProvider client={client}>{children}</QueryClientProvider>
    const { result } = renderHook(() => ({ artifacts: useAudioStudioArtifacts("p"), jobs: useAudioStudioJobs("p") }), { wrapper })
    await waitFor(() => expect(result.current.jobs.isSuccess).toBe(true))
    expect(useAudioStudioStore.getState().jobIntents.p).toBeDefined()
    await waitFor(() => expect(service.artifacts).toHaveBeenCalledTimes(2))
    await act(async () => { finish([{ artifact_id: "newly-completed" }]) })
    await waitFor(() => expect(useAudioStudioStore.getState().jobIntents.p).toBeUndefined())
    expect(result.current.artifacts.data?.[0].artifact_id).toBe("newly-completed")
  })
})

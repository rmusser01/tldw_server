import React from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
const state = vi.hoisted(() => ({ key: "owner-a", clear: (() => {}) as () => void, guards: new Map<string, () => boolean>(), scope: vi.fn() }))
vi.mock("@/hooks/useChatDraftOwner", () => ({ useChatDraftOwner: (clear: () => void) => {
  state.clear = clear
  const key = state.key
  if (!state.guards.has(key)) state.guards.set(key, () => state.key === key)
  return { ownerKey: key, isCurrent: state.guards.get(key) }
} }))
vi.mock("@/services/service-prompts", () => ({ resolveServicePromptScope: (...args: unknown[]) => state.scope(...args) }))
import { AudioStudioSession, useAudioStudioSession } from "../AudioStudioSession"
import { useAudioStudioStore } from "@/store/audio-studio"
const Probe = () => {
  const session = useAudioStudioSession()
  return <><p>Verified {session?.ownerKey}</p>{session?.readOnlyReason ? <p>Read only verified</p> : null}</>
}
beforeEach(() => { vi.resetAllMocks(); state.key = "owner-a"; state.guards.clear(); sessionStorage.clear(); useAudioStudioStore.getState().resetAudioStudio() })
describe("B06 account session", () => {
  it("never publishes a deferred old-owner scope into the new session", async () => {
    let old!: (value: unknown) => void
    state.scope.mockReturnValueOnce(new Promise((resolve) => { old = resolve })).mockResolvedValueOnce({ config: { serverUrl: "https://studio.example", authMode: "multi-user" }, userId: 8 })
    const view = render(<AudioStudioSession><Probe /></AudioStudioSession>)
    expect(screen.queryByText("Verified owner-a")).not.toBeInTheDocument()
    act(() => { state.clear(); state.key = "owner-b"; view.rerender(<AudioStudioSession><Probe /></AudioStudioSession>) })
    await screen.findByText("Verified owner-b")
    await act(async () => { old({ config: { serverUrl: "https://studio.example", authMode: "multi-user" }, userId: 7 }) })
    expect(screen.queryByText("Verified owner-a")).not.toBeInTheDocument()
    expect(screen.getByText("Verified owner-b")).toBeInTheDocument()
  })
  it("shows a truthful recovery state when pending identity cannot be restored", async () => {
    sessionStorage.setItem("tldw:audio-studio:admissions:owner-a", "invalid-json")
    state.scope.mockResolvedValue({ config: { serverUrl: "https://studio.example", authMode: "multi-user" }, userId: 7 })
    render(<AudioStudioSession><Probe /></AudioStudioSession>)
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent(/could not be restored/i))
    expect(screen.queryByText("Verified owner-a")).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Review server jobs without new work" }))
    expect(screen.getByText("Read only verified")).toBeInTheDocument()
    expect(sessionStorage.getItem("tldw:audio-studio:admissions:owner-a")).toBe("invalid-json")
  })
})

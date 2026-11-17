import React from "react"
import { act, fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
const state = vi.hoisted(() => ({ ticket: vi.fn(), current: true }))
vi.mock("@/services/audio-studio", () => ({ mintAudioStudioArtifactMediaTicket: (...args: unknown[]) => state.ticket(...args) }))
vi.mock("../AudioStudioSession", () => ({ useAudioStudioSession: () => ({ ownerKey: "owner", options: { assertCurrent: () => { if (!state.current) throw new Error("owner changed") } } }) }))
import { ArtifactOutput } from "../ArtifactOutput"
const artifact = (id: string) => ({ artifact_id: id, artifact_type: "generated_audio", mime_type: "audio/wav", metadata: {}, created_at: "now" })
beforeEach(() => { vi.resetAllMocks(); state.current = true })
describe("B06 output lifetime", () => {
  it("hides the prior preview immediately when the artifact changes", async () => {
    state.ticket.mockResolvedValue({ ticket_url: "https://studio.example/private-a" })
    const view = render(<ArtifactOutput projectId="p" artifact={artifact("a")} />)
    await act(async () => { fireEvent.click(screen.getByRole("button", { name: "Load audio preview" })) })
    expect(screen.getByLabelText("Audio a")).toHaveAttribute("src", "https://studio.example/private-a")
    view.rerender(<ArtifactOutput projectId="p" artifact={artifact("b")} />)
    expect(screen.queryByLabelText("Audio b")).not.toBeInTheDocument()
  })
  it("never downloads after the owner changes during ticket issuance", async () => {
    let finish!: (value: unknown) => void
    state.ticket.mockReturnValue(new Promise((resolve) => { finish = resolve }))
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {})
    render(<ArtifactOutput projectId="p" artifact={artifact("a")} />)
    fireEvent.click(screen.getByRole("button", { name: "Download generated_audio" }))
    await act(async () => { state.current = false; finish({ ticket_url: "https://studio.example/private-a" }) })
    expect(click).not.toHaveBeenCalled()
    click.mockRestore()
  })
  it("never downloads after its artifact changes during ticket issuance", async () => {
    let finish!: (value: unknown) => void
    state.ticket.mockReturnValue(new Promise((resolve) => { finish = resolve }))
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {})
    const view = render(<ArtifactOutput projectId="p" artifact={artifact("a")} />)
    fireEvent.click(screen.getByRole("button", { name: "Download generated_audio" }))
    view.rerender(<ArtifactOutput projectId="q" artifact={artifact("a")} />)
    await act(async () => { finish({ ticket_url: "https://studio.example/private-a" }) })
    expect(click).not.toHaveBeenCalled()
    click.mockRestore()
  })
})

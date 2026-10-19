import React from "react"
import { act, fireEvent, render, screen, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { sha256 } from "@noble/hashes/sha2.js"
import { bytesToHex, utf8ToBytes } from "@noble/hashes/utils.js"
import type { AudioStudioArtifact } from "@/services/audio-studio"
const mocks = vi.hoisted(() => ({ artifacts: [] as AudioStudioArtifact[], ticket: vi.fn() }))
vi.mock("@/hooks/useAudioStudioProjects", () => ({ useAudioStudioArtifacts: () => ({ data: mocks.artifacts }) }))
vi.mock("@/hooks/useAudioStudioGeneration", () => ({ useCreateAudioStudioRender: () => ({ isPending: false }), useCreateAudioStudioExport: () => ({ isPending: false }) }))
vi.mock("../useAudioStudioJobs", () => ({ useAudioStudioJobs: () => ({ data: [] }), isAudioStudioJobActive: () => false }))
vi.mock("../AudioStudioSession", () => ({ useAudioStudioSession: () => ({ ownerKey: "owner", options: {} }) }))
vi.mock("@/services/audio-studio", async (original) => ({ ...await original<typeof import("@/services/audio-studio")>(), mintAudioStudioArtifactMediaTicket: (...args: unknown[]) => mocks.ticket(...args) }))
import { RenderExportPanel } from "../RenderExportPanel"
import { useAudioStudioStore } from "@/store/audio-studio"
const sectionAudio = (id: string, text: string): AudioStudioArtifact => ({ artifact_id: "audio-" + id, artifact_type: "generated_audio", source_resource_kind: "section", source_resource_id: id, source_revision_id: "r1", mime_type: "audio/wav", metadata: { source_provider_hint: "pocket_tts", source_text_hash: bytesToHex(sha256(utf8ToBytes(text))), model: "pocket_tts", voice: "custom:reference" }, created_at: "2026-10-16T00:00:00Z" })
beforeEach(() => {
  vi.clearAllMocks()
  useAudioStudioStore.setState({ projects: [], activeProject: null, activeProjectId: null, dirtyProjectIds: {}, jobIntents: {} })
  useAudioStudioStore.getState().setProjects([{ project_id: "p", title: "Ordered", workflow: "narration", status: "draft", current_revision_id: "r2", sections: [{ section_id: "first", title: "First saved section", body_text: "First", workflow: "narration", order: 1 }, { section_id: "second", title: "Second saved section", body_text: "Second", workflow: "narration", order: 0 }], tracks: [], clips: [], settings: { speech: { model: "pocket_tts", voice: "custom:reference", provider_hint: "pocket_tts", format: "wav" } } }])
  useAudioStudioStore.getState().setActiveProjectId("p")
  mocks.artifacts = [sectionAudio("first", "First"), sectionAudio("second", "Second"), { artifact_id: "mix", artifact_type: "preview_mix", source_revision_id: "r2", mime_type: "audio/wav", metadata: { source_artifacts: [{ artifact_id: "audio-second", source_revision_id: "r1" }, { artifact_id: "audio-first", source_revision_id: "r1" }] }, created_at: "2026-10-16T00:00:01Z" }]
  mocks.ticket.mockResolvedValue({ ticket_url: "https://studio.example/mix" })
})
describe("B06 saved speech timeline", () => {
  it("plays the current rendered sections in their saved order through the native player", async () => {
    render(<RenderExportPanel />)
    const timeline = screen.getByRole("region", { name: "Saved speech timeline" })
    expect(within(timeline).getAllByRole("listitem").map(item => item.textContent)).toEqual(["Second saved section", "First saved section"])
    await act(async () => { fireEvent.click(within(timeline).getByRole("button", { name: "Load audio preview" })) })
    expect(within(timeline).getByLabelText("Audio mix")).toHaveAttribute("src", "https://studio.example/mix")
    expect(mocks.ticket).toHaveBeenCalledWith("p", "mix", "playback", {})
    expect(screen.getAllByText("preview_mix: mix")).toHaveLength(1)
  })
  it("keeps an older mix recoverable but does not present it as the current timeline after regeneration", () => {
    mocks.artifacts.push({ ...sectionAudio("first", "First"), artifact_id: "audio-first-new", created_at: "2026-10-16T00:00:02Z" })
    render(<RenderExportPanel />)
    const timeline = screen.getByRole("region", { name: "Saved speech timeline" })
    expect(within(timeline).queryByRole("button", { name: "Load audio preview" })).not.toBeInTheDocument()
    expect(within(timeline).getByText(/Create a preview render/)).toBeInTheDocument()
    expect(screen.getByText("preview_mix: mix")).toBeInTheDocument()
  })
})

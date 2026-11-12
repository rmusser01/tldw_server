import { describe, expect, it } from "vitest"
import { sha256 } from "@noble/hashes/sha2.js"
import { bytesToHex, utf8ToBytes } from "@noble/hashes/utils.js"
import { getCurrentSectionArtifacts, buildSectionRenderRequest, getSpeechOptions, getCurrentSectionMix } from "../generationPayload"
import type { AudioStudioProject } from "@/store/audio-studio"
import type { AudioStudioArtifact } from "@/services/audio-studio"
const project = { project_id: "p", title: "Two", workflow: "narration", status: "draft", current_revision_id: "r2", sections: [{ section_id: "s2", title: "Two", body_text: "Second", workflow: "narration", order: 0 }, { section_id: "s1", title: "One", body_text: "First", workflow: "narration", order: 1 }], tracks: [], clips: [], settings: { speech: { model: "pocket_tts", voice: "custom:reference", provider_hint: "pocket_tts", format: "wav" } } } satisfies AudioStudioProject
const artifact = (section: string, text: string, id: string): AudioStudioArtifact => ({ artifact_id: id, artifact_type: "generated_audio", source_resource_kind: "section", source_resource_id: section, source_revision_id: "r1", mime_type: "audio/wav", metadata: { source_provider_hint: "pocket_tts", source_text_hash: bytesToHex(sha256(utf8ToBytes(text))), model: "pocket_tts", voice: "custom:reference" }, created_at: "2026-10-16T00:00:00Z" })
describe("B06 saved speech output", () => {
  it("retains unchanged audio after another section changes and renders the new order", () => {
    const outputs = [artifact("s1", "First", "a1"), artifact("s2", "Second", "a2")]
    expect(getCurrentSectionArtifacts(project, outputs).map((a) => a?.artifact_id)).toEqual(["a2", "a1"])
    expect(buildSectionRenderRequest(project, outputs)).toMatchObject({ render_type: "preview_mix", target_resource_kind: "render", target_revision_id: "r2", options: { output_format: "wav", artifact_refs: [{ artifact_id: "a2", source_revision_id: "r1" }, { artifact_id: "a1", source_revision_id: "r1" }] } })
  })
  it("refuses stale text and changed voices without discarding older artifacts", () => {
    const outputs = [artifact("s1", "First", "a1"), artifact("s2", "Old", "a2")]
    expect(() => buildSectionRenderRequest(project, outputs)).toThrow(/Generate/)
    expect(getCurrentSectionArtifacts({ ...project, settings: { speech: { model: "pocket_tts", voice: "other" } } }, outputs).every((a) => !a)).toBe(true)
    expect(outputs).toHaveLength(2)
  })
  it("uses the selected section voice and project model, never a hidden default", () => {
    expect(getSpeechOptions(project, { ...project.sections[0], settings: { voice: "guest" } })).toMatchObject({ model: "pocket_tts", voice: "guest", provider_hint: "pocket_tts", format: "wav" })
    expect(() => getSpeechOptions({ ...project, settings: {} }, project.sections[0])).toThrow(/model and voice/)
  })
  it("requires a new render after regeneration at the same project revision", () => {
    const outputs = [artifact("s1", "First", "a1"), artifact("s2", "Second", "a2")]
    const oldMix: AudioStudioArtifact = { artifact_id: "mix-old", artifact_type: "preview_mix", source_revision_id: "r2", metadata: { source_artifacts: [{ artifact_id: "a2", source_revision_id: "r1" }, { artifact_id: "a1", source_revision_id: "r1" }] }, created_at: "2026-10-16T00:00:01Z" }
    expect(getCurrentSectionMix(project, [...outputs, oldMix])?.artifact_id).toBe("mix-old")
    const newer = { ...artifact("s1", "First", "a1-new"), created_at: "2026-10-16T00:00:02Z" }
    expect(getCurrentSectionMix(project, [...outputs, oldMix, newer])).toBeUndefined()
  })

  it("does not reuse a different requested provider's speech", () => {
    const outputs = [artifact("s1", "First", "a1"), artifact("s2", "Second", "a2")]
    const other = { ...project, settings: { speech: { ...project.settings.speech, provider_hint: "other-service" } } }
    expect(getCurrentSectionArtifacts(other, outputs).every((artifact) => !artifact)).toBe(true)
  })

})


describe("B06 Music render", () => {
  const music = { ...project, workflow: "music" as const, sections: [], settings: { music: { provider: "ace_step", prompt: "Observatory", duration: 15 } } };
  const cue: AudioStudioArtifact = { artifact_id: "music-new", artifact_type: "generated_audio", provider: "ace_step", source_resource_kind: "track", source_resource_id: "music-cue-1", source_revision_id: "r2", mime_type: "audio/wav", metadata: { task_id: "upstream-1" }, created_at: "2026-10-16T00:00:03Z" };
  it("renders the current saved music cue without requiring speech sections", () => {
    expect(buildSectionRenderRequest(music, [cue])).toMatchObject({ options: { artifact_refs: [{ artifact_id: "music-new", source_revision_id: "r2" }] } });
  });
  it("keeps music mixes current and refuses old revisions or a replaced cue", () => {
    const mix: AudioStudioArtifact = { artifact_id: "music-mix", artifact_type: "preview_mix", source_revision_id: "r2", metadata: { source_artifacts: [{ artifact_id: "music-new", source_revision_id: "r2" }] }, created_at: "2026-10-16T00:00:04Z" };
    expect(getCurrentSectionMix(music, [cue, mix])?.artifact_id).toBe("music-mix");
    expect(() => buildSectionRenderRequest(music, [{ ...cue, source_revision_id: "old" }])).toThrow(/Generate/);
    expect(getCurrentSectionMix(music, [cue, mix, { ...cue, artifact_id: "replacement", created_at: "2026-10-16T00:00:05Z" }])).toBeUndefined();
  });
});

import type { AudioStudioProject } from "@/store/audio-studio"

export const getProjectRevisionId = (
  project: Pick<AudioStudioProject, "current_revision_id" | "revision_id"> | null
) => project?.current_revision_id ?? project?.revision_id ?? null

export const getMusicTrackTargetId = (
  project: Pick<AudioStudioProject, "tracks"> | null
) =>
  project?.tracks.find((track) => track.kind === "music")?.track_id ??
  "music-cue-1"

export const getFirstSectionTargetId = (
  project: Pick<AudioStudioProject, "sections"> | null
) =>
  project?.sections.find((section) => Boolean(section.section_id))?.section_id ??
  null

export const createAudioStudioIdempotencyKey = (
  kind: string,
  projectId: string
) => {
  const random =
    typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
      ? crypto.randomUUID()
      : `${Date.now()}-${Math.random().toString(36).slice(2)}`

  return `${kind}-${projectId}-${random}`.slice(0, 200)
}

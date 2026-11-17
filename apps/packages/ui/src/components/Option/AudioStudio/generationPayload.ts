import { sha256 } from '@noble/hashes/sha2.js';
import { bytesToHex, utf8ToBytes } from '@noble/hashes/utils.js';
import type {
  AudioStudioArtifact,
  AudioStudioSection,
  AudioStudioRenderCreateRequest,
} from '@/services/audio-studio';
import type { AudioStudioProject } from '@/store/audio-studio';

const LOCAL_DRAFT_REVISION_ID = 'local-draft';

export const isUsableProjectRevisionId = (revisionId?: string | null) =>
  Boolean(revisionId && revisionId !== LOCAL_DRAFT_REVISION_ID);

export const getProjectRevisionId = (
  project: Pick<AudioStudioProject, 'current_revision_id' | 'revision_id'> | null,
) => {
  const revisionId = project?.current_revision_id ?? project?.revision_id ?? null;
  return isUsableProjectRevisionId(revisionId) ? revisionId : null;
};

export const getMusicTrackTargetId = (project: Pick<AudioStudioProject, 'tracks'> | null) =>
  project?.tracks.find((track) => track.kind === 'music')?.track_id ?? 'music-cue-1';

export const getFirstSectionTargetId = (project: Pick<AudioStudioProject, 'sections'> | null) =>
  project?.sections.find((section) => Boolean(section.section_id))?.section_id ?? null;

export const createAudioStudioIdempotencyKey = (kind: string, projectId: string) => {
  const random =
    typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
      ? crypto.randomUUID()
      : `${Date.now()}-${Math.random().toString(36).slice(2)}`;

  return `${kind}-${projectId}-${random}`.slice(0, 200);
};

export const getSpeechOptions = (project: AudioStudioProject, section: AudioStudioSection) => {
  const saved = project.settings.speech as Record<string, unknown> | undefined;
  const model = typeof saved?.model === 'string' ? saved.model : '';
  const voice =
    typeof section.settings?.voice === 'string' && section.settings.voice
      ? section.settings.voice
      : saved?.voice;
  if (
    !model ||
    typeof voice !== 'string' ||
    !voice ||
    typeof saved?.provider_hint !== 'string' ||
    !saved.provider_hint
  )
    throw new Error('Choose a provider, model and voice, then save the project.');
  return {
    ...saved,
    model,
    voice,
    format: 'wav',
    provider_hint: typeof saved?.provider_hint === 'string' ? saved.provider_hint : undefined,
  };
};

export const getCurrentSectionArtifacts = (
  project: AudioStudioProject,
  artifacts: AudioStudioArtifact[],
) =>
  [...project.sections]
    .sort((a, b) => a.order - b.order)
    .map((section) => {
      let options: ReturnType<typeof getSpeechOptions>;
      try {
        options = getSpeechOptions(project, section);
      } catch {
        return undefined;
      }
      const hash = bytesToHex(sha256(utf8ToBytes(section.body_text ?? '')));
      return [...artifacts]
        .sort((a, b) => b.created_at.localeCompare(a.created_at))
        .find(
          (artifact) =>
            artifact.artifact_type === 'generated_audio' &&
            artifact.source_resource_kind === 'section' &&
            artifact.source_resource_id === section.section_id &&
            artifact.metadata.source_text_hash === hash &&
            artifact.metadata.model === options.model &&
            artifact.metadata.voice === options.voice &&
            artifact.metadata.source_provider_hint === options.provider_hint &&
            artifact.mime_type === 'audio/wav',
        );
    });

const getCurrentRenderArtifacts = (project: AudioStudioProject, artifacts: AudioStudioArtifact[]) => {
  if (project.workflow !== 'music') return getCurrentSectionArtifacts(project, artifacts);
  const provider = (project.settings.music as { provider?: string } | undefined)?.provider ?? 'ace_step';
  return [[...artifacts]
    .sort((a, b) => b.created_at.localeCompare(a.created_at))
    .find((artifact) => artifact.artifact_type === 'generated_audio'
      && artifact.provider === provider && artifact.source_resource_kind === 'track'
      && artifact.source_resource_id === getMusicTrackTargetId(project)
      && artifact.source_revision_id === getProjectRevisionId(project)
      && artifact.mime_type === 'audio/wav')];
};

export const buildSectionRenderRequest = (
  project: AudioStudioProject,
  artifacts: AudioStudioArtifact[],
): AudioStudioRenderCreateRequest => {
  const current = getCurrentRenderArtifacts(project, artifacts);
  if (!current.length || current.some((artifact) => !artifact))
    throw new Error(project.workflow === 'music' ? 'Generate music for this saved revision before rendering.' : 'Generate current audio for every section before rendering.');
  const revision = getProjectRevisionId(project);
  if (!revision) throw new Error('Save the project before rendering.');
  const key = createAudioStudioIdempotencyKey('render', project.project_id);
  return {
    idempotency_key: key,
    target_resource_kind: 'render',
    target_resource_id: key.slice(-100),
    target_revision_id: revision,
    render_type: 'preview_mix',
    options: {
      output_format: 'wav',
      artifact_refs: current.map((artifact) => ({
        artifact_id: artifact!.artifact_id,
        source_revision_id: artifact!.source_revision_id,
      })),
    },
  };
};

export const getCurrentSectionMix = (
  project: AudioStudioProject,
  artifacts: AudioStudioArtifact[],
) => {
  const current = getCurrentRenderArtifacts(project, artifacts);
  if (!current.length || current.some((artifact) => !artifact)) return undefined;
  return [...artifacts]
    .sort((a, b) => b.created_at.localeCompare(a.created_at))
    .find((mix) => {
      const sources = mix.metadata.source_artifacts;
      return (
        ['preview_mix', 'final_mix'].includes(mix.artifact_type) &&
        mix.source_revision_id === getProjectRevisionId(project) &&
        Array.isArray(sources) &&
        sources.length === current.length &&
        sources.every(
          (source, index) =>
            source?.artifact_id === current[index]?.artifact_id &&
            source?.source_revision_id === current[index]?.source_revision_id,
        )
      );
    });
};

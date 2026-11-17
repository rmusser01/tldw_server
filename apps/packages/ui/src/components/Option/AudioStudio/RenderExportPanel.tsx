import { useAudioStudioSession } from './AudioStudioSession';
import React from 'react';
import { Button, Typography } from 'antd';
import { useAudioStudioArtifacts } from '@/hooks/useAudioStudioProjects';
import {
  useCreateAudioStudioRender,
  useCreateAudioStudioExport,
} from '@/hooks/useAudioStudioGeneration';
import { useAudioStudioStore } from '@/store/audio-studio';
import {
  buildSectionRenderRequest,
  createAudioStudioIdempotencyKey,
  getProjectRevisionId,
  getCurrentSectionMix,
} from './generationPayload';
import { isAudioStudioJobActive, useAudioStudioJobs } from './useAudioStudioJobs';
import { ArtifactOutput } from './ArtifactOutput';

export const RenderExportPanel: React.FC = () => {
  const session = useAudioStudioSession();
  const project = useAudioStudioStore((state) => state.activeProject);
  const intent = useAudioStudioStore((state) =>
    project ? state.jobIntents[project.project_id] : undefined,
  );
  const dirty = useAudioStudioStore((state) =>
    project ? Boolean(state.dirtyProjectIds[project.project_id]) : false,
  );
  const artifacts = useAudioStudioArtifacts(project?.project_id ?? null);
  const jobs = useAudioStudioJobs(project?.project_id ?? null);
  const render = useCreateAudioStudioRender(project?.project_id ?? null);
  const exportJob = useCreateAudioStudioExport(project?.project_id ?? null);
  const [error, setError] = React.useState<string | null>(null);
  let reason =
    session?.readOnlyReason ??
    (!project
      ? 'Select a project first.'
      : dirty
        ? 'Save your changes before rendering.'
        : artifacts.isError || jobs.isError
          ? 'Reload project jobs and saved audio before rendering.'
          : undefined);
  let payload: ReturnType<typeof buildSectionRenderRequest> | undefined;
  if (project && !reason) {
    try {
      payload = buildSectionRenderRequest(project, artifacts.data ?? []);
    } catch (failure) {
      reason = failure instanceof Error ? failure.message : 'Generate section audio first.';
    }
  }
  const mix = project ? getCurrentSectionMix(project, artifacts.data ?? []) : undefined;
  const busy =
    jobs.isLoading ||
    Boolean(intent) ||
    render.isPending ||
    exportJob.isPending ||
    (jobs.data ?? []).some((job) => isAudioStudioJobActive(job.status));
  const queue = async (kind: 'render' | 'export') => {
    setError(null);
    try {
      if (!project || reason || !payload) return;
      if (kind === 'render') await render.mutateAsync(payload);
      else {
        if (!mix) return;
        const key = createAudioStudioIdempotencyKey('export', project.project_id);
        await exportJob.mutateAsync({
          idempotency_key: key,
          target_resource_kind: 'export',
          target_resource_id: key.slice(-100),
          target_revision_id: getProjectRevisionId(project)!,
          export_type: 'single_audio',
          source_render_id: mix.source_resource_id ?? undefined,
          options: {
            artifact_refs: [
              { artifact_id: mix.artifact_id, source_revision_id: mix.source_revision_id },
            ],
          },
        });
      }
    } catch (failure) {
      setError(
        failure instanceof Error
          ? failure.message
          : 'The job could not be queued. Earlier audio is retained.',
      );
    }
  };
  return (
    <section className="space-y-3 rounded-md border border-border bg-surface p-3">
      <Typography.Text strong>Render & Export</Typography.Text>
      <p className="text-xs text-text-muted">
        {project?.workflow === 'music' ? 'Render the generated music for this saved revision.' : 'Render saved speech sections in their listed order.'} Timeline clip effects are not included in this render.
      </p>
      <Button
        block
        disabled={Boolean(reason) || busy}
        title={reason}
        onClick={() => void queue('render')}
      >
        Create preview render
      </Button>
      <Button
        block
        disabled={Boolean(reason) || !mix || busy}
        title={
          reason ?? (!mix ? 'Create a preview render for this saved revision first.' : undefined)
        }
        onClick={() => void queue('export')}
      >
        Create export
      </Button>
      {reason ? <p className="text-xs text-text-muted">{reason}</p> : null}
      {error ? <p role="alert">{error}</p> : null}
      {artifacts.isError ? (
        <p role="alert">
          Saved audio could not load.{' '}
          <Button onClick={() => void artifacts.refetch()}>Reload audio</Button>
        </p>
      ) : null}
      {project && project.workflow !== 'music' ? (
        <section aria-label="Saved speech timeline" className="space-y-2">
          <Typography.Text strong>Timeline preview</Typography.Text>
          <ol className="list-inside list-decimal text-sm">
            {[...(project.sections ?? [])]
              .sort((first, second) => first.order - second.order)
              .map((section) => <li key={section.section_id}>{section.title || 'Untitled section'}</li>)}
          </ol>
          {mix && !dirty ? (
            <ArtifactOutput projectId={project.project_id} artifact={mix} />
          ) : (
            <p className="text-xs text-text-muted">Create a preview render after saving and generating all sections to hear them in this order.</p>
          )}
        </section>
      ) : null}
      <Typography.Text strong>Saved audio & exports</Typography.Text>
      {(artifacts.data ?? [])
        .filter(
          (artifact) =>
            (artifact.mime_type?.startsWith('audio/') || artifact.artifact_type === 'package') &&
            !(project?.workflow !== 'music' && !dirty && artifact.artifact_id === mix?.artifact_id),
        )
        .map((artifact) =>
          project ? (
            <ArtifactOutput
              key={`${project.project_id}:${artifact.artifact_id}`}
              projectId={project.project_id}
              artifact={artifact}
            />
          ) : null,
        )}
    </section>
  );
};

import React from 'react';
import { Button, Input, Typography } from 'antd';
import { useRetryAudioStudioAdmission } from '@/hooks/useAudioStudioGeneration';
import { useQuery } from '@tanstack/react-query';
import { useAudioStudioStore } from '@/store/audio-studio';
import { getAudioStudioSpeechProviders, getAudioStudioVoices } from '@/services/audio-studio';
import { useAudioStudioGenerationActions } from './useAudioStudioGenerationActions';
import { useAudioStudioSession } from './AudioStudioSession';
import {
  isAudioStudioJobActive,
  useAudioStudioJobs,
  useCancelAudioStudioJob,
} from './useAudioStudioJobs';

export const GenerationPanel: React.FC = () => {
  const project = useAudioStudioStore((state) => state.activeProject);
  const update = useAudioStudioStore((state) => state.updateProjectLocal);
  const session = useAudioStudioSession();
  const [validationError, setValidationError] = React.useState<string | null>(null);
  const generation = useAudioStudioGenerationActions();
  const jobs = useAudioStudioJobs(project?.project_id ?? null);
  const retry = useRetryAudioStudioAdmission(project?.project_id ?? null);
  const cancel = useCancelAudioStudioJob(project?.project_id ?? null);
  const speech = (project?.settings.speech ?? {}) as Record<string, string>;
  const providers = useQuery({
    queryKey: ['audio-studio', 'speech-providers', session?.ownerKey],
    queryFn: () => getAudioStudioSpeechProviders(session?.options),
  });
  const voices = useQuery({
    queryKey: ['audio-studio', 'voices', session?.ownerKey, speech.provider_hint, speech.model],
    queryFn: () => getAudioStudioVoices(speech.provider_hint, speech.model, session?.options),
    enabled: Boolean(speech.provider_hint && speech.model),
  });
  const change = (values: Record<string, string>) => {
    if (project)
      update(project.project_id, {
        settings: { ...project.settings, speech: { ...speech, ...values, format: 'wav' } },
      });
  };
  const modelIds = providers.data?.providers[speech.provider_hint]?.models ?? [];
  const manualModel = Boolean(speech.provider_hint && providers.data?.providers[speech.provider_hint] && !modelIds.length);
  const selectClass = 'w-full rounded border border-border bg-surface p-2';
  return (
    <section className="space-y-3 rounded-md border border-border bg-surface p-3">
      <Typography.Text strong>Speech settings & jobs</Typography.Text>
      {project?.workflow !== 'music' ? (
        <>
          <label className="block">
            Speech provider
            <select
              className={selectClass}
              disabled={Boolean(retry.intent || session?.readOnlyReason)}
              aria-label="Speech provider"
              value={speech.provider_hint ?? ''}
              onChange={(e) => change({ provider_hint: e.target.value, model: '', voice: '' })}
            >
              <option value="">Choose a provider</option>
              {Object.entries(providers.data?.providers ?? {}).map(([id, info]) => (
                <option key={id} value={id}>
                  {info.display_name ?? id}
                </option>
              ))}
            </select>
          </label>
          <label className="block">
            Speech model
            {manualModel ? <Input
              aria-label="Speech model"
              value={speech.model ?? ''}
              disabled={Boolean(retry.intent || session?.readOnlyReason)}
              placeholder="Supported model ID"
              onChange={(e) => change({ model: e.target.value, voice: '' })}
            /> : <select
              className={selectClass}
              disabled={Boolean(retry.intent || session?.readOnlyReason)}
              aria-label="Speech model"
              value={speech.model ?? ''}
              onChange={(e) => change({ model: e.target.value, voice: '' })}
            >
              <option value="">Choose a model</option>
              {modelIds.map((id) => <option key={id} value={id}>{id}</option>)}
            </select>}
          </label>
          {manualModel ? <p className="text-xs text-text-muted">This provider does not publish a model list. Enter its supported model ID; generation is validated by the server.</p> : null}
          <label className="block">
            Project voice
            <select
              className={selectClass}
              disabled={Boolean(retry.intent || session?.readOnlyReason)}
              aria-label="Project voice"
              value={speech.voice ?? ''}
              onChange={(e) => change({ voice: e.target.value })}
            >
              <option value="">Choose a voice</option>
              {(voices.data ?? []).map((voice) => {
                const id = voice.voice_id ?? voice.id ?? voice.name;
                return (
                  <option key={id} value={id}>
                    {voice.name ?? id}
                  </option>
                );
              })}
            </select>
          </label>
          {speech.provider_hint && speech.model ? <Button size="small" disabled={voices.isFetching} onClick={() => void voices.refetch()}>Reload voices</Button> : null}
          {voices.data?.length === 0 ? <p className="text-xs text-text-muted">No compatible voices found. Add a custom voice for this provider in Speech, then reload voices.</p> : null}
          <p className="text-xs text-text-muted">
            Save these settings, then generate each section. WAV preserves audio for rendering.
          </p>
          {providers.isError || voices.isError ? (
            <p role="alert">
              Voice settings could not load. Check the server connection and retry.
            </p>
          ) : null}
        </>
      ) : (
        <p>Music requires a configured music provider. Speech output does not replace Music.</p>
      )}
      <Button
        disabled={
          Boolean(session?.readOnlyReason) ||
          project?.workflow === 'music' ||
          Boolean(generation.speechDisabledReason) ||
          generation.isPending ||
          jobs.isLoading ||
          (jobs.data ?? []).some((job) => isAudioStudioJobActive(job.status))
        }
        title={
          project?.workflow === 'music'
            ? 'Use the saved Music prompt to generate a cue.'
            : generation.speechDisabledReason
        }
        onClick={() => {
          setValidationError(null);
          try {
            void generation.queueSpeechGeneration()?.catch(() => {});
          } catch (error) {
            setValidationError(
              error instanceof Error ? error.message : 'Check the saved speech settings.',
            );
          }
        }}
      >
        {project?.workflow === 'music' ? 'Queue music generation' : 'Queue speech generation'}
      </Button>
      {validationError ? <p role="alert">{validationError}</p> : null}
      {generation.error ? (
        <p role="alert">
          Generation could not be queued. Check the selected model and voice; saved text is
          retained.
        </p>
      ) : null}
      {retry.intent ? (
        <div role="status" className="text-xs">
          <p>
            Pending {retry.intent.kind} admission for {retry.intent.body.target_resource_id}. Its
            identity is retained across reloads.
          </p>
          {!retry.intent.job_id ? (
            <Button
              disabled={retry.isPending}
              onClick={() => {
                if (retry.intent) void retry.mutateAsync(retry.intent.body).catch(() => {});
              }}
            >
              Retry pending admission
            </Button>
          ) : null}
          {retry.error ? <p role="alert">{retry.error.message}</p> : null}
        </div>
      ) : null}
      <Typography.Text strong>Project jobs</Typography.Text>
      {jobs.isLoading ? <p role="status">Loading jobs…</p> : null}
      {jobs.isError ? (
        <p role="alert">
          Jobs could not load. <Button onClick={() => void jobs.refetch()}>Reload jobs</Button>
        </p>
      ) : null}
      {(jobs.data ?? []).map((job) => (
        <div key={job.job_id} className="rounded border border-border p-2 text-xs">
          <p>
            {job.job_type}: {job.status} — {job.target_resource_id}
          </p>
          <p className="break-all">{job.job_id}</p>
          {job.error ? (
            <p role="alert">
              {job.error} Your text and earlier audio are retained. Correct the settings and
              generate the section again.
            </p>
          ) : null}
          {job.status === 'skipped' ? (
            <p>
              The project changed before this job ran. Save and generate the intended section again.
            </p>
          ) : null}
          {job.status === 'queued' ? (
            <Button
              size="small"
              disabled={cancel.isPending}
              onClick={() => {
                void cancel.mutateAsync(job.job_id).catch(() => {});
              }}
            >
              Cancel queued job
            </Button>
          ) : null}
        </div>
      ))}
      {cancel.isError ? (
        <p role="alert">
          Cancellation was not applied; the job may already have started. Reload its status.
        </p>
      ) : null}
    </section>
  );
};

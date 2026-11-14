import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { useAudioStudioStore } from '@/store/audio-studio';
import { getAudioStudioProviders } from '@/services/audio-studio';
import { useAudioStudioSession } from './AudioStudioSession';
import { Button, Input, Slider, Typography } from 'antd';
import { useAudioStudioGenerationActions } from './useAudioStudioGenerationActions';

const { TextArea } = Input;
const { Text } = Typography;

export const MusicWorkflow: React.FC = () => {
  const { isPending, hasPendingIntent, error, musicDisabledReason, queueMusicGeneration } =
    useAudioStudioGenerationActions();
  const project = useAudioStudioStore((state) => state.activeProject);
  const update = useAudioStudioStore((state) => state.updateProjectLocal);
  const session = useAudioStudioSession();
  const catalog = useQuery({
    queryKey: ['audio-studio', 'providers', session?.ownerKey],
    queryFn: () => getAudioStudioProviders(session?.options),
  });
  const music = (project?.settings.music ?? {}) as Record<string, unknown>;
  const prompt = typeof music.prompt === 'string' ? music.prompt : '';
  const lyrics = typeof music.lyrics === 'string' ? music.lyrics : '';
  const style = typeof music.style === 'string' ? music.style : '';
  const provider = typeof music.provider === 'string' ? music.provider : 'ace_step';
  const duration =
    typeof music.duration === 'number' && Number.isFinite(music.duration)
      ? Math.max(15, Math.min(180, music.duration))
      : 45;
  const change = (values: Record<string, unknown>) => {
    if (project)
      update(project.project_id, {
        settings: { ...project.settings, music: { ...music, ...values } },
      });
  };
  const available = catalog.data?.providers.some(
    (item) => item.provider_id === provider && item.supported_kinds.includes('music'),
  );
  const trimmedPrompt = prompt.trim();
  const disabledReason =
    session?.readOnlyReason ??
    musicDisabledReason ??
    (catalog.isLoading
      ? 'Checking music providers…'
      : !available
        ? 'Configure a music service before generating Music.'
        : undefined) ??
    (!trimmedPrompt ? 'Enter a prompt before generating music.' : undefined);

  const submitMusicGeneration = () => {
    if (disabledReason) return;

    void queueMusicGeneration(
      {
        prompt: trimmedPrompt,
        lyrics,
        style,
        duration,
      },
      provider,
    )?.catch(() => {});
  };

  return (
    <section className="min-w-0 rounded-md border border-border bg-surface p-4">
      <fieldset
        disabled={!project || Boolean(session?.readOnlyReason) || isPending || hasPendingIntent}
        className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_240px]"
      >
        <div className="space-y-3">
          <label className="block">
            <Text strong className="mb-1 block">
              Prompt
            </Text>
            <TextArea
              aria-label="Prompt"
              placeholder="Warm documentary intro with restrained percussion"
              autoSize={{ minRows: 5, maxRows: 10 }}
              value={prompt}
              onChange={(event) => change({ prompt: event.target.value })}
            />
          </label>
          <label className="block">
            <Text strong className="mb-1 block">
              Lyrics
            </Text>
            <TextArea
              aria-label="Lyrics"
              placeholder="Optional lyrics or vocal direction"
              autoSize={{ minRows: 5, maxRows: 10 }}
              value={lyrics}
              onChange={(event) => change({ lyrics: event.target.value })}
            />
          </label>
        </div>
        <div className="space-y-4">
          <label className="block">
            <Text strong className="mb-1 block">
              Style
            </Text>
            <Input
              aria-label="Style"
              placeholder="cinematic, ambient"
              value={style}
              onChange={(event) => change({ style: event.target.value })}
            />
          </label>
          <label className="block">
            <Text strong className="mb-1 block">
              Provider
            </Text>
            <select
              aria-label="Provider"
              className="w-full rounded-md border border-border bg-surface px-3 py-2 text-sm"
              value={provider}
              onChange={(event) => change({ provider: event.target.value })}
            >
              <option value="ace_step">ACE-Step</option>
            </select>
          </label>
          <div>
            <Text strong className="mb-2 block">
              Duration
            </Text>
            <Slider
              min={15}
              max={180}
              value={duration}
              onChange={(value) => {
                if (typeof value === 'number') change({ duration: value });
              }}
            />
          </div>
          <Button
            type="primary"
            block
            disabled={Boolean(disabledReason) || isPending}
            title={disabledReason}
            loading={isPending}
            onClick={submitMusicGeneration}
          >
            Generate music
          </Button>
        </div>
      </fieldset>
      {!catalog.isLoading && !available ? (
        <p role="status">
          Configure a music service before generating Music. A speech provider cannot produce a
          music cue.
        </p>
      ) : null}
      {catalog.isError ? (
        <p role="alert">
          Music provider discovery failed.{' '}
          <Button onClick={() => void catalog.refetch()}>Reload music providers</Button>
        </p>
      ) : null}
      {error ? <p role="alert">{error.message}</p> : null}
    </section>
  );
};

import { useAudioStudioSession } from './AudioStudioSession';
import React from 'react';
import { Button, Input, Typography } from 'antd';
import { useAudioStudioStore } from '@/store/audio-studio';
import { useAudioStudioGenerationActions } from './useAudioStudioGenerationActions';
import { isAudioStudioJobActive, useAudioStudioJobs } from './useAudioStudioJobs';

export const SpeechSections: React.FC = () => {
  const session = useAudioStudioSession();
  const project = useAudioStudioStore((state) => state.activeProject);
  const update = useAudioStudioStore((state) => state.updateProjectLocal);
  const generation = useAudioStudioGenerationActions();
  const jobs = useAudioStudioJobs(project?.project_id ?? null);
  const [error, setError] = React.useState<string | null>(null);
  if (!project) return <p>Select or create a project to write sections.</p>;
  const sections = [...project.sections].sort((a, b) => a.order - b.order);
  const busy =
    Boolean(session?.readOnlyReason) ||
    generation.isPending ||
    generation.hasPendingIntent ||
    (jobs.data ?? []).some((job) => isAudioStudioJobActive(job.status));
  const edit = (id: string, values: Partial<(typeof sections)[number]>) =>
    update(project.project_id, {
      sections: project.sections.map((section) =>
        section.section_id === id ? { ...section, ...values } : section,
      ),
    });
  const move = (index: number, delta: number) => {
    const ordered = [...sections];
    [ordered[index], ordered[index + delta]] = [ordered[index + delta], ordered[index]];
    update(project.project_id, {
      sections: ordered.map((section, order) => ({ ...section, order })),
    });
  };
  const generate = async (id: string) => {
    setError(null);
    try {
      await generation.queueSpeechGenerationForSection(id);
    } catch (failure) {
      setError(
        failure instanceof Error
          ? failure.message
          : 'Speech could not be queued. Your saved text is retained.',
      );
    }
  };
  return (
    <section className="space-y-4 rounded-md border border-border bg-surface p-4">
      <Typography.Title level={3}>Speech sections</Typography.Title>
      <p className="text-sm text-text-muted">
        Write the spoken text in playback order. Save changes before generating. Each section can
        use a different voice.
      </p>
      {sections.map((section, index) => (
        <fieldset
          key={section.section_id}
          className="space-y-2 rounded border border-border p-3"
          disabled={busy}
        >
          <legend>Section {index + 1}</legend>
          <Input
            aria-label={`Section ${index + 1} title`}
            value={section.title}
            onChange={(e) => edit(section.section_id, { title: e.target.value })}
          />
          <Input.TextArea
            aria-label={`Section ${index + 1} text`}
            value={section.body_text ?? ''}
            autoSize={{ minRows: 4, maxRows: 16 }}
            onChange={(e) => edit(section.section_id, { body_text: e.target.value })}
          />
          <Input
            aria-label={`Section ${index + 1} voice override`}
            placeholder="Optional voice ID; blank uses the project voice"
            value={typeof section.settings?.voice === 'string' ? section.settings.voice : ''}
            onChange={(e) =>
              edit(section.section_id, { settings: { ...section.settings, voice: e.target.value } })
            }
          />
          <div className="flex flex-wrap gap-2">
            <Button
              disabled={busy || index === 0}
              onClick={() => move(index, -1)}
              aria-label={`Move section ${index + 1} up`}
            >
              Move up
            </Button>
            <Button
              disabled={busy || index === sections.length - 1}
              onClick={() => move(index, 1)}
              aria-label={`Move section ${index + 1} down`}
            >
              Move down
            </Button>
            <Button
              disabled={
                busy || Boolean(generation.speechDraftDisabledReason) || !section.body_text?.trim()
              }
              title={generation.speechDraftDisabledReason}
              onClick={() => void generate(section.section_id)}
            >
              Generate section {index + 1}
            </Button>
          </div>
          <small className="text-text-muted">Saved identity: {section.section_id}</small>
        </fieldset>
      ))}
      <Button
        disabled={busy}
        onClick={() =>
          update(project.project_id, {
            sections: [
              ...sections,
              {
                section_id: `section_${crypto.randomUUID()}`,
                workflow: project.workflow,
                title: `Section ${sections.length + 1}`,
                body_text: '',
                order: sections.length,
              },
            ],
          })
        }
      >
        Add section
      </Button>
      {error ? (
        <p role="alert" className="text-danger">
          {error}
        </p>
      ) : null}
    </section>
  );
};

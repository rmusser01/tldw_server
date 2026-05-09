import React, { FormEvent, useMemo, useState } from 'react';
import { Button } from '@web/components/ui/Button';
import { Input } from '@web/components/ui/Input';
import {
  createVNPlayIdempotencyKey,
  getVNPlayErrorInfo,
  isRecoverableVNPlayConflict,
} from '@web/components/vn-play/runtime';
import { submitVNPlayTurn } from '@web/lib/api/vnPlay';
import type { VNPlayEvent, VNPlayMode, VNPlayTurnResponse } from '@web/types/vn-play';

export interface DialoguePanelProps {
  events: VNPlayEvent[];
  mode: VNPlayMode;
  sceneVersion: number;
  sessionId: number;
  onError?: (error: unknown) => void;
  onTurn: (response: VNPlayTurnResponse) => void;
}

interface DialogueLine {
  speaker: string;
  text: string;
}

function latestDialogue(events: VNPlayEvent[]): DialogueLine[] {
  for (const event of [...events].reverse()) {
    const payload = event.event_payload ?? {};
    const rawDialogue = payload.dialogue;
    if (Array.isArray(rawDialogue)) {
      const lines = rawDialogue
        .filter((item): item is Record<string, unknown> => item !== null && typeof item === 'object')
        .map((item) => ({
          speaker: typeof item.speaker === 'string' && item.speaker.trim() ? item.speaker : 'Narrator',
          text: typeof item.text === 'string' ? item.text : '',
        }))
        .filter((line) => line.text.trim());
      if (lines.length > 0) return lines;
    }

    const narration = payload.narrative_text ?? payload.narration;
    if (typeof narration === 'string' && narration.trim()) {
      return [{ speaker: 'Narrator', text: narration }];
    }
  }
  return [];
}

export default function DialoguePanel({
  events,
  mode,
  sceneVersion,
  sessionId,
  onError,
  onTurn,
}: DialoguePanelProps) {
  const [inputText, setInputText] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const dialogue = useMemo(() => latestDialogue(events), [events]);

  const submitFreeformTurn = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const trimmed = inputText.trim();
    if (!trimmed) return;

    setIsSubmitting(true);
    setError(null);
    try {
      const response = await submitVNPlayTurn(sessionId, {
        input_text: trimmed,
        client_scene_version: sceneVersion,
        idempotency_key: createVNPlayIdempotencyKey('freeform'),
      });
      setInputText('');
      onTurn(response);
    } catch (turnError) {
      const errorInfo = getVNPlayErrorInfo(turnError);
      setError(isRecoverableVNPlayConflict(turnError) ? null : errorInfo.message);
      onError?.(turnError);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <section className="rounded-md border border-border bg-bg p-4">
      <h3 className="mb-3 text-sm font-semibold uppercase tracking-normal text-text-muted">Dialogue</h3>
      {dialogue.length > 0 ? (
        <div className="mb-4 grid gap-2">
          {dialogue.map((line, index) => (
            <p key={`${line.speaker}-${index}`} className="text-sm">
              <span className="font-medium">{line.speaker}: </span>
              {line.text}
            </p>
          ))}
        </div>
      ) : (
        <p className="mb-4 text-sm text-text-muted">No dialogue events.</p>
      )}

      {mode === 'freeform' && (
        <form className="grid gap-2" onSubmit={submitFreeformTurn}>
          <Input
            label="Freeform input"
            value={inputText}
            onChange={(event) => setInputText(event.target.value)}
          />
          <Button loading={isSubmitting} type="submit">
            Send turn
          </Button>
        </form>
      )}

      {error && (
        <div className="mt-3 rounded-md border border-danger/30 bg-danger/10 px-3 py-2 text-sm text-danger">
          {error}
        </div>
      )}
    </section>
  );
}

import React, { FormEvent, useState } from 'react';
import { Badge } from '@web/components/ui/Badge';
import { Button } from '@web/components/ui/Button';
import { Input } from '@web/components/ui/Input';
import {
  createVNPlayIdempotencyKey,
  getVNPlayErrorInfo,
  isRecoverableVNPlayConflict,
} from '@web/components/vn-play/runtime';
import { submitVNPlayTurn } from '@web/lib/api/vnPlay';
import type { VNPlayChoice, VNPlayTurnResponse } from '@web/types/vn-play';

export interface ChoicePanelProps {
  choices: VNPlayChoice[];
  sceneVersion: number;
  sessionId: number;
  onError?: (error: unknown) => void;
  onTurn: (response: VNPlayTurnResponse) => void;
}

function choiceMetadata(choice: VNPlayChoice): Record<string, unknown> {
  return choice.metadata && typeof choice.metadata === 'object' ? choice.metadata : {};
}

function metadataString(metadata: Record<string, unknown>, key: string): string | null {
  const value = metadata[key];
  return typeof value === 'string' && value.trim() ? value : null;
}

function isGeneratedChoice(choice: VNPlayChoice): boolean {
  const metadata = choiceMetadata(choice);
  return (
    choice.source === 'generated' ||
    metadataString(metadata, 'source') === 'generated' ||
    Boolean(metadataString(metadata, 'generation_point_key'))
  );
}

export default function ChoicePanel({
  choices,
  sceneVersion,
  sessionId,
  onError,
  onTurn,
}: ChoicePanelProps) {
  const [customAction, setCustomAction] = useState('');
  const [submittingChoiceId, setSubmittingChoiceId] = useState<string | null>(null);
  const [isSubmittingAction, setIsSubmittingAction] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submitChoice = async (choiceId: string) => {
    setSubmittingChoiceId(choiceId);
    setError(null);
    try {
      const response = await submitVNPlayTurn(sessionId, {
        choice_id: choiceId,
        client_scene_version: sceneVersion,
        idempotency_key: createVNPlayIdempotencyKey('choice'),
      });
      onTurn(response);
    } catch (turnError) {
      const errorInfo = getVNPlayErrorInfo(turnError);
      setError(isRecoverableVNPlayConflict(turnError) ? null : errorInfo.message);
      onError?.(turnError);
    } finally {
      setSubmittingChoiceId(null);
    }
  };

  const submitCustomAction = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const text = customAction.trim();
    if (!text) return;

    setIsSubmittingAction(true);
    setError(null);
    try {
      const response = await submitVNPlayTurn(sessionId, {
        custom_action: { text },
        client_scene_version: sceneVersion,
        idempotency_key: createVNPlayIdempotencyKey('action'),
      });
      setCustomAction('');
      onTurn(response);
    } catch (turnError) {
      const errorInfo = getVNPlayErrorInfo(turnError);
      setError(isRecoverableVNPlayConflict(turnError) ? null : errorInfo.message);
      onError?.(turnError);
    } finally {
      setIsSubmittingAction(false);
    }
  };

  return (
    <section className="rounded-md border border-border bg-bg p-4">
      <h3 className="mb-3 text-sm font-semibold uppercase tracking-normal text-text-muted">Choices</h3>
      {choices.length > 0 ? (
        <div className="grid gap-2">
          {choices.map((choice) => {
            const metadata = choiceMetadata(choice);
            const generated = isGeneratedChoice(choice);
            const generationPointKey = metadataString(metadata, 'generation_point_key');
            const status = metadataString(metadata, 'status');
            return (
              <Button
                key={choice.id}
                className="h-auto justify-start text-left"
                loading={submittingChoiceId === choice.id}
                onClick={() => void submitChoice(choice.id)}
                type="button"
                variant="secondary"
              >
                <span className="grid gap-1">
                  <span>{choice.text}</span>
                  {(generated || generationPointKey || status) && (
                    <span className="flex flex-wrap gap-1">
                      {generated && <Badge variant="info">Generated</Badge>}
                      {generationPointKey && <Badge variant="neutral">{generationPointKey}</Badge>}
                      {status && <Badge variant="neutral">{status}</Badge>}
                    </span>
                  )}
                </span>
              </Button>
            );
          })}
        </div>
      ) : (
        <p className="text-sm text-text-muted">No choices are available yet.</p>
      )}

      <form className="mt-4 grid gap-2" onSubmit={submitCustomAction}>
        <Input
          label="Custom action"
          value={customAction}
          onChange={(event) => setCustomAction(event.target.value)}
        />
        <Button loading={isSubmittingAction} type="submit" variant="secondary">
          Send action
        </Button>
      </form>

      {error && (
        <div className="mt-3 rounded-md border border-danger/30 bg-danger/10 px-3 py-2 text-sm text-danger">
          {error}
        </div>
      )}
    </section>
  );
}

import React, { FormEvent, useState } from 'react';
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
          {choices.map((choice) => (
            <Button
              key={choice.id}
              className="justify-start text-left"
              loading={submittingChoiceId === choice.id}
              onClick={() => void submitChoice(choice.id)}
              type="button"
              variant="secondary"
            >
              {choice.text}
            </Button>
          ))}
        </div>
      ) : (
        <p className="text-sm text-text-muted">No choices.</p>
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

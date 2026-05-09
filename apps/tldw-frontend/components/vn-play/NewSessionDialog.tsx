import React, { FormEvent, useEffect, useState } from 'react';
import { Button } from '@web/components/ui/Button';
import { Input } from '@web/components/ui/Input';
import type { VNPlayMode, VNPlaySessionCreate } from '@web/types/vn-play';

export interface NewSessionDialogProps {
  initialMode: VNPlayMode;
  isCreating: boolean;
  open: boolean;
  onClose: () => void;
  onCreateSession: (request: VNPlaySessionCreate) => Promise<void>;
}

function parsePositiveInteger(value: string): number | null {
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : null;
}

export default function NewSessionDialog({
  initialMode,
  isCreating,
  open,
  onClose,
  onCreateSession,
}: NewSessionDialogProps) {
  const [mode, setMode] = useState<VNPlayMode>(initialMode);
  const [title, setTitle] = useState('Untitled VN play session');
  const [primaryCharacterId, setPrimaryCharacterId] = useState('1');
  const [vnAssetPackId, setVnAssetPackId] = useState('1');
  const [linkedChatId, setLinkedChatId] = useState('');
  const [contentRating, setContentRating] = useState('general');
  const [formError, setFormError] = useState<string | null>(null);

  useEffect(() => {
    if (open) {
      setMode(initialMode);
      setFormError(null);
    }
  }, [initialMode, open]);

  if (!open) return null;

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const parsedPrimaryCharacterId = parsePositiveInteger(primaryCharacterId);
    const parsedPackId = parsePositiveInteger(vnAssetPackId);
    const trimmedTitle = title.trim();

    if (!trimmedTitle || !parsedPrimaryCharacterId || !parsedPackId) {
      setFormError('Enter a title, character ID, and asset pack ID.');
      return;
    }

    setFormError(null);
    await onCreateSession({
      mode,
      title: trimmedTitle,
      primary_character_id: parsedPrimaryCharacterId,
      vn_asset_pack_id: parsedPackId,
      linked_chat_id: linkedChatId.trim() || null,
      content_rating: contentRating.trim() || 'general',
    });
  };

  return (
    <div className="rounded-md border border-border bg-surface p-4">
      <div className="mb-4 flex items-start justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold">New VN play session</h2>
          <p className="text-sm text-text-muted">{mode === 'story' ? 'Story/CYOA' : 'Freeform'}</p>
        </div>
        <Button onClick={onClose} size="sm" type="button" variant="ghost">
          Close
        </Button>
      </div>

      <form className="grid gap-3 sm:grid-cols-2" onSubmit={handleSubmit}>
        <label className="block text-sm font-medium text-text">
          Mode
          <select
            className="mt-1 block w-full rounded-md border-border bg-bg shadow-sm focus:border-primary focus:ring-primary"
            value={mode}
            onChange={(event) => setMode(event.target.value as VNPlayMode)}
          >
            <option value="freeform">Freeform</option>
            <option value="story">Story/CYOA</option>
          </select>
        </label>
        <Input label="Title" value={title} onChange={(event) => setTitle(event.target.value)} />
        <Input
          inputMode="numeric"
          label="Primary character ID"
          value={primaryCharacterId}
          onChange={(event) => setPrimaryCharacterId(event.target.value)}
        />
        <Input
          inputMode="numeric"
          label="VN asset pack ID"
          value={vnAssetPackId}
          onChange={(event) => setVnAssetPackId(event.target.value)}
        />
        <Input
          label="Linked chat ID"
          placeholder="Optional"
          value={linkedChatId}
          onChange={(event) => setLinkedChatId(event.target.value)}
        />
        <Input
          label="Content rating"
          value={contentRating}
          onChange={(event) => setContentRating(event.target.value)}
        />

        {formError && (
          <div className="rounded-md border border-danger/30 bg-danger/10 px-3 py-2 text-sm text-danger sm:col-span-2">
            {formError}
          </div>
        )}

        <div className="flex flex-wrap justify-end gap-2 sm:col-span-2">
          <Button onClick={onClose} type="button" variant="secondary">
            Cancel
          </Button>
          <Button loading={isCreating} type="submit">
            Create session
          </Button>
        </div>
      </form>
    </div>
  );
}

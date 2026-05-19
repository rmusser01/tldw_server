import React, { FormEvent } from 'react';
import { Plus } from 'lucide-react';
import { Button } from '@web/components/ui/Button';
import { Input } from '@web/components/ui/Input';

export interface PackSetupProps {
  title: string;
  primaryCharacterId: string;
  isCreating?: boolean;
  onTitleChange: (value: string) => void;
  onPrimaryCharacterIdChange: (value: string) => void;
  onCreatePack: (event: FormEvent<HTMLFormElement>) => void;
}

export default function PackSetup({
  title,
  primaryCharacterId,
  isCreating = false,
  onTitleChange,
  onPrimaryCharacterIdChange,
  onCreatePack,
}: PackSetupProps) {
  return (
    <section className="rounded-md border border-border bg-surface p-4">
      <h2 className="mb-4 text-lg font-semibold">Setup</h2>
      <form className="grid gap-3 md:grid-cols-[minmax(0,1fr)_180px_auto]" onSubmit={onCreatePack}>
        <Input
          label="Pack title"
          value={title}
          onChange={(event) => onTitleChange(event.target.value)}
        />
        <Input
          label="Primary character ID"
          inputMode="numeric"
          min={1}
          type="number"
          value={primaryCharacterId}
          onChange={(event) => onPrimaryCharacterIdChange(event.target.value)}
        />
        <div className="flex items-end">
          <Button className="w-full gap-2" loading={isCreating} type="submit">
            <Plus aria-hidden className="h-4 w-4" />
            Create pack
          </Button>
        </div>
      </form>
    </section>
  );
}

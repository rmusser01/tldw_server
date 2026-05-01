import React from 'react';
import { Badge } from '@web/components/ui/Badge';
import type { VNAssetPack } from '@web/types/vn-assets';

export interface PackListProps {
  packs: VNAssetPack[];
  selectedPackId?: number | null;
  onSelectPack: (pack: VNAssetPack) => void;
}

export default function PackList({ packs, selectedPackId, onSelectPack }: PackListProps) {
  return (
    <aside className="rounded-md border border-border bg-surface p-4">
      <div className="mb-3 flex items-center justify-between gap-2">
        <h2 className="text-sm font-semibold uppercase tracking-normal text-text-muted">Packs</h2>
        <Badge variant="neutral">{packs.length}</Badge>
      </div>
      {packs.length === 0 ? (
        <p className="text-sm text-text-muted">No asset packs yet.</p>
      ) : (
        <div className="flex flex-col gap-2">
          {packs.map((pack) => {
            const selected = selectedPackId === pack.id;

            return (
              <button
                key={pack.id}
                aria-pressed={selected}
                className={`rounded-md border px-3 py-2 text-left text-sm transition-colors ${
                  selected
                    ? 'border-primary bg-primary/10 text-primary'
                    : 'border-border bg-bg hover:bg-surface2'
                }`}
                type="button"
                onClick={() => onSelectPack(pack)}
              >
                <span className="block font-medium">{pack.title}</span>
                <span className="block text-xs text-text-muted">
                  Character {pack.primary_character_id}
                </span>
              </button>
            );
          })}
        </div>
      )}
    </aside>
  );
}

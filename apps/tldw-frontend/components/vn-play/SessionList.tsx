import React from 'react';
import { Badge } from '@web/components/ui/Badge';
import { Button } from '@web/components/ui/Button';
import type { VNPlayMode, VNPlaySession } from '@web/types/vn-play';

export type VNPlayModeFilter = 'all' | VNPlayMode;

export interface SessionListProps {
  modeFilter: VNPlayModeFilter;
  selectedSessionId?: number | null;
  sessions: VNPlaySession[];
  onModeFilterChange: (mode: VNPlayModeFilter) => void;
  onSelectSession: (session: VNPlaySession) => void;
}

const modeFilters: Array<{ key: VNPlayModeFilter; label: string }> = [
  { key: 'all', label: 'All' },
  { key: 'freeform', label: 'Freeform' },
  { key: 'story', label: 'Story' },
  { key: 'scripted_story', label: 'Scripted Story' },
];

function modeLabel(mode: VNPlayMode): string {
  if (mode === 'scripted_story') return 'Scripted Story';
  return mode === 'story' ? 'Story' : 'Freeform';
}

export default function SessionList({
  modeFilter,
  selectedSessionId,
  sessions,
  onModeFilterChange,
  onSelectSession,
}: SessionListProps) {
  return (
    <aside className="rounded-md border border-border bg-surface p-4">
      <div className="mb-3 flex items-center justify-between gap-2">
        <h2 className="text-sm font-semibold uppercase tracking-normal text-text-muted">Sessions</h2>
        <Badge variant="neutral">{sessions.length}</Badge>
      </div>

      <div className="mb-4 flex flex-wrap gap-1" role="tablist" aria-label="VN play session mode">
        {modeFilters.map((filter) => {
          const active = modeFilter === filter.key;
          return (
            <Button
              key={filter.key}
              aria-selected={active}
              onClick={() => onModeFilterChange(filter.key)}
              role="tab"
              size="xs"
              type="button"
              variant={active ? 'primary' : 'secondary'}
            >
              {filter.label}
            </Button>
          );
        })}
      </div>

      {sessions.length === 0 ? (
        <p className="text-sm text-text-muted">No VN play sessions.</p>
      ) : (
        <div className="flex flex-col gap-2">
          {sessions.map((session) => {
            const selected = selectedSessionId === session.id;

            return (
              <button
                key={session.id}
                className={`rounded-md border px-3 py-2 text-left text-sm transition-colors ${
                  selected
                    ? 'border-primary bg-primary/10 text-primary'
                    : 'border-border bg-bg hover:bg-surface2'
                }`}
                type="button"
                onClick={() => onSelectSession(session)}
              >
                <span className="block font-medium">{session.title}</span>
                <span className="mt-1 flex items-center gap-2 text-xs text-text-muted">
                  <span>{modeLabel(session.mode)}</span>
                  <span>Scene {session.scene_version ?? session.scene_state?.scene_version ?? 0}</span>
                </span>
              </button>
            );
          })}
        </div>
      )}
    </aside>
  );
}

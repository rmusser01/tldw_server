import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { BookOpen, MessageSquarePlus } from 'lucide-react';
import { Badge } from '@web/components/ui/Badge';
import { Button } from '@web/components/ui/Button';
import NewSessionDialog from '@web/components/vn-play/NewSessionDialog';
import SessionList, { VNPlayModeFilter } from '@web/components/vn-play/SessionList';
import { createVNPlaySession, listVNPlaySessions } from '@web/lib/api/vnPlay';
import type { VNPlayMode, VNPlaySession, VNPlaySessionCreate } from '@web/types/vn-play';

function sessionModeLabel(mode: VNPlayMode): string {
  return mode === 'story' ? 'Story/CYOA' : 'Freeform';
}

export default function VNPlayWorkspace() {
  const [sessions, setSessions] = useState<VNPlaySession[]>([]);
  const [selectedSession, setSelectedSession] = useState<VNPlaySession | null>(null);
  const [modeFilter, setModeFilter] = useState<VNPlayModeFilter>('all');
  const [dialogMode, setDialogMode] = useState<VNPlayMode>('freeform');
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [isCreating, setIsCreating] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function loadSessions() {
      setIsLoading(true);
      setError(null);
      try {
        const nextSessions = await listVNPlaySessions();
        if (cancelled) return;
        setSessions(nextSessions);
        setSelectedSession(nextSessions[0] ?? null);
      } catch (loadError) {
        if (!cancelled) {
          setError(loadError instanceof Error ? loadError.message : 'Failed to load VN play sessions');
        }
      } finally {
        if (!cancelled) {
          setIsLoading(false);
        }
      }
    }

    void loadSessions();
    return () => {
      cancelled = true;
    };
  }, []);

  const filteredSessions = useMemo(() => {
    if (modeFilter === 'all') return sessions;
    return sessions.filter((session) => session.mode === modeFilter);
  }, [modeFilter, sessions]);

  const handleNewSession = useCallback((mode: VNPlayMode) => {
    setDialogMode(mode);
    setIsDialogOpen(true);
  }, []);

  const handleCreateSession = useCallback(async (request: VNPlaySessionCreate) => {
    setIsCreating(true);
    setError(null);
    try {
      const created = await createVNPlaySession(request);
      setSessions((previous) => [created, ...previous.filter((session) => session.id !== created.id)]);
      setSelectedSession(created);
      setModeFilter('all');
      setIsDialogOpen(false);
    } catch (createError) {
      setError(createError instanceof Error ? createError.message : 'Failed to create VN play session');
    } finally {
      setIsCreating(false);
    }
  }, []);

  const selectedMode = selectedSession ? sessionModeLabel(selectedSession.mode) : null;
  const sceneVersion = selectedSession?.scene_state?.scene_version ?? selectedSession?.scene_version ?? 0;

  return (
    <main className="min-h-screen bg-bg text-text">
      <div className="mx-auto flex w-full max-w-7xl flex-col gap-6 px-6 py-6">
        <header className="flex flex-col gap-3 border-b border-border pb-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex flex-wrap items-center gap-3">
              <h1 className="text-2xl font-semibold">VN play</h1>
              <Badge variant={selectedSession ? 'success' : 'neutral'}>
                {selectedSession ? selectedMode : 'No session'}
              </Badge>
            </div>
            <div className="flex flex-wrap gap-2">
              <Button className="gap-2" onClick={() => handleNewSession('freeform')} type="button">
                <MessageSquarePlus aria-hidden className="h-4 w-4" />
                New Freeform
              </Button>
              <Button className="gap-2" onClick={() => handleNewSession('story')} type="button" variant="secondary">
                <BookOpen aria-hidden className="h-4 w-4" />
                New Story
              </Button>
            </div>
          </div>
        </header>

        {isLoading && <p className="text-sm text-text-muted">Loading VN play sessions...</p>}
        {error && (
          <div className="rounded-md border border-danger/30 bg-danger/10 px-3 py-2 text-sm text-danger">
            {error}
          </div>
        )}

        <NewSessionDialog
          initialMode={dialogMode}
          isCreating={isCreating}
          open={isDialogOpen}
          onClose={() => setIsDialogOpen(false)}
          onCreateSession={handleCreateSession}
        />

        <section className="grid gap-4 lg:grid-cols-[280px_minmax(0,1fr)]">
          <SessionList
            modeFilter={modeFilter}
            selectedSessionId={selectedSession?.id}
            sessions={filteredSessions}
            onModeFilterChange={setModeFilter}
            onSelectSession={setSelectedSession}
          />

          <section className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(300px,380px)]">
            <div className="rounded-md border border-border bg-surface p-4">
              <h2 className="mb-4 text-lg font-semibold">Scene</h2>
              {selectedSession ? (
                <div className="grid gap-4">
                  <div>
                    <p className="text-xs uppercase tracking-normal text-text-muted">Selected session</p>
                    <p className="font-medium">Selected session: {selectedSession.title}</p>
                  </div>
                  <div className="min-h-72 rounded-md border border-border bg-bg p-4">
                    <div className="flex h-full min-h-64 items-center justify-center text-sm text-text-muted">
                      Scene preview
                    </div>
                  </div>
                  <div className="rounded-md border border-border bg-bg p-4">
                    <h3 className="mb-2 text-sm font-semibold uppercase tracking-normal text-text-muted">
                      Dialogue
                    </h3>
                    <p className="text-sm text-text-muted">No dialogue events.</p>
                  </div>
                </div>
              ) : (
                <p className="text-sm text-text-muted">No session selected.</p>
              )}
            </div>

            <aside className="rounded-md border border-border bg-surface p-4">
              <h2 className="mb-4 text-lg font-semibold">Runtime inspector</h2>
              {selectedSession ? (
                <dl className="grid gap-3 text-sm">
                  <div>
                    <dt className="text-xs uppercase tracking-normal text-text-muted">Mode</dt>
                    <dd className="font-medium">{selectedMode}</dd>
                  </div>
                  <div>
                    <dt className="text-xs uppercase tracking-normal text-text-muted">Character</dt>
                    <dd className="font-medium">Character {selectedSession.primary_character_id}</dd>
                  </div>
                  <div>
                    <dt className="text-xs uppercase tracking-normal text-text-muted">Asset pack</dt>
                    <dd className="font-medium">Pack {selectedSession.vn_asset_pack_id}</dd>
                  </div>
                  <div>
                    <dt className="text-xs uppercase tracking-normal text-text-muted">Scene version</dt>
                    <dd className="font-medium">{sceneVersion}</dd>
                  </div>
                </dl>
              ) : (
                <p className="text-sm text-text-muted">No session metadata.</p>
              )}
            </aside>
          </section>
        </section>
      </div>
    </main>
  );
}

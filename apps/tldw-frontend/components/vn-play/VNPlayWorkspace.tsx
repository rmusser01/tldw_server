import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { BookOpen, MessageSquarePlus } from 'lucide-react';
import { Badge } from '@web/components/ui/Badge';
import { Button } from '@web/components/ui/Button';
import ChoicePanel from '@web/components/vn-play/ChoicePanel';
import DialoguePanel from '@web/components/vn-play/DialoguePanel';
import NewSessionDialog from '@web/components/vn-play/NewSessionDialog';
import {
  createVNPlayIdempotencyKey,
  getVNPlayErrorInfo,
  isRecoverableVNPlayConflict,
} from '@web/components/vn-play/runtime';
import SceneInspector from '@web/components/vn-play/SceneInspector';
import SceneStage from '@web/components/vn-play/SceneStage';
import SessionList, { VNPlayModeFilter } from '@web/components/vn-play/SessionList';
import {
  createVNPlayCheckpoint,
  createVNPlaySession,
  getVNPlaySession,
  listVNPlayBranches,
  listVNPlayCheckpoints,
  listVNPlayEvents,
  listVNPlaySessions,
  restoreVNPlaySession,
  retryLastVNPlayTurn,
} from '@web/lib/api/vnPlay';
import type {
  VNPlayBranch,
  VNPlayCheckpoint,
  VNPlayChoice,
  VNPlayEvent,
  VNPlayMode,
  VNPlaySceneState,
  VNPlaySession,
  VNPlaySessionCreate,
  VNPlayTurnResponse,
} from '@web/types/vn-play';

function sessionModeLabel(mode: VNPlayMode): string {
  return mode === 'story' ? 'Story/CYOA' : 'Freeform';
}

function isVNPlayChoice(choice: VNPlayChoice | Record<string, unknown>): choice is VNPlayChoice {
  return (
    choice !== null &&
    typeof choice === 'object' &&
    typeof choice.id === 'string' &&
    typeof choice.text === 'string'
  );
}

function recoveryCopy(status: string | null): string | null {
  if (status === 'stale_scene_version') {
    return 'Scene changed on another client. The latest state was reloaded before you continue.';
  }
  if (status === 'turn_in_progress') {
    return 'A turn is already in progress for this session. Wait for it to finish, then reload if needed.';
  }
  if (status === 'turn_failed') {
    return 'The last turn failed before completion. You can retry the stored turn request without duplicating the user input.';
  }
  return null;
}

export default function VNPlayWorkspace() {
  const [sessions, setSessions] = useState<VNPlaySession[]>([]);
  const [selectedSession, setSelectedSession] = useState<VNPlaySession | null>(null);
  const [events, setEvents] = useState<VNPlayEvent[]>([]);
  const [checkpoints, setCheckpoints] = useState<VNPlayCheckpoint[]>([]);
  const [branches, setBranches] = useState<VNPlayBranch[]>([]);
  const [modeFilter, setModeFilter] = useState<VNPlayModeFilter>('all');
  const [dialogMode, setDialogMode] = useState<VNPlayMode>('freeform');
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [isCreating, setIsCreating] = useState(false);
  const [isCreatingCheckpoint, setIsCreatingCheckpoint] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [isRetryingTurn, setIsRetryingTurn] = useState(false);
  const [restoringCheckpointId, setRestoringCheckpointId] = useState<number | null>(null);
  const [turnStatus, setTurnStatus] = useState<string | null>(null);
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

  useEffect(() => {
    if (!selectedSession) {
      setEvents([]);
      setCheckpoints([]);
      setBranches([]);
      return;
    }

    let cancelled = false;
    async function loadSessionCollections() {
      try {
        const [nextEvents, nextCheckpoints, nextBranches] = await Promise.all([
          listVNPlayEvents(selectedSession.id),
          listVNPlayCheckpoints(selectedSession.id),
          listVNPlayBranches(selectedSession.id),
        ]);
        if (!cancelled) {
          setEvents(nextEvents);
          setCheckpoints(nextCheckpoints);
          setBranches(nextBranches);
        }
      } catch {
        if (!cancelled) {
          setEvents([]);
          setCheckpoints([]);
          setBranches([]);
        }
      }
    }

    void loadSessionCollections();
    return () => {
      cancelled = true;
    };
  }, [selectedSession?.id]);

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
      setEvents([]);
      setCheckpoints([]);
      setBranches([]);
      setModeFilter('all');
      setIsDialogOpen(false);
    } catch (createError) {
      setError(createError instanceof Error ? createError.message : 'Failed to create VN play session');
    } finally {
      setIsCreating(false);
    }
  }, []);

  const reloadSessionCollections = useCallback(async (sessionId: number) => {
    const [nextEvents, nextCheckpoints, nextBranches] = await Promise.all([
      listVNPlayEvents(sessionId),
      listVNPlayCheckpoints(sessionId),
      listVNPlayBranches(sessionId),
    ]);
    setEvents(nextEvents);
    setCheckpoints(nextCheckpoints);
    setBranches(nextBranches);
  }, []);

  const reloadSelectedSession = useCallback(async (sessionId: number) => {
    const nextSession = await getVNPlaySession(sessionId);
    setSelectedSession(nextSession);
    setSessions((previous) =>
      previous.map((session) => (session.id === nextSession.id ? nextSession : session))
    );
    await reloadSessionCollections(sessionId);
    return nextSession;
  }, [reloadSessionCollections]);

  const handleTurn = useCallback(async (response: VNPlayTurnResponse) => {
    if (!selectedSession) return;

    setError(null);
    setTurnStatus(response.status);
    const responseEvents = response.events ?? [];
    if (responseEvents.length > 0) {
      setEvents((previous) => {
        const byId = new Map(previous.map((event) => [event.id, event]));
        for (const event of responseEvents) {
          byId.set(event.id, event);
        }
        return [...byId.values()].sort((left, right) => left.sequence_number - right.sequence_number);
      });
    }

    const responseScene = response.scene_state ?? response.current_scene ?? null;
    if (response.session) {
      setSelectedSession(response.session);
      setSessions((previous) =>
        previous.map((session) => (session.id === response.session?.id ? response.session : session))
      );
      return;
    }

    if (responseScene) {
      setSelectedSession((previous) =>
        previous && previous.id === selectedSession.id
          ? {
              ...previous,
              scene_version: response.scene_version,
              scene_state: responseScene,
              current_scene: responseScene,
            }
          : previous
      );
      setSessions((previous) =>
        previous.map((session) =>
          session.id === selectedSession.id
            ? {
                ...session,
                scene_version: response.scene_version,
                scene_state: responseScene,
                current_scene: responseScene,
              }
            : session
        )
      );
    }

    try {
      await reloadSelectedSession(selectedSession.id);
      if (responseEvents.length > 0) {
        setEvents((previous) => {
          const byId = new Map(previous.map((event) => [event.id, event]));
          for (const event of responseEvents) {
            byId.set(event.id, event);
          }
          return [...byId.values()].sort((left, right) => left.sequence_number - right.sequence_number);
        });
      }
    } catch {
      // Keep response-derived state when the follow-up refresh is unavailable.
    }
  }, [reloadSelectedSession, selectedSession]);

  const handleTurnError = useCallback(async (turnError: unknown) => {
    const errorInfo = getVNPlayErrorInfo(turnError);
    const isConflict = isRecoverableVNPlayConflict(turnError);

    if (isConflict && selectedSession) {
      setTurnStatus(
        errorInfo.code === 'turn_in_progress' || /turn_in_progress/i.test(errorInfo.message)
          ? 'turn_in_progress'
          : 'stale_scene_version'
      );
      setError(null);
      try {
        await reloadSelectedSession(selectedSession.id);
      } catch {
        setError(errorInfo.message);
      }
      return;
    }

    setTurnStatus('turn_failed');
    setError(errorInfo.message);
  }, [reloadSelectedSession, selectedSession]);

  const selectedMode = selectedSession ? sessionModeLabel(selectedSession.mode) : null;
  const sceneState: VNPlaySceneState | null =
    selectedSession?.scene_state ?? selectedSession?.current_scene ?? null;
  const sceneVersion = sceneState?.scene_version ?? selectedSession?.scene_version ?? 0;
  const choices = (sceneState?.visible_choices ?? []).filter(isVNPlayChoice);
  const recoveryMessage = recoveryCopy(turnStatus);

  const handleCreateCheckpoint = useCallback(async (label: string) => {
    if (!selectedSession) return;

    setIsCreatingCheckpoint(true);
    setError(null);
    try {
      await createVNPlayCheckpoint(selectedSession.id, {
        label,
        scene_version: sceneVersion,
      });
      await reloadSelectedSession(selectedSession.id);
    } catch (checkpointError) {
      setError(checkpointError instanceof Error ? checkpointError.message : 'Failed to create checkpoint');
    } finally {
      setIsCreatingCheckpoint(false);
    }
  }, [reloadSelectedSession, sceneVersion, selectedSession]);

  const handleRestoreCheckpoint = useCallback(async (checkpointId: number) => {
    if (!selectedSession) return;

    setRestoringCheckpointId(checkpointId);
    setError(null);
    try {
      const restored = await restoreVNPlaySession(selectedSession.id, {
        checkpoint_id: checkpointId,
        idempotency_key: createVNPlayIdempotencyKey('restore'),
      });
      setSelectedSession(restored);
      setSessions((previous) =>
        previous.map((session) => (session.id === restored.id ? restored : session))
      );
      await reloadSessionCollections(selectedSession.id);
    } catch (restoreError) {
      setError(restoreError instanceof Error ? restoreError.message : 'Failed to restore checkpoint');
    } finally {
      setRestoringCheckpointId(null);
    }
  }, [reloadSessionCollections, selectedSession]);

  const handleRetryLastTurn = useCallback(async () => {
    if (!selectedSession) return;

    setIsRetryingTurn(true);
    setError(null);
    try {
      const response = await retryLastVNPlayTurn(selectedSession.id, {
        client_scene_version: sceneVersion,
        idempotency_key: createVNPlayIdempotencyKey('retry'),
      });
      await handleTurn(response);
    } catch (retryError) {
      setError(retryError instanceof Error ? retryError.message : 'Failed to retry last turn');
    } finally {
      setIsRetryingTurn(false);
    }
  }, [handleTurn, sceneVersion, selectedSession]);

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
        {recoveryMessage && (
          <div className="rounded-md border border-warn/30 bg-warn/10 px-3 py-2 text-sm text-warn">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <span>{recoveryMessage}</span>
              {turnStatus === 'turn_failed' && selectedSession && (
                <Button
                  loading={isRetryingTurn}
                  onClick={() => void handleRetryLastTurn()}
                  size="sm"
                  type="button"
                  variant="secondary"
                >
                  Retry last turn
                </Button>
              )}
            </div>
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
              {selectedSession && sceneState ? (
                <div className="grid gap-4">
                  <div>
                    <p className="text-xs uppercase tracking-normal text-text-muted">Selected session</p>
                    <p className="font-medium">Selected session: {selectedSession.title}</p>
                  </div>
                  <SceneStage events={events} sceneState={sceneState} showDialogue={false} />
                  <DialoguePanel
                    events={events}
                    mode={selectedSession.mode}
                    sceneVersion={sceneVersion}
                    sessionId={selectedSession.id}
                    onError={handleTurnError}
                    onTurn={(response) => void handleTurn(response)}
                  />
                  {selectedSession.mode === 'story' && (
                    <ChoicePanel
                      choices={choices}
                      sceneVersion={sceneVersion}
                      sessionId={selectedSession.id}
                      onError={handleTurnError}
                      onTurn={(response) => void handleTurn(response)}
                    />
                  )}
                </div>
              ) : (
                <p className="text-sm text-text-muted">No session selected.</p>
              )}
            </div>

            {selectedSession && sceneState ? (
              <SceneInspector
                branches={branches}
                checkpoints={checkpoints}
                isCreatingCheckpoint={isCreatingCheckpoint}
                restoringCheckpointId={restoringCheckpointId}
                sceneState={sceneState}
                session={selectedSession}
                onCreateCheckpoint={handleCreateCheckpoint}
                onRestoreCheckpoint={handleRestoreCheckpoint}
              />
            ) : (
              <aside className="rounded-md border border-border bg-surface p-4">
                <h2 className="mb-4 text-lg font-semibold">Runtime inspector</h2>
                <p className="text-sm text-text-muted">No session metadata.</p>
              </aside>
            )}
          </section>
        </section>
      </div>
    </main>
  );
}

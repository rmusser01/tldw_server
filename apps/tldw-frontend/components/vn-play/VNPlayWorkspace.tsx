import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { BookOpen, MessageSquarePlus, RefreshCw, RotateCcw } from 'lucide-react';
import { Badge } from '@web/components/ui/Badge';
import { Button } from '@web/components/ui/Button';
import ChoicePanel from '@web/components/vn-play/ChoicePanel';
import DialoguePanel from '@web/components/vn-play/DialoguePanel';
import NewSessionDialog from '@web/components/vn-play/NewSessionDialog';
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

type RecoveryKind = 'retry_available' | 'stale_scene_version' | 'turn_in_progress';

interface RecoveryState {
  detail?: string;
  kind: RecoveryKind;
  message: string;
  title: string;
}

const RECOVERABLE_TURN_STATUSES = new Set(['model_failed', 'parse_failed', 'abandoned', 'cancelled']);

function idempotencyKey(prefix: string): string {
  const uuid = globalThis.crypto?.randomUUID?.();
  return `${prefix}-${uuid ?? `${Date.now()}-${Math.random().toString(36).slice(2)}`}`;
}

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

function describeTurnError(error: unknown): { detail: string; status?: number } {
  const status = typeof error === 'object' && error !== null && 'status' in error
    ? Number((error as { status?: number }).status)
    : undefined;
  const detail = error instanceof Error ? error.message : String(error);
  return { detail, status };
}

export default function VNPlayWorkspace() {
  const [sessions, setSessions] = useState<VNPlaySession[]>([]);
  const [selectedSession, setSelectedSession] = useState<VNPlaySession | null>(null);
  const [events, setEvents] = useState<VNPlayEvent[]>([]);
  const [branches, setBranches] = useState<VNPlayBranch[]>([]);
  const [checkpoints, setCheckpoints] = useState<VNPlayCheckpoint[]>([]);
  const [checkpointLabel, setCheckpointLabel] = useState('');
  const [modeFilter, setModeFilter] = useState<VNPlayModeFilter>('all');
  const [dialogMode, setDialogMode] = useState<VNPlayMode>('freeform');
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [isCreating, setIsCreating] = useState(false);
  const [isCreatingCheckpoint, setIsCreatingCheckpoint] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [recoveryAction, setRecoveryAction] = useState<RecoveryKind | null>(null);
  const [recoveryState, setRecoveryState] = useState<RecoveryState | null>(null);
  const [restoringCheckpointId, setRestoringCheckpointId] = useState<number | null>(null);
  const [turnStatus, setTurnStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const selectedSessionId = selectedSession?.id ?? null;

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
    if (selectedSessionId === null) {
      setEvents([]);
      setBranches([]);
      setCheckpoints([]);
      return;
    }

    let cancelled = false;
    async function loadSessionMetadata() {
      try {
        const [nextEvents, nextBranches, nextCheckpoints] = await Promise.all([
          listVNPlayEvents(selectedSessionId),
          listVNPlayBranches(selectedSessionId),
          listVNPlayCheckpoints(selectedSessionId),
        ]);
        if (!cancelled) {
          setEvents(nextEvents);
          setBranches(nextBranches);
          setCheckpoints(nextCheckpoints);
        }
      } catch {
        if (!cancelled) {
          setEvents([]);
          setBranches([]);
          setCheckpoints([]);
        }
      }
    }

    void loadSessionMetadata();
    return () => {
      cancelled = true;
    };
  }, [selectedSessionId]);

  const filteredSessions = useMemo(() => {
    if (modeFilter === 'all') return sessions;
    return sessions.filter((session) => session.mode === modeFilter);
  }, [modeFilter, sessions]);

  const handleNewSession = useCallback((mode: VNPlayMode) => {
    setDialogMode(mode);
    setIsDialogOpen(true);
  }, []);

  const handleSelectSession = useCallback((session: VNPlaySession) => {
    setSelectedSession(session);
    setCheckpointLabel('');
    setRecoveryState(null);
    setTurnStatus(null);
  }, []);

  const handleCreateSession = useCallback(async (request: VNPlaySessionCreate) => {
    setIsCreating(true);
    setError(null);
    try {
      const created = await createVNPlaySession(request);
      setSessions((previous) => [created, ...previous.filter((session) => session.id !== created.id)]);
      setSelectedSession(created);
      setEvents([]);
      setBranches([]);
      setCheckpoints([]);
      setRecoveryState(null);
      setTurnStatus(null);
      setCheckpointLabel('');
      setModeFilter('all');
      setIsDialogOpen(false);
    } catch (createError) {
      setError(createError instanceof Error ? createError.message : 'Failed to create VN play session');
    } finally {
      setIsCreating(false);
    }
  }, []);

  const refreshSelectedSession = useCallback(async (sessionId: number) => {
    const [nextSession, nextEvents, nextBranches, nextCheckpoints] = await Promise.all([
      getVNPlaySession(sessionId),
      listVNPlayEvents(sessionId),
      listVNPlayBranches(sessionId),
      listVNPlayCheckpoints(sessionId),
    ]);
    setSelectedSession(nextSession);
    setSessions((previous) =>
      previous.map((session) => (session.id === nextSession.id ? nextSession : session))
    );
    setEvents(nextEvents);
    setBranches(nextBranches);
    setCheckpoints(nextCheckpoints);
    return nextSession;
  }, []);

  const selectedMode = selectedSession ? sessionModeLabel(selectedSession.mode) : null;
  const sceneState: VNPlaySceneState | null =
    selectedSession?.scene_state ?? selectedSession?.current_scene ?? null;
  const sceneVersion = sceneState?.scene_version ?? selectedSession?.scene_version ?? 0;
  const choices = (sceneState?.visible_choices ?? []).filter(isVNPlayChoice);

  const handleTurn = useCallback(async (response: VNPlayTurnResponse) => {
    if (!selectedSession) return;

    setTurnStatus(response.status);
    if (RECOVERABLE_TURN_STATUSES.has(response.status)) {
      setRecoveryState({
        detail: response.error_message ?? response.error_code ?? response.status,
        kind: 'retry_available',
        message: 'The last turn did not complete. Retry it with a fresh idempotency key.',
        title: 'Turn can be retried',
      });
    } else {
      setRecoveryState(null);
    }
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
      try {
        await refreshSelectedSession(response.session.id);
      } catch {
        // Keep response-derived state when the follow-up refresh is unavailable.
      }
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
      await refreshSelectedSession(selectedSession.id);
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
  }, [refreshSelectedSession, selectedSession]);

  const handleTurnError = useCallback(async (turnError: unknown) => {
    const { detail, status } = describeTurnError(turnError);
    const isConflict = status === 409 || /stale_scene_version|turn_in_progress/i.test(detail);

    if (isConflict && selectedSession) {
      const kind: RecoveryKind = /turn_in_progress/i.test(detail)
        ? 'turn_in_progress'
        : 'stale_scene_version';
      setTurnStatus(kind);
      setRecoveryState({
        detail,
        kind,
        message: kind === 'turn_in_progress'
          ? 'A turn is already running for this session. Poll the session before submitting another action.'
          : 'The scene changed on the server. Reload the session before resubmitting.',
        title: kind === 'turn_in_progress' ? 'Turn is still in progress' : 'Scene changed on the server',
      });
      try {
        await refreshSelectedSession(selectedSession.id);
      } catch {
        setError(detail);
      }
      return;
    }

    if (selectedSession) {
      setRecoveryState({
        detail,
        kind: 'retry_available',
        message: 'The last turn failed before completion. Retry it with a fresh idempotency key.',
        title: 'Turn can be retried',
      });
    }
    setError(detail);
  }, [refreshSelectedSession, selectedSession]);

  const handleCreateCheckpoint = useCallback(async () => {
    if (!selectedSession) return;
    const label = checkpointLabel.trim() || `Scene ${sceneVersion}`;
    setIsCreatingCheckpoint(true);
    setError(null);
    try {
      await createVNPlayCheckpoint(selectedSession.id, {
        label,
        scene_version: sceneVersion,
      });
      setCheckpointLabel('');
      await refreshSelectedSession(selectedSession.id);
    } catch (checkpointError) {
      setError(checkpointError instanceof Error ? checkpointError.message : 'Failed to create checkpoint');
    } finally {
      setIsCreatingCheckpoint(false);
    }
  }, [checkpointLabel, refreshSelectedSession, sceneVersion, selectedSession]);

  const handleRestoreCheckpoint = useCallback(async (checkpointId: number) => {
    if (!selectedSession) return;
    setRestoringCheckpointId(checkpointId);
    setError(null);
    try {
      await restoreVNPlaySession(selectedSession.id, {
        checkpoint_id: checkpointId,
        idempotency_key: idempotencyKey('restore'),
      });
      setRecoveryState(null);
      await refreshSelectedSession(selectedSession.id);
    } catch (restoreError) {
      setError(restoreError instanceof Error ? restoreError.message : 'Failed to restore checkpoint');
    } finally {
      setRestoringCheckpointId(null);
    }
  }, [refreshSelectedSession, selectedSession]);

  const handleRetryLastTurn = useCallback(async () => {
    if (!selectedSession) return;
    setRecoveryAction('retry_available');
    setError(null);
    try {
      const response = await retryLastVNPlayTurn(selectedSession.id, {
        client_scene_version: sceneVersion,
        idempotency_key: idempotencyKey('retry'),
      });
      await handleTurn(response);
    } catch (retryError) {
      setError(retryError instanceof Error ? retryError.message : 'Failed to retry turn');
    } finally {
      setRecoveryAction(null);
    }
  }, [handleTurn, sceneVersion, selectedSession]);

  const handleRecoveryRefresh = useCallback(async (kind: RecoveryKind) => {
    if (!selectedSession) return;
    setRecoveryAction(kind);
    setError(null);
    try {
      await refreshSelectedSession(selectedSession.id);
      if (kind !== 'retry_available') {
        setRecoveryState(null);
      }
    } catch (refreshError) {
      setError(refreshError instanceof Error ? refreshError.message : 'Failed to refresh session');
    } finally {
      setRecoveryAction(null);
    }
  }, [refreshSelectedSession, selectedSession]);

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
        {recoveryState && (
          <div className="grid gap-3 rounded-md border border-warn/30 bg-warn/10 px-3 py-3 text-sm text-warn">
            <div>
              <p className="font-medium">{recoveryState.title}</p>
              <p>{recoveryState.message}</p>
              {recoveryState.detail && (
                <p className="mt-1 text-xs opacity-80">{recoveryState.detail}</p>
              )}
            </div>
            <div className="flex flex-wrap gap-2">
              {recoveryState.kind === 'retry_available' ? (
                <Button
                  className="gap-2"
                  loading={recoveryAction === 'retry_available'}
                  onClick={() => void handleRetryLastTurn()}
                  type="button"
                  variant="secondary"
                >
                  <RotateCcw aria-hidden className="h-4 w-4" />
                  Retry last turn
                </Button>
              ) : (
                <Button
                  className="gap-2"
                  loading={recoveryAction === recoveryState.kind}
                  onClick={() => void handleRecoveryRefresh(recoveryState.kind)}
                  type="button"
                  variant="secondary"
                >
                  <RefreshCw aria-hidden className="h-4 w-4" />
                  {recoveryState.kind === 'turn_in_progress' ? 'Poll session' : 'Reload session'}
                </Button>
              )}
            </div>
          </div>
        )}
        {!recoveryState && turnStatus && turnStatus !== 'completed' && (
          <div className="rounded-md border border-warn/30 bg-warn/10 px-3 py-2 text-sm text-warn">
            {turnStatus}
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
            onSelectSession={handleSelectSession}
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
                checkpointLabel={checkpointLabel}
                checkpoints={checkpoints}
                isCreatingCheckpoint={isCreatingCheckpoint}
                restoringCheckpointId={restoringCheckpointId}
                sceneState={sceneState}
                session={selectedSession}
                onCheckpointLabelChange={setCheckpointLabel}
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

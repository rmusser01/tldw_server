import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { BookOpen, MessageSquarePlus, ScrollText } from 'lucide-react';
import Link from 'next/link';
import { Badge } from '@web/components/ui/Badge';
import { Button } from '@web/components/ui/Button';
import BranchTimelinePanel from '@web/components/vn-play/BranchTimelinePanel';
import ChoicePanel from '@web/components/vn-play/ChoicePanel';
import DialoguePanel from '@web/components/vn-play/DialoguePanel';
import GenerationInspector from '@web/components/vn-play/GenerationInspector';
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
  getVNPlayBranchNavigation,
  getVNPlaySession,
  activateVNPlayGenerationRevision,
  cancelVNPlayGenerationRequest,
  confirmVNPlayGenerationRequest,
  listVNPlayBranches,
  listVNPlayCheckpoints,
  listVNPlayEvents,
  listVNPlayGenerations,
  listVNPlaySessions,
  regenerateVNPlayGeneration,
  restoreVNPlayBranch,
  restoreVNPlaySession,
  retryLastVNPlayTurn,
} from '@web/lib/api/vnPlay';
import { isAdmin } from '@web/lib/authz';
import type {
  VNPlayBranch,
  VNPlayBranchNavigationResponse,
  VNPlayBranchRestoreTarget,
  VNPlayCheckpoint,
  VNPlayChoice,
  VNPlayEvent,
  VNPlayGenerationHistoryItem,
  VNPlayOffsetPagination,
  VNPlayMode,
  VNPlaySceneState,
  VNPlaySession,
  VNPlaySessionCreate,
  VNPlayTurnResponse,
} from '@web/types/vn-play';

const GENERATION_HISTORY_PAGE_SIZE = 25;

function sessionModeLabel(mode: VNPlayMode): string {
  if (mode === 'scripted_story') return 'Scripted Story';
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
  if (status === 'restore_action_in_progress') {
    return 'A branch restore is already in progress for this session. Wait for it to finish, then reload if needed.';
  }
  if (status === 'turn_failed') {
    return 'The last turn failed before completion. You can retry the stored turn request without duplicating the user input.';
  }
  return null;
}

function recoverableConflictStatus(errorInfo: ReturnType<typeof getVNPlayErrorInfo>): string {
  if (errorInfo.code === 'turn_in_progress' || /turn_in_progress/i.test(errorInfo.message)) {
    return 'turn_in_progress';
  }
  if (
    errorInfo.code === 'restore_action_in_progress' ||
    /restore_action_in_progress/i.test(errorInfo.message)
  ) {
    return 'restore_action_in_progress';
  }
  return 'stale_scene_version';
}

function mergeSessionScene(
  session: VNPlaySession,
  scene: VNPlaySceneState | null | undefined,
  sceneVersion?: number
): VNPlaySession {
  const nextSceneVersion = sceneVersion ?? scene?.scene_version ?? session.scene_version;
  if (!scene) {
    return {
      ...session,
      scene_version: nextSceneVersion,
    };
  }
  return {
    ...session,
    scene_version: nextSceneVersion,
    scene_state: scene,
    current_scene: scene,
  };
}

function canViewDebugForCurrentUser(): boolean {
  if (typeof window === 'undefined') return true;
  const hasJwtToken = Boolean(window.localStorage.getItem('access_token'));
  if (!hasJwtToken) return true;
  const userValue = window.localStorage.getItem('user');
  if (!userValue) return false;
  try {
    return isAdmin(JSON.parse(userValue));
  } catch {
    return false;
  }
}

type VNPlayWorkspaceProps = {
  initialSessionId?: number;
  generationInspectorRoute?: boolean;
};

export default function VNPlayWorkspace({
  generationInspectorRoute = false,
  initialSessionId,
}: VNPlayWorkspaceProps = {}) {
  const [sessions, setSessions] = useState<VNPlaySession[]>([]);
  const [selectedSession, setSelectedSession] = useState<VNPlaySession | null>(null);
  const [events, setEvents] = useState<VNPlayEvent[]>([]);
  const [checkpoints, setCheckpoints] = useState<VNPlayCheckpoint[]>([]);
  const [branches, setBranches] = useState<VNPlayBranch[]>([]);
  const [branchNavigation, setBranchNavigation] = useState<VNPlayBranchNavigationResponse | null>(null);
  const [branchTimelineError, setBranchTimelineError] = useState<string | null>(null);
  const [generations, setGenerations] = useState<VNPlayGenerationHistoryItem[]>([]);
  const [generationPagination, setGenerationPagination] = useState<VNPlayOffsetPagination | null>(null);
  const [modeFilter, setModeFilter] = useState<VNPlayModeFilter>('all');
  const [dialogMode, setDialogMode] = useState<VNPlayMode>('freeform');
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [isCreating, setIsCreating] = useState(false);
  const [isCreatingCheckpoint, setIsCreatingCheckpoint] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [isLoadingGenerations, setIsLoadingGenerations] = useState(false);
  const [isLoadingBranchNavigation, setIsLoadingBranchNavigation] = useState(false);
  const [isRetryingTurn, setIsRetryingTurn] = useState(false);
  const [restoringCheckpointId, setRestoringCheckpointId] = useState<number | null>(null);
  const [restoringBranchId, setRestoringBranchId] = useState<number | null>(null);
  const [restoringBranchTarget, setRestoringBranchTarget] = useState<VNPlayBranchRestoreTarget | null>(null);
  const [turnStatus, setTurnStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const selectedSessionId = selectedSession?.id;
  const selectedSessionIdRef = useRef<number | null>(selectedSessionId ?? null);
  const restoreOperationRef = useRef(0);
  const restoreInFlightRef = useRef(false);
  const canViewGenerationDebug = useMemo(() => canViewDebugForCurrentUser(), []);

  useEffect(() => {
    const nextSelectedSessionId = selectedSessionId ?? null;
    if (selectedSessionIdRef.current !== nextSelectedSessionId) {
      restoreOperationRef.current += 1;
      restoreInFlightRef.current = false;
      setRestoringBranchId(null);
      setRestoringBranchTarget(null);
    }
    selectedSessionIdRef.current = nextSelectedSessionId;
  }, [selectedSessionId]);

  const fetchBranchNavigation = useCallback(async (
    sessionId: number,
    mode?: VNPlayMode | null
  ): Promise<{ error: string | null; navigation: VNPlayBranchNavigationResponse | null }> => {
    if (mode !== 'story' && mode !== 'scripted_story') {
      return { error: null, navigation: null };
    }

    try {
      const nextBranchNavigation = await getVNPlayBranchNavigation(sessionId);
      return { error: null, navigation: nextBranchNavigation };
    } catch (branchError) {
      return {
        error: branchError instanceof Error ? branchError.message : 'Failed to load branch timeline',
        navigation: null,
      };
    }
  }, []);

  const reloadBranchNavigation = useCallback(async (
    sessionId: number,
    mode?: VNPlayMode | null
  ): Promise<VNPlayBranchNavigationResponse | null> => {
    setIsLoadingBranchNavigation(mode === 'story');
    try {
      const result = await fetchBranchNavigation(sessionId, mode);
      setBranchTimelineError(result.error);
      const nextBranchNavigation = result.navigation;
      setBranchNavigation(nextBranchNavigation);
      return nextBranchNavigation;
    } finally {
      setIsLoadingBranchNavigation(false);
    }
  }, [fetchBranchNavigation]);

  useEffect(() => {
    let cancelled = false;

    async function loadSessions() {
      setIsLoading(true);
      setError(null);
      try {
        const nextSessions = await listVNPlaySessions();
        if (cancelled) return;
        setSessions(nextSessions);
        setSelectedSession(
          initialSessionId
            ? nextSessions.find((session) => session.id === initialSessionId) ?? nextSessions[0] ?? null
            : nextSessions[0] ?? null
        );
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
  }, [initialSessionId]);

  useEffect(() => {
    if (!initialSessionId || sessions.length === 0) return;
    setSelectedSession((previous) => {
      if (previous?.id === initialSessionId) return previous;
      return sessions.find((session) => session.id === initialSessionId) ?? previous;
    });
  }, [initialSessionId, sessions]);

  useEffect(() => {
    if (!selectedSessionId) {
      setEvents([]);
      setCheckpoints([]);
      setBranches([]);
      setBranchNavigation(null);
      setBranchTimelineError(null);
      setIsLoadingBranchNavigation(false);
      setGenerations([]);
      setGenerationPagination(null);
      return;
    }

    let cancelled = false;
    async function loadSessionCollections() {
      setIsLoadingGenerations(true);
      setIsLoadingBranchNavigation(
        selectedSession?.mode === 'story' || selectedSession?.mode === 'scripted_story'
      );
      try {
        const [
          nextEvents,
          nextCheckpoints,
          nextBranches,
          nextGenerations,
          nextBranchNavigation,
        ] = await Promise.all([
          listVNPlayEvents(selectedSessionId),
          listVNPlayCheckpoints(selectedSessionId),
          listVNPlayBranches(selectedSessionId),
          listVNPlayGenerations(selectedSessionId, { limit: GENERATION_HISTORY_PAGE_SIZE, offset: 0 }),
          fetchBranchNavigation(selectedSessionId, selectedSession?.mode),
        ]);
        if (!cancelled) {
          setEvents(nextEvents);
          setCheckpoints(nextCheckpoints);
          setBranches(nextBranches);
          setBranchNavigation(nextBranchNavigation.navigation);
          setBranchTimelineError(nextBranchNavigation.error);
          setGenerations(nextGenerations.items ?? []);
          setGenerationPagination(nextGenerations.pagination ?? null);
        }
      } catch {
        if (!cancelled) {
          setEvents([]);
          setCheckpoints([]);
          setBranches([]);
          setBranchNavigation(null);
          setBranchTimelineError(null);
          setGenerations([]);
          setGenerationPagination(null);
        }
      } finally {
        if (!cancelled) {
          setIsLoadingGenerations(false);
          setIsLoadingBranchNavigation(false);
        }
      }
    }

    void loadSessionCollections();
    return () => {
      cancelled = true;
    };
  }, [fetchBranchNavigation, selectedSession?.mode, selectedSessionId]);

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
      setBranchNavigation(null);
      setBranchTimelineError(null);
      setGenerations([]);
      setGenerationPagination(null);
      setModeFilter('all');
      setIsDialogOpen(false);
    } catch (createError) {
      setError(createError instanceof Error ? createError.message : 'Failed to create VN play session');
    } finally {
      setIsCreating(false);
    }
  }, []);

  const reloadSessionCollections = useCallback(async (
    sessionId: number,
    mode?: VNPlayMode | null,
    nextBranchNavigation?: VNPlayBranchNavigationResponse | null
  ) => {
    const [nextEvents, nextCheckpoints, nextBranches, nextGenerations] = await Promise.all([
      listVNPlayEvents(sessionId),
      listVNPlayCheckpoints(sessionId),
      listVNPlayBranches(sessionId),
      listVNPlayGenerations(sessionId, { limit: GENERATION_HISTORY_PAGE_SIZE, offset: 0 }),
    ]);
    if (nextBranchNavigation !== undefined) {
      setBranchNavigation(nextBranchNavigation);
      setBranchTimelineError(null);
      setIsLoadingBranchNavigation(false);
    } else {
      await reloadBranchNavigation(sessionId, mode ?? selectedSession?.mode);
    }
    setEvents(nextEvents);
    setCheckpoints(nextCheckpoints);
    setBranches(nextBranches);
    setGenerations(nextGenerations.items ?? []);
    setGenerationPagination(nextGenerations.pagination ?? null);
  }, [reloadBranchNavigation, selectedSession?.mode]);

  const loadMoreGenerations = useCallback(async () => {
    if (!selectedSessionId || !generationPagination?.has_more) return;
    const offset = generationPagination.next_offset ?? generationPagination.offset + generationPagination.limit;
    setIsLoadingGenerations(true);
    try {
      const nextGenerations = await listVNPlayGenerations(selectedSessionId, {
        limit: GENERATION_HISTORY_PAGE_SIZE,
        offset,
      });
      setGenerations((previous) => [...previous, ...(nextGenerations.items ?? [])]);
      setGenerationPagination(nextGenerations.pagination ?? null);
    } finally {
      setIsLoadingGenerations(false);
    }
  }, [generationPagination, selectedSessionId]);

  const reloadSelectedSession = useCallback(async (sessionId: number) => {
    const nextSession = await getVNPlaySession(sessionId);
    setSelectedSession(nextSession);
    setSessions((previous) =>
      previous.map((session) => (session.id === nextSession.id ? nextSession : session))
    );
    await reloadSessionCollections(sessionId, nextSession.mode);
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
      try {
        await reloadSessionCollections(response.session.id, response.session.mode);
      } catch {
        // Keep response-derived session state when the follow-up collection refresh is unavailable.
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
  }, [reloadSelectedSession, reloadSessionCollections, selectedSession]);

  const handleTurnError = useCallback(async (turnError: unknown) => {
    const errorInfo = getVNPlayErrorInfo(turnError);
    const isConflict = isRecoverableVNPlayConflict(turnError);

    if (isConflict && selectedSession) {
      setTurnStatus(recoverableConflictStatus(errorInfo));
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
        client_scene_version: sceneVersion,
        idempotency_key: createVNPlayIdempotencyKey('restore'),
      });
      setSelectedSession(restored);
      setSessions((previous) =>
        previous.map((session) => (session.id === restored.id ? restored : session))
      );
      await reloadSessionCollections(selectedSession.id, restored.mode);
    } catch (restoreError) {
      setError(restoreError instanceof Error ? restoreError.message : 'Failed to restore checkpoint');
    } finally {
      setRestoringCheckpointId(null);
    }
  }, [reloadSessionCollections, sceneVersion, selectedSession]);

  const handleRestoreBranch = useCallback(async (
    branchId: number,
    target: VNPlayBranchRestoreTarget
  ) => {
    if (!selectedSession || restoringBranchId !== null || restoreInFlightRef.current) return;

    const sessionId = selectedSession.id;
    const operationId = restoreOperationRef.current + 1;
    restoreOperationRef.current = operationId;
    restoreInFlightRef.current = true;
    const isCurrentRestore = () =>
      restoreOperationRef.current === operationId && selectedSessionIdRef.current === sessionId;
    setRestoringBranchId(branchId);
    setRestoringBranchTarget(target);
    setError(null);
    try {
      const restored = await restoreVNPlayBranch(sessionId, branchId, {
        client_scene_version: sceneVersion,
        idempotency_key: createVNPlayIdempotencyKey('restore-branch'),
        target,
      });
      if (!isCurrentRestore()) return;
      const nextSession = mergeSessionScene(restored.session, restored.current_scene, restored.scene_version);
      setTurnStatus(restored.status);
      setSelectedSession(nextSession);
      setSessions((previous) =>
        previous.map((session) => (session.id === nextSession.id ? nextSession : session))
      );
      setBranchNavigation(restored.branch_navigation);
      await reloadSessionCollections(nextSession.id, nextSession.mode, restored.branch_navigation);
    } catch (restoreError) {
      const errorInfo = getVNPlayErrorInfo(restoreError);
      if (isRecoverableVNPlayConflict(restoreError)) {
        if (!isCurrentRestore()) return;
        setTurnStatus(recoverableConflictStatus(errorInfo));
        setError(null);
        try {
          await reloadSelectedSession(sessionId);
          if (!isCurrentRestore()) return;
        } catch {
          if (!isCurrentRestore()) return;
          setError(errorInfo.message);
        }
        return;
      }
      if (!isCurrentRestore()) return;
      setError(errorInfo.message);
    } finally {
      if (isCurrentRestore()) {
        restoreInFlightRef.current = false;
        setRestoringBranchId(null);
        setRestoringBranchTarget(null);
      }
    }
  }, [
    reloadSelectedSession,
    reloadSessionCollections,
    restoringBranchId,
    sceneVersion,
    selectedSession,
  ]);

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

  const handleGenerationAction = useCallback(async (
    action: () => Promise<VNPlayTurnResponse>
  ) => {
    if (!selectedSession) return;
    setError(null);
    try {
      const response = await action();
      await handleTurn(response);
    } catch (actionError) {
      const errorInfo = getVNPlayErrorInfo(actionError);
      setError(errorInfo.message);
      if (isRecoverableVNPlayConflict(actionError)) {
        try {
          await reloadSelectedSession(selectedSession.id);
        } catch {
          // Preserve the original action error if refresh also fails.
        }
      }
    }
  }, [handleTurn, reloadSelectedSession, selectedSession]);

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
              <Button
                className="gap-2"
                onClick={() => handleNewSession('scripted_story')}
                type="button"
                variant="secondary"
              >
                <ScrollText aria-hidden className="h-4 w-4" />
                New Scripted Story
              </Button>
              <Link
                className="inline-flex items-center rounded-md border border-border px-3 py-2 text-sm font-medium text-text hover:bg-surface"
                href="/vn-scripts"
              >
                VN scripts
              </Link>
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
            <div className="grid gap-4">
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
                    {(selectedSession.mode === 'story' || selectedSession.mode === 'scripted_story') && (
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

              {(selectedSession?.mode === 'story' || selectedSession?.mode === 'scripted_story') && (
                <>
                  {branchTimelineError && (
                    <div className="rounded-md border border-warn/30 bg-warn/10 px-3 py-2 text-sm text-warn">
                      {branchTimelineError}
                    </div>
                  )}
                  <BranchTimelinePanel
                    isLoading={isLoadingBranchNavigation}
                    navigation={branchNavigation}
                    onRestoreBranch={handleRestoreBranch}
                    restoringBranchId={restoringBranchId}
                    restoreTarget={restoringBranchTarget}
                  />
                </>
              )}
            </div>

            {selectedSession && sceneState ? (
              <>
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
                <aside className="rounded-md border border-border bg-surface p-4 xl:col-start-2">
                  {!generationInspectorRoute && (
                    <Link
                      className="mb-3 inline-flex text-xs font-medium text-primary hover:text-primaryStrong"
                      href={`/vn-play/sessions/${selectedSession.id}/generations`}
                    >
                      Open generation inspector
                    </Link>
                  )}
                  <GenerationInspector
                    canViewDebug={canViewGenerationDebug}
                    generations={generations}
                    hasMore={Boolean(generationPagination?.has_more)}
                    isLoading={isLoadingGenerations}
                    sceneState={sceneState}
                    sessionId={selectedSession.id}
                    onLoadMore={loadMoreGenerations}
                    onActivateRevision={(item) =>
                      handleGenerationAction(() =>
                        activateVNPlayGenerationRevision(selectedSession.id, item.generation_id, item.id, {
                          client_scene_version: sceneVersion,
                          idempotency_key: createVNPlayIdempotencyKey('generation-activate'),
                        })
                      )
                    }
                    onCancelRequest={(generationRequestId) =>
                      handleGenerationAction(() =>
                        cancelVNPlayGenerationRequest(selectedSession.id, generationRequestId, {
                          client_scene_version: sceneVersion,
                          idempotency_key: createVNPlayIdempotencyKey('generation-cancel'),
                        })
                      )
                    }
                    onConfirmRequest={(generationRequestId) =>
                      handleGenerationAction(() =>
                        confirmVNPlayGenerationRequest(selectedSession.id, generationRequestId, {
                          client_scene_version: sceneVersion,
                          idempotency_key: createVNPlayIdempotencyKey('generation-confirm'),
                        })
                      )
                    }
                    onRegenerate={(item) =>
                      handleGenerationAction(() =>
                        regenerateVNPlayGeneration(selectedSession.id, item.generation_id, {
                          client_scene_version: sceneVersion,
                          idempotency_key: createVNPlayIdempotencyKey('generation-regenerate'),
                        })
                      )
                    }
                  />
                </aside>
              </>
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

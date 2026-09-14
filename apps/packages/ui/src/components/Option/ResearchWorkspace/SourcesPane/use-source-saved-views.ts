import React from "react";
import { tldwClient } from "@/services/tldw/TldwApiClient";
import type {
  WorkspaceSourceSavedViewConflictDetail,
  WorkspaceSourceSavedViewResponse,
  WorkspaceSourceSavedViewValidResponse,
} from "@/services/tldw/domains/workspace-api";
import type { WorkspaceSourceSavedViewStateV1 } from "@/types/workspace-source-saved-view";
import {
  DEFAULT_SOURCE_LIST_VIEW_STATE,
  type SourceListViewState,
} from "./source-list-view";
import {
  SOURCE_SAVED_VIEW_SCHEMA_VERSION,
  applySavedSourceViewState,
  deserializeSourceViewState,
  getSourceListViewStateSignature,
  getSourceSavedViewNameLength,
  getSourceViewStateSignature,
  serializeSourceListViewState,
  type SourceViewStateValidationIssue,
} from "./source-saved-views";

export interface SourceSavedViewRequestError {
  message: string;
  retryable: true;
}

export interface SourceSavedViewDuplicateConflict {
  viewId: string;
  version: number;
  name: string;
  state: WorkspaceSourceSavedViewStateV1;
}

export interface SourceSavedViewLimitState {
  limit: number;
  retryable: false;
  guidance: string;
}

export interface SourceSavedViewVersionConflict {
  viewId: string;
  currentVersion: number;
  retryable: true;
}

export type SourceSavedViewMutation = "create" | "replace" | "reset" | "delete";

interface ControllerState {
  generation: number;
  views: WorkspaceSourceSavedViewResponse[];
  loading: boolean;
  listError: SourceSavedViewRequestError | null;
  activeViewId: string | null;
  activeSnapshot: WorkspaceSourceSavedViewStateV1 | null;
  activeSignature: string | null;
  serializationIssues: SourceViewStateValidationIssue[];
  duplicateConflict: SourceSavedViewDuplicateConflict | null;
  limitState: SourceSavedViewLimitState | null;
  versionConflict: SourceSavedViewVersionConflict | null;
  mutation: SourceSavedViewMutation | null;
  mutationError: SourceSavedViewRequestError | null;
  announcement: string | null;
}

interface OperationToken {
  generation: number;
  lifecycle: number;
  epoch: number;
}

type WithoutVersion<T> = T extends unknown ? Omit<T, "version"> : never;

interface PatchOptions {
  kind: "replace" | "reset";
  viewId: string;
  version: number;
  body: WithoutVersion<
    Parameters<typeof tldwClient.updateWorkspaceSourceView>[2]
  >;
  onSuccess: (
    view: WorkspaceSourceSavedViewValidResponse,
    token: OperationToken,
  ) => boolean;
}

interface VersionRetry extends OperationToken {
  run: (version: number) => Promise<void>;
}

interface MutationRetry extends OperationToken {
  run: () => Promise<void>;
}

const emptyState = (generation: number): ControllerState => ({
  generation,
  views: [],
  loading: false,
  listError: null,
  activeViewId: null,
  activeSnapshot: null,
  activeSignature: null,
  serializationIssues: [],
  duplicateConflict: null,
  limitState: null,
  versionConflict: null,
  mutation: null,
  mutationError: null,
  announcement: null,
});

const clearTransientState = (state: ControllerState): ControllerState => ({
  ...state,
  listError: null,
  serializationIssues: [],
  duplicateConflict: null,
  limitState: null,
  versionConflict: null,
  mutationError: null,
});

const sameValidationIssues = (
  left: SourceViewStateValidationIssue[],
  right: SourceViewStateValidationIssue[],
): boolean =>
  left.length === right.length &&
  left.every(
    (issue, index) =>
      issue.field === right[index]?.field &&
      issue.message === right[index]?.message,
  );

const isRecord = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === "object" && !Array.isArray(value);

const hasStatus = (error: unknown, status: number): boolean =>
  isRecord(error) && error.status === status;

const isPositiveInteger = (value: unknown): value is number =>
  Number.isSafeInteger(value) && Number(value) > 0;

const parseConflictDetail = (
  error: unknown,
): WorkspaceSourceSavedViewConflictDetail | null => {
  if (!isRecord(error) || error.status !== 409 || !isRecord(error.details)) {
    return null;
  }
  const detail = error.details.detail;
  if (!isRecord(detail) || typeof detail.code !== "string") return null;

  if (
    detail.code === "source_view_name_exists" &&
    typeof detail.view_id === "string" &&
    detail.view_id.trim().length > 0 &&
    !detail.view_id.includes("/") &&
    isPositiveInteger(detail.version)
  ) {
    return {
      code: detail.code,
      view_id: detail.view_id,
      version: detail.version,
    };
  }
  if (detail.code === "source_view_limit_reached" && detail.limit === 100) {
    return { code: detail.code, limit: detail.limit };
  }
  if (
    detail.code === "source_view_version_conflict" &&
    typeof detail.view_id === "string" &&
    detail.view_id.trim().length > 0 &&
    !detail.view_id.includes("/") &&
    isPositiveInteger(detail.current_version)
  ) {
    return {
      code: detail.code,
      view_id: detail.view_id,
      current_version: detail.current_version,
    };
  }
  return null;
};

const requestError = (error: unknown): SourceSavedViewRequestError => ({
  message:
    error instanceof Error && error.message.trim()
      ? error.message
      : "Saved views request failed.",
  retryable: true,
});

const validResponse = (
  response: WorkspaceSourceSavedViewResponse,
): response is WorkspaceSourceSavedViewValidResponse =>
  response.valid && deserializeSourceViewState(response.state) !== null;

const upsertView = (
  views: WorkspaceSourceSavedViewResponse[],
  view: WorkspaceSourceSavedViewResponse,
): WorkspaceSourceSavedViewResponse[] => [
  view,
  ...views.filter((candidate) => candidate.id !== view.id),
];

const removeView = (
  state: ControllerState,
  viewId: string,
  announcement: string,
): ControllerState => ({
  ...clearTransientState(state),
  views: state.views.filter((candidate) => candidate.id !== viewId),
  activeViewId: state.activeViewId === viewId ? null : state.activeViewId,
  activeSnapshot:
    state.activeViewId === viewId ? null : state.activeSnapshot,
  activeSignature:
    state.activeViewId === viewId ? null : state.activeSignature,
  mutation: null,
  announcement,
});

const defaultWireState = (): WorkspaceSourceSavedViewStateV1 => {
  const serialized = serializeSourceListViewState(
    DEFAULT_SOURCE_LIST_VIEW_STATE,
  );
  if (!serialized.ok) throw new Error("Default source view state is invalid");
  return serialized.state;
};

export const useSourceSavedViews = (
  workspaceId: string | null,
  workspaceExists: boolean,
  currentState: SourceListViewState,
  onApplyState: (state: SourceListViewState) => void,
) => {
  const identityRef = React.useRef(workspaceId);
  const generationRef = React.useRef(0);
  const mountedRef = React.useRef(false);
  const lifecycleRef = React.useRef(0);
  const operationEpochRef = React.useRef(0);
  const mutationInFlightRef = React.useRef(false);
  const hasAuthoritativeViewsRef = React.useRef(false);
  const identityPending = identityRef.current !== workspaceId;
  const renderGeneration =
    generationRef.current + (identityPending ? 1 : 0);

  const [state, setState] = React.useState<ControllerState>(() =>
    emptyState(renderGeneration),
  );
  const versionRetryRef = React.useRef<VersionRetry | null>(null);
  const mutationRetryRef = React.useRef<MutationRetry | null>(null);

  const isGenerationCurrent = React.useCallback(
    (generation: number) =>
      mountedRef.current && generationRef.current === generation,
    [],
  );

  const isOperationCurrent = React.useCallback(
    (token: OperationToken) =>
      mountedRef.current &&
      generationRef.current === token.generation &&
      lifecycleRef.current === token.lifecycle &&
      operationEpochRef.current === token.epoch,
    [],
  );

  const beginOperation = React.useCallback(
    (generation: number): OperationToken | null => {
      if (!isGenerationCurrent(generation)) return null;
      operationEpochRef.current += 1;
      versionRetryRef.current = null;
      mutationRetryRef.current = null;
      return {
        generation,
        lifecycle: lifecycleRef.current,
        epoch: operationEpochRef.current,
      };
    },
    [isGenerationCurrent],
  );

  const beginMutation = React.useCallback(
    (generation: number): OperationToken | null => {
      if (mutationInFlightRef.current) return null;
      const token = beginOperation(generation);
      if (token) mutationInFlightRef.current = true;
      return token;
    },
    [beginOperation],
  );

  const invalidateMutationRetries = React.useCallback(() => {
    operationEpochRef.current += 1;
    versionRetryRef.current = null;
    mutationRetryRef.current = null;
  }, []);

  const commitGeneration = React.useCallback(
    (
      generation: number,
      update: (current: ControllerState) => ControllerState,
    ) => {
      if (!isGenerationCurrent(generation)) return false;
      setState((current) => {
        if (!isGenerationCurrent(generation)) return current;
        return update(
          current.generation === generation ? current : emptyState(generation),
        );
      });
      return true;
    },
    [isGenerationCurrent],
  );

  const commitOperation = React.useCallback(
    (
      token: OperationToken,
      update: (current: ControllerState) => ControllerState,
    ) => {
      if (!isOperationCurrent(token)) return false;
      setState((current) => {
        if (!isOperationCurrent(token)) return current;
        return update(
          current.generation === token.generation
            ? current
            : emptyState(token.generation),
        );
      });
      return true;
    },
    [isOperationCurrent],
  );

  const load = React.useCallback(
    async (
      generation: number,
      targetWorkspaceId: string,
      prepare?: (current: ControllerState) => ControllerState,
    ): Promise<OperationToken | null> => {
      if (!workspaceExists) return null;
      const token = beginOperation(generation);
      if (!token) return null;
      commitOperation(token, (current) => {
        const loadingState = {
          ...current,
          loading: true,
          listError: null,
        };
        return prepare ? prepare(loadingState) : loadingState;
      });
      try {
        const response =
          await tldwClient.listWorkspaceSourceViews(targetWorkspaceId);
        if (!isOperationCurrent(token)) return null;
        hasAuthoritativeViewsRef.current = true;
        commitOperation(token, (current) => ({
          ...current,
          views: response.items,
          loading: false,
          listError: null,
        }));
        return token;
      } catch (error) {
        if (!isOperationCurrent(token)) return null;
        commitOperation(token, (current) => ({
          ...current,
          loading: false,
          listError: requestError(error),
        }));
        return token;
      }
    },
    [beginOperation, commitOperation, isOperationCurrent, workspaceExists],
  );

  const reconcileOperation = React.useCallback(
    async (token: OperationToken, targetWorkspaceId: string) => {
      if (!workspaceExists) return;
      if (!isOperationCurrent(token)) return;
      commitOperation(token, (current) => ({
        ...current,
        loading: true,
        listError: null,
      }));
      try {
        const response =
          await tldwClient.listWorkspaceSourceViews(targetWorkspaceId);
        if (!isOperationCurrent(token)) return;
        hasAuthoritativeViewsRef.current = true;
        commitOperation(token, (current) => ({
          ...current,
          views: response.items,
          loading: false,
          listError: null,
        }));
      } catch (error) {
        commitOperation(token, (current) => ({
          ...current,
          loading: false,
          listError: requestError(error),
        }));
      }
    },
    [commitOperation, isOperationCurrent, workspaceExists],
  );

  React.useLayoutEffect(() => {
    mountedRef.current = true;
    lifecycleRef.current += 1;
    return () => {
      mountedRef.current = false;
      lifecycleRef.current += 1;
      operationEpochRef.current += 1;
      mutationInFlightRef.current = false;
      hasAuthoritativeViewsRef.current = false;
      versionRetryRef.current = null;
      mutationRetryRef.current = null;
    };
  }, []);

  React.useLayoutEffect(() => {
    if (identityRef.current !== workspaceId) {
      identityRef.current = workspaceId;
      generationRef.current += 1;
    }
    operationEpochRef.current += 1;
    mutationInFlightRef.current = false;
    hasAuthoritativeViewsRef.current = false;
    versionRetryRef.current = null;
    mutationRetryRef.current = null;
    setState(emptyState(generationRef.current));
  }, [workspaceId]);

  React.useEffect(() => {
    if (
      workspaceId === null ||
      !workspaceExists ||
      identityRef.current !== workspaceId ||
      !mountedRef.current
    ) {
      return;
    }
    void load(generationRef.current, workspaceId);
  }, [load, workspaceExists, workspaceId]);

  const exposed =
    !identityPending && state.generation === renderGeneration
      ? state
      : emptyState(renderGeneration);

  const serializedCurrentState = React.useMemo(
    () => serializeSourceListViewState(currentState),
    [currentState],
  );

  React.useLayoutEffect(() => {
    if (exposed.serializationIssues.length === 0) return;
    const nextIssues = [
      ...exposed.serializationIssues.filter((issue) => issue.field === "name"),
      ...(serializedCurrentState.ok === false
        ? serializedCurrentState.issues
        : []),
    ];
    if (sameValidationIssues(exposed.serializationIssues, nextIssues)) return;
    commitGeneration(renderGeneration, (current) => ({
      ...current,
      serializationIssues: nextIssues,
    }));
  }, [
    commitGeneration,
    exposed.serializationIssues,
    renderGeneration,
    serializedCurrentState,
  ]);

  const retry = React.useCallback(async () => {
    if (
      workspaceId === null ||
      !workspaceExists ||
      identityRef.current !== workspaceId ||
      mutationInFlightRef.current ||
      !isGenerationCurrent(renderGeneration)
    ) {
      return;
    }
    await load(renderGeneration, workspaceId);
  }, [
    isGenerationCurrent,
    load,
    renderGeneration,
    workspaceExists,
    workspaceId,
  ]);

  const applyView = React.useCallback(
    (view: WorkspaceSourceSavedViewResponse) => {
      if (
        !workspaceExists ||
        !view.valid ||
        !isGenerationCurrent(renderGeneration)
      )
        return;
      const canonical = deserializeSourceViewState(view.state);
      if (canonical === null) return;
      const signature = getSourceViewStateSignature(canonical);
      if (signature === null) return;
      if (
        commitGeneration(renderGeneration, (current) => ({
          ...clearTransientState(current),
          activeViewId: view.id,
          activeSnapshot: canonical,
          activeSignature: signature,
        }))
      ) {
        versionRetryRef.current = null;
        mutationRetryRef.current = null;
        onApplyState(applySavedSourceViewState(currentState, canonical));
      }
    },
    [
      commitGeneration,
      currentState,
      isGenerationCurrent,
      onApplyState,
      renderGeneration,
      workspaceExists,
    ],
  );

  const finishValidMutation = React.useCallback(
    (
      token: OperationToken,
      view: WorkspaceSourceSavedViewValidResponse,
      announcement: string,
      activeState: WorkspaceSourceSavedViewStateV1 = view.state,
    ) => {
      const signature = getSourceViewStateSignature(activeState);
      if (signature === null) return false;
      if (!isOperationCurrent(token)) return false;
      versionRetryRef.current = null;
      mutationRetryRef.current = null;
      return commitOperation(token, (current) => ({
        ...clearTransientState(current),
        views: upsertView(current.views, view),
        loading: false,
        activeViewId: view.id,
        activeSnapshot: activeState,
        activeSignature: signature,
        mutation: null,
        announcement,
      }));
    },
    [commitOperation, isOperationCurrent],
  );

  const performPatch = React.useCallback(
    async (
      generation: number,
      targetWorkspaceId: string,
      options: PatchOptions,
    ): Promise<void> => {
      if (!workspaceExists) return;
      const token = beginMutation(generation);
      if (!token) return;
      const needsReconciliation = !hasAuthoritativeViewsRef.current;
      commitOperation(token, (current) => ({
        ...current,
        loading: false,
        mutation: options.kind,
        mutationError: null,
        limitState: null,
        versionConflict: null,
        announcement: null,
      }));
      try {
        const response = await tldwClient.updateWorkspaceSourceView(
          targetWorkspaceId,
          options.viewId,
          { version: options.version, ...options.body },
        );
        if (!isOperationCurrent(token)) return;
        mutationInFlightRef.current = false;
        if (!validResponse(response)) {
          throw new Error("The server returned an invalid saved view.");
        }
        const committed = options.onSuccess(response, token);
        if (committed && needsReconciliation) {
          void reconcileOperation(token, targetWorkspaceId);
        }
      } catch (error) {
        if (!isOperationCurrent(token)) return;
        mutationInFlightRef.current = false;
        if (hasStatus(error, 404)) {
          versionRetryRef.current = null;
          mutationRetryRef.current = null;
          const committed = commitOperation(token, (current) =>
            removeView(current, options.viewId, "Saved view no longer exists."),
          );
          if (committed) {
            await reconcileOperation(token, targetWorkspaceId);
          }
          return;
        }
        const detail = parseConflictDetail(error);
        if (detail?.code === "source_view_version_conflict") {
          const refreshToken = await load(
            generation,
            targetWorkspaceId,
            (current) => ({
              ...current,
              duplicateConflict:
                current.duplicateConflict?.viewId === detail.view_id
                  ? {
                      ...current.duplicateConflict,
                      version: detail.current_version,
                    }
                  : current.duplicateConflict,
              versionConflict: {
                viewId: detail.view_id,
                currentVersion: detail.current_version,
                retryable: true,
              },
              mutation: null,
              mutationError: null,
            }),
          );
          if (refreshToken && isOperationCurrent(refreshToken)) {
            versionRetryRef.current = {
              ...refreshToken,
              run: (version) =>
                performPatch(generation, targetWorkspaceId, {
                  ...options,
                  version,
                }),
            };
            commitOperation(refreshToken, (current) => ({ ...current }));
          }
          return;
        }
        mutationRetryRef.current = {
          ...token,
          run: () => performPatch(generation, targetWorkspaceId, options),
        };
        commitOperation(token, (current) => ({
          ...current,
          mutation: null,
          mutationError: requestError(error),
        }));
      }
    },
    [
      beginMutation,
      commitOperation,
      isOperationCurrent,
      load,
      reconcileOperation,
      workspaceExists,
    ],
  );

  const createView = React.useCallback(
    async (rawName: string) => {
      if (
        workspaceId === null ||
        !workspaceExists ||
        !isGenerationCurrent(renderGeneration)
      ) {
        return;
      }
      const name = rawName.trim();
      const issues: SourceViewStateValidationIssue[] = [];
      if (!name || getSourceSavedViewNameLength(name) > 120) {
        issues.push({
          field: "name",
          message: "Name must contain between 1 and 120 characters.",
        });
      }
      const serialized = serializeSourceListViewState(currentState);
      if (serialized.ok === false) issues.push(...serialized.issues);
      if (issues.length > 0 || serialized.ok === false) {
        commitGeneration(renderGeneration, (current) => ({
          ...current,
          serializationIssues: issues,
          mutation: null,
          mutationError: null,
        }));
        return;
      }

      const run = async () => {
        const token = beginMutation(renderGeneration);
        if (!token) return;
        const needsReconciliation = !hasAuthoritativeViewsRef.current;
        commitOperation(token, (current) => ({
          ...current,
          loading: false,
          serializationIssues: [],
          duplicateConflict: null,
          limitState: null,
          versionConflict: null,
          mutation: "create",
          mutationError: null,
          announcement: null,
        }));
        try {
          const response = await tldwClient.createWorkspaceSourceView(
            workspaceId,
            {
              name,
              schema_version: SOURCE_SAVED_VIEW_SCHEMA_VERSION,
              state: serialized.state,
            },
          );
          if (!isOperationCurrent(token)) return;
          mutationInFlightRef.current = false;
          if (!validResponse(response)) {
            throw new Error("The server returned an invalid saved view.");
          }
          const committed = finishValidMutation(
            token,
            response,
            "Saved view created.",
            serialized.state,
          );
          if (committed && needsReconciliation) {
            void reconcileOperation(token, workspaceId);
          }
        } catch (error) {
          if (!isOperationCurrent(token)) return;
          mutationInFlightRef.current = false;
          const detail = parseConflictDetail(error);
          if (detail?.code === "source_view_name_exists") {
            mutationRetryRef.current = null;
            commitOperation(token, (current) => ({
              ...current,
              duplicateConflict: {
                viewId: detail.view_id,
                version: detail.version,
                name,
                state: serialized.state,
              },
              mutation: null,
              mutationError: null,
            }));
            return;
          }
          if (detail?.code === "source_view_limit_reached") {
            mutationRetryRef.current = null;
            commitOperation(token, (current) => ({
              ...current,
              limitState: {
                limit: detail.limit,
                retryable: false,
                guidance:
                  "Delete an existing saved view before creating another.",
              },
              mutation: null,
              mutationError: null,
            }));
            return;
          }
          mutationRetryRef.current = { ...token, run };
          commitOperation(token, (current) => ({
            ...current,
            mutation: null,
            mutationError: requestError(error),
          }));
        }
      };
      await run();
    },
    [
      beginMutation,
      commitGeneration,
      commitOperation,
      currentState,
      finishValidMutation,
      isGenerationCurrent,
      isOperationCurrent,
      reconcileOperation,
      renderGeneration,
      workspaceExists,
      workspaceId,
    ],
  );

  const confirmReplace = React.useCallback(async () => {
    const conflict = exposed.duplicateConflict;
    if (
      workspaceId === null ||
      !workspaceExists ||
      conflict === null ||
      !isGenerationCurrent(renderGeneration)
    ) {
      return;
    }
    await performPatch(renderGeneration, workspaceId, {
      kind: "replace",
      viewId: conflict.viewId,
      version: conflict.version,
      body: {
        name: conflict.name,
        schema_version: SOURCE_SAVED_VIEW_SCHEMA_VERSION,
        state: conflict.state,
      },
      onSuccess: (view, token) =>
        finishValidMutation(
          token,
          view,
          "Saved view replaced.",
          conflict.state,
        ),
    });
  }, [
    exposed.duplicateConflict,
    finishValidMutation,
    isGenerationCurrent,
    performPatch,
    renderGeneration,
    workspaceExists,
    workspaceId,
  ]);

  const dismissDuplicateConflict = React.useCallback(() => {
    if (mutationInFlightRef.current || !isGenerationCurrent(renderGeneration))
      return;
    invalidateMutationRetries();
    commitGeneration(renderGeneration, (current) => ({
      ...current,
      duplicateConflict: null,
      versionConflict: null,
      mutationError: null,
    }));
  }, [
    commitGeneration,
    invalidateMutationRetries,
    isGenerationCurrent,
    renderGeneration,
  ]);

  const dismissMutationFailure = React.useCallback(() => {
    if (mutationInFlightRef.current || !isGenerationCurrent(renderGeneration))
      return;
    invalidateMutationRetries();
    commitGeneration(renderGeneration, (current) => ({
      ...current,
      versionConflict: null,
      mutationError: null,
    }));
  }, [
    commitGeneration,
    invalidateMutationRetries,
    isGenerationCurrent,
    renderGeneration,
  ]);

  const replaceView = React.useCallback(
    async (view: WorkspaceSourceSavedViewResponse) => {
      if (
        workspaceId === null ||
        !workspaceExists ||
        !view.valid ||
        !isGenerationCurrent(renderGeneration)
      ) {
        return;
      }
      const serialized = serializeSourceListViewState(currentState);
      if (serialized.ok === false) {
        commitGeneration(renderGeneration, (current) => ({
          ...current,
          serializationIssues: serialized.issues,
        }));
        return;
      }
      await performPatch(renderGeneration, workspaceId, {
        kind: "replace",
        viewId: view.id,
        version: view.version,
        body: {
          schema_version: SOURCE_SAVED_VIEW_SCHEMA_VERSION,
          state: serialized.state,
        },
        onSuccess: (response, token) =>
          finishValidMutation(
            token,
            response,
            "Saved view replaced.",
            serialized.state,
          ),
      });
    },
    [
      commitGeneration,
      currentState,
      finishValidMutation,
      isGenerationCurrent,
      performPatch,
      renderGeneration,
      workspaceExists,
      workspaceId,
    ],
  );

  const resetView = React.useCallback(
    async (view: WorkspaceSourceSavedViewResponse) => {
      if (
        workspaceId === null ||
        !workspaceExists ||
        !isGenerationCurrent(renderGeneration)
      )
        return;
      const wireState = defaultWireState();
      await performPatch(renderGeneration, workspaceId, {
        kind: "reset",
        viewId: view.id,
        version: view.version,
        body: {
          schema_version: SOURCE_SAVED_VIEW_SCHEMA_VERSION,
          state: wireState,
        },
        onSuccess: (response, token) => {
          const committed = finishValidMutation(
            token,
            response,
            "Saved view reset.",
            wireState,
          );
          if (committed) {
            onApplyState(applySavedSourceViewState(currentState, wireState));
          }
          return committed;
        },
      });
    },
    [
      currentState,
      finishValidMutation,
      isGenerationCurrent,
      onApplyState,
      performPatch,
      renderGeneration,
      workspaceExists,
      workspaceId,
    ],
  );

  const deleteView = React.useCallback(
    async (view: WorkspaceSourceSavedViewResponse) => {
      if (
        workspaceId === null ||
        !workspaceExists ||
        !isGenerationCurrent(renderGeneration)
      )
        return;
      const run = async () => {
        const token = beginMutation(renderGeneration);
        if (!token) return;
        const needsReconciliation = !hasAuthoritativeViewsRef.current;
        const finishDelete = () => {
          if (!isOperationCurrent(token)) return false;
          mutationInFlightRef.current = false;
          versionRetryRef.current = null;
          mutationRetryRef.current = null;
          return commitOperation(token, (current) =>
            removeView(current, view.id, "Saved view deleted."),
          );
        };
        commitOperation(token, (current) => ({
          ...current,
          loading: false,
          mutation: "delete",
          mutationError: null,
          announcement: null,
        }));
        try {
          await tldwClient.deleteWorkspaceSourceView(workspaceId, view.id);
          const committed = finishDelete();
          if (committed && needsReconciliation) {
            void reconcileOperation(token, workspaceId);
          }
        } catch (error) {
          if (!isOperationCurrent(token)) return;
          if (hasStatus(error, 404)) {
            const committed = finishDelete();
            if (committed && needsReconciliation) {
              void reconcileOperation(token, workspaceId);
            }
            return;
          }
          mutationInFlightRef.current = false;
          mutationRetryRef.current = { ...token, run };
          commitOperation(token, (current) => ({
            ...current,
            mutation: null,
            mutationError: requestError(error),
          }));
        }
      };
      await run();
    },
    [
      beginMutation,
      commitOperation,
      isGenerationCurrent,
      isOperationCurrent,
      reconcileOperation,
      renderGeneration,
      workspaceExists,
      workspaceId,
    ],
  );

  const retryMutation = React.useCallback(async () => {
    const retryOperation = mutationRetryRef.current;
    if (
      !workspaceExists ||
      !retryOperation ||
      !isOperationCurrent(retryOperation)
    )
      return;
    await retryOperation.run();
  }, [isOperationCurrent, workspaceExists]);

  const retryVersionConflict = React.useCallback(async () => {
    const retryOperation = versionRetryRef.current;
    const conflict =
      state.generation === generationRef.current ? state.versionConflict : null;
    if (
      !workspaceExists ||
      !retryOperation ||
      !isOperationCurrent(retryOperation) ||
      conflict === null
    ) {
      return;
    }
    await retryOperation.run(conflict.currentVersion);
  }, [
    isOperationCurrent,
    state.generation,
    state.versionConflict,
    workspaceExists,
  ]);

  const currentSignature = getSourceListViewStateSignature(currentState);
  const modified =
    exposed.activeViewId !== null &&
    (currentSignature === null || currentSignature !== exposed.activeSignature);

  return {
    available: workspaceId !== null && workspaceExists,
    generation: renderGeneration,
    views: exposed.views,
    loading: exposed.loading,
    listError: exposed.listError,
    activeViewId: exposed.activeViewId,
    activeSnapshot: exposed.activeSnapshot,
    currentSignature,
    modified,
    serializationIssues: exposed.serializationIssues,
    duplicateConflict: exposed.duplicateConflict,
    limitState: exposed.limitState,
    versionConflict: exposed.versionConflict,
    mutation: exposed.mutation,
    busy: exposed.mutation !== null,
    mutationError: exposed.mutationError,
    announcement: exposed.announcement,
    canRetryMutation:
      exposed.mutationError !== null &&
      mutationRetryRef.current !== null &&
      mutationRetryRef.current.generation === renderGeneration &&
      isOperationCurrent(mutationRetryRef.current),
    canRetryVersion:
      versionRetryRef.current !== null &&
      exposed.versionConflict !== null &&
      isOperationCurrent(versionRetryRef.current),
    load: retry,
    retry,
    retryMutation,
    retryVersionConflict,
    applyView,
    createView,
    confirmReplace,
    dismissDuplicateConflict,
    dismissMutationFailure,
    replaceView,
    resetView,
    deleteView,
  };
};

export type SourceSavedViewsController = ReturnType<typeof useSourceSavedViews>;

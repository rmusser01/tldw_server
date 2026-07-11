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

interface PatchOptions {
  kind: "replace" | "reset";
  viewId: string;
  version: number;
  body: Omit<
    Parameters<typeof tldwClient.updateWorkspaceSourceView>[2],
    "version"
  >;
  onSuccess: (view: WorkspaceSourceSavedViewValidResponse) => void;
}

interface VersionRetry {
  generation: number;
  run: (version: number) => Promise<void>;
}

interface MutationRetry {
  generation: number;
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

const isRecord = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === "object" && !Array.isArray(value);

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
): response is WorkspaceSourceSavedViewValidResponse => response.valid;

const upsertView = (
  views: WorkspaceSourceSavedViewResponse[],
  view: WorkspaceSourceSavedViewResponse,
): WorkspaceSourceSavedViewResponse[] => [
  view,
  ...views.filter((candidate) => candidate.id !== view.id),
];

const defaultWireState = (): WorkspaceSourceSavedViewStateV1 => {
  const serialized = serializeSourceListViewState(
    DEFAULT_SOURCE_LIST_VIEW_STATE,
  );
  if (!serialized.ok) throw new Error("Default source view state is invalid");
  return serialized.state;
};

export const useSourceSavedViews = (
  workspaceId: string | null,
  currentState: SourceListViewState,
  onApplyState: (state: SourceListViewState) => void,
) => {
  const identityRef = React.useRef(workspaceId);
  const generationRef = React.useRef(0);
  if (identityRef.current !== workspaceId) {
    identityRef.current = workspaceId;
    generationRef.current += 1;
  }
  const renderGeneration = generationRef.current;

  const [state, setState] = React.useState<ControllerState>(() =>
    emptyState(renderGeneration),
  );
  const versionRetryRef = React.useRef<VersionRetry | null>(null);
  const mutationRetryRef = React.useRef<MutationRetry | null>(null);

  const commit = React.useCallback(
    (
      generation: number,
      update: (current: ControllerState) => ControllerState,
    ) => {
      if (generationRef.current !== generation) return false;
      setState((current) =>
        update(
          current.generation === generation ? current : emptyState(generation),
        ),
      );
      return true;
    },
    [],
  );

  const load = React.useCallback(
    async (generation: number, targetWorkspaceId: string) => {
      commit(generation, (current) => ({
        ...current,
        loading: true,
        listError: null,
      }));
      try {
        const response =
          await tldwClient.listWorkspaceSourceViews(targetWorkspaceId);
        commit(generation, (current) => ({
          ...current,
          views: response.items,
          loading: false,
          listError: null,
        }));
      } catch (error) {
        commit(generation, (current) => ({
          ...current,
          loading: false,
          listError: requestError(error),
        }));
      }
    },
    [commit],
  );

  React.useEffect(() => {
    const generation = renderGeneration;
    versionRetryRef.current = null;
    mutationRetryRef.current = null;
    setState(emptyState(generation));
    if (workspaceId !== null) void load(generation, workspaceId);
  }, [load, renderGeneration, workspaceId]);

  const exposed =
    state.generation === renderGeneration
      ? state
      : emptyState(renderGeneration);

  const retry = React.useCallback(async () => {
    if (workspaceId === null || renderGeneration !== generationRef.current) {
      return;
    }
    await load(renderGeneration, workspaceId);
  }, [load, renderGeneration, workspaceId]);

  const applyView = React.useCallback(
    (view: WorkspaceSourceSavedViewResponse) => {
      if (!view.valid || renderGeneration !== generationRef.current) return;
      const canonical = deserializeSourceViewState(view.state);
      if (canonical === null) return;
      const signature = getSourceViewStateSignature(canonical);
      if (signature === null) return;
      onApplyState(applySavedSourceViewState(currentState, canonical));
      commit(renderGeneration, (current) => ({
        ...current,
        activeViewId: view.id,
        activeSnapshot: canonical,
        activeSignature: signature,
      }));
    },
    [commit, currentState, onApplyState, renderGeneration],
  );

  const finishValidMutation = React.useCallback(
    (
      generation: number,
      view: WorkspaceSourceSavedViewValidResponse,
      announcement: string,
      activeState: WorkspaceSourceSavedViewStateV1 = view.state,
    ) => {
      const signature = getSourceViewStateSignature(activeState);
      if (signature === null) return false;
      versionRetryRef.current = null;
      mutationRetryRef.current = null;
      return commit(generation, (current) => ({
        ...current,
        views: upsertView(current.views, view),
        activeViewId: view.id,
        activeSnapshot: activeState,
        activeSignature: signature,
        duplicateConflict: null,
        limitState: null,
        versionConflict: null,
        mutation: null,
        mutationError: null,
        announcement,
      }));
    },
    [commit],
  );

  const performPatch = React.useCallback(
    async (
      generation: number,
      targetWorkspaceId: string,
      options: PatchOptions,
    ): Promise<void> => {
      if (generation !== generationRef.current) return;
      commit(generation, (current) => ({
        ...current,
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
        if (generation !== generationRef.current) return;
        if (!validResponse(response)) {
          throw new Error("The server returned an invalid saved view.");
        }
        options.onSuccess(response);
      } catch (error) {
        if (generation !== generationRef.current) return;
        const detail = parseConflictDetail(error);
        if (detail?.code === "source_view_version_conflict") {
          versionRetryRef.current = {
            generation,
            run: (version) =>
              performPatch(generation, targetWorkspaceId, {
                ...options,
                version,
              }),
          };
          commit(generation, (current) => ({
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
          }));
          await load(generation, targetWorkspaceId);
          return;
        }
        mutationRetryRef.current = {
          generation,
          run: () => performPatch(generation, targetWorkspaceId, options),
        };
        commit(generation, (current) => ({
          ...current,
          mutation: null,
          mutationError: requestError(error),
        }));
      }
    },
    [commit, load],
  );

  const createView = React.useCallback(
    async (rawName: string) => {
      if (workspaceId === null || renderGeneration !== generationRef.current) {
        return;
      }
      const name = rawName.trim();
      const issues: SourceViewStateValidationIssue[] = [];
      if (!name || name.length > 120) {
        issues.push({
          field: "name",
          message: "Name must contain between 1 and 120 characters.",
        });
      }
      const serialized = serializeSourceListViewState(currentState);
      if (!serialized.ok) issues.push(...serialized.issues);
      if (issues.length > 0 || !serialized.ok) {
        commit(renderGeneration, (current) => ({
          ...current,
          serializationIssues: issues,
          mutation: null,
          mutationError: null,
        }));
        return;
      }

      const run = async () => {
        if (renderGeneration !== generationRef.current) return;
        commit(renderGeneration, (current) => ({
          ...current,
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
          if (renderGeneration !== generationRef.current) return;
          if (!validResponse(response)) {
            throw new Error("The server returned an invalid saved view.");
          }
          finishValidMutation(
            renderGeneration,
            response,
            "Saved view created.",
            serialized.state,
          );
        } catch (error) {
          if (renderGeneration !== generationRef.current) return;
          const detail = parseConflictDetail(error);
          if (detail?.code === "source_view_name_exists") {
            mutationRetryRef.current = null;
            commit(renderGeneration, (current) => ({
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
            commit(renderGeneration, (current) => ({
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
          mutationRetryRef.current = { generation: renderGeneration, run };
          commit(renderGeneration, (current) => ({
            ...current,
            mutation: null,
            mutationError: requestError(error),
          }));
        }
      };
      await run();
    },
    [commit, currentState, finishValidMutation, renderGeneration, workspaceId],
  );

  const confirmReplace = React.useCallback(async () => {
    const conflict = exposed.duplicateConflict;
    if (
      workspaceId === null ||
      conflict === null ||
      renderGeneration !== generationRef.current
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
      onSuccess: (view) => {
        finishValidMutation(
          renderGeneration,
          view,
          "Saved view replaced.",
          conflict.state,
        );
      },
    });
  }, [
    exposed.duplicateConflict,
    finishValidMutation,
    performPatch,
    renderGeneration,
    workspaceId,
  ]);

  const replaceView = React.useCallback(
    async (view: WorkspaceSourceSavedViewResponse) => {
      if (workspaceId === null || !view.valid) return;
      const serialized = serializeSourceListViewState(currentState);
      if (!serialized.ok) {
        commit(renderGeneration, (current) => ({
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
          name: view.name,
          schema_version: SOURCE_SAVED_VIEW_SCHEMA_VERSION,
          state: serialized.state,
        },
        onSuccess: (response) => {
          finishValidMutation(
            renderGeneration,
            response,
            "Saved view replaced.",
            serialized.state,
          );
        },
      });
    },
    [
      commit,
      currentState,
      finishValidMutation,
      performPatch,
      renderGeneration,
      workspaceId,
    ],
  );

  const resetView = React.useCallback(
    async (view: WorkspaceSourceSavedViewResponse) => {
      if (workspaceId === null || renderGeneration !== generationRef.current)
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
        onSuccess: (response) => {
          if (
            finishValidMutation(
              renderGeneration,
              response,
              "Saved view reset.",
              wireState,
            )
          ) {
            onApplyState(applySavedSourceViewState(currentState, wireState));
          }
        },
      });
    },
    [
      currentState,
      finishValidMutation,
      onApplyState,
      performPatch,
      renderGeneration,
      workspaceId,
    ],
  );

  const deleteView = React.useCallback(
    async (view: WorkspaceSourceSavedViewResponse) => {
      if (workspaceId === null || renderGeneration !== generationRef.current)
        return;
      const run = async () => {
        if (renderGeneration !== generationRef.current) return;
        commit(renderGeneration, (current) => ({
          ...current,
          mutation: "delete",
          mutationError: null,
          announcement: null,
        }));
        try {
          await tldwClient.deleteWorkspaceSourceView(workspaceId, view.id);
          if (renderGeneration !== generationRef.current) return;
          mutationRetryRef.current = null;
          commit(renderGeneration, (current) => ({
            ...current,
            views: current.views.filter(
              (candidate) => candidate.id !== view.id,
            ),
            activeViewId:
              current.activeViewId === view.id ? null : current.activeViewId,
            activeSnapshot:
              current.activeViewId === view.id ? null : current.activeSnapshot,
            activeSignature:
              current.activeViewId === view.id ? null : current.activeSignature,
            mutation: null,
            mutationError: null,
            announcement: "Saved view deleted.",
          }));
        } catch (error) {
          if (renderGeneration !== generationRef.current) return;
          mutationRetryRef.current = { generation: renderGeneration, run };
          commit(renderGeneration, (current) => ({
            ...current,
            mutation: null,
            mutationError: requestError(error),
          }));
        }
      };
      await run();
    },
    [commit, renderGeneration, workspaceId],
  );

  const retryMutation = React.useCallback(async () => {
    const retryOperation = mutationRetryRef.current;
    if (retryOperation?.generation !== generationRef.current) return;
    await retryOperation.run();
  }, []);

  const retryVersionConflict = React.useCallback(async () => {
    const retryOperation = versionRetryRef.current;
    const conflict =
      state.generation === generationRef.current ? state.versionConflict : null;
    if (
      retryOperation?.generation !== generationRef.current ||
      conflict === null
    ) {
      return;
    }
    await retryOperation.run(conflict.currentVersion);
  }, [state.generation, state.versionConflict]);

  const currentSignature = getSourceListViewStateSignature(currentState);
  const modified =
    exposed.activeViewId !== null &&
    (currentSignature === null || currentSignature !== exposed.activeSignature);

  return {
    available: workspaceId !== null,
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
    canRetryMutation: mutationRetryRef.current?.generation === renderGeneration,
    load: retry,
    retry,
    retryMutation,
    retryVersionConflict,
    applyView,
    createView,
    confirmReplace,
    replaceView,
    resetView,
    deleteView,
  };
};

export type SourceSavedViewsController = ReturnType<typeof useSourceSavedViews>;

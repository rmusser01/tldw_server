import React from "react";
import { useStoreMessageOption } from "@/store/option";
import { useStorage } from "@plasmohq/storage/hook";

/**
 * Normalizes a model string: trims whitespace, returns null for empty/non-string values.
 */
export const normalizeSelectedModel = (
  value: string | null | undefined,
): string | null => {
  if (typeof value !== "string") return null;
  let trimmed = value.trim();
  try {
    const parsed: unknown = JSON.parse(trimmed);
    if (typeof parsed === "string") trimmed = parsed.trim();
  } catch {
    // Plain model identifiers are the normal storage format.
  }
  return trimmed.length > 0 ? trimmed : null;
};

/** Freeze a specific selection revision, including a synchronously claimed own selection. */
export const createSelectedModelGuard = (
  value: string | null | undefined,
  revision: number
) => {
  const normalized = normalizeSelectedModel(value)
  return () => {
    const current = useStoreMessageOption.getState()
    return (
      current.selectedModelRevision === revision &&
      normalizeSelectedModel(current.selectedModel) === normalized
    )
  }
}

/** Capture an inherited model selection without issuing a preference write. */
export const captureSelectedModelGuard = () => {
  const { selectedModel, selectedModelRevision } =
    useStoreMessageOption.getState()
  return createSelectedModelGuard(selectedModel, selectedModelRevision)
}

// ponytail: serialize this document's model writes; use a browser lock if
// preferences need ordered writes across independent windows (TASK-13549).
let selectedModelWrites: Promise<void> = Promise.resolve();
const persistSelectedModel = (
  value: string | null,
  revision: number,
  write: (value: string | null) => Promise<void>
): Promise<boolean> => {
  const isCurrent = () => {
    const current = useStoreMessageOption.getState()
    return (
      current.selectedModelRevision === revision &&
      normalizeSelectedModel(current.selectedModel) === value
    )
  }
  const pending = selectedModelWrites.then(async () => {
    if (!isCurrent()) return false
    await write(value)
    return isCurrent()
  })
  // Keep the serial queue usable after failure without turning failure into ACK.
  selectedModelWrites = pending.then(
    () => undefined,
    () => undefined
  )
  return pending
}

export type SelectedModelSaveOperation = {
  completion: Promise<boolean>
  isCurrent: () => boolean
  retry: () => Promise<boolean>
}

/**
 * Consolidated model selection hook.
 *
 * Encapsulates the 3-way sync between:
 * - Zustand store (`useStoreMessageOption.selectedModel`)
 * - Browser storage (`useStorage("selectedModel")`)
 * - Derived `selectedModel` value (store takes precedence, storage as fallback)
 */
export const useSelectedModel = () => {
  const selectedModelFromStore = useStoreMessageOption((s) => s.selectedModel);
  const selectedModelRevision = useStoreMessageOption((s) => s.selectedModelRevision);
  const selectedModelStorageHandled = useStoreMessageOption(
    (s) => s.selectedModelStorageHandled,
  );
  const setSelectedModelInStore = useStoreMessageOption(
    (s) => s.setSelectedModel,
  );
  const [
    storedSelectedModel,
    setStoredSelectedModel,
    selectedModelStorageMeta,
  ] = useStorage<string | null>("selectedModel", null);

  const selectedModel = React.useMemo(
    () =>
      selectedModelStorageHandled
        ? normalizeSelectedModel(selectedModelFromStore)
        : (normalizeSelectedModel(selectedModelFromStore) ??
          normalizeSelectedModel(storedSelectedModel)),
    [selectedModelFromStore, storedSelectedModel, selectedModelStorageHandled],
  );

  const beginSelectedModelSave = React.useCallback(
    (
      nextOrUpdater: string | null | ((current: string | null) => string | null)
    ) => {
      const resolved =
        typeof nextOrUpdater === "function"
          ? nextOrUpdater(selectedModel)
          : nextOrUpdater
      const normalized = normalizeSelectedModel(resolved)
      const revision = useStoreMessageOption.getState().selectedModelRevision + 1
      const isCurrent = createSelectedModelGuard(normalized, revision)
      setSelectedModelInStore(normalized, { storageHandled: true })
      return {
        completion: persistSelectedModel(
          normalized,
          revision,
          setStoredSelectedModel
        ),
        isCurrent,
        retry: () =>
          persistSelectedModel(normalized, revision, setStoredSelectedModel)
      } satisfies SelectedModelSaveOperation
    },
    [selectedModel, setSelectedModelInStore, setStoredSelectedModel]
  )

  const setSelectedModel = React.useCallback(
    (next: Parameters<typeof beginSelectedModelSave>[0]) =>
      beginSelectedModelSave(next).completion,
    [beginSelectedModelSave]
  )

  const retrySelectedModelSave = React.useCallback(
    () => persistSelectedModel(selectedModel, selectedModelRevision, setStoredSelectedModel),
    [selectedModel, selectedModelRevision, setStoredSelectedModel],
  );

  // Sync effect: hydrate store from storage or push store to storage
  React.useEffect(() => {
    if (selectedModelStorageMeta.isLoading) return;
    // All mounted owners share one write claim, published before storage I/O.
    // Keep it after failure; a later selection can explicitly claim a new write.
    const current = useStoreMessageOption.getState();
    if (
      current.selectedModel !== selectedModelFromStore ||
      current.selectedModelStorageHandled
    )
      return;
    const normalizedStoreModel = normalizeSelectedModel(selectedModelFromStore);
    const normalizedStoredModel = normalizeSelectedModel(storedSelectedModel);
    if (!normalizedStoreModel && normalizedStoredModel) {
      setSelectedModelInStore(normalizedStoredModel);
      return;
    }

    if (normalizedStoreModel !== normalizedStoredModel) {
      const revision = current.selectedModelRevision + 1;
      setSelectedModelInStore(normalizedStoreModel, { storageHandled: true });
      // Explicit writes return their rejection to the caller. Hydration sync is
      // best effort and must not create an unhandled background rejection.
      void persistSelectedModel(normalizedStoreModel, revision, setStoredSelectedModel).catch(() => undefined);
    }
  }, [
    selectedModelFromStore,
    selectedModelStorageHandled,
    setSelectedModelInStore,
    setStoredSelectedModel,
    storedSelectedModel,
    selectedModelStorageMeta.isLoading,
  ]);

  return {
    selectedModel,
    setSelectedModel,
    beginSelectedModelSave,
    retrySelectedModelSave,
    selectedModelIsLoading: selectedModelStorageMeta.isLoading,
  };
};

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

  const setSelectedModel = React.useCallback(
    (
      nextOrUpdater:
        | string
        | null
        | ((current: string | null) => string | null),
    ) => {
      const resolved =
        typeof nextOrUpdater === "function"
          ? nextOrUpdater(selectedModel)
          : nextOrUpdater;
      const normalized = normalizeSelectedModel(resolved);
      setSelectedModelInStore(normalized, { storageHandled: true });
      return setStoredSelectedModel(normalized);
    },
    [selectedModel, setSelectedModelInStore, setStoredSelectedModel],
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
      setSelectedModelInStore(normalizedStoreModel, { storageHandled: true });
      // Explicit writes return their rejection to the caller. Hydration sync is
      // best effort and must not create an unhandled background rejection.
      void setStoredSelectedModel(normalizedStoreModel).catch(() => undefined);
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
    selectedModelIsLoading: selectedModelStorageMeta.isLoading,
  };
};

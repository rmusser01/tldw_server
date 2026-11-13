export const OPEN_MODEL_SELECTOR_EVENT = "tldw:open-model-selector";

export type ModelSelectorOpenDetail = {
  returnFocusSelector?: string;
};

/**
 * Ask the chat composer on this page to open its model picker. A picker that
 * takes the request cancels the event, so this returns false when none did
 * (no composer is mounted) and the caller can fall back to the chat settings.
 */
export const openModelSelector = (
  options: { returnFocusSelector?: string } = {},
): boolean => {
  if (typeof window === "undefined") return false;
  const event = new CustomEvent<ModelSelectorOpenDetail>(
    OPEN_MODEL_SELECTOR_EVENT,
    {
      cancelable: true,
      detail: { returnFocusSelector: options.returnFocusSelector },
    },
  );
  return !window.dispatchEvent(event);
};

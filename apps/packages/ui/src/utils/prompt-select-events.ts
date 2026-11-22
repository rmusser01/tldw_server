export const OPEN_PROMPT_SELECT_EVENT = "tldw:open-prompt-select";

export type PromptSelectOpenDetail = {
  returnFocusSelector?: string;
  source?: string;
  /** Start the picker's search with this text (`/prompt <text>`). */
  search?: string;
};

export function dispatchOpenPromptSelect(
  detail: PromptSelectOpenDetail = {},
) {
  if (typeof window === "undefined") return;
  window.dispatchEvent(
    new CustomEvent<PromptSelectOpenDetail>(OPEN_PROMPT_SELECT_EVENT, {
      detail,
    }),
  );
}

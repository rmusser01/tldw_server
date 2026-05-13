export const OPEN_PROMPT_SELECT_EVENT = "tldw:open-prompt-select";

export type PromptSelectOpenDetail = {
  source?: string;
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

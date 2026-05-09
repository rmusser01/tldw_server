export const OPEN_ASSISTANT_SELECT_EVENT = "tldw:open-assistant-select"

export type AssistantSelectTab = "character" | "persona"

export type AssistantSelectOpenDetail = {
  tab?: AssistantSelectTab
  source?: string
}

export function dispatchOpenAssistantSelect(
  detail: AssistantSelectOpenDetail = {}
) {
  if (typeof window === "undefined") return
  window.dispatchEvent(
    new CustomEvent<AssistantSelectOpenDetail>(OPEN_ASSISTANT_SELECT_EVENT, {
      detail
    })
  )
}

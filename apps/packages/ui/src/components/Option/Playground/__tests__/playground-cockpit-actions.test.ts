// @vitest-environment jsdom
import { describe, expect, it } from "vitest";

import {
  OPEN_ACTOR_SETTINGS_EVENT,
  OPEN_ASSISTANT_SELECT_EVENT,
  OPEN_KNOWLEDGE_PANEL_EVENT,
  OPEN_MCP_SETTINGS_EVENT,
  OPEN_MODEL_SETTINGS_EVENT,
  OPEN_PROMPT_SELECT_EVENT,
  SET_TEMPORARY_CHAT_EVENT,
  TOGGLE_WEB_SEARCH_EVENT,
  openActorSettings,
  openAssistantSelector,
  openMcpSettings,
  openModelSettings,
  openPromptSelector,
  openSearchAndContext,
  setTemporaryChatFromCockpit,
  toggleWebSearchFromCockpit,
} from "../playground-cockpit-actions";

const nextCustomEvent = <TDetail = unknown>(eventName: string) =>
  new Promise<CustomEvent<TDetail>>((resolve) => {
    const handler = (event: Event) => {
      window.removeEventListener(eventName, handler);
      resolve(event as CustomEvent<TDetail>);
    };
    window.addEventListener(eventName, handler);
  });

describe("playground cockpit actions", () => {
  it("opens Search & Context with the existing knowledge-panel event shape", async () => {
    const event = nextCustomEvent<{ tab: "search" }>(
      OPEN_KNOWLEDGE_PANEL_EVENT,
    );

    openSearchAndContext({ tab: "search" });

    await expect(event).resolves.toMatchObject({
      detail: { tab: "search" },
    });
  });

  it("opens model settings through the existing model-settings event", async () => {
    const event = nextCustomEvent(OPEN_MODEL_SETTINGS_EVENT);

    openModelSettings();

    await expect(event).resolves.toMatchObject({ type: OPEN_MODEL_SETTINGS_EVENT });
  });

  it("opens actor settings through the existing actor-settings event", async () => {
    const event = nextCustomEvent(OPEN_ACTOR_SETTINGS_EVENT);

    openActorSettings();

    await expect(event).resolves.toMatchObject({ type: OPEN_ACTOR_SETTINGS_EVENT });
  });

  it("opens the assistant selector as the cockpit character/persona primary path", async () => {
    const event = nextCustomEvent<{ tab: "persona"; source: string }>(
      OPEN_ASSISTANT_SELECT_EVENT,
    );

    openAssistantSelector({ tab: "persona" });

    await expect(event).resolves.toMatchObject({
      type: OPEN_ASSISTANT_SELECT_EVENT,
      detail: { tab: "persona", source: "playground-cockpit" },
    });
  });

  it("opens MCP settings directly instead of only the composer MCP popover", async () => {
    const event = nextCustomEvent(OPEN_MCP_SETTINGS_EVENT);

    openMcpSettings();

    await expect(event).resolves.toMatchObject({ type: OPEN_MCP_SETTINGS_EVENT });
  });

  it("opens the shared prompt selector from the cockpit rail", async () => {
    const event = nextCustomEvent<{
      returnFocusSelector: string;
      source: string;
    }>(OPEN_PROMPT_SELECT_EVENT);

    openPromptSelector({
      returnFocusSelector: "[data-testid='cockpit-prompt-select-trigger']",
    });

    await expect(event).resolves.toMatchObject({
      type: OPEN_PROMPT_SELECT_EVENT,
      detail: {
        returnFocusSelector: "[data-testid='cockpit-prompt-select-trigger']",
        source: "playground-cockpit",
      },
    });
  });

  it("toggles web search through the cockpit bridge event", async () => {
    const event = nextCustomEvent(TOGGLE_WEB_SEARCH_EVENT);

    toggleWebSearchFromCockpit();

    await expect(event).resolves.toMatchObject({ type: TOGGLE_WEB_SEARCH_EVENT });
  });

  it("sets temporary-chat state through the cockpit bridge event", async () => {
    const event = nextCustomEvent<{ next: boolean }>(SET_TEMPORARY_CHAT_EVENT);

    setTemporaryChatFromCockpit(true);

    await expect(event).resolves.toMatchObject({
      detail: { next: true },
    });
  });
});

// @vitest-environment jsdom
import React from "react";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { ServerChatHistoryItem } from "@/hooks/useServerChatHistory";
import { CharacterChatSessionsPanel } from "../CharacterChatSessionsPanel";

const historyHookMock = vi.hoisted(() => vi.fn());
const selectServerChatMock = vi.hoisted(() => vi.fn());
const formatRelativeTimeMock = vi.hoisted(() =>
  vi.fn((value: string) => `relative:${value}`),
);

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      _key: string,
      defaultValueOrOptions?: string | { defaultValue?: string },
      interpolationOptions?: Record<string, unknown>,
    ) => {
      const defaultValue =
        typeof defaultValueOrOptions === "string"
          ? defaultValueOrOptions
          : defaultValueOrOptions?.defaultValue || _key;
      const values =
        typeof defaultValueOrOptions === "object" && defaultValueOrOptions
          ? defaultValueOrOptions
          : interpolationOptions;

      return defaultValue.replace(/\{\{(\w+)\}\}/g, (_match, key) =>
        values?.[key] == null ? `{{${key}}}` : String(values[key]),
      );
    },
  }),
}));

vi.mock("@/hooks/useServerChatHistory", () => ({
  SERVER_CHAT_HISTORY_OVERVIEW_PAGE_SIZE: 25,
  useServerChatHistory: (...args: unknown[]) => historyHookMock(...args),
}));

vi.mock("@/hooks/chat/useSelectServerChat", () => ({
  useSelectServerChat: () => selectServerChatMock,
}));

vi.mock("@/utils/dateFormatters", () => ({
  formatRelativeTime: (...args: Parameters<typeof formatRelativeTimeMock>) =>
    formatRelativeTimeMock(...args),
}));

const makeChat = (
  id: string,
  overrides: Partial<ServerChatHistoryItem> = {},
): ServerChatHistoryItem =>
  ({
    id,
    title: `Chat ${id}`,
    created_at: "2026-05-20T00:00:00.000Z",
    updated_at: "2026-05-20T00:10:00.000Z",
    createdAtMs: Date.parse("2026-05-20T00:00:00.000Z"),
    updatedAtMs: Date.parse("2026-05-20T00:10:00.000Z"),
    state: "in-progress",
    ...overrides,
  }) as ServerChatHistoryItem;

describe("CharacterChatSessionsPanel", () => {
  beforeEach(() => {
    historyHookMock.mockReset();
    selectServerChatMock.mockReset();
    formatRelativeTimeMock.mockClear();
    historyHookMock.mockReturnValue({
      data: [],
      total: 0,
      isLoading: false,
      sidebarRefreshState: "ready",
      hasUsableData: false,
      isShowingStaleData: false,
    });
  });

  it("fetches character-scoped recent sessions and prioritizes the active character", () => {
    const currentCharacterChat = makeChat("chat-current", {
      title: "Mira Act I",
      character_id: "mira",
    });
    const nextMiraChat = makeChat("chat-next", {
      title: "Mira Act II",
      character_id: "mira",
    });
    const otherCharacterChat = makeChat("chat-other", {
      title: "Rook Side Quest",
      character_id: "rook",
    });
    historyHookMock.mockReturnValue({
      data: [otherCharacterChat, nextMiraChat, currentCharacterChat],
      total: 3,
      isLoading: false,
      sidebarRefreshState: "ready",
      hasUsableData: true,
      isShowingStaleData: false,
    });

    render(
      <CharacterChatSessionsPanel
        activeCharacterId="mira"
        activeCharacterName="Mira"
        activeServerChatId="chat-current"
      />,
    );

    expect(historyHookMock).toHaveBeenCalledWith(
      "",
      expect.objectContaining({
        enabled: true,
        filterMode: "character",
        limit: 5,
        mode: "overview",
        page: 1,
      }),
    );

    const currentCharacterList = screen.getByRole("list", {
      name: "Recent sessions for Mira",
    });
    expect(
      within(currentCharacterList).getByText("Mira Act I"),
    ).toBeInTheDocument();
    expect(
      within(currentCharacterList).getByText("Mira Act II"),
    ).toBeInTheDocument();
    expect(
      within(currentCharacterList).queryByText("Rook Side Quest"),
    ).not.toBeInTheDocument();

    const otherCharactersList = screen.getByRole("list", {
      name: "Other character sessions",
    });
    expect(
      within(otherCharactersList).getByText("Rook Side Quest"),
    ).toBeInTheDocument();

    expect(
      screen.getByRole("button", { name: "Current Mira Act I" }),
    ).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Resume Mira Act II" }));
    expect(selectServerChatMock).toHaveBeenCalledWith(nextMiraChat);
  });

  it("uses character assistant identity when character_id is missing", () => {
    const assistantBackedChat = makeChat("chat-assistant-backed", {
      title: "Mira Assistant Identity",
      character_id: null,
      assistant_kind: "character",
      assistant_id: "mira",
    });
    const personaBackedChat = makeChat("chat-persona", {
      title: "Garden Helper",
      character_id: null,
      assistant_kind: "persona",
      assistant_id: "mira",
    });
    historyHookMock.mockReturnValue({
      data: [personaBackedChat, assistantBackedChat],
      total: 2,
      isLoading: false,
      sidebarRefreshState: "ready",
      hasUsableData: true,
      isShowingStaleData: false,
    });

    render(
      <CharacterChatSessionsPanel
        activeCharacterId="mira"
        activeCharacterName="Mira"
        activeServerChatId={null}
      />,
    );

    const currentCharacterList = screen.getByRole("list", {
      name: "Recent sessions for Mira",
    });
    expect(
      within(currentCharacterList).getByText("Mira Assistant Identity"),
    ).toBeInTheDocument();
    expect(
      within(currentCharacterList).queryByText("Garden Helper"),
    ).not.toBeInTheDocument();

    expect(
      within(
        screen.getByRole("list", { name: "Other character sessions" }),
      ).getByText("Garden Helper"),
    ).toBeInTheDocument();
  });

  it("shows local loading, empty, and refresh-error states distinct from saved setups", () => {
    historyHookMock.mockReturnValueOnce({
      data: [],
      total: 0,
      isLoading: true,
      sidebarRefreshState: "idle",
      hasUsableData: false,
      isShowingStaleData: false,
    });
    const { rerender } = render(
      <CharacterChatSessionsPanel
        activeCharacterId="mira"
        activeCharacterName="Mira"
        activeServerChatId={null}
      />,
    );
    expect(
      screen.getByText("Loading character sessions..."),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("status", { name: "Loading character sessions" }),
    ).toHaveTextContent("Loading character sessions...");

    historyHookMock.mockReturnValueOnce({
      data: [],
      total: 0,
      isLoading: false,
      sidebarRefreshState: "ready",
      hasUsableData: false,
      isShowingStaleData: false,
    });
    rerender(
      <CharacterChatSessionsPanel
        activeCharacterId="mira"
        activeCharacterName="Mira"
        activeServerChatId={null}
      />,
    );
    expect(
      screen.getByText("No character conversations yet."),
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        "Recent character conversations stay separate from saved role-play setups.",
      ),
    ).toBeInTheDocument();

    historyHookMock.mockReturnValueOnce({
      data: [],
      total: 0,
      isLoading: false,
      sidebarRefreshState: "recoverable-error",
      hasUsableData: false,
      isShowingStaleData: false,
    });
    rerender(
      <CharacterChatSessionsPanel
        activeCharacterId="mira"
        activeCharacterName="Mira"
        activeServerChatId={null}
      />,
    );
    expect(
      screen.getByText("Unable to refresh character sessions right now."),
    ).toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent(
      "Unable to refresh character sessions right now.",
    );
  });

  it("announces stale and hard-failure session states", () => {
    const staleChat = makeChat("chat-stale", {
      title: "Stale Mira",
      character_id: "mira",
    });
    historyHookMock.mockReturnValueOnce({
      data: [staleChat],
      total: 1,
      isLoading: false,
      sidebarRefreshState: "ready",
      hasUsableData: true,
      isShowingStaleData: true,
    });
    const { rerender } = render(
      <CharacterChatSessionsPanel
        activeCharacterId="mira"
        activeCharacterName="Mira"
        activeServerChatId={null}
      />,
    );

    expect(screen.getByRole("status", { name: "Character session refresh" }))
      .toHaveTextContent(
        "Showing character sessions from the last successful refresh.",
      );

    historyHookMock.mockReturnValueOnce({
      data: [],
      total: 0,
      isLoading: false,
      sidebarRefreshState: "hard-error",
      hasUsableData: false,
      isShowingStaleData: false,
    });
    rerender(
      <CharacterChatSessionsPanel
        activeCharacterId="mira"
        activeCharacterName="Mira"
        activeServerChatId={null}
      />,
    );

    expect(screen.getByRole("alert")).toHaveTextContent(
      "Character sessions could not be loaded.",
    );
  });

  it("normalizes timestamps before rendering relative session age", () => {
    const chatWithPaddedTimestamp = makeChat("chat-padded", {
      title: "Padded Timestamp",
      character_id: "mira",
      updated_at: "  2026-05-20T00:10:00.000Z  ",
    });
    historyHookMock.mockReturnValue({
      data: [chatWithPaddedTimestamp],
      total: 1,
      isLoading: false,
      sidebarRefreshState: "ready",
      hasUsableData: true,
      isShowingStaleData: false,
    });

    render(
      <CharacterChatSessionsPanel
        activeCharacterId="mira"
        activeCharacterName="Mira"
        activeServerChatId={null}
      />,
    );

    expect(formatRelativeTimeMock).toHaveBeenCalledWith(
      "2026-05-20T00:10:00.000Z",
      expect.any(Function),
      { compact: true },
    );
    expect(
      screen.getByText("relative:2026-05-20T00:10:00.000Z"),
    ).toBeInTheDocument();
  });

  it("shows hard history load failures as an error instead of an empty state", () => {
    historyHookMock.mockReturnValue({
      data: [],
      total: 0,
      isLoading: false,
      sidebarRefreshState: "hard-error",
      hasUsableData: false,
      isShowingStaleData: false,
    });

    render(
      <CharacterChatSessionsPanel
        activeCharacterId="mira"
        activeCharacterName="Mira"
        activeServerChatId={null}
      />,
    );

    expect(
      screen.getByText("Character sessions could not be loaded."),
    ).toBeInTheDocument();
    expect(
      screen.queryByText("No character conversations yet."),
    ).not.toBeInTheDocument();
  });
});

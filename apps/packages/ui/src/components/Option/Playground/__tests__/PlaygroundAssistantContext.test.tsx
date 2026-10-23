// @vitest-environment jsdom
import React from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { PlaygroundAssistantContext } from "../PlaygroundAssistantContext";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?:
        | string
        | { defaultValue?: string; [key: string]: unknown },
    ) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions;
      const template = fallbackOrOptions?.defaultValue || key;
      return template.replace(/\{\{(\w+)\}\}/g, (_, token: string) => {
        const value = fallbackOrOptions?.[token];
        return value == null ? `{{${token}}}` : String(value);
      });
    },
  }),
}));

const renderAssistant = (
  overrides: Partial<
    React.ComponentProps<typeof PlaygroundAssistantContext>
  > = {},
) => {
  const props = {
    assistantSummary: {
      mode: "character" as const,
      name: "Mira Vale",
      detail: "Character selected",
    },
    onOpenAssistantSelect: vi.fn(),
    onClearAssistant: vi.fn(),
    onInspectAssistant: vi.fn(),
    onOpenSceneDirector: vi.fn(),
    ...overrides,
  };
  render(<PlaygroundAssistantContext {...props} />);
  return props;
};
describe("assistant controls in Context", () => {
  it("keeps assistant selection available without empty mode badges", () => {
    const props = renderAssistant({
      assistantSummary: { mode: "none", name: null },
    });
    fireEvent.click(
      screen.getByRole("button", { name: "Select character or persona" }),
    );
    expect(props.onOpenAssistantSelect).toHaveBeenCalledOnce();
    expect(screen.queryByText("None")).toBeNull();
  });
  it("shows a clear assistant action when a character is selected", () => {
    const props = renderAssistant();

    fireEvent.click(screen.getByRole("button", { name: "Clear assistant" }));

    expect(props.onClearAssistant).toHaveBeenCalledTimes(1);
  });

  it("shows a clear assistant action when a persona is selected", () => {
    const props = renderAssistant({
      assistantSummary: {
        mode: "persona",
        name: "Research Persona",
        detail: "Persona selected",
      },
    });

    fireEvent.click(screen.getByRole("button", { name: "Clear assistant" }));

    expect(props.onClearAssistant).toHaveBeenCalledTimes(1);
  });

  it("opens the supplied manage path for selected assistants", () => {
    const props = renderAssistant();

    fireEvent.click(screen.getByRole("button", { name: "Manage assistant" }));

    expect(props.onInspectAssistant).toHaveBeenCalledTimes(1);
  });

  it("keeps Scene Director secondary to the primary assistant selector", () => {
    const props = renderAssistant();

    fireEvent.click(
      screen.getByRole("button", { name: "Open Scene Director" }),
    );

    expect(props.onOpenSceneDirector).toHaveBeenCalledTimes(1);
    expect(props.onOpenAssistantSelect).not.toHaveBeenCalled();
  });

  it("keeps Scene Director character-only and explains persona mode", () => {
    renderAssistant({
      assistantSummary: {
        mode: "persona",
        name: "Research Persona",
        detail: "Persona selected",
      },
    });

    expect(
      screen.queryByRole("button", { name: "Open Scene Director" }),
    ).toBeNull();
    expect(
      screen.getByText(
        "Scene Director is available for character-backed chats.",
      ),
    ).toBeInTheDocument();
  });
});

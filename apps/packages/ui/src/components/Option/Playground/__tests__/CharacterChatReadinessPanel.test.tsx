// @vitest-environment jsdom
import React from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { buildCharacterChatReadiness } from "@/utils/chat-model-availability";

import { CharacterChatReadinessPanel } from "../CharacterChatReadinessPanel";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      _key: string,
      fallbackOrOptions?: string | { defaultValue?: string; [key: string]: unknown },
    ) => {
      if (_key === "characterChatReadiness.missingRestoredCharacter.title") {
        return typeof fallbackOrOptions === "string"
          ? fallbackOrOptions
          : `Translated restored character ${fallbackOrOptions?.id}`;
      }
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions;
      const template = fallbackOrOptions?.defaultValue || _key;
      return template.replace(/\{\{(\w+)\}\}/g, (_, token: string) => {
        const value = fallbackOrOptions?.[token];
        return value == null ? `{{${token}}}` : String(value);
      });
    },
  }),
}));

describe("CharacterChatReadinessPanel", () => {
  it("announces a missing model with selected character context", () => {
    const action = vi.fn();

    render(
      <CharacterChatReadinessPanel
        readiness={buildCharacterChatReadiness({
          isServerConnected: true,
          selectedCharacter: { id: "ariadne", name: "Ariadne" },
          selectedModel: null,
        })}
        characterName="Ariadne"
        onAction={action}
      />,
    );

    const panel = screen.getByRole("status", {
      name: "Character Chat setup status",
    });
    expect(panel).toHaveAttribute("aria-live", "polite");
    expect(panel).toHaveTextContent(
      "Choose a chat model before chatting as Ariadne",
    );
    expect(panel).toHaveTextContent(
      "return here to continue with Ariadne",
    );

    fireEvent.click(screen.getByRole("button", { name: "Open model settings" }));
    expect(action).toHaveBeenCalledWith("open-model-settings");
  });

  it("shows an unavailable selected model as a model-settings recovery", () => {
    const action = vi.fn();

    render(
      <CharacterChatReadinessPanel
        readiness={buildCharacterChatReadiness({
          isServerConnected: true,
          selectedCharacter: { id: "ariadne", name: "Ariadne" },
          selectedModel: "missing-model",
          availableModels: [{ model: "gpt-4o-mini" }],
        })}
        characterName="Ariadne"
        onAction={action}
      />,
    );

    expect(screen.getByRole("status")).toHaveTextContent(
      "Choose a chat model before chatting as Ariadne",
    );
    fireEvent.click(screen.getByRole("button", { name: "Open model settings" }));
    expect(action).toHaveBeenCalledWith("open-model-settings");
  });

  it("surfaces a restored route character that can no longer be loaded", () => {
    const chooseCharacter = vi.fn();
    const retry = vi.fn();

    render(
      <CharacterChatReadinessPanel
        readiness={buildCharacterChatReadiness({
          isServerConnected: true,
          selectedCharacter: { id: "missing-character", name: "Character missing-character" },
          selectedModel: "gpt-4o-mini",
        })}
        missingCharacter={{
          id: "missing-character",
          reason: "missing",
        }}
        onAction={vi.fn()}
        onChooseCharacter={chooseCharacter}
        onRetryMissingCharacter={retry}
      />,
    );

    const panel = screen.getByRole("status", {
      name: "Character Chat setup status",
    });
    expect(panel).toHaveTextContent(
      "Translated restored character missing-character",
    );
    expect(panel).toHaveTextContent("Choose another character or retry loading it.");

    fireEvent.click(screen.getByRole("button", { name: "Choose character" }));
    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(chooseCharacter).toHaveBeenCalledTimes(1);
    expect(retry).toHaveBeenCalledTimes(1);
  });

  it("does not render a panel for the ready state", () => {
    const { container } = render(
      <CharacterChatReadinessPanel
        readiness={buildCharacterChatReadiness({
          isServerConnected: true,
          selectedCharacter: { id: "ariadne", name: "Ariadne" },
          selectedModel: "gpt-4o-mini",
          availableModels: [{ model: "gpt-4o-mini" }],
        })}
        characterName="Ariadne"
        onAction={vi.fn()}
      />,
    );

    expect(container).toBeEmptyDOMElement();
  });
});

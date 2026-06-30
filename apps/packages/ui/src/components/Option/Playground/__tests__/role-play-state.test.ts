import { describe, expect, it } from "vitest";

import {
  deriveRolePlayState,
  getDefaultRolePlayGenerationStyle,
  type RolePlayTemplateMetadata,
} from "../role-play-state";

const characterActorTemplate: RolePlayTemplateMetadata = {
  id: "character-actor",
  title: "Character Actor",
  category: "roleplay",
  content: "Stay in character and respond as the selected persona.",
  description: "Roleplay as a specific character or persona",
  tags: ["roleplay", "character", "acting"],
};

describe("deriveRolePlayState", () => {
  it("is inactive when no role-play identity, behavior, scene, preset, or context is present", () => {
    const state = deriveRolePlayState({});

    expect(state).toEqual({
      active: false,
      identity: null,
      behavior: null,
      scene: null,
      generationStyle: null,
      context: {
        pinnedCount: 0,
        hasExternalContext: false,
      },
    });
  });

  it("represents selected characters as a character identity layer", () => {
    const state = deriveRolePlayState({
      identity: {
        kind: "character",
        id: "char-mira",
        name: "Mira",
      },
    });

    expect(state.active).toBe(true);
    expect(state.identity).toEqual({
      kind: "character",
      id: "char-mira",
      name: "Mira",
    });
  });

  it("represents selected personas as distinct from character identity", () => {
    const state = deriveRolePlayState({
      identity: {
        kind: "persona",
        id: "persona-mentor",
        name: "Stern Mentor",
      },
    });

    expect(state.active).toBe(true);
    expect(state.identity).toEqual({
      kind: "persona",
      id: "persona-mentor",
      name: "Stern Mentor",
    });
  });

  it("keeps applied role-play templates named as behavior templates", () => {
    const state = deriveRolePlayState({
      systemPrompt: characterActorTemplate.content,
      behaviorTemplate: characterActorTemplate,
    });

    expect(state.active).toBe(true);
    expect(state.behavior).toEqual({
      source: "template",
      templateId: "character-actor",
      title: "Character Actor",
      modified: false,
    });
  });

  it("marks behavior templates as modified when the prompt text changes after apply", () => {
    const state = deriveRolePlayState({
      systemPrompt: `${characterActorTemplate.content}\nAdd a noir tone.`,
      behaviorTemplate: characterActorTemplate,
    });

    expect(state.behavior).toEqual({
      source: "modified-template",
      templateId: "character-actor",
      title: "Character Actor",
      modified: true,
    });
  });

  it("represents custom prompt text without a template as a custom system prompt", () => {
    const state = deriveRolePlayState({
      systemPrompt: "Speak like a sarcastic ship computer.",
    });

    expect(state.behavior).toEqual({
      source: "custom",
      modified: false,
      title: "Custom",
    });
  });

  it("exposes non-custom generation presets as a generation style layer", () => {
    const state = deriveRolePlayState({
      generationStyle: {
        key: "creative",
        label: "Creative",
      },
    });

    expect(state.active).toBe(true);
    expect(state.generationStyle).toEqual({
      key: "creative",
      label: "Creative",
    });
  });

  it("centralizes generation style reset on Balanced", () => {
    expect(getDefaultRolePlayGenerationStyle()).toEqual({
      key: "balanced",
      label: "Balanced",
    });
  });

  it("summarizes pinned and external context without exposing source-management actions", () => {
    const state = deriveRolePlayState({
      context: {
        pinnedCount: 2,
        hasExternalContext: true,
      },
    });

    expect(state.active).toBe(true);
    expect(state.context).toEqual({
      pinnedCount: 2,
      hasExternalContext: true,
    });
    expect("onClick" in state.context).toBe(false);
  });
});

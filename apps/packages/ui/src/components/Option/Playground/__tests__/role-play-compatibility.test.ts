import { describe, expect, it } from "vitest";

import { deriveRolePlayCompatibility } from "../role-play-compatibility";

const baseInput = {
  hasCharacter: true,
  hasPersona: false,
  compareModeActive: false,
  isImageCommand: false,
  hasContextFiles: false,
  hasSelectedDocuments: false,
  hasDocumentContext: false,
  hasSelectedKnowledge: false,
  fileRetrievalEnabled: false,
  hasScopedRagMediaIds: false,
  ragPinnedResultsLength: 0,
  hasCustomPrompt: false,
};

describe("deriveRolePlayCompatibility", () => {
  it("reports no role-play context when no character or persona is selected", () => {
    expect(
      deriveRolePlayCompatibility({
        ...baseInput,
        hasCharacter: false,
      }),
    ).toEqual(
      expect.objectContaining({
        status: "none",
        reasonCode: "no_identity",
      }),
    );
  });

  it("reports selected character context as included in the plain character flow", () => {
    expect(deriveRolePlayCompatibility(baseInput)).toEqual(
      expect.objectContaining({
        status: "included",
        reasonCode: "character_flow",
      }),
    );
  });

  it("reports selected persona context separately from character context", () => {
    expect(
      deriveRolePlayCompatibility({
        ...baseInput,
        hasCharacter: false,
        hasPersona: true,
      }),
    ).toEqual(
      expect.objectContaining({
        status: "included",
        reasonCode: "persona_flow",
      }),
    );
  });

  it("reports custom prompt steering as an override risk for character role-play", () => {
    expect(
      deriveRolePlayCompatibility({
        ...baseInput,
        hasCustomPrompt: true,
      }),
    ).toEqual(
      expect.objectContaining({
        status: "override-risk",
        reasonCode: "custom_prompt",
      }),
    );
  });

  it("reports pinned sources as blended with the character request path", () => {
    expect(
      deriveRolePlayCompatibility({
        ...baseInput,
        ragPinnedResultsLength: 2,
      }),
    ).toEqual(
      expect.objectContaining({
        status: "blended",
        reasonCode: "rag_sources",
      }),
    );
  });

  it.each([
    ["selected knowledge", { hasSelectedKnowledge: true }, "rag_sources"],
    [
      "scoped file retrieval RAG",
      { fileRetrievalEnabled: true, hasScopedRagMediaIds: true },
      "rag_sources",
    ],
    ["context files", { hasContextFiles: true }, "context_files"],
    ["selected documents", { hasSelectedDocuments: true }, "documents"],
    ["document context", { hasDocumentContext: true }, "documents"],
    ["image command", { isImageCommand: true }, "image_command"],
    ["compare mode", { compareModeActive: true }, "compare_mode"],
  ] as const)(
    "reports character context as excluded for %s",
    (_label, override, reasonCode) => {
      expect(
        deriveRolePlayCompatibility({
          ...baseInput,
          ...override,
        }),
      ).toEqual(
        expect.objectContaining({
          status: "excluded",
          reasonCode,
        }),
      );
    },
  );
});

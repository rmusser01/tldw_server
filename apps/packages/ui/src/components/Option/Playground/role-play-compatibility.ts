export type RolePlayContextStatus =
  | "none"
  | "included"
  | "blended"
  | "excluded"
  | "override-risk";

export type RolePlayCompatibilityReasonCode =
  | "no_identity"
  | "character_flow"
  | "persona_flow"
  | "custom_prompt"
  | "rag_sources"
  | "compare_mode"
  | "context_files"
  | "documents"
  | "image_command";

export type RolePlayCompatibility = {
  status: RolePlayContextStatus;
  reasonCode: RolePlayCompatibilityReasonCode;
  messageKey: string;
};

export type RolePlayCompatibilityInput = {
  hasCharacter: boolean;
  hasPersona: boolean;
  compareModeActive: boolean;
  isImageCommand: boolean;
  hasContextFiles: boolean;
  hasSelectedDocuments: boolean;
  hasDocumentContext: boolean;
  hasSelectedKnowledge: boolean;
  fileRetrievalEnabled: boolean;
  hasScopedRagMediaIds: boolean;
  ragPinnedResultsLength: number;
  hasCustomPrompt: boolean;
};

const compatibility = (
  status: RolePlayContextStatus,
  reasonCode: RolePlayCompatibilityReasonCode,
): RolePlayCompatibility => ({
  status,
  reasonCode,
  messageKey: `playground:composer.rolePlayCompatibility.${reasonCode}`,
});

export const deriveRolePlayCompatibility = (
  input: RolePlayCompatibilityInput,
): RolePlayCompatibility => {
  if (!input.hasCharacter && !input.hasPersona) {
    return compatibility("none", "no_identity");
  }

  if (input.isImageCommand) {
    return compatibility("excluded", "image_command");
  }

  if (input.hasContextFiles) {
    return compatibility("excluded", "context_files");
  }

  if (input.hasSelectedDocuments || input.hasDocumentContext) {
    return compatibility("excluded", "documents");
  }

  if (
    input.hasSelectedKnowledge ||
    (input.fileRetrievalEnabled && input.hasScopedRagMediaIds)
  ) {
    return compatibility("excluded", "rag_sources");
  }

  if (input.compareModeActive && input.hasCharacter) {
    return compatibility("excluded", "compare_mode");
  }

  if (input.hasPersona && !input.hasCharacter) {
    return compatibility("included", "persona_flow");
  }

  if (input.hasCustomPrompt) {
    return compatibility("override-risk", "custom_prompt");
  }

  if (input.ragPinnedResultsLength > 0) {
    return compatibility("blended", "rag_sources");
  }

  return compatibility("included", "character_flow");
};

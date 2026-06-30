export type RolePlayIdentityKind = "character" | "persona" | "assistant";
export type RolePlayPromptSource =
  | "none"
  | "template"
  | "custom"
  | "modified-template";

export type RolePlayIdentity = {
  kind: RolePlayIdentityKind;
  id?: string;
  name?: string;
};

export type RolePlayTemplateMetadata = {
  id: string;
  title: string;
  category: string;
  content: string;
  description?: string;
  tags?: string[];
};

export type RolePlayBehaviorState = {
  source: Exclude<RolePlayPromptSource, "none">;
  templateId?: string;
  title?: string;
  modified: boolean;
};

export type RolePlayState = {
  active: boolean;
  identity: RolePlayIdentity | null;
  behavior: RolePlayBehaviorState | null;
  scene: { active: boolean; summary?: string } | null;
  generationStyle: { key: string; label: string } | null;
  context: { pinnedCount: number; hasExternalContext: boolean };
};

export type DeriveRolePlayStateInput = {
  identity?: RolePlayIdentity | null;
  systemPrompt?: string | null;
  selectedSystemPrompt?: string | null;
  selectedQuickPrompt?: string | null;
  behaviorTemplate?: RolePlayTemplateMetadata | null;
  scene?: { active?: boolean; summary?: string | null } | null;
  generationStyle?: { key: string; label: string } | null;
  context?: {
    pinnedCount?: number | null;
    hasExternalContext?: boolean | null;
  } | null;
};

const normalizePromptText = (value: string | null | undefined) =>
  String(value || "").trim();

const normalizeIdentity = (
  identity: RolePlayIdentity | null | undefined,
): RolePlayIdentity | null => {
  if (!identity) return null;
  const name = normalizePromptText(identity.name);
  const id = normalizePromptText(identity.id);
  if (!name && !id) return null;
  return {
    kind: identity.kind,
    ...(id ? { id } : {}),
    ...(name ? { name } : {}),
  };
};

const deriveBehavior = (
  input: DeriveRolePlayStateInput,
): RolePlayBehaviorState | null => {
  const promptText = normalizePromptText(input.systemPrompt);
  const hasSelectedPrompt =
    normalizePromptText(input.selectedSystemPrompt).length > 0 ||
    normalizePromptText(input.selectedQuickPrompt).length > 0;

  if (!promptText && !hasSelectedPrompt) return null;

  if (input.behaviorTemplate && promptText) {
    const templateText = normalizePromptText(input.behaviorTemplate.content);
    const modified = promptText !== templateText;
    return {
      source: modified ? "modified-template" : "template",
      templateId: input.behaviorTemplate.id,
      title: input.behaviorTemplate.title,
      modified,
    };
  }

  return {
    source: "custom",
    title: "Custom",
    modified: false,
  };
};

const deriveScene = (
  scene: DeriveRolePlayStateInput["scene"],
): RolePlayState["scene"] => {
  if (!scene?.active) return null;
  const summary = normalizePromptText(scene.summary);
  return {
    active: true,
    ...(summary ? { summary } : {}),
  };
};

export const getDefaultRolePlayGenerationStyle = (label = "Balanced") => ({
  key: "balanced",
  label,
});

export const deriveRolePlayState = (
  input: DeriveRolePlayStateInput,
): RolePlayState => {
  const identity = normalizeIdentity(input.identity);
  const behavior = deriveBehavior(input);
  const scene = deriveScene(input.scene);
  const generationStyle = input.generationStyle
    ? {
        key: input.generationStyle.key,
        label: input.generationStyle.label,
      }
    : null;
  const context = {
    pinnedCount: Math.max(0, Number(input.context?.pinnedCount || 0)),
    hasExternalContext: Boolean(input.context?.hasExternalContext),
  };
  const active = Boolean(
    identity ||
    behavior ||
    scene ||
    generationStyle ||
    context.pinnedCount > 0 ||
    context.hasExternalContext,
  );

  return {
    active,
    identity,
    behavior,
    scene,
    generationStyle,
    context,
  };
};

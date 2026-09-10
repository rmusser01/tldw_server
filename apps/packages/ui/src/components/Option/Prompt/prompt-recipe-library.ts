import type { SavedRecipeSource } from "@/components/Common/PromptAssist/recipes/types"
import { parseStructuredPromptDefinitionForTransport } from "@/services/structured-prompt-transport"
import type { PromptCapabilities } from "@/services/prompts-api"
import {
  renderSingleTextRecipeTemplate,
  type SingleTextRecipeDefinitionV2
} from "./structured-prompt-utils"

export type RecipeTargetFilter = "recipe" | "recipe_system" | "recipe_user"

export type PromptRecipeClassification =
  | { kind: "ordinary" }
  | {
      kind: "recipe"
      target: "system" | "user"
      definition: SingleTextRecipeDefinitionV2
    }
  | { kind: "quarantined_recipe" }

export const getRecipePersistenceState = (
  isOnline: boolean,
  capabilities: PromptCapabilities | undefined
): { available: boolean; reason: string } => {
  if (!isOnline) {
    return {
      available: false,
      reason:
        "Recipe saving is unavailable offline. You can still edit, preview, and apply this local draft."
    }
  }
  if (!capabilities) {
    return {
      available: false,
      reason:
        "Checking whether this server supports recipe saving. You can still edit, preview, and apply."
    }
  }
  if (
    capabilities.availability === "available" &&
    capabilities.single_text_recipe_v2.supported === true
  ) {
    return { available: true, reason: "" }
  }
  return {
    available: false,
    reason:
      capabilities.availability === "available"
        ? "This server does not support recipe saving yet. You can still edit, preview, and apply."
        : "Recipe saving is unavailable because server capabilities could not be confirmed. You can still edit, preview, and apply."
  }
}

const clone = <T>(value: T): T => structuredClone(value)

const isRecipeLike = (prompt: Record<string, unknown>): boolean => {
  const definition = prompt.structuredPromptDefinition
  const inner =
    definition && typeof definition === "object" && !Array.isArray(definition)
      ? (definition as Record<string, unknown>)
      : null
  return (
    (prompt.promptFormat === "structured" &&
      prompt.promptSchemaVersion !== 1) ||
    inner?.schema_version === 2 ||
    inner?.definition_kind === "single_text_recipe" ||
    (inner?.assembly_config != null &&
      typeof inner.assembly_config === "object" &&
      (inner.assembly_config as Record<string, unknown>).assembly_mode ===
        "single_text")
  )
}

export const classifyPromptRecipe = (
  value: unknown
): PromptRecipeClassification => {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return { kind: "ordinary" }
  }
  const prompt = value as Record<string, unknown>
  if (!isRecipeLike(prompt)) return { kind: "ordinary" }

  try {
    const definition = clone(prompt.structuredPromptDefinition)
    const parsed = parseStructuredPromptDefinitionForTransport(
      definition,
      prompt.promptFormat,
      prompt.promptSchemaVersion
    )
    if (
      parsed?.schema_version !== 2 ||
      parsed.definition_kind !== "single_text_recipe"
    ) {
      return { kind: "quarantined_recipe" }
    }
    const safeDefinition = clone(parsed)
    return {
      kind: "recipe",
      target: safeDefinition.assembly_config.target_role,
      definition: safeDefinition
    }
  } catch {
    return { kind: "quarantined_recipe" }
  }
}

export const matchesRecipeTarget = (
  prompt: unknown,
  filter: RecipeTargetFilter
): boolean => {
  const classification = classifyPromptRecipe(prompt)
  if (classification.kind !== "recipe") return false
  if (filter === "recipe") return true
  return (
    classification.target === (filter === "recipe_system" ? "system" : "user")
  )
}

export const cloneSavedRecipeSource = (prompt: unknown): SavedRecipeSource => {
  const classification = classifyPromptRecipe(prompt)
  if (classification.kind !== "recipe") {
    throw new Error("invalid_saved_recipe")
  }
  const record = prompt as Record<string, unknown>
  const id = record.id
  if (typeof id !== "string" && typeof id !== "number") {
    throw new Error("invalid_saved_recipe")
  }
  const name =
    typeof record.name === "string" && record.name.trim()
      ? record.name
      : typeof record.title === "string" && record.title.trim()
        ? record.title
        : "Untitled recipe"
  return {
    source_kind: "saved",
    id: String(id),
    name,
    definition: clone(classification.definition)
  }
}

export const buildRecipePromptFields = (
  definition: SingleTextRecipeDefinitionV2
) => {
  const parsed = parseStructuredPromptDefinitionForTransport(
    clone(definition),
    "structured",
    2
  )
  if (
    parsed?.schema_version !== 2 ||
    parsed.definition_kind !== "single_text_recipe"
  ) {
    throw new Error("invalid_recipe_definition")
  }
  const canonical = clone(parsed)
  const snapshot = renderSingleTextRecipeTemplate(canonical)
  return {
    promptFormat: "structured" as const,
    promptSchemaVersion: 2 as const,
    structuredPromptDefinition: canonical,
    content: snapshot.rendered_text,
    system_prompt: snapshot.legacy.system_prompt,
    user_prompt: snapshot.legacy.user_prompt,
    is_system: canonical.assembly_config.target_role === "system"
  }
}

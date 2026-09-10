export type MultiMessagePromptDefinitionV1 = {
  schema_version: 1
  format: "structured"
  variables: Array<Record<string, any>>
  blocks: Array<Record<string, any>>
  assembly_config?: Record<string, any>
}

export type SingleTextRecipeDefinitionV2 = {
  schema_version: 2
  format: "structured"
  definition_kind: "single_text_recipe"
  variables: Array<Record<string, any>>
  blocks: Array<Record<string, any>>
  assembly_config: {
    assembly_mode: "single_text"
    target_role: "system" | "user"
    render_format: "xml" | "markdown" | "freeform"
    block_separator?: string
  }
}

export type ParsedStructuredPromptDefinition =
  | MultiMessagePromptDefinitionV1
  | SingleTextRecipeDefinitionV2

// Editor/Dexie callers remain intentionally open until their Task 5/7 model
// migrations. Every sync boundary narrows this transport shape through the
// discriminated parser below before using or persisting it.
export type StructuredPromptDefinition = Record<string, any>

const RUNTIME_VALUE_KEYS = new Set([
  "runtime_values",
  "variable_values",
  "resolved_values"
])

/** Validate untrusted sync/transport JSON before it enters local storage. */
export const parseStructuredPromptDefinitionForTransport = (
  value: unknown
): ParsedStructuredPromptDefinition => {
  const stack: Array<{ value: unknown; depth: number }> = [{ value, depth: 0 }]
  const seen = new WeakSet<object>()
  let nodes = 0
  while (stack.length > 0) {
    const current = stack.pop()!
    nodes += 1
    if (nodes > 10_000 || current.depth > 32) {
      throw new Error("invalid_prompt_definition_structure")
    }
    if (current.value == null || typeof current.value !== "object") continue
    if (seen.has(current.value)) {
      throw new Error("invalid_prompt_definition_structure")
    }
    seen.add(current.value)
    if (Array.isArray(current.value)) {
      for (const item of current.value) {
        stack.push({ value: item, depth: current.depth + 1 })
      }
      continue
    }
    const record = current.value as Record<string, unknown>
    if (Object.keys(record).some((key) => RUNTIME_VALUE_KEYS.has(key))) {
      throw new Error("invalid_recipe_runtime_values")
    }
    for (const item of Object.values(record)) {
      stack.push({ value: item, depth: current.depth + 1 })
    }
  }

  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("invalid_prompt_definition")
  }
  const definition = value as Record<string, unknown>
  const rawVersion = definition.schema_version ?? 1
  const version = rawVersion === 1 || rawVersion === "1" ? 1 : rawVersion
  if (
    definition.format !== "structured" ||
    !Array.isArray(definition.variables) ||
    !Array.isArray(definition.blocks)
  ) {
    throw new Error("invalid_prompt_definition")
  }
  if (version === 1) {
    return {
      ...definition,
      schema_version: 1
    } as MultiMessagePromptDefinitionV1
  }
  const assembly = definition.assembly_config
  if (
    version !== 2 ||
    definition.definition_kind !== "single_text_recipe" ||
    !assembly ||
    typeof assembly !== "object" ||
    Array.isArray(assembly)
  ) {
    throw new Error("invalid_prompt_definition")
  }
  const config = assembly as Record<string, unknown>
  if (
    config.assembly_mode !== "single_text" ||
    !["system", "user"].includes(String(config.target_role)) ||
    !["xml", "markdown", "freeform"].includes(String(config.render_format))
  ) {
    throw new Error("invalid_prompt_definition")
  }
  return definition as SingleTextRecipeDefinitionV2
}

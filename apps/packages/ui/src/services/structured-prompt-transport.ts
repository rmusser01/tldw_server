import {
  extractSingleTextRecipeTemplateVariables,
  renderSingleTextRecipe,
  SINGLE_TEXT_RECIPE_LIMITS,
  singleTextRecipeCodePointLength,
  type SingleTextRecipeDefinitionV2
} from "@/components/Option/Prompt/structured-prompt-utils"

export type { SingleTextRecipeDefinitionV2 }

type PromptVariableDefinitionV1 = {
  name: string
  label: string | null
  description: string | null
  required: boolean
  default_value: unknown
  input_type: string
  options: string[] | null
  max_length: number | null
}

type PromptBlockRoleV1 = "system" | "developer" | "user" | "assistant"

type PromptBlockDefinitionV1 = {
  id: string
  name: string
  role: PromptBlockRoleV1
  kind: string | null
  content: string
  enabled: boolean
  order: number
  is_template: boolean
}

export type MultiMessagePromptDefinitionV1 = {
  schema_version: 1
  format: "structured"
  variables: PromptVariableDefinitionV1[]
  blocks: PromptBlockDefinitionV1[]
  assembly_config: {
    legacy_system_roles: string[]
    legacy_user_roles: string[]
    block_separator: string
  }
}

export type ParsedStructuredPromptDefinition =
  | MultiMessagePromptDefinitionV1
  | SingleTextRecipeDefinitionV2

// Editor/Dexie callers remain open until their Task 5/7 model migrations.
// Transport code still cannot consume these unknown elements before parsing.
export type StructuredPromptDefinition = {
  [key: string]: unknown
  schema_version?: number | string
  format?: string
  variables?: unknown[]
  blocks?: unknown[]
  assembly_config?: unknown
}

const RUNTIME_VALUE_KEYS = new Set([
  "runtime_values",
  "variable_values",
  "resolved_values"
])
const JSON_MAX_DEPTH = 32
const JSON_MAX_NODES = 10_000
const VARIABLE_NAME_PATTERN = /^[A-Za-z0-9_]+$/
const XML_SECTION_KEY_PATTERN = /^[A-Za-z_][A-Za-z0-9_.-]*$/
// Python v1 accepts wider integers, but sync must round-trip exactly through JSON/JS.
const MAX_SAFE_INTEGER_TEXT = String(Number.MAX_SAFE_INTEGER)
// Pydantic integer whitespace: includes NEL, unlike JS trim; excludes BOM and FS–US.
const PYDANTIC_INTEGER_EDGE_WHITESPACE =
  /^[\t\n\v\f\r \u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+|[\t\n\v\f\r \u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+$/g
// Python str.strip() whitespace parity for IDs used by backend duplicate checks.
const PYTHON_EDGE_WHITESPACE =
  /^[\t\n\v\f\r \u001c-\u001f\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+|[\t\n\v\f\r \u001c-\u001f\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+$/g

const pythonStrip = (value: string): string =>
  value.replace(PYTHON_EDGE_WHITESPACE, "")

const fail = (code = "invalid_prompt_definition"): never => {
  throw new Error(code)
}

const isRecord = (value: unknown): value is Record<string, unknown> => {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    return false
  }
  const prototype = Object.getPrototypeOf(value)
  return prototype === Object.prototype || prototype === null
}

const requireRecord = (value: unknown): Record<string, unknown> =>
  isRecord(value) ? value : fail()

const requireArray = (value: unknown): unknown[] =>
  Array.isArray(value) ? value : fail()

const requireAllowedKeys = (
  value: Record<string, unknown>,
  allowed: ReadonlySet<string>
): void => {
  if (Object.keys(value).some((key) => !allowed.has(key))) fail()
}

const requireString = (
  value: unknown,
  options: { nullable?: boolean; min?: number; max?: number } = {}
): string | null => {
  if (value === null && options.nullable) return null
  if (typeof value !== "string") return fail()
  const length = singleTextRecipeCodePointLength(value)
  if (
    (options.min !== undefined && length < options.min) ||
    (options.max !== undefined && length > options.max)
  ) {
    fail()
  }
  return value
}

const requireOptionalString = (
  value: unknown,
  options: { nullable?: boolean; min?: number; max?: number } = {}
): void => {
  if (value !== undefined) requireString(value, options)
}

const requireOptionalBoolean = (value: unknown): void => {
  if (value !== undefined && typeof value !== "boolean") fail()
}

const requireInteger = (value: unknown, minimum?: number): number => {
  if (
    typeof value !== "number" ||
    !Number.isFinite(value) ||
    !Number.isInteger(value)
  ) {
    return fail()
  }
  if (minimum !== undefined && value < minimum) return fail()
  return value
}

const coerceV1Boolean = (value: unknown, defaultValue: boolean): boolean => {
  if (value === undefined) return defaultValue
  if (typeof value === "boolean") return value
  if (typeof value === "number" && Number.isFinite(value)) {
    if (value === 0) return false
    if (value === 1) return true
    return fail()
  }
  if (typeof value === "string") {
    const normalized = value.toLowerCase()
    if (["0", "off", "f", "false", "n", "no"].includes(normalized)) {
      return false
    }
    if (["1", "on", "t", "true", "y", "yes"].includes(normalized)) {
      return true
    }
  }
  return fail()
}

const coerceV1Integer = (value: unknown, nullable = false): number | null => {
  if (value === null && nullable) return null
  if (typeof value === "boolean") return value ? 1 : 0
  if (typeof value === "number") {
    if (!Number.isSafeInteger(value)) return fail()
    return value === 0 ? 0 : value
  }
  if (typeof value === "string") {
    const normalized = value.replace(PYDANTIC_INTEGER_EDGE_WHITESPACE, "")
    if (!/^[+-]?\d(?:_?\d)*(?:\.0+)?$/.test(normalized)) return fail()
    const integerText = normalized.replace(/_/g, "").replace(/\.0+$/, "")
    const digits = integerText.replace(/^[+-]?0*/, "") || "0"
    if (
      digits.length > MAX_SAFE_INTEGER_TEXT.length ||
      (digits.length === MAX_SAFE_INTEGER_TEXT.length &&
        digits > MAX_SAFE_INTEGER_TEXT)
    ) {
      return fail()
    }
    const parsed = Number(integerText)
    if (!Number.isSafeInteger(parsed)) return fail()
    return parsed === 0 ? 0 : parsed
  }
  return fail()
}

const requireV1Role = (value: unknown): PromptBlockRoleV1 => {
  switch (value) {
    case "system":
    case "developer":
    case "user":
    case "assistant":
      return value
    default:
      return fail()
  }
}

const v1NullableString = (value: unknown): string | null =>
  value === undefined ? null : requireString(value, { nullable: true })

const v1StringArray = (value: unknown, fallback: string[]): string[] => {
  if (value === undefined) return [...fallback]
  if (!Array.isArray(value) || value.some((item) => typeof item !== "string")) {
    return fail()
  }
  return value.map((item) => String(item))
}

const requireOptionalStringArray = (value: unknown, nullable = true): void => {
  if (value === undefined) return
  if (value === null) {
    if (nullable) return
    fail()
  }
  if (!Array.isArray(value) || value.some((item) => typeof item !== "string")) {
    fail()
  }
}

const assertBoundedJsonTree = (value: unknown): void => {
  const stack: Array<{ value: unknown; depth: number }> = [{ value, depth: 0 }]
  const seen = new WeakSet<object>()
  let nodes = 0
  while (stack.length > 0) {
    const current = stack.pop()!
    nodes += 1
    if (nodes > JSON_MAX_NODES || current.depth > JSON_MAX_DEPTH) {
      fail("invalid_prompt_definition_structure")
    }
    const item = current.value
    if (
      item === null ||
      typeof item === "string" ||
      typeof item === "boolean"
    ) {
      continue
    }
    if (typeof item === "number" && Number.isFinite(item)) continue
    if (typeof item !== "object") return fail()
    if (seen.has(item)) fail("invalid_prompt_definition_structure")
    seen.add(item)
    if (Array.isArray(item)) {
      for (const child of item) {
        stack.push({ value: child, depth: current.depth + 1 })
      }
      continue
    }
    if (!isRecord(item)) return fail()
    if (Object.keys(item).some((key) => RUNTIME_VALUE_KEYS.has(key))) {
      fail("invalid_recipe_runtime_values")
    }
    for (const child of Object.values(item)) {
      stack.push({ value: child, depth: current.depth + 1 })
    }
  }
}

const validateVariable = (value: unknown, recipe: boolean): string => {
  const variable = requireRecord(value)
  requireAllowedKeys(
    variable,
    new Set([
      "name",
      "label",
      "description",
      "required",
      "default_value",
      "input_type",
      "options",
      "max_length"
    ])
  )
  const name = requireString(variable.name, {
    min: 1,
    max: recipe ? SINGLE_TEXT_RECIPE_LIMITS.max_key_length : undefined
  })!
  if (recipe && !VARIABLE_NAME_PATTERN.test(name)) fail()
  requireOptionalString(variable.label, {
    nullable: true,
    max: recipe ? SINGLE_TEXT_RECIPE_LIMITS.max_label_length : undefined
  })
  requireOptionalString(variable.description, { nullable: true })
  requireOptionalBoolean(variable.required)
  requireOptionalString(variable.input_type)
  requireOptionalStringArray(variable.options)
  if (variable.max_length !== undefined && variable.max_length !== null) {
    requireInteger(variable.max_length, 1)
  }
  return name
}

const validateV1 = (
  definition: Record<string, unknown>
): MultiMessagePromptDefinitionV1 => {
  requireAllowedKeys(
    definition,
    new Set([
      "schema_version",
      "format",
      "variables",
      "blocks",
      "assembly_config"
    ])
  )
  if (definition.format !== undefined && definition.format !== "structured") {
    fail()
  }
  const variables =
    definition.variables === undefined ? [] : requireArray(definition.variables)
  const blocks =
    definition.blocks === undefined ? [] : requireArray(definition.blocks)
  const declared = new Set<string>()
  const parsedVariables: PromptVariableDefinitionV1[] = []
  for (const value of variables) {
    const variable = requireRecord(value)
    requireAllowedKeys(
      variable,
      new Set([
        "name",
        "label",
        "description",
        "required",
        "default_value",
        "input_type",
        "options",
        "max_length"
      ])
    )
    const name = requireString(variable.name, { min: 1 })!
    const normalizedName = pythonStrip(name)
    if (normalizedName && declared.has(normalizedName)) {
      fail("duplicate_variable_name")
    }
    if (normalizedName) declared.add(normalizedName)
    const options = variable.options
    if (
      options !== undefined &&
      options !== null &&
      (!Array.isArray(options) ||
        options.some((item) => typeof item !== "string"))
    ) {
      fail()
    }
    const maxLength =
      variable.max_length === undefined
        ? null
        : coerceV1Integer(variable.max_length, true)
    if (maxLength !== null && maxLength < 1) fail()
    parsedVariables.push({
      name,
      label: v1NullableString(variable.label),
      description: v1NullableString(variable.description),
      required: coerceV1Boolean(variable.required, false),
      default_value:
        variable.default_value === undefined
          ? null
          : structuredClone(variable.default_value),
      input_type:
        variable.input_type === undefined
          ? "text"
          : requireString(variable.input_type)!,
      options: Array.isArray(options)
        ? options.map((item) => String(item))
        : null,
      max_length: maxLength
    })
  }

  const blockIds = new Set<string>()
  const parsedBlocks: PromptBlockDefinitionV1[] = []
  for (const value of blocks) {
    const block = requireRecord(value)
    requireAllowedKeys(
      block,
      new Set([
        "id",
        "name",
        "role",
        "kind",
        "content",
        "enabled",
        "order",
        "is_template"
      ])
    )
    const id = requireString(block.id, { min: 1 })!
    const name = requireString(block.name, { min: 1 })!
    const role = requireV1Role(block.role)
    const kind = v1NullableString(block.kind)
    const content = requireString(block.content)!
    const enabled = coerceV1Boolean(block.enabled, true)
    const order = coerceV1Integer(block.order)
    const isTemplate = coerceV1Boolean(block.is_template, false)
    const normalizedId = pythonStrip(id)
    if (normalizedId && blockIds.has(normalizedId)) fail("duplicate_block_id")
    if (normalizedId) blockIds.add(normalizedId)
    if (isTemplate) {
      for (const name of extractSingleTextRecipeTemplateVariables(content)) {
        if (!declared.has(name)) fail("unknown_variable_reference")
      }
    }
    parsedBlocks.push({
      id,
      name,
      role,
      kind,
      content,
      enabled,
      order: order!,
      is_template: isTemplate
    })
  }

  let legacySystemRoles = ["system", "developer"]
  let legacyUserRoles = ["user"]
  let blockSeparator = "\n\n"
  if (definition.assembly_config !== undefined) {
    const config = requireRecord(definition.assembly_config)
    requireAllowedKeys(
      config,
      new Set(["legacy_system_roles", "legacy_user_roles", "block_separator"])
    )
    legacySystemRoles = v1StringArray(
      config.legacy_system_roles,
      legacySystemRoles
    )
    legacyUserRoles = v1StringArray(config.legacy_user_roles, legacyUserRoles)
    blockSeparator =
      config.block_separator === undefined
        ? blockSeparator
        : requireString(config.block_separator)!
  }
  return {
    schema_version: 1,
    format: "structured",
    variables: parsedVariables,
    blocks: parsedBlocks,
    assembly_config: {
      legacy_system_roles: legacySystemRoles,
      legacy_user_roles: legacyUserRoles,
      block_separator: blockSeparator
    }
  }
}

const validateV2 = (
  definition: Record<string, unknown>
): SingleTextRecipeDefinitionV2 => {
  requireAllowedKeys(
    definition,
    new Set([
      "schema_version",
      "format",
      "definition_kind",
      "variables",
      "blocks",
      "assembly_config"
    ])
  )
  if (
    definition.format !== "structured" ||
    definition.definition_kind !== "single_text_recipe"
  ) {
    fail()
  }
  const variables =
    definition.variables === undefined ? [] : definition.variables
  const blocks = definition.blocks === undefined ? [] : definition.blocks
  if (!Array.isArray(variables)) return fail()
  if (variables.length > SINGLE_TEXT_RECIPE_LIMITS.max_variables) fail()
  if (!Array.isArray(blocks)) return fail()
  if (blocks.length > SINGLE_TEXT_RECIPE_LIMITS.max_blocks) fail()

  const declared = new Set<string>()
  const variableNames: string[] = []
  for (const value of variables) {
    const name = validateVariable(value, true)
    if (declared.has(name)) fail("duplicate_variable_name")
    declared.add(name)
    variableNames.push(name)
  }

  const config = requireRecord(definition.assembly_config)
  requireAllowedKeys(
    config,
    new Set([
      "assembly_mode",
      "target_role",
      "render_format",
      "block_separator"
    ])
  )
  if (config.assembly_mode !== "single_text") fail()
  if (config.target_role !== "system" && config.target_role !== "user") fail()
  if (!["xml", "markdown", "freeform"].includes(String(config.render_format))) {
    fail()
  }
  requireOptionalString(config.block_separator, {
    max: SINGLE_TEXT_RECIPE_LIMITS.max_separator_length
  })

  const blockIds = new Set<string>()
  const xmlKeys = new Set<string>()
  for (const value of blocks) {
    const block = requireRecord(value)
    requireAllowedKeys(
      block,
      new Set([
        "id",
        "name",
        "role",
        "kind",
        "content",
        "enabled",
        "order",
        "is_template",
        "section_key"
      ])
    )
    const id = requireString(block.id, {
      min: 1,
      max: SINGLE_TEXT_RECIPE_LIMITS.max_key_length
    })!
    const normalizedId = pythonStrip(id)
    if (!normalizedId) fail("invalid_block_id")
    if (blockIds.has(normalizedId)) fail("duplicate_block_id")
    blockIds.add(normalizedId)
    requireString(block.name, {
      min: 1,
      max: SINGLE_TEXT_RECIPE_LIMITS.max_label_length
    })
    if (block.role !== config.target_role) fail("invalid_block_role")
    requireOptionalString(block.kind, { nullable: true })
    const content = requireString(block.content, {
      max: SINGLE_TEXT_RECIPE_LIMITS.max_content_length
    })!
    requireOptionalBoolean(block.enabled)
    const order = requireInteger(block.order)
    if (Math.abs(order) > SINGLE_TEXT_RECIPE_LIMITS.max_abs_order) fail()
    requireOptionalBoolean(block.is_template)
    requireOptionalString(block.section_key, {
      nullable: true,
      max: SINGLE_TEXT_RECIPE_LIMITS.max_key_length
    })
    if (block.is_template === true) {
      for (const name of extractSingleTextRecipeTemplateVariables(content)) {
        if (!declared.has(name)) fail("unknown_variable_reference")
      }
    }
    if (config.render_format === "xml" && block.enabled !== false) {
      const sectionKey = block.section_key
      if (
        typeof sectionKey !== "string" ||
        !XML_SECTION_KEY_PATTERN.test(sectionKey)
      ) {
        return fail("invalid_section_key")
      }
      if (xmlKeys.has(sectionKey)) fail("duplicate_section_key")
      xmlKeys.add(sectionKey)
    }
  }

  const parsed = definition as SingleTextRecipeDefinitionV2
  const safeValues = Object.fromEntries(
    variableNames.map((name) => [name, "x"])
  )
  renderSingleTextRecipe(parsed, safeValues)
  renderSingleTextRecipe(
    {
      ...parsed,
      variables: [],
      blocks: (parsed.blocks ?? []).map((block) => ({
        ...block,
        is_template: false
      }))
    },
    null
  )
  return parsed
}

/**
 * Validate an untrusted definition together with its outer persisted identity.
 * Legacy identity returns null; structured identity returns the narrowed v1/v2 union.
 */
export const parseStructuredPromptDefinitionForTransport = (
  value: unknown,
  promptFormat: unknown,
  promptSchemaVersion: unknown
): ParsedStructuredPromptDefinition | null => {
  if (value !== undefined) assertBoundedJsonTree(value)
  if (
    promptFormat === undefined ||
    promptFormat === null ||
    promptFormat === "legacy"
  ) {
    if (value !== undefined && value !== null) fail()
    if (promptSchemaVersion !== undefined && promptSchemaVersion !== null)
      fail()
    return null
  }
  if (promptFormat !== "structured") fail()
  if (promptSchemaVersion !== 1 && promptSchemaVersion !== 2) {
    fail("unsupported_schema_version")
  }
  const definition = requireRecord(value)
  const rawVersion =
    definition.schema_version === undefined ? 1 : definition.schema_version
  if (promptSchemaVersion === 1) {
    if (coerceV1Integer(rawVersion) !== 1) fail()
    return validateV1(definition)
  }
  if (rawVersion !== 2) fail()
  if (promptSchemaVersion === 2) return validateV2(definition)
  return fail("unsupported_schema_version")
}

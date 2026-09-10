import type { StructuredPromptDefinition } from "@/services/prompt-studio"

type StructuredPromptBlock = {
  id: string
  name: string
  role: "system" | "developer" | "user" | "assistant"
  kind?: string
  content: string
  enabled: boolean
  order: number
  is_template: boolean
}

type StructuredPromptVariable = {
  name: string
  label?: string
  description?: string
  required?: boolean
  input_type?: string
}

type StructuredPromptAssemblyConfig = {
  legacy_system_roles: string[]
  legacy_user_roles: string[]
  block_separator: string
}

const LEGACY_TEMPLATE_PATTERN =
  /{{\s*([a-zA-Z0-9_]+)\s*}}|\{([a-zA-Z0-9_]+)\}|\$([a-zA-Z0-9_]+)|<([a-zA-Z0-9_]+)>/g

const DEFAULT_ASSEMBLY_CONFIG: StructuredPromptAssemblyConfig = {
  legacy_system_roles: ["system", "developer"],
  legacy_user_roles: ["user"],
  block_separator: "\n\n"
}

const matchVariableName = (
  groups: Array<string | undefined>
): string => {
  for (const group of groups) {
    if (group) return group
  }
  return ""
}

export const extractLegacyPromptVariables = (
  ...templates: Array<string | null | undefined>
): string[] => {
  const variables: string[] = []

  for (const template of templates) {
    LEGACY_TEMPLATE_PATTERN.lastIndex = 0
    let match = LEGACY_TEMPLATE_PATTERN.exec(template || "")
    while (match) {
      const variableName = matchVariableName(match.slice(1))
      if (variableName && !variables.includes(variableName)) {
        variables.push(variableName)
      }
      match = LEGACY_TEMPLATE_PATTERN.exec(template || "")
    }
  }

  return variables
}

export const normalizeLegacyPromptTemplate = (
  template?: string | null
): string => {
  if (!template) return ""

  LEGACY_TEMPLATE_PATTERN.lastIndex = 0
  return template.replace(LEGACY_TEMPLATE_PATTERN, (...args) => {
    const groups = args.slice(1, 5) as Array<string | undefined>
    return `{{${matchVariableName(groups)}}}`
  })
}

export const createDefaultStructuredPromptDefinition =
  (): StructuredPromptDefinition => ({
    schema_version: 1,
    format: "structured",
    assembly_config: { ...DEFAULT_ASSEMBLY_CONFIG },
    variables: [],
    blocks: [
      {
        id: "task",
        name: "Task",
        role: "user",
        kind: "task",
        content: "Describe the task here.",
        enabled: true,
        order: 10,
        is_template: false
      }
    ]
  })

export const createStructuredPromptDefinition = ({
  variables = [],
  blocks
}: {
  variables?: StructuredPromptVariable[]
  blocks: Array<
    Omit<StructuredPromptBlock, "order" | "enabled" | "is_template"> &
      Partial<Pick<StructuredPromptBlock, "order" | "enabled" | "is_template">>
  >
}): StructuredPromptDefinition => ({
  schema_version: 1,
  format: "structured",
  assembly_config: { ...DEFAULT_ASSEMBLY_CONFIG },
  variables: variables.map((variable) => ({
    name: variable.name,
    label: variable.label,
    description: variable.description,
    required: variable.required === true,
    input_type: variable.input_type || "text"
  })),
  blocks: blocks.map((block, index) => ({
    ...block,
    enabled: block.enabled !== false,
    order:
      typeof block.order === "number" && Number.isFinite(block.order)
        ? block.order
        : (index + 1) * 10,
    is_template: block.is_template === true
  }))
})

export const convertLegacyPromptToStructuredDefinition = (
  systemPrompt?: string | null,
  userPrompt?: string | null
): StructuredPromptDefinition => {
  const variables = extractLegacyPromptVariables(systemPrompt, userPrompt)
  const blocks: StructuredPromptBlock[] = []
  const normalizedSystemPrompt = normalizeLegacyPromptTemplate(systemPrompt)
  const normalizedUserPrompt = normalizeLegacyPromptTemplate(userPrompt)

  if (normalizedSystemPrompt.trim()) {
    blocks.push({
      id: "legacy_system",
      name: "System Instructions",
      role: "system",
      kind: "instructions",
      content: normalizedSystemPrompt.trim(),
      enabled: true,
      order: 10,
      is_template: normalizedSystemPrompt.includes("{{")
    })
  }

  if (normalizedUserPrompt.trim()) {
    blocks.push({
      id: "legacy_user",
      name: "User Prompt",
      role: "user",
      kind: "task",
      content: normalizedUserPrompt.trim(),
      enabled: true,
      order: blocks.length === 0 ? 10 : 20,
      is_template: normalizedUserPrompt.includes("{{")
    })
  }

  return {
    schema_version: 1,
    format: "structured",
    assembly_config: { ...DEFAULT_ASSEMBLY_CONFIG },
    variables: variables.map((variableName) => ({
      name: variableName,
      label: variableName.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase()),
      required: true,
      input_type: "textarea"
    })),
    blocks:
      blocks.length > 0
        ? blocks
        : (createDefaultStructuredPromptDefinition().blocks as StructuredPromptBlock[])
  }
}

export const renderStructuredPromptLegacySnapshot = (
  definition: StructuredPromptDefinition | null | undefined
): {
  systemPrompt: string
  userPrompt: string
  content: string
} => {
  if (definition?.schema_version === 2) {
    const result = renderSingleTextRecipeTemplate(
      definition as SingleTextRecipeDefinitionV2
    )
    return {
      systemPrompt: result.legacy.system_prompt,
      userPrompt: result.legacy.user_prompt,
      content: result.rendered_text
    }
  }
  const blocks = Array.isArray(definition?.blocks)
    ? [...definition.blocks]
        .filter((block: any) => block?.enabled !== false)
        .sort((left: any, right: any) => {
          const leftOrder =
            typeof left?.order === "number" && Number.isFinite(left.order)
              ? left.order
              : 0
          const rightOrder =
            typeof right?.order === "number" && Number.isFinite(right.order)
              ? right.order
              : 0
          return leftOrder - rightOrder
        })
    : []

  const assemblyConfig = (
    definition?.assembly_config &&
    typeof definition.assembly_config === "object" &&
    !Array.isArray(definition.assembly_config)
      ? definition.assembly_config
      : DEFAULT_ASSEMBLY_CONFIG
  ) as Partial<StructuredPromptAssemblyConfig>
  const separator =
    typeof assemblyConfig.block_separator === "string"
      ? assemblyConfig.block_separator
      : DEFAULT_ASSEMBLY_CONFIG.block_separator
  const systemRoles = Array.isArray(assemblyConfig.legacy_system_roles)
    ? assemblyConfig.legacy_system_roles.map((role) => String(role))
    : DEFAULT_ASSEMBLY_CONFIG.legacy_system_roles
  const userRoles = Array.isArray(assemblyConfig.legacy_user_roles)
    ? assemblyConfig.legacy_user_roles.map((role) => String(role))
    : DEFAULT_ASSEMBLY_CONFIG.legacy_user_roles

  const pickContent = (roles: string[]) =>
    blocks
      .filter((block: any) => roles.includes(String(block?.role || "")))
      .map((block: any) =>
        typeof block?.content === "string" ? block.content.trim() : ""
      )
      .filter((content: string) => content.length > 0)
      .join(separator)

  const systemPrompt = pickContent(systemRoles)
  const userPrompt = pickContent(userRoles)

  return {
    systemPrompt,
    userPrompt,
    content: userPrompt || systemPrompt || ""
  }
}

const normalizeForStableComparison = (value: unknown): unknown => {
  if (Array.isArray(value)) {
    return value.map((item) => normalizeForStableComparison(item))
  }

  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>)
        .filter(([, item]) => item !== undefined)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, item]) => [key, normalizeForStableComparison(item)])
    )
  }

  return value
}

export const stableSerializePromptSnapshot = (value: unknown): string =>
  JSON.stringify(normalizeForStableComparison(value)) ?? ""

/** Renderer input; unknown stored JSON must be validated before use. */
export type SingleTextRecipeDefinitionV2 = {
  schema_version: 2
  format: "structured"
  definition_kind: "single_text_recipe"
  assembly_config: {
    assembly_mode: "single_text"
    target_role: "system" | "user"
    render_format: "xml" | "markdown" | "freeform"
    block_separator?: string
  }
  variables?: Array<Omit<StructuredPromptVariable, "label" | "description"> & {
    label?: string | null
    description?: string | null
    default_value?: unknown
    options?: string[] | null
    max_length?: number | null
  }>
  blocks?: Array<Omit<StructuredPromptBlock, "enabled" | "is_template" | "kind" | "role"> & {
    role: "system" | "user"
    kind?: string | null
    section_key?: string | null
    enabled?: boolean
    is_template?: boolean
  }>
}

export type SingleTextRecipeRenderResult = {
  definition_kind: "single_text_recipe"
  rendered_text: string
  legacy: { system_prompt: string; user_prompt: string }
}

// Shared transport/renderer mirror of the backend's SINGLE_TEXT_RECIPE_LIMITS.
export const SINGLE_TEXT_RECIPE_LIMITS = Object.freeze({
  max_blocks: 100,
  max_variables: 100,
  max_label_length: 200,
  max_key_length: 128,
  max_content_length: 20000,
  max_separator_length: 20,
  max_abs_order: 9007199254740991,
  max_rendered_output_length: 100000
})

// Python str/regex whitespace, including U+001C–001F and U+0085, excluding BOM.
const RECIPE_TEMPLATE_PATTERN =
  // eslint-disable-next-line no-control-regex -- Python whitespace intentionally includes these four control separators.
  /{{[\t\n\v\f\r \u001c-\u001f\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]*([a-zA-Z0-9_]+)[\t\n\v\f\r \u001c-\u001f\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]*}}/g

export class SingleTextRecipeRenderError extends Error {
  constructor(public readonly code: string, public readonly variable_name?: string) {
    super(code)
    this.name = "SingleTextRecipeRenderError"
  }
}

export const singleTextRecipeCodePointLength = (text: string): number => {
  let length = 0
  // Iterate without allocating an array of characters for oversized inputs.
  for (const _character of text) length += 1
  return length
}

export const extractSingleTextRecipeTemplateVariables = (
  text: string
): string[] =>
  Array.from(
    text.matchAll(new RegExp(RECIPE_TEMPLATE_PATTERN.source, "g")),
    (match) => match[1]
  )

/**
 * Compile a prevalidated v2 definition to one string, never a message array.
 * Null/omitted defaults are absent. Selected runtime/default values must be
 * strings (including an explicit empty string). Unknown runtime keys are ignored.
 */
export const renderSingleTextRecipe = (
  definition: SingleTextRecipeDefinitionV2,
  runtimeValues: Readonly<Record<string, unknown>> | null = null
): SingleTextRecipeRenderResult => {
  if (definition.schema_version !== 2) {
    throw new SingleTextRecipeRenderError("unsupported_schema_version")
  }
  const declarations = definition.variables ?? []
  const declaredNames = new Set(declarations.map((variable) => variable.name))
  const blocks = definition.blocks ?? []
  // Guard even disabled blocks: the persisted v2 order domain is lossless in JS.
  for (const block of blocks) {
    if (!Number.isInteger(block.order)) {
      throw new SingleTextRecipeRenderError("int_type")
    }
    if (block.order > SINGLE_TEXT_RECIPE_LIMITS.max_abs_order) {
      throw new SingleTextRecipeRenderError("less_than_equal")
    }
    if (block.order < -SINGLE_TEXT_RECIPE_LIMITS.max_abs_order) {
      throw new SingleTextRecipeRenderError("greater_than_equal")
    }
  }
  // Match backend validation: disabled templates still require declared names.
  for (const block of blocks) {
    if (!block.is_template) continue
    for (const variableName of extractSingleTextRecipeTemplateVariables(block.content)) {
      if (!declaredNames.has(variableName)) {
        throw new SingleTextRecipeRenderError("unknown_variable_reference", variableName)
      }
    }
  }

  const resolved = new Map<string, string>()
  for (const variable of declarations) {
    let value: unknown
    if (runtimeValues && Object.prototype.hasOwnProperty.call(runtimeValues, variable.name)) {
      value = runtimeValues[variable.name]
    } else if (variable.default_value != null) {
      value = variable.default_value
    } else if (variable.required) {
      throw new SingleTextRecipeRenderError("missing_required_variable", variable.name)
    } else {
      value = ""
    }
    if (typeof value !== "string") {
      throw new SingleTextRecipeRenderError("invalid_variable_value", variable.name)
    }
    resolved.set(variable.name, value)
  }

  const config = definition.assembly_config
  const separator = config.block_separator ?? "\n\n"
  const separatorLength = singleTextRecipeCodePointLength(separator)
  const valueLengths = new Map<string, number>()
  const rendered: string[] = []
  let outputLength = 0
  const ordered = blocks.map((block, index) => ({ block, index }))
    .sort((left, right) => left.block.order - right.block.order || left.index - right.index)
  for (const { block } of ordered) {
    if (block.enabled === false) continue
    let contentLength = singleTextRecipeCodePointLength(block.content)
    if (block.is_template) {
      for (const match of block.content.matchAll(RECIPE_TEMPLATE_PATTERN)) {
        if (!valueLengths.has(match[1])) {
          valueLengths.set(match[1], singleTextRecipeCodePointLength(resolved.get(match[1])!))
        }
        contentLength += valueLengths.get(match[1])! - singleTextRecipeCodePointLength(match[0])
      }
    }
    let prefix = ""
    let suffix = ""
    if (config.render_format === "xml") {
      prefix = `<${block.section_key}>`
      suffix = `</${block.section_key}>`
    } else if (config.render_format === "markdown") {
      prefix = `## ${block.name}\n\n`
    }
    outputLength += contentLength + singleTextRecipeCodePointLength(prefix) + singleTextRecipeCodePointLength(suffix)
      + (rendered.length ? separatorLength : 0)
    if (outputLength > SINGLE_TEXT_RECIPE_LIMITS.max_rendered_output_length) {
      throw new SingleTextRecipeRenderError("rendered_output_too_large")
    }
    // Only allocate substituted content once its exact expansion fits the budget.
    const content = block.is_template
      ? block.content.replace(RECIPE_TEMPLATE_PATTERN, (_, name: string) => resolved.get(name)!)
      : block.content
    if (config.render_format === "xml" && content.includes(suffix)) {
      throw new SingleTextRecipeRenderError("closing_tag_collision")
    }
    rendered.push(prefix + content + suffix)
  }
  const text = rendered.join(separator)
  return {
    definition_kind: "single_text_recipe",
    rendered_text: text,
    legacy: {
      system_prompt: config.target_role === "system" ? text : "",
      user_prompt: config.target_role === "user" ? text : ""
    }
  }
}

/** Compile the authored recipe template for storage without resolving defaults. */
export const renderSingleTextRecipeTemplate = (
  definition: SingleTextRecipeDefinitionV2
): SingleTextRecipeRenderResult =>
  renderSingleTextRecipe({
    ...definition,
    variables: [],
    blocks: (definition.blocks ?? []).map((block) => ({
      ...block,
      is_template: false
    }))
  })

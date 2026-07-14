import type { SkillResponse } from "@/types/skill"
import { DEFAULT_SCHEMA, load as parseYaml, Type } from "js-yaml"

const SUPPORTING_FILE_NAME_REGEX = /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,99}$/
export const SKILL_NAME_REGEX = /^[a-z][a-z0-9-]{0,63}$/
export const MAX_SKILLS_BULK_SELECTION = 100

const PY_YAML_COMPATIBLE_SCHEMA = DEFAULT_SCHEMA.extend({
  implicit: [
    new Type("tag:tldw.ai,2026:pyyaml-bool", {
      kind: "scalar",
      resolve: (value) =>
        typeof value === "string" && /^(?:yes|no|on|off)$/i.test(value),
      construct: (value) => /^(?:yes|on)$/i.test(String(value))
    })
  ]
})

export const limitSkillSelection = (requestedNames: string[]): {
  names: string[]
  limited: boolean
} => {
  const uniqueNames = Array.from(new Set(requestedNames))
  return {
    names: uniqueNames.slice(0, MAX_SKILLS_BULK_SELECTION),
    limited: uniqueNames.length > MAX_SKILLS_BULK_SELECTION
  }
}

export interface SupportingFileFormEntry {
  filename?: string
  content?: string
  originalFilename?: string
}

export const SKILL_TEMPLATE_IDS = ["summarizer", "explainer", "extractor", "blank"] as const

export type SkillTemplateId = (typeof SKILL_TEMPLATE_IDS)[number]

export interface SkillTemplateOption {
  id: SkillTemplateId
  label: string
  description: string
  defaultName: string
  argumentHint: string
  body: string
}

export interface SkillGuidedDraft {
  name: string
  description: string
  argumentHint: string
  instructions: string
  context: "inline" | "fork"
  userInvocable: boolean
  allowModelInvocation: boolean
  model: string
  allowedTools: string
}

const quoteYamlString = (value: string): string => JSON.stringify(value)

const pushYamlLine = (lines: string[], key: string, value: string | null | undefined): void => {
  if (typeof value !== "string") return
  const trimmed = value.trim()
  if (!trimmed) return
  lines.push(`${key}: ${quoteYamlString(trimmed)}`)
}

const serializeSkillFrontmatter = (skill: SkillResponse): string => {
  const lines: string[] = [`name: ${quoteYamlString(skill.name)}`]

  pushYamlLine(lines, "description", skill.description)
  pushYamlLine(lines, "argument-hint", skill.argument_hint)

  if (skill.disable_model_invocation) {
    lines.push("disable-model-invocation: true")
  }
  if (!skill.user_invocable) {
    lines.push("user-invocable: false")
  }
  if (skill.allowed_tools && skill.allowed_tools.length > 0) {
    lines.push(`allowed-tools: ${quoteYamlString(skill.allowed_tools.join(", "))}`)
  }
  pushYamlLine(lines, "model", skill.model)

  if (skill.context === "fork") {
    lines.push("context: fork")
  }

  return lines.join("\n")
}

export const buildInitialSkillContent = (skill: SkillResponse): string => {
  if (skill.raw_content && skill.raw_content.trim()) {
    return skill.raw_content
  }
  const frontmatter = serializeSkillFrontmatter(skill)
  const body = skill.content || ""
  return `---\n${frontmatter}\n---\n\n${body}`
}

export const buildDuplicateSkillContent = (
  skill: SkillResponse,
  duplicateName: string
): string => {
  const content = buildInitialSkillContent(skill)
  const newline = content.includes("\r\n") ? "\r\n" : "\n"
  const lines = content.split(/\r?\n/)

  if (lines[0]?.trim() !== "---") return content

  const closingIndex = lines.findIndex(
    (line, index) => index > 0 && line.trim() === "---"
  )
  if (closingIndex < 0) return content

  const nameIndex = lines.findIndex(
    (line, index) =>
      index > 0
      && index < closingIndex
      && /^(?:name|'name'|"name")\s*:/.test(line)
  )
  const nameLine = `name: ${quoteYamlString(duplicateName)}`
  if (nameIndex >= 0) lines[nameIndex] = nameLine
  else lines.splice(1, 0, nameLine)

  return lines.join(newline)
}

export const SKILL_TEMPLATE_OPTIONS: SkillTemplateOption[] = [
  {
    id: "summarizer",
    label: "Summarizer",
    description: "Condense source material into a short, useful answer.",
    defaultName: "summarizer-skill",
    argumentHint: "[brief|medium|detailed] [text or topic]",
    body: `Summarize the following source material.

Input:
$ARGUMENTS

Return:
- A concise summary at the requested depth
- Key points the user should keep
- Any uncertainty or missing context`
  },
  {
    id: "explainer",
    label: "Explainer",
    description: "Teach a concept step by step for a specific audience.",
    defaultName: "explainer-skill",
    argumentHint: "[concept] [audience or current understanding]",
    body: `Explain the following concept.

Topic and audience:
$ARGUMENTS

Return:
- A plain-language explanation
- One concrete example
- A quick check for understanding`
  },
  {
    id: "extractor",
    label: "Extractor",
    description: "Pull structured facts or fields from messy input.",
    defaultName: "extractor-skill",
    argumentHint: "[fields to extract] [source text]",
    body: `Extract structured information from the following input.

Extraction request:
$ARGUMENTS

Return:
- The requested fields in a clear list or table
- Missing fields marked as unknown
- Source snippets when they help verify the extraction`
  },
  {
    id: "blank",
    label: "Blank",
    description: "Start from valid frontmatter and write custom instructions.",
    defaultName: "blank-skill",
    argumentHint: "[input]",
    body: `Write the instructions for this skill.

Input:
$ARGUMENTS

Return the result the user should receive.`
  }
]

const getSkillTemplateOption = (templateId: SkillTemplateId): SkillTemplateOption =>
  SKILL_TEMPLATE_OPTIONS.find((template) => template.id === templateId) ??
  SKILL_TEMPLATE_OPTIONS[0]

const normalizeSkillTemplateName = (name: string, fallback: string): string => {
  const normalized = name
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")

  if (!normalized) {
    return fallback
  }

  const validStart = /^[a-z]/.test(normalized) ? normalized : `skill-${normalized}`
  return validStart.slice(0, 64).replace(/-+$/g, "") || fallback
}

export const buildGuidedDraftFromTemplate = (
  templateId: SkillTemplateId,
  name: string
): SkillGuidedDraft => {
  const template = getSkillTemplateOption(templateId)
  return {
    name: normalizeSkillTemplateName(name, template.defaultName),
    description: template.description,
    argumentHint: template.argumentHint,
    instructions: template.body,
    context: "inline",
    userInvocable: true,
    allowModelInvocation: true,
    model: "",
    allowedTools: ""
  }
}

export const buildGuidedDraftFromSkill = (skill: SkillResponse): SkillGuidedDraft => ({
  name: skill.name,
  description: skill.description ?? "",
  argumentHint: skill.argument_hint ?? "",
  instructions: skill.content ?? "",
  context: skill.context === "fork" ? "fork" : "inline",
  userInvocable: skill.user_invocable,
  allowModelInvocation: !skill.disable_model_invocation,
  model: skill.model ?? "",
  allowedTools: (skill.allowed_tools ?? []).join(", ")
})

export const parseAllowedTools = (value: string): string[] => {
  const seen = new Set<string>()
  const tools: string[] = []

  for (const candidate of value.split(/[\n,]/)) {
    const tool = candidate.trim()
    if (!tool || seen.has(tool)) continue
    seen.add(tool)
    tools.push(tool)
  }

  return tools
}

export const serializeGuidedSkillContent = (draft: SkillGuidedDraft): string => {
  const lines = [`name: ${quoteYamlString(draft.name.trim())}`]

  pushYamlLine(lines, "description", draft.description)
  pushYamlLine(lines, "argument-hint", draft.argumentHint)
  if (!draft.allowModelInvocation) lines.push("disable-model-invocation: true")
  if (!draft.userInvocable) lines.push("user-invocable: false")

  const allowedTools = parseAllowedTools(draft.allowedTools)
  if (allowedTools.length > 0) {
    lines.push(`allowed-tools: ${quoteYamlString(allowedTools.join(", "))}`)
  }
  pushYamlLine(lines, "model", draft.model)
  lines.push(`context: ${draft.context === "fork" ? "fork" : "inline"}`)

  return `---\n${lines.join("\n")}\n---\n\n${draft.instructions.trim()}`
}

export const validateGuidedSkillDraft = (draft: SkillGuidedDraft): string[] => {
  const errors: string[] = []
  if (!SKILL_NAME_REGEX.test(draft.name.trim())) {
    errors.push(
      "Name must start with a lowercase letter and use only lowercase letters, numbers, and hyphens (max 64 characters)."
    )
  }
  if (!draft.description.trim()) errors.push("Description is required.")
  if (!draft.instructions.trim()) errors.push("Instructions are required.")
  return errors
}

export const validateRawSkillContent = (
  content: string,
  canonicalName?: string
): string[] => {
  const normalized = content.replace(/\r\n/g, "\n")
  if (!normalized.trim()) return ["Skill content is required."]

  const lines = normalized.split("\n")
  if (lines[0].trim() !== "---") return []

  const closingIndex = lines.findIndex(
    (line, index) => index > 0 && line.trim() === "---"
  )
  if (closingIndex < 0) {
    return ["Frontmatter starts with --- but has no closing --- delimiter."]
  }
  if (!lines.slice(closingIndex + 1).join("\n").trim()) {
    return ["Skill instructions are required after frontmatter."]
  }
  let parsedFrontmatter: unknown
  try {
    parsedFrontmatter = parseYaml(
      lines.slice(1, closingIndex).join("\n"),
      { schema: PY_YAML_COMPATIBLE_SCHEMA }
    )
  } catch {
    return ["Frontmatter must be valid YAML."]
  }
  if (canonicalName) {
    const frontmatter = parsedFrontmatter
      && typeof parsedFrontmatter === "object"
      && !Array.isArray(parsedFrontmatter)
      ? parsedFrontmatter as Record<string, unknown>
      : null
    const parsedName = frontmatter?.name
    if (parsedName !== undefined && typeof parsedName !== "string") {
      return ["Frontmatter name must be a string."]
    }
    if (
      typeof parsedName === "string"
      && parsedName.trim().toLowerCase() !== canonicalName.trim().toLowerCase()
    ) {
      return [
        `Frontmatter name "${parsedName}" must match canonical name "${canonicalName}".`
      ]
    }
  }
  return []
}

export const buildSkillTemplateContent = (
  templateId: SkillTemplateId,
  name: string
): string => {
  return serializeGuidedSkillContent(buildGuidedDraftFromTemplate(templateId, name))
}

const validateSupportingFilename = (filename: string): void => {
  if (!SUPPORTING_FILE_NAME_REGEX.test(filename)) {
    throw new Error(
      "Supporting file names must be 1-100 chars and use letters, numbers, dot, underscore, or hyphen."
    )
  }
  if (filename.toLowerCase() === "skill.md") {
    throw new Error("SKILL.md is reserved and cannot be used as a supporting file name.")
  }
}

const normalizeSupportingRows = (
  rows: SupportingFileFormEntry[] | undefined
): SupportingFileFormEntry[] => {
  const entries = rows ?? []
  const dedupe = new Set<string>()
  const normalized: SupportingFileFormEntry[] = []

  for (const row of entries) {
    const filename = (row.filename ?? "").trim()
    const content = row.content ?? ""
    const originalFilename = (row.originalFilename ?? "").trim()

    if (!filename && !content && !originalFilename) {
      continue
    }
    if (!filename) {
      throw new Error("Each supporting file needs a filename.")
    }

    validateSupportingFilename(filename)

    if (dedupe.has(filename)) {
      throw new Error(`Duplicate supporting file name: ${filename}`)
    }
    dedupe.add(filename)

    normalized.push({
      filename,
      content,
      originalFilename: originalFilename || undefined
    })
  }

  return normalized
}

export const buildSupportingFilesForCreate = (
  rows: SupportingFileFormEntry[] | undefined
): Record<string, string> | undefined => {
  const normalized = normalizeSupportingRows(rows)
  if (!normalized.length) {
    return undefined
  }

  const files: Record<string, string> = {}
  for (const row of normalized) {
    files[row.filename!] = row.content ?? ""
  }
  return Object.keys(files).length ? files : undefined
}

export const buildSupportingFilesForUpdate = (
  initialFiles: Record<string, string> | null | undefined,
  rows: SupportingFileFormEntry[] | undefined
): Record<string, string | null> | undefined => {
  const initial = initialFiles ?? {}
  const normalized = normalizeSupportingRows(rows)
  const remainingInitial = new Set(Object.keys(initial))
  const updates: Record<string, string | null> = {}

  for (const row of normalized) {
    const filename = row.filename!
    const content = row.content ?? ""
    const originalFilename = row.originalFilename

    if (originalFilename && Object.prototype.hasOwnProperty.call(initial, originalFilename)) {
      remainingInitial.delete(originalFilename)
      if (filename !== originalFilename) {
        updates[originalFilename] = null
        updates[filename] = content
      } else if (initial[originalFilename] !== content) {
        updates[filename] = content
      }
      continue
    }

    if (!Object.prototype.hasOwnProperty.call(initial, filename) || initial[filename] !== content) {
      updates[filename] = content
    }
  }

  for (const deletedName of remainingInitial) {
    updates[deletedName] = null
  }

  return Object.keys(updates).length ? updates : undefined
}

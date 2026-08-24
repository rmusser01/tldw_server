import { dump, load } from "js-yaml"
import type {
  ChatMacroDefinition,
  ChatMacroOutputProfile,
  ChatMacroSettings,
  ChatMacroStep
} from "@/services/chat-macros"

const MAX_IMPORT_BYTES = 500_000
const MAX_BRANCHES = 6
const MAX_TIMEOUT_SECONDS = 3_600
const GUIDED_CONTEXT_SURFACES = ["chat", "chat-workspace", "research-workspace", "acp"]

type GuidedBranch = {
  id: string
  label: string
  output: string
  prompt: string
}

export interface GuidedMacroDraft {
  name: string
  command: string
  description: string
  outputProfile: string
  maxBranches: number
  maxConcurrency: number
  timeoutSeconds: number
  branches: GuidedBranch[]
  merge: { id: string; output: string; prompt: string }
}

export type ParsedMacroSource =
  | { mode: "guided"; draft: GuidedMacroDraft }
  | { mode: "source"; raw: string; error?: string }

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value)

const hasOnlyKeys = (value: Record<string, unknown>, keys: string[]): boolean =>
  Object.keys(value).every((key) => keys.includes(key))

const isStringArray = (value: unknown): value is string[] =>
  Array.isArray(value) && value.every((item) => typeof item === "string")

const isEmptyRecord = (value: unknown): boolean => isRecord(value) && Object.keys(value).length === 0

const boundedInteger = (value: unknown, fallback: number, maximum: number): number => {
  if (typeof value !== "number" || !Number.isFinite(value)) return fallback
  return Math.max(1, Math.min(maximum, Math.floor(value)))
}

const guidedDraftFromDefinition = (
  definition: Record<string, unknown>
): GuidedMacroDraft | null => {
  if (
    !hasOnlyKeys(definition, [
      "schema_version",
      "name",
      "command",
      "description",
      "enabled",
      "args",
      "context",
      "execution",
      "steps",
      "output_profile",
      "permissions"
    ])
    || definition.schema_version !== 1
    || typeof definition.name !== "string"
    || typeof definition.command !== "string"
    || typeof definition.description !== "string"
    || definition.enabled !== true
    || !isEmptyRecord(definition.args)
    || typeof definition.output_profile !== "string"
  ) {
    return null
  }

  const context = definition.context
  if (
    !isRecord(context)
    || !hasOnlyKeys(context, [
      "surfaces",
      "include_chat_history",
      "include_workspace_context",
      "retrieval",
      "snapshot_at_dispatch"
    ])
    || !isStringArray(context.surfaces)
    || context.surfaces.length !== GUIDED_CONTEXT_SURFACES.length
    || context.surfaces.some((surface, index) => surface !== GUIDED_CONTEXT_SURFACES[index])
    || context.include_chat_history !== true
    || context.include_workspace_context !== "auto"
    || context.retrieval !== "auto"
    || context.snapshot_at_dispatch !== true
  ) {
    return null
  }

  const execution = definition.execution
  if (
    !isRecord(execution)
    || !hasOnlyKeys(execution, [
      "mode_default",
      "branch_strategy",
      "max_branches",
      "max_concurrency",
      "timeout_seconds",
      "retries_per_branch",
      "merge_retries",
      "partial_failure",
      "retain_scratch_branches"
    ])
    || execution.mode_default !== "background"
    || execution.branch_strategy !== "auto"
    || typeof execution.max_branches !== "number"
    || typeof execution.max_concurrency !== "number"
    || typeof execution.timeout_seconds !== "number"
    || !Number.isInteger(execution.max_branches)
    || execution.max_branches < 1
    || execution.max_branches > MAX_BRANCHES
    || !Number.isInteger(execution.max_concurrency)
    || execution.max_concurrency < 1
    || execution.max_concurrency > execution.max_branches
    || !Number.isInteger(execution.timeout_seconds)
    || execution.timeout_seconds < 1
    || execution.timeout_seconds > MAX_TIMEOUT_SECONDS
    || execution.retries_per_branch !== 1
    || execution.merge_retries !== 1
    || execution.partial_failure !== "best_effort"
    || execution.retain_scratch_branches !== false
  ) {
    return null
  }

  const permissions = definition.permissions
  if (
    !isRecord(permissions)
    || !hasOnlyKeys(permissions, ["tool_calls", "skills"])
    || !isStringArray(permissions.tool_calls)
    || !isStringArray(permissions.skills)
    || permissions.tool_calls.length > 0
    || permissions.skills.length > 0
    || !Array.isArray(definition.steps)
    || definition.steps.length < 3
  ) {
    return null
  }

  const branchSteps = definition.steps.slice(0, -2)
  const merge = definition.steps.at(-2)
  const post = definition.steps.at(-1)
  if (
    !isRecord(merge)
    || !isRecord(post)
    || branchSteps.length === 0
    || branchSteps.length > execution.max_branches
  ) {
    return null
  }

  const branches: GuidedBranch[] = []
  for (const step of branchSteps) {
    if (
      !isRecord(step)
      || !hasOnlyKeys(step, ["id", "type", "label", "output", "prompt"])
      || step.type !== "branch_prompt"
      || typeof step.id !== "string"
      || typeof step.label !== "string"
      || typeof step.output !== "string"
      || typeof step.prompt !== "string"
    ) {
      return null
    }
    branches.push({ id: step.id, label: step.label, output: step.output, prompt: step.prompt })
  }

  const branchOutputs = branches.map((branch) => branch.output)
  if (
    !hasOnlyKeys(merge, ["id", "type", "consumes", "output", "prompt"])
    || merge.type !== "merge"
    || typeof merge.id !== "string"
    || typeof merge.output !== "string"
    || typeof merge.prompt !== "string"
    || !isStringArray(merge.consumes)
    || merge.consumes.length !== branchOutputs.length
    || merge.consumes.some((output, index) => output !== branchOutputs[index])
    || !hasOnlyKeys(post, ["id", "type", "consumes"])
    || post.id !== "post"
    || post.type !== "post_result"
    || !isStringArray(post.consumes)
    || post.consumes.length !== 1
    || post.consumes[0] !== merge.output
  ) {
    return null
  }

  return {
    name: definition.name,
    command: definition.command,
    description: definition.description,
    outputProfile: definition.output_profile,
    maxBranches: execution.max_branches,
    maxConcurrency: execution.max_concurrency,
    timeoutSeconds: execution.timeout_seconds,
    branches,
    merge: { id: merge.id, output: merge.output, prompt: merge.prompt }
  }
}

export const createBlankMacroDraft = (): GuidedMacroDraft => ({
  name: "",
  command: "",
  description: "",
  outputProfile: "default",
  maxBranches: 6,
  maxConcurrency: 3,
  timeoutSeconds: 180,
  branches: [{ id: "branch_1", label: "Branch 1", output: "branch_1", prompt: "" }],
  merge: { id: "merge", output: "final", prompt: "" }
})

export const serializeGuidedMacro = (draft: GuidedMacroDraft): string => {
  if (draft.branches.length < 1 || draft.branches.length > MAX_BRANCHES) {
    throw new Error(`Guided macros require between 1 and ${MAX_BRANCHES} branches.`)
  }
  const maxBranches = boundedInteger(draft.maxBranches, 6, MAX_BRANCHES)
  if (draft.branches.length > maxBranches) {
    throw new Error("Guided macro branch count cannot exceed max branches.")
  }
  const maxConcurrency = Math.min(
    maxBranches,
    boundedInteger(draft.maxConcurrency, 3, MAX_BRANCHES)
  )
  const definition: ChatMacroDefinition = {
    schema_version: 1,
    name: draft.name,
    command: draft.command,
    description: draft.description,
    enabled: true,
    args: {},
    context: {
      surfaces: GUIDED_CONTEXT_SURFACES,
      include_chat_history: true,
      include_workspace_context: "auto",
      retrieval: "auto",
      snapshot_at_dispatch: true
    },
    execution: {
      mode_default: "background",
      branch_strategy: "auto",
      max_branches: maxBranches,
      max_concurrency: maxConcurrency,
      timeout_seconds: boundedInteger(draft.timeoutSeconds, 180, MAX_TIMEOUT_SECONDS),
      retries_per_branch: 1,
      merge_retries: 1,
      partial_failure: "best_effort",
      retain_scratch_branches: false
    },
    steps: [
      ...draft.branches.map<ChatMacroStep>((branch) => ({
        id: branch.id,
        type: "branch_prompt",
        label: branch.label,
        output: branch.output,
        prompt: branch.prompt
      })),
      {
        id: draft.merge.id,
        type: "merge",
        consumes: draft.branches.map((branch) => branch.output),
        output: draft.merge.output,
        prompt: draft.merge.prompt
      },
      { id: "post", type: "post_result", consumes: [draft.merge.output] }
    ],
    output_profile: draft.outputProfile,
    permissions: { tool_calls: [], skills: [] }
  }

  return dump(definition, { lineWidth: -1, noRefs: true })
}

export const parseMacroSource = (raw: string): ParsedMacroSource => {
  let parsed: unknown
  try {
    parsed = load(raw)
  } catch {
    return { mode: "source", raw, error: "Macro YAML is invalid." }
  }
  if (!isRecord(parsed)) {
    return { mode: "source", raw, error: "Macro YAML must be a mapping." }
  }

  const draft = guidedDraftFromDefinition(parsed)
  return draft ? { mode: "guided", draft } : { mode: "source", raw }
}

export const readMacroImport = async (file: File): Promise<string> => {
  if (!/\.ya?ml$/i.test(file.name)) {
    throw new Error("Macro imports must use a .yaml or .yml file.")
  }
  if (file.size > MAX_IMPORT_BYTES) {
    throw new Error("Macro imports cannot exceed 500,000 bytes.")
  }
  return file.text()
}

export const outputProfilesToSettings = (
  settings: ChatMacroSettings,
  profiles: Record<string, ChatMacroOutputProfile>
): ChatMacroSettings => ({ ...settings, output_profiles: profiles })

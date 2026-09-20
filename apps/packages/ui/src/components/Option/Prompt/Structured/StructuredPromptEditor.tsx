import React from "react"
import type {
  StructuredPromptDefinition,
  StructuredPromptPreviewResponse
} from "@/services/prompt-studio"
import { BlockListPanel } from "./BlockListPanel"
import { BlockEditorPanel } from "./BlockEditorPanel"
import { VariableEditorPanel } from "./VariableEditorPanel"

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
  required?: boolean
  input_type?: string
  label?: string
  description?: string
  default_value?: unknown
  options?: string[] | null
  max_length?: number
}

type StructuredPromptAssemblyConfig = {
  legacy_system_roles: string[]
  legacy_user_roles: string[]
  block_separator: string
}

type StructuredPromptEditorProps = {
  value: StructuredPromptDefinition | null | undefined
  onChange: (nextValue: StructuredPromptDefinition) => void
  previewResult: StructuredPromptPreviewResponse | null
  previewLoading: boolean
  onPreview: (variables: Record<string, string>) => void
}

const makeDefaultBlock = (): StructuredPromptBlock => ({
  id: `block-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
  name: "Task",
  role: "user",
  kind: "task",
  content: "Describe the task here.",
  enabled: true,
  order: 10,
  is_template: false
})

const DEFAULT_ASSEMBLY_CONFIG: StructuredPromptAssemblyConfig = {
  legacy_system_roles: ["system", "developer"],
  legacy_user_roles: ["user"],
  block_separator: "\n\n"
}

const reindexBlocks = (blocks: StructuredPromptBlock[]): StructuredPromptBlock[] =>
  blocks.map((block, index) => ({
    ...block,
    order: (index + 1) * 10
  }))

const normalizeDefinition = (
  value: StructuredPromptDefinition | null | undefined
) => {
  const candidate = value && typeof value === "object" ? value : {}
  const assemblyConfigCandidate =
    candidate.assembly_config &&
    typeof candidate.assembly_config === "object" &&
    !Array.isArray(candidate.assembly_config)
      ? (candidate.assembly_config as Record<string, unknown>)
      : null
  const assemblyConfig: StructuredPromptAssemblyConfig = {
    legacy_system_roles: Array.isArray(assemblyConfigCandidate?.legacy_system_roles)
      ? assemblyConfigCandidate.legacy_system_roles
          .filter((role): role is string => typeof role === "string")
      : DEFAULT_ASSEMBLY_CONFIG.legacy_system_roles,
    legacy_user_roles: Array.isArray(assemblyConfigCandidate?.legacy_user_roles)
      ? assemblyConfigCandidate.legacy_user_roles
          .filter((role): role is string => typeof role === "string")
      : DEFAULT_ASSEMBLY_CONFIG.legacy_user_roles,
    block_separator:
      typeof assemblyConfigCandidate?.block_separator === "string"
        ? assemblyConfigCandidate.block_separator
        : DEFAULT_ASSEMBLY_CONFIG.block_separator
  }
  const blocks = Array.isArray(candidate.blocks)
    ? reindexBlocks(
        candidate.blocks.map((block: any, index: number) => ({
          id: typeof block?.id === "string" ? block.id : `block-${index + 1}`,
          name: typeof block?.name === "string" ? block.name : `Block ${index + 1}`,
          role:
            block?.role === "system" ||
            block?.role === "developer" ||
            block?.role === "assistant"
              ? block.role
              : "user",
          kind: typeof block?.kind === "string" ? block.kind : undefined,
          content: typeof block?.content === "string" ? block.content : "",
          enabled: block?.enabled !== false,
          order:
            typeof block?.order === "number" && Number.isFinite(block.order)
              ? block.order
              : (index + 1) * 10,
          is_template: block?.is_template === true
        }))
      )
    : [makeDefaultBlock()]

  const variables = Array.isArray(candidate.variables)
    ? candidate.variables.map((variable: any) => ({
        name: typeof variable?.name === "string" ? variable.name : "",
        required: variable?.required === true,
        input_type:
          typeof variable?.input_type === "string" ? variable.input_type : "text",
        label: typeof variable?.label === "string" ? variable.label : undefined,
        description:
          typeof variable?.description === "string"
            ? variable.description
            : undefined,
        default_value:
          variable && Object.prototype.hasOwnProperty.call(variable, "default_value")
            ? variable.default_value
            : undefined,
        options:
          variable?.options === null
            ? null
            : Array.isArray(variable?.options)
              ? variable.options.filter(
                  (option: unknown): option is string => typeof option === "string"
                )
              : undefined,
        max_length:
          typeof variable?.max_length === "number" &&
          Number.isFinite(variable.max_length) &&
          variable.max_length >= 1
            ? variable.max_length
            : undefined
      }))
    : []

  return {
    schema_version:
      typeof candidate.schema_version === "number" &&
      Number.isFinite(candidate.schema_version)
        ? candidate.schema_version
        : 1,
    format: "structured",
    assembly_config: assemblyConfig,
    variables,
    blocks
  }
}

export const StructuredPromptEditor: React.FC<StructuredPromptEditorProps> = ({
  value,
  onChange,
  previewResult,
  previewLoading,
  onPreview
}) => {
  const definition = React.useMemo(() => normalizeDefinition(value), [value])
  const [selectedBlockId, setSelectedBlockId] = React.useState<string | null>(
    definition.blocks[0]?.id || null
  )
  const [previewValues, setPreviewValues] = React.useState<Record<string, string>>(
    {}
  )

  React.useEffect(() => {
    if (!definition.blocks.some((block: any) => block.id === selectedBlockId)) {
      setSelectedBlockId(definition.blocks[0]?.id || null)
    }
  }, [definition.blocks, selectedBlockId])

  React.useEffect(() => {
    setPreviewValues((currentValues) => {
      const allowedNames = new Set(
        definition.variables
          .map((variable: any) => variable.name)
          .filter((name: string) => name.length > 0)
      )
      return Object.fromEntries(
        Object.entries(currentValues).filter(([name]) => allowedNames.has(name))
      )
    })
  }, [definition.variables])

  const selectedBlock =
    definition.blocks.find((block: any) => block.id === selectedBlockId) || null

  const updateDefinition = (nextValue: StructuredPromptDefinition) => {
    onChange(normalizeDefinition(nextValue))
  }

  const updateBlocks = (blocks: StructuredPromptBlock[]) => {
    updateDefinition({
      ...definition,
      blocks: reindexBlocks(blocks)
    })
  }

  const handleAddBlock = () => {
    const nextBlock = makeDefaultBlock()
    updateBlocks([...definition.blocks, nextBlock as StructuredPromptBlock])
    setSelectedBlockId(nextBlock.id)
  }

  const handleMoveBlock = (
    blockId: string,
    direction: "up" | "down"
  ) => {
    const currentIndex = definition.blocks.findIndex((block: any) => block.id === blockId)
    if (currentIndex < 0) return
    const targetIndex = direction === "up" ? currentIndex - 1 : currentIndex + 1
    if (targetIndex < 0 || targetIndex >= definition.blocks.length) return

    const nextBlocks = [...definition.blocks]
    const [moved] = nextBlocks.splice(currentIndex, 1)
    nextBlocks.splice(targetIndex, 0, moved)
    updateBlocks(nextBlocks as StructuredPromptBlock[])
  }

  const handleRemoveBlock = (blockId: string) => {
    const nextBlocks = definition.blocks.filter((block: any) => block.id !== blockId)
    const fallbackBlocks =
      nextBlocks.length > 0 ? nextBlocks : ([makeDefaultBlock()] as StructuredPromptBlock[])
    updateBlocks(fallbackBlocks as StructuredPromptBlock[])
    if (selectedBlockId === blockId) {
      setSelectedBlockId(fallbackBlocks[0]?.id || null)
    }
  }

  const handleUpdateBlock = (updates: Partial<StructuredPromptBlock>) => {
    if (!selectedBlock) return
    updateBlocks(
      definition.blocks.map((block: any) =>
        block.id === selectedBlock.id ? { ...block, ...updates } : block
      ) as StructuredPromptBlock[]
    )
  }

  const handleVariablesChange = (variables: StructuredPromptVariable[]) => {
    updateDefinition({
      ...definition,
      variables
    })
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-3 rounded-xl border border-border bg-surface1 p-4">
        <div>
          <h3 className="text-sm font-semibold text-text">
            Structured prompt builder
          </h3>
          <p className="text-sm text-text-muted">
            Build prompts from ordered blocks and preview the assembled backend messages.
          </p>
        </div>
        <button
          type="button"
          onClick={() => onPreview(previewValues)}
          data-testid="structured-preview-button"
          className="rounded-md bg-primary px-3 py-2 text-sm font-medium text-white hover:opacity-90 disabled:opacity-60"
          disabled={previewLoading}
        >
          {previewLoading ? "Previewing..." : "Preview"}
        </button>
      </div>

      <div className="grid gap-4 xl:grid-cols-[18rem_minmax(0,1fr)]">
        <BlockListPanel
          blocks={definition.blocks as StructuredPromptBlock[]}
          selectedBlockId={selectedBlockId}
          onSelect={setSelectedBlockId}
          onAddBlock={handleAddBlock}
          onMoveBlock={handleMoveBlock}
          onRemoveBlock={handleRemoveBlock}
        />
        <BlockEditorPanel
          block={selectedBlock as StructuredPromptBlock | null}
          onChange={handleUpdateBlock}
        />
      </div>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <VariableEditorPanel
          variables={definition.variables as StructuredPromptVariable[]}
          previewValues={previewValues}
          onVariablesChange={handleVariablesChange}
          onPreviewValuesChange={setPreviewValues}
        />

        <section className="rounded-xl border border-border bg-surface1 p-4">
          <div className="mb-3">
            <h3 className="text-sm font-semibold text-text">Preview</h3>
            <p className="text-xs text-text-muted">
              Live server-side assembly using the Prompt Studio preview endpoint.
            </p>
          </div>

          {!previewResult && (
            <p className="text-sm text-text-muted">
              Run preview to inspect the assembled role-based messages and legacy snapshot.
            </p>
          )}

          {previewResult && (
            <div className="space-y-4" data-testid="structured-preview-panel">
              <div className="space-y-2">
                {previewResult.assembled_messages.map((message, index) => (
                  <div
                    key={`${message.role}-${index}`}
                    className="rounded-lg border border-border bg-background p-3"
                  >
                    <div className="mb-1 text-xs font-medium uppercase tracking-wide text-text-muted">
                      {message.role}
                    </div>
                    <div className="whitespace-pre-wrap text-sm text-text">
                      {message.content}
                    </div>
                  </div>
                ))}
              </div>

              <div className="grid gap-3 sm:grid-cols-2">
                <div className="rounded-lg border border-border bg-background p-3">
                  <div className="mb-1 text-xs font-medium uppercase tracking-wide text-text-muted">
                    Legacy system
                  </div>
                  <div className="whitespace-pre-wrap text-sm text-text">
                    {previewResult.legacy_system_prompt || "No system output"}
                  </div>
                </div>
                <div className="rounded-lg border border-border bg-background p-3">
                  <div className="mb-1 text-xs font-medium uppercase tracking-wide text-text-muted">
                    Legacy user
                  </div>
                  <div className="whitespace-pre-wrap text-sm text-text">
                    {previewResult.legacy_user_prompt || "No user output"}
                  </div>
                </div>
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  )
}

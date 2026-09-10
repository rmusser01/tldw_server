import React from "react"

type StructuredPromptBlock = {
  id: string
  name: string
  role: "system" | "developer" | "user" | "assistant"
  content: string
  enabled: boolean
  order: number
  is_template: boolean
}

type BlockEditorPanelProps = {
  block: StructuredPromptBlock | null
  onChange: (updates: Partial<StructuredPromptBlock>) => void
  allowedRoles?: readonly StructuredPromptBlock["role"][]
  showRole?: boolean
  sectionKey?: string | null
  onSectionKeyChange?: (sectionKey: string) => void
  sectionKeyError?: string | null
  nameInputRef?: React.Ref<HTMLInputElement>
}

const ALL_ROLES: readonly StructuredPromptBlock["role"][] = [
  "system",
  "developer",
  "user",
  "assistant"
]

const roleLabel = (role: StructuredPromptBlock["role"]): string =>
  role.charAt(0).toUpperCase() + role.slice(1)

export const BlockEditorPanel: React.FC<BlockEditorPanelProps> = ({
  block,
  onChange,
  allowedRoles = ALL_ROLES,
  showRole = true,
  sectionKey,
  onSectionKeyChange,
  sectionKeyError,
  nameInputRef
}) => {
  if (!block) {
    return (
      <section className="rounded-xl border border-border bg-surface1 p-4">
        <h3 className="text-sm font-semibold text-text">Block editor</h3>
        <p className="mt-2 text-sm text-text-muted">
          Select a block to edit its role, content, and template behavior.
        </p>
      </section>
    )
  }

  return (
    <section className="rounded-xl border border-border bg-surface1 p-4">
      <div className="mb-3">
        <h3 className="text-sm font-semibold text-text">Block editor</h3>
        <p className="text-xs text-text-muted">
          Keep each block focused on one job: identity, task, constraints, or
          examples.
        </p>
      </div>

      <div className="space-y-3">
        <label className="block">
          <span className="mb-1 block text-xs font-medium uppercase tracking-wide text-text-muted">
            Name
          </span>
          <input
            ref={nameInputRef}
            type="text"
            aria-label="Block name"
            value={block.name}
            onChange={(event) => onChange({ name: event.target.value })}
            data-testid="structured-block-name"
            className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm text-text"
          />
        </label>

        {showRole ? (
          <label className="block">
            <span className="mb-1 block text-xs font-medium uppercase tracking-wide text-text-muted">
              Role
            </span>
            <select
              aria-label="Block role"
              value={block.role}
              onChange={(event) =>
                onChange({
                  role: event.target.value as StructuredPromptBlock["role"]
                })
              }
              data-testid="structured-block-role"
              className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm text-text"
            >
              {allowedRoles.map((role) => (
                <option key={role} value={role}>
                  {roleLabel(role)}
                </option>
              ))}
            </select>
          </label>
        ) : null}

        {onSectionKeyChange ? (
          <label className="block">
            <span className="mb-1 block text-xs font-medium uppercase tracking-wide text-text-muted">
              Section key
            </span>
            <input
              type="text"
              aria-label="Section key"
              aria-invalid={Boolean(sectionKeyError)}
              aria-describedby={
                sectionKeyError ? "recipe-section-key-error" : undefined
              }
              value={sectionKey ?? ""}
              onChange={(event) => onSectionKeyChange(event.target.value)}
              className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm text-text aria-invalid:border-danger"
            />
            {sectionKeyError ? (
              <span
                id="recipe-section-key-error"
                role="alert"
                className="mt-1 block text-xs text-danger"
              >
                {sectionKeyError}
              </span>
            ) : null}
          </label>
        ) : null}

        <label className="block">
          <span className="mb-1 block text-xs font-medium uppercase tracking-wide text-text-muted">
            Content
          </span>
          <textarea
            aria-label="Block content"
            value={block.content}
            onChange={(event) => onChange({ content: event.target.value })}
            rows={8}
            data-testid="structured-block-content"
            className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm text-text"
          />
        </label>

        <div className="grid gap-3 sm:grid-cols-2">
          <label className="flex items-center gap-2 text-sm text-text">
            <input
              type="checkbox"
              checked={block.enabled}
              aria-label="Block enabled"
              onChange={(event) => onChange({ enabled: event.target.checked })}
              data-testid="structured-block-enabled"
            />
            Enabled
          </label>
          <label className="flex items-center gap-2 text-sm text-text">
            <input
              type="checkbox"
              checked={block.is_template}
              aria-label="Block uses variables"
              onChange={(event) =>
                onChange({ is_template: event.target.checked })
              }
              data-testid="structured-block-template"
            />
            Uses variables
          </label>
        </div>
      </div>
    </section>
  )
}

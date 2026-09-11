import React from "react"
import { Plus, Trash2 } from "lucide-react"

type StructuredPromptVariable = {
  name: string
  required?: boolean
  input_type?: string
  label?: string | null
  description?: string | null
  default_value?: unknown
  options?: string[] | null
  max_length?: number | null
}

type VariableEditorPanelProps = {
  variables: StructuredPromptVariable[]
  previewValues: Record<string, string>
  onVariablesChange?: (variables: StructuredPromptVariable[]) => void
  onPreviewValuesChange?: (values: Record<string, string>) => void
  runtimeValues?: Readonly<Record<string, string>>
  onVariableChange?: (
    variableName: string,
    updates: Partial<StructuredPromptVariable>
  ) => void
  onAddVariable?: () => void
  onRemoveVariable?: (variableName: string) => void
  onRuntimeValueChange?: (variableName: string, value: string) => void
  showDeclarationFields?: boolean
  variableNameDrafts?: Readonly<Record<string, string>>
  variableNameErrors?: Readonly<Record<string, string>>
  onVariableNameDraftChange?: (variableName: string, value: string) => void
  onVariableNameCommit?: (variableName: string) => void
}

export const VariableEditorPanel: React.FC<VariableEditorPanelProps> = ({
  variables,
  previewValues,
  onVariablesChange,
  onPreviewValuesChange,
  runtimeValues,
  onVariableChange,
  onAddVariable,
  onRemoveVariable,
  onRuntimeValueChange,
  showDeclarationFields = false,
  variableNameDrafts,
  variableNameErrors,
  onVariableNameDraftChange,
  onVariableNameCommit
}) => {
  const updateVariable = (
    index: number,
    updates: Partial<StructuredPromptVariable>
  ) => {
    if (onVariableChange) {
      onVariableChange(variables[index].name, updates)
      return
    }
    const next = variables.map((variable, currentIndex) =>
      currentIndex === index ? { ...variable, ...updates } : variable
    )
    onVariablesChange?.(next)
  }

  const removeVariable = (index: number) => {
    const removedName = variables[index]?.name
    if (removedName && onRemoveVariable) {
      onRemoveVariable(removedName)
      return
    }
    const next = variables.filter((_, currentIndex) => currentIndex !== index)
    onVariablesChange?.(next)
    const nextPreviewValues = { ...previewValues }
    if (removedName) {
      delete nextPreviewValues[removedName]
      onPreviewValuesChange?.(nextPreviewValues)
    }
  }

  const addVariable = () => {
    if (onAddVariable) {
      onAddVariable()
      return
    }
    onVariablesChange?.([
      ...variables,
      {
        name: `variable_${variables.length + 1}`,
        required: false,
        input_type: "text"
      }
    ])
  }

  const currentValues = runtimeValues ?? previewValues

  const updateCurrentValue = (name: string, value: string) => {
    if (onRuntimeValueChange) {
      onRuntimeValueChange(name, value)
      return
    }
    onPreviewValuesChange?.({ ...previewValues, [name]: value })
  }

  return (
    <section className="rounded-xl border border-border bg-surface p-4">
      <div className="mb-3 flex items-center justify-between gap-2">
        <div>
          <h3 className="text-sm font-semibold text-text">Variables</h3>
          <p className="text-xs text-text-muted">
            Define reusable prompt inputs and optional preview values.
          </p>
        </div>
        <button
          type="button"
          onClick={addVariable}
          data-testid="structured-variable-add"
          className="inline-flex items-center gap-1 rounded-md border border-border px-2 py-1 text-xs font-medium text-text hover:bg-surface2"
        >
          <Plus className="size-3" />
          Add variable
        </button>
      </div>

      <div className="space-y-3">
        {variables.length === 0 && (
          <p className="text-sm text-text-muted">
            No variables yet. Add one to support dynamic prompt assembly.
          </p>
        )}

        {variables.map((variable, index) => (
          <div
            key={`${variable.name}-${index}`}
            className="rounded-lg border border-border bg-bg p-3"
          >
            <div className="mb-2 flex items-center justify-between gap-2">
              <span className="text-xs font-medium uppercase tracking-wide text-text-muted">
                Variable {index + 1}
              </span>
              <button
                type="button"
                onClick={() => removeVariable(index)}
                aria-label={`Remove ${variable.label || variable.name || `variable ${index + 1}`}`}
                className="inline-flex min-h-11 min-w-11 items-center justify-center rounded border border-border text-danger hover:bg-danger/5"
              >
                <Trash2 className="size-3" />
              </button>
            </div>

            <div className="grid gap-3 sm:grid-cols-2">
              <label className="block">
                <span className="mb-1 block text-xs font-medium uppercase tracking-wide text-text-muted">
                  Name
                </span>
                <input
                  type="text"
                  aria-label="Variable name"
                  aria-invalid={Boolean(variableNameErrors?.[variable.name])}
                  aria-describedby={
                    variableNameErrors?.[variable.name]
                      ? `structured-variable-name-error-${index}`
                      : undefined
                  }
                  value={variableNameDrafts?.[variable.name] ?? variable.name}
                  onChange={(event) => {
                    if (onVariableNameDraftChange) {
                      onVariableNameDraftChange(
                        variable.name,
                        event.target.value
                      )
                    } else {
                      updateVariable(index, { name: event.target.value })
                    }
                  }}
                  onBlur={() => onVariableNameCommit?.(variable.name)}
                  data-testid={`structured-variable-name-${index}`}
                  className="w-full rounded-md border border-border bg-surface px-3 py-2 text-sm text-text aria-invalid:border-danger"
                />
                {variableNameErrors?.[variable.name] ? (
                  <span
                    id={`structured-variable-name-error-${index}`}
                    role="alert"
                    className="mt-1 block text-xs text-danger"
                  >
                    {variableNameErrors[variable.name]}
                  </span>
                ) : null}
              </label>

              <label className="block">
                <span className="mb-1 block text-xs font-medium uppercase tracking-wide text-text-muted">
                  Input type
                </span>
                <select
                  aria-label="Variable input type"
                  value={variable.input_type || "text"}
                  onChange={(event) =>
                    updateVariable(index, { input_type: event.target.value })
                  }
                  className="w-full rounded-md border border-border bg-surface px-3 py-2 text-sm text-text"
                >
                  <option value="text">Text</option>
                  <option value="textarea">Textarea</option>
                  <option value="number">Number</option>
                  <option value="boolean">Boolean</option>
                  <option value="select">Select</option>
                  <option value="json">JSON</option>
                </select>
              </label>
            </div>

            {showDeclarationFields ? (
              <div className="mt-3 grid gap-3 sm:grid-cols-2">
                <label className="block">
                  <span className="mb-1 block text-xs font-medium uppercase tracking-wide text-text-muted">
                    Label
                  </span>
                  <input
                    type="text"
                    aria-label="Variable label"
                    value={variable.label ?? ""}
                    onChange={(event) =>
                      updateVariable(index, {
                        label: event.target.value || null
                      })
                    }
                    className="w-full rounded-md border border-border bg-surface px-3 py-2 text-sm text-text"
                  />
                </label>
                <label className="block">
                  <span className="mb-1 block text-xs font-medium uppercase tracking-wide text-text-muted">
                    Maximum length
                  </span>
                  <input
                    type="number"
                    min={1}
                    aria-label="Variable maximum length"
                    value={variable.max_length ?? ""}
                    onChange={(event) =>
                      updateVariable(index, {
                        max_length: event.target.value
                          ? Number(event.target.value)
                          : null
                      })
                    }
                    className="w-full rounded-md border border-border bg-surface px-3 py-2 text-sm text-text"
                  />
                </label>
                <label className="block sm:col-span-2">
                  <span className="mb-1 block text-xs font-medium uppercase tracking-wide text-text-muted">
                    Description
                  </span>
                  <input
                    type="text"
                    aria-label="Variable description"
                    value={variable.description ?? ""}
                    onChange={(event) =>
                      updateVariable(index, {
                        description: event.target.value || null
                      })
                    }
                    className="w-full rounded-md border border-border bg-surface px-3 py-2 text-sm text-text"
                  />
                </label>
                <label className="block sm:col-span-2">
                  <span className="mb-1 block text-xs font-medium uppercase tracking-wide text-text-muted">
                    Options, one per line
                  </span>
                  <textarea
                    rows={2}
                    aria-label="Variable options"
                    value={(variable.options ?? []).join("\n")}
                    onChange={(event) =>
                      updateVariable(index, {
                        options: event.target.value
                          .split("\n")
                          .map((option) => option.trim())
                          .filter(Boolean)
                      })
                    }
                    className="w-full rounded-md border border-border bg-surface px-3 py-2 text-sm text-text"
                  />
                </label>
                <label className="flex items-center gap-2 text-sm text-text sm:col-span-2">
                  <input
                    type="checkbox"
                    aria-label={`Use a saved starter default for ${variable.label || variable.name}`}
                    checked={variable.default_value != null}
                    onChange={(event) =>
                      updateVariable(index, {
                        default_value: event.target.checked ? "" : null
                      })
                    }
                  />
                  Save a starter default
                </label>
                {variable.default_value != null ? (
                  <label className="block sm:col-span-2">
                    <span className="mb-1 block text-xs font-medium uppercase tracking-wide text-text-muted">
                      Starter default (saved)
                    </span>
                    <textarea
                      rows={2}
                      aria-label={`Starter default for ${variable.label || variable.name} (saved)`}
                      value={
                        typeof variable.default_value === "string"
                          ? variable.default_value
                          : ""
                      }
                      onChange={(event) =>
                        updateVariable(index, {
                          default_value: event.target.value
                        })
                      }
                      className="w-full rounded-md border border-border bg-surface px-3 py-2 text-sm text-text"
                    />
                  </label>
                ) : null}
              </div>
            ) : null}

            <label className="mt-3 flex items-center gap-2 text-sm text-text">
              <input
                type="checkbox"
                aria-label="Variable required"
                checked={!!variable.required}
                onChange={(event) =>
                  updateVariable(index, { required: event.target.checked })
                }
              />
              Required
            </label>
          </div>
        ))}
      </div>

      {variables.length > 0 && (
        <div className="mt-4 space-y-3 border-t border-border pt-4">
          <div>
            <h4 className="text-sm font-semibold text-text">Preview inputs</h4>
            <p className="text-xs text-text-muted">
              {runtimeValues
                ? "Current inputs used only for this preview. They are not saved."
                : "Sample values passed to the backend preview endpoint."}
            </p>
          </div>
          {variables.map((variable) => (
            <label key={`preview-${variable.name}`} className="block">
              <span className="mb-1 block text-xs font-medium uppercase tracking-wide text-text-muted">
                {runtimeValues
                  ? `${variable.label || variable.name} current value (not saved)`
                  : variable.name}
              </span>
              {variable.input_type === "textarea" ? (
                <textarea
                  rows={2}
                  aria-label={`Current value for ${variable.label || variable.name} (not saved)`}
                  value={currentValues[variable.name] || ""}
                  onChange={(event) =>
                    updateCurrentValue(variable.name, event.target.value)
                  }
                  data-testid={`structured-preview-variable-${variable.name}`}
                  className="w-full rounded-md border border-border bg-bg px-3 py-2 text-sm text-text"
                />
              ) : (
                <input
                  type="text"
                  aria-label={
                    runtimeValues
                      ? `Current value for ${variable.label || variable.name} (not saved)`
                      : undefined
                  }
                  value={currentValues[variable.name] || ""}
                  onChange={(event) =>
                    updateCurrentValue(variable.name, event.target.value)
                  }
                  data-testid={`structured-preview-variable-${variable.name}`}
                  className="w-full rounded-md border border-border bg-bg px-3 py-2 text-sm text-text"
                />
              )}
            </label>
          ))}
        </div>
      )}
    </section>
  )
}

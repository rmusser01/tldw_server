import React from "react"

import { McpToolPicker } from "./McpToolPicker"
import { PERSONA_STARTER_COMMAND_TEMPLATES } from "./personaStarterCommandTemplates"

type DefaultCommand = {
  template_key?: string
  custom?: {
    name: string
    phrases: string[]
    tool_name: string
    slot_map: Record<string, string>
    requires_confirmation: boolean
  }
}

type SetupStarterCommandsStepProps = {
  saving: boolean
  error?: string | null
  defaultCommands?: DefaultCommand[]
  onCreateFromTemplate: (templateKey: string) => void
  onCreateMcpStarter: (toolName: string, phrase: string) => void
  onSkip: () => void
}

export const SetupStarterCommandsStep: React.FC<SetupStarterCommandsStepProps> = ({
  saving,
  error = null,
  defaultCommands,
  onCreateFromTemplate,
  onCreateMcpStarter,
  onSkip
}) => {
  const defaultTemplateKeys = React.useMemo(() => {
    if (!defaultCommands) return new Set<string>()
    const keys = new Set<string>()
    for (const cmd of defaultCommands) {
      if (cmd.template_key) keys.add(cmd.template_key)
    }
    return keys
  }, [defaultCommands])

  const [checkedKeys, setCheckedKeys] = React.useState<Set<string>>(
    () => new Set(defaultTemplateKeys)
  )

  // Sync checked keys when defaultTemplateKeys changes (new archetype selected)
  React.useEffect(() => {
    setCheckedKeys(new Set(defaultTemplateKeys))
  }, [defaultTemplateKeys])

  const [toolName, setToolName] = React.useState("")
  const [phrase, setPhrase] = React.useState("")
  const retryHint =
    error && /starter command/i.test(error)
      ? "Try a starter template again, add an MCP starter instead, or continue without starter commands."
      : null

  const handleCreateMcpStarter = React.useCallback(() => {
    const normalizedToolName = String(toolName || "").trim()
    const normalizedPhrase = String(phrase || "").trim()
    if (!normalizedToolName || !normalizedPhrase) return
    onCreateMcpStarter(normalizedToolName, normalizedPhrase)
  }, [onCreateMcpStarter, phrase, toolName])

  return (
    <div className="space-y-3">
      <div>
        <div className="text-sm font-semibold text-text">Starter commands</div>
        <div className="text-xs text-text-muted">
          Add a useful command now or continue explicitly without one.
        </div>
      </div>
      {error ? (
        <div className="rounded-md border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-200">
          <div>{error}</div>
          {retryHint ? (
            <div className="mt-1 text-xs text-red-100">{retryHint}</div>
          ) : null}
        </div>
      ) : null}
      <div className="space-y-2">
        {PERSONA_STARTER_COMMAND_TEMPLATES.map((template) => {
          const isChecked = checkedKeys.has(template.key)
          return (
            <button
              key={template.key}
              type="button"
              aria-label={template.label}
              aria-pressed={isChecked}
              className={[
                "flex w-full items-start justify-between rounded-lg border px-3 py-3 text-left disabled:cursor-not-allowed disabled:opacity-60",
                isChecked
                  ? "border-accent bg-accent/10"
                  : "border-border bg-surface2"
              ].join(" ")}
              disabled={saving}
              onClick={() => {
                const wasChecked = checkedKeys.has(template.key)
                setCheckedKeys((prev) => {
                  const next = new Set(prev)
                  if (next.has(template.key)) {
                    next.delete(template.key)
                  } else {
                    next.add(template.key)
                  }
                  return next
                })
                if (!wasChecked) {
                  onCreateFromTemplate(template.key)
                }
              }}
            >
              <div>
                <div className="text-sm font-medium text-text">{template.label}</div>
                <div className="mt-1 text-xs text-text-muted">{template.description}</div>
              </div>
              {isChecked ? (
                <span className="ml-2 mt-0.5 text-sm text-accent" aria-hidden="true">
                  &#10003;
                </span>
              ) : null}
            </button>
          )
        })}
      </div>
      <div className="space-y-2 rounded-lg border border-border bg-surface2 p-3">
        <div className="text-sm font-medium text-text">Add MCP starter</div>
        <McpToolPicker value={toolName} onChange={setToolName} disabled={saving} />
        <input
          type="text"
          value={phrase}
          aria-label="MCP starter phrase"
          placeholder="Phrase users can say"
          className="w-full rounded-md border border-border bg-surface px-3 py-2 text-sm text-text"
          disabled={saving}
          onChange={(event) => setPhrase(event.target.value)}
        />
        <button
          type="button"
          className="rounded-md border border-border px-3 py-2 text-sm font-medium text-text disabled:cursor-not-allowed disabled:opacity-60"
          disabled={saving || !String(toolName || "").trim() || !String(phrase || "").trim()}
          onClick={handleCreateMcpStarter}
        >
          Add MCP starter
        </button>
      </div>
      <button
        type="button"
        className="rounded-md border border-border px-3 py-2 text-sm font-medium text-text disabled:cursor-not-allowed disabled:opacity-60"
        disabled={saving}
        onClick={onSkip}
      >
        Continue without starter commands
      </button>
    </div>
  )
}

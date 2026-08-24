import React from "react"
import { Tooltip } from "antd"
import {
  ChevronDown,
  ChevronUp,
  Clipboard,
  Copy,
  Download,
  Plus,
  Save,
  Trash2,
  Upload
} from "lucide-react"
import { load } from "js-yaml"
import { useTranslation } from "react-i18next"

import { useConfirmDanger } from "@/components/Common/confirm-danger"
import {
  createChatMacro,
  deleteChatMacro,
  getChatMacro,
  updateChatMacro,
  validateChatMacro,
  type ChatMacroSummary
} from "@/services/chat-macros"
import { downloadBlob } from "@/utils/download-blob"
import {
  createBlankMacroDraft,
  parseMacroSource,
  readMacroImport,
  serializeGuidedMacro,
  type GuidedMacroDraft
} from "./chat-macro-editor-utils"

const MAX_BRANCHES = 6
const MAX_TIMEOUT_SECONDS = 3_600

type EditorMode = "guided" | "source"
type BusyAction = "save" | "delete" | null

export interface ChatMacroEditorProps {
  selected: ChatMacroSummary | null
  outputProfileNames: string[]
  onSaved: (name: string) => void
  onDeleted: (name: string) => void
  onCloneRequested: (macro: ChatMacroSummary) => void
}

const responseError = (status: number, error?: string): string =>
  error || `Request failed (${status})`

const clampInteger = (value: string, fallback: number, maximum: number): number => {
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return fallback
  return Math.max(1, Math.min(maximum, Math.floor(parsed)))
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value)

const sourceName = (raw: string): string | null => {
  const parsed = parseMacroSource(raw)
  if (parsed.mode === "guided") return parsed.draft.name || null

  try {
    const document = load(raw)
    return isRecord(document) && typeof document.name === "string"
      ? document.name.trim() || null
      : null
  } catch {
    return null
  }
}

const iconButtonClassName =
  "inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-md border border-border bg-surface text-text transition-colors hover:bg-surface2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus disabled:cursor-not-allowed disabled:opacity-50"

const fieldClassName =
  "mt-1 h-9 w-full rounded-md border border-border bg-background px-2.5 text-sm text-text outline-none focus:border-primary focus:ring-2 focus:ring-focus disabled:cursor-not-allowed disabled:opacity-60"

const sourceFieldClassName =
  "mt-1 min-h-[280px] w-full resize-y rounded-md border border-border bg-background px-3 py-2 font-mono text-sm leading-6 text-text outline-none focus:border-primary focus:ring-2 focus:ring-focus disabled:cursor-not-allowed disabled:opacity-60"

export const ChatMacroEditor = ({
  selected,
  outputProfileNames,
  onSaved,
  onDeleted,
  onCloneRequested
}: ChatMacroEditorProps) => {
  const { t } = useTranslation()
  const confirmDanger = useConfirmDanger()
  const inputRef = React.useRef<HTMLInputElement>(null)
  const requestGeneration = React.useRef(0)
  const [draft, setDraft] = React.useState<GuidedMacroDraft>(createBlankMacroDraft)
  const [sourceRaw, setSourceRaw] = React.useState("")
  const [serverRaw, setServerRaw] = React.useState("")
  const [mode, setMode] = React.useState<EditorMode>("guided")
  const [loading, setLoading] = React.useState(false)
  const [loadedDetailName, setLoadedDetailName] = React.useState<string | null>(null)
  const [busyAction, setBusyAction] = React.useState<BusyAction>(null)
  const [validationError, setValidationError] = React.useState<string | null>(null)
  const [validationMessage, setValidationMessage] = React.useState<string | null>(null)

  const label = React.useCallback(
    (key: string, defaultValue: string) => t(`chatMacroEditor.${key}`, defaultValue),
    [t]
  )

  const isBuiltin = selected?.source === "builtin" || selected?.immutable === true
  const isCreate = selected === null
  const isBusy = busyAction !== null
  const hasCurrentDetail = selected === null || loadedDetailName === selected.name

  React.useEffect(() => {
    const generation = ++requestGeneration.current
    setValidationError(null)
    setValidationMessage(null)

    if (!selected) {
      setDraft(createBlankMacroDraft())
      setSourceRaw("")
      setServerRaw("")
      setMode("guided")
      setLoadedDetailName(null)
      setLoading(false)
      return
    }

    setDraft(createBlankMacroDraft())
    setSourceRaw("")
    setServerRaw("")
    setMode("guided")
    setLoadedDetailName(null)
    setLoading(true)
    void (async () => {
      try {
        const response = await getChatMacro(selected.name)
        if (generation !== requestGeneration.current) return

        if (!response.ok || !response.data) {
          setValidationError(responseError(response.status, response.error))
          return
        }

        const parsed = parseMacroSource(response.data.raw)
        setServerRaw(response.data.raw)
        setSourceRaw(response.data.raw)
        if (parsed.mode === "guided") {
          setDraft(parsed.draft)
          setMode("guided")
        } else {
          setDraft((current) => ({
            ...current,
            name: response.data.definition.name || selected.name,
            command: response.data.definition.command || selected.command,
            description: response.data.definition.description || ""
          }))
          setMode("source")
        }
        setLoadedDetailName(selected.name)
      } catch (error) {
        if (generation === requestGeneration.current) {
          setValidationError(error instanceof Error ? error.message : label("detailLoadError", "Unable to load macro details."))
        }
      } finally {
        if (generation === requestGeneration.current) setLoading(false)
      }
    })()
  }, [selected])

  const updateDraft = <K extends keyof GuidedMacroDraft>(key: K, value: GuidedMacroDraft[K]) => {
    setDraft((current) => ({ ...current, [key]: value }))
  }

  const updateBranch = (
    index: number,
    key: "label" | "output" | "prompt",
    value: string
  ) => {
    setDraft((current) => ({
      ...current,
      branches: current.branches.map((branch, branchIndex) =>
        branchIndex === index ? { ...branch, [key]: value } : branch
      )
    }))
  }

  const moveBranch = (index: number, direction: -1 | 1) => {
    setDraft((current) => {
      const destination = index + direction
      if (destination < 0 || destination >= current.branches.length) return current
      const branches = [...current.branches]
      ;[branches[index], branches[destination]] = [branches[destination], branches[index]]
      return { ...current, branches }
    })
  }

  const removeBranch = (index: number) => {
    setDraft((current) => {
      if (current.branches.length <= 1) return current
      return { ...current, branches: current.branches.filter((_, branchIndex) => branchIndex !== index) }
    })
  }

  const addBranch = () => {
    setDraft((current) => {
      if (current.branches.length >= current.maxBranches || current.branches.length >= MAX_BRANCHES) {
        return current
      }
      const suffix = current.branches.reduce((largest, branch) => {
        const candidate = Number(branch.id.replace(/^branch_/, ""))
        return Number.isFinite(candidate) ? Math.max(largest, candidate) : largest
      }, 0) + 1
      return {
        ...current,
        branches: [
          ...current.branches,
          {
            id: `branch_${suffix}`,
            label: `Branch ${suffix}`,
            output: `branch_${suffix}`,
            prompt: ""
          }
        ]
      }
    })
  }

  const switchToSource = () => {
    if (mode === "source") return
    try {
      setSourceRaw(serializeGuidedMacro(draft))
      setValidationError(null)
      setValidationMessage(null)
      setMode("source")
    } catch (error) {
      setValidationError(error instanceof Error ? error.message : label("guidedSourceError", "Unable to generate YAML."))
    }
  }

  const switchToGuided = () => {
    if (mode === "guided") return
    const parsed = parseMacroSource(sourceRaw)
    if (parsed.mode === "source") {
      setValidationError(
        parsed.error || label("guidedUnavailable", "This YAML cannot be edited in Guided mode.")
      )
      return
    }
    setDraft(parsed.draft)
    setValidationError(null)
    setValidationMessage(null)
    setMode("guided")
  }

  const handleImport = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    event.target.value = ""
    if (!file) return

    try {
      const raw = await readMacroImport(file)
      const parsed = parseMacroSource(raw)
      setSourceRaw(raw)
      setServerRaw("")
      setValidationError(null)
      setValidationMessage(null)
      if (parsed.mode === "guided") {
        setDraft(selected ? { ...parsed.draft, name: selected.name } : parsed.draft)
      } else if (!selected) {
        const importedName = sourceName(raw)
        if (importedName) updateDraft("name", importedName)
      }
      setMode("source")
    } catch (error) {
      setValidationError(error instanceof Error ? error.message : label("importError", "Unable to import YAML."))
    }
  }

  const rawForSave = (): string | null => {
    if (mode === "source") return sourceRaw
    try {
      return serializeGuidedMacro(draft)
    } catch (error) {
      setValidationError(error instanceof Error ? error.message : label("guidedSourceError", "Unable to generate YAML."))
      return null
    }
  }

  const saveMacro = async (event: React.FormEvent) => {
    event.preventDefault()
    if (isBuiltin || isBusy || loading || !hasCurrentDetail) return

    const raw = rawForSave()
    if (raw === null) return

    setBusyAction("save")
    setValidationError(null)
    setValidationMessage(null)
    try {
      const validation = await validateChatMacro(raw)
      if (!validation.ok || !validation.data?.valid) {
        setValidationError(validation.data?.error || responseError(validation.status, validation.error))
        return
      }

      const expectedName = selected?.name || draft.name
      const validatedName = isRecord(validation.data.macro) && typeof validation.data.macro.name === "string"
        ? validation.data.macro.name
        : null
      if (validatedName !== expectedName) {
        setValidationError(label("validatedNameMismatch", "Validated macro name must match the selected macro."))
        return
      }

      const response = isCreate
        ? await createChatMacro({ name: draft.name, raw })
        : await updateChatMacro(selected.name, { raw })
      if (!response.ok || !response.data) {
        setValidationError(responseError(response.status, response.error))
        return
      }

      setServerRaw(response.data.raw)
      setSourceRaw(response.data.raw)
      setValidationMessage(label("validationPassed", "Server validation passed."))
      onSaved(response.data.summary.name)
    } catch (error) {
      setValidationError(error instanceof Error ? error.message : label("saveError", "Unable to save macro."))
    } finally {
      setBusyAction(null)
    }
  }

  const copyYaml = async () => {
    if (loading || !hasCurrentDetail) return
    const raw = serverRaw || rawForSave()
    if (raw === null) return
    try {
      await navigator.clipboard.writeText(raw)
      setValidationMessage(label("copied", "YAML copied."))
    } catch {
      setValidationError(label("copyError", "Unable to copy YAML."))
    }
  }

  const downloadYaml = () => {
    if (loading || !hasCurrentDetail) return
    const raw = serverRaw || rawForSave()
    if (raw === null) return
    const name = selected?.name || draft.name || "macro"
    downloadBlob(new Blob([raw], { type: "text/yaml" }), `${name}.yaml`)
  }

  const removeMacro = async () => {
    if (!selected || isBuiltin || isBusy || loading || !hasCurrentDetail) return
    setBusyAction("delete")
    try {
      const confirmed = await confirmDanger({
        title: label("deleteTitle", "Delete macro?"),
        content: label("deleteContent", "This macro will be permanently removed."),
        okText: label("confirmDelete", "Delete"),
        cancelText: label("cancel", "Cancel"),
        autoFocusButton: "cancel"
      })
      if (!confirmed) return

      const response = await deleteChatMacro(selected.name)
      if (!response.ok) {
        setValidationError(responseError(response.status, response.error))
        return
      }
      onDeleted(selected.name)
    } catch (error) {
      setValidationError(error instanceof Error ? error.message : label("deleteError", "Unable to delete macro."))
    } finally {
      setBusyAction(null)
    }
  }

  const actionLabel = busyAction === "save"
    ? label("saving", "Saving")
    : label("save", "Save macro")

  return (
    <form
      aria-label={label("editor", "Macro editor")}
      className="w-full border-y border-border py-4 text-text"
      onSubmit={(event) => void saveMacro(event)}
    >
      <div className="flex flex-wrap items-end justify-between gap-3 border-b border-border pb-3">
        <div>
          <h2 className="text-base font-semibold">{label("title", "Macro editor")}</h2>
          {selected ? <p className="mt-1 text-xs text-text-muted">/{selected.command}</p> : null}
        </div>
        <div className="flex h-9 items-center rounded-md border border-border bg-surface2 p-0.5" role="group" aria-label={label("mode", "Editor mode")}>
          <button
            type="button"
            aria-pressed={mode === "guided"}
            className={[
              "h-7 rounded-sm px-3 text-xs font-medium focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus",
              mode === "guided" ? "bg-surface text-text shadow-sm" : "text-text-muted hover:text-text"
            ].join(" ")}
            onClick={switchToGuided}
          >
            {label("guided", "Guided")}
          </button>
          <button
            type="button"
            aria-pressed={mode === "source"}
            className={[
              "h-7 rounded-sm px-3 text-xs font-medium focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus",
              mode === "source" ? "bg-surface text-text shadow-sm" : "text-text-muted hover:text-text"
            ].join(" ")}
            onClick={switchToSource}
          >
            {label("yaml", "YAML")}
          </button>
        </div>
      </div>

      <div className="grid gap-x-5 gap-y-4 py-4 xl:grid-cols-[minmax(0,0.78fr)_minmax(0,1.22fr)]">
        <section className="grid gap-3 sm:grid-cols-2 xl:col-span-2" aria-label={label("identity", "Macro identity")}>
          <label className="block text-xs font-medium text-text-muted" htmlFor="chat-macro-editor-name">
            {label("name", "Name")}
            <input
              id="chat-macro-editor-name"
              className={fieldClassName}
              value={draft.name}
              readOnly={!isCreate}
              disabled={loading}
              onChange={(event) => updateDraft("name", event.target.value)}
            />
          </label>
          <label className="block text-xs font-medium text-text-muted" htmlFor="chat-macro-editor-command">
            {label("command", "Command")}
            <input
              id="chat-macro-editor-command"
              className={fieldClassName}
              value={draft.command}
              readOnly={isBuiltin}
              disabled={loading}
              onChange={(event) => updateDraft("command", event.target.value)}
            />
          </label>
          <label className="block text-xs font-medium text-text-muted xl:col-span-2" htmlFor="chat-macro-editor-description">
            {label("description", "Description")}
            <input
              id="chat-macro-editor-description"
              className={fieldClassName}
              value={draft.description}
              disabled={loading || isBuiltin}
              onChange={(event) => updateDraft("description", event.target.value)}
            />
          </label>
        </section>

        {mode === "guided" ? (
          <>
            <section className="space-y-3 border-t border-border pt-4 xl:border-t-0 xl:pt-0" aria-label={label("execution", "Execution settings")}>
              <h3 className="text-sm font-semibold">{label("execution", "Execution settings")}</h3>
              <label className="block text-xs font-medium text-text-muted" htmlFor="chat-macro-editor-profile">
                {label("outputProfile", "Output profile")}
                <select
                  id="chat-macro-editor-profile"
                  className={fieldClassName}
                  value={draft.outputProfile}
                  disabled={loading || isBuiltin}
                  onChange={(event) => updateDraft("outputProfile", event.target.value)}
                >
                  {outputProfileNames.map((profile) => (
                    <option key={profile} value={profile}>{profile}</option>
                  ))}
                </select>
              </label>
              <div className="grid grid-cols-3 gap-2">
                <label className="block text-xs font-medium text-text-muted" htmlFor="chat-macro-editor-max-branches">
                  {label("maxBranches", "Max branches")}
                  <input
                    id="chat-macro-editor-max-branches"
                    type="number"
                    min={1}
                    max={MAX_BRANCHES}
                    className={fieldClassName}
                    value={draft.maxBranches}
                    disabled={loading || isBuiltin}
                    onChange={(event) => {
                      const maxBranches = clampInteger(event.target.value, draft.maxBranches, MAX_BRANCHES)
                      setDraft((current) => ({
                        ...current,
                        maxBranches,
                        maxConcurrency: Math.min(current.maxConcurrency, maxBranches)
                      }))
                    }}
                  />
                </label>
                <label className="block text-xs font-medium text-text-muted" htmlFor="chat-macro-editor-concurrency">
                  {label("concurrency", "Concurrency")}
                  <input
                    id="chat-macro-editor-concurrency"
                    type="number"
                    min={1}
                    max={draft.maxBranches}
                    className={fieldClassName}
                    value={draft.maxConcurrency}
                    disabled={loading || isBuiltin}
                    onChange={(event) => updateDraft(
                      "maxConcurrency",
                      clampInteger(event.target.value, draft.maxConcurrency, draft.maxBranches)
                    )}
                  />
                </label>
                <label className="block text-xs font-medium text-text-muted" htmlFor="chat-macro-editor-timeout">
                  {label("timeout", "Timeout (s)")}
                  <input
                    id="chat-macro-editor-timeout"
                    type="number"
                    min={1}
                    max={MAX_TIMEOUT_SECONDS}
                    className={fieldClassName}
                    value={draft.timeoutSeconds}
                    disabled={loading || isBuiltin}
                    onChange={(event) => updateDraft(
                      "timeoutSeconds",
                      clampInteger(event.target.value, draft.timeoutSeconds, MAX_TIMEOUT_SECONDS)
                    )}
                  />
                </label>
              </div>
              <label className="block text-xs font-medium text-text-muted" htmlFor="chat-macro-editor-merge-prompt">
                {label("mergePrompt", "Merge prompt")}
                <textarea
                  id="chat-macro-editor-merge-prompt"
                  className="mt-1 min-h-28 w-full resize-y rounded-md border border-border bg-background px-2.5 py-2 text-sm text-text outline-none focus:border-primary focus:ring-2 focus:ring-focus disabled:cursor-not-allowed disabled:opacity-60"
                  value={draft.merge.prompt}
                  disabled={loading || isBuiltin}
                  onChange={(event) => setDraft((current) => ({
                    ...current,
                    merge: { ...current.merge, prompt: event.target.value }
                  }))}
                />
              </label>
            </section>

            <section className="min-w-0 space-y-3 border-t border-border pt-4 xl:border-l xl:border-t-0 xl:pl-5 xl:pt-0" aria-label={label("branches", "Branches")}>
              <div className="flex items-center justify-between gap-3">
                <h3 className="text-sm font-semibold">{label("branches", "Branches")}</h3>
                <button
                  type="button"
                  className="inline-flex h-9 items-center gap-1.5 rounded-md border border-border bg-surface px-3 text-sm font-medium text-text transition-colors hover:bg-surface2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus disabled:cursor-not-allowed disabled:opacity-50"
                  disabled={loading || isBuiltin || draft.branches.length >= draft.maxBranches || draft.branches.length >= MAX_BRANCHES}
                  onClick={addBranch}
                >
                  <Plus aria-hidden="true" size={16} />
                  {label("addBranch", "Add branch")}
                </button>
              </div>
              <div className="divide-y divide-border border-y border-border">
                {draft.branches.map((branch, index) => (
                  <div key={branch.id} className="grid gap-2 py-3 sm:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_auto]">
                    <label className="block text-xs font-medium text-text-muted" htmlFor={`chat-macro-editor-branch-label-${branch.id}`}>
                      {label("branchLabel", "Branch label")}
                      <input
                        id={`chat-macro-editor-branch-label-${branch.id}`}
                        aria-label={`${label("branchLabel", "Branch label")} ${index + 1}`}
                        className={fieldClassName}
                        value={branch.label}
                        disabled={loading || isBuiltin}
                        onChange={(event) => updateBranch(index, "label", event.target.value)}
                      />
                    </label>
                    <label className="block text-xs font-medium text-text-muted" htmlFor={`chat-macro-editor-branch-output-${branch.id}`}>
                      {label("branchOutput", "Branch output")}
                      <input
                        id={`chat-macro-editor-branch-output-${branch.id}`}
                        aria-label={`${label("branchOutput", "Branch output")} ${index + 1}`}
                        className={fieldClassName}
                        value={branch.output}
                        disabled={loading || isBuiltin}
                        onChange={(event) => updateBranch(index, "output", event.target.value)}
                      />
                    </label>
                    <div className="flex items-end gap-1">
                      <Tooltip title={label("moveBranchUp", "Move branch up")}>
                        <button
                          type="button"
                          aria-label={`${label("moveBranchUp", "Move branch up")} ${index + 1}`}
                          className={iconButtonClassName}
                          disabled={loading || isBuiltin || index === 0}
                          onClick={() => moveBranch(index, -1)}
                        >
                          <ChevronUp aria-hidden="true" size={16} />
                        </button>
                      </Tooltip>
                      <Tooltip title={label("moveBranchDown", "Move branch down")}>
                        <button
                          type="button"
                          aria-label={`${label("moveBranchDown", "Move branch down")} ${index + 1}`}
                          className={iconButtonClassName}
                          disabled={loading || isBuiltin || index === draft.branches.length - 1}
                          onClick={() => moveBranch(index, 1)}
                        >
                          <ChevronDown aria-hidden="true" size={16} />
                        </button>
                      </Tooltip>
                      <Tooltip title={label("deleteBranch", "Delete branch")}>
                        <button
                          type="button"
                          aria-label={`${label("deleteBranch", "Delete branch")} ${index + 1}`}
                          className={iconButtonClassName}
                          disabled={loading || isBuiltin || draft.branches.length === 1}
                          onClick={() => removeBranch(index)}
                        >
                          <Trash2 aria-hidden="true" size={16} />
                        </button>
                      </Tooltip>
                    </div>
                    <label className="block text-xs font-medium text-text-muted sm:col-span-3" htmlFor={`chat-macro-editor-branch-prompt-${branch.id}`}>
                      {label("branchPrompt", "Branch prompt")}
                      <textarea
                        id={`chat-macro-editor-branch-prompt-${branch.id}`}
                        aria-label={`${label("branchPrompt", "Branch prompt")} ${index + 1}`}
                        className="mt-1 min-h-24 w-full resize-y rounded-md border border-border bg-background px-2.5 py-2 text-sm text-text outline-none focus:border-primary focus:ring-2 focus:ring-focus disabled:cursor-not-allowed disabled:opacity-60"
                        value={branch.prompt}
                        disabled={loading || isBuiltin}
                        onChange={(event) => updateBranch(index, "prompt", event.target.value)}
                      />
                    </label>
                  </div>
                ))}
              </div>
            </section>
          </>
        ) : (
          <section className="min-w-0 xl:col-span-2" aria-label={label("source", "Macro source")}>
            <label className="block text-xs font-medium text-text-muted" htmlFor="chat-macro-editor-yaml">
              {label("sourceYaml", "Macro YAML")}
              <textarea
                id="chat-macro-editor-yaml"
                className={sourceFieldClassName}
                value={sourceRaw}
                disabled={loading || isBuiltin}
                onChange={(event) => setSourceRaw(event.target.value)}
              />
            </label>
          </section>
        )}
      </div>

      <input
        ref={inputRef}
        type="file"
        className="sr-only"
        accept=".yaml,.yml,text/yaml,text/plain"
        aria-hidden="true"
        tabIndex={-1}
        onChange={(event) => void handleImport(event)}
      />
      <div className="flex flex-wrap items-center gap-2 border-t border-border pt-3">
        <Tooltip title={label("upload", "Upload macro YAML")}>
          <button
            type="button"
            aria-label={label("upload", "Upload macro YAML")}
            className={iconButtonClassName}
            disabled={loading || isBusy || isBuiltin}
            onClick={() => inputRef.current?.click()}
          >
            <Upload aria-hidden="true" size={16} />
          </button>
        </Tooltip>
        <Tooltip title={label("download", "Download macro YAML")}>
          <button
            type="button"
            aria-label={label("download", "Download macro YAML")}
            className={iconButtonClassName}
            disabled={loading || !hasCurrentDetail || !serverRaw && !sourceRaw && !draft.name}
            onClick={downloadYaml}
          >
            <Download aria-hidden="true" size={16} />
          </button>
        </Tooltip>
        <Tooltip title={label("copy", "Copy macro YAML")}>
          <button
            type="button"
            aria-label={label("copy", "Copy macro YAML")}
            className={iconButtonClassName}
            disabled={loading || !hasCurrentDetail || !serverRaw && !sourceRaw && !draft.name}
            onClick={() => void copyYaml()}
          >
            <Clipboard aria-hidden="true" size={16} />
          </button>
        </Tooltip>
        <div className="ml-auto flex items-center gap-2">
          {isBuiltin && selected ? (
            <button
              type="button"
              className="inline-flex h-9 items-center gap-1.5 rounded-md border border-border bg-surface px-3 text-sm font-medium text-text transition-colors hover:bg-surface2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus"
              onClick={() => onCloneRequested(selected)}
            >
              <Copy aria-hidden="true" size={16} />
              {label("clone", "Clone macro")}
            </button>
          ) : (
            <button
              type="submit"
              className="inline-flex h-9 min-w-[118px] items-center justify-center gap-1.5 rounded-md bg-primary px-3 text-sm font-medium text-white transition-colors hover:bg-primaryStrong focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus disabled:cursor-not-allowed disabled:opacity-50"
              disabled={loading || !hasCurrentDetail || isBusy}
            >
              <Save aria-hidden="true" size={16} />
              {actionLabel}
            </button>
          )}
          {!isCreate && !isBuiltin ? (
            <button
              type="button"
              aria-label={label("delete", "Delete macro")}
              className="inline-flex h-9 w-9 items-center justify-center rounded-md border border-danger/50 bg-surface text-danger transition-colors hover:bg-danger/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus disabled:cursor-not-allowed disabled:opacity-50"
              disabled={loading || !hasCurrentDetail || isBusy}
              onClick={() => void removeMacro()}
            >
              <Trash2 aria-hidden="true" size={16} />
            </button>
          ) : null}
        </div>
      </div>
      <div className="mt-2 min-h-5 text-sm" aria-live="polite">
        {validationError ? <p className="text-danger" role="alert">{validationError}</p> : null}
        {!validationError && validationMessage ? <p className="text-success" role="status">{validationMessage}</p> : null}
      </div>
    </form>
  )
}

export default ChatMacroEditor

import React, { useEffect, useMemo, useRef, useState } from "react"
import { ChevronDown, Sparkles } from "lucide-react"
import { cn } from "@/libs/utils"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { getProviderDisplayName } from "@/utils/provider-registry"

type LlmProviderConfig = {
  name: string
  display_name?: string
  models?: string[]
  models_info?: Array<{
    id?: string
    name?: string
    model_id?: string
    display_name?: string
  }>
  default_model?: string | null
}

type LlmProvidersResponse = {
  providers?: LlmProviderConfig[]
  default_provider?: string | null
}

const SERVER_DEFAULT_PROVIDER_VALUE = "__server_default__"

type AnswerModelMenuProps = {
  generationProvider: string | null
  generationModel: string | null
  onGenerationProviderChange: (provider: string | null) => void
  onGenerationModelChange: (model: string | null) => void
  className?: string
  menuAlign?: "left" | "right"
}

export function AnswerModelMenu({
  generationProvider,
  generationModel,
  onGenerationProviderChange,
  onGenerationModelChange,
  className,
  menuAlign = "left",
}: AnswerModelMenuProps) {
  const [open, setOpen] = useState(false)
  const [providerCatalog, setProviderCatalog] =
    useState<LlmProvidersResponse | null>(null)
  const [providerStatus, setProviderStatus] = useState<"loading" | "ready" | "error">("loading")
  const containerRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    let cancelled = false

    void (async () => {
      try {
        setProviderStatus("loading")
        await tldwClient.initialize()
        const response = (await tldwClient.getProviders()) as LlmProvidersResponse
        if (!cancelled) {
          setProviderCatalog(response)
          setProviderStatus("ready")
        }
      } catch {
        if (!cancelled) {
          setProviderCatalog(null)
          setProviderStatus("error")
        }
      }
    })()

    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (!open) return

    const handlePointerDown = (event: MouseEvent) => {
      const target = event.target as Node
      if (containerRef.current?.contains(target)) return
      setOpen(false)
    }

    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setOpen(false)
      }
    }

    document.addEventListener("mousedown", handlePointerDown)
    document.addEventListener("keydown", handleEscape)
    return () => {
      document.removeEventListener("mousedown", handlePointerDown)
      document.removeEventListener("keydown", handleEscape)
    }
  }, [open])

  const providerEntries = useMemo(
    () => providerCatalog?.providers ?? [],
    [providerCatalog]
  )

  const effectiveProviderKey =
    generationProvider || providerCatalog?.default_provider || null

  const selectedProviderConfig = useMemo(
    () =>
      providerEntries.find((provider) => provider.name === effectiveProviderKey) ??
      null,
    [effectiveProviderKey, providerEntries]
  )

  const providerOptions = useMemo(
    () => [
      { value: SERVER_DEFAULT_PROVIDER_VALUE, label: "Server default" },
      ...providerEntries.map((provider) => ({
        value: provider.name,
        label:
          provider.display_name ||
          getProviderDisplayName(provider.name) ||
          provider.name,
      })),
    ],
    [providerEntries]
  )

  const modelSuggestions = useMemo(() => {
    const seen = new Set<string>()
    const suggestions: string[] = []
    const push = (value?: string | null) => {
      const normalized = String(value || "").trim()
      if (!normalized || seen.has(normalized)) return
      seen.add(normalized)
      suggestions.push(normalized)
    }

    selectedProviderConfig?.models?.forEach(push)
    selectedProviderConfig?.models_info?.forEach((model) =>
      push(model.model_id || model.id || model.name)
    )
    push(selectedProviderConfig?.default_model)
    return suggestions
  }, [selectedProviderConfig])

  const providerLabel = effectiveProviderKey
    ? selectedProviderConfig?.display_name ||
      getProviderDisplayName(effectiveProviderKey) ||
      effectiveProviderKey
    : "Default"

  const trimmedGenerationModel = generationModel?.trim() || ""
  const summary = trimmedGenerationModel
    ? trimmedGenerationModel
    : generationProvider
      ? `${providerLabel} default`
      : "Server default"
  const resolvedDefaultDetail =
    !generationProvider && !trimmedGenerationModel && effectiveProviderKey
      ? ` (${providerLabel} default when available)`
      : ""

  return (
    <div className={cn("relative", className)} ref={containerRef}>
      <button
        type="button"
        onClick={() => setOpen((previous) => !previous)}
        className="inline-flex h-7 items-center gap-1 rounded-full border border-border bg-surface px-2.5 text-[11px] font-medium text-text-muted hover:bg-surface2 hover:text-text transition-colors"
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-label="Choose answer model"
        title={`Answer generation uses ${summary}${resolvedDefaultDetail}`}
      >
        <Sparkles className="h-3.5 w-3.5" />
        <span className="max-w-[11rem] truncate">AI: {summary}</span>
        <ChevronDown className="h-3 w-3" />
      </button>

      {open ? (
        <div
          role="dialog"
          aria-label="Answer model controls"
          className={cn(
            "absolute z-40 mt-2 w-[22rem] max-w-[85vw] rounded-lg border border-border/80 bg-surface p-3 shadow-lg",
            menuAlign === "right" ? "right-0" : "left-0"
          )}
        >
          <div className="mb-2">
            <p className="text-xs font-semibold uppercase tracking-wide text-text-muted">
              Answer model
            </p>
            <p className="text-xs text-text-muted">
              Applies to final answer generation for this search session.
            </p>
          </div>

          {providerStatus === "loading" ? (
            <p role="status" className="mb-2 rounded-md border border-border bg-surface2 px-2 py-1.5 text-xs text-text-muted">
              Loading answer providers...
            </p>
          ) : null}

          {providerStatus === "error" ? (
            <p role="status" className="mb-2 rounded-md border border-warn/40 bg-warn/10 px-2 py-1.5 text-xs text-warn">
              Provider list failed to load. You can still type a model manually.
            </p>
          ) : null}

          <label className="mb-2 block text-[11px] font-medium text-text-muted">
            Provider
            <select
              aria-label="Answer provider"
              value={generationProvider ?? SERVER_DEFAULT_PROVIDER_VALUE}
              onChange={(event) =>
                onGenerationProviderChange(
                  event.target.value === SERVER_DEFAULT_PROVIDER_VALUE
                    ? null
                    : event.target.value
                )
              }
              className="mt-1 h-9 w-full rounded-md border border-border bg-surface2 px-2 text-sm text-text outline-none focus:border-primary"
            >
              {providerOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>

          <label className="block text-[11px] font-medium text-text-muted">
            Model
            <input
              aria-label="Answer model"
              list="knowledge-answer-model-options"
              value={generationModel ?? ""}
              onChange={(event) =>
                onGenerationModelChange(event.target.value.trim() || null)
              }
              placeholder={
                selectedProviderConfig?.default_model
                  ? `Default: ${selectedProviderConfig.default_model}`
                  : "Use provider default"
              }
              className="mt-1 h-9 w-full rounded-md border border-border bg-surface2 px-2 text-sm text-text outline-none placeholder:text-text-muted focus:border-primary"
            />
            <datalist id="knowledge-answer-model-options">
              {modelSuggestions.map((model) => (
                <option key={model} value={model} />
              ))}
            </datalist>
          </label>
        </div>
      ) : null}
    </div>
  )
}

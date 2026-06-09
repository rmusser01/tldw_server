import { Plus, Search, X } from "lucide-react"
import type {
  ExplainerDepthPreset,
  ExplainerGrounding,
  ExplainerOutputIntent,
  ExplainerSelectedSource,
  ExplainerSourceCandidate
} from "./types"

type ExplainerSourcePickerProps = {
  query: string
  results: ExplainerSourceCandidate[]
  selectedSources: ExplainerSelectedSource[]
  grounding: ExplainerGrounding
  outputIntent: ExplainerOutputIntent
  depthPreset: ExplainerDepthPreset
  isSearching?: boolean
  isCreating?: boolean
  onQueryChange: (value: string) => void
  onSearch: () => void
  onAddSource: (source: ExplainerSourceCandidate) => void
  onRemoveSource: (source: ExplainerSelectedSource) => void
  onGroundingChange: (value: ExplainerGrounding) => void
  onOutputIntentChange: (value: ExplainerOutputIntent) => void
  onDepthPresetChange: (value: ExplainerDepthPreset) => void
  onCreate: () => void
}

const sourceKey = (source: Pick<ExplainerSelectedSource, "sourceId" | "sourceType">) =>
  `${source.sourceType}:${source.sourceId}`

export const ExplainerSourcePicker = ({
  query,
  results,
  selectedSources,
  grounding,
  outputIntent,
  depthPreset,
  isSearching = false,
  isCreating = false,
  onQueryChange,
  onSearch,
  onAddSource,
  onRemoveSource,
  onGroundingChange,
  onOutputIntentChange,
  onDepthPresetChange,
  onCreate
}: ExplainerSourcePickerProps) => {
  const selectedKeys = new Set(selectedSources.map(sourceKey))

  return (
    <section
      aria-label="Source setup"
      className="grid gap-4 border-b border-border bg-surface px-4 py-4 xl:grid-cols-[minmax(320px,1fr)_minmax(260px,360px)_220px_auto]"
    >
      <div className="grid gap-3">
        <label className="grid gap-2">
          <span className="text-xs font-semibold uppercase tracking-wide text-text-muted">
            Source search
          </span>
          <div className="flex gap-2">
            <input
              className="h-10 min-w-0 flex-1 rounded-md border border-border bg-surface2 px-3 text-sm text-text outline-none focus:border-primary focus:ring-2 focus:ring-focus"
              value={query}
              onChange={(event) => onQueryChange(event.target.value)}
              placeholder="Search media and notes"
            />
            <button
              type="button"
              className="inline-flex h-10 items-center gap-2 rounded-md border border-border bg-surface px-3 text-sm font-medium text-text transition-colors hover:bg-surface2 disabled:cursor-not-allowed disabled:opacity-60"
              disabled={!query.trim() || isSearching}
              onClick={onSearch}
            >
              <Search className="h-4 w-4" aria-hidden="true" />
              Search sources
            </button>
          </div>
        </label>

        <div className="grid max-h-40 min-w-0 gap-2 overflow-auto overflow-x-hidden" aria-live="polite">
          {results.map((source) => {
            const key = sourceKey(source)
            const isSelected = selectedKeys.has(key)
            return (
              <div
                key={key}
                className="grid min-w-0 grid-cols-[minmax(0,1fr)_auto] items-start gap-3 rounded-md border border-border bg-surface2 px-3 py-2"
              >
                <div className="min-w-0">
                  <p className="truncate text-sm font-medium text-text">{source.title}</p>
                  <p className="truncate text-xs text-text-muted">
                    {source.sourceType}
                    {source.description ? ` · ${source.description}` : ""}
                  </p>
                </div>
                <button
                  type="button"
                  className="inline-flex h-8 shrink-0 items-center gap-1 rounded-md border border-border bg-surface px-2 text-xs font-medium text-text transition-colors hover:bg-surface2 disabled:cursor-not-allowed disabled:opacity-60"
                  disabled={isSelected}
                  aria-label={`Add ${source.title}`}
                  onClick={() => onAddSource(source)}
                >
                  <Plus className="h-3.5 w-3.5" aria-hidden="true" />
                  Add
                </button>
              </div>
            )
          })}
        </div>
      </div>

      <div className="grid gap-2">
        <h2 className="text-xs font-semibold uppercase tracking-wide text-text-muted">
          Selected sources
        </h2>
        <div className="grid max-h-44 gap-2 overflow-auto rounded-md border border-border bg-surface2 p-2">
          {selectedSources.length === 0 ? (
            <p className="px-2 py-3 text-sm text-text-muted">
              Select at least one source for source-only explanations.
            </p>
          ) : (
            selectedSources.map((source) => (
              <div
                key={sourceKey(source)}
                className="grid min-w-0 grid-cols-[minmax(0,1fr)_auto] items-center gap-2 rounded bg-surface px-2 py-2"
              >
                <span className="min-w-0 truncate text-sm text-text">{source.title}</span>
                <button
                  type="button"
                  className="inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-md text-text-muted transition-colors hover:bg-surface2 hover:text-text"
                  aria-label={`Remove ${source.title}`}
                  onClick={() => onRemoveSource(source)}
                >
                  <X className="h-4 w-4" aria-hidden="true" />
                </button>
              </div>
            ))
          )}
        </div>
      </div>

      <div className="grid gap-3">
        <label className="grid gap-2">
          <span className="text-xs font-semibold uppercase tracking-wide text-text-muted">
            Grounding mode
          </span>
          <select
            className="h-10 rounded-md border border-border bg-surface2 px-3 text-sm text-text outline-none focus:border-primary focus:ring-2 focus:ring-focus"
            value={grounding}
            onChange={(event) => onGroundingChange(event.target.value as ExplainerGrounding)}
          >
            <option value="source_only">Source-only</option>
            <option value="source_led">Source-led</option>
            <option value="open">Open explainer</option>
          </select>
        </label>
        <label className="grid gap-2">
          <span className="text-xs font-semibold uppercase tracking-wide text-text-muted">
            Output intent
          </span>
          <select
            className="h-10 rounded-md border border-border bg-surface2 px-3 text-sm text-text outline-none focus:border-primary focus:ring-2 focus:ring-focus"
            value={outputIntent}
            onChange={(event) => onOutputIntentChange(event.target.value as ExplainerOutputIntent)}
          >
            <option value="explain">Explain</option>
            <option value="plan">Plan</option>
            <option value="both">Both</option>
          </select>
        </label>
        <label className="grid gap-2">
          <span className="text-xs font-semibold uppercase tracking-wide text-text-muted">
            Depth preset
          </span>
          <select
            className="h-10 rounded-md border border-border bg-surface2 px-3 text-sm text-text outline-none focus:border-primary focus:ring-2 focus:ring-focus"
            value={depthPreset}
            onChange={(event) => onDepthPresetChange(event.target.value as ExplainerDepthPreset)}
          >
            <option value="quick">Quick</option>
            <option value="standard">Standard</option>
            <option value="deep">Deep</option>
          </select>
        </label>
      </div>

      <div className="flex items-end">
        <button
          type="button"
          className="inline-flex h-10 items-center gap-2 rounded-md bg-primary px-4 text-sm font-semibold text-white transition-colors hover:bg-primaryStrong disabled:cursor-not-allowed disabled:bg-surface2 disabled:text-text-muted"
          disabled={selectedSources.length === 0 || isCreating}
          onClick={onCreate}
        >
          <Plus className="h-4 w-4" aria-hidden="true" />
          Create Explainer
        </button>
      </div>
    </section>
  )
}

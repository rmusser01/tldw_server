import { Play } from "lucide-react"
import type { ExplainerDepthPreset, ExplainerOutputIntent } from "./types"

type ExplainerGoalComposerProps = {
  goal: string
  outputIntent: ExplainerOutputIntent
  depthPreset: ExplainerDepthPreset
  isCreating?: boolean
  onGoalChange: (value: string) => void
  onOutputIntentChange: (value: ExplainerOutputIntent) => void
  onDepthPresetChange: (value: ExplainerDepthPreset) => void
  onCreate: () => void
}

export const ExplainerGoalComposer = ({
  goal,
  outputIntent,
  depthPreset,
  isCreating = false,
  onGoalChange,
  onOutputIntentChange,
  onDepthPresetChange,
  onCreate
}: ExplainerGoalComposerProps) => (
  <section
    aria-label="Goal setup"
    className="grid gap-4 border-b border-border bg-surface px-4 py-4 lg:grid-cols-[minmax(280px,1fr)_220px_180px_auto]"
  >
    <label className="grid gap-2">
      <span className="text-xs font-semibold uppercase tracking-wide text-text-muted">
        Learning goal
      </span>
      <textarea
        className="min-h-[88px] rounded-md border border-border bg-surface2 px-3 py-2 text-sm text-text outline-none transition-colors focus:border-primary focus:ring-2 focus:ring-focus"
        value={goal}
        onChange={(event) => onGoalChange(event.target.value)}
        placeholder="Explain transformer attention"
      />
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

    <div className="flex items-end">
      <button
        type="button"
        className="inline-flex h-10 items-center gap-2 rounded-md bg-primary px-4 text-sm font-semibold text-white transition-colors hover:bg-primaryStrong disabled:cursor-not-allowed disabled:bg-surface2 disabled:text-text-muted"
        disabled={!goal.trim() || isCreating}
        onClick={onCreate}
      >
        <Play className="h-4 w-4" aria-hidden="true" />
        Create Explainer
      </button>
    </div>
  </section>
)

import { Play } from "lucide-react"
import type { ExplainerDepthPreset, ExplainerOutputIntent } from "./types"

type ExplainerTemplate = {
  name: string
  description: string
  goal: string
  outputIntent: ExplainerOutputIntent
  depthPreset: ExplainerDepthPreset
}

const TEMPLATES: ExplainerTemplate[] = [
  {
    name: "Explain to a newcomer",
    description: "Plain-language walkthrough for someone new to the field.",
    goal: "Explain <topic> to someone new to the field, using plain language and concrete examples.",
    outputIntent: "explain",
    depthPreset: "standard"
  },
  {
    name: "Prepare a study plan",
    description: "A sequenced plan with milestones and practice.",
    goal: "Build a study plan for learning <topic>, with milestones and practice exercises.",
    outputIntent: "plan",
    depthPreset: "deep"
  },
  {
    name: "Quick refresher",
    description: "The essentials in a few minutes.",
    goal: "Give me a quick refresher on <topic> — just the essentials I am likely to have forgotten.",
    outputIntent: "explain",
    depthPreset: "quick"
  },
  {
    name: "Deep dive with practice",
    description: "Full depth plus hands-on next steps.",
    goal: "Explain <topic> in depth and give me hands-on practice steps to test my understanding.",
    outputIntent: "both",
    depthPreset: "deep"
  }
]

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
    className="grid gap-4 border-b border-border bg-surface px-4 py-4"
  >
    <section aria-label="Explainer templates" className="grid gap-2">
      <h2 className="text-xs font-semibold uppercase tracking-wide text-text-muted">
        Start from a template
      </h2>
      <div className="flex flex-wrap gap-2">
        {TEMPLATES.map((template) => (
          <button
            key={template.name}
            type="button"
            title={template.description}
            className="inline-flex h-8 items-center rounded-full border border-border bg-surface2 px-3 text-xs font-medium text-text-muted transition-colors hover:border-primary hover:text-text"
            onClick={() => {
              onGoalChange(template.goal)
              onOutputIntentChange(template.outputIntent)
              onDepthPresetChange(template.depthPreset)
            }}
          >
            {template.name}
          </button>
        ))}
      </div>
    </section>

    <div className="grid gap-4 lg:grid-cols-[minmax(280px,1fr)_220px_180px_auto]">
    <div className="grid gap-2">
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
      <p className="text-xs text-text-muted">
        Goal sessions use open grounding: the model answers from its own knowledge, without
        source citations. Use the Sources tab for cited explanations.
      </p>
    </div>

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
    </div>
  </section>
)

import { RotateCw } from "lucide-react"
import type { ExplainerNode, ExplainerSession } from "./types"
import {
  getExplainerEvidenceLabel,
  getExplainerNodeStatusLabel
} from "./tree"

type ExplainerDetailPanelProps = {
  session: ExplainerSession | null
  node: ExplainerNode | null
  isExpanding?: boolean
  onExpand: (nodeId: string) => void
}

export const ExplainerDetailPanel = ({
  session,
  node,
  isExpanding = false,
  onExpand
}: ExplainerDetailPanelProps) => {
  if (!session || !node) {
    return (
      <section
        aria-label="Explainer detail"
        className="flex min-h-0 flex-1 items-center justify-center bg-bg p-6"
      >
        <div className="max-w-md rounded-lg border border-border bg-surface px-5 py-6 text-center">
          <h2 className="text-base font-semibold text-text">No explainer selected</h2>
          <p className="mt-2 text-sm text-text-muted">
            Create a goal session or select sources to begin a persisted explanation.
          </p>
        </div>
      </section>
    )
  }

  return (
    <section
      aria-label="Explainer detail"
      className="min-h-0 flex-1 overflow-auto bg-bg"
    >
      <article className="mx-auto flex max-w-3xl flex-col gap-5 px-6 py-6">
        <header className="border-b border-border pb-4">
          <div className="mb-3 flex flex-wrap items-center gap-2 text-xs">
            <span className="rounded-full bg-surface px-2 py-1 font-medium text-text-muted">
              {getExplainerNodeStatusLabel(node.status)}
            </span>
            <span className="rounded-full bg-accent/10 px-2 py-1 font-medium text-accent">
              {getExplainerEvidenceLabel(node.evidenceState)}
            </span>
            {node.outsideKnowledgeUsed ? (
              <span className="rounded-full bg-warn/10 px-2 py-1 font-medium text-warn">
                Outside knowledge used
              </span>
            ) : null}
          </div>
          <h2 className="text-2xl font-semibold leading-tight text-text">{node.title}</h2>
          <p className="mt-2 text-sm text-text-muted">
            {session.outputIntent === "both" ? "Explain and plan" : session.outputIntent}
            {" · "}
            {session.grounding.replace("_", "-")}
            {" · "}
            {session.depthPreset}
          </p>
        </header>

        <div className="prose prose-sm max-w-none text-text">
          {node.body ? (
            <p className="whitespace-pre-wrap text-[15px] leading-7">{node.body}</p>
          ) : (
            <p className="rounded-md border border-border bg-surface px-4 py-4 text-sm text-text-muted">
              This node has no generated body yet.
            </p>
          )}
        </div>

        {node.questionOptions?.length ? (
          <section aria-label="Clarifying answers" className="grid gap-2">
            <h3 className="text-sm font-semibold text-text">Clarifying answers</h3>
            <div className="flex flex-wrap gap-2">
              {node.questionOptions.map((option) => {
                const id = typeof option.id === "string" ? option.id : ""
                const label = typeof option.label === "string" ? option.label : id
                const selected = id && id === node.selectedOptionId
                return (
                  <span
                    key={id || label}
                    className={[
                      "rounded-full border px-3 py-1 text-xs font-medium",
                      selected
                        ? "border-primary bg-primary/10 text-primary"
                        : "border-border bg-surface text-text-muted"
                    ].join(" ")}
                  >
                    {label}
                  </span>
                )
              })}
            </div>
          </section>
        ) : null}

        <section aria-label="Citations" className="grid gap-3">
          <h3 className="text-sm font-semibold text-text">Citations</h3>
          {node.citations.length === 0 ? (
            <p className="rounded-md border border-border bg-surface px-4 py-3 text-sm text-text-muted">
              No citations recorded for this node.
            </p>
          ) : (
            node.citations.map((citation) => (
              <figure
                key={citation.id}
                className="rounded-md border border-border bg-surface px-4 py-3"
              >
                <figcaption className="mb-2 text-sm font-semibold text-text">
                  {citation.title}
                  {citation.locationLabel ? (
                    <span className="font-normal text-text-muted">
                      {" "}
                      ({citation.locationLabel})
                    </span>
                  ) : null}
                </figcaption>
                <blockquote className="text-sm leading-6 text-text-muted">
                  {citation.excerpt}
                </blockquote>
              </figure>
            ))
          )}
        </section>

        <footer className="flex flex-wrap items-center gap-2 border-t border-border pt-4">
          <button
            type="button"
            className="inline-flex h-9 items-center gap-2 rounded-md bg-primary px-3 text-sm font-semibold text-white transition-colors hover:bg-primaryStrong disabled:cursor-not-allowed disabled:bg-surface2 disabled:text-text-muted"
            disabled={isExpanding}
            onClick={() => onExpand(node.id)}
          >
            <RotateCw className="h-4 w-4" aria-hidden="true" />
            Expand node
          </button>
        </footer>
      </article>
    </section>
  )
}

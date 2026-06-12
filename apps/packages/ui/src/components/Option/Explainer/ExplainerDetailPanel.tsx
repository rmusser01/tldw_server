import { useEffect, useState } from "react"
import { Sparkles, Trash2 } from "lucide-react"
import type { ExplainerNode, ExplainerSession } from "./types"
import {
  getExplainerDepthLabel,
  getExplainerEvidenceChipClass,
  getExplainerEvidenceDescription,
  getExplainerEvidenceLabel,
  getExplainerGroundingLabel,
  getExplainerIntentLabel,
  getExplainerNodeStatusLabel
} from "./tree"

// Media items use the library's established deep-link; notes have no
// single-note URL yet, so fall back to the manager. External URLs win.
const citationHref = (citation: { url?: string | null; sourceType: string; sourceId: string }): string => {
  if (citation.url) return citation.url
  if (citation.sourceType === "media") return `/media?id=${encodeURIComponent(citation.sourceId)}`
  if (citation.sourceType === "note") return "/notes"
  return "/media"
}

type ExplainerAnswerPayload = {
  selectedOptionId?: string
  selectedCustomAnswer?: string
}

type ExplainerDetailPanelProps = {
  session: ExplainerSession | null
  node: ExplainerNode | null
  isExpanding?: boolean
  generatingNodeId?: string | null
  sectionRef?: React.Ref<HTMLElement>
  onExpand: (nodeId: string) => void
  onDeleteNode?: (nodeId: string) => void
  onAnswerQuestion?: (nodeId: string, answer: ExplainerAnswerPayload) => void
}

export const ExplainerDetailPanel = ({
  session,
  node,
  isExpanding = false,
  generatingNodeId = null,
  sectionRef,
  onExpand,
  onDeleteNode,
  onAnswerQuestion
}: ExplainerDetailPanelProps) => {
  const [confirmingDelete, setConfirmingDelete] = useState(false)
  const [customAnswer, setCustomAnswer] = useState("")

  useEffect(() => {
    setConfirmingDelete(false)
    setCustomAnswer("")
  }, [node?.id])

  if (!session || !node) {
    return (
      <section
        ref={sectionRef}
        aria-label="Explainer detail"
        className="flex min-h-0 flex-1 items-center justify-center bg-bg p-6"
      >
        <div className="max-w-md rounded-lg border border-border bg-surface px-5 py-6 text-center">
          <h2 className="text-base font-semibold text-text">No explainer yet</h2>
          <p className="mt-2 text-sm text-text-muted">
            Set a learning goal or pick sources to build an explanation that is saved
            automatically and backed by citations.
          </p>
        </div>
      </section>
    )
  }

  const generating = node.id === generatingNodeId

  return (
    <section
      ref={sectionRef}
      aria-label="Explainer detail"
      className="min-h-0 flex-1 overflow-auto bg-bg"
    >
      <article className="mx-auto flex max-w-3xl flex-col gap-5 px-6 py-6">
        <header className="border-b border-border pb-4">
          <div className="mb-3 flex flex-wrap items-center gap-2 text-xs">
            {generating ? (
              <span className="rounded-full bg-primary/10 px-2 py-1 font-medium text-primary">
                Generating
              </span>
            ) : node.status !== "complete" ? (
              <span className="rounded-full bg-surface px-2 py-1 font-medium text-text-muted">
                {getExplainerNodeStatusLabel(node.status)}
              </span>
            ) : null}
            <span
              className={`rounded-full px-2 py-1 font-medium ${getExplainerEvidenceChipClass(node.evidenceState)}`}
              title={getExplainerEvidenceDescription(node.evidenceState)}
            >
              {getExplainerEvidenceLabel(node.evidenceState)}
            </span>
            {node.outsideKnowledgeUsed ? (
              <span
                className="rounded-full bg-warn/10 px-2 py-1 font-medium text-warn"
                title="Parts of this node come from the model's own knowledge rather than the selected sources."
              >
                Outside knowledge used
              </span>
            ) : null}
          </div>
          <h2 className="text-2xl font-semibold leading-tight text-text">{node.title}</h2>
          <p className="mt-2 text-sm text-text-muted">
            {`Intent: ${getExplainerIntentLabel(session.outputIntent)} · Grounding: ${getExplainerGroundingLabel(session.grounding)} · Depth: ${getExplainerDepthLabel(session.depthPreset)}`}
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
          (() => {
            const answered = Boolean(node.selectedOptionId || node.selectedCustomAnswer)
            if (!answered && onAnswerQuestion) {
              return (
                <section aria-label="Clarifying question" className="grid gap-3 rounded-md border border-primary/40 bg-primary/5 px-4 py-4">
                  <h3 className="text-sm font-semibold text-text">
                    Answer to shape the next breakdown
                  </h3>
                  <div className="flex flex-wrap gap-2">
                    {node.questionOptions.map((option) => {
                      const id = typeof option.id === "string" ? option.id : ""
                      const label = typeof option.label === "string" ? option.label : id
                      return (
                        <button
                          key={id || label}
                          type="button"
                          className="rounded-full border border-border bg-surface px-3 py-1 text-xs font-medium text-text transition-colors hover:border-primary hover:text-primary"
                          onClick={() => onAnswerQuestion(node.id, { selectedOptionId: id })}
                        >
                          {label}
                        </button>
                      )
                    })}
                  </div>
                  <div className="flex flex-wrap items-center gap-2">
                    <input
                      aria-label="Custom answer"
                      className="h-9 min-w-[220px] flex-1 rounded-md border border-border bg-surface2 px-3 text-sm text-text outline-none focus:border-primary focus:ring-2 focus:ring-focus"
                      placeholder="Or answer in your own words"
                      value={customAnswer}
                      onChange={(event) => setCustomAnswer(event.target.value)}
                      onKeyDown={(event) => {
                        if (event.key === "Enter" && customAnswer.trim()) {
                          onAnswerQuestion(node.id, { selectedCustomAnswer: customAnswer.trim() })
                        }
                      }}
                    />
                    <button
                      type="button"
                      className="inline-flex h-9 items-center rounded-md bg-primary px-3 text-sm font-semibold text-white transition-colors hover:bg-primaryStrong disabled:cursor-not-allowed disabled:bg-surface2 disabled:text-text-muted"
                      disabled={!customAnswer.trim()}
                      onClick={() =>
                        onAnswerQuestion(node.id, { selectedCustomAnswer: customAnswer.trim() })
                      }
                    >
                      Submit answer
                    </button>
                  </div>
                </section>
              )
            }
            return (
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
                  {node.selectedCustomAnswer ? (
                    <span className="rounded-full border border-primary bg-primary/10 px-3 py-1 text-xs font-medium text-primary">
                      {node.selectedCustomAnswer}
                    </span>
                  ) : null}
                </div>
              </section>
            )
          })()
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
                <a
                  href={citationHref(citation)}
                  className="mt-2 inline-block text-xs font-medium text-primary underline underline-offset-2 hover:opacity-90"
                  {...(citation.url ? { target: "_blank", rel: "noreferrer" } : {})}
                >
                  Open source
                </a>
              </figure>
            ))
          )}
        </section>

        <footer className="flex flex-wrap items-center gap-2 border-t border-border pt-4">
          <button
            type="button"
            className="inline-flex h-9 items-center gap-2 rounded-md bg-primary px-3 text-sm font-semibold text-white transition-colors hover:bg-primaryStrong disabled:cursor-not-allowed disabled:bg-surface2 disabled:text-text-muted"
            disabled={isExpanding || generating}
            onClick={() => onExpand(node.id)}
          >
            <Sparkles className="h-4 w-4" aria-hidden="true" />
            Break down
          </button>
          {onDeleteNode && node.parentId !== null ? (
            confirmingDelete ? (
              <>
                <button
                  type="button"
                  className="inline-flex h-9 items-center gap-2 rounded-md bg-danger px-3 text-sm font-semibold text-white transition-colors hover:opacity-90"
                  onClick={() => {
                    setConfirmingDelete(false)
                    onDeleteNode(node.id)
                  }}
                >
                  <Trash2 className="h-4 w-4" aria-hidden="true" />
                  Confirm delete
                </button>
                <button
                  type="button"
                  className="inline-flex h-9 items-center rounded-md border border-border bg-surface px-3 text-sm font-medium text-text transition-colors hover:bg-surface2"
                  onClick={() => setConfirmingDelete(false)}
                >
                  Cancel
                </button>
              </>
            ) : (
              <button
                type="button"
                className="inline-flex h-9 items-center gap-2 rounded-md border border-border bg-surface px-3 text-sm font-medium text-text-muted transition-colors hover:bg-surface2 hover:text-danger"
                onClick={() => setConfirmingDelete(true)}
              >
                <Trash2 className="h-4 w-4" aria-hidden="true" />
                Delete node
              </button>
            )
          ) : null}
        </footer>
      </article>
    </section>
  )
}

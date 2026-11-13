import React from "react"
import { useTranslation } from "react-i18next"

type GraphNodeKind = "note" | "tag" | "source"

const NODE_KINDS: Array<{
  kind: GraphNodeKind
  defaultLabel: string
  swatchClassName: string
}> = [
  { kind: "note", defaultLabel: "Note", swatchClassName: "rounded-full bg-primary" },
  { kind: "tag", defaultLabel: "Tag", swatchClassName: "rounded-[3px] bg-accent" },
  {
    kind: "source",
    defaultLabel: "Source",
    swatchClassName: "rotate-45 border border-border-strong bg-surface2"
  }
]

type NotesGraphLegendProps = {
  /** Node types in the loaded graph; the legend lists only these (NL-14). */
  nodeTypes: ReadonlySet<string>
}

/**
 * What the graph's shapes and lines mean: a circle is a note, a rounded
 * square a tag, a diamond a source; solid lines are saved links, dotted ones
 * similar content and dashed ones suggestions.
 */
const NotesGraphLegend: React.FC<NotesGraphLegendProps> = ({ nodeTypes }) => {
  const { t } = useTranslation("option")
  const kinds = NODE_KINDS.filter(({ kind }) => nodeTypes.has(kind))

  return (
    <div className="absolute bottom-2 left-2 flex max-w-[calc(100%-1rem)] flex-col gap-1 border border-border bg-elevated px-2 py-1 text-xs text-text shadow-sm">
      {kinds.length > 0 ? (
        <ul
          data-testid="notes-graph-node-legend"
          aria-label={t("notesSearch.graphNodeLegend", {
            defaultValue: "Node shapes"
          })}
          className="flex flex-wrap gap-x-3 gap-y-1">
          {kinds.map(({ kind, defaultLabel, swatchClassName }) => (
            <li key={kind} className="flex items-center gap-1.5">
              <span
                className={`inline-block h-2.5 w-2.5 ${swatchClassName}`}
                aria-hidden="true"
              />
              {t(`notesSearch.graphNodeType.${kind}`, { defaultValue: defaultLabel })}
            </li>
          ))}
        </ul>
      ) : null}
      <ul
        data-testid="notes-graph-edge-legend"
        aria-label={t("notesSearch.graphEdgeLegend", {
          defaultValue: "Relationship legend"
        })}
        className="flex flex-wrap gap-x-3 gap-y-1">
        <li className="flex items-center gap-1.5">
          <span className="w-5 border-t-2 border-text-muted" aria-hidden="true" />
          {t("notesSearch.graphLegendAuthoritative", {
            defaultValue: "Saved links"
          })}
        </li>
        <li className="flex items-center gap-1.5">
          <span
            data-testid="notes-graph-semantic-legend-swatch"
            className="w-5 border-t-2 border-dotted border-primary"
            aria-hidden="true"
          />
          {t("notesSearch.graphSimilarContent", {
            defaultValue: "Similar content"
          })}
        </li>
        <li className="flex items-center gap-1.5">
          <span
            className="w-5 border-t-2 border-dashed border-warn"
            aria-hidden="true"
          />
          {t("notesSearch.graphSuggestions", { defaultValue: "Suggestions" })}
        </li>
      </ul>
    </div>
  )
}

export default NotesGraphLegend

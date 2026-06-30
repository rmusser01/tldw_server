import React from "react"

/**
 * Horizontal faceted meta row used by V5 Radial Command. Each facet shows
 * a short key (e.g. "src", "mdl", "tmp") plus a value, where the whole
 * facet toggles between muted (default) and primary tint (active).
 *
 * Designer source: composer-redesign.html .v5-meta / .facet.
 *
 * Unlike V3's BriefField (which renders as a tall key/value card), Facet
 * is a single-line inline element — keys and values sit on the same row
 * in the meta strip above V5's pill composer.
 */

export interface FacetProps {
  /** Short 3–4 char key, e.g. "src", "mdl". */
  fieldKey?: string
  /** Value label — accepts any node for glyph + text combos. */
  value: React.ReactNode
  active?: boolean
  onClick?: () => void
  /** Accessible label for icon-only facets. */
  "aria-label"?: string
}

export const Facet: React.FC<FacetProps> = ({
  fieldKey,
  value,
  active = false,
  onClick,
  "aria-label": ariaLabel,
}) => {
  const wrapperCls = active ? "text-primary" : "text-text-subtle hover:text-text"
  const valueCls = active ? "text-primary" : "text-text-muted"
  const base =
    "inline-flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-wider transition-colors"

  const inner = (
    <>
      {fieldKey && <span>{fieldKey}</span>}
      <span className={valueCls}>{value}</span>
    </>
  )

  if (onClick) {
    return (
      <button
        type="button"
        className={`${base} ${wrapperCls} cursor-pointer`}
        onClick={onClick}
        aria-label={ariaLabel}
        aria-pressed={active}
      >
        {inner}
      </button>
    )
  }
  return (
    <span
      className={`${base} ${wrapperCls}`}
      aria-label={ariaLabel}
    >
      {inner}
    </span>
  )
}

export interface FacetSpec {
  id: string
  fieldKey?: string
  value: React.ReactNode
  active?: boolean
  onClick?: () => void
  "aria-label"?: string
}

export interface FacetRowProps {
  facets: FacetSpec[]
  /** Optional trailing content — e.g. token meter slot. */
  trailing?: React.ReactNode
  /** Accessible label for the whole row. Defaults to "Composer facets". */
  "aria-label"?: string
}

export const FacetRow: React.FC<FacetRowProps> = ({
  facets,
  trailing,
  "aria-label": ariaLabel = "Composer facets",
}) => (
  <div
    role="group"
    aria-label={ariaLabel}
    className="flex items-center gap-4 px-4 pt-2.5 flex-wrap"
  >
    {facets.map((facet) => (
      <Facet
        key={facet.id}
        fieldKey={facet.fieldKey}
        value={facet.value}
        active={facet.active}
        onClick={facet.onClick}
        aria-label={facet["aria-label"]}
      />
    ))}
    {trailing && <span className="ml-auto">{trailing}</span>}
  </div>
)

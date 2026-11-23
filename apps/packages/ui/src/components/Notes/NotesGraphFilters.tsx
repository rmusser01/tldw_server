import React from "react"
import { useTranslation } from "react-i18next"

const TAG_SETTLE_MS = 300

type NotesGraphFiltersProps = {
  tagOptions: string[]
  tag: string | null
  linkedOnly: boolean
  onTagChange: (tag: string | null) => void
  onLinkedOnlyChange: (linkedOnly: boolean) => void
}

/**
 * Graph filters (NL-11): a tag, which the server applies and which works for
 * a library of any size, and "Linked notes only", which hides the notes that
 * no shown link joins to another note.
 */
const NotesGraphFilters: React.FC<NotesGraphFiltersProps> = ({
  tagOptions,
  tag,
  linkedOnly,
  onTagChange,
  onLinkedOnlyChange
}) => {
  const { t } = useTranslation(["option"])
  const tagLabel = t("option:notesSearch.graphFilterTag", { defaultValue: "Tag" })
  // A native select fires change on every arrow key; wait for the reader to
  // settle before refetching the graph (NL-11 review).
  const [pendingTag, setPendingTag] = React.useState<string | null | undefined>(
    undefined
  )
  const onTagChangeRef = React.useRef(onTagChange)
  React.useEffect(() => {
    onTagChangeRef.current = onTagChange
  }, [onTagChange])
  React.useEffect(() => {
    if (pendingTag === undefined) return
    const timer = window.setTimeout(() => {
      setPendingTag(undefined)
      onTagChangeRef.current(pendingTag)
    }, TAG_SETTLE_MS)
    return () => window.clearTimeout(timer)
  }, [pendingTag])
  const options = React.useMemo(
    () => (tag && !tagOptions.includes(tag) ? [tag, ...tagOptions] : tagOptions),
    [tag, tagOptions]
  )

  return (
    <fieldset
      className="flex min-h-11 min-w-0 flex-wrap items-end gap-2 border border-border bg-surface px-2 py-1"
      data-testid="notes-graph-filters">
      <legend className="sr-only">
        {t("option:notesSearch.graphFilters", { defaultValue: "Graph filters" })}
      </legend>
      <label className="flex flex-col gap-0.5 text-xs text-text-muted">
        {tagLabel}
        <select
          aria-label={tagLabel}
          className="h-8 max-w-[180px] border border-border bg-bg px-2 text-sm text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-focus"
          value={(pendingTag === undefined ? tag : pendingTag) ?? ""}
          onChange={(event) => setPendingTag(event.target.value || null)}>
          <option value="">
            {t("option:notesSearch.graphFilterAnyTag", {
              defaultValue: "All tags"
            })}
          </option>
          {options.map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
      </label>
      <label className="flex h-9 items-center gap-2 whitespace-nowrap text-sm text-text">
        <input
          type="checkbox"
          checked={linkedOnly}
          onChange={() => onLinkedOnlyChange(!linkedOnly)}
        />
        {t("option:notesSearch.graphFilterLinkedOnly", {
          defaultValue: "Linked notes only"
        })}
      </label>
    </fieldset>
  )
}

export default NotesGraphFilters

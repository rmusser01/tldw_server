import React, { useId, useState } from "react"
import { Input } from "antd"
import { ChevronDown } from "lucide-react"
import { useTranslation } from "react-i18next"

import { useIngestWizard } from "./IngestWizardContext"
import type { ConferenceItemMetadataOverride, WizardQueueItem } from "./types"

const fields = ["title", "speaker", "track", "tags"] as const
const columns = "grid grid-cols-[44px_minmax(0,1.4fr)_minmax(0,1fr)_minmax(0,0.8fr)_minmax(0,1fr)] gap-2"

type SourceRowProps = {
  item: WizardQueueItem
  number: number
  update: (patch: Partial<ConferenceItemMetadataOverride>) => void
}

function SourceRow({ item, number, update }: SourceRowProps) {
  const { t } = useTranslation("option")
  const panelId = useId()
  const [expanded, setExpanded] = useState(number === 1)
  const [tagDraft, setTagDraft] = useState<string | null>(null)
  const override = item.conferenceOverride
  const sourceLabel = t("quickIngest.metadata.source", { number })
  const original = item.url || item.fileName || item.id
  const title = item.playlist?.sourceTitle || item.fileName || original

  return (
    <div
      role="group"
      aria-label={sourceLabel}
      className={`qi-metadata-row ${columns} border-b border-border px-2 py-2 last:border-b-0`}
    >
      <button
        type="button"
        aria-expanded={expanded}
        aria-controls={panelId}
        className="qi-metadata-toggle hidden min-w-0 items-center gap-2 text-start"
        onClick={() => setExpanded(!expanded)}
      >
        <ChevronDown
          aria-hidden="true" size={16}
          className={`shrink-0 text-text-muted ${expanded ? "" : "-rotate-90"}`}
        />
        <span className="min-w-0">
          <span className="block text-xs text-text-muted">{sourceLabel}</span>
          <span className="block truncate text-sm font-medium text-text">{override?.title || title}</span>
        </span>
      </button>
      <label className="qi-metadata-include flex cursor-pointer items-center gap-2 text-sm">
        <input
          type="checkbox"
          checked={override?.selected ?? true}
          aria-label={t("quickIngest.metadata.includeItem", { number })}
          onChange={(event) => update({ selected: event.target.checked })}
        />
        <span className="qi-metadata-label hidden">{t("quickIngest.metadata.include")}</span>
      </label>
      <div
        id={panelId}
        data-expanded={expanded}
        className="qi-metadata-fields contents"
        onFocusCapture={() => setExpanded(true)}
      >
        {fields.map((field) => (
          <label key={field} className="block min-w-0">
            <span className="qi-metadata-label mb-1 hidden text-xs font-medium text-text">
              {t(`quickIngest.metadata.${field}`)}
            </span>
            {field === "tags" ? (
              <Input
                size="small"
                aria-label={t("quickIngest.metadata.tagsItem", { number })}
                placeholder={t("quickIngest.metadata.tagExample")}
                value={tagDraft ?? override?.tags?.join(", ") ?? ""}
                onChange={(event) => setTagDraft(event.target.value)}
                onBlur={(event) => {
                  if (tagDraft === null) return
                  update({ tags: event.target.value.split(",").map((tag) => tag.trim()).filter(Boolean) })
                  setTagDraft(null)
                }}
                onKeyDown={(event) => {
                  if (event.key === "Enter") event.currentTarget.blur()
                }}
              />
            ) : (
              <Input
                size="small"
                aria-label={t(`quickIngest.metadata.${field}Item`, { number })}
                value={override?.[field] ?? ""}
                placeholder={field === "title" ? title : undefined}
                onChange={(event) => update({ [field]: event.target.value })}
              />
            )}
            {field === "title" && (
              <span className="qi-metadata-origin mt-1 block truncate text-[11px] text-text-muted">
                {original}
              </span>
            )}
          </label>
        ))}
      </div>
    </div>
  )
}

export const ItemMetadataTable: React.FC = () => {
  const { t } = useTranslation("option")
  const { state: { queueItems }, setQueueItems } = useIngestWizard()
  if (queueItems.length === 0) return null

  return (
    <div className="qi-metadata-editor mt-3 min-w-0 rounded-md border border-border">
      <div aria-hidden="true" className={`qi-metadata-header ${columns} border-b border-border bg-surface2 px-2 py-1.5 text-[11px] font-medium uppercase text-text-muted`}>
        <span>{t("quickIngest.metadata.use")}</span>
        {fields.map((field) => <span key={field}>{t(`quickIngest.metadata.${field}`)}</span>)}
      </div>
      <div className="qi-metadata-list max-h-72 overflow-y-auto">
        {queueItems.map((item, index) => (
          <SourceRow
            key={item.id}
            item={item}
            number={index + 1}
            update={(patch) => setQueueItems(queueItems.map((entry) =>
              entry.id === item.id
                ? { ...entry, conferenceOverride: { selected: true, ...entry.conferenceOverride, ...patch } }
                : entry
            ))}
          />
        ))}
      </div>
    </div>
  )
}

export default ItemMetadataTable

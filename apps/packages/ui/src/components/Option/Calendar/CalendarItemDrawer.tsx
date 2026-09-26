import React from "react"
import { Button, Divider, Drawer, Input, Popconfirm, Select, Space, Tooltip, Typography, message } from "antd"
import { Trash2 } from "lucide-react"
import { safeExternalUrl } from "@/utils/safe-external-url"
import type {
  CalendarItemKind,
  CalendarItemUpdateRequest,
  CalendarLinkResponse,
  CalendarResponse,
  CalendarSourceOwner,
  CalendarViewItemResponse
} from "@/services/calendar"
import {
  copyCalendarItemIntoTldw,
  createCalendarAnnotation,
  createCalendarItem,
  createCalendarLink,
  listCalendarLinks,
  deleteCalendarLink,
  deleteCalendarItem,
  updateCalendarItem,
  updateCalendarLocalTags
} from "@/services/calendar"
import { CalendarOwnershipBadge } from "./CalendarOwnershipBadge"

const toInputDateTime = (value?: string | null): string =>
  value ? value.slice(0, 16) : ""

const toUpdatedDateTime = (
  value: string, original: string | null | undefined, allDay: boolean, itemTimezone: unknown
): string | null => {
  if (!value) return null
  if (allDay || !value.includes("T") || /(?:Z|[+-]\d{2}:\d{2})$/i.test(value)) return value
  if (typeof itemTimezone === "string" && itemTimezone) return value
  // Inputs show the timestamp's wall time, not a browser-local conversion.
  const offset = original?.includes("T") ? original.match(/(?:Z|[+-]\d{2}:\d{2})$/i)?.[0] : null
  return offset ? `${value}${offset}` : value
}

const parseTags = (value: string): string[] =>
  value
    .split(",")
    .map((tag) => tag.trim())
    .filter(Boolean)

const inferKind = (item: CalendarViewItemResponse | null): CalendarItemKind => {
  if (!item) return "event"
  if (item.kind === "todo") return "todo"
  return "event"
}

const itemId = (item: CalendarViewItemResponse): string | number =>
  item.calendar_item_id ?? item.id

const itemLocalTags = (item: CalendarViewItemResponse | null): string[] | null => {
  if (!item) return []
  if (Array.isArray(item.local_tags)) return item.local_tags
  const metadataTags = item.metadata?.local_tags ?? item.metadata?.tags
  return Array.isArray(metadataTags) ? metadataTags.map(String) : null
}

const sameTags = (left: string[] | null, right: string[]): boolean =>
  Array.isArray(left) &&
  left.length === right.length &&
  left.every((tag, index) => tag === right[index])

export interface CalendarItemDrawerProps {
  open: boolean
  item: CalendarViewItemResponse | null
  calendars: CalendarResponse[]
  onClose: () => void
  onSaved: () => void | Promise<void>
}

export const CalendarItemDrawer: React.FC<CalendarItemDrawerProps> = ({
  open,
  item,
  calendars,
  onClose,
  onSaved
}) => {
  const [kind, setKind] = React.useState<CalendarItemKind>("event")
  const [calendarId, setCalendarId] = React.useState<number | null>(null)
  const [title, setTitle] = React.useState("")
  const [description, setDescription] = React.useState("")
  const [location, setLocation] = React.useState("")
  const [startAt, setStartAt] = React.useState("")
  const [endAt, setEndAt] = React.useState("")
  const [dueAt, setDueAt] = React.useState("")
  const [tags, setTags] = React.useState("")
  const [annotation, setAnnotation] = React.useState("")
  const [linkLabel, setLinkLabel] = React.useState("")
  const [linkUrl, setLinkUrl] = React.useState("")
  const [links, setLinks] = React.useState<CalendarLinkResponse[]>([])
  const [linksLoading, setLinksLoading] = React.useState(false)
  const [linksError, setLinksError] = React.useState(false)
  const [initialTags, setInitialTags] = React.useState<string[] | null>([])
  const [tagsDirty, setTagsDirty] = React.useState(false)
  const [saving, setSaving] = React.useState(false)

  const selectedCalendar =
    calendars.find((calendar) => calendar.id === (item?.calendar_id ?? calendarId)) ?? null
  const isCreate = !item
  const isProviderOwned = item?.source_owner === "provider"
  const isLinkedProjection = item?.source_owner === "linked_projection"
  const canEditItemFields = Boolean(isCreate || (!isProviderOwned && !isLinkedProjection && !item?.read_only_reason))
  const canEditLocalContext = Boolean(item?.calendar_item_id && !isLinkedProjection)
  const isOccurrence = item?.recurrence_id != null || item?.occurrence_index != null
  const canEditTemporalFields = canEditItemFields && !isOccurrence

  React.useEffect(() => {
    if (!open) return
    const nextKind = inferKind(item)
    setKind(nextKind)
    setCalendarId(item?.calendar_id ?? calendars[0]?.id ?? null)
    setTitle(item?.title ?? "")
    setDescription(item?.description ?? "")
    setLocation(item?.location ?? "")
    setStartAt(toInputDateTime(item?.start_at))
    setEndAt(toInputDateTime(item?.end_at))
    setDueAt(toInputDateTime(item?.due_at))
    const nextTags = itemLocalTags(item)
    setInitialTags(nextTags)
    setTags(Array.isArray(nextTags) ? nextTags.join(", ") : "")
    setTagsDirty(false)
    setAnnotation("")
    setLinkLabel("")
    setLinkUrl("")
  }, [calendars, item, open])

  React.useEffect(() => {
    let active = true
    setLinks(item?.links ?? [])
    setLinksError(false)
    if (!open || !item?.calendar_item_id || isLinkedProjection) {
      setLinksLoading(false)
      return () => { active = false }
    }
    setLinksLoading(true)
    listCalendarLinks(item.calendar_item_id)
      .then((result) => { if (active) setLinks(result.items) })
      .catch(() => { if (active) setLinksError(true) })
      .finally(() => { if (active) setLinksLoading(false) })
    return () => { active = false }
  }, [open, item, isLinkedProjection])

  const handleRemoveLink = async (linkId: number) => {
    if (!item?.calendar_item_id) return
    setSaving(true)
    try {
      await deleteCalendarLink(item.calendar_item_id, linkId)
      setLinks((current) => current.filter((link) => link.id !== linkId))
      await onSaved()
    } catch (error) {
      message.error(error instanceof Error ? error.message : "Unable to remove link")
    } finally {
      setSaving(false)
    }
  }

  const refreshAndClose = async () => {
    await onSaved()
    onClose()
  }

  const handleSave = async () => {
    if (!calendarId || (!title.trim() && canEditItemFields)) return
    setSaving(true)
    try {
      const currentTags = parseTags(tags)
      const shouldSaveTags = tagsDirty || (initialTags !== null && !sameTags(initialTags, currentTags))

      if (item && !canEditItemFields) {
        await saveLocalContext(currentTags, shouldSaveTags)
        await refreshAndClose()
        return
      }

      const payload = {
        calendar_id: calendarId,
        kind,
        title: title.trim(),
        description: description.trim() || null,
        location: location.trim() || null,
        start_at: kind === "event" ? startAt || null : null,
        end_at: kind === "event" ? endAt || null : null,
        due_at: kind === "todo" ? dueAt || null : null,
        status: kind === "todo" ? "needs_action" : "confirmed"
      }

      if (item) {
        const updates: CalendarItemUpdateRequest = {
          title: payload.title,
          description: payload.description,
          location: payload.location,
          source_owner: item.source_owner as CalendarSourceOwner,
          provider_owned: false
        }
        if (canEditTemporalFields) {
          const kindChanged = kind !== inferKind(item)
          if (kindChanged) {
            updates.kind = kind
            updates.status = payload.status
          }
          // Occurrence timestamps are not master timestamps; unchanged fields
          // must also stay omitted to preserve offsets and sub-minute precision.
          for (const [field, input] of [
            ["start_at", startAt],
            ["end_at", endAt],
            ["due_at", dueAt]
          ] as const) {
            if (kindChanged || input !== toInputDateTime(item[field])) {
              const active = field === "due_at" ? kind === "todo" : kind === "event"
              updates[field] = active ? toUpdatedDateTime(input, item[field], item.all_day, item.metadata?.timezone) : null
            }
          }
        }
        if (shouldSaveTags) {
          updates.local_tags = currentTags
        }
        await updateCalendarItem(itemId(item), updates)
      } else {
        await createCalendarItem({
          ...payload,
          timezone: selectedCalendar?.timezone ?? null,
          local_tags: currentTags
        })
      }

      await saveLocalContext(currentTags, false)

      await refreshAndClose()
    } catch (error: any) {
      message.error(error?.message || "Unable to save calendar item")
    } finally {
      setSaving(false)
    }
  }

  const saveLocalContext = async (currentTags: string[], shouldSaveTags: boolean) => {
    if (!item?.calendar_item_id) return

    if (shouldSaveTags) {
      await updateCalendarLocalTags(item.calendar_item_id, { tags: currentTags })
    }

    if (annotation.trim()) {
      await createCalendarAnnotation(item.calendar_item_id, {
        body: annotation.trim(),
        tags: []
      })
    }

    if (linkLabel.trim() || linkUrl.trim()) {
      await createCalendarLink(item.calendar_item_id, {
        target_type: "note",
        target_id: linkUrl.trim() || linkLabel.trim(),
        label: linkLabel.trim() || null,
        url: linkUrl.trim() || null,
        metadata: {}
      })
    }
  }

  const handleDelete = async () => {
    if (!item || isOccurrence) return
    setSaving(true)
    try {
      await deleteCalendarItem({
        calendar_item_id: item.calendar_item_id,
        id: item.id,
        source_owner: item.source_owner,
        read_only_reason: item.read_only_reason
      })
      await refreshAndClose()
    } catch (error: any) {
      message.error(error?.message || "Unable to delete calendar item")
    } finally {
      setSaving(false)
    }
  }

  const handleCopy = async () => {
    if (!item) return
    setSaving(true)
    try {
      await copyCalendarItemIntoTldw(itemId(item), {
        target_calendar_id: calendarId ?? calendars[0]?.id ?? null
      })
      await refreshAndClose()
    } catch (error: any) {
      message.error(error?.message || "Unable to copy calendar item")
    } finally {
      setSaving(false)
    }
  }

  return (
    <Drawer
      title={isCreate ? "New calendar item" : item?.title ?? "Calendar item"}
      open={open}
      onClose={onClose}
      size="default"
      getContainer={false}
      destroyOnHidden
      extra={
        item ? (
          <CalendarOwnershipBadge item={item} calendar={selectedCalendar} />
        ) : null
      }
    >
      {isLinkedProjection && item ? (
        <div className="flex flex-col gap-4">
          <div>
            <Typography.Title level={4} style={{ marginTop: 0 }}>
              {item.title}
            </Typography.Title>
            {item.read_only_reason ? (
              <Typography.Paragraph type="secondary">
                {item.read_only_reason}
              </Typography.Paragraph>
            ) : null}
          </div>
          {item.link?.url ? (
            <Button href={item.link.url}>
              {item.link.label || "Open source"}
            </Button>
          ) : null}
        </div>
      ) : (
        <div className="flex flex-col gap-4">
          {item?.read_only_reason ? (
            <Typography.Text type="secondary">{item.read_only_reason}</Typography.Text>
          ) : null}

          <label className="flex flex-col gap-1">
            <Typography.Text>Calendar</Typography.Text>
            <Select
              aria-label="Calendar"
              value={calendarId ?? undefined}
              disabled={!isCreate || !canEditItemFields}
              onChange={(value) => setCalendarId(Number(value))}
              options={calendars.map((calendar) => ({
                value: calendar.id,
                label: calendar.name
              }))}
            />
          </label>

          <fieldset className="flex gap-4 border-0 p-0">
            <legend className="sr-only">Kind</legend>
            <label className="inline-flex items-center gap-2">
              <input
                type="radio"
                name="calendar-kind"
                checked={kind === "event"}
                disabled={!canEditTemporalFields}
                onChange={() => setKind("event")}
              />
              Event
            </label>
            <label className="inline-flex items-center gap-2">
              <input
                type="radio"
                name="calendar-kind"
                checked={kind === "todo"}
                disabled={!canEditTemporalFields}
                onChange={() => setKind("todo")}
              />
              Todo
            </label>
          </fieldset>

          <label className="flex flex-col gap-1">
            <Typography.Text>Title</Typography.Text>
            <Input
              aria-label="Title"
              value={title}
              disabled={!canEditItemFields}
              onChange={(event) => setTitle(event.target.value)}
            />
          </label>

          <label className="flex flex-col gap-1">
            <Typography.Text>Description</Typography.Text>
            <Input.TextArea
              aria-label="Description"
              value={description}
              disabled={!canEditItemFields}
              autoSize={{ minRows: 2, maxRows: 4 }}
              onChange={(event) => setDescription(event.target.value)}
            />
          </label>

          <label className="flex flex-col gap-1">
            <Typography.Text>Location</Typography.Text>
            <Input
              aria-label="Location"
              value={location}
              disabled={!canEditItemFields}
              onChange={(event) => setLocation(event.target.value)}
            />
          </label>

          {kind === "event" ? (
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              <label className="flex flex-col gap-1">
                <Typography.Text>Start</Typography.Text>
                <Input
                  aria-label="Start"
                  value={startAt}
                  disabled={!canEditTemporalFields}
                  placeholder="2026-06-05T09:00"
                  onChange={(event) => setStartAt(event.target.value)}
                />
              </label>
              <label className="flex flex-col gap-1">
                <Typography.Text>End</Typography.Text>
                <Input
                  aria-label="End"
                  value={endAt}
                  disabled={!canEditTemporalFields}
                  placeholder="2026-06-05T10:00"
                  onChange={(event) => setEndAt(event.target.value)}
                />
              </label>
            </div>
          ) : (
            <label className="flex flex-col gap-1">
              <Typography.Text>Due</Typography.Text>
              <Input
                aria-label="Due"
                value={dueAt}
                disabled={!canEditTemporalFields}
                placeholder="2026-06-05T17:00"
                onChange={(event) => setDueAt(event.target.value)}
              />
            </label>
          )}

          <label className="flex flex-col gap-1">
            <Typography.Text>Tags</Typography.Text>
            <Input
              aria-label="Tags"
              value={tags}
              disabled={!isCreate && !canEditLocalContext}
              onChange={(event) => {
                setTags(event.target.value)
                setTagsDirty(true)
              }}
            />
          </label>

          {canEditLocalContext ? (
            <>
              <Divider className="my-1" />
              <label className="flex flex-col gap-1">
                <Typography.Text>Annotation</Typography.Text>
                <Input.TextArea
                  aria-label="Annotation"
                  value={annotation}
                  autoSize={{ minRows: 2, maxRows: 4 }}
                  onChange={(event) => setAnnotation(event.target.value)}
                />
              </label>
              {linksLoading ? <Typography.Text type="secondary" role="status">Loading links</Typography.Text> : null}
              {linksError ? <Typography.Text type="danger" role="alert">Links unavailable</Typography.Text> : null}
              {links.map((link) => {
                const label = link.label || link.target_id
                const href = safeExternalUrl(link.url)
                return (
                  <div key={link.id} className="flex min-w-0 items-center justify-between gap-2">
                    {href ? <Typography.Link href={href} className="min-w-0 break-words">{label}</Typography.Link>
                      : <Typography.Text className="min-w-0 break-words">{label}</Typography.Text>}
                    <Popconfirm title="Remove this link?" okText="Remove link" cancelText="Cancel"
                      onConfirm={() => handleRemoveLink(link.id)}>
                      <Tooltip title={`Remove ${label}`}>
                        <Button type="text" aria-label={`Remove ${label}`} disabled={saving}
                          icon={<Trash2 size={16} />} />
                      </Tooltip>
                    </Popconfirm>
                  </div>
                )
              })}
              {!links.length && item.link?.url && safeExternalUrl(item.link.url) ? (
                <Button href={safeExternalUrl(item.link.url) ?? undefined}>{item.link.label || "Open link"}</Button>
              ) : null}
              <div className="flex flex-col gap-2">
                <label className="flex flex-col gap-1">
                  <Typography.Text>Link label</Typography.Text>
                  <Input
                    aria-label="Link label"
                    value={linkLabel}
                    onChange={(event) => setLinkLabel(event.target.value)}
                  />
                </label>
                <label className="flex flex-col gap-1">
                  <Typography.Text>Link URL</Typography.Text>
                  <Input
                    aria-label="Link URL"
                    value={linkUrl}
                    onChange={(event) => setLinkUrl(event.target.value)}
                  />
                </label>
              </div>
            </>
          ) : null}

          <Space wrap>
            {isProviderOwned ? (
              <Button loading={saving} onClick={handleCopy}>
                Copy into tldw
              </Button>
            ) : null}
            {canEditItemFields ? (
              <Button type="primary" loading={saving} onClick={handleSave}>
                Save item
              </Button>
            ) : null}
            {!canEditItemFields && canEditLocalContext ? (
              <Button type="primary" loading={saving} onClick={handleSave}>
                Save context
              </Button>
            ) : null}
            {canEditItemFields && item && !isOccurrence ? (
              <Popconfirm
                title="Delete this calendar item?"
                okText="Confirm delete"
                cancelText="Cancel"
                okButtonProps={{ danger: true }}
                onConfirm={() => void handleDelete()}
              >
                <Button danger loading={saving}>Delete item</Button>
              </Popconfirm>
            ) : null}
          </Space>
        </div>
      )}
    </Drawer>
  )
}

export default CalendarItemDrawer

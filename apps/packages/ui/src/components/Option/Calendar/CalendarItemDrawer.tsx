import React from "react"
import { Button, Divider, Drawer, Input, Select, Space, Typography, message } from "antd"
import type {
  CalendarItemKind,
  CalendarResponse,
  CalendarViewItemResponse
} from "@/services/calendar"
import {
  copyCalendarItemIntoTldw,
  createCalendarAnnotation,
  createCalendarItem,
  createCalendarLink,
  deleteCalendarItem,
  updateCalendarItem
} from "@/services/calendar"
import { CalendarOwnershipBadge } from "./CalendarOwnershipBadge"

const toInputDateTime = (value?: string | null): string =>
  value ? value.slice(0, 16) : ""

const parseTags = (value: string): string[] =>
  value
    .split(",")
    .map((tag) => tag.trim())
    .filter(Boolean)

const inferKind = (item: CalendarViewItemResponse | null): CalendarItemKind => {
  if (!item) return "event"
  if (item.metadata?.kind === "todo" || (item.due_at && !item.start_at)) return "todo"
  return "event"
}

const itemId = (item: CalendarViewItemResponse): string | number =>
  item.calendar_item_id ?? item.id

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
  const [saving, setSaving] = React.useState(false)

  const selectedCalendar =
    calendars.find((calendar) => calendar.id === (item?.calendar_id ?? calendarId)) ?? null
  const isCreate = !item
  const isProviderOwned = item?.source_owner === "provider"
  const isLinkedProjection = item?.source_owner === "linked_projection"
  const isReadOnly = Boolean(isProviderOwned || isLinkedProjection || item?.read_only_reason)

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
    const metadataTags = item?.metadata?.local_tags ?? item?.metadata?.tags
    setTags(Array.isArray(metadataTags) ? metadataTags.join(", ") : "")
    setAnnotation("")
    setLinkLabel("")
    setLinkUrl("")
  }, [calendars, item, open])

  const refreshAndClose = async () => {
    await onSaved()
    onClose()
  }

  const handleSave = async () => {
    if (!calendarId || !title.trim()) return
    setSaving(true)
    try {
      const payload = {
        calendar_id: calendarId,
        kind,
        title: title.trim(),
        description: description.trim() || null,
        location: location.trim() || null,
        start_at: kind === "event" ? startAt || null : null,
        end_at: kind === "event" ? endAt || null : null,
        due_at: kind === "todo" ? dueAt || null : null,
        status: kind === "todo" ? "needs_action" : "confirmed",
        local_tags: parseTags(tags)
      }

      if (item) {
        await updateCalendarItem(itemId(item), {
          kind: payload.kind,
          title: payload.title,
          description: payload.description,
          location: payload.location,
          start_at: payload.start_at,
          end_at: payload.end_at,
          due_at: payload.due_at,
          status: payload.status,
          local_tags: payload.local_tags,
          source_owner: item.source_owner,
          provider_owned: false
        })
      } else {
        await createCalendarItem(payload)
      }

      if (item?.calendar_item_id && annotation.trim()) {
        await createCalendarAnnotation(item.calendar_item_id, {
          body: annotation.trim(),
          tags: []
        })
      }

      if (item?.calendar_item_id && (linkLabel.trim() || linkUrl.trim())) {
        await createCalendarLink(item.calendar_item_id, {
          target_type: "note",
          target_id: linkUrl.trim() || linkLabel.trim(),
          label: linkLabel.trim() || null,
          url: linkUrl.trim() || null,
          metadata: {}
        })
      }

      await refreshAndClose()
    } catch (error: any) {
      message.error(error?.message || "Unable to save calendar item")
    } finally {
      setSaving(false)
    }
  }

  const handleDelete = async () => {
    if (!item) return
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
              disabled={isReadOnly}
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
                disabled={isReadOnly}
                onChange={() => setKind("event")}
              />
              Event
            </label>
            <label className="inline-flex items-center gap-2">
              <input
                type="radio"
                name="calendar-kind"
                checked={kind === "todo"}
                disabled={isReadOnly}
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
              disabled={isReadOnly}
              onChange={(event) => setTitle(event.target.value)}
            />
          </label>

          <label className="flex flex-col gap-1">
            <Typography.Text>Description</Typography.Text>
            <Input.TextArea
              aria-label="Description"
              value={description}
              disabled={isReadOnly}
              autoSize={{ minRows: 2, maxRows: 4 }}
              onChange={(event) => setDescription(event.target.value)}
            />
          </label>

          <label className="flex flex-col gap-1">
            <Typography.Text>Location</Typography.Text>
            <Input
              aria-label="Location"
              value={location}
              disabled={isReadOnly}
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
                  disabled={isReadOnly}
                  placeholder="2026-06-05T09:00"
                  onChange={(event) => setStartAt(event.target.value)}
                />
              </label>
              <label className="flex flex-col gap-1">
                <Typography.Text>End</Typography.Text>
                <Input
                  aria-label="End"
                  value={endAt}
                  disabled={isReadOnly}
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
                disabled={isReadOnly}
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
              disabled={isReadOnly}
              onChange={(event) => setTags(event.target.value)}
            />
          </label>

          {!isReadOnly && item?.calendar_item_id ? (
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
              {item.link?.url ? (
                <Button href={item.link.url}>{item.link.label || "Open link"}</Button>
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
            {!isReadOnly ? (
              <Button type="primary" loading={saving} onClick={handleSave}>
                Save item
              </Button>
            ) : null}
            {!isReadOnly && item ? (
              <Button danger loading={saving} onClick={handleDelete}>
                Delete item
              </Button>
            ) : null}
          </Space>
        </div>
      )}
    </Drawer>
  )
}

export default CalendarItemDrawer

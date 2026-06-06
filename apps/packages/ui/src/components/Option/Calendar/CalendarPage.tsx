import React, { useMemo, useState } from "react"
import { Button, Segmented, Spin, Typography } from "antd"
import { useQuery } from "@tanstack/react-query"
import { useNavigate } from "react-router-dom"
import { RecoveryCallout, buildCapabilityState } from "@/components/ui/state"
import { useCanonicalConnectionConfig } from "@/hooks/useCanonicalConnectionConfig"
import {
  getCalendarAgenda,
  getCalendarWeek,
  listCalendars,
  type CalendarResponse,
  type CalendarViewItemResponse
} from "@/services/calendar"
import { CalendarAgenda } from "./CalendarAgenda"
import {
  CalendarFilterRail,
  type CalendarKindFilter,
  type CalendarSourceFilter
} from "./CalendarFilterRail"
import { CalendarItemDrawer } from "./CalendarItemDrawer"
import { CalendarWeekView } from "./CalendarWeekView"

const CALENDAR_SUPPORT_PATH = "/api/v1/calendar/calendars"
const ALL_SOURCES: CalendarSourceFilter[] = ["local", "org", "provider", "linked"]
const ALL_KINDS: CalendarKindFilter[] = ["event", "todo"]

const startOfDay = (date: Date): Date =>
  new Date(date.getFullYear(), date.getMonth(), date.getDate())

const addDays = (date: Date, days: number): Date => {
  const next = new Date(date)
  next.setDate(next.getDate() + days)
  return next
}

const startOfWeek = (date: Date): Date => {
  const day = startOfDay(date)
  const offset = day.getDay() === 0 ? -6 : 1 - day.getDay()
  return addDays(day, offset)
}

const inferItemKind = (item: CalendarViewItemResponse): CalendarKindFilter =>
  item.kind === "todo" ? "todo" : "event"

const sourceFilterForItem = (
  item: CalendarViewItemResponse,
  calendarsById: Map<number, CalendarResponse>
): CalendarSourceFilter => {
  if (item.source_owner === "linked_projection") return "linked"
  if (item.source_owner === "provider") return "provider"
  const calendar = item.calendar_id ? calendarsById.get(item.calendar_id) : null
  return calendar?.org_id ? "org" : "local"
}

export const CalendarPage: React.FC = () => {
  const navigate = useNavigate()
  const { config: connectionConfig, loading: connectionConfigLoading } =
    useCanonicalConnectionConfig()
  const [calendarSupported, setCalendarSupported] = useState<boolean | null>(null)
  const [selectedCalendarIds, setSelectedCalendarIds] = useState<number[] | null>(null)
  const [selectedSources, setSelectedSources] =
    useState<CalendarSourceFilter[]>(ALL_SOURCES)
  const [selectedKinds, setSelectedKinds] = useState<CalendarKindFilter[]>(ALL_KINDS)
  const [view, setView] = useState<"agenda" | "week">("agenda")
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [selectedItem, setSelectedItem] = useState<CalendarViewItemResponse | null>(null)

  const now = useMemo(() => new Date(), [])
  const agendaStart = useMemo(() => startOfDay(now), [now])
  const agendaEnd = useMemo(() => addDays(agendaStart, 14), [agendaStart])
  const weekStart = useMemo(() => startOfWeek(now), [now])

  React.useEffect(() => {
    if (connectionConfigLoading) return

    const serverUrl = connectionConfig?.serverUrl?.trim()
    if (!serverUrl) {
      setCalendarSupported(true)
      return
    }

    let cancelled = false

    const probeCalendarSupport = async () => {
      try {
        const response = await fetch(`${serverUrl}/openapi.json`)
        if (!response.ok) {
          if (!cancelled) setCalendarSupported(true)
          return
        }

        const spec = await response.json()
        const paths =
          spec && typeof spec === "object" && spec.paths && typeof spec.paths === "object"
            ? (spec.paths as Record<string, unknown>)
            : null

        if (!cancelled) {
          setCalendarSupported(Boolean(paths && CALENDAR_SUPPORT_PATH in paths))
        }
      } catch {
        if (!cancelled) setCalendarSupported(true)
      }
    }

    void probeCalendarSupport()

    return () => {
      cancelled = true
    }
  }, [connectionConfig?.serverUrl, connectionConfigLoading])

  const calendarsQuery = useQuery({
    queryKey: ["calendar", "calendars"],
    queryFn: listCalendars,
    enabled: calendarSupported === true
  })

  const calendars = calendarsQuery.data?.items ?? []

  React.useEffect(() => {
    if (selectedCalendarIds !== null || calendars.length === 0) return
    setSelectedCalendarIds(calendars.map((calendar) => calendar.id))
  }, [calendars, selectedCalendarIds])

  const selectedCalendarQueryIds =
    selectedCalendarIds ?? calendars.map((calendar) => calendar.id)
  const hasExplicitEmptyCalendarSelection =
    selectedCalendarIds !== null && selectedCalendarIds.length === 0

  const agendaQuery = useQuery({
    queryKey: [
      "calendar",
      "agenda",
      agendaStart.toISOString(),
      agendaEnd.toISOString(),
      selectedCalendarQueryIds
    ],
    queryFn: () =>
      getCalendarAgenda({
        start_at: agendaStart.toISOString(),
        end_at: agendaEnd.toISOString(),
        calendar_ids: selectedCalendarQueryIds,
        include_scheduled_tasks: true
      }),
    enabled:
      calendarSupported === true &&
      calendarsQuery.isSuccess &&
      !hasExplicitEmptyCalendarSelection
  })

  const weekQuery = useQuery({
    queryKey: ["calendar", "week", weekStart.toISOString(), selectedCalendarQueryIds],
    queryFn: () =>
      getCalendarWeek({
        week_start: weekStart.toISOString(),
        calendar_ids: selectedCalendarQueryIds,
        include_scheduled_tasks: true
      }),
    enabled:
      calendarSupported === true &&
      calendarsQuery.isSuccess &&
      !hasExplicitEmptyCalendarSelection
  })

  const calendarsById = useMemo(
    () => new Map(calendars.map((calendar) => [calendar.id, calendar])),
    [calendars]
  )

  const filterItems = React.useCallback(
    (items: CalendarViewItemResponse[]) => {
      if (hasExplicitEmptyCalendarSelection) return []
      return items.filter((item) => {
        const source = sourceFilterForItem(item, calendarsById)
        const kind = inferItemKind(item)
        if (!selectedSources.includes(source)) return false
        if (!selectedKinds.includes(kind)) return false
        if (item.calendar_id && !selectedCalendarQueryIds.includes(item.calendar_id)) {
          return false
        }
        return true
      })
    },
    [
      calendarsById,
      hasExplicitEmptyCalendarSelection,
      selectedCalendarQueryIds,
      selectedKinds,
      selectedSources
    ]
  )

  const agendaItems = useMemo(
    () => filterItems(agendaQuery.data?.items ?? []),
    [agendaQuery.data?.items, filterItems]
  )
  const weekItems = useMemo(
    () => filterItems(weekQuery.data?.items ?? []),
    [filterItems, weekQuery.data?.items]
  )

  const openCreateDrawer = () => {
    setSelectedItem(null)
    setDrawerOpen(true)
  }

  const openItemDrawer = (item: CalendarViewItemResponse) => {
    setSelectedItem(item)
    setDrawerOpen(true)
  }

  const closeDrawer = () => {
    setDrawerOpen(false)
    setSelectedItem(null)
  }

  const refreshViews = async () => {
    await Promise.all([
      calendarsQuery.refetch(),
      agendaQuery.refetch(),
      weekQuery.refetch()
    ])
  }

  const unsupportedState =
    calendarSupported === false
      ? buildCapabilityState({
          featureName: "Calendar",
          capabilityName: "calendar workspace",
          endpoint: CALENDAR_SUPPORT_PATH,
          method: "GET",
          serverUrl: connectionConfig?.serverUrl,
          reason: "unsupported"
        })
      : null

  const loadError = calendarsQuery.error ?? agendaQuery.error ?? weekQuery.error
  const loadErrorState = loadError
    ? buildCapabilityState({
        featureName: "Calendar",
        capabilityName: "calendar workspace",
        endpoint: CALENDAR_SUPPORT_PATH,
        method: "GET",
        serverUrl: connectionConfig?.serverUrl,
        error: loadError
      })
    : null

  const partialWarnings = [
    ...(agendaQuery.data?.partial && !agendaQuery.data?.warnings?.length
      ? ["Agenda returned partial calendar data"]
      : agendaQuery.data?.warnings ?? []),
    ...(weekQuery.data?.partial && !weekQuery.data?.warnings?.length
      ? ["Week view returned partial calendar data"]
      : weekQuery.data?.warnings ?? [])
  ]
  const partialState = partialWarnings.length
    ? buildCapabilityState({
        featureName: "Calendar",
        capabilityName: "calendar workspace",
        endpoint: CALENDAR_SUPPORT_PATH,
        method: "GET",
        serverUrl: connectionConfig?.serverUrl,
        reason: "partial",
        partialErrors: partialWarnings,
        title: "Calendar data is partially available",
        message: "Some calendar data loaded, but one or more calendar sources reported partial results."
      })
    : null

  const loading =
    connectionConfigLoading ||
    calendarSupported === null ||
    calendarsQuery.isLoading ||
    agendaQuery.isLoading ||
    weekQuery.isLoading

  return (
    <main className="mx-auto flex w-full max-w-7xl flex-col gap-5 p-6" aria-labelledby="calendar-page-title">
      <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div>
          <Typography.Title id="calendar-page-title" level={2} style={{ marginBottom: 0 }}>
            Calendar
          </Typography.Title>
          <Typography.Text type="secondary">
            {calendars.length} calendars / {agendaItems.length} agenda items
          </Typography.Text>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Segmented
            aria-label="Calendar view"
            value={view}
            onChange={(value) => setView(value as "agenda" | "week")}
            options={[
              { label: "Agenda", value: "agenda" },
              { label: "Week", value: "week" }
            ]}
          />
          <Button type="primary" onClick={openCreateDrawer}>
            New item
          </Button>
        </div>
      </div>

      {loading ? <Spin /> : null}

      {calendarSupported === false ? (
        <RecoveryCallout
          state={unsupportedState?.state ?? "unavailable"}
          title="Calendar is unavailable on this server"
          message={
            "The connected server does not advertise calendar endpoints."
          }
          diagnostics={unsupportedState?.diagnostics}
          primaryAction={{
            label: "Health & diagnostics",
            onClick: () => navigate("/settings/health")
          }}
        />
      ) : null}

      {loadErrorState ? (
        <RecoveryCallout
          state={loadErrorState.state}
          title={loadErrorState.title}
          message={loadErrorState.message}
          diagnostics={loadErrorState.diagnostics}
          primaryAction={{
            label: "Try again",
            onClick: () => {
              void refreshViews()
            }
          }}
          secondaryActions={[
            {
              label: "Health & diagnostics",
              onClick: () => navigate("/settings/health")
            }
          ]}
        />
      ) : null}

      {partialState ? (
        <RecoveryCallout
          state={partialState.state}
          title={partialState.title}
          message={partialState.message}
          diagnostics={partialState.diagnostics}
          primaryAction={{
            label: "Try again",
            onClick: () => {
              void refreshViews()
            }
          }}
        />
      ) : null}

      {calendarSupported === false ? null : (
        <div className="flex flex-col gap-4 md:flex-row md:items-start">
          <CalendarFilterRail
            calendars={calendars}
            selectedCalendarIds={selectedCalendarQueryIds}
            selectedSources={selectedSources}
            selectedKinds={selectedKinds}
            onCalendarChange={setSelectedCalendarIds}
            onSourceChange={setSelectedSources}
            onKindChange={setSelectedKinds}
          />
          {view === "agenda" ? (
            <CalendarAgenda
              calendars={calendars}
              items={agendaItems}
              onSelectItem={openItemDrawer}
            />
          ) : (
            <CalendarWeekView
              calendars={calendars}
              items={weekItems}
              weekStart={weekStart}
              onSelectItem={openItemDrawer}
            />
          )}
        </div>
      )}

      <CalendarItemDrawer
        open={drawerOpen}
        item={selectedItem}
        calendars={calendars}
        onClose={closeDrawer}
        onSaved={refreshViews}
      />
    </main>
  )
}

export default CalendarPage

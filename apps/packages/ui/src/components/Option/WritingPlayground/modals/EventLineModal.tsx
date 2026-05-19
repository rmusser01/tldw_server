import { useQuery } from "@tanstack/react-query"
import { Empty, Modal, Timeline, Tag, Typography } from "antd"
import { useWritingPlaygroundStore } from "@/store/writing-playground"
import { listManuscriptPlotLines } from "@/services/writing-playground"

type EventLineModalProps = {
  open: boolean
  onClose: () => void
}

const EVENT_TYPE_COLORS: Record<string, string> = {
  setup: "blue",
  conflict: "red",
  action: "orange",
  emotional: "purple",
  plot: "gray",
  resolution: "green",
}

export function EventLineModal({ open, onClose }: EventLineModalProps) {
  const { activeProjectId } = useWritingPlaygroundStore()

  const { data: plotLinesData } = useQuery({
    queryKey: ["manuscript-plot-lines", activeProjectId],
    queryFn: () => listManuscriptPlotLines(activeProjectId!),
    enabled: open && !!activeProjectId,
  })

  const plotLines = (plotLinesData as { plot_lines?: unknown[] } | undefined)?.plot_lines ?? []

  const allEvents = plotLines.flatMap((pl: Record<string, unknown>) =>
    ((pl.events as Record<string, unknown>[] | undefined) || []).map((ev: Record<string, unknown>) => ({
      ...ev,
      plotLineTitle: pl.title,
    }))
  )

  // Sort events by sequence or created_at if available
  const sortedEvents = [...allEvents].sort((a: Record<string, unknown>, b: Record<string, unknown>) => {
    if (a.sequence != null && b.sequence != null) return (a.sequence as number) - (b.sequence as number)
    const aTime = (a.created_at as string) || ""
    const bTime = (b.created_at as string) || ""
    return aTime.localeCompare(bTime)
  })

  return (
    <Modal title="Event Line" open={open} onCancel={onClose} footer={null} width={600}>
      {!activeProjectId ? (
        <Empty description="Select a project first" />
      ) : sortedEvents.length === 0 ? (
        <Empty description="No plot events yet. Add events to your plot lines to see the timeline." />
      ) : (
        <div className="max-h-[500px] overflow-y-auto pr-2">
          <Timeline
            items={sortedEvents.map((ev: Record<string, unknown>, idx: number) => ({
              key: (ev.id as string) || idx,
              color: EVENT_TYPE_COLORS[ev.event_type as string] || "gray",
              children: (
                <div className="flex flex-col gap-1">
                  <div className="flex items-center gap-2">
                    <Typography.Text strong>{(ev.title as string) || (ev.description as string) || "Untitled event"}</Typography.Text>
                    {ev.event_type && (
                      <Tag color={EVENT_TYPE_COLORS[ev.event_type as string] || "default"} className="!text-[10px]">
                        {ev.event_type as string}
                      </Tag>
                    )}
                  </div>
                  {ev.description && ev.title && (
                    <Typography.Text type="secondary" className="text-xs">
                      {ev.description as string}
                    </Typography.Text>
                  )}
                  <Typography.Text type="secondary" className="text-[10px]">
                    Plot line: {ev.plotLineTitle as string}
                  </Typography.Text>
                </div>
              ),
            }))}
          />
        </div>
      )}
    </Modal>
  )
}

export default EventLineModal

import { useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { Button, Empty, Modal, Table, Tag, Typography, message } from "antd"
import { useWritingPlaygroundStore } from "@/store/writing-playground"
import {
  listManuscriptPlotLines,
  listManuscriptPlotHoles,
  analyzeProjectPlotHoles,
} from "@/services/writing-playground"

type PlotTrackerModalProps = {
  open: boolean
  onClose: () => void
}

const STATUS_COLORS: Record<string, string> = {
  active: "blue",
  resolved: "green",
  abandoned: "default",
}

const SEVERITY_COLORS: Record<string, string> = {
  critical: "red",
  high: "orange",
  medium: "gold",
  low: "default",
}

export function PlotTrackerModal({ open, onClose }: PlotTrackerModalProps) {
  const { activeProjectId } = useWritingPlaygroundStore()
  const [detecting, setDetecting] = useState(false)

  const { data: plotLinesData } = useQuery({
    queryKey: ["manuscript-plot-lines", activeProjectId],
    queryFn: () => listManuscriptPlotLines(activeProjectId!),
    enabled: open && !!activeProjectId,
  })

  const { data: plotHolesData, refetch: refetchHoles } = useQuery({
    queryKey: ["manuscript-plot-holes", activeProjectId],
    queryFn: () => listManuscriptPlotHoles(activeProjectId!),
    enabled: open && !!activeProjectId,
  })

  const plotLines = (plotLinesData as { plot_lines?: unknown[] } | undefined)?.plot_lines ?? []
  const plotHoles = (plotHolesData as { plot_holes?: unknown[] } | undefined)?.plot_holes ?? []

  const handleDetect = async () => {
    if (!activeProjectId) return
    setDetecting(true)
    try {
      await analyzeProjectPlotHoles(activeProjectId)
      await refetchHoles()
    } catch {
      message.error("Failed to detect plot holes")
    } finally {
      setDetecting(false)
    }
  }

  const plotLineColumns = [
    {
      title: "Title",
      dataIndex: "title",
      key: "title",
      render: (text: string) => <Typography.Text strong>{text}</Typography.Text>,
    },
    {
      title: "Status",
      dataIndex: "status",
      key: "status",
      width: 100,
      render: (status: string) => (
        <Tag color={STATUS_COLORS[status] || "default"}>{status}</Tag>
      ),
    },
    {
      title: "Description",
      dataIndex: "description",
      key: "description",
      ellipsis: true,
    },
  ]

  const plotHoleColumns = [
    {
      title: "Description",
      dataIndex: "description",
      key: "description",
      ellipsis: true,
    },
    {
      title: "Severity",
      dataIndex: "severity",
      key: "severity",
      width: 100,
      render: (severity: string) => (
        <Tag color={SEVERITY_COLORS[severity] || "default"}>{severity}</Tag>
      ),
    },
    {
      title: "Location",
      dataIndex: "location",
      key: "location",
      width: 150,
      ellipsis: true,
    },
  ]

  return (
    <Modal title="Plot Tracker" open={open} onCancel={onClose} footer={null} width={800}>
      {!activeProjectId ? (
        <Empty description="Select a project first" />
      ) : (
        <div className="flex flex-col gap-4">
          <div>
            <Typography.Title level={5} className="!mb-2">Plot Lines</Typography.Title>
            <Table
              dataSource={plotLines}
              columns={plotLineColumns}
              rowKey="id"
              size="small"
              pagination={false}
              locale={{ emptyText: "No plot lines defined yet" }}
            />
          </div>

          <div>
            <div className="flex items-center justify-between mb-2">
              <Typography.Title level={5} className="!mb-0">Plot Holes</Typography.Title>
              <Button
                size="small"
                type="primary"
                loading={detecting}
                onClick={handleDetect}
              >
                AI Detect
              </Button>
            </div>
            <Table
              dataSource={plotHoles}
              columns={plotHoleColumns}
              rowKey="id"
              size="small"
              pagination={false}
              locale={{ emptyText: "No plot holes detected" }}
            />
          </div>
        </div>
      )}
    </Modal>
  )
}

export default PlotTrackerModal

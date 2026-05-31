import React, { useState, useRef, useCallback, useEffect } from "react"
import {
  Card,
  Table,
  Button,
  Input,
  Form,
  Tag,
  Space,
  Popconfirm,
  message
} from "antd"
import {
  deriveAdminGuardFromError,
  sanitizeAdminErrorMessage
} from "./admin-error-utils"
import { Alert } from "@/components/ui/primitives"
import { tldwClient } from "@/services/tldw/TldwApiClient"

const pageContainerStyle: React.CSSProperties = {
  padding: "24px",
  maxWidth: 1200
}

const WatchlistsPage: React.FC = () => {
  // Admin guard state
  const [adminGuard, setAdminGuard] = useState<"forbidden" | "notFound" | null>(null)

  // Watchlists state
  const [watchlists, setWatchlists] = useState<any[]>([])
  const [watchlistsLoading, setWatchlistsLoading] = useState(false)
  const [watchlistForm] = Form.useForm()
  const [creatingWatchlist, setCreatingWatchlist] = useState(false)

  // Alerts state
  const [alerts, setAlerts] = useState<any[]>([])
  const [alertsLoading, setAlertsLoading] = useState(false)

  const initialLoadRef = useRef(false)

  const markAdminGuardFromError = useCallback((err: any) => {
    const guardState = deriveAdminGuardFromError(err)
    if (guardState) setAdminGuard(guardState)
  }, [])

  // ── Watchlists ──

  const loadWatchlists = useCallback(async () => {
    setWatchlistsLoading(true)
    try {
      const result = await tldwClient.listWatchlists()
      setWatchlists(Array.isArray(result) ? result : [])
    } catch (err) {
      markAdminGuardFromError(err)
    } finally {
      setWatchlistsLoading(false)
    }
  }, [markAdminGuardFromError])

  const handleCreateWatchlist = async () => {
    try {
      const values = await watchlistForm.validateFields()
      setCreatingWatchlist(true)
      await tldwClient.createWatchlist({
        name: values.name.trim(),
        description: values.description?.trim() || undefined
      })
      watchlistForm.resetFields()
      message.success("Watchlist created")
      await loadWatchlists()
    } catch (err: any) {
      if (err?.errorFields) return
      message.error(sanitizeAdminErrorMessage(err, "Failed to create watchlist"))
    } finally {
      setCreatingWatchlist(false)
    }
  }

  const handleDeleteWatchlist = async (id: string) => {
    try {
      await tldwClient.deleteWatchlist(id)
      message.success("Watchlist deleted")
      await loadWatchlists()
    } catch (err: any) {
      message.error(sanitizeAdminErrorMessage(err, "Failed to delete watchlist"))
    }
  }

  // ── Alerts ──

  const loadAlerts = useCallback(async () => {
    setAlertsLoading(true)
    try {
      const result = await tldwClient.listMonitoringAlerts({ limit: 100 })
      const items = result?.items ?? (Array.isArray(result) ? result : [])
      setAlerts(items)
    } catch (err) {
      markAdminGuardFromError(err)
    } finally {
      setAlertsLoading(false)
    }
  }, [markAdminGuardFromError])

  const handleAcknowledgeAlert = async (alertId: number) => {
    try {
      await tldwClient.acknowledgeAlert(alertId)
      message.success("Alert acknowledged")
      await loadAlerts()
    } catch (err: any) {
      message.error(sanitizeAdminErrorMessage(err, "Failed to acknowledge alert"))
    }
  }

  const handleDismissAlert = async (alertId: number) => {
    try {
      await tldwClient.dismissAlert(alertId)
      message.success("Alert dismissed")
      await loadAlerts()
    } catch (err: any) {
      message.error(sanitizeAdminErrorMessage(err, "Failed to dismiss alert"))
    }
  }

  // ── Initial Load ──

  useEffect(() => {
    if (initialLoadRef.current) return
    initialLoadRef.current = true
    void loadWatchlists()
    void loadAlerts()
  }, [loadWatchlists, loadAlerts])

  // ── Watchlists Table Columns ──

  const watchlistColumns = [
    {
      title: "Name",
      dataIndex: "name",
      key: "name"
    },
    {
      title: "Description",
      dataIndex: "description",
      key: "description",
      render: (val: string) => val || "\u2014"
    },
    {
      title: "Scope",
      dataIndex: "scope_type",
      key: "scope_type",
      render: (val: string) => <Tag>{val || "user"}</Tag>
    },
    {
      title: "Enabled",
      dataIndex: "enabled",
      key: "enabled",
      render: (val: boolean) => (
        <Tag color={val !== false ? "green" : "default"}>
          {val !== false ? "Yes" : "No"}
        </Tag>
      )
    },
    {
      title: "Rules",
      dataIndex: "rules",
      key: "rules",
      render: (rules: any[]) => (rules?.length ?? 0)
    },
    {
      title: "Actions",
      key: "actions",
      render: (_: any, record: any) => (
        <Popconfirm
          title={`Delete watchlist "${record.name}"?`}
          onConfirm={() => handleDeleteWatchlist(record.id)}
        >
          <Button size="small" danger>Delete</Button>
        </Popconfirm>
      )
    }
  ]

  // ── Alerts Table Columns ──

  const alertColumns = [
    {
      title: "ID",
      dataIndex: "id",
      key: "id",
      width: 60
    },
    {
      title: "Source",
      dataIndex: "source",
      key: "source"
    },
    {
      title: "Category",
      dataIndex: "rule_category",
      key: "rule_category",
      render: (val: string) => val || "\u2014"
    },
    {
      title: "Severity",
      dataIndex: "rule_severity",
      key: "rule_severity",
      render: (severity: string) => {
        const color = severity === "critical" ? "red" : severity === "warning" ? "orange" : "blue"
        return <Tag color={color}>{severity || "info"}</Tag>
      }
    },
    {
      title: "Snippet",
      dataIndex: "text_snippet",
      key: "text_snippet",
      ellipsis: true,
      render: (val: string) => val || "\u2014"
    },
    {
      title: "Created",
      dataIndex: "created_at",
      key: "created_at",
      render: (val: string) => val ? new Date(val).toLocaleString() : "\u2014"
    },
    {
      title: "Actions",
      key: "actions",
      render: (_: any, record: any) => (
        <Space size="small">
          <Button size="small" onClick={() => handleAcknowledgeAlert(record.id)}>
            Acknowledge
          </Button>
          <Popconfirm
            title="Dismiss this alert?"
            onConfirm={() => handleDismissAlert(record.id)}
          >
            <Button size="small" danger>Dismiss</Button>
          </Popconfirm>
        </Space>
      )
    }
  ]

  // ── Render ──

  if (adminGuard === "forbidden") {
    return (
      <div style={pageContainerStyle}>
        <Alert variant="error" title="Access Denied">
          You don't have permission to access watchlists administration.
        </Alert>
      </div>
    )
  }
  if (adminGuard === "notFound") {
    return (
      <div style={pageContainerStyle}>
        <Alert variant="warning" title="Not Available">
          Watchlists administration is not available on this server.
        </Alert>
      </div>
    )
  }

  return (
    <div style={pageContainerStyle}>
      <h2 style={{ marginBottom: 16 }}>Watchlists &amp; Alerts</h2>

      {/* Watchlists Card */}
      <Card
        title="Watchlists"
        style={{ marginBottom: 16 }}
        extra={
          <Button onClick={() => loadWatchlists()} size="small">
            Refresh
          </Button>
        }
      >
        <div style={{ marginBottom: 16 }}>
          <Form form={watchlistForm} layout="inline">
            <Form.Item
              name="name"
              rules={[{ required: true, message: "Name is required" }]}
            >
              <Input placeholder="Watchlist name" style={{ width: 200 }} />
            </Form.Item>
            <Form.Item name="description">
              <Input placeholder="Description (optional)" style={{ width: 250 }} />
            </Form.Item>
            <Form.Item>
              <Button type="primary" onClick={handleCreateWatchlist} loading={creatingWatchlist}>
                Create Watchlist
              </Button>
            </Form.Item>
          </Form>
        </div>
        <Table
          dataSource={watchlists}
          columns={watchlistColumns}
          rowKey="id"
          loading={watchlistsLoading}
          pagination={false}
          size="small"
        />
      </Card>

      {/* Alerts Card */}
      <Card
        title="Monitoring Alerts"
        extra={
          <Button onClick={() => loadAlerts()} size="small">
            Refresh
          </Button>
        }
      >
        <Table
          dataSource={alerts}
          columns={alertColumns}
          rowKey="id"
          loading={alertsLoading}
          pagination={{ pageSize: 20 }}
          size="small"
        />
      </Card>
    </div>
  )
}

export default WatchlistsPage

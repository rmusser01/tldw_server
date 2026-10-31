import React, { useState, useRef, useCallback, useEffect } from "react"
import { useTranslation } from "react-i18next"
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
  const { t } = useTranslation(["settings", "common"])
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
      message.success(t("settings:adminWatchlists.created", "Watchlist created"))
      await loadWatchlists()
    } catch (err: any) {
      if (err?.errorFields) return
      message.error(sanitizeAdminErrorMessage(err, t("settings:adminWatchlists.createFailed", "Failed to create watchlist")))
    } finally {
      setCreatingWatchlist(false)
    }
  }

  const handleDeleteWatchlist = async (id: string) => {
    try {
      await tldwClient.deleteWatchlist(id)
      message.success(t("settings:adminWatchlists.deleted", "Watchlist deleted"))
      await loadWatchlists()
    } catch (err: any) {
      message.error(sanitizeAdminErrorMessage(err, t("settings:adminWatchlists.deleteFailed", "Failed to delete watchlist")))
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
      message.success(t("settings:adminWatchlists.alertAcknowledged", "Alert acknowledged"))
      await loadAlerts()
    } catch (err: any) {
      message.error(sanitizeAdminErrorMessage(err, t("settings:adminWatchlists.acknowledgeFailed", "Failed to acknowledge alert")))
    }
  }

  const handleDismissAlert = async (alertId: number) => {
    try {
      await tldwClient.dismissAlert(alertId)
      message.success(t("settings:adminWatchlists.alertDismissed", "Alert dismissed"))
      await loadAlerts()
    } catch (err: any) {
      message.error(sanitizeAdminErrorMessage(err, t("settings:adminWatchlists.dismissFailed", "Failed to dismiss alert")))
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
      title: t("settings:adminWatchlists.colName", "Name"),
      dataIndex: "name",
      key: "name"
    },
    {
      title: t("settings:adminWatchlists.colDescription", "Description"),
      dataIndex: "description",
      key: "description",
      render: (val: string) => val || "\u2014"
    },
    {
      title: t("settings:adminWatchlists.colScope", "Scope"),
      dataIndex: "scope_type",
      key: "scope_type",
      render: (val: string) => <Tag>{val || t("settings:adminWatchlists.scopeUser", "user")}</Tag>
    },
    {
      title: t("settings:adminWatchlists.colEnabled", "Enabled"),
      dataIndex: "enabled",
      key: "enabled",
      render: (val: boolean) => (
        <Tag color={val !== false ? "green" : "default"}>
          {val !== false ? t("common:yes", "Yes") : t("common:no", "No")}
        </Tag>
      )
    },
    {
      title: t("settings:adminWatchlists.colRules", "Rules"),
      dataIndex: "rules",
      key: "rules",
      render: (rules: any[]) => (rules?.length ?? 0)
    },
    {
      title: t("settings:adminWatchlists.colActions", "Actions"),
      key: "actions",
      render: (_: any, record: any) => (
        <Popconfirm
          title={t("settings:adminWatchlists.deleteConfirm", {
            defaultValue: 'Delete watchlist "{{name}}"?',
            name: record.name
          })}
          onConfirm={() => handleDeleteWatchlist(record.id)}
        >
          <Button size="small" danger>{t("common:delete", "Delete")}</Button>
        </Popconfirm>
      )
    }
  ]

  // ── Alerts Table Columns ──

  const alertColumns = [
    {
      title: t("settings:adminWatchlists.colId", "ID"),
      dataIndex: "id",
      key: "id",
      width: 60
    },
    {
      title: t("settings:adminWatchlists.colSource", "Source"),
      dataIndex: "source",
      key: "source"
    },
    {
      title: t("settings:adminWatchlists.colCategory", "Category"),
      dataIndex: "rule_category",
      key: "rule_category",
      render: (val: string) => val || "\u2014"
    },
    {
      title: t("settings:adminWatchlists.colSeverity", "Severity"),
      dataIndex: "rule_severity",
      key: "rule_severity",
      render: (severity: string) => {
        const color = severity === "critical" ? "red" : severity === "warning" ? "orange" : "blue"
        return <Tag color={color}>{severity || t("settings:adminWatchlists.severityInfo", "info")}</Tag>
      }
    },
    {
      title: t("settings:adminWatchlists.colSnippet", "Snippet"),
      dataIndex: "text_snippet",
      key: "text_snippet",
      ellipsis: true,
      render: (val: string) => val || "\u2014"
    },
    {
      title: t("settings:adminWatchlists.colCreated", "Created"),
      dataIndex: "created_at",
      key: "created_at",
      render: (val: string) => val ? new Date(val).toLocaleString() : "\u2014"
    },
    {
      title: t("settings:adminWatchlists.colActions", "Actions"),
      key: "actions",
      render: (_: any, record: any) => (
        <Space size="small">
          <Button size="small" onClick={() => handleAcknowledgeAlert(record.id)}>
            {t("settings:adminWatchlists.acknowledge", "Acknowledge")}
          </Button>
          <Popconfirm
            title={t("settings:adminWatchlists.dismissConfirm", "Dismiss this alert?")}
            onConfirm={() => handleDismissAlert(record.id)}
          >
            <Button size="small" danger>{t("settings:adminWatchlists.dismiss", "Dismiss")}</Button>
          </Popconfirm>
        </Space>
      )
    }
  ]

  // ── Render ──

  if (adminGuard === "forbidden") {
    return (
      <div style={pageContainerStyle}>
        <Alert variant="error" title={t("settings:adminWatchlists.forbiddenTitle", "Access Denied")}>
          {t(
            "settings:adminWatchlists.forbiddenBody",
            "You don't have permission to access watchlists administration."
          )}
        </Alert>
      </div>
    )
  }
  if (adminGuard === "notFound") {
    return (
      <div style={pageContainerStyle}>
        <Alert variant="warning" title={t("settings:adminWatchlists.notFoundTitle", "Not Available")}>
          {t(
            "settings:adminWatchlists.notFoundBody",
            "Watchlists administration is not available on this server."
          )}
        </Alert>
      </div>
    )
  }

  return (
    <div style={pageContainerStyle}>
      <h1 style={{ marginBottom: 16, fontSize: "1.5rem", fontWeight: 600 }}>{t("settings:adminWatchlists.title", "Watchlists & Alerts")}</h1>

      {/* Watchlists Card */}
      <Card
        title={t("settings:adminWatchlists.watchlistsCardTitle", "Watchlists")}
        style={{ marginBottom: 16 }}
        extra={
          <Button onClick={() => loadWatchlists()} size="small">
            {t("common:refresh", "Refresh")}
          </Button>
        }
      >
        <div style={{ marginBottom: 16 }}>
          <Form form={watchlistForm} layout="inline">
            <Form.Item
              name="name"
              rules={[{ required: true, message: t("settings:adminWatchlists.nameRequired", "Name is required") }]}
            >
              <Input placeholder={t("settings:adminWatchlists.namePlaceholder", "Watchlist name")} style={{ width: 200 }} />
            </Form.Item>
            <Form.Item name="description">
              <Input placeholder={t("settings:adminWatchlists.descriptionPlaceholder", "Description (optional)")} style={{ width: 250 }} />
            </Form.Item>
            <Form.Item>
              <Button type="primary" onClick={handleCreateWatchlist} loading={creatingWatchlist}>
                {t("settings:adminWatchlists.createWatchlist", "Create Watchlist")}
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
        title={t("settings:adminWatchlists.alertsCardTitle", "Monitoring Alerts")}
        extra={
          <Button onClick={() => loadAlerts()} size="small">
            {t("common:refresh", "Refresh")}
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

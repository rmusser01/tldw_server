import React, { useCallback, useEffect, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { Card, Select, Space, Statistic, Table, Tag } from "antd"

import { tldwClient } from "@/services/tldw/TldwApiClient"
import {
  fetchScrapedItemSmartCounts,
  fetchScrapedItems,
  fetchWatchlistRuns,
  fetchWatchlistSources
} from "@/services/watchlists"
import type {
  ScrapedItem,
  ScrapedItemSmartCounts,
  WatchlistRun,
  WatchlistSource
} from "@/types/watchlists"
import { Alert } from "@/components/ui/primitives"
import {
  deriveAdminGuardFromError,
  sanitizeAdminErrorMessage
} from "./admin-error-utils"

/**
 * Fleet oversight for watchlists (#2922): a read-only, user-scoped view of
 * any user's feeds, collected items, and run history. The decision behind
 * this page: the /admin watchlists surface is FLEET oversight with an
 * explicit user selector (like API Keys) - the personal triage tool stays
 * on /watchlists. The backend already gates cross-user reads behind
 * WATCHLIST_SHARING_MODE + admin claims via target_user_id.
 */

type PrivacyState = "allowed" | "private_only" | null

const extractDetail = (err: unknown): string => {
  const message = err instanceof Error ? err.message : String(err ?? "")
  return message
}

const WatchlistsOversightPage: React.FC = () => {
  const { t } = useTranslation(["settings", "common"])
  const [adminGuard, setAdminGuard] = useState<"forbidden" | "notFound" | null>(null)
  const [users, setUsers] = useState<any[]>([])
  const [usersLoading, setUsersLoading] = useState(false)
  const [usersError, setUsersError] = useState<string | null>(null)
  const [selectedUserId, setSelectedUserId] = useState<number | null>(null)

  const [loading, setLoading] = useState(false)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [privacy, setPrivacy] = useState<PrivacyState>(null)
  const [sources, setSources] = useState<WatchlistSource[]>([])
  const [items, setItems] = useState<ScrapedItem[]>([])
  const [runs, setRuns] = useState<WatchlistRun[]>([])
  const [counts, setCounts] = useState<ScrapedItemSmartCounts | null>(null)

  const initialLoadRef = useRef(false)

  const markAdminGuardFromError = useCallback((err: unknown) => {
    const guardState = deriveAdminGuardFromError(err)
    if (guardState) setAdminGuard(guardState)
  }, [])

  const loadUsers = useCallback(async () => {
    setUsersLoading(true)
    setUsersError(null)
    try {
      const result = await tldwClient.listAdminUsers({ limit: 100 })
      const loaded = result.users || []
      setUsers(loaded)
      // Single-user servers have exactly one account - select it directly
      // instead of asking the operator to search for themselves.
      if (loaded.length === 1) {
        setSelectedUserId((current) => current ?? loaded[0].id)
      }
    } catch (err) {
      markAdminGuardFromError(err)
      setUsersError(
        sanitizeAdminErrorMessage(
          err,
          t("settings:adminWatchlistsOversight.usersLoadFailed", "Failed to load the user list.")
        )
      )
    } finally {
      setUsersLoading(false)
    }
  }, [markAdminGuardFromError, t])

  useEffect(() => {
    if (initialLoadRef.current) return
    initialLoadRef.current = true
    void loadUsers()
  }, [loadUsers])

  const loadUserData = useCallback(
    async (userId: number) => {
      setLoading(true)
      setLoadError(null)
      setPrivacy(null)
      try {
        const scope = { target_user_id: userId }
        const [sourcesResp, itemsResp, runsResp, countsResp] = await Promise.all([
          fetchWatchlistSources({ ...scope, size: 100 }),
          fetchScrapedItems({ ...scope, size: 50, sort: "created_desc" }),
          fetchWatchlistRuns({ ...scope, size: 25 }),
          fetchScrapedItemSmartCounts(scope)
        ])
        setSources(sourcesResp?.items ?? [])
        setItems(itemsResp?.items ?? [])
        setRuns(runsResp?.items ?? [])
        setCounts(countsResp ?? null)
        setPrivacy("allowed")
      } catch (err) {
        // Sharing can be disabled per deployment; render that as a designed
        // state, not an error wall.
        if (/watchlists_private_only_mode|watchlists_admin_same_org_required/.test(extractDetail(err))) {
          setPrivacy("private_only")
        } else {
          markAdminGuardFromError(err)
          setLoadError(
            sanitizeAdminErrorMessage(
              err,
              t(
                "settings:adminWatchlistsOversight.loadFailed",
                "Failed to load this user's watchlist data."
              )
            )
          )
        }
      } finally {
        setLoading(false)
      }
    },
    [markAdminGuardFromError, t]
  )

  useEffect(() => {
    if (selectedUserId) {
      void loadUserData(selectedUserId)
    } else {
      setSources([])
      setItems([])
      setRuns([])
      setCounts(null)
      setPrivacy(null)
    }
  }, [selectedUserId, loadUserData])

  if (adminGuard === "forbidden") {
    return (
      <Alert variant="error" title={t("settings:adminWatchlistsOversight.forbiddenTitle", "Access Denied")}>
        {t(
          "settings:adminWatchlistsOversight.forbiddenBody",
          "You don't have permission to inspect other users' watchlists."
        )}
      </Alert>
    )
  }
  if (adminGuard === "notFound") {
    return (
      <Alert variant="warning" title={t("settings:adminWatchlistsOversight.notFoundTitle", "Not Available")}>
        {t(
          "settings:adminWatchlistsOversight.notFoundBody",
          "Watchlist oversight endpoints are not available on this server."
        )}
      </Alert>
    )
  }

  const sourceColumns = [
    {
      title: t("settings:adminWatchlistsOversight.colFeed", "Feed"),
      dataIndex: "name",
      key: "name"
    },
    {
      title: t("settings:adminWatchlistsOversight.colUrl", "URL"),
      dataIndex: "url",
      key: "url",
      ellipsis: true,
      render: (value: string) => <code>{value}</code>
    },
    {
      title: t("settings:adminWatchlistsOversight.colActive", "Active"),
      dataIndex: "active",
      key: "active",
      width: 90,
      render: (value: boolean) =>
        value ? (
          <Tag color="green">{t("settings:adminWatchlistsOversight.active", "Active")}</Tag>
        ) : (
          <Tag>{t("settings:adminWatchlistsOversight.inactive", "Inactive")}</Tag>
        )
    },
    {
      title: t("settings:adminWatchlistsOversight.colLastScraped", "Last scraped"),
      dataIndex: "last_scraped_at",
      key: "last_scraped_at",
      width: 180,
      render: (value: string | null) =>
        value ? new Date(value).toLocaleString() : "—"
    }
  ]

  const itemColumns = [
    {
      title: t("settings:adminWatchlistsOversight.colTitle", "Item"),
      dataIndex: "title",
      key: "title",
      ellipsis: true,
      render: (value: string | null) =>
        value || t("settings:adminWatchlistsOversight.untitled", "Untitled")
    },
    {
      title: t("settings:adminWatchlistsOversight.colPublished", "Published"),
      dataIndex: "published_at",
      key: "published_at",
      width: 180,
      render: (value: string | null) =>
        value ? new Date(value).toLocaleString() : "—"
    },
    {
      title: t("settings:adminWatchlistsOversight.colReviewed", "Reviewed"),
      dataIndex: "reviewed",
      key: "reviewed",
      width: 100,
      render: (value: boolean) =>
        value ? (
          <Tag color="green">{t("settings:adminWatchlistsOversight.reviewed", "Reviewed")}</Tag>
        ) : (
          <Tag color="blue">{t("settings:adminWatchlistsOversight.unread", "Unread")}</Tag>
        )
    }
  ]

  const runColumns = [
    {
      title: t("settings:adminWatchlistsOversight.colRunStarted", "Started"),
      dataIndex: "started_at",
      key: "started_at",
      width: 180,
      render: (value: string | null) =>
        value ? new Date(value).toLocaleString() : "—"
    },
    {
      title: t("settings:adminWatchlistsOversight.colRunStatus", "Status"),
      dataIndex: "status",
      key: "status",
      width: 120,
      render: (value: string) => (
        <Tag color={value === "success" ? "green" : value === "failed" ? "red" : "blue"}>
          {value}
        </Tag>
      )
    },
    {
      title: t("settings:adminWatchlistsOversight.colRunError", "Error"),
      dataIndex: "error_msg",
      key: "error_msg",
      ellipsis: true,
      render: (value: string | null) => value || "—"
    }
  ]

  return (
    <div style={{ padding: "24px", maxWidth: 1200 }}>
      <h1 style={{ marginBottom: 4, fontSize: "1.5rem", fontWeight: 600 }}>
        {t("settings:adminWatchlistsOversight.title", "Watchlists Oversight")}
      </h1>
      <p style={{ marginBottom: 16, color: "var(--color-text-secondary, #888)" }}>
        {t(
          "settings:adminWatchlistsOversight.description",
          "Read-only fleet view of any user's watchlist feeds, collected items, and run health. Users manage their own watchlists on the Watchlists page."
        )}
      </p>

      {usersError && (
        <Alert
          variant="error"
          title={t("settings:adminWatchlistsOversight.usersErrorTitle", "Unable to load users")}
          className="mb-4"
        >
          {usersError}
        </Alert>
      )}

      <Card size="small" style={{ marginBottom: 16 }}>
        <Space>
          <span>{t("settings:adminWatchlistsOversight.selectUser", "Select User:")}</span>
          <Select
            showSearch
            placeholder={t("settings:adminWatchlistsOversight.searchUsers", "Search users...")}
            style={{ width: 300 }}
            loading={usersLoading}
            value={selectedUserId}
            onChange={(val) => setSelectedUserId(val)}
            optionFilterProp="label"
            options={users.map((u: any) => ({
              value: u.id,
              label: `${u.username} (${u.email || t("settings:adminWatchlistsOversight.noEmail", "no email")})`
            }))}
          />
        </Space>
      </Card>

      {!selectedUserId && !usersLoading && !usersError && (
        <Card size="small">
          <p style={{ margin: 0, color: "var(--color-text-secondary, #888)" }}>
            {users.length === 0
              ? t("settings:adminWatchlistsOversight.noUsers", "No users were found on this server.")
              : t(
                  "settings:adminWatchlistsOversight.selectUserHint",
                  "Select a user above to inspect their watchlist activity."
                )}
          </p>
        </Card>
      )}

      {privacy === "private_only" && (
        <Alert
          variant="info"
          title={t(
            "settings:adminWatchlistsOversight.privateTitle",
            "Watchlist sharing is disabled on this server"
          )}
        >
          {t(
            "settings:adminWatchlistsOversight.privateBody",
            "This deployment runs with WATCHLIST_SHARING_MODE set to private_only (or same-org restrictions apply), so admins cannot inspect other users' watchlists. Change the sharing mode on the server to enable fleet oversight."
          )}
        </Alert>
      )}

      {loadError && (
        <Alert
          variant="error"
          title={t("settings:adminWatchlistsOversight.loadErrorTitle", "Unable to load watchlist data")}
        >
          {loadError}
        </Alert>
      )}

      {selectedUserId && privacy === "allowed" && (
        <>
          <Card size="small" style={{ marginBottom: 16 }} data-testid="oversight-summary">
            <Space size="large" wrap>
              <Statistic
                title={t("settings:adminWatchlistsOversight.statFeeds", "Feeds")}
                value={sources.length}
              />
              <Statistic
                title={t("settings:adminWatchlistsOversight.statItems", "Collected items")}
                value={counts?.all ?? items.length}
              />
              <Statistic
                title={t("settings:adminWatchlistsOversight.statUnread", "Unread")}
                value={counts?.unread ?? 0}
              />
              <Statistic
                title={t("settings:adminWatchlistsOversight.statRuns", "Recent runs")}
                value={runs.length}
              />
            </Space>
          </Card>

          <Card
            title={t("settings:adminWatchlistsOversight.feedsCardTitle", "Feeds")}
            size="small"
            style={{ marginBottom: 16 }}
          >
            <Table
              dataSource={sources}
              columns={sourceColumns}
              rowKey="id"
              loading={loading}
              pagination={sources.length > 10 ? { pageSize: 10 } : false}
              size="small"
              locale={{
                emptyText: t(
                  "settings:adminWatchlistsOversight.feedsEmpty",
                  "This user has no watchlist feeds."
                )
              }}
            />
          </Card>

          <Card
            title={t("settings:adminWatchlistsOversight.itemsCardTitle", "Latest collected items")}
            size="small"
            style={{ marginBottom: 16 }}
          >
            <Table
              dataSource={items}
              columns={itemColumns}
              rowKey="id"
              loading={loading}
              pagination={items.length > 10 ? { pageSize: 10 } : false}
              size="small"
              locale={{
                emptyText: t(
                  "settings:adminWatchlistsOversight.itemsEmpty",
                  "No collected items for this user yet."
                )
              }}
            />
          </Card>

          <Card
            title={t("settings:adminWatchlistsOversight.runsCardTitle", "Recent runs")}
            size="small"
          >
            <Table
              dataSource={runs}
              columns={runColumns}
              rowKey="id"
              loading={loading}
              pagination={false}
              size="small"
              locale={{
                emptyText: t(
                  "settings:adminWatchlistsOversight.runsEmpty",
                  "No scrape runs recorded for this user."
                )
              }}
            />
          </Card>
        </>
      )}
    </div>
  )
}

export default WatchlistsOversightPage

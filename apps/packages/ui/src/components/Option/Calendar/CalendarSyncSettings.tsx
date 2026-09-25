import React, { useMemo, useState } from "react"
import {
  Alert,
  Button,
  Drawer,
  Empty,
  Input,
  InputNumber,
  Select,
  Space,
  Spin,
  Tag,
  Typography,
  message
} from "antd"
import { Plus, RefreshCw, Search, Trash2 } from "lucide-react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import type {
  CalendarResponse,
  ExternalCalendarAccountResponse,
  ExternalCalendarBindingResponse,
  ExternalCalendarDiscoveryResponse
} from "@/services/calendar"
import {
  createCalDavAccount,
  createExternalCalendarBinding,
  deleteCalDavAccount,
  discoverExternalCalendars,
  listCalDavAccounts,
  listExternalCalendarBindings,
  revokeCalDavAccount,
  triggerCalendarSync,
  verifyCalDavAccount
} from "@/services/calendar"

interface CalendarSyncSettingsProps {
  calendars: CalendarResponse[]
  onChanged?: () => void | Promise<void>
}

type DiscoveredCalendar = ExternalCalendarDiscoveryResponse["items"][number]

interface AccountDraft {
  display_name: string
  server_url: string
  username: string
  password: string
}

interface BindingDraft {
  calendar_id: number | null
  lookback_days: number | null
  lookahead_days: number | null
}

const DEFAULT_ACCOUNT_DRAFT: AccountDraft = {
  display_name: "",
  server_url: "",
  username: "",
  password: ""
}

const DEFAULT_LOOKBACK_DAYS = 30
const DEFAULT_LOOKAHEAD_DAYS = 120

const queryKeys = {
  accounts: ["calendar", "external", "accounts"] as const,
  bindings: ["calendar", "external", "bindings"] as const
}

const metadataValue = (
  account: ExternalCalendarAccountResponse,
  key: string
): string | null => {
  const value = account.account_metadata?.[key]
  return typeof value === "string" && value.trim() ? value : null
}

const strategyLabel = (value?: Record<string, unknown> | null): string | null => {
  const strategy = value?.sync_strategy
  if (typeof strategy !== "string" || !strategy.trim()) return null
  return strategy.replace(/_/g, " ")
}

const titleCase = (value: string): string =>
  value ? `${value.charAt(0).toUpperCase()}${value.slice(1)}` : value

const bindingKey = (accountId: number, remoteCalendarId: string): string =>
  `${accountId}:${remoteCalendarId}`

const normalizeNumber = (value: number | null, fallback: number): number =>
  typeof value === "number" && Number.isFinite(value) ? value : fallback

export const CalendarSyncSettings: React.FC<CalendarSyncSettingsProps> = ({
  calendars,
  onChanged
}) => {
  const queryClient = useQueryClient()
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [accountDraft, setAccountDraft] = useState<AccountDraft>(DEFAULT_ACCOUNT_DRAFT)
  const [discoveredByAccount, setDiscoveredByAccount] = useState<
    Record<number, DiscoveredCalendar[]>
  >({})
  const [bindingDrafts, setBindingDrafts] = useState<Record<string, BindingDraft>>({})

  const accountsQuery = useQuery({
    queryKey: queryKeys.accounts,
    queryFn: listCalDavAccounts
  })

  const accounts = accountsQuery.data?.items ?? []

  const bindingsQuery = useQuery({
    queryKey: queryKeys.bindings,
    queryFn: async () => {
      const entries = await Promise.all(
        accounts.map(async (account) => {
          const response = await listExternalCalendarBindings(account.id)
          return [account.id, response.items] as const
        })
      )
      return new Map<number, ExternalCalendarBindingResponse[]>(entries)
    },
    enabled: accounts.length > 0
  })

  const accountBindings = bindingsQuery.data ?? new Map<number, ExternalCalendarBindingResponse[]>()
  const calendarOptions = useMemo(
    () =>
      calendars.map((calendar) => ({
        label: calendar.name,
        value: calendar.id
      })),
    [calendars]
  )

  const invalidateSyncQueries = async () => {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: queryKeys.accounts }),
      queryClient.invalidateQueries({ queryKey: queryKeys.bindings })
    ])
  }

  const notifyChanged = async () => {
    await invalidateSyncQueries()
    await onChanged?.()
  }

  const addAccountMutation = useMutation({
    mutationFn: async (draft: AccountDraft) => {
      const account = await createCalDavAccount({
        display_name: draft.display_name.trim(),
        server_url: draft.server_url.trim(),
        username: draft.username.trim(),
        password: draft.password
      })
      if (draft.password) {
        await verifyCalDavAccount(account.id, { password: draft.password })
      }
      return account
    },
    onSuccess: async () => {
      setDrawerOpen(false)
      setAccountDraft(DEFAULT_ACCOUNT_DRAFT)
      await notifyChanged()
    },
    onError: (error: Error) => {
      message.error(error.message || "Unable to add CalDAV account")
    }
  })

  const discoverMutation = useMutation({
    mutationFn: async (accountId: number) => {
      const response = await discoverExternalCalendars(accountId)
      return { accountId, items: response.items }
    },
    onSuccess: ({ accountId, items }) => {
      setDiscoveredByAccount((current) => ({
        ...current,
        [accountId]: items
      }))
      setBindingDrafts((current) => {
        const next = { ...current }
        for (const item of items) {
          next[bindingKey(accountId, item.remote_calendar_id)] = {
            calendar_id: calendars[0]?.id ?? null,
            lookback_days: DEFAULT_LOOKBACK_DAYS,
            lookahead_days: DEFAULT_LOOKAHEAD_DAYS
          }
        }
        return next
      })
    },
    onError: (error: Error) => {
      message.error(error.message || "Unable to discover remote calendars")
    }
  })

  const bindMutation = useMutation({
    mutationFn: async ({
      accountId,
      remoteCalendar,
      draft
    }: {
      accountId: number
      remoteCalendar: DiscoveredCalendar
      draft: BindingDraft
    }) => {
      if (!draft.calendar_id) {
        throw new Error("Choose a local calendar before binding")
      }
      return await createExternalCalendarBinding({
        account_id: accountId,
        calendar_id: draft.calendar_id,
        remote_calendar_id: remoteCalendar.remote_calendar_id,
        remote_display_name: remoteCalendar.remote_display_name ?? null,
        lookback_days: normalizeNumber(draft.lookback_days, DEFAULT_LOOKBACK_DAYS),
        lookahead_days: normalizeNumber(draft.lookahead_days, DEFAULT_LOOKAHEAD_DAYS),
        provider_capabilities: remoteCalendar.provider_capabilities ?? null
      })
    },
    onSuccess: async () => {
      await notifyChanged()
    },
    onError: (error: Error) => {
      message.error(error.message || "Unable to bind remote calendar")
    }
  })

  const syncNowMutation = useMutation({
    mutationFn: async (bindingId: number) =>
      await triggerCalendarSync(bindingId, { reason: "manual" }),
    onSuccess: async () => {
      await notifyChanged()
    },
    onError: (error: Error) => {
      message.error(error.message || "Unable to queue calendar sync")
    }
  })

  const revokeAccountMutation = useMutation({
    mutationFn: async (accountId: number) => await revokeCalDavAccount(accountId),
    onSuccess: async () => {
      await notifyChanged()
    },
    onError: (error: Error) => {
      message.error(error.message || "Unable to revoke CalDAV account")
    }
  })

  const deleteAccountMutation = useMutation({
    mutationFn: async (accountId: number) => await deleteCalDavAccount(accountId),
    onSuccess: async () => {
      await notifyChanged()
    },
    onError: (error: Error) => {
      message.error(error.message || "Unable to delete CalDAV account")
    }
  })

  const updateBindingDraft = (
    accountId: number,
    remoteCalendarId: string,
    patch: Partial<BindingDraft>
  ) => {
    const key = bindingKey(accountId, remoteCalendarId)
    setBindingDrafts((current) => ({
      ...current,
      [key]: {
        calendar_id: calendars[0]?.id ?? null,
        lookback_days: DEFAULT_LOOKBACK_DAYS,
        lookahead_days: DEFAULT_LOOKAHEAD_DAYS,
        ...current[key],
        ...patch
      }
    }))
  }

  const handleDeleteAccount = (account: ExternalCalendarAccountResponse) => {
    if (!window.confirm(`Delete CalDAV account "${account.display_name}"?`)) return
    deleteAccountMutation.mutate(account.id)
  }

  const loading = accountsQuery.isLoading || (accounts.length > 0 && bindingsQuery.isLoading)

  return (
    <section className="flex min-w-0 flex-1 flex-col gap-4" aria-labelledby="calendar-sync-title">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <Typography.Title id="calendar-sync-title" level={3} style={{ margin: 0 }}>
            Sync
          </Typography.Title>
          <Typography.Text type="secondary">
            CalDAV accounts import provider-owned calendars into local tldw calendars.
          </Typography.Text>
        </div>
        <Button
          type="primary"
          icon={<Plus size={16} aria-hidden="true" />}
          onClick={() => setDrawerOpen(true)}
        >
          Add CalDAV account
        </Button>
      </div>

      {calendars.length === 0 ? (
        <Alert
          type="warning"
          showIcon
          message="Create a local calendar before binding remote calendars."
        />
      ) : null}

      {loading ? <Spin /> : null}

      {!loading && accounts.length === 0 ? (
        <Empty description="No CalDAV accounts" />
      ) : null}

      <div className="grid gap-3">
        {!loading
          ? accounts.map((account) => {
              const bindings = accountBindings.get(account.id) ?? []
              const discovered = discoveredByAccount[account.id] ?? []
              return (
                <section
                  key={account.id}
                  role="region"
                  aria-label={account.display_name}
                  className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm"
                >
              <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                <div className="min-w-0">
                  <div className="flex flex-wrap items-center gap-2">
                    <Typography.Title level={4} style={{ margin: 0 }}>
                      {account.display_name}
                    </Typography.Title>
                    <Tag>{account.status || "active"}</Tag>
                  </div>
                  <Typography.Text type="secondary" className="block">
                    {metadataValue(account, "username") ?? "No username"} /{" "}
                    {metadataValue(account, "server_url") ?? "No server URL"}
                  </Typography.Text>
                </div>
                <Space wrap>
                  <Button
                    icon={<Search size={16} aria-hidden="true" />}
                    loading={discoverMutation.isPending}
                    onClick={() => discoverMutation.mutate(account.id)}
                  >
                    Discover calendars
                  </Button>
                  <Button
                    icon={<RefreshCw size={16} aria-hidden="true" />}
                    onClick={() => revokeAccountMutation.mutate(account.id)}
                  >
                    Revoke account
                  </Button>
                  <Button
                    danger
                    icon={<Trash2 size={16} aria-hidden="true" />}
                    onClick={() => handleDeleteAccount(account)}
                  >
                    Delete account
                  </Button>
                </Space>
              </div>

              <div className="mt-4 grid gap-3">
                {bindings.length === 0 ? (
                  <Typography.Text type="secondary">
                    No remote calendars are bound yet.
                  </Typography.Text>
                ) : null}
                {bindings.map((binding) => {
                  const strategy = strategyLabel(binding.provider_capabilities)
                  return (
                    <div
                      key={binding.id}
                      className="flex flex-col gap-2 rounded-md border border-slate-200 p-3 md:flex-row md:items-center md:justify-between"
                    >
                      <div>
                        <Typography.Text strong>
                          {binding.remote_display_name || binding.remote_calendar_id}
                        </Typography.Text>
                        <div className="flex flex-wrap gap-2 text-sm text-slate-600">
                          {strategy ? <span>{strategy}</span> : null}
                          <span>{binding.lookback_days}d back</span>
                          <span>{binding.lookahead_days}d ahead</span>
                        </div>
                        {binding.last_error ? (
                          <Typography.Text type="danger" className="block">
                            Sync error: {binding.last_error}
                          </Typography.Text>
                        ) : null}
                      </div>
                      <Button
                        icon={<RefreshCw size={16} aria-hidden="true" />}
                        loading={syncNowMutation.isPending}
                        onClick={() => syncNowMutation.mutate(binding.id)}
                      >
                        Sync now
                      </Button>
                    </div>
                  )
                })}
              </div>

              {discovered.length ? (
                <section
                  role="region"
                  aria-label="Discovered calendars"
                  className="mt-4 grid gap-3 rounded-md border border-dashed border-slate-300 p-3"
                >
                  {discovered.map((remoteCalendar) => {
                    const key = bindingKey(account.id, remoteCalendar.remote_calendar_id)
                    const draft =
                      bindingDrafts[key] ?? {
                        calendar_id: calendars[0]?.id ?? null,
                        lookback_days: DEFAULT_LOOKBACK_DAYS,
                        lookahead_days: DEFAULT_LOOKAHEAD_DAYS
                      }
                    const remoteName =
                      remoteCalendar.remote_display_name || remoteCalendar.remote_calendar_id
                    const remoteStrategy = strategyLabel(remoteCalendar.provider_capabilities)
                    return (
                      <div
                        key={remoteCalendar.remote_calendar_id}
                        className="grid gap-3 rounded-md border border-slate-200 bg-slate-50 p-3"
                      >
                        <div className="flex flex-wrap items-center gap-2">
                          <Typography.Text strong>{remoteName}</Typography.Text>
                          {remoteStrategy ? <Tag>{titleCase(remoteStrategy)}</Tag> : null}
                        </div>
                        <div className="grid gap-3 md:grid-cols-[minmax(180px,1fr)_140px_140px_auto] md:items-end">
                          <label className="grid gap-1 text-sm font-medium text-slate-700">
                            Import into
                            <Select
                              value={draft.calendar_id ?? undefined}
                              options={calendarOptions}
                              onChange={(calendar_id) =>
                                updateBindingDraft(account.id, remoteCalendar.remote_calendar_id, {
                                  calendar_id
                                })
                              }
                            />
                          </label>
                          <label className="grid gap-1 text-sm font-medium text-slate-700">
                            Lookback days
                            <InputNumber
                              aria-label="Lookback days"
                              min={0}
                              max={3650}
                              value={draft.lookback_days}
                              onChange={(value) =>
                                updateBindingDraft(account.id, remoteCalendar.remote_calendar_id, {
                                  lookback_days: value
                                })
                              }
                            />
                          </label>
                          <label className="grid gap-1 text-sm font-medium text-slate-700">
                            Lookahead days
                            <InputNumber
                              aria-label="Lookahead days"
                              min={0}
                              max={3650}
                              value={draft.lookahead_days}
                              onChange={(value) =>
                                updateBindingDraft(account.id, remoteCalendar.remote_calendar_id, {
                                  lookahead_days: value
                                })
                              }
                            />
                          </label>
                          <Button
                            type="primary"
                            loading={bindMutation.isPending}
                            disabled={!draft.calendar_id}
                            onClick={() =>
                              bindMutation.mutate({
                                accountId: account.id,
                                remoteCalendar,
                                draft
                              })
                            }
                          >
                            Bind {remoteName}
                          </Button>
                        </div>
                      </div>
                    )
                  })}
                </section>
              ) : null}
                </section>
              )
            })
          : null}
      </div>

      <Drawer
        title="Add CalDAV account"
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        getContainer={false}
        destroyOnHidden
        extra={
          <Button
            type="primary"
            loading={addAccountMutation.isPending}
            onClick={() => addAccountMutation.mutate(accountDraft)}
          >
            Save and verify account
          </Button>
        }
      >
        <div className="grid gap-4">
          <label className="grid gap-1 text-sm font-medium text-slate-700">
            Account name
            <Input
              value={accountDraft.display_name}
              onChange={(event) =>
                setAccountDraft((current) => ({
                  ...current,
                  display_name: event.target.value
                }))
              }
            />
          </label>
          <label className="grid gap-1 text-sm font-medium text-slate-700">
            Server URL
            <Input
              value={accountDraft.server_url}
              onChange={(event) =>
                setAccountDraft((current) => ({
                  ...current,
                  server_url: event.target.value
                }))
              }
            />
          </label>
          <label className="grid gap-1 text-sm font-medium text-slate-700">
            Username
            <Input
              value={accountDraft.username}
              onChange={(event) =>
                setAccountDraft((current) => ({
                  ...current,
                  username: event.target.value
                }))
              }
            />
          </label>
          <label className="grid gap-1 text-sm font-medium text-slate-700">
            Password
            <Input.Password
              value={accountDraft.password}
              onChange={(event) =>
                setAccountDraft((current) => ({
                  ...current,
                  password: event.target.value
                }))
              }
            />
          </label>
        </div>
      </Drawer>
    </section>
  )
}

export default CalendarSyncSettings

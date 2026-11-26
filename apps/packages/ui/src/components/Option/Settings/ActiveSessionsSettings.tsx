import React, { useCallback, useEffect, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { Button, Modal, Table } from "antd"
import { Alert, Badge } from "@/components/ui/primitives"
import { EmptyState, LoadingState } from "@/components/ui/feedback"
import { useCanonicalConnectionConfig } from "@/hooks/useCanonicalConnectionConfig"
import {
  isInteractiveRequiredError,
  listAuthSessions,
  revokeAuthSession,
  revokeOtherAuthSessions,
  type AuthSession
} from "@/services/auth-sessions"

const timeFormat = new Intl.DateTimeFormat(undefined, {
  dateStyle: "medium",
  timeStyle: "short"
})
const formatTime = (iso: string) => {
  const date = new Date(iso)
  return Number.isNaN(date.getTime()) ? iso : timeFormat.format(date)
}

type ActionError = "interactive" | "generic"

export const ActiveSessionsSettings = () => {
  const { t } = useTranslation(["settings", "common"])
  const { config } = useCanonicalConnectionConfig()
  const [sessions, setSessions] = useState<AuthSession[] | null>(null)
  const [loadFailed, setLoadFailed] = useState(false)
  const [actionError, setActionError] = useState<ActionError | null>(null)
  const [status, setStatus] = useState("")
  const [busy, setBusy] = useState(false)
  const headingRef = useRef<HTMLHeadingElement>(null)

  /** Returns the fresh list (null on failure). */
  const load = useCallback(async (): Promise<AuthSession[] | null> => {
    try {
      const rows = await listAuthSessions()
      setSessions(rows)
      setLoadFailed(false)
      return rows
    } catch {
      setLoadFailed(true)
      return null
    }
  }, [])

  useEffect(() => {
    void load()
  }, [load])

  const others = (sessions ?? []).filter((s) => !s.is_current)
  const readOnly = !!sessions?.length && !sessions.some((s) => s.is_current)
  const remaining = (rows: AuthSession[] | null) =>
    t("settings:activeSessions.remaining", {
      count: (rows ?? []).filter((s) => !s.is_current).length,
      defaultValue: "{count, plural, one {# other session remains.} other {# other sessions remain.}}"
    })

  /** Runs a revoke, reloads, announces the result and moves focus to the heading. */
  const act = async (revoke: () => Promise<string>, withRemaining: boolean) => {
    setBusy(true)
    setActionError(null)
    try {
      const done = await revoke()
      const rows = await load()
      setStatus(withRemaining ? `${done} ${remaining(rows)}` : done)
      headingRef.current?.focus()
    } catch (error) {
      setActionError(isInteractiveRequiredError(error) ? "interactive" : "generic")
    } finally {
      setBusy(false)
    }
  }

  const signOut = (session: AuthSession) =>
    act(async () => {
      await revokeAuthSession(session.id)
      return t("settings:activeSessions.signedOut", {
        device: session.device_label,
        defaultValue: `Signed out ${session.device_label}.`
      })
    }, true)

  const signOutOthers = () =>
    Modal.confirm({
      title: t("settings:activeSessions.confirmTitle", "Sign out all other sessions?"),
      content: t("settings:activeSessions.confirmContent", {
        count: others.length,
        defaultValue:
          "{count, plural, one {# other session} other {# other sessions}} will be signed out. This device stays signed in."
      }),
      okText: t("settings:activeSessions.confirmOk", "Sign out other sessions"),
      cancelText: t("common:cancel", "Cancel"),
      onOk: () =>
        act(async () => {
          const { sessions_revoked } = await revokeOtherAuthSessions()
          return t("settings:activeSessions.signedOutOthers", {
            count: sessions_revoked,
            defaultValue:
              "{count, plural, one {Signed out # other session.} other {Signed out # other sessions.}}"
          })
        }, false)
    })

  const columns = [
    {
      title: t("settings:activeSessions.colDevice", "Device"),
      key: "device",
      render: (_: unknown, s: AuthSession) => (
        <span className="flex flex-wrap items-center gap-2">
          <span className="font-medium">{s.device_label}</span>
          {s.is_current && (
            <Badge variant="primary">
              {t("settings:activeSessions.thisDevice", "This device")}
            </Badge>
          )}
        </span>
      )
    },
    {
      title: t("settings:activeSessions.colIp", "IP address"),
      dataIndex: "ip_address",
      key: "ip",
      render: (ip: string | null) => ip ?? "-"
    },
    {
      title: t("settings:activeSessions.colSignedIn", "Signed in"),
      dataIndex: "created_at",
      key: "created",
      render: (iso: string) => <span title={iso}>{formatTime(iso)}</span>
    },
    {
      title: t("settings:activeSessions.colLastActive", "Last active"),
      dataIndex: "last_activity",
      key: "active",
      render: (iso: string) => <span title={iso}>{formatTime(iso)}</span>
    },
    {
      title: "",
      key: "actions",
      render: (_: unknown, s: AuthSession) =>
        s.is_current ? null : (
          <Button
            size="small"
            disabled={readOnly || busy}
            aria-label={t("settings:activeSessions.signOutAria", {
              device: s.device_label,
              time: formatTime(s.created_at),
              defaultValue: `Sign out ${s.device_label}, signed in ${formatTime(s.created_at)}`
            })}
            onClick={() => void signOut(s)}>
            {t("settings:activeSessions.signOut", "Sign out")}
          </Button>
        )
    }
  ]

  let body: React.ReactNode
  if (loadFailed && sessions === null) {
    body = (
      <Alert
        variant="error"
        title={t("settings:activeSessions.loadErrorTitle", "Could not load your sessions")}
        action={{
          label: t("settings:activeSessions.tryAgain", "Try again"),
          onClick: () => void load()
        }}
      />
    )
  } else if (sessions === null) {
    body = <LoadingState mode="spinner" label={t("settings:activeSessions.loading", "Loading your sessions...")} />
  } else if (sessions.length === 0) {
    body = (
      <EmptyState
        title={t("settings:activeSessions.emptyTitle", "No sessions to show")}
        description={t(
          "settings:activeSessions.emptyDescription",
          "Sessions appear here for browsers signed in to this account."
        )}
      />
    )
  } else {
    body = (
      <div className="overflow-x-auto">
        <Table
          dataSource={sessions}
          columns={columns}
          rowKey="id"
          pagination={false}
          size="small"
        />
      </div>
    )
  }

  return (
    <div className="mx-auto max-w-4xl px-4 py-6">
      <h1
        ref={headingRef}
        tabIndex={-1}
        className="text-xl font-semibold text-text outline-none">
        {t("settings:activeSessions.title", "Active sessions")}
      </h1>
      <p className="mt-1 text-sm text-text-muted">
        {t(
          "settings:activeSessions.intro",
          "These are the browsers and devices signed in to your account. Sign out any you do not recognise."
        )}
      </p>
      {config?.authMode === "single-user" && (
        <p className="mt-1 text-sm text-text-muted">
          {t("settings:activeSessions.singleUserNote", "If you did not sign in, rotate your API key.")}
        </p>
      )}
      <div className="my-4 flex justify-end">
        <Button disabled={readOnly || busy || others.length === 0} onClick={signOutOthers}>
          {t("settings:activeSessions.signOutOthers", "Sign out all other sessions")}
        </Button>
      </div>
      <p role="status" className="mb-3 text-sm text-text-muted">
        {status}
      </p>
      {readOnly && (
        <Alert
          variant="info"
          className="mb-4"
          title={t("settings:activeSessions.readOnlyTitle", "Sessions are read-only on this connection")}>
          {t(
            "settings:activeSessions.readOnlyBody",
            "This connection uses an API key. Sign in with the browser to sign sessions out."
          )}
        </Alert>
      )}
      {actionError === "interactive" && (
        <Alert variant="warning" className="mb-4">
          {t("settings:activeSessions.interactiveRequired", "Sign in from a browser to sign sessions out.")}
        </Alert>
      )}
      {actionError === "generic" && (
        <Alert variant="error" className="mb-4">
          {t("settings:activeSessions.signOutError", "Could not sign out that session. Try again.")}
        </Alert>
      )}
      {body}
    </div>
  )
}

export default ActiveSessionsSettings

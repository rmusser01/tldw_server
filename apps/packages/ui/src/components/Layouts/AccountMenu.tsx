import React from "react"
import { Link } from "react-router-dom"
import { useTranslation } from "react-i18next"
import { Bell, ChevronRight, LogOut, Settings, UserRound } from "lucide-react"
import { tldwAuth, type UserInfo } from "@/services/tldw/TldwAuth"
import { watchChatAccountChanges } from "@/services/chat-account-boundary"
import type { NotificationLifecycleState } from "@/services/notification-lifecycle"

type AccountMenuProps = {
  onClose: () => void
  onOpenNotifications?: () => void
  notificationCount?: number
  notificationState?: NotificationLifecycleState
  onRetryNotifications?: () => void | Promise<void>
}

export function AccountMenu({
  onClose,
  onOpenNotifications,
  notificationCount = 0,
  notificationState = "active",
  onRetryNotifications
}: AccountMenuProps) {
  const { t } = useTranslation("option")
  const [user, setUser] = React.useState<UserInfo | null>(null)
  const [loading, setLoading] = React.useState(true)
  const [signingOut, setSigningOut] = React.useState(false)
  const [logoutFailed, setLogoutFailed] = React.useState(false)
  const currentRef = React.useRef(true)

  React.useEffect(() => {
    let current = true
    currentRef.current = true
    const unwatch = watchChatAccountChanges((invalidated) => {
      if (!invalidated) return
      current = false
      currentRef.current = false
      setUser(null)
      setLoading(false)
      onClose()
    })
    const load = async () => {
      try {
        const identity = await tldwAuth.getCurrentUser()
        if (current) setUser(identity)
      } catch {
        if (current) setUser(null)
      } finally {
        if (current) setLoading(false)
      }
    }
    void load()
    return () => {
      current = false
      currentRef.current = false
      unwatch()
    }
  }, [onClose])

  const signOut = async () => {
    if (signingOut || !currentRef.current) return
    setSigningOut(true)
    setLogoutFailed(false)
    try {
      await tldwAuth.logout()
      if (currentRef.current) onClose()
    } catch {
      if (currentRef.current) setLogoutFailed(true)
    } finally {
      if (currentRef.current) setSigningOut(false)
    }
  }

  const rowClass =
    "flex min-h-11 w-full items-center gap-3 rounded-md px-3 py-2 text-sm text-text hover:bg-surface2 focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
  const notificationStatus = {
    idle: null,
    active: null,
    connecting: t(
      "header.notificationsConnectingAnnouncement",
      "Notifications are connecting."
    ),
    degraded: t(
      "header.notificationsReconnectingAnnouncement",
      "Notifications are reconnecting."
    ),
    "auth-required": t(
      "header.notificationsSignInDescription",
      "Sign in again to resume personal notifications."
    ),
    unavailable: t(
      "header.notificationsUnavailableDescription",
      "Notifications unavailable for this account."
    )
  }[notificationState]

  return (
    <div>
      <div className="border-b border-border px-3 pb-3">
        {user ? (
          <>
            <p className="break-words text-sm font-semibold text-text">
              {user.username}
            </p>
            {user.email ? (
              <p className="mt-1 break-all text-xs text-text-muted">
                {user.email}
              </p>
            ) : null}
          </>
        ) : (
          <p role="status" className="text-sm text-text-muted">
            {loading
              ? t("header.accountLoading", "Loading account details...")
              : t("header.accountUnavailable", "Account details unavailable")}
          </p>
        )}
      </div>
      <nav
        aria-label={t("header.account", "Account")}
        className="mt-2 space-y-1">
        <Link to="/settings/tldw" onClick={onClose} className={rowClass}>
          <UserRound className="h-4 w-4 shrink-0" aria-hidden="true" />
          <span>{t("header.accountSettings", "Account settings")}</span>
          <ChevronRight
            className="ml-auto h-4 w-4 shrink-0"
            aria-hidden="true"
          />
        </Link>
        <Link to="/settings" onClick={onClose} className={rowClass}>
          <Settings className="h-4 w-4 shrink-0" aria-hidden="true" />
          <span>{t("settings", "Settings")}</span>
          <ChevronRight
            className="ml-auto h-4 w-4 shrink-0"
            aria-hidden="true"
          />
        </Link>
        {onOpenNotifications ? (
          <>
            <button
              type="button"
              className={rowClass}
              onClick={() => {
                onClose()
                onOpenNotifications()
              }}>
              <Bell className="h-4 w-4 shrink-0" aria-hidden="true" />
              <span>{t("header.notifications", "Notifications")}</span>
              {notificationCount > 0 ? (
                <span className="ml-auto text-xs text-text-muted">
                  {t("header.notificationsUnreadCount", {
                    count: notificationCount,
                    defaultValue: `${notificationCount} unread`
                  })}
                </span>
              ) : null}
              <ChevronRight
                className={`${notificationCount > 0 ? "" : "ml-auto"} h-4 w-4 shrink-0`}
                aria-hidden="true"
              />
            </button>
            {notificationStatus ? (
              <p className="px-3 text-xs text-text-muted">
                {notificationStatus}
              </p>
            ) : null}
            {(notificationState === "degraded" ||
              notificationState === "unavailable") &&
            onRetryNotifications ? (
              <button
                type="button"
                onClick={() => void onRetryNotifications()}
                className={rowClass}>
                {t("header.notificationsTryAgain", "Try again")}
              </button>
            ) : null}
          </>
        ) : null}
      </nav>
      <div className="mt-2 border-t border-border pt-2">
        <button
          type="button"
          onClick={() => void signOut()}
          disabled={signingOut}
          className={`${rowClass} disabled:cursor-wait disabled:opacity-60`}>
          <LogOut className="h-4 w-4 shrink-0" aria-hidden="true" />
          {signingOut
            ? t("header.signingOut", "Signing out...")
            : t("header.signOut", "Sign out")}
        </button>
        {logoutFailed ? (
          <p role="alert" className="px-3 text-xs text-danger">
            {t("header.signOutFailed", "Could not sign out. Try again.")}
          </p>
        ) : null}
      </div>
    </div>
  )
}

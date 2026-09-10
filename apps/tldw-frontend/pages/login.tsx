import React from "react"
import dynamic from "next/dynamic"
import Link from "next/link"
import { useRouter } from "next/router"

import { RouteRedirect } from "@web/components/navigation/RouteRedirect"
import { isHostedTldwDeployment } from "@/services/tldw/deployment-mode"

const TldwSettings = dynamic(
  () => import("@/components/Option/Settings/tldw").then((m) => m.TldwSettings),
  { ssr: false }
)

type LoginTarget = {
  resolved: boolean
  serverUrl: string | null
  multiUser: boolean
}

/**
 * Focused sign-in screen for self-host multi-user servers (#2919).
 *
 * Multi-user visitors used to be redirected into the full settings shell
 * with a below-the-fold "Login Required" banner - there was no login
 * screen at all. This page puts username/password front and center once a
 * multi-user server is configured; server configuration itself stays in
 * settings, one small link away.
 */
const SelfHostLogin: React.FC<{ serverUrl: string }> = ({ serverUrl }) => {
  const router = useRouter()
  const [username, setUsername] = React.useState("")
  const [password, setPassword] = React.useState("")
  const [error, setError] = React.useState<string | null>(null)
  const [submitting, setSubmitting] = React.useState(false)

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault()
    if (!username.trim() || !password) {
      setError("Enter your username and password.")
      return
    }
    setSubmitting(true)
    setError(null)
    try {
      const { tldwAuth } = await import("@/services/tldw/TldwAuth")
      await tldwAuth.login({ username: username.trim(), password })
      void router.push("/")
    } catch (err) {
      const { mapMultiUserLoginErrorMessage } = await import(
        "@/services/auth-errors"
      )
      // The mapper is i18n-aware; the login page renders English defaults,
      // matching the rest of the web shell's fallback behavior.
      setError(
        mapMultiUserLoginErrorMessage(
          ((key: string, fallback?: string) => fallback ?? key) as never,
          err,
          "settings"
        )
      )
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-bg px-4">
      <div className="w-full max-w-sm rounded-xl border border-border bg-surface p-8 shadow-sm">
        <h1 className="text-2xl font-semibold text-text">Sign in to tldw</h1>
        <p className="mt-1 text-sm text-text-muted">
          {serverUrl}{" "}
          <Link className="underline hover:text-text" href="/settings/tldw">
            Change server
          </Link>
        </p>
        <form className="mt-6 space-y-4" onSubmit={handleSubmit}>
          <label className="block">
            <span className="text-sm font-medium text-text">Username</span>
            <input
              name="username"
              autoComplete="username"
              autoFocus
              value={username}
              onChange={(event) => setUsername(event.target.value)}
              className="mt-1 w-full rounded-md border border-border bg-bg px-3 py-2 text-sm text-text outline-none focus:border-primary"
            />
          </label>
          <label className="block">
            <span className="text-sm font-medium text-text">Password</span>
            <input
              name="password"
              type="password"
              autoComplete="current-password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              className="mt-1 w-full rounded-md border border-border bg-bg px-3 py-2 text-sm text-text outline-none focus:border-primary"
            />
          </label>
          {error ? (
            <p role="alert" className="text-sm text-danger">
              {error}
            </p>
          ) : null}
          <button
            type="submit"
            disabled={submitting}
            className="w-full rounded-md bg-primary px-3 py-2 text-sm font-medium text-white hover:bg-primaryStrong disabled:opacity-60"
          >
            {submitting ? "Signing in…" : "Sign in"}
          </button>
        </form>
        <p className="mt-4 text-xs text-text-muted">
          Trouble signing in? Server URL and auth mode are configured in{" "}
          <Link className="underline hover:text-text" href="/settings/tldw">
            tldw settings
          </Link>
          .
        </p>
      </div>
    </div>
  )
}

const LoginPage = () => {
  const hostedMode = isHostedTldwDeployment()
  const [target, setTarget] = React.useState<LoginTarget>({
    resolved: false,
    serverUrl: null,
    multiUser: false
  })

  React.useEffect(() => {
    if (hostedMode) return
    let cancelled = false
    void (async () => {
      try {
        const { tldwClient } = await import("@/services/tldw/TldwApiClient")
        const config = await tldwClient.getConfig()
        if (cancelled) return
        setTarget({
          resolved: true,
          serverUrl:
            typeof config?.serverUrl === "string" ? config.serverUrl : null,
          multiUser: config?.authMode === "multi-user"
        })
      } catch {
        if (!cancelled) {
          setTarget({ resolved: true, serverUrl: null, multiUser: false })
        }
      }
    })()
    return () => {
      cancelled = true
    }
  }, [hostedMode])

  if (hostedMode) {
    return (
      <div className="min-h-screen bg-bg">
        <div className="mx-auto w-full max-w-4xl px-4 py-10 sm:px-6 lg:px-8">
          <TldwSettings />
        </div>
      </div>
    )
  }

  if (!target.resolved) {
    return <div className="min-h-screen bg-bg" />
  }

  if (target.multiUser && target.serverUrl) {
    return <SelfHostLogin serverUrl={target.serverUrl} />
  }

  // No multi-user server configured: signing in is not the next step -
  // configuring a server is.
  return (
    <RouteRedirect
      to="/settings/tldw"
      title="Connect a server first"
      description="Configure your tldw server URL and authentication mode in settings; multi-user servers then sign in here."
    />
  )
}

export default LoginPage

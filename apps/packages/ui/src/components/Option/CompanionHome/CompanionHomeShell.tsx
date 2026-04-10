import { useQuery } from "@tanstack/react-query"
import React from "react"
import { Link } from "react-router-dom"
import { browser } from "wxt/browser"

import { useIsConnected } from "@/hooks/useConnectionState"
import { useSafeDemoMode } from "@/context/demo-mode"
import { fetchChatModels } from "@/services/tldw-server"
import { CompanionHomePage } from "./CompanionHomePage"

type QuickAction = {
  href: string
  label: string
  description: string
  /** When true, opens options.html at the given hash route in a new tab. */
  external?: boolean
}

type CompanionHomeShellProps = {
  surface: "options" | "sidepanel"
  onPersonalizationEnabled?: () => void
}

export function CompanionHomeShell({
  surface,
  onPersonalizationEnabled
}: CompanionHomeShellProps) {
  const { demoEnabled, setDemoEnabled } = useSafeDemoMode()
  const demoExitPath = surface === "sidepanel" ? "/settings" : "/setup"

  const isConnected = useIsConnected()
  const { data: models = [] } = useQuery({
    queryKey: ["companion-home:chatModels"],
    queryFn: () => fetchChatModels({ returnEmpty: true }),
    enabled: isConnected && !demoEnabled,
    staleTime: 30_000,
  })
  const needsProvider = isConnected && !demoEnabled && models.length === 0

  const actions: QuickAction[] =
    surface === "sidepanel"
      ? [
          {
            href: "/?view=chat",
            label: "Open Chat",
            description: "Jump back into the sidepanel chat workspace."
          },
          {
            href: "/settings",
            label: "Open Settings",
            description: "Adjust connection and sidepanel behavior."
          },
          {
            href: "/media",
            label: "Open Media Library",
            description: "Browse and manage your ingested media collection.",
            external: true
          },
          {
            href: "/quiz",
            label: "Open Quizzes",
            description: "Review and take quizzes generated from your content.",
            external: true
          }
        ]
      : [
          {
            href: "/chat",
            label: "Open Chat",
            description: "Continue active work in the main chat workspace."
          },
          {
            href: "/knowledge",
            label: "Open Knowledge",
            description: "Review sources, notes, and captured knowledge from the main workspace."
          },
          {
            href: "/media-multi",
            label: "Open Analysis",
            description: "Review and compare media from the main workspace."
          }
        ]

  const openOptionsRoute = React.useCallback((hash: string) => {
    const url = browser.runtime.getURL(`/options.html#${hash}`)
    if (browser.tabs?.create) {
      browser.tabs.create({ url }).catch(() => {
        window.open(url, "_blank", "noopener,noreferrer")
      })
      return
    }
    window.open(url, "_blank", "noopener,noreferrer")
  }, [])

  return (
    <section
      className="mx-auto flex w-full max-w-6xl flex-col gap-6 px-6 py-8"
      data-testid="companion-home-shell"
    >
      {demoEnabled && (
        <div className="rounded-2xl border border-amber-500/30 bg-amber-500/5 px-5 py-4">
          <div className="flex items-center justify-between gap-3">
            <div>
              <p className="text-sm font-medium text-text">
                You are in demo mode
              </p>
              <p className="mt-1 text-xs text-text-muted">
                Chat and search use sample data only. Connect a server for full functionality.
              </p>
            </div>
            <Link
              to={demoExitPath}
              onClick={() => setDemoEnabled(false)}
              className="shrink-0 rounded-lg border border-border bg-bg px-3 py-1.5 text-xs font-medium text-text transition-colors hover:bg-surface-hover"
            >
              Connect a server
            </Link>
          </div>
        </div>
      )}

      {needsProvider && (
        <div className="rounded-2xl border border-primary/30 bg-primary/5 px-5 py-4">
          <div className="flex items-center justify-between gap-3">
            <div>
              <p className="text-sm font-medium text-text">
                Configure an LLM provider to start chatting
              </p>
              <p className="mt-1 text-xs text-text-muted">
                Add an API key for OpenAI, Anthropic, or another provider in your server&apos;s .env file, then restart.
              </p>
            </div>
            <Link
              to="/settings/model"
              className="shrink-0 rounded-lg bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground transition-colors hover:bg-primary/90"
            >
              Open Model Settings
            </Link>
          </div>
        </div>
      )}
      <CompanionHomePage
        onPersonalizationEnabled={onPersonalizationEnabled}
        surface={surface}
      />

      <div className="rounded-3xl border border-border/80 bg-surface/90 p-5 shadow-sm backdrop-blur-sm">
        <div>
          <h2 className="text-lg font-semibold text-text">Quick actions</h2>
          <p className="mt-1 text-sm text-text-muted">
            Jump to the main features.
          </p>
        </div>
        <div className="grid gap-3 pt-4 sm:grid-cols-2 xl:grid-cols-3" data-testid="companion-home-quick-actions">
          {actions.map((action) => {
            const testId = `companion-home-action-${action.href.replace(/\//g, "-").replace(/^-/, "")}`
            const cardClass =
              "rounded-2xl border border-border bg-bg/60 px-4 py-4 transition-colors hover:border-primary/40 hover:bg-primary/5 text-left"

            if (action.external) {
              return (
                <button
                  key={action.href}
                  type="button"
                  onClick={() => openOptionsRoute(action.href)}
                  data-testid={testId}
                  className={cardClass}
                >
                  <div className="text-sm font-semibold text-text">{action.label}</div>
                  <p className="mt-2 text-sm leading-6 text-text-muted">{action.description}</p>
                </button>
              )
            }

            return (
              <Link
                key={action.href}
                to={action.href}
                data-testid={testId}
                className={cardClass}
              >
                <div className="text-sm font-semibold text-text">{action.label}</div>
                <p className="mt-2 text-sm leading-6 text-text-muted">{action.description}</p>
              </Link>
            )
          })}
        </div>
      </div>
    </section>
  )
}

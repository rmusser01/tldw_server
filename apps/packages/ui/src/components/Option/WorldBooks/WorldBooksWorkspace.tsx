import React from "react"
import { useTranslation } from "react-i18next"
import { useNavigate } from "react-router-dom"
import FeatureEmptyState from "@/components/Common/FeatureEmptyState"
import { PageShell } from "@/components/Common/PageShell"
import WorkspaceConnectionGate from "@/components/Common/WorkspaceConnectionGate"
import { useDemoMode } from "@/context/demo-mode"
import { WorldBooksManager } from "./Manager"
import { useServerCapabilities } from "@/hooks/useServerCapabilities"
import { useLayoutUiStore } from "@/store/layout-ui"

export const WorldBooksWorkspace: React.FC = () => {
  const { t } = useTranslation(["option", "common"])
  const navigate = useNavigate()
  const { demoEnabled } = useDemoMode()
  const { capabilities, loading: capsLoading } = useServerCapabilities()
  const chatSidebarCollapsed = useLayoutUiStore(
    (state) => state.chatSidebarCollapsed
  )
  const pageShellMaxWidthClassName = "max-w-none"

  const worldBooksUnsupported =
    !capsLoading && capabilities && !capabilities.hasWorldBooks

  return (
    <WorkspaceConnectionGate
      featureName={t("option:header.modeWorldBooks", "World Books")}
      setupDescription={t("option:worldBooksEmpty.connectDescription", {
        defaultValue:
          "To use World Books, first connect to your tldw server so world knowledge can be saved and retrieved."
      })}
      maxWidthClassName={pageShellMaxWidthClassName}
      renderDemo={
        demoEnabled
          ? () => (
              <FeatureEmptyState
                title={
                  <span className="inline-flex items-center gap-2">
                    <span className="rounded-full bg-primary/10 px-2 py-0.5 text-[11px] font-medium text-primary">
                      Demo
                    </span>
                    <span>
                      {t("option:worldBooksEmpty.demoTitle", {
                        defaultValue: "Explore World Books in demo mode"
                      })}
                    </span>
                  </span>
                }
                description={t("option:worldBooksEmpty.demoDescription", {
                  defaultValue:
                    "This demo shows how World Books can organize structured knowledge about your worlds, settings, or products."
                })}
                examples={[
                  t("option:worldBooksEmpty.demoExample1", {
                    defaultValue:
                      "See example entries like a fantasy setting, product glossary, or campaign notes."
                  }),
                  t("option:worldBooksEmpty.demoExample2", {
                    defaultValue:
                      "When you connect, you’ll be able to create world books that tldw can use while chatting."
                  })
                ]}
              />
            )
          : undefined
      }
    >
      {worldBooksUnsupported ? (
        <FeatureEmptyState
          title={
            <span className="inline-flex items-center gap-2">
              <span className="rounded-full bg-warn/10 px-2 py-0.5 text-[11px] font-medium text-warn">
                Feature unavailable
              </span>
              <span>
                {t("option:worldBooksEmpty.offlineTitle", {
                  defaultValue: "World Books API not available on this server"
                })}
              </span>
            </span>
          }
          description={t("option:worldBooksEmpty.offlineDescription", {
            defaultValue:
              "This tldw server does not advertise the World Books endpoints (for example, /api/v1/characters/world-books). Upgrade your server to a version that includes World Books to use this workspace."
          })}
          examples={[
            t("option:worldBooksEmpty.offlineExample1", {
              defaultValue:
                "Open Health & diagnostics to confirm your server version and available APIs."
            }),
            t("option:worldBooksEmpty.offlineExample2", {
              defaultValue:
                "After upgrading, reload the extension and return to World Books."
            })
          ]}
          primaryActionLabel={t("settings:healthSummary.diagnostics", {
            defaultValue: "Health & diagnostics"
          })}
          onPrimaryAction={() => navigate("/settings/health")}
        />
      ) : (
        <PageShell
          className="space-y-4"
          maxWidthClassName={pageShellMaxWidthClassName}
        >
          <div className="space-y-4" data-testid="world-books-tutorial-shell">
            <div className="space-y-1">
              <h1 className="text-lg font-semibold text-text">
                {t("option:header.modeWorldBooks", "World Books")}
              </h1>
              <p className="text-xs text-text-muted">
                {t("option:worldBooksEmpty.headerDescription", {
                  defaultValue:
                    "Create and manage structured knowledge bases that characters and chats can reference."
                })}
              </p>
            </div>
            <WorldBooksManager />
          </div>
        </PageShell>
      )}
    </WorkspaceConnectionGate>
  )
}

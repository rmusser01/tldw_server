import React from "react"
import { useTranslation } from "react-i18next"
import FeatureEmptyState from "@/components/Common/FeatureEmptyState"
import { PageShell } from "@/components/Common/PageShell"
import WorkspaceConnectionGate from "@/components/Common/WorkspaceConnectionGate"
import { useDemoMode } from "@/context/demo-mode"
import { useLayoutUiStore } from "@/store/layout-ui"
import { translateMessage } from "@/i18n/translateMessage"
import { DictionariesManager } from "./Manager"

export const DictionariesWorkspace: React.FC = () => {
  const { t } = useTranslation(["option", "common"])
  const { demoEnabled } = useDemoMode()
  const chatSidebarCollapsed = useLayoutUiStore(
    (state) => state.chatSidebarCollapsed
  )
  const pageShellMaxWidthClassName = chatSidebarCollapsed
    ? "max-w-none"
    : "max-w-5xl"

  return (
    <WorkspaceConnectionGate
      featureName={translateMessage(
        t,
        "option:header.modeDictionaries",
        "Chat dictionaries"
      )}
      setupDescription={translateMessage(
        t,
        "option:dictionariesEmpty.connectDescription",
        "To use Chat dictionaries, first connect to your tldw server so substitutions can be stored."
      )}
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
                      {t("option:dictionariesEmpty.demoTitle", {
                        defaultValue: "Explore Chat dictionaries in demo mode"
                      })}
                    </span>
                  </span>
                }
                description={translateMessage(
                  t,
                  "option:dictionariesEmpty.demoDescription",
                  "This demo shows how Chat dictionaries can normalize names, acronyms, and terms before they reach the model."
                )}
                examples={[
                  translateMessage(
                    t,
                    "option:dictionariesEmpty.demoExample1",
                    "Create example dictionaries for product names, project codenames, or company jargon."
                  ),
                  translateMessage(
                    t,
                    "option:dictionariesEmpty.demoExample2",
                    "When you connect, you’ll be able to activate dictionaries across all chats."
                  )
                ]}
              />
            )
          : undefined
      }
    >
      <PageShell
        className="space-y-4"
        maxWidthClassName={pageShellMaxWidthClassName}
      >
        <div className="space-y-1">
          <h1 className="text-lg font-semibold text-text">
            {translateMessage(
              t,
              "option:header.modeDictionaries",
              "Chat dictionaries"
            )}
          </h1>
          <p className="text-xs text-text-muted">
            {translateMessage(
              t,
              "option:dictionariesEmpty.headerDescription",
              "Define reusable substitutions so tldw understands your organization’s names, acronyms, and terminology."
            )}
          </p>
          <a
            href="https://github.com/rmusser01/tldw_server/blob/main/Docs/User_Guides/WebUI_Extension/Chat_Dictionaries_Guide.md"
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs text-text-muted hover:text-text underline"
          >
            {translateMessage(
              t,
              "option:dictionariesEmpty.learnMore",
              "Learn more about dictionaries"
            )}
          </a>
        </div>
        <DictionariesManager />
      </PageShell>
    </WorkspaceConnectionGate>
  )
}

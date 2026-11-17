import React, { useEffect } from "react"
import { Tabs } from "antd"
import { DismissibleBetaAlert } from "@/components/Common/DismissibleBetaAlert"
import type { TabsProps } from "antd"
import { Table2, Plus } from "lucide-react"
import { useTranslation } from "react-i18next"
import { PageShell } from "@/components/Common/PageShell"
import WorkspaceConnectionGate from "@/components/Common/WorkspaceConnectionGate"
import { useDataTablesStore } from "@/store/data-tables"
import { watchChatAccountChanges } from "@/services/chat-account-boundary"
import { consumeDataTablesPrefill } from "@/utils/data-tables-prefill"
import { DataTablesList } from "./DataTablesList"
import { CreateTableWizard } from "./CreateTableWizard"

/**
 * DataTablesPage
 *
 * Main container for the Data Tables Studio feature.
 * Provides a tabbed interface for viewing saved tables and creating new ones.
 */
export const DataTablesPage: React.FC = () => {
  const { t } = useTranslation(["dataTables", "common"])

  const activeTab = useDataTablesStore((s) => s.activeTab)
  const setActiveTab = useDataTablesStore((s) => s.setActiveTab)
  const addSource = useDataTablesStore((s) => s.addSource)
  const setWizardStep = useDataTablesStore((s) => s.setWizardStep)
  const setGeneratedTable = useDataTablesStore((s) => s.setGeneratedTable)
  const resetStore = useDataTablesStore((s) => s.resetStore)
  const resetWizard = useDataTablesStore((s) => s.resetWizard)

  // Reset store on unmount — use ref to avoid re-firing if selector returns new reference
  const wizardAuthorityRef = React.useRef(0)
  const resetStoreRef = React.useRef(resetStore)
  resetStoreRef.current = resetStore
  useEffect(() => {
    const release = watchChatAccountChanges((invalidated) => {
      if (!invalidated) return
      wizardAuthorityRef.current += 1
      resetStoreRef.current()
    })
    return () => {
      release()
      resetStoreRef.current()
    }
  }, [])

  useEffect(() => {
    let isActive = true
    const applyPrefill = async () => {
      const authority = wizardAuthorityRef.current
      const payload = await consumeDataTablesPrefill()
      if (!payload || !isActive || authority !== wizardAuthorityRef.current) return

      resetWizard()

      if (payload.kind === "chat") {
        addSource(payload.source)
        setWizardStep("prompt")
        return
      }

      if (payload.kind === "artifact") {
        if (payload.source) {
          addSource(payload.source)
        }
        setGeneratedTable(payload.table)
        setWizardStep("preview")
      }
    }

    void applyPrefill()

    return () => {
      isActive = false
    }
  }, [addSource, resetWizard, setGeneratedTable, setWizardStep])

  const tabItems: TabsProps["items"] = [
    {
      key: "tables",
      label: (
        <span className="flex items-center gap-2">
          <Table2 className="h-4 w-4" />
          {t("dataTables:tabs.tables", "My Tables")}
        </span>
      ),
      children: <DataTablesList />
    },
    {
      key: "create",
      label: (
        <span className="flex items-center gap-2">
          <Plus className="h-4 w-4" />
          {t("dataTables:tabs.create", "Create Table")}
        </span>
      ),
      children: <CreateTableWizard />
    }
  ]

  return (
    <WorkspaceConnectionGate
      featureName={t("dataTables:title", "Data Tables")}
      setupDescription={t(
        "dataTables:setupRequired",
        "Data Tables depends on your connected tldw server to generate and save structured tables."
      )}
      maxWidthClassName="max-w-6xl"
    >
      <PageShell className="py-6" maxWidthClassName="max-w-6xl">
        <div className="mb-6">
          <h1 className="text-2xl font-semibold text-text">
            {t("dataTables:title", "Data Tables Studio")}
          </h1>
          <p className="mt-1 text-sm text-text-muted">
            {t(
              "dataTables:dataTablesDescription",
              "Generate structured tables from your chats, documents, and knowledge base using natural language prompts."
            )}
          </p>
        </div>

        <DismissibleBetaAlert
          storageKey="beta-dismissed:dataTables"
          message={t("dataTables:betaNotice", "Beta Feature")}
          description={t(
            "dataTables:betaDescription",
            "Data Tables is currently in beta. Table generation requires backend support."
          )}
          className="mb-6"
        />

        <Tabs
          activeKey={activeTab}
          onChange={(key) => setActiveTab(key as typeof activeTab)}
          items={tabItems}
          className="data-tables-tabs"
        />
      </PageShell>
    </WorkspaceConnectionGate>
  )
}

export default DataTablesPage

/**
 * EvaluationsPage
 *
 * Main container for the Evaluations module.
 * Provides a tabbed interface for managing evaluations, runs, datasets, webhooks, and history.
 */

import React, { useEffect } from "react"
import { Tabs } from "antd"
import { DismissibleBetaAlert } from "@/components/Common/DismissibleBetaAlert"
import type { TabsProps } from "antd"
import { BarChart3, ClipboardCheck, Database, FlaskConical, History, Play, Webhook } from "lucide-react"
import { useTranslation } from "react-i18next"
import { useNavigate, useSearchParams } from "react-router-dom"
import { PageShell } from "@/components/Common/PageShell"
import WorkspaceConnectionGate from "@/components/Common/WorkspaceConnectionGate"
import { Alert } from "@/components/ui/primitives/Alert"
import { useEvaluationsStore, type EvaluationsTab as EvaluationsTabType } from "@/store/evaluations"
import { DatasetsTab } from "./tabs/DatasetsTab"
import { EvaluationsTab } from "./tabs/EvaluationsTab"
import { HistoryTab } from "./tabs/HistoryTab"
import { RecipesTab } from "./tabs/RecipesTab"
import { RunsTab } from "./tabs/RunsTab"
import { SyntheticReviewTab } from "./tabs/SyntheticReviewTab"
import { WebhooksTab } from "./tabs/WebhooksTab"

export const EvaluationsPage: React.FC = () => {
  const { t } = useTranslation(["evaluations", "common"])
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const tourActive = searchParams.get("tour") === "1"

  const activeTab = useEvaluationsStore((s) => s.activeTab)
  const setActiveTab = useEvaluationsStore((s) => s.setActiveTab)
  const setSelectedEvalId = useEvaluationsStore((s) => s.setSelectedEvalId)
  const setSelectedRunId = useEvaluationsStore((s) => s.setSelectedRunId)
  const resetStore = useEvaluationsStore((s) => s.resetStore)

  // Sync URL params to store on mount
  useEffect(() => {
    const tabFromQuery = searchParams.get("tab") as EvaluationsTabType | null
    const evalIdFromQuery = searchParams.get("evaluationId")
    const runIdFromQuery = searchParams.get("runId")

    if (
      tabFromQuery &&
      ["recipes", "synthetic-review", "evaluations", "runs", "datasets", "webhooks", "history"].includes(
        tabFromQuery
      )
    ) {
      setActiveTab(tabFromQuery)
    }
    if (evalIdFromQuery) {
      setSelectedEvalId(evalIdFromQuery)
    }
    if (runIdFromQuery) {
      setSelectedRunId(runIdFromQuery)
    }
  }, [searchParams, setActiveTab, setSelectedEvalId, setSelectedRunId])

  // Reset store on unmount — use ref to avoid re-firing if selector returns new reference
  const resetStoreRef = React.useRef(resetStore)
  resetStoreRef.current = resetStore
  useEffect(() => {
    return () => {
      resetStoreRef.current()
    }
  }, [])

  useEffect(() => {
    if (typeof document === "undefined") return
    const root = document.documentElement
    if (tourActive) {
      root.dataset.evaluationsTour = "on"
    } else {
      delete root.dataset.evaluationsTour
    }
    return () => {
      delete root.dataset.evaluationsTour
    }
  }, [tourActive])

  // Sync store to URL params
  const handleTabChange = (key: string) => {
    const tab = key as EvaluationsTabType
    setActiveTab(tab)

    const params = new URLSearchParams(searchParams)
    params.set("tab", tab)
    navigate(`?${params.toString()}`, { replace: true })
  }

  const tabItems: TabsProps["items"] = [
    {
      key: "recipes",
      label: (
        <span
          className="flex items-center gap-2"
          data-testid="evaluations-tab-recipes"
        >
          <FlaskConical className="h-4 w-4" />
          {t("evaluations:tabRecipes", "Recipes")}
        </span>
      ),
      children: <RecipesTab />
    },
    {
      key: "synthetic-review",
      label: (
        <span
          className="flex items-center gap-2"
          data-testid="evaluations-tab-synthetic-review"
        >
          <ClipboardCheck className="h-4 w-4" />
          {t("evaluations:tabSyntheticReview", "Review")}
        </span>
      ),
      children: <SyntheticReviewTab />
    },
    {
      key: "evaluations",
      label: (
        <span
          className="flex items-center gap-2"
          data-testid="evaluations-tab-evaluations"
        >
          <BarChart3 className="h-4 w-4" />
          {t("evaluations:tabEvaluations", "Evaluations")}
        </span>
      ),
      children: <EvaluationsTab />
    },
    {
      key: "runs",
      label: (
        <span className="flex items-center gap-2" data-testid="evaluations-tab-runs">
          <Play className="h-4 w-4" />
          {t("evaluations:tabRuns", "Runs")}
        </span>
      ),
      children: <RunsTab />
    },
    {
      key: "datasets",
      label: (
        <span
          className="flex items-center gap-2"
          data-testid="evaluations-tab-datasets"
        >
          <Database className="h-4 w-4" />
          {t("evaluations:tabDatasets", "Datasets")}
        </span>
      ),
      children: <DatasetsTab />
    },
    {
      key: "webhooks",
      label: (
        <span
          className="flex items-center gap-2"
          data-testid="evaluations-tab-webhooks"
        >
          <Webhook className="h-4 w-4" />
          {t("evaluations:tabWebhooks", "Webhooks")}
        </span>
      ),
      children: <WebhooksTab />
    },
    {
      key: "history",
      label: (
        <span
          className="flex items-center gap-2"
          data-testid="evaluations-tab-history"
        >
          <History className="h-4 w-4" />
          {t("evaluations:tabHistory", "History")}
        </span>
      ),
      children: <HistoryTab />
    }
  ]

  return (
    <WorkspaceConnectionGate
      featureName={t("evaluations:title", "Evaluations")}
      setupDescription={t(
        "evaluations:setupRequired",
        "Evaluations depends on your connected tldw server to create runs, inspect datasets, and review metrics."
      )}
      maxWidthClassName="max-w-6xl"
    >
      <PageShell className="py-6" maxWidthClassName="max-w-6xl">
      <div className="mb-6">
        <h1 className="text-2xl font-semibold text-text" data-testid="evaluations-page-title">
          {t("evaluations:title", "Evaluations")}
        </h1>
        <p className="mt-1 text-sm text-text-muted">
          {t(
            "evaluations:subtitle",
            "Define evaluations against your tldw server and inspect recent runs."
          )}
        </p>
      </div>

      <DismissibleBetaAlert
        storageKey="beta-dismissed:evaluations"
        message={t("evaluations:betaNotice", "Beta Feature")}
        description={t(
          "evaluations:betaDescription",
          "Evaluations is currently in beta. Some features may be incomplete or change."
        )}
        className="mb-6"
      />

      {tourActive && (
        <Alert
          variant="info"
          title={t("evaluations:tourTitle", "Evaluations tour")}
          className="mb-6"
        >
          {t(
            "evaluations:tourDescription",
            "Tour mode highlights key actions. Remove ?tour=1 from the URL to exit."
          )}
        </Alert>
      )}

      {tourActive && (
        <style>{`
          [data-evaluations-tour="on"] [data-eval-tour] {
            outline: 2px dashed rgba(59, 130, 246, 0.8);
            outline-offset: 4px;
            box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.12);
            border-radius: 8px;
          }
        `}</style>
      )}

      <Tabs
        activeKey={activeTab}
        onChange={handleTabChange}
        items={tabItems}
        className="evaluations-tabs"
        data-testid="evaluations-tabs"
      />
      </PageShell>
    </WorkspaceConnectionGate>
  )
}

export default EvaluationsPage

import React from "react"
import { Button, Empty, Spin, Tag, Typography } from "antd"
import { useNavigate } from "react-router-dom"
import { useTranslation } from "react-i18next"

import { PageShell } from "@/components/Common/PageShell"
import { RecoveryCallout, buildCapabilityState } from "@/components/ui/state"
import { useIngestionSourcesQuery } from "@/hooks/use-ingestion-sources"
import { useServerCapabilities } from "@/hooks/useServerCapabilities"
import { SourcesAvailabilityGate } from "./SourcesAvailabilityGate"
import { SourceListTable } from "./SourceListTable"

type SourcesWorkspacePageProps = {
  mode?: "user" | "admin"
}

export const SourcesWorkspacePage: React.FC<SourcesWorkspacePageProps> = ({
  mode = "user"
}) => {
  const { t } = useTranslation(["sources", "common", "option"])
  const navigate = useNavigate()
  const capabilityState = useServerCapabilities()
  const sourcesQuery = useIngestionSourcesQuery(undefined, {
    enabled:
      !capabilityState.loading &&
      capabilityState.capabilities?.hasIngestionSources !== false
  })
  const loadErrorState = sourcesQuery.error
    ? buildCapabilityState({
        featureName: t("sources:title", "Sources"),
        capabilityName: "ingestion source management",
        endpoint: "/api/v1/ingestion-sources",
        method: "GET",
        error: sourcesQuery.error
      })
    : null

  return (
    <SourcesAvailabilityGate capabilityState={capabilityState}>
      <PageShell className="space-y-6 py-6" maxWidthClassName="max-w-6xl">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Typography.Title level={2} className="!mb-0">
                {t("sources:title", "Sources")}
              </Typography.Title>
              {mode === "admin" && <Tag color="gold">Admin view</Tag>}
            </div>
            <Typography.Paragraph type="secondary" className="!mb-0">
              {t(
                "sources:description",
                "Manage local folders and archive snapshots that sync into notes or media."
              )}
            </Typography.Paragraph>
          </div>
          <Button
            type="primary"
            onClick={() => {
              navigate("/sources/new")
            }}>
            {t("sources:actions.new", "New source")}
          </Button>
        </div>

        {sourcesQuery.isLoading ? (
          <div className="flex justify-center py-10">
            <Spin />
          </div>
        ) : null}

        {!sourcesQuery.isLoading && sourcesQuery.error ? (
          <RecoveryCallout
            state={loadErrorState?.state ?? "error"}
            title={loadErrorState?.title ?? "Unable to load sources"}
            message={
              loadErrorState?.message ??
              "The ingestion source overview could not be loaded."
            }
            diagnostics={loadErrorState?.diagnostics}
            primaryAction={{
              label: t("common:actions.retry", "Try again"),
              onClick: () => {
                void sourcesQuery.refetch()
              }
            }}
            secondaryActions={[
              {
                label: t("option:healthDiagnostics", "Health & diagnostics"),
                onClick: () => navigate("/settings/health")
              }
            ]}
          />
        ) : null}

        {!sourcesQuery.isLoading &&
        !sourcesQuery.error &&
        (sourcesQuery.data?.total ?? 0) === 0 ? (
          <Empty description={t("sources:states.empty", "No ingestion sources yet.")} />
        ) : null}

        {!sourcesQuery.isLoading &&
        !sourcesQuery.error &&
        (sourcesQuery.data?.sources?.length ?? 0) > 0 ? (
          <SourceListTable sources={sourcesQuery.data?.sources ?? []} />
        ) : null}
      </PageShell>
    </SourcesAvailabilityGate>
  )
}

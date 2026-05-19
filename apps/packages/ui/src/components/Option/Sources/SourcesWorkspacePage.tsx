import React from "react"
import { Button, Spin, Tag, Typography } from "antd"
import { useNavigate } from "react-router-dom"
import { useTranslation } from "react-i18next"

import { PageShell } from "@/components/Common/PageShell"
import {
  StatePanel,
  buildCapabilityState,
  classifyCapabilityError,
  messageFromError,
  statusFromError
} from "@/components/ui/state"
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
  const sourceFeatureName = t("sources:title", "Sources")
  const sourceCapabilityName = t(
    "sources:capability.ingestionSources",
    "ingestion sources"
  )
  const queryError = sourcesQuery.error
  const queryErrorState = queryError
    ? buildCapabilityState({
        kind: classifyCapabilityError(queryError),
        featureName: sourceFeatureName,
        capabilityName: sourceCapabilityName,
        method: "GET",
        endpoint: "/api/v1/ingestion-sources",
        status: statusFromError(queryError),
        rawMessage: messageFromError(queryError) || "Failed to load sources",
        primaryAction: {
          label: t("common:actions.retry", "Try again"),
          onClick: () => {
            void sourcesQuery.refetch?.()
          }
        }
      })
    : null
  const emptyState = buildCapabilityState({
    kind: "empty",
    featureName: sourceFeatureName,
    primaryAction: {
      label: t("sources:actions.create", "Create source"),
      onClick: () => {
        navigate("/sources/new")
      }
    }
  })

  return (
    <SourcesAvailabilityGate capabilityState={capabilityState}>
      <PageShell className="space-y-6 py-6" maxWidthClassName="max-w-6xl">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Typography.Title level={1} className="!mb-0 !text-2xl">
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
          <div
            className="flex justify-center py-10"
            data-testid="sources-loading-state"
            role="status"
            aria-label={t("sources:states.loading", "Loading sources")}
          >
            <Spin />
          </div>
        ) : null}

        {!sourcesQuery.isLoading && queryErrorState ? (
          <StatePanel
            state={queryErrorState.state}
            title={queryErrorState.title}
            message={queryErrorState.message}
            diagnostics={queryErrorState.diagnostics}
            primaryAction={queryErrorState.primaryAction}
            role="alert"
          />
        ) : null}

        {!sourcesQuery.isLoading &&
        !sourcesQuery.error &&
        (sourcesQuery.data?.total ?? 0) === 0 ? (
          <StatePanel
            state={emptyState.state}
            title={emptyState.title}
            message={emptyState.message}
            primaryAction={emptyState.primaryAction}
          />
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

import React from "react"
import { Button, Empty, Space, Spin, Tag, Typography } from "antd"
import { useTranslation } from "react-i18next"

import { PageShell } from "@/components/Common/PageShell"
import { Alert } from "@/components/ui/primitives"
import {
  useIngestionSourceDetailQuery,
  useIngestionSourceItemsQuery,
  useReattachIngestionSourceItemMutation,
  useSyncIngestionSourceMutation,
  useUploadIngestionSourceArchiveMutation
} from "@/hooks/use-ingestion-sources"
import { SourceForm } from "./SourceForm"
import { SourceItemsTable, type SourceItemsFilter } from "./SourceItemsTable"
import { SourceStatusPanels } from "./SourceStatusPanels"

type SourceDetailPageProps = {
  sourceId: string
  mode?: "user" | "admin"
}

const matchesFilter = (syncStatus: string, filter: SourceItemsFilter): boolean => {
  if (filter === "detached") {
    return syncStatus === "conflict_detached"
  }
  if (filter === "degraded") {
    return syncStatus.includes("degraded")
  }
  return true
}

export const SourceDetailPage: React.FC<SourceDetailPageProps> = ({
  sourceId,
  mode = "user"
}) => {
  const { t } = useTranslation(["sources", "common"])
  const [itemsFilter, setItemsFilter] = React.useState<SourceItemsFilter>("all")
  const fileInputRef = React.useRef<HTMLInputElement | null>(null)

  const detailQuery = useIngestionSourceDetailQuery(sourceId)
  const itemsQuery = useIngestionSourceItemsQuery(sourceId)
  const syncMutation = useSyncIngestionSourceMutation()
  const uploadMutation = useUploadIngestionSourceArchiveMutation(sourceId)
  const reattachMutation = useReattachIngestionSourceItemMutation(sourceId)

  const detail = detailQuery.data
  const items = React.useMemo(() => {
    const allItems = itemsQuery.data?.items ?? []
    return allItems.filter((item) => matchesFilter(item.sync_status, itemsFilter))
  }, [itemsFilter, itemsQuery.data?.items])

  if (detailQuery.isLoading) {
    return (
      <PageShell className="py-6" maxWidthClassName="max-w-6xl">
        <div className="flex justify-center py-10">
          <Spin />
        </div>
      </PageShell>
    )
  }

  if (!detail) {
    return (
      <PageShell className="py-6" maxWidthClassName="max-w-6xl">
        <Empty description="Source not found." />
      </PageShell>
    )
  }

  return (
    <PageShell className="space-y-6 py-6" maxWidthClassName="max-w-6xl">
      <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Typography.Title level={2} className="!mb-0">
              {typeof detail.config?.label === "string" && detail.config.label.length > 0
                ? detail.config.label
                : `${detail.source_type} ${detail.id}`}
            </Typography.Title>
            {mode === "admin" ? <Tag color="gold">Admin view</Tag> : null}
          </div>
          <Typography.Text type="secondary">
            {`${detail.source_type} -> ${detail.sink_type} (${detail.policy})`}
          </Typography.Text>
          <SourceStatusPanels source={detail} />
        </div>

        <Space wrap>
          <Button
            type="primary"
            onClick={() => {
              void syncMutation.mutateAsync(sourceId)
            }}
            loading={Boolean((syncMutation as { isPending?: boolean }).isPending)}>
            {t("sources:actions.sync", "Sync now")}
          </Button>
          {detail.source_type === "archive_snapshot" ? (
            <>
              <Button onClick={() => fileInputRef.current?.click()}>
                {t("sources:actions.uploadArchive", "Upload archive")}
              </Button>
              <input
                ref={fileInputRef}
                type="file"
                className="hidden"
                onChange={(event) => {
                  const file = event.target.files?.[0]
                  if (file) {
                    void uploadMutation.mutateAsync(file)
                    event.currentTarget.value = ""
                  }
                }}
              />
            </>
          ) : null}
        </Space>
      </div>

      {detail.last_error ? (
        <Alert variant="error" title={detail.last_error} />
      ) : null}

      {detail.last_successful_snapshot_id || detail.last_successful_sync_summary ? (
        <Alert
          variant="info"
          title="Source identity is locked after the first successful sync."
        />
      ) : null}

      <div className="rounded-2xl border border-border/70 bg-surface/80 p-4">
        <SourceForm mode="edit" source={detail} />
      </div>

      <div className="space-y-3">
        <Typography.Title level={4} className="!mb-0">
          Tracked items
        </Typography.Title>
        <SourceItemsTable
          items={items}
          filter={itemsFilter}
          onFilterChange={setItemsFilter}
          onReattach={(itemId) => {
            void reattachMutation.mutateAsync(itemId)
          }}
          isReattaching={Boolean((reattachMutation as { isPending?: boolean }).isPending)}
        />
      </div>
    </PageShell>
  )
}

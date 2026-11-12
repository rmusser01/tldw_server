import React from "react"
import { useQuery } from "@tanstack/react-query"
import { Alert, Descriptions, Drawer, Empty, Skeleton, Table, Tag } from "antd"
import { useTranslation } from "react-i18next"
import { Button } from "@/components/Common/Button"
import { useCanonicalConnectionConfig } from "@/hooks/useCanonicalConnectionConfig"
import { loadCallerCapabilities } from "@/services/caller-capabilities"
import { buildChatSurfaceScopeKeyFromConfig, derivePromptAssistAuthorizationRevision } from "@/services/chat-surface-scope"
import { getTestRun, listTestRuns, type SavedTestRun, type TestRunReadScope } from "@/services/prompt-studio"

type Props = { projectId: number; onClose: () => void }

const isSavedRun = (value: unknown, projectId: number): value is SavedTestRun => {
  if (!value || typeof value !== "object") return false
  const row = value as SavedTestRun
  return [row.id, row.prompt_id, row.test_case_id].every(id => Number.isSafeInteger(id) && id > 0) &&
    row.project_id === projectId && typeof row.model_name === "string"
}

const Snapshot = ({ value }: { value: unknown }) => value === null || value === undefined
  ? <span>Not recorded</span>
  : <pre className="whitespace-pre-wrap break-words rounded bg-surface2 p-2 text-xs">{JSON.stringify(value, null, 2)}</pre>

const HistoryContent = ({ projectId, scope }: { projectId: number; scope: TestRunReadScope }) => {
  const { t } = useTranslation(["settings", "common"])
  const [page, setPage] = React.useState(1)
  const [selectedId, setSelectedId] = React.useState<number | null>(null)
  const [showPrompt, setShowPrompt] = React.useState(false)
  const list = useQuery({
    queryKey: ["prompt-studio", "saved-test-runs", scope.scopeKey, projectId, page],
    queryFn: async ({ signal }) => {
      const response = await listTestRuns(projectId, page, scope, signal)
      const data = response.data
      if (!response.ok || !data?.success || !Array.isArray(data.data) ||
        !data.data.every(row => isSavedRun(row, projectId)) ||
        !Number.isSafeInteger(data.metadata?.total) || data.metadata!.total < 0) {
        throw new Error("Unable to load saved test runs.")
      }
      return { rows: data.data, total: data.metadata!.total }
    },
    retry: false, staleTime: 0, gcTime: 0
  })
  const detail = useQuery({
    queryKey: ["prompt-studio", "saved-test-run", scope.scopeKey, projectId, selectedId],
    queryFn: async ({ signal }) => {
      const response = await getTestRun(selectedId!, scope, signal)
      const row = response.data?.data
      if (!response.ok || !response.data?.success || !isSavedRun(row, projectId) || row.id !== selectedId)
        throw new Error("Unable to load saved test run.")
      return row
    },
    enabled: selectedId !== null, retry: false, staleTime: 0, gcTime: 0
  })
  const version = useQuery({
    queryKey: ["prompt-studio", "saved-test-run-version", scope.scopeKey, projectId, selectedId],
    queryFn: async ({ signal }) => {
      const response = await getTestRun(selectedId!, scope, signal)
      const row = response.data?.data
      if (!response.ok || !response.data?.success || !isSavedRun(row, projectId) ||
        row.id !== selectedId || row.prompt_id !== detail.data?.prompt_id ||
        row.prompt_version_number !== detail.data?.prompt_version_number)
        throw new Error("Unable to load original prompt version.")
      return row
    },
    enabled: showPrompt && !!detail.data, retry: false, staleTime: 0, gcTime: 0
  })
  const row = detail.data
  return <div className="space-y-4">
    {list.isPending ? <Skeleton active /> : list.isError
      ? <Alert type="error" showIcon title={t("managePrompts.studio.history.listFailed", { defaultValue: "Unable to load saved test runs." })} />
      : <>
        <Table<SavedTestRun> rowKey="id" dataSource={list.data?.rows} size="small" pagination={false}
          locale={{ emptyText: <Empty description="No saved test runs" /> }} columns={[
            { title: "Run", render: (_, record) => `#${record.id}` },
            { title: "Prompt version", render: (_, record) => `${record.prompt_name ?? "Prompt"} #${record.prompt_id} (${record.prompt_version_number == null ? "version not recorded" : `v${record.prompt_version_number}`})` },
            { title: "Recorded", render: (_, record) => record.created_at ?? "Not recorded" },
            { title: "Result", render: (_, record) => <Tag>{record.error_message ? "Failed" : "Saved"}</Tag> },
            { title: "", render: (_, record) => <Button ariaLabel={`Open run #${record.id}`} onClick={() => {
              setSelectedId(record.id); setShowPrompt(false)
            }}>Open</Button> }
          ]} />
        <div className="flex items-center gap-3">
          <Button disabled={page === 1} onClick={() => { setPage(page - 1); setSelectedId(null); setShowPrompt(false) }}>Previous page</Button>
          <span>Page {page} · {list.data?.total ?? 0} saved runs</span>
          <Button disabled={page * 20 >= (list.data?.total ?? 0)} onClick={() => { setPage(page + 1); setSelectedId(null); setShowPrompt(false) }}>Next page</Button>
        </div>
      </>}
    {selectedId !== null && (detail.isPending ? <Skeleton active /> : detail.isError
      ? <Alert type="error" showIcon title="Unable to load saved test run." />
      : row && <section aria-label={`Saved run #${row.id}`} className="space-y-3">
        <h3>Saved run #{row.id}</h3>
        {row.error_message && <Alert type="error" showIcon title={row.error_message} />}
        <Descriptions column={1} items={[
          { key: "prompt", label: "Original prompt version", children: `#${row.prompt_id} (${row.prompt_version_number == null ? "version not recorded" : `v${row.prompt_version_number}`})` },
          { key: "model", label: "Model", children: row.model_name || "Not recorded" },
          { key: "params", label: "Provider and parameters", children: <Snapshot value={row.model_params} /> },
          { key: "inputs", label: "Saved inputs", children: <Snapshot value={row.inputs} /> },
          { key: "expected", label: "Saved expected output", children: <Snapshot value={row.expected_outputs} /> },
          { key: "outputs", label: "Saved actual output", children: <Snapshot value={row.outputs} /> },
          { key: "time", label: "Recorded", children: row.created_at ?? "Not recorded" },
          { key: "duration", label: "Execution time", children: row.execution_time_ms == null ? "Not recorded" : `${row.execution_time_ms} ms` },
          { key: "tokens", label: "Tokens used", children: row.tokens_used ?? "Not recorded" },
          { key: "cost", label: "Cost estimate", children: row.cost_estimate ?? "Not recorded" },
          { key: "scores", label: "Recorded scores", children: <Snapshot value={row.scores} /> }
        ]} />
        <Button onClick={() => setShowPrompt(true)}>Open prompt version #{row.prompt_id}</Button>
        {showPrompt && (version.isPending ? <Skeleton active /> : version.isError
          ? <Alert type="error" showIcon title="Unable to load original prompt version." />
          : <section aria-label="Original prompt template">
            <h4>Original prompt template</h4>
            {version.data?.prompt_version_number == null ? <Alert type="warning" title="Original prompt version is unavailable." /> : <>
              {version.data.prompt_deleted && <Alert type="warning" title="Original prompt version is deleted; showing its saved template." />}
              <Snapshot value={version.data.prompt_system_prompt} />
              <Snapshot value={version.data.prompt_user_prompt} />
              {version.data.prompt_definition && <Snapshot value={version.data.prompt_definition} />}
            </>}
          </section>)}
      </section>)}
  </div>
}

export const TestRunHistoryDrawer = ({ projectId, onClose }: Props) => {
  const { config, authorityLoading } = useCanonicalConnectionConfig()
  const [revision, setRevision] = React.useState(0)
  const connection = `${buildChatSurfaceScopeKeyFromConfig(config)}:${config?.authSource ?? "manual"}:${derivePromptAssistAuthorizationRevision(config)}`
  const authority = useQuery({
    queryKey: ["prompt-studio", "test-run-history-owner", connection, projectId, revision],
    queryFn: ({ signal }) => loadCallerCapabilities(signal),
    enabled: !!config && !authorityLoading, retry: false, staleTime: 0, gcTime: 0
  })
  const scope = authority.data && { userId: authority.data.capabilities.user_id, scopeKey: authority.data.scopeKey }
  return <Drawer open title="Test run history" onClose={onClose} size="large" destroyOnHidden
    extra={<Button disabled={authorityLoading || authority.isFetching} onClick={() => setRevision(value => value + 1)}>Refresh</Button>}>
    {authorityLoading || authority.isPending ? <Skeleton active /> : authority.isError
      ? <Alert type="error" showIcon title="Unable to resolve history owner." />
      : scope && <HistoryContent key={`${scope.scopeKey}:${projectId}:${revision}`} projectId={projectId} scope={scope} />}
  </Drawer>
}

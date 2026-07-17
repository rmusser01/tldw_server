import React from "react"
import { useQuery } from "@tanstack/react-query"
import { Alert, Button, Drawer, Spin, Tag } from "antd"
import { Copy, MessageSquare, Pen, Play, Plus } from "lucide-react"
import { useTranslation } from "react-i18next"
import { tldwClient } from "@/services/tldw/TldwApiClient"

interface SkillDetailsDrawerProps {
  scopeKey: string | null
  skillName: string | null
  onClose: () => void
  onTest: (skillName: string) => void
  onEdit: (skillName: string) => void
  onUseInChat: (skillName: string) => void
  onCopyInvocation: (skillName: string) => void
  onDuplicate: (skillName: string) => void
}

const formatTimestamp = (value: string): string => {
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString()
}

export const SkillDetailsDrawer: React.FC<SkillDetailsDrawerProps> = ({
  scopeKey,
  skillName,
  onClose,
  onTest,
  onEdit,
  onUseInChat,
  onCopyInvocation,
  onDuplicate
}) => {
  const { t } = useTranslation(["option", "common"])
  const detailsQuery = useQuery({
    queryKey: ["skill-details", scopeKey, skillName],
    queryFn: ({ signal }) => tldwClient.getSkill(skillName!, { signal }),
    enabled: Boolean(scopeKey && skillName)
  })

  const actionName = detailsQuery.data?.name ?? skillName
  const actionButtons = actionName ? (
    <div className="flex flex-wrap gap-2">
      <Button
        type="primary"
        className="min-h-11"
        icon={<MessageSquare size={14} />}
        onClick={() => onUseInChat(actionName)}
      >
        {t("option:skills.useInChat", { defaultValue: "Use in chat" })}
      </Button>
      <Button
        className="min-h-11"
        icon={<Copy size={14} />}
        onClick={() => onCopyInvocation(actionName)}
      >
        {t("option:skills.copyInvocationAction", { defaultValue: "Copy invocation" })}
      </Button>
      <Button className="min-h-11" icon={<Play size={14} />} onClick={() => onTest(actionName)}>
        {t("option:skills.testRun", { defaultValue: "Test run" })}
      </Button>
      <Button className="min-h-11" icon={<Pen size={14} />} onClick={() => onEdit(actionName)}>
        {t("common:edit", { defaultValue: "Edit" })}
      </Button>
      <Button className="min-h-11" icon={<Plus size={14} />} onClick={() => onDuplicate(actionName)}>
        {t("option:skills.duplicate", { defaultValue: "Duplicate" })}
      </Button>
    </div>
  ) : null

  return (
    <Drawer
      title={t("option:skills.detailsTitle", {
        defaultValue: "Skill details: {{name}}",
        name: skillName ?? ""
      })}
      open={Boolean(skillName)}
      onClose={onClose}
      size={640}
      styles={{ wrapper: { maxWidth: "100vw" } }}
      destroyOnHidden
    >
      {detailsQuery.isLoading && (
        <div className="flex min-h-40 items-center justify-center" role="status">
          <Spin />
          <span className="sr-only">
            {t("option:skills.loadingDetails", { defaultValue: "Loading skill details" })}
          </span>
        </div>
      )}

      {detailsQuery.isError && (
        <Alert
          type="error"
          showIcon
          title={t("option:skills.detailsLoadError", {
            defaultValue: "Failed to load skill details"
          })}
          action={(
            <Button
              size="small"
              className="min-h-11"
              onClick={() => void detailsQuery.refetch()}
            >
              {t("common:tryAgain", { defaultValue: "Try again" })}
            </Button>
          )}
        />
      )}

      {detailsQuery.data && (
        <div className="flex flex-col gap-6">
          <section aria-labelledby="skill-details-overview">
            <h2 id="skill-details-overview" className="m-0 text-base font-semibold text-text">
              {detailsQuery.data.name}
            </h2>
            <p className="mb-0 mt-1 text-sm text-text-muted">
              {detailsQuery.data.description
                || t("option:skills.noDescriptionProvided", {
                  defaultValue: "No description provided."
                })}
            </p>
          </section>

          <dl className="grid grid-cols-1 gap-4 border-y border-border py-4 text-sm sm:grid-cols-2">
            <div>
              <dt className="font-medium text-text-muted">
                {t("option:skills.argumentHint", { defaultValue: "Argument hint" })}
              </dt>
              <dd className="m-0 mt-1 font-mono text-text">
                {detailsQuery.data.argument_hint || "-"}
              </dd>
            </div>
            <div>
              <dt className="font-medium text-text-muted">
                {t("option:skills.mode", { defaultValue: "Mode" })}
              </dt>
              <dd className="m-0 mt-1"><Tag>{detailsQuery.data.context}</Tag></dd>
            </div>
            <div>
              <dt className="font-medium text-text-muted">
                {t("option:skills.visibility", { defaultValue: "Visibility" })}
              </dt>
              <dd className="m-0 mt-1">
                {detailsQuery.data.user_invocable
                  ? t("option:skills.visibleInChatState", { defaultValue: "Visible in chat" })
                  : t("option:skills.hiddenFromChatState", { defaultValue: "Hidden from chat" })}
              </dd>
            </div>
            <div>
              <dt className="font-medium text-text-muted">
                {t("option:skills.model", { defaultValue: "Model" })}
              </dt>
              <dd className="m-0 mt-1 font-mono text-text">
                {detailsQuery.data.model || t("option:skills.defaultModel", { defaultValue: "Default" })}
              </dd>
            </div>
            <div>
              <dt className="font-medium text-text-muted">
                {t("option:skills.versionLabel", { defaultValue: "Version" })}
              </dt>
              <dd className="m-0 mt-1 font-mono text-text">
                {detailsQuery.data.version}
              </dd>
            </div>
            <div>
              <dt className="font-medium text-text-muted">
                {t("option:skills.createdAt", { defaultValue: "Created" })}
              </dt>
              <dd className="m-0 mt-1 text-text">
                <time dateTime={detailsQuery.data.created_at}>
                  {formatTimestamp(detailsQuery.data.created_at)}
                </time>
              </dd>
            </div>
            <div>
              <dt className="font-medium text-text-muted">
                {t("option:skills.lastUpdated", { defaultValue: "Last updated" })}
              </dt>
              <dd className="m-0 mt-1 text-text">
                <time dateTime={detailsQuery.data.last_modified}>
                  {formatTimestamp(detailsQuery.data.last_modified)}
                </time>
              </dd>
            </div>
          </dl>

          {detailsQuery.data.runtime && (
            <section aria-labelledby="skill-details-runtime">
              <h3 id="skill-details-runtime" className="m-0 text-sm font-semibold text-text">
                {t("option:skills.runtimeImpactTitle", { defaultValue: "Runtime impact" })}
              </h3>
              <div className="mt-2 flex flex-wrap gap-1">
                <Tag color={detailsQuery.data.runtime.execution_mode === "fork" ? "blue" : "green"}>
                  {detailsQuery.data.runtime.execution_mode === "fork"
                    ? t("option:skills.runtimeFork", { defaultValue: "Fork" })
                    : t("option:skills.runtimeInline", { defaultValue: "Inline" })}
                </Tag>
                <Tag color={detailsQuery.data.runtime.test_run_may_call_model ? "orange" : "default"}>
                  {detailsQuery.data.runtime.test_run_may_call_model
                    ? t("option:skills.runtimeMayCallModel", { defaultValue: "Test may call model" })
                    : t("option:skills.runtimePromptOnly", { defaultValue: "Prompt only by default" })}
                </Tag>
                {detailsQuery.data.runtime.declares_tools && (
                  <Tag color="geekblue">
                    {t("option:skills.runtimeDeclaredTools", {
                      defaultValue: `${detailsQuery.data.runtime.declared_tool_count} tools declared`,
                      count: detailsQuery.data.runtime.declared_tool_count
                    })}
                  </Tag>
                )}
                {detailsQuery.data.runtime.model_override && (
                  <Tag>
                    {t("option:skills.runtimeModelOverride", {
                      defaultValue: "Model override"
                    })}: {detailsQuery.data.runtime.model_override}
                  </Tag>
                )}
                {!detailsQuery.data.runtime.auto_invocation_enabled && (
                  <Tag color="warning">
                    {t("option:skills.runtimeAutoOff", {
                      defaultValue: "Auto invocation off"
                    })}
                  </Tag>
                )}
              </div>
            </section>
          )}

          <section aria-labelledby="skill-details-instructions">
            <h3 id="skill-details-instructions" className="m-0 text-sm font-semibold text-text">
              {t("option:skills.instructionsLabel", { defaultValue: "Instructions" })}
            </h3>
            <pre className="mt-2 max-h-80 overflow-auto whitespace-pre-wrap break-words rounded-md border border-border bg-surface p-3 text-sm text-text">
              {detailsQuery.data.content}
            </pre>
          </section>

          <section aria-labelledby="skill-details-tools">
            <h3 id="skill-details-tools" className="m-0 text-sm font-semibold text-text">
              {t("option:skills.declaredTools", { defaultValue: "Declared tools" })}
            </h3>
            <div className="mt-2 flex flex-wrap gap-1">
              {detailsQuery.data.allowed_tools?.length
                ? detailsQuery.data.allowed_tools.map((tool) => <Tag key={tool}>{tool}</Tag>)
                : <span className="text-sm text-text-muted">{t("common:none", { defaultValue: "None" })}</span>}
            </div>
          </section>

          <section aria-labelledby="skill-details-files">
            <h3 id="skill-details-files" className="m-0 text-sm font-semibold text-text">
              {t("option:skills.supportingFiles", { defaultValue: "Supporting files" })}
            </h3>
            {Object.keys(detailsQuery.data.supporting_files ?? {}).length ? (
              <ul className="mb-0 mt-2 pl-5 font-mono text-sm text-text">
                {Object.keys(detailsQuery.data.supporting_files ?? {}).map((filename) => (
                  <li key={filename}>{filename}</li>
                ))}
              </ul>
            ) : (
              <p className="mb-0 mt-2 text-sm text-text-muted">
                {t("option:skills.noSupportingFiles", { defaultValue: "No supporting files." })}
              </p>
            )}
          </section>

          <div className="border-t border-border pt-4">{actionButtons}</div>
        </div>
      )}
    </Drawer>
  )
}

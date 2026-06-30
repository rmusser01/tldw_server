import React from "react"
import { useMutation } from "@tanstack/react-query"
import { Alert, Button, Input, Modal, Tag } from "antd"
import { useTranslation } from "react-i18next"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import type { SkillExecutionResult, SkillRuntimeMetadata } from "@/types/skill"

interface SkillPreviewProps {
  skillName: string | null
  runtime?: SkillRuntimeMetadata | null
  onClose: () => void
}

type SkillRunMode = "dry-run" | "test-run"

export const SkillPreview: React.FC<SkillPreviewProps> = ({
  skillName,
  runtime,
  onClose
}) => {
  const { t } = useTranslation(["option", "common"])
  const [args, setArgs] = React.useState("")
  const [result, setResult] = React.useState<SkillExecutionResult | null>(null)
  const [activeRunMode, setActiveRunMode] = React.useState<SkillRunMode | null>(null)
  const skillRunPendingRef = React.useRef(false)

  React.useEffect(() => {
    if (!skillName) {
      setArgs("")
      setResult(null)
      setActiveRunMode(null)
      skillRunPendingRef.current = false
    }
  }, [skillName])

  const executeMutation = useMutation({
    mutationFn: ({ dryRun }: { dryRun: boolean }) =>
      tldwClient.executeSkill(skillName!, args, { dryRun }),
    onSuccess: (data: SkillExecutionResult) => {
      setResult(data)
    },
    onSettled: () => {
      skillRunPendingRef.current = false
      setActiveRunMode(null)
    }
  })

  const handleRun = (dryRun: boolean) => {
    if (!skillName || skillRunPendingRef.current || executeMutation.isPending) return

    skillRunPendingRef.current = true
    setActiveRunMode(dryRun ? "dry-run" : "test-run")
    executeMutation.mutate({ dryRun })
  }

  const handleRunTest = () => handleRun(false)
  const handleRenderOnly = () => handleRun(true)

  const errorMessage =
    executeMutation.error instanceof Error
      ? executeMutation.error.message
      : t("option:skills.testRunError", { defaultValue: "Execution failed" })
  const runtimeToolLabel = runtime
    ? t("option:skills.runtimeDeclaredTools", {
        defaultValue: `${runtime.declared_tool_count} tools declared`,
        count: runtime.declared_tool_count
      })
    : ""
  const testRunDisclosure = runtime
    ? runtime.execution_mode === "fork"
      ? runtime.test_run_may_call_model
        ? t("option:skills.testRunForkMayCallModelDisclosure", {
            defaultValue:
              "Run test uses fork execution and may call the configured model for this skill."
          })
        : t("option:skills.testRunForkPromptOnlyDisclosure", {
            defaultValue:
              "Run test uses fork execution for this skill; model calls are disabled."
          })
      : runtime.test_run_may_call_model
        ? t("option:skills.testRunInlineMayCallModelDisclosure", {
            defaultValue:
              "Run test uses inline prompt execution and may call the configured model for this skill."
          })
        : t("option:skills.testRunInlineDisclosure", {
            defaultValue: "Run test uses inline prompt execution for this skill."
          })
    : ""

  return (
    <Modal
      title={t("option:skills.testRunTitle", {
        defaultValue: "Test run",
        name: skillName
      })}
      open={Boolean(skillName)}
      onCancel={onClose}
      footer={null}
      width={640}
      destroyOnHidden
    >
      <div className="flex flex-col gap-4">
        <div>
          <label className="mb-1 block text-sm font-medium">
            {t("option:skills.previewArgs", {
              defaultValue: "Test Arguments"
            })}
          </label>
          <Input
            value={args}
            onChange={(e) => setArgs(e.target.value)}
            placeholder={t("option:skills.previewArgsPlaceholder", {
              defaultValue: "Enter test arguments..."
            })}
            onPressEnter={handleRunTest}
            disabled={executeMutation.isPending}
          />
        </div>

        {runtime ? (
          <div className="rounded-md border border-border bg-surface p-3">
            <h3 className="m-0 text-sm font-semibold text-text">
              {t("option:skills.runtimeImpactTitle", { defaultValue: "Runtime impact" })}
            </h3>
            <div className="mt-2 flex flex-wrap gap-1">
              <Tag color={runtime.execution_mode === "fork" ? "blue" : "green"}>
                {runtime.execution_mode === "fork"
                  ? t("option:skills.runtimeFork", { defaultValue: "Fork" })
                  : t("option:skills.runtimeInline", { defaultValue: "Inline" })}
              </Tag>
              <Tag color={runtime.test_run_may_call_model ? "orange" : "default"}>
                {runtime.test_run_may_call_model
                  ? t("option:skills.runtimeMayCallModel", { defaultValue: "Test may call model" })
                  : t("option:skills.runtimePromptOnly", { defaultValue: "Prompt only by default" })}
              </Tag>
              {runtime.declares_tools && (
                <Tag color="geekblue">{runtimeToolLabel}</Tag>
              )}
              {runtime.model_override && (
                <Tag>
                  {t("option:skills.runtimeModelOverride", {
                    defaultValue: "Model override"
                  })}
                </Tag>
              )}
              {!runtime.auto_invocation_enabled && (
                <Tag color="warning">
                  {t("option:skills.runtimeAutoOff", {
                    defaultValue: "Auto invocation off"
                  })}
                </Tag>
              )}
            </div>
            <div className="mt-2 space-y-1 text-sm text-text-muted">
              <p className="m-0">
                {t("option:skills.renderOnlyDisclosure", {
                  defaultValue:
                    "Render prompt only does not invoke fork, model, or tool execution."
                })}
              </p>
              <p className="m-0">
                {testRunDisclosure}
              </p>
              {runtime.declares_tools && (
                <p className="m-0">
                  {t("option:skills.declaredToolsDisclosure", {
                    defaultValue:
                      "Declared tools are declarations, not availability guarantees."
                  })}
                </p>
              )}
            </div>
          </div>
        ) : (
          <p className="m-0 text-sm text-text-muted">
            {t("option:skills.testRunDisclosure", {
              defaultValue:
                "This renders the skill with your arguments. Fork-mode skills may call the configured model and allowed tools."
            })}
          </p>
        )}

        <div className="flex flex-col gap-2 sm:flex-row">
          <Button
            onClick={handleRenderOnly}
            loading={executeMutation.isPending && activeRunMode === "dry-run"}
            disabled={executeMutation.isPending}
          >
            {t("option:skills.renderOnlyAction", { defaultValue: "Render prompt only" })}
          </Button>
          <Button
            type="primary"
            onClick={handleRunTest}
            loading={executeMutation.isPending && activeRunMode === "test-run"}
            disabled={executeMutation.isPending}
          >
            {t("option:skills.testRunAction", { defaultValue: "Run test" })}
          </Button>
        </div>

        {executeMutation.isError && (
          <Alert role="alert" type="error" showIcon title={errorMessage} />
        )}

        {result && (
          <div className="flex flex-col gap-3">
            <div className="flex items-center gap-2">
              <Tag color={result.dry_run ? "default" : "green"}>
                {result.dry_run
                  ? t("option:skills.dryRenderResult", { defaultValue: "Dry render" })
                  : t("option:skills.executedResult", { defaultValue: "Executed test" })}
              </Tag>
              <Tag color={result.execution_mode === "fork" ? "blue" : "green"}>
                {result.execution_mode}
              </Tag>
              {result.model_override && (
                <Tag>{result.model_override}</Tag>
              )}
              {result.allowed_tools?.map((tool) => (
                <Tag key={tool} color="orange">
                  {tool}
                </Tag>
              ))}
            </div>

            <div>
              <label className="mb-1 block text-sm font-medium">
                {t("option:skills.previewRendered", {
                  defaultValue: "Rendered Prompt"
                })}
              </label>
              <Input.TextArea
                value={result.rendered_prompt}
                readOnly
                rows={10}
                className="font-mono text-xs"
              />
            </div>

            {result.fork_output && (
              <div>
                <label className="mb-1 block text-sm font-medium">
                  {t("option:skills.previewForkOutput", {
                    defaultValue: "Fork Output"
                  })}
                </label>
                <Input.TextArea
                  value={result.fork_output}
                  readOnly
                  rows={6}
                  className="font-mono text-xs"
                />
              </div>
            )}
          </div>
        )}
      </div>
    </Modal>
  )
}

import React from "react"
import { useMutation } from "@tanstack/react-query"
import { Alert, Button, Input, Modal, Tag } from "antd"
import { useTranslation } from "react-i18next"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import type { SkillExecutionResult, SkillRuntimeMetadata } from "@/types/skill"
import { sanitizeServerErrorMessage } from "@/utils/server-error-message"

interface SkillPreviewProps {
  skillName: string | null
  runtime?: SkillRuntimeMetadata | null
  onClose: () => void
  onAfterClose?: () => void
}

type SkillRunMode = "dry-run" | "test-run"

interface SkillExecutionVariables {
  requestId: number
  skillName: string
  args: string
  dryRun: boolean
  signal: AbortSignal
}

export const SkillPreview: React.FC<SkillPreviewProps> = ({
  skillName,
  runtime,
  onClose,
  onAfterClose
}) => {
  const { t } = useTranslation(["option", "common"])
  const [args, setArgs] = React.useState("")
  const [result, setResult] = React.useState<SkillExecutionResult | null>(null)
  const [runError, setRunError] = React.useState<unknown>(null)
  const [activeRunMode, setActiveRunMode] = React.useState<SkillRunMode | null>(null)
  const requestIdRef = React.useRef(0)
  const activeRequestRef = React.useRef<{
    requestId: number
    skillName: string
    controller: AbortController
  } | null>(null)
  const activeSkillNameRef = React.useRef(skillName)

  const executeMutation = useMutation({
    mutationFn: (variables: SkillExecutionVariables) =>
      tldwClient.executeSkill(variables.skillName, variables.args, {
        dryRun: variables.dryRun,
        signal: variables.signal
      }),
    onSuccess: (data: SkillExecutionResult, variables: SkillExecutionVariables) => {
      const activeRequest = activeRequestRef.current
      if (
        variables.signal.aborted
        || !activeRequest
        || activeRequest.requestId !== variables.requestId
        || activeRequest.skillName !== variables.skillName
        || activeSkillNameRef.current !== variables.skillName
      ) {
        return
      }
      setResult(data)
    },
    onError: (error: unknown, variables: SkillExecutionVariables) => {
      const activeRequest = activeRequestRef.current
      if (
        variables.signal.aborted
        || !activeRequest
        || activeRequest.requestId !== variables.requestId
        || activeSkillNameRef.current !== variables.skillName
      ) {
        return
      }
      setRunError(error)
    },
    onSettled: (_data, _error, variables: SkillExecutionVariables) => {
      if (activeRequestRef.current?.requestId !== variables.requestId) return
      activeRequestRef.current = null
      setActiveRunMode(null)
    }
  })
  const resetMutation = executeMutation.reset

  const resetExecution = React.useCallback(() => {
    requestIdRef.current += 1
    activeRequestRef.current?.controller.abort()
    activeRequestRef.current = null
    resetMutation()
    setResult(null)
    setRunError(null)
    setActiveRunMode(null)
  }, [resetMutation])

  React.useEffect(() => {
    activeSkillNameRef.current = skillName
    resetExecution()
    setArgs("")
  }, [resetExecution, skillName])

  React.useEffect(
    () => () => {
      requestIdRef.current += 1
      activeRequestRef.current?.controller.abort()
      activeRequestRef.current = null
    },
    []
  )

  const handleRun = (dryRun: boolean) => {
    if (!skillName || activeRequestRef.current) return

    const controller = new AbortController()
    const requestId = requestIdRef.current + 1
    requestIdRef.current = requestId
    activeRequestRef.current = { requestId, skillName, controller }
    setResult(null)
    setRunError(null)
    setActiveRunMode(dryRun ? "dry-run" : "test-run")
    executeMutation.mutate({
      requestId,
      skillName,
      args,
      dryRun,
      signal: controller.signal
    })
  }

  const handleRunTest = () => handleRun(false)
  const handleRenderOnly = () => handleRun(true)

  const fallbackErrorMessage = t("option:skills.testRunError", {
    defaultValue: "Execution failed"
  })
  const errorMessage = sanitizeServerErrorMessage(runError, fallbackErrorMessage)
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
  const isRunning = activeRunMode !== null
  const runStatusMessage = isRunning && skillName
    ? activeRunMode === "dry-run"
      ? t("option:skills.renderingPromptStatus", {
          defaultValue: `Rendering prompt for ${skillName}`,
          name: skillName
        })
      : t("option:skills.runningTestStatus", {
          defaultValue: `Running test for ${skillName}`,
          name: skillName
        })
    : result && skillName
      ? result.dry_run
        ? t("option:skills.renderedPromptReadyStatus", {
            defaultValue: `Rendered prompt ready for ${skillName}`,
            name: skillName
          })
        : t("option:skills.testResultReadyStatus", {
            defaultValue: `Test result ready for ${skillName}`,
            name: skillName
          })
      : null

  return (
    <Modal
      title={t("option:skills.testRunTitle", {
        defaultValue: `Test run: ${skillName ?? ""}`,
        name: skillName
      })}
      open={Boolean(skillName)}
      onCancel={() => {
        resetExecution()
        onClose()
      }}
      afterClose={onAfterClose}
      footer={null}
      width={640}
      destroyOnHidden
    >
      <div className="flex flex-col gap-4">
        <div role="status" aria-live="polite" className="sr-only">
          {runStatusMessage ?? ""}
        </div>

        <div>
          <label htmlFor="skill-preview-arguments" className="mb-1 block text-sm font-medium">
            {t("option:skills.previewArgs", {
              defaultValue: "Test Arguments"
            })}
          </label>
          <Input
            id="skill-preview-arguments"
            value={args}
            onChange={(e) => setArgs(e.target.value)}
            placeholder={t("option:skills.previewArgsPlaceholder", {
              defaultValue: "Enter test arguments..."
            })}
            onPressEnter={handleRenderOnly}
            disabled={isRunning}
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
            loading={isRunning && activeRunMode === "dry-run"}
            disabled={isRunning}
          >
            {t("option:skills.renderOnlyAction", { defaultValue: "Render prompt only" })}
          </Button>
          <Button
            type="primary"
            onClick={handleRunTest}
            loading={isRunning && activeRunMode === "test-run"}
            disabled={isRunning}
          >
            {t("option:skills.testRunAction", { defaultValue: "Run test" })}
          </Button>
        </div>

        {runError && (
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
              <label htmlFor="skill-preview-rendered-prompt" className="mb-1 block text-sm font-medium">
                {t("option:skills.previewRendered", {
                  defaultValue: "Rendered Prompt"
                })}
              </label>
              <Input.TextArea
                id="skill-preview-rendered-prompt"
                value={result.rendered_prompt}
                readOnly
                rows={10}
                className="font-mono text-xs"
              />
            </div>

            {result.fork_output && (
              <div>
                <label htmlFor="skill-preview-fork-output" className="mb-1 block text-sm font-medium">
                  {t("option:skills.previewForkOutput", {
                    defaultValue: "Fork Output"
                  })}
                </label>
                <Input.TextArea
                  id="skill-preview-fork-output"
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

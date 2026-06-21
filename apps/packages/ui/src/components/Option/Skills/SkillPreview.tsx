import React from "react"
import { useMutation } from "@tanstack/react-query"
import { Alert, Button, Input, Modal, Tag } from "antd"
import { useTranslation } from "react-i18next"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import type { SkillExecutionResult } from "@/types/skill"

interface SkillPreviewProps {
  skillName: string | null
  onClose: () => void
}

export const SkillPreview: React.FC<SkillPreviewProps> = ({
  skillName,
  onClose
}) => {
  const { t } = useTranslation(["option", "common"])
  const [args, setArgs] = React.useState("")
  const [result, setResult] = React.useState<SkillExecutionResult | null>(null)

  React.useEffect(() => {
    if (!skillName) {
      setArgs("")
      setResult(null)
    }
  }, [skillName])

  const executeMutation = useMutation({
    mutationFn: () => tldwClient.executeSkill(skillName!, args),
    onSuccess: (data: SkillExecutionResult) => {
      setResult(data)
    }
  })

  const handleRunTest = () => {
    if (skillName) {
      executeMutation.mutate()
    }
  }

  const errorMessage =
    executeMutation.error instanceof Error
      ? executeMutation.error.message
      : t("option:skills.testRunError", { defaultValue: "Execution failed" })

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
          />
        </div>

        <p className="m-0 text-sm text-text-muted">
          {t("option:skills.testRunDisclosure", {
            defaultValue:
              "This renders the skill with your arguments. Fork-mode skills may call the configured model and allowed tools."
          })}
        </p>

        <Button
          type="primary"
          onClick={handleRunTest}
          loading={executeMutation.isPending}
        >
          {t("option:skills.testRunAction", { defaultValue: "Run test" })}
        </Button>

        {executeMutation.isError && (
          <Alert role="alert" type="error" showIcon title={errorMessage} />
        )}

        {result && (
          <div className="flex flex-col gap-3">
            <div className="flex items-center gap-2">
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

import { useMutation, useQueryClient } from "@tanstack/react-query"
import { Drawer, Tabs, notification, Upload, Spin } from "antd"
import type { UploadProps } from "antd"
import { Download, Upload as UploadIcon, FileJson, FileSpreadsheet } from "lucide-react"
import React, { useState } from "react"
import { useTranslation } from "react-i18next"
import {
  exportTestCases,
  importTestCases,
  createBulkTestCases,
  type TestCaseExportFormat,
  type TestCaseCreatePayload
} from "@/services/prompt-studio"
import { Button } from "@/components/Common/Button"
import { Alert as DsAlert } from "@/components/ui/primitives"

type TestCaseBulkPanelProps = {
  open: boolean
  projectId: number
  onClose: () => void
}

export const TestCaseBulkPanel: React.FC<TestCaseBulkPanelProps> = ({
  open,
  projectId,
  onClose
}) => {
  const { t } = useTranslation(["settings", "common"])
  const queryClient = useQueryClient()
  const [exportFormat, setExportFormat] = useState<TestCaseExportFormat>("json")

  // Export mutation
  const exportMutation = useMutation({
    mutationFn: (format: TestCaseExportFormat) =>
      exportTestCases(projectId, { format }),
    onSuccess: (response, format) => {
      const data = (response as any)?.data
      let content: string
      let mimeType: string
      let extension: string

      if (format === "csv") {
        content = typeof data === "string" ? data : JSON.stringify(data)
        mimeType = "text/csv"
        extension = "csv"
      } else {
        content = JSON.stringify(data, null, 2)
        mimeType = "application/json"
        extension = "json"
      }

      const blob = new Blob([content], { type: mimeType })
      const url = URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = `test_cases_project_${projectId}_${new Date().toISOString().slice(0, 10)}.${extension}`
      a.click()
      URL.revokeObjectURL(url)

      notification.success({
        message: t("managePrompts.studio.testCases.exportSuccess", {
          defaultValue: "Test cases exported"
        })
      })
    },
    onError: (error: any) => {
      notification.error({
        message: t("common:error", { defaultValue: "Error" }),
        description: error?.message || t("common:unknownError")
      })
    }
  })

  // Import mutation
  const importMutation = useMutation({
    mutationFn: async (file: File) => {
      const text = await file.text()
      const isJson = file.name.endsWith(".json")

      if (isJson) {
        const data = JSON.parse(text) as TestCaseCreatePayload[]
        return createBulkTestCases({
          project_id: projectId,
          test_cases: data.map((tc) => ({
            name: tc.name,
            description: tc.description,
            inputs: tc.inputs,
            expected_outputs: tc.expected_outputs,
            tags: tc.tags,
            is_golden: tc.is_golden
          }))
        })
      } else {
        return importTestCases({
          project_id: projectId,
          format: "csv",
          data: text
        })
      }
    },
    onSuccess: (response) => {
      queryClient.invalidateQueries({
        queryKey: ["prompt-studio", "test-cases", projectId]
      })
      const count = (response as any)?.data?.data?.length ?? 0
      notification.success({
        message: t("managePrompts.studio.testCases.importSuccess", {
          defaultValue: "{{count}} test cases imported",
          count
        })
      })
    },
    onError: (error: any) => {
      notification.error({
        message: t("common:error", { defaultValue: "Error" }),
        description: error?.message || t("common:unknownError")
      })
    }
  })

  const handleExport = () => {
    exportMutation.mutate(exportFormat)
  }

  const uploadProps: UploadProps = {
    name: "file",
    accept: ".json,.csv",
    showUploadList: false,
    beforeUpload: (file) => {
      importMutation.mutate(file)
      return false
    }
  }

  const downloadCsvTemplate = () => {
    const template = `name,description,inputs,expected_outputs,tags,is_golden
"Simple query","Basic test case","{""query"": ""Hello""}","{""response"": ""Hi there!""}","greeting,simple",false
"Complex query","Test with multiple inputs","{""query"": ""Help"", ""context"": ""support""}","",""customer"",true`

    const blob = new Blob([template], { type: "text/csv" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = "test_cases_template.csv"
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <Drawer
      open={open}
      onClose={onClose}
      title={t("managePrompts.studio.testCases.bulkTitle", {
        defaultValue: "Import / Export Test Cases"
      })}
      size={500}
      destroyOnHidden
    >
      <Tabs
        items={[
          {
            key: "export",
            label: (
              <span className="flex items-center gap-2">
                <Download className="size-4" />
                {t("managePrompts.studio.testCases.export", {
                  defaultValue: "Export"
                })}
              </span>
            ),
            children: (
              <div className="space-y-4">
                <DsAlert
                  variant="info"
                  title={t("managePrompts.studio.testCases.exportInfo", {
                    defaultValue:
                      "Export all test cases from this project to a file."
                  })}
                />

                <div className="space-y-2">
                  <label className="text-sm font-medium">
                    {t("managePrompts.studio.testCases.exportFormat", {
                      defaultValue: "Format"
                    })}
                  </label>
                  <div className="flex gap-2">
                    <button
                      onClick={() => setExportFormat("json")}
                      className={`flex items-center gap-2 px-4 py-2 rounded-md border ${
                        exportFormat === "json"
                          ? "border-primary bg-primary/10 text-primary"
                          : "border-border hover:bg-surface2"
                      }`}
                    >
                      <FileJson className="size-4" />
                      JSON
                    </button>
                    <button
                      onClick={() => setExportFormat("csv")}
                      className={`flex items-center gap-2 px-4 py-2 rounded-md border ${
                        exportFormat === "csv"
                          ? "border-primary bg-primary/10 text-primary"
                          : "border-border hover:bg-surface2"
                      }`}
                    >
                      <FileSpreadsheet className="size-4" />
                      CSV
                    </button>
                  </div>
                </div>

                <Button
                  type="primary"
                  onClick={handleExport}
                  loading={exportMutation.isPending}
                  className="w-full"
                >
                  <Download className="size-4 mr-1" />
                  {t("managePrompts.studio.testCases.downloadBtn", {
                    defaultValue: "Download"
                  })}
                </Button>
              </div>
            )
          },
          {
            key: "import",
            label: (
              <span className="flex items-center gap-2">
                <UploadIcon className="size-4" />
                {t("managePrompts.studio.testCases.import", {
                  defaultValue: "Import"
                })}
              </span>
            ),
            children: (
              <div className="space-y-4">
                <DsAlert
                  variant="info"
                  title={t("managePrompts.studio.testCases.importInfo", {
                    defaultValue:
                      "Import test cases from a JSON or CSV file. New test cases will be added to the project."
                  })}
                />

                <Upload.Dragger {...uploadProps} disabled={importMutation.isPending}>
                  {importMutation.isPending ? (
                    <div className="py-4">
                      <Spin size="large" />
                      <p className="mt-2 text-text-muted">
                        {t("managePrompts.studio.testCases.importing", {
                          defaultValue: "Importing..."
                        })}
                      </p>
                    </div>
                  ) : (
                    <div className="py-4">
                      <p className="text-3xl text-text-muted">
                        <UploadIcon className="size-10 mx-auto mb-2" />
                      </p>
                      <p className="text-sm">
                        {t("managePrompts.studio.testCases.dragDropHint", {
                          defaultValue:
                            "Click or drag a file here to import"
                        })}
                      </p>
                      <p className="text-xs text-text-muted mt-1">
                        {t("managePrompts.studio.testCases.supportedFormats", {
                          defaultValue: "Supports JSON and CSV files"
                        })}
                      </p>
                    </div>
                  )}
                </Upload.Dragger>

                <div className="border-t border-border pt-4">
                  <p className="text-sm font-medium mb-2">
                    {t("managePrompts.studio.testCases.needTemplate", {
                      defaultValue: "Need a template?"
                    })}
                  </p>
                  <Button type="secondary" onClick={downloadCsvTemplate}>
                    <Download className="size-4 mr-1" />
                    {t("managePrompts.studio.testCases.downloadTemplate", {
                      defaultValue: "Download CSV Template"
                    })}
                  </Button>
                </div>
              </div>
            )
          }
        ]}
      />
    </Drawer>
  )
}

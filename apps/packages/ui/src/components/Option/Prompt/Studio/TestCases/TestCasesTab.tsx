import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import {
  Table,
  Skeleton,
  Input,
  Tag,
  Tooltip,
  notification,
  Dropdown,
  Select
} from "antd"
import type { MenuProps } from "antd"
import {
  Plus,
  Search,
  TestTube,
  Star,
  Trash2,
  Pen,
  MoreHorizontal,
  Upload,
  Download,
  Sparkles,
  Play
} from "lucide-react"
import React, { useMemo, useState } from "react"
import { useTranslation } from "react-i18next"
import { usePromptStudioStore } from "@/store/prompt-studio"
import {
  listTestCases,
  deleteTestCase,
  updateTestCase,
  type TestCase
} from "@/services/prompt-studio"
import { useConfirmDanger } from "@/components/Common/confirm-danger"
import FeatureEmptyState from "@/components/Common/FeatureEmptyState"
import { Button } from "@/components/Common/Button"
import { TestCaseFormModal } from "./TestCaseFormModal"
import { TestCaseBulkPanel } from "./TestCaseBulkPanel"
import { TestCaseGenerateModal } from "./TestCaseGenerateModal"
import { TestCaseRunModal } from "./TestCaseRunModal"

export const TestCasesTab: React.FC = () => {
  const { t } = useTranslation(["settings", "common"])
  const queryClient = useQueryClient()
  const confirmDanger = useConfirmDanger()

  const [searchText, setSearchText] = useState("")
  const [goldenFilter, setGoldenFilter] = useState<boolean | undefined>(undefined)
  const [tagFilter, setTagFilter] = useState<string[]>([])
  const [bulkPanelOpen, setBulkPanelOpen] = useState(false)
  const [generateModalOpen, setGenerateModalOpen] = useState(false)
  const [runModalOpen, setRunModalOpen] = useState(false)

  const selectedProjectId = usePromptStudioStore((s) => s.selectedProjectId)
  const selectedTestCaseIds = usePromptStudioStore((s) => s.selectedTestCaseIds)
  const setSelectedTestCaseIds = usePromptStudioStore((s) => s.setSelectedTestCaseIds)
  const clearTestCaseSelection = usePromptStudioStore((s) => s.clearTestCaseSelection)
  const isTestCaseModalOpen = usePromptStudioStore((s) => s.isTestCaseModalOpen)
  const setTestCaseModalOpen = usePromptStudioStore((s) => s.setTestCaseModalOpen)
  const editingTestCaseId = usePromptStudioStore((s) => s.editingTestCaseId)
  const setEditingTestCaseId = usePromptStudioStore((s) => s.setEditingTestCaseId)

  // Fetch test cases
  const { data: testCasesResponse, status: testCasesStatus } = useQuery({
    queryKey: [
      "prompt-studio",
      "test-cases",
      selectedProjectId,
      { is_golden: goldenFilter }
    ],
    queryFn: () =>
      listTestCases(selectedProjectId!, {
        per_page: 100,
        is_golden: goldenFilter
      }),
    enabled: selectedProjectId !== null
  })

  const testCases: TestCase[] = (testCasesResponse as any)?.data?.data ?? []

  // Extract all tags for filter
  const allTags = useMemo(() => {
    const tagSet = new Set<string>()
    testCases.forEach((tc) => {
      tc.tags?.forEach((tag) => tagSet.add(tag))
    })
    return Array.from(tagSet)
  }, [testCases])

  // Toggle golden mutation
  const toggleGoldenMutation = useMutation({
    mutationFn: ({ id, isGolden }: { id: number; isGolden: boolean }) =>
      updateTestCase(id, { is_golden: isGolden }),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: ["prompt-studio", "test-cases", selectedProjectId]
      })
    },
    onError: (error: any) => {
      notification.error({
        message: t("common:error", { defaultValue: "Error" }),
        description: error?.message || t("common:unknownError")
      })
    }
  })

  // Delete mutation
  const deleteMutation = useMutation({
    mutationFn: (id: number) => deleteTestCase(id),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: ["prompt-studio", "test-cases", selectedProjectId]
      })
      notification.success({
        message: t("managePrompts.studio.testCases.deleted", {
          defaultValue: "Test case deleted"
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

  // Filter test cases
  const filteredTestCases = useMemo(() => {
    let items = testCases
    if (searchText.trim()) {
      const q = searchText.toLowerCase()
      items = items.filter(
        (tc) =>
          tc.name?.toLowerCase().includes(q) ||
          tc.description?.toLowerCase().includes(q) ||
          JSON.stringify(tc.inputs).toLowerCase().includes(q)
      )
    }
    if (tagFilter.length > 0) {
      items = items.filter((tc) =>
        tc.tags?.some((tag) => tagFilter.includes(tag))
      )
    }
    return items.sort((a, b) => {
      // Golden first, then by created date
      if (a.is_golden !== b.is_golden) return a.is_golden ? -1 : 1
      return (
        new Date(b.created_at || 0).getTime() -
        new Date(a.created_at || 0).getTime()
      )
    })
  }, [testCases, searchText, tagFilter])

  const handleOpenCreate = () => {
    setEditingTestCaseId(null)
    setTestCaseModalOpen(true)
  }

  const handleOpenEdit = (testCase: TestCase) => {
    setEditingTestCaseId(testCase.id)
    setTestCaseModalOpen(true)
  }

  const handleDelete = async (testCase: TestCase) => {
    const ok = await confirmDanger({
      title: t("managePrompts.studio.testCases.deleteConfirmTitle", {
        defaultValue: "Delete test case?"
      }),
      content: t("managePrompts.studio.testCases.deleteConfirmContent", {
        defaultValue: "This will permanently delete this test case."
      }),
      okText: t("common:delete", { defaultValue: "Delete" }),
      cancelText: t("common:cancel", { defaultValue: "Cancel" })
    })
    if (ok) {
      deleteMutation.mutate(testCase.id)
    }
  }

  const getTestCaseActions = (testCase: TestCase): MenuProps["items"] => [
    {
      key: "edit",
      icon: <Pen className="size-4" />,
      label: t("common:edit", { defaultValue: "Edit" }),
      onClick: () => handleOpenEdit(testCase)
    },
    {
      key: "golden",
      icon: <Star className={`size-4 ${testCase.is_golden ? "fill-warn text-warn" : ""}`} />,
      label: testCase.is_golden
        ? t("managePrompts.studio.testCases.unmarkGolden", {
            defaultValue: "Unmark as golden"
          })
        : t("managePrompts.studio.testCases.markGolden", {
            defaultValue: "Mark as golden"
          }),
      onClick: () =>
        toggleGoldenMutation.mutate({
          id: testCase.id,
          isGolden: !testCase.is_golden
        })
    },
    { type: "divider" },
    {
      key: "delete",
      icon: <Trash2 className="size-4" />,
      label: t("common:delete", { defaultValue: "Delete" }),
      danger: true,
      onClick: () => handleDelete(testCase)
    }
  ]

  if (!selectedProjectId) {
    return (
      <FeatureEmptyState
        title={t("managePrompts.studio.testCases.noProjectSelected", {
          defaultValue: "Select a project first"
        })}
        description={t("managePrompts.studio.testCases.noProjectSelectedDesc", {
          defaultValue:
            "Go to the Projects tab and select a project to manage its test cases."
        })}
        examples={[]}
      />
    )
  }

  if (testCasesStatus === "pending") {
    return <Skeleton paragraph={{ rows: 8 }} />
  }

  if (testCasesStatus === "success" && testCases.length === 0) {
    return (
      <>
        <FeatureEmptyState
          title={t("managePrompts.studio.testCases.emptyTitle", {
            defaultValue: "No test cases yet"
          })}
          description={t("managePrompts.studio.testCases.emptyDescription", {
            defaultValue:
              "Create test cases to evaluate your prompts with specific inputs and expected outputs."
          })}
          examples={[
            t("managePrompts.studio.testCases.emptyExample1", {
              defaultValue:
                "Test cases help you measure prompt performance systematically."
            }),
            t("managePrompts.studio.testCases.emptyExample2", {
              defaultValue:
                "Mark important test cases as 'golden' for regression testing."
            })
          ]}
          primaryActionLabel={t("managePrompts.studio.testCases.createBtn", {
            defaultValue: "Create Test Case"
          })}
          onPrimaryAction={handleOpenCreate}
        />
        <TestCaseFormModal
          open={isTestCaseModalOpen}
          testCaseId={editingTestCaseId}
          projectId={selectedProjectId}
          onClose={() => {
            setTestCaseModalOpen(false)
            setEditingTestCaseId(null)
          }}
        />
      </>
    )
  }

  return (
    <div className="space-y-4">
      {/* Bulk selection bar */}
      {selectedTestCaseIds.length > 0 && (
        <div className="flex items-center gap-3 p-2 bg-primary/10 rounded-md border border-primary/30">
          <span className="text-sm text-primary">
            {t("managePrompts.studio.testCases.selected", {
              defaultValue: "{{count}} selected",
              count: selectedTestCaseIds.length
            })}
          </span>
          <Button
            type="ghost"
            size="sm"
            onClick={() => setRunModalOpen(true)}
          >
            <Play className="size-3 mr-1" />
            {t("managePrompts.studio.testCases.runSelected", {
              defaultValue: "Run"
            })}
          </Button>
          <button
            onClick={clearTestCaseSelection}
            className="ml-auto text-sm text-text-muted hover:text-text"
          >
            {t("common:clearSelection", { defaultValue: "Clear selection" })}
          </button>
        </div>
      )}

      {/* Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <Button type="primary" onClick={handleOpenCreate}>
            <Plus className="size-4 mr-1" />
            {t("managePrompts.studio.testCases.createBtn", {
              defaultValue: "Create"
            })}
          </Button>
          <Button type="secondary" onClick={() => setBulkPanelOpen(true)}>
            <Upload className="size-4 mr-1" />
            {t("managePrompts.studio.testCases.importExport", {
              defaultValue: "Import/Export"
            })}
          </Button>
          <Button type="secondary" onClick={() => setGenerateModalOpen(true)}>
            <Sparkles className="size-4 mr-1" />
            {t("managePrompts.studio.testCases.generate", {
              defaultValue: "Generate"
            })}
          </Button>
        </div>

        <div className="flex items-center gap-2">
          <Input
            placeholder={t(
              "managePrompts.studio.testCases.searchPlaceholder",
              {
                defaultValue: "Search test cases..."
              }
            )}
            prefix={<Search className="size-4 text-text-muted" />}
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
            allowClear
            style={{ width: 200 }}
          />
          <Select
            placeholder={t("managePrompts.studio.testCases.filterGolden", {
              defaultValue: "All"
            })}
            value={goldenFilter}
            onChange={(v) => setGoldenFilter(v)}
            allowClear
            style={{ width: 120 }}
            options={[
              {
                label: t("managePrompts.studio.testCases.goldenOnly", {
                  defaultValue: "Golden only"
                }),
                value: true
              },
              {
                label: t("managePrompts.studio.testCases.nonGolden", {
                  defaultValue: "Non-golden"
                }),
                value: false
              }
            ]}
          />
          {allTags.length > 0 && (
            <Select
              mode="multiple"
              placeholder={t("managePrompts.studio.testCases.filterTags", {
                defaultValue: "Filter by tags"
              })}
              value={tagFilter}
              onChange={(v) => setTagFilter(v)}
              allowClear
              style={{ minWidth: 150 }}
              options={allTags.map((tag) => ({ label: tag, value: tag }))}
            />
          )}
        </div>
      </div>

      {/* Test cases table */}
      <Table<TestCase>
        dataSource={filteredTestCases}
        rowKey="id"
        size="middle"
        pagination={{
          pageSize: 20,
          showSizeChanger: true,
          showTotal: (total) =>
            t("managePrompts.studio.testCases.totalCount", {
              defaultValue: "{{count}} test cases",
              count: total
            })
        }}
        rowSelection={{
          selectedRowKeys: selectedTestCaseIds,
          onChange: (keys) => setSelectedTestCaseIds(keys as number[])
        }}
        onRow={(record) => ({
          onDoubleClick: () => handleOpenEdit(record)
        })}
        columns={[
          {
            title: "",
            width: 40,
            render: (_, record) =>
              record.is_golden ? (
                <Tooltip
                  title={t("managePrompts.studio.testCases.goldenTooltip", {
                    defaultValue: "Golden test case"
                  })}
                >
                  <Star className="size-5 fill-warn text-warn" />
                </Tooltip>
              ) : (
                <TestTube className="size-5 text-text-muted" />
              )
          },
          {
            title: t("managePrompts.studio.testCases.columns.name", {
              defaultValue: "Name"
            }),
            key: "name",
            render: (_, record) => (
              <div className="flex flex-col">
                <span className="font-medium">
                  {record.name || `Test Case #${record.id}`}
                </span>
                {record.description && (
                  <span className="text-xs text-text-muted line-clamp-1">
                    {record.description}
                  </span>
                )}
              </div>
            )
          },
          {
            title: t("managePrompts.studio.testCases.columns.inputs", {
              defaultValue: "Inputs"
            }),
            key: "inputs",
            render: (_, record) => (
              <pre className="text-xs bg-surface2 p-1 rounded max-w-xs overflow-hidden text-ellipsis">
                {JSON.stringify(record.inputs, null, 0).substring(0, 80)}
                {JSON.stringify(record.inputs).length > 80 && "..."}
              </pre>
            )
          },
          {
            title: t("managePrompts.studio.testCases.columns.expectedOutputs", {
              defaultValue: "Expected"
            }),
            key: "expected_outputs",
            render: (_, record) =>
              record.expected_outputs ? (
                <pre className="text-xs bg-surface2 p-1 rounded max-w-xs overflow-hidden text-ellipsis">
                  {JSON.stringify(record.expected_outputs, null, 0).substring(
                    0,
                    80
                  )}
                  {JSON.stringify(record.expected_outputs).length > 80 && "..."}
                </pre>
              ) : (
                <span className="text-text-muted text-sm">-</span>
              )
          },
          {
            title: t("managePrompts.studio.testCases.columns.tags", {
              defaultValue: "Tags"
            }),
            key: "tags",
            width: 150,
            render: (_, record) =>
              record.tags && record.tags.length > 0 ? (
                <div className="flex flex-wrap gap-1">
                  {record.tags.slice(0, 3).map((tag) => (
                    <Tag key={tag}>{tag}</Tag>
                  ))}
                  {record.tags.length > 3 && (
                    <Tag>+{record.tags.length - 3}</Tag>
                  )}
                </div>
              ) : null
          },
          {
            title: "",
            key: "actions",
            width: 50,
            render: (_, record) => (
              <Dropdown
                menu={{ items: getTestCaseActions(record) }}
                trigger={["click"]}
                placement="bottomRight"
              >
                <button
                  onClick={(e) => e.stopPropagation()}
                  className="p-1 rounded hover:bg-surface2"
                >
                  <MoreHorizontal className="size-4" />
                </button>
              </Dropdown>
            )
          }
        ]}
      />

      {/* Modals */}
      <TestCaseFormModal
        open={isTestCaseModalOpen}
        testCaseId={editingTestCaseId}
        projectId={selectedProjectId}
        onClose={() => {
          setTestCaseModalOpen(false)
          setEditingTestCaseId(null)
        }}
      />

      <TestCaseBulkPanel
        open={bulkPanelOpen}
        projectId={selectedProjectId}
        onClose={() => setBulkPanelOpen(false)}
      />

      <TestCaseGenerateModal
        open={generateModalOpen}
        projectId={selectedProjectId}
        onClose={() => setGenerateModalOpen(false)}
      />

      <TestCaseRunModal
        open={runModalOpen}
        projectId={selectedProjectId}
        testCaseIds={selectedTestCaseIds}
        onClose={() => setRunModalOpen(false)}
      />
    </div>
  )
}

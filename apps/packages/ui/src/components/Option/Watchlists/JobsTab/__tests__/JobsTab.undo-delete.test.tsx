import React from "react"
import { act, fireEvent, screen, waitFor } from "@testing-library/react"
import { render } from "../../shared/__tests__/render-with-watchlists-query"
import { Modal } from "antd"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { JobsTab } from "../JobsTab"

const mocks = vi.hoisted(() => ({
  changeAccount: { current: () => {} },
  fetchJobRunsMock: vi.fn(),
  fetchWatchlistJobsMock: vi.fn(),
  fetchWatchlistSourcesMock: vi.fn(),
  fetchWatchlistGroupsMock: vi.fn(),
  deleteWatchlistJobMock: vi.fn(),
  restoreWatchlistJobMock: vi.fn(),
  updateWatchlistJobMock: vi.fn(),
  triggerWatchlistRunMock: vi.fn(),
  messageErrorMock: vi.fn(),
  messageSuccessMock: vi.fn(),
  showUndoNotificationMock: vi.fn(),
  storeStateRef: { current: {} as Record<string, any> }
}))

vi.mock("@/services/chat-account-boundary", () => ({
  watchChatAccountChanges: (listener: (invalidated: boolean) => void) => {
    mocks.changeAccount.current = () => listener(true)
    return () => undefined
  }
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, defaultValue?: unknown, options?: Record<string, unknown>) => {
      if (typeof defaultValue !== "string") return _key
      if (!options) return defaultValue
      return defaultValue.replace(/\{\{(\w+)\}\}/g, (_, token) => String(options[token] ?? ""))
    }
  })
}))

vi.mock("antd", () => {
  const Button = ({ children, onClick, danger = false, loading = false }: any) => (
    <button
      type="button"
      data-testid={danger ? "danger-button" : "button"}
      disabled={Boolean(loading)}
      onClick={() => onClick?.()}
    >
      {children}
    </button>
  )

  const Table = ({ dataSource = [], columns = [] }: any) => (
    <div data-testid="jobs-table">
      {dataSource.map((record: any, rowIndex: number) => (
        <div key={record.id ?? rowIndex}>
          {columns.map((column: any, columnIndex: number) => {
            const key = String(column.key ?? column.dataIndex ?? columnIndex)
            if (key !== "actions") return null
            const value = column.dataIndex ? record[column.dataIndex] : undefined
            return (
              <div key={key}>
                {column.render ? column.render(value, record, rowIndex) : null}
              </div>
            )
          })}
        </div>
      ))}
    </div>
  )

  const Popconfirm = ({ children, onConfirm }: any) => (
    <span onClick={() => onConfirm?.()}>{children}</span>
  )

  return {
    Button,
    Popconfirm,
    Space: ({ children }: any) => <>{children}</>,
    Modal: {
      confirm: vi.fn(() => ({
        destroy: vi.fn(),
        update: vi.fn()
      }))
    },
    Switch: () => null,
    Table,
    Tag: ({ children }: any) => <span>{children}</span>,
    Tooltip: ({ children }: any) => <>{children}</>,
    message: {
      success: mocks.messageSuccessMock,
      error: mocks.messageErrorMock,
      warning: vi.fn()
    }
  }
})

vi.mock("@/hooks/useUndoNotification", () => ({
  useUndoNotification: () => ({
    showUndoNotification: mocks.showUndoNotificationMock
  })
}))

vi.mock("@/services/watchlists", () => ({
  deleteWatchlistJob: (...args: any[]) => mocks.deleteWatchlistJobMock(...args),
  fetchJobRuns: (...args: any[]) => mocks.fetchJobRunsMock(...args),
  fetchWatchlistJobs: (...args: any[]) => mocks.fetchWatchlistJobsMock(...args),
  fetchWatchlistSources: (...args: any[]) => mocks.fetchWatchlistSourcesMock(...args),
  fetchWatchlistGroups: (...args: any[]) => mocks.fetchWatchlistGroupsMock(...args),
  restoreWatchlistJob: (...args: any[]) => mocks.restoreWatchlistJobMock(...args),
  triggerWatchlistRun: (...args: any[]) => mocks.triggerWatchlistRunMock(...args),
  updateWatchlistJob: (...args: any[]) => mocks.updateWatchlistJobMock(...args)
}))

vi.mock("@/store/watchlists", () => ({
  useWatchlistsStore: (selector: (state: Record<string, any>) => unknown) =>
    selector(mocks.storeStateRef.current)
}))

vi.mock("@/utils/dateFormatters", () => ({
  formatRelativeTime: () => "just now"
}))

vi.mock("../JobFormModal", () => ({
  JobFormModal: () => null
}))

vi.mock("../JobPreviewModal", () => ({
  JobPreviewModal: () => null
}))

type DeleteConfirmConfig = {
  onOk?: () => unknown | Promise<unknown>
  onCancel?: () => unknown | Promise<unknown>
}

const getPendingDeleteConfirmation = async (): Promise<DeleteConfirmConfig> => {
  await waitFor(() => {
    expect(Modal.confirm).toHaveBeenCalled()
  })
  const calls = vi.mocked(Modal.confirm).mock.calls
  const config = calls[calls.length - 1][0] as DeleteConfirmConfig
  expect(typeof config.onOk).toBe("function")
  return config
}

vi.mock("../../shared", () => ({
  CronDisplay: () => <span>Cron</span>,
  StatusTag: () => <span>Status</span>
}))

const baseState = (overrides: Record<string, unknown> = {}) => ({
  jobs: [],
  jobsLoading: false,
  jobsTotal: 0,
  jobsPage: 1,
  jobsPageSize: 20,
  jobFormOpen: false,
  jobFormEditId: null,
  setJobs: vi.fn(),
  setJobsLoading: vi.fn(),
  setJobsPage: vi.fn(),
  setJobsPageSize: vi.fn(),
  openJobForm: vi.fn(),
  closeJobForm: vi.fn(),
  addJob: vi.fn(),
  updateJobInList: vi.fn(),
  removeJob: vi.fn(),
  addRun: vi.fn(),
  ...overrides
})

describe("JobsTab undo delete flow", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    const job = {
      id: 41,
      name: "Morning Monitor",
      description: "Daily scan",
      active: true,
      scope: {
        sources: [1],
        groups: [],
        tags: ["news"]
      },
      schedule_expr: "0 9 * * *",
      timezone: "UTC",
      output_prefs: {},
      created_at: "2026-02-18T00:00:00Z",
      updated_at: "2026-02-18T00:00:00Z",
      last_run_at: null,
      next_run_at: null
    }

    mocks.storeStateRef.current = baseState()

    let deleted = false
    mocks.fetchWatchlistJobsMock.mockImplementation(async () => ({
      items: deleted ? [] : [job],
      total: deleted ? 0 : 1,
      page: 1,
      size: 20,
      has_more: false
    }))
    mocks.fetchWatchlistSourcesMock.mockResolvedValue({ items: [], total: 0, has_more: false })
    mocks.fetchWatchlistGroupsMock.mockResolvedValue({ items: [], total: 0, has_more: false })
    mocks.fetchJobRunsMock.mockResolvedValue({
      items: [],
      total: 0,
      page: 1,
      size: 10,
      has_more: false
    })
    mocks.deleteWatchlistJobMock.mockImplementation(async () => {
      deleted = true
      return {
        success: true,
        job_id: 41,
        restore_window_seconds: 10,
        restore_expires_at: "2026-02-18T00:00:10Z"
      }
    })
    mocks.restoreWatchlistJobMock.mockImplementation(async () => {
      deleted = false
      return job
    })
    mocks.updateWatchlistJobMock.mockResolvedValue(job)
    mocks.triggerWatchlistRunMock.mockResolvedValue({
      id: 77,
      job_id: 41,
      status: "queued",
      started_at: "2026-02-18T00:00:00Z",
      finished_at: null,
      stats: {}
    })
  })

  it("deletes a monitor and restores it when undo callback is executed", async () => {
    render(<JobsTab />)

    await waitFor(() => {
      expect(screen.getByTestId("jobs-table")).toBeInTheDocument()
    })

    fireEvent.click(await screen.findByTestId("danger-button"))
    const confirmConfig = await getPendingDeleteConfirmation()
    expect(mocks.deleteWatchlistJobMock).not.toHaveBeenCalled()

    await confirmConfig.onOk?.()

    await waitFor(() => {
      expect(mocks.deleteWatchlistJobMock).toHaveBeenCalledWith(41)
      expect(screen.queryByTestId("danger-button")).not.toBeInTheDocument()
      expect(mocks.showUndoNotificationMock).toHaveBeenCalledTimes(1)
    })

    const undoOptions = mocks.showUndoNotificationMock.mock.calls[0][0]
    expect(undoOptions.title).toBe("Monitor deleted")
    expect(undoOptions.duration).toBe(10)
    expect(undoOptions.description).toBe(
      "Undo restores this monitor's schedule, feed scope, and delivery settings for 10 seconds."
    )

    await undoOptions.onUndo()

    expect(mocks.restoreWatchlistJobMock).toHaveBeenCalledWith(41)
    expect(await screen.findByTestId("danger-button")).toBeInTheDocument()
  })

  it("does not delete a monitor when the confirmation is cancelled", async () => {
    render(<JobsTab />)

    await waitFor(() => {
      expect(screen.getByTestId("jobs-table")).toBeInTheDocument()
    })

    fireEvent.click(await screen.findByTestId("danger-button"))
    const confirmConfig = await getPendingDeleteConfirmation()

    expect(typeof confirmConfig.onCancel).toBe("function")
    await confirmConfig.onCancel?.()

    expect(mocks.deleteWatchlistJobMock).not.toHaveBeenCalled()
    expect(screen.getByTestId("danger-button")).toBeInTheDocument()
    expect(mocks.showUndoNotificationMock).not.toHaveBeenCalled()
  })

  it("does not restore a monitor through an undo callback from a previous account", async () => {
    render(<JobsTab />)
    fireEvent.click(await screen.findByTestId("danger-button"))
    const confirmation = await getPendingDeleteConfirmation()
    await confirmation.onOk?.()
    await waitFor(() => expect(mocks.showUndoNotificationMock).toHaveBeenCalledTimes(1))
    const undo = mocks.showUndoNotificationMock.mock.calls[0][0].onUndo

    act(() => { mocks.changeAccount.current() })
    await expect(undo()).rejects.toMatchObject({ name: "AbortError" })

    expect(mocks.restoreWatchlistJobMock).not.toHaveBeenCalled()
  })

  it("refreshes monitors immediately while undo is available", async () => {
    render(<JobsTab />)

    await waitFor(() => {
      expect(screen.getByTestId("jobs-table")).toBeInTheDocument()
    })

    fireEvent.click(await screen.findByTestId("danger-button"))
    const confirmConfig = await getPendingDeleteConfirmation()
    expect(mocks.showUndoNotificationMock).not.toHaveBeenCalled()

    await confirmConfig.onOk?.()

    await waitFor(() => {
      expect(mocks.fetchWatchlistJobsMock).toHaveBeenCalledTimes(2)
      expect(mocks.showUndoNotificationMock).toHaveBeenCalledTimes(1)
    })
    expect(mocks.restoreWatchlistJobMock).not.toHaveBeenCalled()
  })

  it("falls back to default undo window when backend returns invalid restore timing", async () => {
    mocks.deleteWatchlistJobMock.mockResolvedValueOnce({
      success: true,
      job_id: 41,
      restore_window_seconds: 0,
      restore_expires_at: null
    })

    render(<JobsTab />)

    await waitFor(() => {
      expect(screen.getByTestId("jobs-table")).toBeInTheDocument()
    })

    fireEvent.click(await screen.findByTestId("danger-button"))
    const confirmConfig = await getPendingDeleteConfirmation()
    expect(mocks.showUndoNotificationMock).not.toHaveBeenCalled()

    await confirmConfig.onOk?.()

    await waitFor(() => {
      expect(mocks.showUndoNotificationMock).toHaveBeenCalledTimes(1)
    })
    const undoOptions = mocks.showUndoNotificationMock.mock.calls[0][0]
    expect(undoOptions.duration).toBe(10)
    expect(undoOptions.description).toBe(
      "Undo restores this monitor's schedule, feed scope, and delivery settings for 10 seconds."
    )
  })
})

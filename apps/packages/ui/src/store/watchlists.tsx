import { createWithEqualityFn } from "zustand/traditional"
import type { WatchlistTab } from "@/types/watchlists"

interface WatchlistsViewState {
  activeTab: WatchlistTab
  selectedWatchlistId: number | null
  selectedGroupId: number | null
  selectedTagName: string | null
  sourcesSearch: string
  sourcesPage: number
  sourcesPageSize: number
  jobsPage: number
  jobsPageSize: number
  selectedRunId: number | null
  runsJobFilter: number | null
  runsStatusFilter: string | null
  runsPage: number
  runsPageSize: number
  pollingActive: boolean
  selectedOutputId: number | null
  outputsJobFilter: number | null
  outputsRunFilter: number | null
  outputsPage: number
  outputsPageSize: number
  itemsSelectedSourceId: number | null
  itemsStatusFilter: string
  itemsSmartFilter: string
  itemsSearchQuery: string
  sourceFormOpen: boolean
  sourceFormEditId: number | null
  jobFormOpen: boolean
  jobFormEditId: number | null
  runDetailOpen: boolean
  outputPreviewOpen: boolean
}

interface WatchlistsViewActions {
  setActiveTab: (tab: WatchlistTab) => void
  setSelectedWatchlistId: (id: number | null) => void
  setSelectedGroupId: (id: number | null) => void
  setSelectedTagName: (name: string | null) => void
  setSourcesSearch: (search: string) => void
  setSourcesPage: (page: number) => void
  setSourcesPageSize: (size: number) => void
  setJobsPage: (page: number) => void
  setJobsPageSize: (size: number) => void
  setRunsJobFilter: (jobId: number | null) => void
  setRunsStatusFilter: (status: string | null) => void
  setRunsPage: (page: number) => void
  setRunsPageSize: (size: number) => void
  setPollingActive: (active: boolean) => void
  setOutputsJobFilter: (jobId: number | null) => void
  setOutputsRunFilter: (runId: number | null) => void
  setOutputsPage: (page: number) => void
  setOutputsPageSize: (size: number) => void
  setItemsSelectedSourceId: (sourceId: number | null) => void
  setItemsStatusFilter: (status: string) => void
  setItemsSmartFilter: (smartFilter: string) => void
  setItemsSearchQuery: (query: string) => void
  openSourceForm: (editId?: number | null) => void
  closeSourceForm: () => void
  openJobForm: (editId?: number | null) => void
  closeJobForm: () => void
  openRunDetail: (runId: number) => void
  closeRunDetail: () => void
  openOutputPreview: (outputId: number) => void
  closeOutputPreview: () => void
  resetStore: () => void
}

const initialPages = { sourcesPage: 1, jobsPage: 1, runsPage: 1, outputsPage: 1 }
const initialState: WatchlistsViewState = {
  ...initialPages,
  activeTab: "overview",
  selectedWatchlistId: null,
  selectedGroupId: null,
  selectedTagName: null,
  sourcesSearch: "",
  sourcesPageSize: 20,
  jobsPageSize: 20,
  selectedRunId: null,
  runsJobFilter: null,
  runsStatusFilter: null,
  runsPageSize: 20,
  pollingActive: false,
  selectedOutputId: null,
  outputsJobFilter: null,
  outputsRunFilter: null,
  outputsPageSize: 20,
  itemsSelectedSourceId: null,
  itemsStatusFilter: "all",
  itemsSmartFilter: "all",
  itemsSearchQuery: "",
  sourceFormOpen: false,
  sourceFormEditId: null,
  jobFormOpen: false,
  jobFormEditId: null,
  runDetailOpen: false,
  outputPreviewOpen: false
}

export const useWatchlistsStore = createWithEqualityFn<WatchlistsViewState & WatchlistsViewActions>()((set) => ({
  ...initialState,
  setActiveTab: (activeTab) => set({ activeTab }),
  setSelectedWatchlistId: (selectedWatchlistId) => set((state) =>
    state.selectedWatchlistId === selectedWatchlistId
      ? state
      : {
          ...initialPages,
          selectedWatchlistId,
          runsJobFilter: state.selectedWatchlistId == null ? state.runsJobFilter : null
        }
  ),
  setSelectedGroupId: (selectedGroupId) => set({ selectedGroupId, sourcesPage: 1 }),
  setSelectedTagName: (selectedTagName) => set({ selectedTagName, sourcesPage: 1 }),
  setSourcesSearch: (sourcesSearch) => set({ sourcesSearch, sourcesPage: 1 }),
  setSourcesPage: (sourcesPage) => set({ sourcesPage }),
  setSourcesPageSize: (sourcesPageSize) => set({ sourcesPageSize, sourcesPage: 1 }),
  setJobsPage: (jobsPage) => set({ jobsPage }),
  setJobsPageSize: (jobsPageSize) => set({ jobsPageSize, jobsPage: 1 }),
  setRunsJobFilter: (runsJobFilter) => set({ runsJobFilter, runsPage: 1 }),
  setRunsStatusFilter: (runsStatusFilter) => set({ runsStatusFilter, runsPage: 1 }),
  setRunsPage: (runsPage) => set({ runsPage }),
  setRunsPageSize: (runsPageSize) => set({ runsPageSize, runsPage: 1 }),
  setPollingActive: (pollingActive) => set({ pollingActive }),
  setOutputsJobFilter: (outputsJobFilter) => set({ outputsJobFilter, outputsPage: 1 }),
  setOutputsRunFilter: (outputsRunFilter) => set({ outputsRunFilter, outputsPage: 1 }),
  setOutputsPage: (outputsPage) => set({ outputsPage }),
  setOutputsPageSize: (outputsPageSize) => set({ outputsPageSize, outputsPage: 1 }),
  setItemsSelectedSourceId: (itemsSelectedSourceId) => set({ itemsSelectedSourceId }),
  setItemsStatusFilter: (itemsStatusFilter) => set({ itemsStatusFilter }),
  setItemsSmartFilter: (itemsSmartFilter) => set({ itemsSmartFilter }),
  setItemsSearchQuery: (itemsSearchQuery) => set({ itemsSearchQuery }),
  openSourceForm: (editId = null) => set({ sourceFormOpen: true, sourceFormEditId: editId }),
  closeSourceForm: () => set({ sourceFormOpen: false, sourceFormEditId: null }),
  openJobForm: (editId = null) => set({ jobFormOpen: true, jobFormEditId: editId }),
  closeJobForm: () => set({ jobFormOpen: false, jobFormEditId: null }),
  openRunDetail: (runId) => set({ runDetailOpen: true, selectedRunId: runId }),
  closeRunDetail: () => set({ runDetailOpen: false }),
  openOutputPreview: (outputId) => set({ outputPreviewOpen: true, selectedOutputId: outputId }),
  closeOutputPreview: () => set({ outputPreviewOpen: false }),
  resetStore: () => set(initialState)
}))

if (typeof window !== "undefined") {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ;(window as any).__tldw_useWatchlistsStore = useWatchlistsStore
}

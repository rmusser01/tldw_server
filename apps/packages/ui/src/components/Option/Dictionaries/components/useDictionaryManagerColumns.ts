import { useDictionaryTableColumns } from "./useDictionaryTableColumns"

type UseDictionaryManagerColumnsParams = {
  activeUpdateMap: Record<number, boolean>
  validationStatus: Record<
    number,
    {
      status: "valid" | "warning" | "error" | "loading" | "unknown"
      message?: string
    }
  >
  useCompactDictionaryActions: boolean
  handleDictionaryActiveToggle: (record: any, checked: boolean) => Promise<void>
  validateDictionary: (dictionaryId: number) => Promise<void>
  openChatContextFromDictionary: (chatRef: unknown, dictionary: unknown) => void
  openDictionaryEditModal: (record: any) => void
  openDictionaryEntriesPanel: (dictionaryId: number) => void
  openQuickAssignModal: (record: any) => void
  exportDictionaryAsJson: (record: any) => Promise<void>
  exportDictionaryAsMarkdown: (record: any) => Promise<void>
  openDictionaryStatsModal: (record: any) => Promise<void>
  openDictionaryVersionHistoryModal: (record: any) => Promise<void>
  duplicateDictionary: (record: any) => Promise<void>
  confirmAndDeleteDictionary: (record: any) => Promise<void>
}

export function useDictionaryManagerColumns({
  activeUpdateMap,
  validationStatus,
  useCompactDictionaryActions,
  handleDictionaryActiveToggle,
  validateDictionary,
  openChatContextFromDictionary,
  openDictionaryEditModal,
  openDictionaryEntriesPanel,
  openQuickAssignModal,
  exportDictionaryAsJson,
  exportDictionaryAsMarkdown,
  openDictionaryStatsModal,
  openDictionaryVersionHistoryModal,
  duplicateDictionary,
  confirmAndDeleteDictionary,
}: UseDictionaryManagerColumnsParams): any[] {
  return useDictionaryTableColumns({
    activeUpdateMap,
    validationStatus,
    useCompactDictionaryActions,
    onToggleActive: handleDictionaryActiveToggle,
    onValidateDictionary: validateDictionary,
    onOpenChatContext: openChatContextFromDictionary,
    onOpenEdit: openDictionaryEditModal,
    onOpenEntries: openDictionaryEntriesPanel,
    onOpenQuickAssign: openQuickAssignModal,
    onExportJson: (value) => {
      void exportDictionaryAsJson(value)
    },
    onExportMarkdown: (value) => {
      void exportDictionaryAsMarkdown(value)
    },
    onOpenStats: (value) => {
      void openDictionaryStatsModal(value)
    },
    onOpenVersions: (value) => {
      void openDictionaryVersionHistoryModal(value)
    },
    onDuplicate: (value) => {
      void duplicateDictionary(value)
    },
    onDelete: (value) => {
      void confirmAndDeleteDictionary(value)
    },
  })
}

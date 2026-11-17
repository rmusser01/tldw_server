import { useDictionaryManagerColumns } from "./useDictionaryManagerColumns"
import { useDictionaryValidationAndActivation } from "./useDictionaryValidationAndActivation"

type UseDictionaryManagerValidationColumnsParams = {
  queryClient: {
    invalidateQueries: (input: { queryKey: readonly unknown[] }) => Promise<unknown>
  }
  notification: {
    error: (config: { message: string; description?: string }) => void
    warning: (config: { message: string; description?: string }) => void
    success: (config: { message: string; description?: string }) => void
    info: (config: { message: string; description?: string }) => void
  }
  confirmDanger: (config: {
    title: string
    content: string
    okText: string
    cancelText: string
  }) => Promise<boolean>
  t: (key: string, fallbackOrOptions?: any) => string
  confirmDeactivationIfNeeded: (dictionary: any, nextIsActive: boolean) => Promise<boolean>
  useCompactDictionaryActions: boolean
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

export function useDictionaryManagerValidationColumns({
  queryClient,
  notification,
  confirmDanger,
  t,
  confirmDeactivationIfNeeded,
  useCompactDictionaryActions,
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
}: UseDictionaryManagerValidationColumnsParams) {
  const {
    validationStatus,
    activeUpdateMap,
    validateDictionary,
    handleDictionaryActiveToggle,
  } = useDictionaryValidationAndActivation({
    queryClient,
    notification,
    confirmDanger,
    t,
    confirmDeactivationIfNeeded,
  })

  return useDictionaryManagerColumns({
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
  })
}

import { useDictionariesAvailability } from "./useDictionariesAvailability"
import { useDictionaryManagerValidationColumns } from "./useDictionaryManagerValidationColumns"

type UseDictionaryManagerAvailabilityColumnsParams = {
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
  useMobileEntriesDrawer: boolean
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
  capsLoading: boolean
  capabilities: any
}

export function useDictionaryManagerAvailabilityColumns({
  queryClient,
  notification,
  confirmDanger,
  t,
  confirmDeactivationIfNeeded,
  useMobileEntriesDrawer,
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
  capsLoading,
  capabilities,
}: UseDictionaryManagerAvailabilityColumnsParams) {
  const {
    dictionariesUnsupported,
    dictionariesUnsupportedTitle,
    dictionariesUnsupportedDescription,
    dictionariesUnsupportedPrimaryActionLabel,
    openHealthDiagnostics,
  } = useDictionariesAvailability({
    capsLoading,
    capabilities,
    t,
  })

  const columns = useDictionaryManagerValidationColumns({
    queryClient,
    notification,
    confirmDanger,
    t,
    confirmDeactivationIfNeeded,
    useCompactDictionaryActions: useMobileEntriesDrawer,
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

  return {
    dictionariesUnsupported,
    dictionariesUnsupportedTitle,
    dictionariesUnsupportedDescription,
    dictionariesUnsupportedPrimaryActionLabel,
    openHealthDiagnostics,
    columns,
  }
}

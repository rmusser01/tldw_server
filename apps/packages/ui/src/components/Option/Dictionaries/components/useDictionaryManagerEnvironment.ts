import { useQueryClient } from "@tanstack/react-query"
import { Form } from "antd"
import React from "react"
import { useChatDraftOwner } from "@/hooks/useChatDraftOwner"
import { useTranslation } from "react-i18next"
import { useConfirmDanger } from "@/components/Common/confirm-danger"
import { useAntdNotification } from "@/hooks/useAntdNotification"
import { useMobile } from "@/hooks/useMediaQuery"
import { useServerCapabilities } from "@/hooks/useServerCapabilities"
import { useServerOnline } from "@/hooks/useServerOnline"
import { useDictionaryChatContextNavigation } from "./useDictionaryChatContextNavigation"
import { useDictionaryManagerFocusRefs } from "./useDictionaryManagerFocusRefs"

export function useDictionaryManagerEnvironment() {
  const { t } = useTranslation(["common", "option"])
  const isOnline = useServerOnline()
  const queryClient = useQueryClient()
  const notification = useAntdNotification()
  const [entryForm] = Form.useForm()
  const [statsFor, setStatsFor] = React.useState<any | null>(null)
  const [versionHistoryFor, setVersionHistoryFor] = React.useState<any | null>(null)
  const [dictionarySearch, setDictionarySearch] = React.useState("")
  const [dictionaryCategoryFilter, setDictionaryCategoryFilter] = React.useState("")
  const [dictionaryTagFilters, setDictionaryTagFilters] = React.useState<string[]>([])
  const {
    createDialogFocusReturnRef,
    editDialogFocusReturnRef,
    entriesDrawerFocusReturnRef,
    importDialogFocusReturnRef,
    quickAssignFocusReturnRef,
    statsDialogFocusReturnRef,
    versionHistoryDialogFocusReturnRef,
  } = useDictionaryManagerFocusRefs()
  const { capabilities, loading: capsLoading } = useServerCapabilities()
  const confirmDanger = useConfirmDanger()
  const owner = useChatDraftOwner(React.useCallback(() => undefined, []))
  const { openChatContextFromDictionary } = useDictionaryChatContextNavigation(owner, notification)
  const useMobileEntriesDrawer = useMobile()

  return {
    owner,
    t,
    isOnline,
    queryClient,
    notification,
    entryForm,
    statsFor,
    setStatsFor,
    versionHistoryFor,
    setVersionHistoryFor,
    dictionarySearch,
    setDictionarySearch,
    dictionaryCategoryFilter,
    setDictionaryCategoryFilter,
    dictionaryTagFilters,
    setDictionaryTagFilters,
    createDialogFocusReturnRef,
    editDialogFocusReturnRef,
    entriesDrawerFocusReturnRef,
    importDialogFocusReturnRef,
    quickAssignFocusReturnRef,
    statsDialogFocusReturnRef,
    versionHistoryDialogFocusReturnRef,
    capabilities,
    capsLoading,
    confirmDanger,
    openChatContextFromDictionary,
    useMobileEntriesDrawer,
  }
}

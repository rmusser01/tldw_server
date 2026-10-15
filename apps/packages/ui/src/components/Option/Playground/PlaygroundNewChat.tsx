import { PencilIcon } from "lucide-react"
import { useClearChat } from "@/hooks/chat/useClearChat"
import { useTranslation } from 'react-i18next';

export const PlaygroundNewChat = () => {
  const clearChat = useClearChat()
  const { t } = useTranslation('optionChat')

  const handleClick = () => {
    clearChat()
  }

  return (
    <button
      onClick={handleClick}
      title={t('newChat')}
      className="flex w-full rounded-md border border-border bg-transparent p-2 text-text-muted transition-colors hover:bg-surface2 hover:text-text focus:outline-none focus-visible:ring-2 focus-visible:ring-focus">
      <PencilIcon className="mx-3 h-5 w-5" aria-hidden="true" />
      <span className="inline-flex text-sm font-semibold">
        {t('newChat')}
      </span>
    </button>
  )
}

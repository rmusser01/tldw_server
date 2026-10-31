import React from "react"
import { Users } from "lucide-react"
import { useTranslation } from "react-i18next"
import {
  useBuddyManagementStore,
  type BuddyManagementTarget
} from "@/store/buddy-management"

export const BuddyManagementButton = ({
  target,
  onConversationSettings
}: {
  target?: BuddyManagementTarget | null
  onConversationSettings?: () => void
}) => {
  const { t } = useTranslation("sidepanel")
  return (
    <button
      type="button"
      onClick={() =>
        useBuddyManagementStore.getState().show(target, onConversationSettings)
      }
      className="inline-flex items-center gap-1 rounded-md px-2 py-2 text-sm font-medium text-text-muted hover:bg-surface2 hover:text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-primary"
    >
      <Users size={16} aria-hidden="true" />
      {t("buddyManagement.entry", { defaultValue: "Buddy & Persona" })}
    </button>
  )
}

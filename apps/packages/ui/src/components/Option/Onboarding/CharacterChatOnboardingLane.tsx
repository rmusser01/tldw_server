import React from "react"
import { MessageCircle, Settings2, Upload, UserPlus } from "lucide-react"
import type { LucideIcon } from "lucide-react"

import { cn } from "@/libs/utils"

type CharacterChatOnboardingLaneProps = {
  className?: string
  onCreateCharacter: () => void
  onImportCharacter: () => void
  onChooseModel: () => void
  onStartCharacterChat: () => void
}

type CharacterChatAction = {
  label: string
  description: string
  icon: LucideIcon
  onClick: () => void
}

export const CharacterChatOnboardingLane: React.FC<
  CharacterChatOnboardingLaneProps
> = ({
  className,
  onCreateCharacter,
  onImportCharacter,
  onChooseModel,
  onStartCharacterChat
}) => {
  const actions: CharacterChatAction[] = [
    {
      label: "Create character",
      description: "Start with a reusable persona.",
      icon: UserPlus,
      onClick: onCreateCharacter
    },
    {
      label: "Import character",
      description: "Bring in a card you already use.",
      icon: Upload,
      onClick: onImportCharacter
    },
    {
      label: "Choose model",
      description: "Confirm the model for character replies.",
      icon: Settings2,
      onClick: onChooseModel
    },
    {
      label: "Start character chat",
      description: "Open chat with character context.",
      icon: MessageCircle,
      onClick: onStartCharacterChat
    }
  ]

  return (
    <section
      aria-label="Character chat onboarding actions"
      data-testid="character-chat-onboarding-lane"
      className={cn(
        "rounded-lg border border-primary/20 bg-primary/5 p-3",
        className
      )}
    >
      <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
        {actions.map((action) => {
          const Icon = action.icon
          return (
            <button
              key={action.label}
              type="button"
              aria-label={action.label}
              onClick={action.onClick}
              className="flex min-h-20 items-start gap-2 rounded-md border border-border bg-surface px-3 py-3 text-left transition-colors hover:border-primary/40 hover:bg-surface2"
            >
              <Icon
                className="mt-0.5 size-4 shrink-0 text-primary"
                aria-hidden={true}
              />
              <span className="min-w-0">
                <span className="block text-sm font-medium text-text">
                  {action.label}
                </span>
                <span className="mt-1 block text-xs leading-5 text-text-muted">
                  {action.description}
                </span>
              </span>
            </button>
          )
        })}
      </div>
    </section>
  )
}

export default CharacterChatOnboardingLane

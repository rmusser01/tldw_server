import { useMemo } from "react"
import { MISSION_CARDS, type MissionCard } from "../mission-cards"
import { useMilestoneStore, type MilestoneId } from "@/store/milestones"
import { useConnectionStore } from "@/store/connection"

export type ResolvedMissionCard = MissionCard & {
  isCompleted: boolean
}

export type UseMissionCardsResult = {
  gettingStartedCards: ResolvedMissionCard[]
  whatsNextCard: ResolvedMissionCard | null
  completedCount: number
  totalCount: number
  allComplete: boolean
}

export function useMissionCards(): UseMissionCardsResult {
  const userPersona = useConnectionStore((s) => s.state.userPersona)
  const completedMilestones = useMilestoneStore((s) => s.completedMilestones)

  return useMemo(() => {
    // Use completedMilestones directly instead of isMilestoneCompleted
    const isCompleted = (id: MilestoneId) => completedMilestones[id] != null

    // 1. Filter by persona
    const personaCards = MISSION_CARDS.filter((card) => {
      if (card.persona === "all") return true
      return card.persona.includes(userPersona)
    })

    // 2. Filter by prerequisite milestones (all must be completed)
    const availableCards = personaCards.filter((card) =>
      card.prerequisiteMilestones.every((m) => isCompleted(m))
    )

    // 3. Resolve completion status
    const resolved: ResolvedMissionCard[] = availableCards.map((card) => ({
      ...card,
      isCompleted: card.linkedMilestone ? isCompleted(card.linkedMilestone) : false
    }))

    // 4. Sort by priority
    resolved.sort((a, b) => a.priority - b.priority)

    // 5. Split into categories
    const gettingStartedCards = resolved.filter((c) => c.category === "getting-started")
    const allCards = resolved

    // 6. Find "what's next" — first non-completed card
    const whatsNextCard = allCards.find((c) => !c.isCompleted) ?? null

    const completedCount = allCards.filter((c) => c.isCompleted).length
    const totalCount = allCards.length

    return {
      gettingStartedCards,
      whatsNextCard,
      completedCount,
      totalCount,
      allComplete: completedCount === totalCount && totalCount > 0
    }
  }, [userPersona, completedMilestones])
}

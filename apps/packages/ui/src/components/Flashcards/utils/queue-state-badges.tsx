import React from "react"

import { Badge, type BadgeVariant } from "@/components/ui/primitives"
import type { Flashcard } from "@/services/flashcards"

const QUEUE_STATE_META: Record<
  Flashcard["queue_state"],
  {
    label: string
    variant: BadgeVariant
  }
> = {
  new: {
    label: "New",
    variant: "info"
  },
  learning: {
    label: "Learning",
    variant: "warning"
  },
  review: {
    label: "Review",
    variant: "success"
  },
  relearning: {
    label: "Relearning",
    variant: "warning"
  },
  suspended: {
    label: "Suspended",
    variant: "danger"
  }
}

const coerceQueueState = (
  queueState: Flashcard["queue_state"] | null | undefined
): Flashcard["queue_state"] => {
  if (queueState && queueState in QUEUE_STATE_META) {
    return queueState
  }
  return "review"
}

export const formatFlashcardQueueStateLabel = (
  queueState: Flashcard["queue_state"] | null | undefined,
  suspendedReason?: Flashcard["suspended_reason"] | null
): string => {
  const normalizedQueueState = coerceQueueState(queueState)

  if (normalizedQueueState !== "suspended") {
    return QUEUE_STATE_META[normalizedQueueState].label
  }

  if (suspendedReason === "leech") {
    return "Suspended (Leech)"
  }

  if (suspendedReason === "manual") {
    return "Suspended (Manual)"
  }

  return QUEUE_STATE_META[queueState].label
}

export interface FlashcardQueueStateBadgeProps {
  card: Pick<Flashcard, "queue_state" | "suspended_reason">
  testId?: string
}

export const FlashcardQueueStateBadge: React.FC<FlashcardQueueStateBadgeProps> = ({
  card,
  testId
}) => {
  const normalizedQueueState = coerceQueueState(card.queue_state)

  return (
    <Badge
      variant={QUEUE_STATE_META[normalizedQueueState].variant}
      size="sm"
      dot
      data-testid={testId}
    >
      {formatFlashcardQueueStateLabel(normalizedQueueState, card.suspended_reason)}
    </Badge>
  )
}

export default FlashcardQueueStateBadge

import type {
  ConversationContextComposition,
  ConversationContextPiece,
  ConversationContextPieceStatus,
  ConversationContextSource
} from "@/types/conversation-context"
import type { ConversationContextCompositionStatus } from "@/hooks/chat/useConversationContextComposition"

type KindSummary = {
  total: number
  active: number
  matched: number
  configured: number
  blocked: number
  missing: number
  skipped: number
  inherited: number
  explicit: number
}

export type ConversationContextSummary = {
  characterConfigured: boolean
  worldbooks: KindSummary
  dictionaries: KindSummary
  warnings: number
}

const emptyKindSummary = (): KindSummary => ({
  total: 0,
  active: 0,
  matched: 0,
  configured: 0,
  blocked: 0,
  missing: 0,
  skipped: 0,
  inherited: 0,
  explicit: 0
})

const incrementStatus = (
  summary: KindSummary,
  status: ConversationContextPieceStatus
) => {
  summary[status] += 1
}

const countSource = (summary: KindSummary, piece: ConversationContextPiece) => {
  if (piece.source === "explicit_chat" || piece.source === "request") {
    summary.explicit += 1
  }
  if (
    piece.source === "character_inherited" ||
    piece.source === "character_start"
  ) {
    summary.inherited += 1
  }
}

export const summarizeConversationContextPieces = (
  composition?: ConversationContextComposition | null
): ConversationContextSummary => {
  const summary: ConversationContextSummary = {
    characterConfigured: false,
    worldbooks: emptyKindSummary(),
    dictionaries: emptyKindSummary(),
    warnings: composition?.warnings.length ?? 0
  }

  for (const piece of composition?.pieces ?? []) {
    summary.warnings += piece.warnings?.length ?? 0

    if (piece.kind === "character") {
      summary.characterConfigured = piece.status !== "missing"
      continue
    }

    const target =
      piece.kind === "worldbook"
        ? summary.worldbooks
        : piece.kind === "dictionary"
          ? summary.dictionaries
          : null
    if (!target) continue

    target.total += 1
    incrementStatus(target, piece.status)
    countSource(target, piece)
  }

  return summary
}

export const formatContextSourceLabel = (
  source: ConversationContextSource
): string => {
  switch (source) {
    case "explicit_chat":
      return "Chat"
    case "workspace":
      return "Workspace"
    case "character_start":
      return "Character start"
    case "character_inherited":
      return "Character inherited"
    case "global":
      return "Global"
    case "request":
    default:
      return "Request"
  }
}

export type ResolvedContextReadiness = {
  readiness: ConversationContextComposition["readiness"]
  label: string
  tone: "ready" | "partial" | "blocked" | "loading"
}

export const resolveContextReadiness = ({
  composition,
  status
}: {
  composition?: ConversationContextComposition | null
  status: ConversationContextCompositionStatus
}): ResolvedContextReadiness => {
  if (status === "loading") {
    return {
      readiness: "partial",
      label: "Composing",
      tone: "loading"
    }
  }

  if (status === "error") {
    return {
      readiness: "blocked",
      label: "Needs attention",
      tone: "blocked"
    }
  }

  const readiness = composition?.readiness ?? "ready"
  if (readiness === "blocked") {
    return { readiness, label: "Blocked", tone: "blocked" }
  }
  if (readiness === "partial") {
    return { readiness, label: "Partial", tone: "partial" }
  }
  return { readiness, label: "Ready", tone: "ready" }
}

export type QuickChatWorkflowGuide = {
  id: string
  title: string
  question: string
  answer: string
  route: string
  routeLabel: string
  tags: string[]
}

export type QuickChatWorkflowCitationHint = {
  title?: string
  source?: string
}

export type QuickChatWorkflowRecommendation = {
  id: string
  title: string
  route: string
  routeLabel: string
  reason: string
  isCurrentRoute: boolean
}

export const QUICK_CHAT_WORKFLOW_GUIDES_STORAGE_KEY =
  "quickChatWorkflowGuidesV1"

const RESEARCH_STUDIO_ROUTE = "/research-studio"

const QUICK_CHAT_ROUTE_ALIASES: Record<string, string> = {
  "/workspace-playground": RESEARCH_STUDIO_ROUTE,
  "/workspace-studio": RESEARCH_STUDIO_ROUTE
}

const canonicalizeQuickChatRoute = (route: string): string =>
  QUICK_CHAT_ROUTE_ALIASES[route] ?? route

export const QUICK_CHAT_WORKFLOW_GUIDES: QuickChatWorkflowGuide[] = [
  {
    id: "ingest-summarize-media",
    title: "Ingest + summarize a source",
    question: "How do I ingest a URL or file and then summarize it quickly?",
    answer:
      "Use Media to ingest and process content first, then jump to Knowledge QA to ask focused questions and get sourced answers.",
    route: "/media",
    routeLabel: "Media",
    tags: ["ingest", "summary", "media", "qa"]
  },
  {
    id: "find-tools-for-goal",
    title: "Find which page fits my goal",
    question: "I know my goal, but not which page has the right tools. Where should I start?",
    answer:
      "Start in Research Studio for guided multi-tool workflows, then open the specialized page it suggests (Media, Knowledge, Characters, or Evaluations).",
    route: RESEARCH_STUDIO_ROUTE,
    routeLabel: "Research Studio",
    tags: ["workflow", "onboarding", "discovery", "navigation"]
  },
  {
    id: "docs-research-qa",
    title: "Ask questions over docs",
    question: "How can I ask questions over my project documentation with citations?",
    answer:
      "Use Knowledge to run RAG over indexed content. Tune retrieval settings if needed, and keep citations enabled for traceability.",
    route: "/knowledge",
    routeLabel: "Knowledge",
    tags: ["docs", "rag", "citations", "research"]
  },
  {
    id: "character-assisted-chat",
    title: "Use a character assistant",
    question: "How do I create a reusable assistant persona for repeated tasks?",
    answer:
      "Create a character in Characters, set system prompt and greeting, then start chats with that persona from the character actions.",
    route: "/characters",
    routeLabel: "Characters",
    tags: ["characters", "persona", "assistant", "chat"]
  },
  {
    id: "world-book-linking",
    title: "Attach lore/reference data",
    question: "How do I connect reusable reference lore to a character?",
    answer:
      "Create entries in World Books and attach them to characters so chats can pull consistent context from shared references.",
    route: "/world-books",
    routeLabel: "World Books",
    tags: ["world books", "lore", "reference", "characters"]
  },
  {
    id: "prompt-iteration",
    title: "Iterate on prompts",
    question: "Where can I manage prompt variants and test prompt improvements?",
    answer:
      "Use Prompts for library management and Prompt Studio flows. Keep baseline and variant prompts side by side while iterating.",
    route: "/prompts",
    routeLabel: "Prompts",
    tags: ["prompts", "prompt studio", "iteration", "testing"]
  },
  {
    id: "evaluation-benchmark",
    title: "Benchmark model quality",
    question: "How can I compare model output quality and reliability?",
    answer:
      "Use Evaluations to run structured quality checks and compare metrics across providers or prompt variants.",
    route: "/evaluations",
    routeLabel: "Evaluations",
    tags: ["evaluation", "benchmark", "quality", "models"]
  },
  {
    id: "note-taking-from-research",
    title: "Capture notes from findings",
    question: "How do I capture and organize findings while researching?",
    answer:
      "Use Notes for persistent capture and categorization while keeping retrieval in Knowledge for source-grounded follow-ups.",
    route: "/notes",
    routeLabel: "Notes",
    tags: ["notes", "research", "organization", "capture"]
  },
  {
    id: "flashcards-from-content",
    title: "Create study material",
    question: "How do I turn content into flashcards or quizzes?",
    answer:
      "Use Flashcards to generate and review cards, then use Quiz for assessment workflows and retention checks.",
    route: "/flashcards",
    routeLabel: "Flashcards",
    tags: ["flashcards", "quiz", "study", "learning"]
  },
  {
    id: "platform-config-health",
    title: "Fix setup/connectivity issues",
    question: "Where should I troubleshoot server or model configuration problems?",
    answer:
      "Open Settings and Health first to verify connection/auth/model availability, then return to your workflow page.",
    route: "/settings/health",
    routeLabel: "Health & Diagnostics",
    tags: ["setup", "diagnostics", "connection", "settings"]
  }
]

const normalizeGuideRoute = (route: unknown): string | null => {
  if (typeof route !== "string") return null
  const trimmed = route.trim()
  if (!trimmed) return null
  const normalized = trimmed.startsWith("/") ? trimmed : `/${trimmed}`
  return canonicalizeQuickChatRoute(normalized)
}

const deriveRouteLabel = (route: string): string => {
  const segments = route
    .split("/")
    .map((segment) => segment.trim())
    .filter((segment) => segment.length > 0)
  if (segments.length === 0) {
    return "Home"
  }
  return segments
    .map((segment) =>
      segment
        .split("-")
        .map((part) => part.trim())
        .filter((part) => part.length > 0)
        .map((part) => part[0].toUpperCase() + part.slice(1))
        .join(" ")
    )
    .join(" / ")
}

const normalizeGuideTags = (value: unknown): string[] => {
  if (Array.isArray(value)) {
    return value
      .map((item) => (typeof item === "string" ? item.trim() : ""))
      .filter((item) => item.length > 0)
  }
  if (typeof value === "string") {
    return value
      .split(",")
      .map((item) => item.trim())
      .filter((item) => item.length > 0)
  }
  return []
}

const sanitizeGuideCandidate = (
  value: unknown,
  index: number
): QuickChatWorkflowGuide | null => {
  if (!value || typeof value !== "object") return null
  const candidate = value as Record<string, unknown>
  const title = typeof candidate.title === "string" ? candidate.title.trim() : ""
  const question =
    typeof candidate.question === "string" ? candidate.question.trim() : ""
  const answer = typeof candidate.answer === "string" ? candidate.answer.trim() : ""
  const route = normalizeGuideRoute(candidate.route)
  const routeLabelInput =
    typeof candidate.routeLabel === "string" ? candidate.routeLabel.trim() : ""

  if (!title || !question || !answer || !route) {
    return null
  }
  const routeLabel = routeLabelInput || deriveRouteLabel(route)

  const rawId = typeof candidate.id === "string" ? candidate.id.trim() : ""
  const id = rawId || `custom-guide-${index + 1}`
  const tags = normalizeGuideTags(candidate.tags)

  return {
    id,
    title,
    question,
    answer,
    route,
    routeLabel,
    tags
  }
}

export const resolveQuickChatWorkflowGuides = (
  rawGuides: unknown,
  fallback: QuickChatWorkflowGuide[] = QUICK_CHAT_WORKFLOW_GUIDES
): QuickChatWorkflowGuide[] => {
  if (!Array.isArray(rawGuides)) {
    return fallback
  }

  const usedIds = new Set<string>()
  const sanitized = rawGuides
    .map((guide, index) => sanitizeGuideCandidate(guide, index))
    .filter((guide): guide is QuickChatWorkflowGuide => guide !== null)
    .map((guide, index) => {
      let nextId = guide.id
      if (usedIds.has(nextId)) {
        nextId = `${guide.id}-${index + 1}`
      }
      usedIds.add(nextId)
      return { ...guide, id: nextId }
    })

  return sanitized.length > 0 ? sanitized : fallback
}

export const parseQuickChatWorkflowGuidesJson = (
  draft: string
): { guides: QuickChatWorkflowGuide[] | null; error?: string } => {
  const trimmed = draft.trim()
  if (!trimmed) {
    return {
      guides: null,
      error: "Guide JSON cannot be empty."
    }
  }

  try {
    const parsed = JSON.parse(trimmed) as unknown
    if (!Array.isArray(parsed)) {
      return {
        guides: null,
        error: "Guide JSON must be an array."
      }
    }
    const resolved = resolveQuickChatWorkflowGuides(parsed, [])
    if (resolved.length === 0) {
      return {
        guides: null,
        error:
          "No valid guide cards were found. Each card requires title, question, answer, and route."
      }
    }
    return { guides: resolved }
  } catch {
    return {
      guides: null,
      error: "Guide JSON is not valid."
    }
  }
}

export const stringifyQuickChatWorkflowGuides = (
  guides: QuickChatWorkflowGuide[]
): string => JSON.stringify(guides, null, 2)

const normalizeForSearch = (value: string): string =>
  value.toLowerCase().trim()

const tokenizeSearchText = (value: string): string[] =>
  normalizeForSearch(value)
    .split(/[^a-z0-9]+/g)
    .filter((token) => token.length > 1)

export const normalizeQuickChatRoutePath = (
  route: string | null | undefined
): string | null => {
  if (typeof route !== "string") return null
  let candidate = route.trim()
  if (!candidate) return null

  if (/^https?:\/\//i.test(candidate)) {
    try {
      const parsed = new URL(candidate)
      candidate = parsed.hash ? parsed.hash.slice(1) : parsed.pathname
    } catch {
      // Fall through and keep raw value.
    }
  }

  const optionsHashIndex = candidate.indexOf("options.html#")
  if (optionsHashIndex >= 0) {
    candidate = candidate.slice(optionsHashIndex + "options.html#".length)
  }

  if (candidate.startsWith("#")) {
    candidate = candidate.slice(1)
  }
  if (!candidate) return null

  const hashless = candidate.split("#")[0] || ""
  const queryless = hashless.split("?")[0] || ""
  const normalizedBase = queryless.trim()
  if (!normalizedBase) return null

  let normalized = normalizedBase.startsWith("/")
    ? normalizedBase
    : `/${normalizedBase}`
  normalized = normalized.replace(/\/{2,}/g, "/")
  if (normalized.length > 1 && normalized.endsWith("/")) {
    normalized = normalized.slice(0, -1)
  }
  return normalized ? canonicalizeQuickChatRoute(normalized) : null
}

export const filterQuickChatWorkflowGuides = (
  query: string,
  guides: QuickChatWorkflowGuide[] = QUICK_CHAT_WORKFLOW_GUIDES
): QuickChatWorkflowGuide[] => {
  const normalized = normalizeForSearch(query)
  if (!normalized) return guides

  return guides.filter((guide) => {
    const haystack = [
      guide.title,
      guide.question,
      guide.answer,
      guide.routeLabel,
      ...guide.tags
    ]
      .join(" ")
      .toLowerCase()
    return haystack.includes(normalized)
  })
}

type RecommendQuickChatWorkflowGuidesOptions = {
  query?: string
  answer?: string
  citations?: QuickChatWorkflowCitationHint[]
  currentRoute?: string | null
  maxResults?: number
  guides?: QuickChatWorkflowGuide[]
}

const toRecommendationReason = ({
  guide,
  matchedTags,
  matchedRouteText,
  isCurrentRoute
}: {
  guide: QuickChatWorkflowGuide
  matchedTags: string[]
  matchedRouteText: boolean
  isCurrentRoute: boolean
}): string => {
  if (matchedTags.length > 0) {
    return `Matched topic: ${matchedTags.slice(0, 2).join(", ")}.`
  }
  if (matchedRouteText) {
    return `Directly related to ${guide.routeLabel}.`
  }
  if (isCurrentRoute) {
    return "Matches your current page context."
  }
  return "Related to your question."
}

export const recommendQuickChatWorkflowGuides = ({
  query = "",
  answer = "",
  citations = [],
  currentRoute = null,
  maxResults = 3,
  guides = QUICK_CHAT_WORKFLOW_GUIDES
}: RecommendQuickChatWorkflowGuidesOptions): QuickChatWorkflowRecommendation[] => {
  const normalizedCurrentRoute = normalizeQuickChatRoutePath(currentRoute)
  const citationText = citations
    .map((citation) => [citation.title || "", citation.source || ""].join(" "))
    .join(" ")
  const searchText = [query, answer, citationText].join(" ").trim()
  if (!searchText) return []

  const normalizedSearch = normalizeForSearch(searchText)
  const queryTokens = new Set(tokenizeSearchText(searchText))
  const safeMax = Math.max(1, Math.min(8, Math.floor(maxResults)))

  return guides
    .map((guide) => {
      const guideCorpus = [
        guide.title,
        guide.question,
        guide.answer,
        guide.routeLabel,
        guide.route,
        ...guide.tags
      ].join(" ")
      const guideTokens = new Set(tokenizeSearchText(guideCorpus))
      let overlapScore = 0
      guideTokens.forEach((token) => {
        if (queryTokens.has(token)) overlapScore += 1
      })

      const matchedTags = guide.tags.filter((tag) =>
        normalizedSearch.includes(normalizeForSearch(tag))
      )
      const normalizedRouteLabel = normalizeForSearch(guide.routeLabel)
      const normalizedRoutePath = normalizeForSearch(guide.route)
      const routeKeyword = normalizeForSearch(guide.route.replace(/\//g, " "))
      const matchedRouteText =
        normalizedSearch.includes(normalizedRouteLabel) ||
        normalizedSearch.includes(normalizedRoutePath) ||
        normalizedSearch.includes(routeKeyword)

      const isCurrentRoute =
        normalizedCurrentRoute !== null && normalizedCurrentRoute === guide.route
      const score =
        overlapScore +
        matchedTags.length * 3 +
        (matchedRouteText ? 2 : 0) +
        (isCurrentRoute ? 0.5 : 0)

      return {
        id: guide.id,
        title: guide.title,
        route: guide.route,
        routeLabel: guide.routeLabel,
        reason: toRecommendationReason({
          guide,
          matchedTags,
          matchedRouteText,
          isCurrentRoute
        }),
        isCurrentRoute,
        score
      }
    })
    .filter((recommendation) => recommendation.score > 0)
    .sort((left, right) => {
      if (right.score !== left.score) {
        return right.score - left.score
      }
      if (left.isCurrentRoute !== right.isCurrentRoute) {
        return left.isCurrentRoute ? -1 : 1
      }
      return left.title.localeCompare(right.title)
    })
    .slice(0, safeMax)
    .map(({ score: _score, ...recommendation }) => recommendation)
}

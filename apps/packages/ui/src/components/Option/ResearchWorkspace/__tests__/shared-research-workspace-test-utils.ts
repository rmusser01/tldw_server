import i18next from "i18next"
import playgroundEn from "@/assets/locale/en/playground.json"
import type {
  SharedChatResponse,
  SharedSourcePage,
  SharedSourcePreview,
  SharedWorkspaceBootstrap
} from "@/types/shared-workspace"

export const createSharedWorkspaceTestI18n = async () => {
  const instance = i18next.createInstance()
  await instance.init({
    lng: "en",
    fallbackLng: false,
    resources: { en: { playground: playgroundEn, common: {} } },
    ns: ["playground", "common"],
    defaultNS: "playground",
    interpolation: { escapeValue: false }
  })
  return instance
}

export const sourcePage: SharedSourcePage = {
  items: [
    {
      source_id: "source-ready",
      title: "Queryable report",
      source_type: "document",
      origin_url: "https://example.test/report",
      origin_host: "example.test",
      state: "ready",
      reason_code: null,
      citation_ready: true,
      retrieval_ready: true,
      position: 1,
      added_at: "2026-08-21T10:00:00Z"
    },
    {
      source_id: "source-processing",
      title: "Processing interview",
      source_type: "audio",
      origin_url: null,
      origin_host: null,
      state: "processing",
      reason_code: "transcription_pending",
      citation_ready: false,
      retrieval_ready: false,
      position: 2,
      added_at: "2026-08-21T11:00:00Z"
    }
  ],
  pagination: { offset: 0, limit: 50, total: 2, has_more: false },
  summary: { total: 2, queryable: 1, processing: 1, failed: 0 },
  partial_errors: []
}

export const buildBootstrap = (
  overrides: Partial<SharedWorkspaceBootstrap> = {}
): SharedWorkspaceBootstrap => ({
  schema_version: 1,
  generated_at: "2026-08-21T12:00:00Z",
  share: {
    share_id: 42,
    access_level: "view_chat_add",
    allow_clone: false,
    owner_display_name: "Avery Owner",
    shared_at: "2026-08-20T12:00:00Z"
  },
  workspace: {
    workspace_id: "workspace-shared",
    name: "Election evidence review",
    description: "Shared evidence for recipient review."
  },
  allowed_actions: {
    inspect_sources: { allowed: true, reason_code: null },
    ask_grounded_questions: { allowed: true, reason_code: null },
    add_sources: { allowed: false, reason_code: "recipient_mutation_disabled" },
    edit_workspace: { allowed: false, reason_code: "recipient_mutation_disabled" },
    clone_workspace: { allowed: false, reason_code: "owner_disabled" }
  },
  generation_default: {
    provider: "anthropic",
    model: "claude-shared",
    ready: true,
    reason_code: null
  },
  source_summary: sourcePage.summary,
  sources: {
    items: sourcePage.items,
    pagination: sourcePage.pagination
  },
  conversation: {
    conversation_id: "conversation-1",
    messages: [
      {
        message_id: "message-existing",
        role: "assistant",
        content: "Existing **grounded** answer.",
        created_at: "2026-08-21T11:30:00Z",
        citations: []
      }
    ],
    next_before: "older-cursor"
  },
  partial_errors: [],
  ...overrides
})

export const preview: SharedSourcePreview = {
  source_id: "source-ready",
  title: "Queryable report",
  source_type: "document",
  origin_url: "https://example.test/report",
  origin_host: "example.test",
  state: "ready",
  reason_code: null,
  content_available: true,
  preview_mode: "content_excerpt",
  unavailable_reason: null,
  text_preview: "Focused source preview",
  text_total_chars: 220,
  text_truncated: false,
  snippets: [
    {
      kind: "chunk",
      text: "Chunk seven evidence",
      start_char: 40,
      end_char: 61,
      chunk_index: 7
    }
  ],
  generated_at: "2026-08-21T12:10:00Z"
}

export const chatResponse: SharedChatResponse = {
  schema_version: 1,
  request_id: "00000000-0000-4000-8000-000000000042",
  conversation_id: "conversation-1",
  turn: {
    user_message: {
      message_id: "message-user-new",
      role: "user",
      content: "What does the report conclude?",
      created_at: "2026-08-21T12:20:00Z"
    },
    assistant_message: {
      message_id: "message-assistant-new",
      role: "assistant",
      content: "The report supports **one conclusion**.",
      created_at: "2026-08-21T12:20:01Z"
    }
  },
  citations: [
    {
      citation_id: "citation-1",
      source_id: "source-ready",
      source_title: "Queryable report",
      locator: { chunk: 7, start_char: 40, end_char: 61 },
      quote: "Evidence from the report.",
      score: 0.91
    }
  ],
  generation: { provider: "anthropic", model: "claude-shared" },
  source_scope: { mode: "all", effective_source_count: 1 },
  replay: { replayed: false }
}

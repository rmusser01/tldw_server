import { tldwClient } from "@/services/tldw/TldwApiClient"
import type { ServerChatSummary } from "@/services/tldw/TldwApiClient"
import { toAllowedPath } from "@/services/tldw/path-utils"
import type {
  PersonaVisualAsset,
  PersonaVisualManifest
} from "@/types/persona-visuals"

export type BuddyConversationSummary = Omit<ServerChatSummary, "created_at"> & {
  created_at?: string | null
}

export type BuddyProfile = {
  id: string
  name: string
  optional_persona_id: string | null
  optional_persona_available: boolean
  display_mode: "dynamic" | "static"
  version: number
  manifest: PersonaVisualManifest
  assets: (Omit<PersonaVisualAsset, "url"> & { content_url: string })[]
  attribution: Record<string, unknown>
}
export type BuddyAttachment = {
  buddy_id: string
  scope_type: "conversation" | "workspace"
  scope_id: string
}
export type BuddyAttachmentState = {
  client_slot: string
  version: number
  attachment: BuddyAttachment | null
  unavailable_reason?: "target_unavailable" | "buddy_unavailable" | null
  target?: { title: string; workspace_id: string | null } | null
}
export type BuddySource =
  | { kind: "starter"; starter_id: string }
  | { kind: "persona_pack"; persona_id: string; pack_id: string }

export async function buddyRequest<T>(
  path: string,
  method = "GET",
  body?: unknown
): Promise<T> {
  const response = await tldwClient.fetchWithAuth(
    toAllowedPath(`/api/v1/buddies${path}`),
    {
      method,
      ...(body === undefined
        ? {}
        : {
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body)
          })
    }
  )
  const payload = await response.json().catch(() => null)
  if (!response.ok) {
    const detail = payload?.detail
    const message = typeof detail === "string" ? detail : detail?.message
    throw new Error(
      message ||
        `Buddy request failed (${response.status}). Refresh and try again.`
    )
  }
  return payload as T
}
export const BUDDY_PAGE_SIZE = 100
type BuddyPage = { limit?: number; offset?: number }
const pageQuery = ({ limit = BUDDY_PAGE_SIZE, offset = 0 }: BuddyPage = {}) =>
  `limit=${Math.min(BUDDY_PAGE_SIZE, Math.max(1, Math.floor(limit)))}&offset=${Math.max(0, Math.floor(offset))}`
export const listBuddies = (page?: BuddyPage) =>
  buddyRequest<{ buddies: BuddyProfile[] }>(`?${pageQuery(page)}`)
export const getBuddy = (id: string) =>
  buddyRequest<BuddyProfile>(`/${encodeURIComponent(id)}`)
export const createBuddy = (body: {
  name: string
  source: BuddySource
  optional_persona_id?: string | null
  display_mode?: "dynamic" | "static"
}) => buddyRequest<BuddyProfile>("", "POST", body)
export const updateBuddy = (
  id: string,
  body: {
    expected_version: number
    name?: string
    optional_persona_id?: string | null
    display_mode?: "dynamic" | "static"
  }
) => buddyRequest<BuddyProfile>(`/${encodeURIComponent(id)}`, "PATCH", body)
export const getBuddyAttachment = () =>
  buddyRequest<BuddyAttachmentState>("/attachment?client_slot=default")
export const putBuddyAttachment = (
  body: BuddyAttachment & { expected_version: number }
) =>
  buddyRequest<BuddyAttachmentState>(
    "/attachment?client_slot=default",
    "PUT",
    body
  )
export const detachBuddy = (version: number) =>
  buddyRequest<BuddyAttachmentState>(
    `/attachment?client_slot=default&expected_version=${version}`,
    "DELETE"
  )
export const listBuddyConversations = (page?: BuddyPage) =>
  buddyRequest<{ conversations: BuddyConversationSummary[] }>(
    `/attachment/conversations?client_slot=default&${pageQuery(page)}`
  )
export const resolveBuddyConversationTarget = (id: string) =>
  buddyRequest<BuddyConversationSummary>(
    `/conversation-targets/${encodeURIComponent(id)}`
  )
export type BuddyTurn = {
  id: string
  client_request_id: string
  conversation_id: string
  conversation_title: string
  status: "queued" | "running" | "completed" | "failed" | "stopped"
  result_message_id: string | null
  error_code: string | null
  created_at: string
  updated_at: string
}
const pageOffsets = (pages: number) =>
  Array.from(
    { length: Math.max(1, Math.floor(pages)) },
    (_, index) => index * BUDDY_PAGE_SIZE
  )
export const listBuddyTurns = async ({
  pages = 1,
  status
}: { pages?: number; status?: "active" } = {}) => {
  const results = await Promise.all(
    pageOffsets(pages).map((offset) =>
      buddyRequest<{ turns: BuddyTurn[] }>(
        `/turns?client_slot=default&${pageQuery({ offset })}${status ? `&status=${status}` : ""}`
      )
    )
  )
  return {
    turns: [
      ...new Map(
        results.flatMap((result) => result.turns).map((turn) => [turn.id, turn])
      ).values()
    ],
    hasMore: results.at(-1)!.turns.length === BUDDY_PAGE_SIZE
  }
}
export const acceptBuddyTurn = (body: {
  conversation_id: string
  text: string
  client_request_id: string
  expected_attachment_version: number
  model?: string
  provider?: string
}) => buddyRequest<BuddyTurn>("/turns?client_slot=default", "POST", body)
export const stopBuddyTurn = (id: string) =>
  buddyRequest<BuddyTurn>(`/turns/${encodeURIComponent(id)}/stop`, "POST")
export type BuddyActivity = {
  conversation_id: string
  title: string
  workspace_id: string | null
  result: { id: string; created_at: string; content: string } | null
  acknowledged: boolean
}
export const listBuddyActivity = async ({
  conversationPages = 1
}: { conversationPages?: number } = {}) => {
  // Activity pages follow conversation pages; an empty result page is not the end.
  const results = await Promise.all(
    pageOffsets(conversationPages).map((offset) =>
      buddyRequest<{ items: BuddyActivity[] }>(
        `/attachment/activity?client_slot=default&${pageQuery({ offset })}`
      )
    )
  )
  return {
    items: [
      ...new Map(
        results
          .flatMap((result) => result.items)
          .map((item) => [item.conversation_id, item])
      ).values()
    ]
  }
}
export const acknowledgeBuddyResult = (body: {
  conversation_id: string
  result_message_id: string
}) =>
  buddyRequest<{ acknowledged: boolean }>(
    "/attachment/acknowledgements?client_slot=default",
    "POST",
    body
  )
export const buddyAssets = (
  buddy: BuddyProfile
): Record<string, PersonaVisualAsset> =>
  Object.fromEntries(
    buddy.assets.map((asset) => [
      asset.id,
      { ...asset, url: asset.content_url }
    ])
  )

export type BuddyReplySettings = {
  model: string | null
  provider: string | null
}

export async function readBuddyConversation(
  attachment: BuddyAttachment,
  conversationId: string,
  workspaceId?: string | null,
  {
    isCurrent = () => true,
    includeReplySettings = false
  }: { isCurrent?: () => boolean; includeReplySettings?: boolean } = {}
) {
  if (
    attachment.scope_type === "conversation" &&
    attachment.scope_id !== conversationId
  ) {
    throw new Error("This Buddy is attached to a different conversation.")
  }
  const scopedWorkspace =
    attachment.scope_type === "workspace" ? attachment.scope_id : workspaceId
  const scope = scopedWorkspace
    ? { type: "workspace" as const, workspaceId: scopedWorkspace }
    : undefined
  const connectionIdentity = (
    config: Awaited<ReturnType<typeof tldwClient.getConfig>>
  ) =>
    JSON.stringify([
      config?.serverUrl,
      config?.authMode,
      config?.apiKey,
      config?.accessToken,
      config?.authSource,
      config?.orgId
    ])
  if (!isCurrent())
    throw new Error("Interaction changed before reading conversation.")
  const connection = connectionIdentity(await tldwClient.getConfig())
  const assertCurrent = async () => {
    if (!isCurrent())
      throw new Error("Interaction changed before reading conversation.")
    if (connection !== connectionIdentity(await tldwClient.getConfig()))
      throw new Error("Connection changed before reading conversation.")
    if (!isCurrent())
      throw new Error("Interaction changed before reading conversation.")
  }
  await assertCurrent()
  const conversation = await tldwClient.getChat(conversationId, { scope })
  await assertCurrent()
  if (
    attachment.scope_type === "workspace" &&
    (conversation.scope_type !== "workspace" ||
      conversation.workspace_id !== attachment.scope_id)
  ) {
    throw new Error(
      "This conversation no longer belongs to the attached workspace."
    )
  }
  // Revalidate with the server even when another surface has cached this chat.
  tldwClient.invalidateChatMessagesCache(conversationId)
  const messages = await tldwClient.listChatMessages(
    conversationId,
    { limit: 100, order: "desc" },
    { scope }
  )
  await assertCurrent()
  const replySettings = includeReplySettings
    ? await tldwClient.requestWithCurrentConfig<BuddyReplySettings>(
        (
          config: NonNullable<Awaited<ReturnType<typeof tldwClient.getConfig>>>
        ) => {
          if (!isCurrent())
            throw new Error(
              "Interaction changed before reading reply settings."
            )
          if (connection !== connectionIdentity(config))
            throw new Error("Connection changed before reading reply settings.")
          // A factory pins this checked config through the extension background
          // handoff as well as direct WebUI transport.
          return {
            path: toAllowedPath(
              `/api/v1/buddies/conversation-targets/${encodeURIComponent(conversationId)}/reply-settings?client_slot=default`
            ),
            method: "GET"
          }
        }
      )
    : null
  await assertCurrent()
  return {
    conversation,
    replySettings,
    messages: [...messages].sort((a, b) =>
      a.created_at.localeCompare(b.created_at)
    )
  }
}

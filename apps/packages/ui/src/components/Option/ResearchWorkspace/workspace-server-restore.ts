import { projectWorkspaceRagSources, readWorkspaceRagSourceProvenance, WORKSPACE_RAG_SOURCES_KEY } from "@/utils/workspace-rag-sources";
import { formatToMessage } from "@/db/dexie/helpers";
import { readWorkspaceServerChatMirror, serverChatMirrorOwnerKey } from "@/db/dexie/server-chat-mirror";
import { canonicalHistoryJson } from "@/db/dexie/history-selection";
import { fetchAllServerChatMessages, mapServerChatMessagesToPlaygroundMessages } from "@/hooks/chat/useServerChatLoader";
import { tldwClient } from "@/services/tldw/TldwApiClient";
import type { Message } from "@/store/option";
import { rememberCanonicalArtifactWorkspace } from "@/services/workspace-artifact-persistence";
import { bindWorkspaceArtifactOrigin } from "@/store/workspace-artifact-codec";
import { recoverWebCapturePin, WebCaptureNotCurrentError } from "@/utils/research-web-capture";
import {
  getResearchWorkspaceOwner,
  readResearchWebCaptures,
  retainResearchWebCapturePins,
} from "@/utils/research-workspace-prefill";
import { normalizeNoteKeyword } from "@/services/note-keywords";
import { resolveKnowledgeNoteProvenance, knowledgeNoteHead, retainKnowledgeNoteProvenance, type KnowledgeNoteHead } from "@/utils/knowledge-note-provenance";
import { bgRequest } from "@/services/background-proxy";
import { loadServicePromptSnapshot } from "@/services/service-prompts";
import { requestScopeFields } from "@/services/tldw/domains/service-prompts";
import type {
  WorkspaceArtifactApiResponse,
  WorkspaceContextResponse,
  WorkspaceListApiResponse,
  WorkspaceNoteApiResponse,
} from "@/services/tldw/domains/workspace-api";
import { createServicePromptScopeChangedError } from "@/services/tldw/service-prompt-scope-error";
import {
  createEmptyWorkspaceSnapshot,
  deriveHistoryFromMessages,
  createSlug,
  useWorkspaceStore,
  type WorkspaceState,
  type WorkspaceChatSession,
} from "@/store/workspace";
import { hydrateWorkspaceFromServer } from "@/store/workspace-api";
import { RESEARCH_WORKSPACE_MIGRATION_TOMBSTONE_PREFIX, readResearchWorkspaceChatReference, readSavedResearchWorkspaceChatReference } from "@/store/workspace-migration";
import { normalizeWorkspaceAssistantDefaults } from "@/types/workspace-assistant-defaults";
import type { WorkspaceSourceStatus } from "@/types/workspace";

type WorkspaceSnapshot = WorkspaceState["workspaceSnapshots"][string];
const INFORMATIONAL_CONTEXT_ERRORS = new Set([
  "jobs_unavailable",
  "media_db_unavailable",
  "membership_summary_unavailable",
]);

type CachedWorkspaceChat = { referenceId: string; session: WorkspaceChatSession };

type MigrationReceipt = {
  id: string;
  deletedAt: number;
  key: string;
  scopeKey?: string;
  value: Record<string, unknown>;
  raw: string;
};

const readMigrationReceipts = (storage: Storage): MigrationReceipt[] => {
  const receipts: MigrationReceipt[] = [];
  for (let index = 0; index < storage.length; index += 1) {
    const key = storage.key(index);
    if (!key?.startsWith(`${RESEARCH_WORKSPACE_MIGRATION_TOMBSTONE_PREFIX}:`))
      continue;
    try {
      const raw = storage.getItem(key) || "null";
      const receipt = JSON.parse(raw);
      if (
        receipt?.contentRetained !== false ||
        receipt.restoreInvalidated === true ||
        typeof receipt.serverWorkspaceId !== "string" ||
        !receipt.serverWorkspaceId.trim() ||
        typeof receipt.migrationId !== "string" ||
        !receipt.migrationId.trim()
      )
        continue;
      const deletedAt = Date.parse(receipt.deletedAt);
      if (Number.isFinite(deletedAt))
        receipts.push({
          id: receipt.serverWorkspaceId,
          deletedAt,
          key,
          value: receipt,
          raw,
          scopeKey:
            typeof receipt.serverScopeKey === "string"
              ? receipt.serverScopeKey
              : undefined,
        });
    } catch {
      // A malformed local receipt grants no authority and cannot identify a server workspace.
    }
  }
  return receipts.sort((a, b) => b.deletedAt - a.deletedAt);
};

/** Untrusted location hint only; callers must resolve it within their captured account. */
export const readMigratedResearchWorkspaceId = (
  storage: Storage,
): string | null => readMigrationReceipts(storage)[0]?.id ?? null;

const cachedMessagesMatchCanonical = (cached: Message[], canonical: Message[], retainLegacyLinearParent = false): boolean => {
  const ids = cached.map(message => message.serverMessageId || message.id);
  if (cached.length !== canonical.length || ids.some(id => !id) ||
      new Set(ids).size !== ids.length) return false;
  try {
    const localIds = new Map<string, string>();
    cached.forEach((message, index) => {
      if (!message.id || localIds.has(message.id)) throw new Error("Ambiguous local message identity");
      localIds.set(message.id, ids[index]!);
    });
    const parentId = (value: unknown) => {
      if (value == null) return null;
      if (typeof value !== "string") throw new Error("Unknown message parent");
      const candidates = new Set([localIds.get(value), ids.includes(value) ? value : undefined].filter(Boolean));
      if (candidates.size !== 1) throw new Error("Unverified message parent");
      return [...candidates][0];
    };
    return cached.every((message, index) => {
      const saved = retainLegacyLinearParent && index === 1
        ? { ...canonical[index], parentMessageId: canonical[0].serverMessageId }
        : canonical[index];
      if (ids[index] !== saved.serverMessageId || message.isBot !== saved.isBot) return false;
      // Optimistic display labels and timing are retained, not canonical payload.
      const display = ["id", "name", "createdAt", "modelId", "modelName", "modelImage", "reasoning_time_taken"];
      const fields = Object.keys(message).filter(key => !display.includes(key) && message[key as keyof Message] !== undefined);
      const payload = (row: Message, local: boolean) => Object.fromEntries(fields.map(key => {
        if (key === "sources" && (message.sources?.length || saved.sources?.length)) {
          const provenance = local ? projectWorkspaceRagSources(row.sources)
            : readWorkspaceRagSourceProvenance(saved.metadataExtra?.[WORKSPACE_RAG_SOURCES_KEY]);
          if (!provenance) throw new Error("Unverified saved citation projection");
          return [key, provenance];
        }
        return [key, key === "parentMessageId" && local ? parentId(row.parentMessageId) : row[key as keyof Message]];
      }));
      return canonicalHistoryJson(payload(message, true)) === canonicalHistoryJson(payload(saved, false));
    });
  } catch {
    return false; // Opaque or unverified local work must remain untouched.
  }
};

/** Restore an owned target or the latest migration while its captured account lease remains live. */
export const restoreMigratedResearchWorkspace = async (options: {
  workspaceId?: string;
  signal: AbortSignal;
  isCurrent?: () => boolean;
  apply: (snapshot: WorkspaceSnapshot, chatSession?: WorkspaceChatSession, verifiedCachedChat?: CachedWorkspaceChat) => void;
  getCachedChat?: () => CachedWorkspaceChat | null;
  storage?: Storage;
}): Promise<boolean> => {
  const requestedId = options.workspaceId;
  if (
    requestedId !== undefined &&
    (!requestedId.trim() || requestedId !== requestedId.trim() || requestedId.length > 128)
  )
    throw new Error("Invalid requested workspace");
  const storage = options.storage ?? window.localStorage;
  const receipts = readMigrationReceipts(storage).filter(
    (receipt) => !requestedId || receipt.id === requestedId,
  );
  if (!receipts.length && !requestedId) return false;
  const scope = await loadServicePromptSnapshot([], { signal: options.signal });
  const assertCurrent = () => {
    if (scope.scopeInvalidatedSignal.aborted)
      throw createServicePromptScopeChangedError();
    if (options.signal.aborted || scope.scopeSignal.aborted || options.isCurrent?.() === false)
      throw new DOMException("Workspace restoration cancelled", "AbortError");
  };
  const request = {
    method: "GET" as const,
    abortSignal: scope.scopeSignal,
    ...requestScopeFields(scope.requestScope),
  };
  try {
    assertCurrent();
    let receipt = receipts.find((value) => value.scopeKey === scope.scopeKey);
    if (!receipt) {
      const unbound = receipts.filter((value) => !value.scopeKey);
      if (unbound.length) {
        const listed = await bgRequest<WorkspaceListApiResponse>({
          ...request,
          path: "/api/v1/workspaces/",
        });
        assertCurrent();
        if (!Array.isArray(listed.items))
          throw new Error("The current account's workspaces could not be listed");
        receipt = unbound.find((value) =>
          listed.items.some(
            (workspace) => workspace.id === value.id && !workspace.deleted && !workspace.archived,
          ),
        );
      }
      if (!receipt && !requestedId) return false;
    }
    const workspaceId = requestedId ?? receipt!.id;
    const invalidateReceipt = () => {
      assertCurrent();
      if (!receipt) return false;
      if (storage.getItem(receipt.key) !== receipt.raw)
        throw new Error("Saved workspace changed during restoration. Retry to load the current chat.");
      // Retain the migration tombstone so deleted legacy content cannot reappear.
      storage.setItem(
        receipt.key,
        JSON.stringify({ ...receipt.value, restoreInvalidated: true }),
      );
      return false;
    };
    const path =
      `/api/v1/workspaces/${encodeURIComponent(workspaceId)}` as const;
    let context: WorkspaceContextResponse;
    try {
      context = await bgRequest<WorkspaceContextResponse>({
        ...request,
        path: `${path}/context`,
      });
    } catch (error) {
      assertCurrent();
      if ((error as { status?: number } | null)?.status === 404)
        return invalidateReceipt();
      throw error;
    }
    assertCurrent();
    if (
      context.workspace_id !== workspaceId ||
      context.workspace.id !== workspaceId
    ) {
      throw new Error("Workspace restoration returned a different workspace");
    }
    if (context.workspace.deleted || context.workspace.archived)
      return invalidateReceipt();
    if (
      context.partial_errors?.some(
        (error) => !INFORMATIONAL_CONTEXT_ERRORS.has(error.code),
      ) ||
      !Array.isArray(context.sources?.items) ||
      context.sources.items.some(
        (source) => source.workspace_id !== workspaceId,
      )
    ) {
      throw new Error("The server workspace could not be restored completely");
    }
    const [artifacts, notes] = await Promise.all([
      bgRequest<WorkspaceArtifactApiResponse[]>({
        ...request,
        path: `${path}/artifacts`,
      }),
      bgRequest<WorkspaceNoteApiResponse[]>({
        ...request,
        path: `${path}/notes`,
      }),
    ]);
    assertCurrent();
    if (
      artifacts.some((artifact) => artifact.workspace_id !== workspaceId) ||
      notes.some((note) => note.workspace_id !== workspaceId)
    ) {
      throw new Error("Workspace restoration returned a different workspace");
    }
    const local = await hydrateWorkspaceFromServer(workspaceId, {
      fetch: async () => ({
        ...context.workspace,
        sources: context.sources.items,
        artifacts,
        notes,
      }),
    });
    assertCurrent();
    const workspace = context.workspace;
    const snapshot = createEmptyWorkspaceSnapshot({
      id: workspaceId,
      name: local.name || "Research Workspace",
      tag: `workspace:${createSlug(local.name) || workspaceId.slice(0, 8)}`,
      createdAt: new Date(workspace.created_at),
      studyMaterialsPolicy: workspace.study_materials_policy,
      assistantDefaults: normalizeWorkspaceAssistantDefaults(
        workspace.assistant_defaults,
      ),
    });
    snapshot.sources = local.sources.map((source, index) => {
      const authoritative = context.sources.items[index];
      const status: WorkspaceSourceStatus =
        authoritative.state === "queryable" ||
        authoritative.state === "partially_queryable"
          ? "ready"
          : ["failed", "missing_media", "blocked_by_permissions"].includes(
                authoritative.state,
              )
            ? "error"
            : "processing";
      return { ...source, status, readiness: authoritative.readiness };
    });
    const owner = await getResearchWorkspaceOwner(scope.requestScope);
    const captureRecords = await readResearchWebCaptures(owner, workspaceId);
    assertCurrent();
    const displayState = useWorkspaceStore.getState();
    const previousSnapshot = displayState.workspaceSnapshots[workspaceId];
    const authoritativeSources = snapshot.sources;
    snapshot.sources = retainResearchWebCapturePins(
      authoritativeSources,
      captureRecords,
      displayState.workspaceId === workspaceId
        ? displayState.sources
        : previousSnapshot?.workspaceId === workspaceId
          ? previousSnapshot.sources
          : [],
      owner,
    );
    for (const [index, source] of snapshot.sources.entries()) {
      if (!source.id.startsWith("web-clipper:")) continue;
      const checkpoint = captureRecords.some(
        (record) =>
          record.pin &&
          record.pin.clipId === source.webCapture?.clipId &&
          record.pin.mediaId === source.mediaId &&
          record.pin.requestedUrl === source.url,
      );
      if (checkpoint) continue;
      try {
        const pin = await recoverWebCapturePin(
          context.sources.items[index],
          {
            requestScope: scope.requestScope,
            signal: scope.scopeSignal,
          },
          assertCurrent,
          source.webCapture,
        );
        assertCurrent();
        if (
          !pin &&
          (source.statusDetails?.statusReason === "capture_unavailable" ||
            captureRecords.some(
              (record) =>
                record.body?.clip_id === source.id.slice("web-clipper:".length),
            ))
        )
          throw new Error("Known capture history unavailable");
        if (pin)
          snapshot.sources[index] = {
            ...(source.statusDetails?.statusReason === "capture_unavailable"
              ? authoritativeSources[index]
              : source),
            webCapture: pin,
            captureOwnerScope: owner,
          };
      } catch (reason) {
        assertCurrent();
        snapshot.sources[index] = {
          ...source,
          // Unknown lookup failure blocks this attempt but cannot label an ordinary clip forever.
          captureOwnerScope:
            source.captureOwnerScope ??
            (reason instanceof WebCaptureNotCurrentError &&
            reason.captureEvidenceKnown
              ? owner
              : undefined),
          status: "error",
          statusMessage:
            "Capture evidence is unavailable. Retry or recapture before Ask.",
          statusDetails: {
            statusReason: "capture_unavailable",
            retryEligible: false,
          },
        };
      }
    }
    snapshot.selectedSourceIds = local.selectedSourceIds.filter(
      (id) =>
        !snapshot.sources.some(
          (source) =>
            source.id === id &&
            ["capture_head_changed", "capture_unavailable"].includes(
              source.statusDetails?.statusReason ?? "",
            ),
        ),
    );
    snapshot.generatedArtifacts = local.artifacts.map(artifact => bindWorkspaceArtifactOrigin(artifact, scope.scopeKey, workspaceId));
    snapshot.workspaceBanner = {
      ...snapshot.workspaceBanner,
      title: workspace.banner_title || "",
      subtitle: workspace.banner_subtitle || "",
    };
    const note = notes[0];
    if (note) {
      let keywords: unknown = [];
      try {
        keywords = JSON.parse(note.keywords_json || "[]");
      } catch {
        // Keywords are optional metadata; retain the note's title and content.
      }
      snapshot.currentNote = {
        id: note.id,
        title: note.title,
        content: note.content,
        keywords: Array.isArray(keywords)
          ? keywords.filter(
              (keyword): keyword is string => typeof keyword === "string",
            )
          : [],
        isDirty: false,
        version: note.version,
      };
      snapshot.notes = note.content;
    }
    // Canonical Quick Notes survive the legacy snapshot tombstone. The marker is
    // descriptive evidence only; server membership and selection stay authoritative.
    const found = await bgRequest<{
      notes?: Array<KnowledgeNoteHead & {
        id: string;
        title: string;
        content: string;
        keywords?: unknown[];
        version: number;
      }>;
    }>({
      ...request,
      path: `/api/v1/notes/search/?tokens=${encodeURIComponent(`workspace:${workspaceId}`)}&limit=100&include_keywords=true`,
    });
    assertCurrent();
    let canonical: NonNullable<typeof found.notes>[number] | undefined;
    const seenNotes = new Set<string>();
    for (const candidate of found.notes || []) {
      if (typeof candidate.id !== "string" || seenNotes.has(candidate.id)) continue;
      seenNotes.add(candidate.id);
      // Search rows may omit metadata or truncate content. Resolve the exact owned head.
      const full = await bgRequest<NonNullable<typeof found.notes>[number]>({
        ...request, path: `/api/v1/notes/${encodeURIComponent(candidate.id)}`,
      });
      assertCurrent();
      if (full.id !== candidate.id) throw new Error("Research note identity changed");
      const belongsToWorkspace = (full.keywords || []).map(normalizeNoteKeyword).includes(`workspace:${workspaceId}`);
      if (belongsToWorkspace || resolveKnowledgeNoteProvenance(full).provenance?.research?.workspace_id === workspaceId) {
        // Several associated Notes require the visible Load choice.
        if (canonical) { canonical = undefined; break; }
        canonical = full;
      }
    }
    if (canonical) {
      const provenance = resolveKnowledgeNoteProvenance(canonical).provenance;
      snapshot.currentNote = {
        id: canonical.id,
        title: canonical.title,
        content: retainKnowledgeNoteProvenance(canonical.content, canonical),
        ...knowledgeNoteHead(canonical),
        keywords: (canonical.keywords || [])
          .map(normalizeNoteKeyword)
          .filter(
            (keyword): keyword is string =>
              keyword !== null &&
              keyword !== snapshot.workspaceTag &&
              keyword !== `workspace:${workspaceId}`,
          ),
        version: canonical.version,
        isDirty: false,
      };
      snapshot.notes = snapshot.currentNote.content;
      snapshot.sources = snapshot.sources.map((source) => {
        const retained = provenance?.research?.workspace_id === workspaceId
          ? provenance.research.sources.find((item) => item.mediaId === source.mediaId)
          : undefined;
        return retained
          ? { ...source, knowledgeQaEvidence: retained.evidence }
          : source;
      });
    }
    const savedChat = readSavedResearchWorkspaceChatReference(storage, workspaceId, scope.scopeKey);
    const markerReference = readResearchWorkspaceChatReference(receipt?.value.chatReference);
    if (savedChat.value && markerReference &&
        canonicalHistoryJson(savedChat.value) !== canonicalHistoryJson(markerReference))
      throw new Error("Saved workspace chat references conflict. Your local work has been kept.");
    const cachedSelection = options.getCachedChat?.();
    const cachedChat = cachedSelection?.session;
    const provisional = !savedChat.value && !markerReference && cachedChat?.messages.length &&
      cachedChat.serverChatId
      ? readResearchWorkspaceChatReference({
          referenceId: cachedSelection!.referenceId, serverChatId: cachedChat.serverChatId,
        }) : undefined;
    const chatReference = savedChat.value ?? markerReference ?? provisional;
    let chatSession: WorkspaceChatSession | undefined;
    if (chatReference) {
      if (chatReference.serverChatId) {
        const head = await bgRequest<{ id: string; scope_type: string; workspace_id: string | null; deleted?: boolean }>({
          ...request,
          path: `/api/v1/chats/${encodeURIComponent(chatReference.serverChatId)}?scope_type=workspace&workspace_id=${encodeURIComponent(workspaceId)}`,
        });
        assertCurrent();
        if (head.id !== chatReference.serverChatId || head.scope_type !== "workspace" ||
            head.workspace_id !== workspaceId || head.deleted === true)
          throw new Error("Saved workspace chat is unavailable in this account and workspace");
      }
      if (!provisional) {
        snapshot.workspaceChatReferenceId = chatReference.referenceId;
        chatSession = { messages: [], history: [], historyId: null, serverChatId: chatReference.serverChatId };
      }
    }
    let verifiedCachedChat: CachedWorkspaceChat | undefined;
    const mirrorBinding = chatReference?.serverChatId && cachedChat?.historyId ? {
      historyId: cachedChat.historyId, chatId: chatReference.serverChatId,
      ownerKey: serverChatMirrorOwnerKey(scope), workspaceId,
    } : undefined;
    const mirror = mirrorBinding ? await readWorkspaceServerChatMirror(mirrorBinding) : undefined;
    assertCurrent();
    if (chatReference?.serverChatId && cachedChat?.serverChatId === chatReference.serverChatId &&
        cachedChat.messages.length) {
      const signal = AbortSignal.any([scope.scopeSignal, AbortSignal.timeout(15000)]);
      const serverMessages = await fetchAllServerChatMessages(
        ({ limit, offset }) => tldwClient.listChatMessages(chatReference.serverChatId!, {
          limit, offset, include_metadata: "true", include_images: "true",
          include_deleted: "false", render_placeholders: "false",
        }, { signal, fresh: true, requestScope: scope.requestScope,
          scope: { type: "workspace", workspaceId } }),
        { signal },
      );
      assertCurrent();
      const canonical = mapServerChatMessagesToPlaygroundMessages({ serverMessages, assistantName: "AI", characterId: null });
      const mirrorMessages = mirror ? formatToMessage(mirror.rows) : undefined;
      // Older manual two-message saves omitted parents. Retain that exact local
      // linear graph; it is not a canonical parent or permission to rewrite it.
      const retainLegacyLinearParent = !!mirror && serverMessages.length === 2 &&
        serverMessages[0].role === "user" && serverMessages[1].role === "assistant" &&
        serverMessages.every(row => row.parent_message_id === null && row.version === 1);
      const matches = (legacy: boolean) => cachedMessagesMatchCanonical(cachedChat.messages, canonical, legacy) &&
        (!mirrorMessages || (mirror!.rows.length === mirrorMessages.length &&
          cachedMessagesMatchCanonical(mirrorMessages, canonical, legacy)));
      if ((matches(false) || (retainLegacyLinearParent && matches(true))) &&
          (cachedChat.history.length === 0 ||
            canonicalHistoryJson(cachedChat.history) === canonicalHistoryJson(deriveHistoryFromMessages(canonical))))
        verifiedCachedChat = cachedSelection!;
    }
    if (provisional && verifiedCachedChat) {
      snapshot.workspaceChatReferenceId = provisional.referenceId;
      chatSession = { messages: [], history: [], historyId: null, serverChatId: provisional.serverChatId };
    }
    if (mirror && verifiedCachedChat) {
      if (canonicalHistoryJson(await readWorkspaceServerChatMirror(mirrorBinding!)) !== canonicalHistoryJson(mirror))
        throw new Error("Workspace Chat mirror changed during restoration");
      // Keep its exact local settings association and optimistic display data.
      chatSession = cachedChat;
    }
    assertCurrent();
    const relevantReceipts = (rows: MigrationReceipt[]) => rows
      .filter(row => row.id === workspaceId && (!row.scopeKey || row.scopeKey === scope.scopeKey))
      .map(row => [row.key, row.raw]);
    if (canonicalHistoryJson(relevantReceipts(readMigrationReceipts(storage))) !==
        canonicalHistoryJson(relevantReceipts(receipts)))
      throw new Error("Saved workspace changed during restoration. Retry to load the current chat.");
    if (storage.getItem(savedChat.key) !== savedChat.raw)
      throw new Error("Saved workspace changed during restoration. Retry to load the current chat.");
    if (receipt) {
      if (storage.getItem(receipt.key) !== receipt.raw)
        throw new Error("Saved workspace changed during restoration. Retry to load the current chat.");
      storage.setItem(
        receipt.key,
        JSON.stringify({ ...receipt.value, serverScopeKey: scope.scopeKey }),
      );
    }
    assertCurrent();
    rememberCanonicalArtifactWorkspace(workspaceId, scope.scopeKey);
    options.apply(snapshot, chatSession, verifiedCachedChat);
    return true;
  } finally {
    scope.release();
  }
};

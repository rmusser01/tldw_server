import React from "react";
import { useTranslation } from "react-i18next";
import {
  useInfiniteQuery,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import { useLiveQuery } from "dexie-react-hooks";
import { useLocation, useNavigate } from "react-router-dom";
import { Button, Image, Modal, Spin } from "antd";
import { MessageSquare, PinIcon } from "lucide-react";
import { db } from "@/db/dexie/schema";
import {
  pageAssistDatabase,
  isReadableOnDeviceHistory,
  type VisibleServerChats,
  watchOnDeviceHistoryChanges,
  type OnDeviceHistoryScope,
} from "@/db/dexie/chat";
import { ensureLocalProfileId } from "@/db/dexie/history-selection";
import type { HistoryInfo } from "@/db/dexie/types";
import { useChatDraftOwner } from "@/hooks/useChatDraftOwner";
import { useStoreChatModelSettings } from "@/store/model";
import { useStoreMessageOption } from "@/store/option";
import { useShallow } from "zustand/react/shallow";
import { useSelectedModel } from "@/hooks/chat/useSelectedModel";
import { useLoadLocalConversation } from "@/hooks/useLoadLocalConversation";
import {
  historySelectionExpansionPath,
  useHistorySelectionContext,
} from "@/hooks/chat/useHistorySelection";
import { captureChatDraftLoad } from "@/hooks/chat/chat-draft-transition";
import { requestSettingsNavigation } from "@/utils/settings-return";
import { MarkdownPreview } from "@/components/Common/MarkdownPreview";
import { cn } from "@/libs/utils";

export interface LocalChatListProps {
  searchQuery: string;
  selectedChatId?: string | null;
  onSelectChat?: (chatId: string) => void;
  readOnly?: boolean;
  includeCachedServerChats?: boolean;
  visibleServerChats?: VisibleServerChats | null;
  className?: string;
}
type PreviewData = NonNullable<
  Awaited<ReturnType<typeof pageAssistDatabase.getOnDeviceChatPreview>>
>;
type Preview = {
  id: string;
  title: string;
  scope: OnDeviceHistoryScope;
  isCurrent: () => boolean;
  status: "loading" | "ready" | "error" | "unavailable";
  data?: PreviewData;
  cached: boolean;
};
type PreparedOpen = {
  row: HistoryInfo;
  token: number;
  address: ReturnType<
    typeof useStoreMessageOption.getState
  >["chatDraftAddress"];
  revision: number;
  profileId: string;
  readEpoch: number;
  isCurrent: () => boolean;
};
const focusRing =
  "focus:outline-none focus-visible:ring-2 focus-visible:ring-focus focus-visible:ring-offset-2 focus-visible:ring-offset-bg";

const returnPreviewFocus = (
  initiator: HTMLElement | null,
  sidebar: Element | null,
  isCurrent: () => boolean = () => true,
) => {
  queueMicrotask(() => {
    if (!isCurrent()) return;
    const active = document.activeElement;
    if (
      active !== document.body &&
      active?.isConnected &&
      !active.closest(".on-device-history-preview")
    )
      return;
    const visible = (element: Element | null): element is HTMLElement => {
      if (!(element instanceof HTMLElement) || !element.isConnected)
        return false;
      for (
        let ancestor: HTMLElement | null = element;
        ancestor;
        ancestor = ancestor.parentElement
      ) {
        const style = getComputedStyle(ancestor);
        if (
          ancestor.hidden ||
          ancestor.hasAttribute("inert") ||
          style.display === "none" ||
          style.visibility === "hidden"
        )
          return false;
      }
      return true;
    };
    const currentSidebar = sidebar?.isConnected
      ? sidebar
      : document.querySelector("[data-chat-sidebar]");
    const fallback =
      currentSidebar?.querySelector(
        "input, [data-testid='chat-sidebar-toggle']",
      ) ?? null;
    if (visible(initiator)) initiator.focus({ preventScroll: true });
    else if (visible(fallback)) fallback.focus({ preventScroll: true });
  });
};

/** Discovery only: management/migration needs a separately qualified write path. */
export function LocalChatList({
  searchQuery,
  selectedChatId,
  onSelectChat,
  readOnly = false,
  includeCachedServerChats = false,
  visibleServerChats,
  className,
}: LocalChatListProps) {
  const { t } = useTranslation(["common", "sidepanel", "option"]);
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const location = useLocation();
  const routeLifetime = React.useRef({ location, epoch: 0 });
  if (routeLifetime.current.location !== location) {
    routeLifetime.current = {
      location,
      epoch: routeLifetime.current.epoch + 1,
    };
  }
  const routeEpoch = routeLifetime.current.epoch;
  const selection = useHistorySelectionContext();
  const historyId = useStoreMessageOption((state) => state.historyId);
  const { setSelectedModel } = useSelectedModel();
  const loadDependencies = useStoreMessageOption(
    useShallow((state) => ({
      setServerChatId: state.setServerChatId,
      setHistoryId: state.setHistoryId,
      setHistory: state.setHistory,
      setMessages: state.setMessages,
      setSelectedSystemPrompt: state.setSelectedSystemPrompt,
      setContextFiles: state.setContextFiles
    }))
  )
  const { setSystemPrompt } = useStoreChatModelSettings();
  const temporary = useStoreMessageOption((state) => state.temporaryChat);
  const draftAddress = useStoreMessageOption((state) => state.chatDraftAddress);
  const draftRevision = useStoreMessageOption(
    (state) => state.chatDraftRevision,
  );
  const previewOnly = readOnly || temporary;
  const [profileRetry, setProfileRetry] = React.useState(0);
  const readEpoch = React.useRef(profileRetry);
  readEpoch.current = profileRetry;
  const profileMutations = React.useRef(0);
  const [preparedOpen, setPreparedOpen] = React.useState<PreparedOpen | null>(
    null,
  );
  const pendingOpen = React.useRef<PreparedOpen | null>(null);
  const previewOnlyNow = React.useRef(previewOnly);
  previewOnlyNow.current = previewOnly;
  const [preview, setPreview] = React.useState<Preview | null>(null);
  const [loadingId, setLoadingId] = React.useState<string | null>(null);
  const [selectionFailed, setSelectionFailed] = React.useState(false);
  const listRoot = React.useRef<HTMLDivElement>(null);
  const previewInitiator = React.useRef<HTMLElement | null>(null);
  const previewWasOpen = React.useRef(false);
  const pendingPreviewFocus = React.useRef<(() => void) | null>(null);
  const mounted = React.useRef(false);
  const request = React.useRef(0);
  const currentScope = React.useRef<OnDeviceHistoryScope | null>(null);
  const {
    ownerKey,
    serverScopeKey,
    isCurrent: ownerCurrent,
    verificationState,
    retryVerification,
  } = useChatDraftOwner(() => {
    request.current++;
    pendingOpen.current = null;
    setPreparedOpen(null);
    setPreview(null);
    setLoadingId(null);
  });

  // Read only the existing profile. A retry/invalidated read must not expose
  // the previous profile while useLiveQuery resolves its next request.
  const storage = useLiveQuery(async () => {
    try {
      const settings = await db.userSettings.get("main");
      return {
        status: "ready" as const,
        profileId: settings?.history_profile_id ?? null,
        epoch: profileRetry,
      };
    } catch {
      return { status: "error" as const, profileId: null, epoch: profileRetry };
    }
  }, [profileRetry]);
  const profileId = storage?.profileId ?? null;
  const ready = storage?.status === "ready" && storage.epoch === profileRetry;
  const scope = React.useMemo<OnDeviceHistoryScope | null>(() => {
    if (!ready) return null;
    const mutation = profileMutations.current;
    const expectedReadEpoch = profileRetry;
    const reader: OnDeviceHistoryScope = {
      profileId,
      serverScopeKey,
      isCurrent: () =>
        mounted.current &&
        currentScope.current === reader &&
        profileMutations.current === mutation &&
        readEpoch.current === expectedReadEpoch &&
        routeLifetime.current.epoch === routeEpoch &&
        (ownerKey === null || ownerCurrent()),
    };
    return reader;
  }, [
    ready,
    profileId,
    serverScopeKey,
    ownerKey,
    ownerCurrent,
    profileRetry,
    routeEpoch,
  ]);
  currentScope.current = scope;
  React.useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      currentScope.current = null;
      pendingOpen.current = null;
    };
  }, []);
  React.useEffect(
    () =>
      watchOnDeviceHistoryChanges((profileChanged) => {
        if (profileChanged) {
          profileMutations.current++;
          currentScope.current = null;
          setPreview(null);
          setProfileRetry((value) => value + 1);
        }
        void queryClient.invalidateQueries({
          queryKey: ["fetchChatHistory", "on-device"],
        });
        void queryClient.invalidateQueries({
          queryKey: ["historyMetadata", "on-device"],
        });
      }),
    [queryClient],
  );
  React.useLayoutEffect(() => {
    setPreview(null);
  }, [draftAddress, draftRevision, previewOnly, scope]);
  React.useLayoutEffect(() => {
    pendingOpen.current = null;
    setPreparedOpen(null);
    setLoadingId(null);
    setSelectionFailed(false);
  }, [draftAddress, draftRevision, previewOnly, routeEpoch]);
  const query = searchQuery.trim();
  const visibleServerReceipt =
    visibleServerChats?.isCurrent() &&
    visibleServerChats.serverScopeKey === serverScopeKey &&
    visibleServerChats.query.trim().toLowerCase() === query.toLowerCase()
      ? visibleServerChats
      : null;
  const visibleServerIds = visibleServerReceipt
    ? [...visibleServerReceipt.ids].sort()
    : [];
  const readOptions = React.useMemo(
    () => ({
      includeCachedServerChats,
      visibleServerChats: visibleServerReceipt,
    }),
    [includeCachedServerChats, visibleServerReceipt],
  );
  const history = useInfiniteQuery({
    queryKey: [
      "fetchChatHistory",
      "on-device",
      query,
      profileId,
      ownerKey,
      includeCachedServerChats,
      visibleServerIds,
    ],
    enabled: !!scope,
    queryFn: async ({ pageParam }) => {
      if (!scope?.isCurrent())
        throw new Error("on_device_history_scope_changed");
      const page = await pageAssistDatabase.getChatHistoriesPaginated(
        pageParam,
        query || undefined,
        scope,
        readOptions,
      );
      if (!scope.isCurrent())
        throw new Error("on_device_history_scope_changed");
      return page;
    },
    initialPageParam: 1,
    getNextPageParam: (last, pages) =>
      last.hasMore ? pages.length + 1 : undefined,
    retry: false,
    refetchOnWindowFocus: false,
  });
  const rows = React.useMemo(() => {
    const unique = new Map<string, HistoryInfo>();
    for (const page of history.data?.pages ?? [])
      for (const row of page.histories) {
        if (
          scope &&
          isReadableOnDeviceHistory(row, scope, readOptions, query) &&
          !unique.has(row.id)
        )
          unique.set(row.id, row);
      }
    return [...unique.values()];
  }, [history.data, scope, readOptions, query]);
  const ids = rows.map((row) => row.id);
  const metadata = useQuery({
    queryKey: ["historyMetadata", "on-device", profileId, ownerKey, ids],
    enabled: !!scope && ids.length > 0,
    queryFn: async () => {
      if (!scope?.isCurrent())
        throw new Error("on_device_history_scope_changed");
      const value = await pageAssistDatabase.getHistoriesWithMetadata(
        ids,
        scope,
        { includeCachedServerChats },
      );
      if (!scope.isCurrent())
        throw new Error("on_device_history_scope_changed");
      return value;
    },
    retry: false,
    refetchOnWindowFocus: false,
  });
  const load = useLoadLocalConversation(
    { ...loadDependencies, setSelectedModel, setSystemPrompt },
    {
      t,
      errorLogPrefix: "Failed to load on-device chat",
      errorDefaultMessage: "Could not open this chat. Try again.",
    },
  );

  const closePreview = () => {
    const token = ++request.current;
    const key = previewLifetime.current.key;
    const sourceScope = scope;
    const initiator = previewInitiator.current;
    const sidebar = listRoot.current?.closest("[data-chat-sidebar]") ?? null;
    pendingPreviewFocus.current = () =>
      returnPreviewFocus(
        initiator,
        sidebar,
        () =>
          mounted.current &&
          request.current === token &&
          previewLifetime.current.key === key &&
          currentScope.current === sourceScope,
      );
    setPreview(null);
  };
  const showPreview = async (
    row: HistoryInfo,
    cached = !!row.server_chat_id,
  ) => {
    if (!scope?.isCurrent()) return;
    pendingPreviewFocus.current = null;
    const token = ++request.current;
    const source = captureChatDraftLoad("switch", scope.isCurrent);
    const current = () => token === request.current && source.isCurrent();
    const initial: Preview = {
      id: row.id,
      title: row.title,
      scope,
      isCurrent: current,
      status: "loading",
      cached,
    };
    setPreview(initial);
    try {
      const data = await pageAssistDatabase.getOnDeviceChatPreview(
        row.id,
        {
          ...scope,
          isCurrent: current,
        },
        { includeCachedServerChats },
      );
      if (!current()) return;
      setPreview({
        ...initial,
        status: data ? "ready" : "unavailable",
        data: data ?? undefined,
      });
    } catch {
      if (current()) setPreview({ ...initial, status: "error" });
    }
  };
  const openSelection = React.useCallback(
    async (
      row: HistoryInfo,
      reader: OnDeviceHistoryScope,
      token: number,
      address: PreparedOpen["address"],
      revision: number,
    ) => {
      try {
        if (
          !(await load(row.id, {
            ...reader,
            isCurrent: () => token === request.current && reader.isCurrent(),
          }))
        ) {
          if (token === request.current && reader.isCurrent())
            setSelectionFailed(true);
          return;
        }
        if (token !== request.current || !reader.isCurrent()) return;
        const state = useStoreMessageOption.getState();
        const current = selection?.getCurrent();
        const reference = selection?.getReference();
        const sameSource =
          state.chatDraftAddress === address &&
          state.chatDraftRevision === revision;
        const acceptedSource =
          state.chatDraftRevision === revision + 1 &&
          state.chatDraftTransition?.sourceAddress === address;
        if (
          (!sameSource && !acceptedSource) ||
          state.chatDraftAddress?.kind !== "local" ||
          state.chatDraftAddress.id !== row.id ||
          current?.owner?.kind !== "local" ||
          current.owner.conversation_id !== row.id ||
          !reference ||
          reference.owner_key !== current.owner.owner_key ||
          reference.conversation_id !== row.id
        )
          return;
        // The explicit reference outranks automatic restore after Layout remounts.
        if (location.pathname !== "/chat")
          navigate(
            historySelectionExpansionPath({ owner: current.owner, reference }),
          );
        onSelectChat?.(row.id);
      } finally {
        if (mounted.current && token === request.current) setLoadingId(null);
      }
    },
    [load, selection, location.pathname, navigate, onSelectChat],
  );

  React.useEffect(() => {
    if (!preparedOpen || pendingOpen.current !== preparedOpen) return;
    const cancel = () => {
      pendingOpen.current = null;
      setPreparedOpen(null);
      if (mounted.current && preparedOpen.token === request.current) {
        setSelectionFailed(true);
        setLoadingId(null);
      }
    };
    if (!preparedOpen.isCurrent()) {
      cancel();
      return;
    }
    if (!ready || profileRetry < preparedOpen.readEpoch) return;
    if (profileId !== preparedOpen.profileId || !scope?.isCurrent()) {
      cancel();
      return;
    }
    // The initializer and read-only observer must agree, in either completion order.
    // Consume this click once, then let the existing qualified loader and H1 own it.
    pendingOpen.current = null;
    setPreparedOpen(null);
    void openSelection(
      preparedOpen.row,
      scope,
      preparedOpen.token,
      preparedOpen.address,
      preparedOpen.revision,
    );
  }, [preparedOpen, ready, profileId, profileRetry, scope, openSelection]);

  const select = async (row: HistoryInfo) => {
    if (previewOnly || row.server_chat_id) {
      await showPreview(row);
      return;
    }
    if (!scope?.isCurrent() || !requestSettingsNavigation("/chat")) return;
    const token = ++request.current;
    pendingOpen.current = null;
    setPreparedOpen(null);
    const source = useStoreMessageOption.getState();
    const address = source.chatDraftAddress;
    const revision = source.chatDraftRevision;
    setLoadingId(row.id);
    setSelectionFailed(false);
    if (scope.profileId !== null) {
      await openSelection(row, scope, token, address, revision);
      return;
    }
    // An explicit normal open may initialize the local profile. Enumeration and
    // Temporary previews never do. Capture source/account before that write.
    const mutation = profileMutations.current;
    const preparing = captureChatDraftLoad(
      "switch",
      () =>
        mounted.current &&
        token === request.current &&
        !previewOnlyNow.current &&
        routeLifetime.current.epoch === routeEpoch &&
        (ownerKey === null || ownerCurrent()) &&
        profileMutations.current - mutation <= 1,
    );
    try {
      const initialized = await ensureLocalProfileId();
      if (!preparing.isCurrent()) {
        if (mounted.current && token === request.current) setLoadingId(null);
        return;
      }
      const needsRead = currentScope.current?.profileId !== initialized;
      const intent: PreparedOpen = {
        row,
        token,
        address,
        revision,
        profileId: initialized,
        readEpoch: readEpoch.current + (needsRead ? 1 : 0),
        isCurrent: preparing.isCurrent,
      };
      pendingOpen.current = intent;
      setPreparedOpen(intent);
      if (needsRead) setProfileRetry((value) => value + 1);
    } catch {
      if (preparing.isCurrent()) {
        setSelectionFailed(true);
        setLoadingId(null);
      }
    }
  };

  const groups = React.useMemo(() => {
    const now = new Date();
    const today = new Date(
      now.getFullYear(),
      now.getMonth(),
      now.getDate(),
    ).getTime();
    const yesterday = today - 86400000;
    const week = today - 7 * 86400000;
    const result = new Map<string, HistoryInfo[]>();
    for (const row of rows) {
      const group = query
        ? "searchResults"
        : row.is_pinned
          ? "pinned"
          : row.createdAt >= today
            ? "today"
            : row.createdAt >= yesterday
              ? "yesterday"
              : row.createdAt >= week
                ? "lastWeek"
                : "older";
      result.set(group, [...(result.get(group) ?? []), row]);
    }
    return [
      "searchResults",
      "pinned",
      "today",
      "yesterday",
      "lastWeek",
      "older",
    ]
      .filter((key) => result.has(key))
      .map((key) => ({ key, items: result.get(key)! }));
  }, [rows, query]);
  const error = (text: string, retry: () => void) => (
    <div role="alert" className="p-3 text-sm text-text">
      <p>{text}</p>
      <Button onClick={retry}>
        {t("common:retry", { defaultValue: "Retry" })}
      </Button>
    </div>
  );
  const visiblePreview =
    (previewOnly || (includeCachedServerChats && preview?.cached)) &&
    preview?.isCurrent()
      ? preview
      : null;
  const alternatives = React.useMemo(() => {
    const replies = new Map<string, PreviewData["messages"]>();
    for (const row of visiblePreview?.data?.messages ?? []) {
      if (
        row.role !== "assistant" ||
        !row.parent_message_id ||
        row.messageType?.includes("compare") ||
        row.clusterId
      )
        continue;
      const group = replies.get(row.parent_message_id) ?? [];
      group.push(row);
      replies.set(row.parent_message_id, group);
    }
    return replies;
  }, [visiblePreview?.data]);
  const previewLifetime = React.useRef({
    scope,
    draftRevision,
    previewOnly,
    key: 0,
  });
  // Dialog keeps closing content for animation. Replace its instance immediately
  // when the source/reader changes so old private content cannot linger.
  if (
    previewLifetime.current.scope !== scope ||
    previewLifetime.current.draftRevision !== draftRevision ||
    previewLifetime.current.previewOnly !== previewOnly
  ) {
    previewLifetime.current = {
      scope,
      draftRevision,
      previewOnly,
      key: previewLifetime.current.key + 1,
    };
  }
  const previewKey = previewLifetime.current.key;

  React.useLayoutEffect(
    () => () => {
      // Replacing the private dialog skips rc-dialog's animated-close focus return.
      // Restore only focus owned by this dialog; a newly activated control wins.
      const active = document.activeElement;
      if (
        !previewWasOpen.current ||
        (active !== document.body &&
          !active?.closest(".on-device-history-preview"))
      )
        return;
      const initiator = previewInitiator.current;
      const sidebar = listRoot.current?.closest("[data-chat-sidebar]");
      returnPreviewFocus(initiator, sidebar ?? null);
    },
    [previewKey],
  );
  React.useLayoutEffect(() => {
    previewWasOpen.current = !!visiblePreview;
  }, [visiblePreview]);

  return (
    <div
      ref={listRoot}
      className={cn("flex flex-col", className)}
      aria-busy={history.isFetching}
    >
      <p className="px-3 py-2 text-xs text-text-muted">
        {t("common:chatSidebar.local.help", {
          defaultValue:
            "Chats stored on this device. Account-bound chats appear for the current account.",
        })}
      </p>
      {includeCachedServerChats && verificationState !== "ready" && (
        <div
          role={verificationState === "error" ? "alert" : "status"}
          className="px-3 pb-2 text-sm"
        >
          <p>
            {verificationState === "error"
              ? t("common:chatSidebar.local.cachedVerifyError", {
                  defaultValue:
                    "Could not verify the account for cached Server copies. Profile-owned chats are still available.",
                })
              : t("common:chatSidebar.local.cachedVerifying", {
                  defaultValue:
                    "Verifying the account for cached Server copies. Profile-owned chats are still available.",
                })}
          </p>
          {verificationState === "error" && (
            <Button onClick={retryVerification}>
              {t("common:chatSidebar.local.retryAccount", {
                defaultValue: "Retry account verification",
              })}
            </Button>
          )}
        </div>
      )}
      {previewOnly && (
        <p role="status" className="px-3 pb-2 text-xs text-text-muted">
          {t("common:chatSidebar.local.temporaryHelp", {
            defaultValue:
              "Your Temporary chat stays open. Select a chat to preview its stored messages.",
          })}
        </p>
      )}
      {!storage ||
      storage.epoch !== profileRetry ||
      (ready && history.isPending) ? (
        <div role="status" className="p-3">
          <Spin size="small" />{" "}
          {t("common:loading.title", { defaultValue: "Loading…" })}
        </div>
      ) : storage.status === "error" ? (
        error(
          t("common:chatSidebar.local.readError", {
            defaultValue: "Could not load chats on this device.",
          }),
          () => setProfileRetry((value) => value + 1),
        )
      ) : history.isError && rows.length === 0 ? (
        error(
          t("common:chatSidebar.local.readError", {
            defaultValue: "Could not load chats on this device.",
          }),
          () => {
            void history.refetch();
          },
        )
      ) : rows.length === 0 ? (
        <p role="status" className="p-3 text-sm text-text-muted">
          {query
            ? t("common:chatSidebar.local.noMatches", {
                defaultValue: "No matching chats on this device.",
              })
            : t("common:chatSidebar.local.empty", {
                defaultValue: "No available chats on this device.",
              })}
        </p>
      ) : null}
      {scope?.isCurrent() &&
        groups.map((group) => (
          <section
            className="mx-2"
            key={group.key}
            aria-label={t(
              group.key === "searchResults"
                ? "common:searchResults"
                : `common:date.${group.key === "lastWeek" ? "last7Days" : group.key}`,
              { defaultValue: group.key },
            )}
          >
            <h3 className="px-3 py-2 text-xs font-semibold text-text-muted">
              {t(
                group.key === "searchResults"
                  ? "common:searchResults"
                  : `common:date.${group.key === "lastWeek" ? "last7Days" : group.key}`,
                { defaultValue: group.key },
              )}
            </h3>
            {group.items.map((row) => {
              const title =
                row.title || t("common:untitled", { defaultValue: "Untitled" });
              const excerpt = metadata.data
                ?.get(row.id)
                ?.lastMessage?.content.replace(/[#*_`~\[\]]/g, "")
                .slice(0, 90);
              return (
                <button
                  key={row.id}
                  type="button"
                  aria-label={
                    row.server_chat_id
                      ? t("common:chatSidebar.local.cachedPreviewAction", {
                          defaultValue:
                            "Preview cached server chat: {{title}} · {{id}}",
                          title,
                          id: row.id,
                        })
                      : previewOnly
                        ? t("common:chatSidebar.local.previewAction", {
                            defaultValue: "Preview chat: {{title}}",
                            title,
                          })
                        : rows.some(
                              (item) =>
                                item.id !== row.id && item.title === row.title,
                            )
                          ? `${title} · ${t("common:chatSidebar.local.chatReference", { defaultValue: "Chat {{id}}", id: row.id })}`
                          : title
                  }
                  aria-current={
                    !previewOnly &&
                    !row.server_chat_id &&
                    (selectedChatId ?? historyId) === row.id
                      ? "true"
                      : undefined
                  }
                  aria-busy={loadingId === row.id}
                  onClick={(event) => {
                    previewInitiator.current = event.currentTarget;
                    void select(row);
                  }}
                  className={cn(
                    "w-full rounded-md px-2 py-2 text-left text-sm hover:bg-surface [&[aria-current=true]]:bg-surface",
                    focusRing,
                  )}
                >
                  <span className="flex min-w-0 items-center gap-2">
                    <MessageSquare
                      aria-hidden="true"
                      className="size-4 shrink-0"
                    />
                    <span className="truncate font-medium">{title}</span>
                    {row.is_pinned && (
                      <PinIcon aria-hidden="true" className="size-3 shrink-0" />
                    )}
                    {loadingId === row.id && <Spin size="small" />}
                  </span>
                  {row.server_chat_id && (
                    <span className="block text-xs text-text-muted">
                      {t("common:chatSidebar.local.cachedOrigin", {
                        defaultValue: "Cached server copy · Read-only",
                      })}
                      {row.server_scope_type === "workspace" &&
                      row.server_workspace_id
                        ? " · " +
                          t("common:chatSidebar.local.cachedWorkspace", {
                            defaultValue: "Workspace {{id}}",
                            id: row.server_workspace_id,
                          })
                        : row.server_scope_type !== "global"
                          ? " · " +
                            t("common:chatSidebar.local.cachedUnknownScope", {
                              defaultValue: "Chat scope unknown",
                            })
                          : ""}
                    </span>
                  )}
                  {rows.filter((item) => item.title === row.title).length >
                    1 && (
                    <span className="block text-xs text-text-muted">
                      {t("common:chatSidebar.local.chatReference", {
                        defaultValue: "Chat {{id}}",
                        id: row.id,
                      })}
                    </span>
                  )}
                  {excerpt && (
                    <span className="block truncate text-xs text-text-muted">
                      {excerpt}
                    </span>
                  )}
                  {Number.isFinite(new Date(row.createdAt).getTime()) && (
                    <time
                      dateTime={new Date(row.createdAt).toISOString()}
                      className="block text-xs text-text-muted"
                    >
                      {new Date(row.createdAt).toLocaleDateString()}
                    </time>
                  )}
                </button>
              );
            })}
          </section>
        ))}
      {history.isError &&
        rows.length > 0 &&
        error(
          t("common:chatSidebar.local.moreError", {
            defaultValue:
              "Could not load more chats. Previously loaded chats are still available.",
          }),
          () => {
            void (history.isFetchNextPageError
              ? history.fetchNextPage()
              : history.refetch());
          },
        )}
      {metadata.isError &&
        rows.length > 0 &&
        error(
          t("common:chatSidebar.local.previewError", {
            defaultValue:
              "Message previews are unavailable. You can still open a chat.",
          }),
          () => {
            void metadata.refetch();
          },
        )}
      {selectionFailed && (
        <p role="alert" className="p-3 text-sm">
          {t("common:chatSidebar.local.openError", {
            defaultValue: "Could not open this chat. Select it again to retry.",
          })}
        </p>
      )}
      {history.hasNextPage && !history.isFetchNextPageError && (
        <Button
          className="m-3"
          loading={history.isFetchingNextPage}
          onClick={() => {
            void history.fetchNextPage();
          }}
        >
          {t("common:loadMore", { defaultValue: "Load more" })}
        </Button>
      )}
      <Modal
        className="on-device-history-preview"
        key={previewKey}
        open={!!visiblePreview}
        destroyOnHidden
        title={
          visiblePreview
            ? t("common:chatSidebar.local.previewTitle", {
                defaultValue: "{{title}} — read-only preview",
                title: visiblePreview.title,
              })
            : ""
        }
        onCancel={closePreview}
        afterClose={() => {
          const restore = pendingPreviewFocus.current;
          pendingPreviewFocus.current = null;
          restore?.();
        }}
        footer={
          <Button onClick={closePreview}>
            {t("common:chatSidebar.local.closePreview", {
              defaultValue: "Close preview",
            })}
          </Button>
        }
      >
        <p className="mb-3 text-sm text-text-muted">
          {visiblePreview?.cached
            ? t("common:chatSidebar.local.cachedPreviewHelp", {
                defaultValue:
                  "These are messages stored on this device. The copy may be incomplete or contain local edits. Your open chat stays unchanged; reconnect and open Server history to continue.",
              })
            : t("common:chatSidebar.local.previewHelp", {
                defaultValue:
                  "Your Temporary chat stays open. These are stored messages in time order; alternatives and comparison results may be included.",
              })}
        </p>
        {visiblePreview?.status === "loading" && (
          <p role="status">
            <Spin size="small" />{" "}
            {t("common:loading.title", { defaultValue: "Loading…" })}
          </p>
        )}
        {visiblePreview?.status === "error" &&
          error(
            t("common:chatSidebar.local.previewReadError", {
              defaultValue: "Could not read this chat.",
            }),
            () => {
              void showPreview(
                {
                  id: visiblePreview.id,
                  title: visiblePreview.title,
                } as HistoryInfo,
                visiblePreview.cached,
              );
            },
          )}
        {visiblePreview?.status === "unavailable" && (
          <p role="status">
            {t("common:chatSidebar.local.unavailable", {
              defaultValue: "This chat is no longer available in this view.",
            })}
          </p>
        )}
        {visiblePreview?.status === "ready" && (
          <div
            tabIndex={0}
            aria-label={t("common:chatSidebar.local.storedMessages", {
              defaultValue: "Stored messages",
            })}
            className={cn("max-h-[55vh] overflow-y-auto space-y-4", focusRing)}
          >
            {!visiblePreview.data?.messages.length && (
              <p role="status">
                {t("common:chatSidebar.local.noMessages", {
                  defaultValue: "No stored messages in this chat.",
                })}
              </p>
            )}
            {visiblePreview.data?.messages.map((row) => {
              const comparison =
                row.messageType?.includes("compare") || !!row.clusterId;
              const siblings =
                !comparison && row.parent_message_id
                  ? (alternatives.get(row.parent_message_id) ?? [])
                  : [];
              const role = t(`common:chatSidebar.local.roles.${row.role}`, {
                defaultValue: row.role,
              });
              const images = [
                ...new Set([
                  ...(row.images ?? []),
                  ...(row.image ? [row.image] : []),
                ]),
              ].filter(Boolean);
              return (
                <article key={row.id} className="border-b border-border pb-3">
                  <p className="text-sm font-semibold">
                    {row.name || role}
                    {row.name && row.name !== role && (
                      <span className="ml-2 text-xs font-normal text-text-muted">
                        {role}
                      </span>
                    )}
                  </p>
                  {siblings.length > 1 && (
                    <p className="text-xs text-text-muted">
                      {t("common:chatSidebar.local.alternative", {
                        defaultValue:
                          "Alternative reply {{index}} of {{count}}",
                        index:
                          siblings.findIndex((other) => other.id === row.id) +
                          1,
                        count: siblings.length,
                      })}
                    </p>
                  )}
                  {comparison && (
                    <p className="text-xs text-text-muted">
                      {row.role === "assistant"
                        ? t("common:chatSidebar.local.comparison", {
                            defaultValue: "Comparison result",
                          })
                        : t("common:chatSidebar.local.comparisonPrompt", {
                            defaultValue: "Comparison prompt",
                          })}
                    </p>
                  )}
                  <MarkdownPreview content={row.content} />
                  {images.map((src, index) => (
                    <Image
                      preview={false}
                      key={src}
                      src={src}
                      loading="lazy"
                      alt={t("common:chatSidebar.local.image", {
                        defaultValue: "Image {{index}} from {{speaker}}",
                        index: index + 1,
                        speaker: row.name || role,
                      })}
                      className="mt-2 max-w-full rounded"
                    />
                  ))}
                </article>
              );
            })}
          </div>
        )}
      </Modal>
    </div>
  );
}

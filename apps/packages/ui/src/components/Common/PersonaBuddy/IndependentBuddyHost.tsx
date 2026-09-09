import React from "react"
import { Button, Modal } from "antd"
import { Settings, Move, RotateCcw, Unplug } from "lucide-react"
import { useTranslation } from "react-i18next"
import { useSafeDemoMode } from "@/context/demo-mode"
import { useCanonicalConnectionConfig } from "@/hooks/useCanonicalConnectionConfig"
import { useBuddyManagementStore } from "@/store/buddy-management"
import {
  clampPersonaBuddyShellPosition,
  usePersonaBuddyShellStore
} from "@/store/persona-buddy-shell"
import {
  BUDDY_PAGE_SIZE,
  buddyAssets,
  detachBuddy,
  getBuddy,
  getBuddyAttachment,
  listBuddies,
  listBuddyConversations,
  updateBuddy,
  type BuddyAttachmentState,
  type BuddyProfile
} from "@/services/buddies"
import type { ServerChatSummary } from "@/services/tldw/TldwApiClient"
import type { PersonaVisualStateId } from "@/types/persona-visuals"
import { BuddyManagementModal } from "./BuddyManagementModal"
import { BuddyInteraction } from "./BuddyInteraction"
import { SpriteFrameRenderer } from "./SpriteFrameRenderer"
import type { BuddyDraftState } from "./buddy-drafts"

const emptyAttachment: BuddyAttachmentState = {
  client_slot: "default",
  version: 0,
  attachment: null
}
export const IndependentBuddySession = ({
  root = "web"
}: {
  root?: "web" | "sidepanel"
}) => {
  const { t } = useTranslation("sidepanel")
  const label = (key: string, text: string) =>
    t(`buddyManagement.${key}`, { defaultValue: text })
  const management = useBuddyManagementStore()
  const setAttached = useBuddyManagementStore((state) => state.setAttached)
  const [profiles, setProfiles] = React.useState<BuddyProfile[]>([])
  const [attachment, setAttachment] = React.useState(emptyAttachment)
  const [conversations, setConversations] = React.useState<ServerChatSummary[]>(
    []
  )
  const [selectedId, setSelectedId] = React.useState("")
  const [open, setOpen] = React.useState(false)
  const [loaded, setLoaded] = React.useState(false)
  const [hasLoaded, setHasLoaded] = React.useState(false)
  const [fetching, setFetching] = React.useState(false)
  const [profilePages, setProfilePages] = React.useState(1)
  const [conversationPages, setConversationPages] = React.useState(1)
  const [hasMoreProfiles, setHasMoreProfiles] = React.useState(false)
  const [hasMoreConversations, setHasMoreConversations] = React.useState(false)
  const [loadedScope, setLoadedScope] = React.useState("")
  const [drafts, setDrafts] = React.useState<Record<string, string>>({})
  const requestKeys = React.useRef(
    new Map<string, { text: string; key: string }>()
  )
  const draftState: BuddyDraftState = React.useMemo(
    () => ({ drafts, setDrafts, requestKeys }),
    [drafts]
  )
  const [error, setError] = React.useState<string | null>(null)
  const [revision, setRevision] = React.useState(0)
  const [attention, setAttention] = React.useState(0)
  const [visualState, setVisualState] =
    React.useState<PersonaVisualStateId>("idle")
  const [reducedMotion, setReducedMotion] = React.useState(false)
  const [viewport, setViewport] = React.useState({ width: 1024, height: 768 })
  const bucket = root === "sidepanel" ? "sidepanel-desktop" : "web-desktop"
  const position = usePersonaBuddyShellStore((s) => s.positions[bucket])
  const drag = React.useRef<{
    pointerX: number
    pointerY: number
    x: number
    y: number
  } | null>(null)
  const replyTargetId = React.useId()
  const expressionsId = React.useId()
  const alive = React.useRef(true)
  React.useEffect(() => {
    alive.current = true
    return () => {
      alive.current = false
    }
  }, [])
  const setPosition = usePersonaBuddyShellStore((s) => s.setPosition)
  const resetPosition = usePersonaBuddyShellStore((s) => s.resetPosition)
  React.useEffect(() => {
    const resize = () =>
      setViewport({ width: window.innerWidth, height: window.innerHeight })
    resize()
    window.addEventListener("resize", resize)
    const query = window.matchMedia("(prefers-reduced-motion: reduce)")
    const changed = () => setReducedMotion(query.matches)
    changed()
    query.addEventListener("change", changed)
    return () => {
      window.removeEventListener("resize", resize)
      query.removeEventListener("change", changed)
    }
  }, [])
  React.useEffect(() => {
    let active = true
    let pending = false
    const refresh = async () => {
      if (pending) return
      pending = true
      setFetching(true)
      try {
        const [collections, next] = await Promise.all([
          Promise.all(
            Array.from({ length: profilePages }, (_, index) =>
              listBuddies({
                limit: BUDDY_PAGE_SIZE,
                offset: index * BUDDY_PAGE_SIZE
              })
            )
          ),
          getBuddyAttachment()
        ])
        if (!active) return
        setAttached(Boolean(next.attachment || next.unavailable_reason))
        const nextScope = next.attachment
          ? `${next.attachment.scope_type}:${next.attachment.scope_id}`
          : ""
        const nextConversationPages =
          nextScope === loadedScope ? conversationPages : 1
        const profiles = [
          ...new Map(
            collections
              .flatMap((page) => page.buddies)
              .map((profile) => [profile.id, profile])
          ).values()
        ]
        const attachedProfile =
          next.attachment &&
          !profiles.some((profile) => profile.id === next.attachment!.buddy_id)
            ? await getBuddy(next.attachment.buddy_id).catch((error) => {
                if (active) {
                  // Keep the independent attachment authoritative while artwork is
                  // unavailable; never fall back to an unrelated legacy Buddy.
                  setAttachment({
                    ...next,
                    attachment: null,
                    unavailable_reason: "buddy_unavailable"
                  })
                  setProfiles([])
                }
                throw error
              })
            : null
        if (!active) return
        const pages = next.attachment
          ? await Promise.all(
              Array.from({ length: nextConversationPages }, (_, index) =>
                listBuddyConversations({
                  limit: BUDDY_PAGE_SIZE,
                  offset: index * BUDDY_PAGE_SIZE
                })
              )
            )
          : []
        if (!active) return
        if (attachedProfile) profiles.push(attachedProfile)
        const rows = [
          ...new Map(
            pages
              .flatMap((page) => page.conversations)
              .map((conversation) => [conversation.id, conversation])
          ).values()
        ]
        setProfiles((previous) => {
          const byId = new Map(previous.map((profile) => [profile.id, profile]))
          return profiles.map((profile) => {
            const existing = byId.get(profile.id)
            // Unchanged polls must not reset the renderer's animation timeline.
            // Persona availability can change independently of the profile version.
            return existing &&
              existing.version === profile.version &&
              existing.optional_persona_available ===
                profile.optional_persona_available
              ? existing
              : profile
          })
        })
        setAttachment(next)
        setConversations(rows)
        setError(null)
        setHasMoreProfiles(
          collections.at(-1)!.buddies.length === BUDDY_PAGE_SIZE
        )
        setHasMoreConversations(
          next.attachment?.scope_type === "workspace" &&
            pages.at(-1)?.conversations.length === BUDDY_PAGE_SIZE
        )
        setConversationPages(nextConversationPages)
        setLoadedScope(nextScope)
        setHasLoaded(true)
        setAttached(Boolean(next.attachment || next.unavailable_reason))
      } catch (e) {
        if (active) {
          setError(e instanceof Error ? e.message : "Buddy unavailable")
          setConversations([])
        }
      } finally {
        pending = false
        if (active) {
          setLoaded(true)
          setFetching(false)
        }
      }
    }
    void refresh()
    const timer = window.setInterval(() => void refresh(), 5000)
    return () => {
      active = false
      window.clearInterval(timer)
    }
  }, [revision, profilePages, conversationPages, loadedScope, setAttached])
  React.useEffect(
    () => () => {
      useBuddyManagementStore.getState().setAttached(false)
      useBuddyManagementStore.getState().close()
    },
    []
  )
  const binding = attachment.attachment
  const profile = profiles.find((p) => p.id === binding?.buddy_id)
  const assets = React.useMemo(
    () => (profile ? buddyAssets(profile) : {}),
    [profile]
  )
  const bindingKey = binding
    ? `${binding.buddy_id}:${binding.scope_type}:${binding.scope_id}`
    : ""
  React.useEffect(() => {
    setSelectedId(
      binding?.scope_type === "conversation" ? binding.scope_id : ""
    )
  }, [binding?.scope_id, binding?.scope_type])
  const conversation = conversations.find((c) => c.id === selectedId) ?? null
  const targetName =
    attachment.target?.title ||
    conversation?.title ||
    label("targetUnavailable", "Target unavailable")
  const clamp = (value: { x: number; y: number }) =>
    clampPersonaBuddyShellPosition(value, bucket, {
      viewportWidth: viewport.width,
      viewportHeight: viewport.height,
      shellWidth: 132,
      shellHeight: 174,
      margin: 16
    })
  const clamped = clamp(position)
  const refresh = () => setRevision((r) => r + 1)
  const changeMode = async (mode: "dynamic" | "static") => {
    if (!profile) return
    try {
      await updateBuddy(profile.id, {
        expected_version: profile.version,
        display_mode: mode
      })
      if (alive.current) refresh()
    } catch (e) {
      if (alive.current)
        setError(e instanceof Error ? e.message : "Could not save display mode")
    }
  }
  const detach = async () => {
    try {
      const next = await detachBuddy(attachment.version)
      if (!alive.current) return
      setAttachment(next)
      setOpen(false)
      management.setAttached(false)
      refresh()
    } catch (e) {
      if (alive.current)
        setError(e instanceof Error ? e.message : "Could not detach Buddy")
    }
  }
  return (
    <>
      {profile && binding ? (
        <div
          className="fixed z-40 flex w-32 flex-col items-center"
          style={{ left: clamped.x, top: clamped.y }}
          data-testid="independent-buddy"
        >
          <button
            type="button"
            onClick={() => setOpen(true)}
            aria-label={`${label("open", "Open")} ${profile.name} — ${targetName}`}
            className="flex min-h-24 w-28 items-center justify-center rounded-md focus-visible:outline focus-visible:outline-2 focus-visible:outline-primary"
          >
            <SpriteFrameRenderer
              manifest={profile.manifest}
              assets={assets}
              state={visualState}
              fitSize={112}
              animate={profile.display_mode === "dynamic" && !reducedMotion}
              fallbackLabel={profile.name}
              className="max-h-28 max-w-28 object-contain"
            />
          </button>
          <span
            className="max-w-full truncate rounded bg-surface px-2 text-xs text-text"
            title={targetName}
          >
            {targetName}
          </span>
          {attention > 0 ? (
            <span
              className="rounded bg-surface px-2 text-xs font-semibold text-text"
              role="status"
            >
              {attention} {label("newResponses", "new responses")}
            </span>
          ) : null}
          <button
            type="button"
            aria-label={label("move", "Move Buddy with arrow keys")}
            title={label(
              "moveHint",
              "Drag to move, or focus and use arrow keys. Home resets position."
            )}
            className="mt-1 touch-none cursor-move rounded bg-surface p-1 text-text-muted focus-visible:outline focus-visible:outline-2 focus-visible:outline-primary"
            onPointerDown={(event) => {
              if (event.button !== 0) return
              event.preventDefault()
              event.currentTarget.focus()
              event.currentTarget.setPointerCapture?.(event.pointerId)
              drag.current = {
                pointerX: event.clientX,
                pointerY: event.clientY,
                x: clamped.x,
                y: clamped.y
              }
            }}
            onPointerMove={(event) => {
              if (!drag.current) return
              setPosition(
                bucket,
                clamp({
                  x: drag.current.x + event.clientX - drag.current.pointerX,
                  y: drag.current.y + event.clientY - drag.current.pointerY
                })
              )
            }}
            onPointerUp={(event) => {
              drag.current = null
              if (event.currentTarget.hasPointerCapture?.(event.pointerId))
                event.currentTarget.releasePointerCapture(event.pointerId)
            }}
            onPointerCancel={() => {
              drag.current = null
            }}
            onKeyDown={(event) => {
              const delta = event.shiftKey ? 40 : 10
              const moves: Record<string, [number, number]> = {
                ArrowLeft: [-delta, 0],
                ArrowRight: [delta, 0],
                ArrowUp: [0, -delta],
                ArrowDown: [0, delta]
              }
              if (event.key === "Home") {
                event.preventDefault()
                resetPosition(bucket)
                return
              }
              const change = moves[event.key]
              if (change) {
                event.preventDefault()
                setPosition(
                  bucket,
                  clamp({ x: clamped.x + change[0], y: clamped.y + change[1] })
                )
              }
            }}
          >
            <Move size={16} aria-hidden="true" />
          </button>
        </div>
      ) : null}
      {attachment.unavailable_reason && !management.open ? (
        <button
          type="button"
          className="fixed bottom-4 right-4 z-40 rounded-md border border-border bg-surface p-3 text-sm text-text"
          onClick={() => management.show()}
        >
          {label("repair", "Buddy attachment needs attention")}
        </button>
      ) : null}
      {binding && profile ? (
        <Modal
          open={open}
          onCancel={() => setOpen(false)}
          title={
            <span
              className="block break-words pr-6"
              style={{ overflowWrap: "anywhere" }}
            >
              {profile.name} — {targetName}
            </span>
          }
          footer={null}
          width={620}
          style={{ maxWidth: "calc(100vw - 24px)", top: 24 }}
          destroyOnHidden={false}
          forceRender
          classNames={{
            container: "!bg-surface !text-text",
            header: "!bg-surface !text-text",
            title: "!text-text",
            body: "!bg-surface !text-text"
          }}
          styles={{
            body: { maxHeight: "calc(100dvh - 140px)", overflowY: "auto" }
          }}
        >
          <div className="space-y-3">
            <p className="text-sm text-text-muted">
              {binding.scope_type === "workspace"
                ? label("workspaceScope", "Attached to workspace")
                : label("conversationScope", "Attached to conversation")}
              : {targetName}
            </p>
            {error ? (
              <p role="alert" className="text-sm text-danger">
                {error}
              </p>
            ) : null}
            {binding.scope_type === "workspace" ? (
              <div>
                <label
                  htmlFor={replyTargetId}
                  className="block text-sm text-text"
                >
                  {label("replyTarget", "Reply to conversation")}
                </label>
                <select
                  id={replyTargetId}
                  className="mt-1 block w-full rounded-md border border-border bg-surface p-2 text-text"
                  value={selectedId}
                  onChange={(e) => setSelectedId(e.target.value)}
                >
                  <option value="">
                    {label("chooseConversation", "Choose a conversation")}
                  </option>
                  {conversations.map((c) => (
                    <option key={c.id} value={c.id}>
                      {c.title}
                    </option>
                  ))}
                </select>
                {hasMoreConversations ? (
                  <Button
                    className="mt-2"
                    loading={fetching}
                    onClick={() => setConversationPages((pages) => pages + 1)}
                  >
                    {label("moreConversations", "Load more conversations")}
                  </Button>
                ) : null}
              </div>
            ) : null}
            <BuddyInteraction
              key={bindingKey}
              draftState={draftState}
              conversationPages={conversationPages}
              attachment={binding}
              conversation={conversation}
              conversations={conversations}
              visible={open}
              attachmentVersion={attachment.version}
              onSelectConversation={setSelectedId}
              onAttentionChange={setAttention}
              onVisualStateChange={setVisualState}
            />
            <details className="border-t border-border pt-3 text-sm text-text">
              <summary className="cursor-pointer font-medium">
                {label("options", "Buddy options")}
              </summary>
              <div className="mt-3 flex flex-wrap items-center gap-2">
                <label htmlFor={expressionsId}>
                  {label("expressions", "Expressions")}
                </label>{" "}
                <select
                  id={expressionsId}
                  value={profile.display_mode}
                  onChange={(e) =>
                    void changeMode(e.target.value as "static" | "dynamic")
                  }
                  className="rounded border border-border bg-surface p-2"
                >
                  <option value="dynamic">{label("dynamic", "Dynamic")}</option>
                  <option value="static">{label("static", "Static")}</option>
                </select>
                {reducedMotion ? (
                  <p>
                    {label(
                      "reducedMotion",
                      "Reduced motion is on; expressions remain static."
                    )}
                  </p>
                ) : null}
                <Button
                  icon={<RotateCcw size={16} />}
                  onClick={() => resetPosition(bucket)}
                >
                  {label("resetPosition", "Reset position")}
                </Button>
                <Button
                  icon={<Settings size={16} />}
                  onClick={() => {
                    setOpen(false)
                    management.show()
                  }}
                >
                  {label("manage", "Manage Buddy & Persona")}
                </Button>
                <Button
                  icon={<Unplug size={16} />}
                  onClick={() => void detach()}
                >
                  {label("detach", "Detach Buddy")}
                </Button>
              </div>
            </details>
          </div>
        </Modal>
      ) : null}
      {management.open && loaded && !hasLoaded && error ? (
        <Modal
          open
          footer={null}
          title={label("title", "Buddy & Persona Management")}
          onCancel={management.close}
        >
          <p role="alert" className="mb-3 text-danger">
            {error}
          </p>
          <Button loading={fetching} onClick={refresh}>
            {label("retry", "Retry")}
          </Button>
        </Modal>
      ) : null}
      {management.open && hasLoaded ? (
        <BuddyManagementModal
          profiles={profiles}
          hasMoreProfiles={hasMoreProfiles}
          loadingMoreProfiles={fetching}
          onLoadMoreProfiles={() => setProfilePages((pages) => pages + 1)}
          collectionError={error}
          onRetryCollections={refresh}
          attachment={attachment}
          target={management.target}
          onClose={management.close}
          onApplied={refresh}
          onConversationSettings={management.conversationSettings}
        />
      ) : null}
      {management.open && !loaded ? (
        <Modal
          open
          footer={null}
          title={label("title", "Buddy & Persona Management")}
          onCancel={management.close}
        >
          <p role="status">{label("loading", "Loading available choices…")}</p>
        </Modal>
      ) : null}
    </>
  )
}

export const IndependentBuddyHost = ({
  root = "web"
}: {
  root?: "web" | "sidepanel"
}) => {
  const { demoEnabled } = useSafeDemoMode()
  const { config, loading } = useCanonicalConnectionConfig()
  // Authentication changes create a new ephemeral UI lifetime. No credential or
  // conversation identity is written to localStorage or exposed in a React key.
  const identity = [
    config?.serverUrl,
    config?.authMode,
    config?.apiKey,
    config?.accessToken,
    config?.authSource,
    config?.orgId
  ]
  const previous = React.useRef({ identity, generation: 0 })
  if (
    previous.current.identity.some((value, index) => value !== identity[index])
  ) {
    previous.current = { identity, generation: previous.current.generation + 1 }
  }
  if (demoEnabled || loading || !config?.serverUrl) return null
  return (
    <IndependentBuddySession key={previous.current.generation} root={root} />
  )
}

import React from "react"
import { Button, Modal, Tooltip } from "antd"
import { Info } from "lucide-react"
import { useTranslation } from "react-i18next"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import type { WorkspaceApiResponse } from "@/services/tldw/domains/workspace-api"
import { listPersonaVisualStarterPacks } from "@/services/persona-visuals"
import {
  buddyAssets,
  createBuddy,
  importBuddyArchive,
  previewBuddyArchive,
  putBuddyAttachment,
  resolveBuddyConversationTarget,
  type BuddyAttachmentState,
  type ReviewedBuddyArchive,
  type BuddyProfile,
  type BuddyConversationSummary
} from "@/services/buddies"
import type { BuddyManagementTarget } from "@/store/buddy-management"
import type { PersonaVisualStarterPackSummary } from "@/types/persona-visuals"
import { BuddyStarterArtwork } from "@/components/PersonaGarden/BuddyStarterArtwork"
import { SpriteFrameRenderer } from "./SpriteFrameRenderer"
import { buddyConversationLabels } from "./buddy-conversation-labels"

const fieldClass =
  "block w-full rounded-md border border-border bg-surface px-3 py-2 text-sm text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-primary"
const StarterChoice = ({
  starter,
  selected,
  disabled,
  onSelect
}: {
  starter: PersonaVisualStarterPackSummary
  selected: boolean
  disabled: boolean
  onSelect: () => void
}) => {
  const { t } = useTranslation("sidepanel")
  const [ready, setReady] = React.useState(false)
  const licenceId = React.useId()
  return (
    <div
      className={`overflow-hidden rounded-md border ${selected ? "border-primary" : "border-border"}`}
    >
      <BuddyStarterArtwork
        starterId={starter.id}
        title={starter.title}
        onReadyChange={setReady}
      />
      <div className="space-y-2 p-3">
        <div className="flex items-center gap-1">
          <h4 className="font-semibold">{starter.title}</h4>
          {/* The licence caption is provenance, not the choice (XP-14). A tap
              opens it on touch, and the button's description reads it out. */}
          {starter.license_label ? (
            <>
              <Tooltip title={starter.license_label} trigger={["hover", "focus", "click"]}>
                {/* The Tooltip clears aria-describedby on its own child while
                    closed, so it wraps the button rather than being it. */}
                <span className="inline-flex">
                  <button
                    type="button"
                    className="inline-flex size-6 items-center justify-center rounded text-text-muted hover:text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-primary"
                    aria-label={`${t("buddyManagement.licenceInfo", { defaultValue: "Artwork licence" })}: ${starter.title}`}
                    aria-describedby={licenceId}
                  >
                    <Info className="size-4" aria-hidden="true" />
                  </button>
                </span>
              </Tooltip>
              <span id={licenceId} className="sr-only">
                {starter.license_label}
              </span>
            </>
          ) : null}
        </div>
        <Button disabled={disabled || !ready} onClick={onSelect}>
          {selected
            ? t("buddyManagement.selected", { defaultValue: "Selected" })
            : t("buddyManagement.useBuddy", { defaultValue: "Use this avatar" })}
        </Button>
      </div>
    </div>
  )
}
export const BuddyManagementModal = ({
  profiles,
  attachment,
  target,
  onClose,
  onApplied,
  onConversationSettings,
  hasMoreProfiles = false,
  loadingMoreProfiles = false,
  onLoadMoreProfiles,
  collectionError,
  onRetryCollections
}: {
  profiles: BuddyProfile[]
  attachment: BuddyAttachmentState
  target: BuddyManagementTarget | null
  onClose: () => void
  onApplied: () => void | Promise<void>
  onConversationSettings?: (() => void) | null
  hasMoreProfiles?: boolean
  loadingMoreProfiles?: boolean
  onLoadMoreProfiles?: () => void
  collectionError?: string | null
  onRetryCollections?: () => void
}) => {
  const { t, i18n } = useTranslation("sidepanel")
  const label = (key: string, text: string) =>
    t(`buddyManagement.${key}`, { defaultValue: text })
  const mounted = React.useRef(true)
  const uploadLifetime = React.useRef<AbortController | null>(null)
  const targetGeneration = React.useRef(0)
  React.useLayoutEffect(() => {
    targetGeneration.current += 1
  }, [target?.scope_type, target?.scope_id])
  React.useEffect(() => {
    const controller = new AbortController()
    uploadLifetime.current = controller
    mounted.current = true
    return () => {
      mounted.current = false
      controller.abort()
    }
  }, [])
  const [createdArtwork, setCreatedArtwork] = React.useState<{
    sourceKey: string
    profile: BuddyProfile
  } | null>(null)
  const createdProfile = createdArtwork?.profile
  const availableProfiles =
    createdProfile && !profiles.some((p) => p.id === createdProfile.id)
      ? [...profiles, createdProfile]
      : profiles
  const [selected, setSelected] = React.useState(
    attachment.attachment?.buddy_id ?? ""
  )
  const [starter, setStarter] =
    React.useState<PersonaVisualStarterPackSummary | null>(null)
  const [reviewedArchive, setReviewedArchive] = React.useState<ReviewedBuddyArchive | null>(null)
  const [reviewingArchive, setReviewingArchive] = React.useState(false)
  const [starters, setStarters] = React.useState<
    PersonaVisualStarterPackSummary[]
  >([])
  const [scope, setScope] = React.useState<"conversation" | "workspace">(
    target?.scope_type ?? attachment.attachment?.scope_type ?? "conversation"
  )
  const [scopeId, setScopeId] = React.useState(
    target?.scope_id ?? attachment.attachment?.scope_id ?? ""
  )
  const [location, setLocation] = React.useState(
    attachment.target?.workspace_id ?? ""
  )
  const [chats, setChats] = React.useState<BuddyConversationSummary[]>([])
  const conversationLabels = React.useMemo(
    () => buddyConversationLabels(chats, i18n.resolvedLanguage),
    [chats, i18n.resolvedLanguage]
  )
  const [chatOffset, setChatOffset] = React.useState(0)
  const [hasMoreChats, setHasMoreChats] = React.useState(false)
  const [loadingChats, setLoadingChats] = React.useState(true)
  const [resolvedTarget, setResolvedTarget] =
    React.useState<BuddyConversationSummary | null>(null)
  const [resolvingTarget, setResolvingTarget] = React.useState(
    target?.scope_type === "conversation"
  )
  const [search, setSearch] = React.useState("")
  const [workspaces, setWorkspaces] = React.useState<WorkspaceApiResponse[]>([])
  const [personas, setPersonas] = React.useState<
    { id: string; name: string }[]
  >([])
  const [workspace, setWorkspace] = React.useState<WorkspaceApiResponse | null>(
    null
  )
  const [defaultPersona, setDefaultPersona] = React.useState("")
  const [defaultDirty, setDefaultDirty] = React.useState(false)
  const [applyAttempt, setApplyAttempt] = React.useState<{ target: BuddyManagementTarget | null } | null>(null)
  const busy = applyAttempt !== null
  const draftLocked = busy &&
    applyAttempt.target?.scope_type === target?.scope_type &&
    applyAttempt.target?.scope_id === target?.scope_id
  const [loading, setLoading] = React.useState(true)
  const [error, setError] = React.useState<string | null>(null)
  const [notice, setNotice] = React.useState<string | null>(null)
  const targetScope = target?.scope_type
  const targetId = target?.scope_id
  React.useEffect(() => {
    if (targetScope === undefined || targetId === undefined) return
    setScope(targetScope)
    setScopeId(targetId)
    setResolvedTarget(null)
    setLocation(targetScope === "workspace" ? targetId : "")
    setChatOffset(0)
    setSearch("")
    setError(null)
    setNotice(null)
    setResolvingTarget(targetScope === "conversation")
    if (targetScope !== "conversation") return
    let active = true
    setResolvingTarget(true)
    void resolveBuddyConversationTarget(targetId)
      .then((result) => {
        if (!active) return
        setResolvedTarget(result)
        setLocation(result.workspace_id ?? "")
        setChatOffset(0)
      })
      .catch((e) => {
        if (active) {
          setScopeId("")
          setError(e.message)
        }
      })
      .finally(() => {
        if (active) setResolvingTarget(false)
      })
    return () => {
      active = false
    }
  }, [targetScope, targetId])
  React.useEffect(() => {
    let active = true
    setLoading(true)
    void Promise.all([
      tldwClient.listWorkspaces(),
      listPersonaVisualStarterPacks(),
      tldwClient
        .fetchWithAuth("/api/v1/persona/catalog", { method: "GET" })
        .then(async (r) => (r.ok ? r.json() : null))
    ])
      .then(([ws, catalog, people]) => {
        if (!active) return
        setWorkspaces(ws.items.filter((w) => !w.deleted && !w.archived))
        setStarters(
          catalog.starter_packs.filter(
            (s) => s.production_status === "art_ready"
          )
        )
        const entries = Array.isArray(people)
          ? people
          : (people?.personas ?? people?.profiles ?? [])
        setPersonas(
          entries.map((p: { id: string; name: string }) => ({
            id: String(p.id),
            name: p.name
          }))
        )
      })
      .catch((e) => {
        if (active) setError(e.message)
      })
      .finally(() => {
        if (active) setLoading(false)
      })
    return () => {
      active = false
    }
  }, [])
  React.useEffect(() => {
    let active = true
    setLoadingChats(true)
    void tldwClient
      .listChats(
        { limit: 100, offset: chatOffset },
        {
          scope: location
            ? { type: "workspace", workspaceId: location }
            : { type: "global" }
        }
      )
      .then((value) => {
        if (!active) return
        setChats((previous) => [
          ...new Map(
            [
              ...(chatOffset ? previous : []),
              ...value,
              ...(resolvedTarget &&
              (resolvedTarget.workspace_id ?? "") === location
                ? [resolvedTarget]
                : [])
            ].map((chat) => [chat.id, chat])
          ).values()
        ])
        setHasMoreChats(value.length === 100)
      })
      .catch((e) => {
        if (active) setError(e.message)
      })
      .finally(() => {
        if (active) setLoadingChats(false)
      })
    return () => {
      active = false
    }
  }, [location, resolvedTarget, chatOffset])
  React.useEffect(() => {
    let active = true
    setWorkspace(null)
    setDefaultDirty(false)
    if (scope !== "workspace" || !scopeId) return
    void tldwClient
      .getWorkspace(scopeId)
      .then((ws) => {
        if (!active) return
        setWorkspace(ws)
        setDefaultPersona(
          ws.assistant_defaults?.assistant_id ??
            ws.assistantDefaults?.assistantId ??
            ""
        )
      })
      .catch((e) => {
        if (active) setError(e.message)
      })
    return () => {
      active = false
    }
  }, [scope, scopeId])
  const reviewArchive = async (file: File) => {
    setReviewedArchive(null)
    setReviewingArchive(true)
    setError(null)
    setNotice(null)
    try {
      const reviewed = await previewBuddyArchive(file, uploadLifetime.current?.signal)
      if (!mounted.current) return
      setReviewedArchive(reviewed)
      setStarter(null)
      setSelected("")
      setSearch("")
    } catch (e) {
      if (mounted.current) {
        setError(e instanceof Error ? e.message : label("reviewFailed", "Could not review the avatar pack."))
      }
    } finally {
      if (mounted.current) setReviewingArchive(false)
    }
  }
  const apply = async () => {
    if (
      busy ||
      reviewingArchive ||
      resolvingTarget ||
      !scopeId ||
      (!selected && !starter && !reviewedArchive) ||
      (scope === "conversation" &&
        (loadingChats || !chats.some((c) => c.id === scopeId)))
    )
      return
    setApplyAttempt({ target })
    setError(null)
    setNotice(null)
    const generation = targetGeneration.current
    let artworkSaved = false
    let defaultSaved = false
    let attachmentSaved = false
    const stopForRetarget = (failure?: unknown) => {
      if (!mounted.current) return true
      if (generation === targetGeneration.current) return false
      setNotice(
        [
          artworkSaved &&
            label("artworkSaved", "Avatar artwork saved to Your avatars."),
          defaultSaved &&
            label(
              "previousDefaultSaved",
              "The previous workspace default was saved."
            ),
          attachmentSaved &&
            label(
              "previousAttachmentSaved",
              "The previous avatar attachment was saved."
            ),
          failure &&
            label("earlierApplyFailed", "The earlier Apply could not finish."),
          label(
            "targetChanged",
            "Target changed. Your new selection is retained; review it before applying."
          )
        ]
          .filter(Boolean)
          .join(" ")
      )
      return true
    }
    try {
      let buddyId = selected
      if (starter || reviewedArchive) {
        const sourceKey = reviewedArchive
          ? "archive:" + reviewedArchive.preview.archive_sha256
          : "starter:" + starter?.id
        let created = createdArtwork?.sourceKey === sourceKey ? createdArtwork.profile : null
        if (!created && reviewedArchive) {
          created = await importBuddyArchive(reviewedArchive, uploadLifetime.current?.signal)
        }
        if (!created && starter) {
          created = await createBuddy({
            name: starter.title,
            source: { kind: "starter", starter_id: starter.id },
            optional_persona_id: null
          })
        }
        if (!mounted.current || !created) return
        artworkSaved = true
        setCreatedArtwork({ sourceKey, profile: created })
        if (stopForRetarget()) return
        buddyId = created.id
        // Retain the created profile on retry rather than creating duplicates.
        setSelected(created.id)
        setStarter(null)
        setReviewedArchive(null)
      }
      if (scope === "workspace" && defaultDirty && workspace) {
        const updated = await tldwClient.patchWorkspace(scopeId, {
          version: workspace.version,
          assistant_defaults: defaultPersona
            ? {
                assistant_kind: "persona",
                assistant_id: defaultPersona,
                persona_memory_mode: "read_only"
              }
            : null
        })
        defaultSaved = true
        if (stopForRetarget()) return
        setWorkspace(updated)
        setDefaultDirty(false)
      }
      await putBuddyAttachment({
        expected_version: attachment.version,
        buddy_id: buddyId,
        scope_type: scope,
        scope_id: scopeId
      })
      attachmentSaved = true
      if (stopForRetarget()) {
        // The write already committed. Refresh its version without replacing
        // the new draft or closing the editor for the new target.
        if (mounted.current) await onApplied()
        return
      }
      await onApplied()
      if (stopForRetarget()) return
      onClose()
    } catch (e) {
      if (stopForRetarget(e)) return
      if (reviewedArchive && artworkSaved && !attachmentSaved) {
        onRetryCollections?.()
      }
      const savedMessage = attachmentSaved
        ? label("attachmentSaved", "Avatar attachment saved. Refresh failed: ")
        : defaultSaved
          ? label(
              "defaultSaved",
              "Workspace default saved. Avatar attachment was not confirmed: "
            )
          : artworkSaved
            ? label(
                "artworkSavedRetry",
                "Avatar saved to Your avatars. Refresh and try attaching it again. "
              )
            : ""
      setError(
        `${savedMessage}${e instanceof Error ? e.message : label("saveFailed", "Could not apply. Your selection is retained; refresh and retry.")}`
      )
    } finally {
      if (mounted.current) setApplyAttempt(null)
    }
  }
  const stagedChanges = Boolean(
    starter ||
    reviewingArchive ||
    reviewedArchive ||
    selected !== (attachment.attachment?.buddy_id ?? "") ||
    scope !==
      (target?.scope_type ??
        attachment.attachment?.scope_type ??
        "conversation") ||
    scopeId !== (target?.scope_id ?? attachment.attachment?.scope_id ?? "") ||
    defaultDirty
  )
  return (
    <Modal
      open
      title={label("title", "Avatar settings")}
      width={760}
      style={{ maxWidth: "calc(100vw - 24px)", top: 24 }}
      styles={{
        container: { background: "var(--surface)", color: "var(--text)" },
        header: { background: "var(--surface)" },
        title: {
          color: "var(--text)",
          paddingRight: 24,
          overflowWrap: "anywhere"
        },
        body: { maxHeight: "calc(100dvh - 190px)", overflowY: "auto" }
      }}
      onCancel={() => {
        if (!busy) onClose()
      }}
      maskClosable={!busy}
      closable={!busy}
      footer={
        <>
          <Button onClick={onClose} disabled={busy}>
            {label("cancel", "Cancel")}
          </Button>
          <Button
            type="primary"
            aria-label={label("apply", "Apply")}
            loading={busy}
            disabled={
              loading ||
              resolvingTarget ||
              !scopeId ||
              reviewingArchive ||
              (!selected && !starter && !reviewedArchive) ||
              (scope === "conversation" &&
                (loadingChats || !chats.some((c) => c.id === scopeId))) ||
              (scope === "workspace" && !workspace)
            }
            onClick={() => void apply()}
          >
            {label("apply", "Apply")}
          </Button>
        </>
      }
    >
      <div className="space-y-5 text-text">
        <p className="text-sm text-text-muted">
          {label(
            "intro",
            "Choose an avatar and where it belongs. The avatar’s artwork doesn’t change the conversation’s Persona."
          )}
        </p>
        {onConversationSettings &&
        (!scopeId || scopeId === target?.scope_id) ? (
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border pb-3">
            <p className="text-sm">
              {label("currentPersona", "Conversation Persona")}:{" "}
              {resolvedTarget?.assistant_name ??
                (resolvedTarget?.assistant_kind === "persona"
                  ? label("personaUnavailable", "Persona unavailable")
                  : label("none", "None"))}
            </p>
            <Button
              disabled={busy || stagedChanges}
              title={
                stagedChanges
                  ? label(
                      "finishSelection",
                      "Apply or cancel your avatar changes before editing conversation settings."
                    )
                  : undefined
              }
              onClick={() => {
                onClose()
                onConversationSettings()
              }}
            >
              {label(
                "editConversation",
                "Edit conversation Persona & behavior"
              )}
            </Button>
          </div>
        ) : null}
        {collectionError ? (
          <div role="alert" className="text-sm text-danger">
            {collectionError}{" "}
            <Button size="small" onClick={onRetryCollections}>
              {label("retry", "Retry")}
            </Button>
          </div>
        ) : null}
        {error ? (
          <p role="alert" className="text-sm text-danger">
            {error}
          </p>
        ) : null}
        {notice ? (
          <p role="status" className="text-sm text-text-muted">
            {notice}
          </p>
        ) : null}
        {loading ? (
          <p role="status">{label("loading", "Loading available choices…")}</p>
        ) : null}
        <label className="block text-sm">
          {label("searchBuddies", "Find an avatar")}
          <input
            type="search"
            className={fieldClass}
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </label>
        <section aria-label={label("yourBuddies", "Your avatars")}>
          <h3 className="mb-2 text-base font-semibold">
            {label("yourBuddies", "Your avatars")}
          </h3>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
            {availableProfiles
              .filter((p) =>
                p.name.toLowerCase().includes(search.toLowerCase())
              )
              .map((p) => (
                <button
                  key={p.id}
                  type="button"
                  disabled={draftLocked || reviewingArchive}
                  aria-pressed={selected === p.id && !starter}
                  aria-label={`${label("use", "Use")} ${p.name}`}
                  onClick={() => {
                    setSelected(p.id)
                    setStarter(null)
                    setReviewedArchive(null)
                  }}
                  className={`min-w-0 rounded-md border p-3 text-sm focus-visible:outline focus-visible:outline-2 focus-visible:outline-primary ${selected === p.id && !starter ? "border-primary bg-primary/10" : "border-border bg-surface"}`}
                >
                  <div className="flex h-32 items-center justify-center overflow-hidden">
                    <SpriteFrameRenderer
                      manifest={p.manifest}
                      assets={buddyAssets(p)}
                      state="idle"
                      animate={false}
                      fitSize={128}
                      fallbackLabel={p.name}
                      className="max-h-32 max-w-full object-contain"
                    />
                  </div>
                  <span className="mt-2 block font-medium">{p.name}</span>
                </button>
              ))}
          </div>
          {hasMoreProfiles ? (
            <Button
              className="mt-3"
              loading={loadingMoreProfiles}
              onClick={onLoadMoreProfiles}
            >
              {label("loadMoreBuddies", "Load more avatars")}
            </Button>
          ) : null}
        </section>
        <details open={!availableProfiles.length}>
          <summary className="cursor-pointer text-base font-semibold">
            {label("gallery", "Choose a ready-made avatar")}
          </summary>
          <div className="mt-3 grid gap-3 sm:grid-cols-2">
            {starters
              .filter((s) =>
                s.title.toLowerCase().includes(search.toLowerCase())
              )
              .map((s) => (
                <StarterChoice
                  key={s.id}
                  starter={s}
                  selected={starter?.id === s.id}
                  disabled={draftLocked || reviewingArchive}
                  onSelect={() => {
                    setStarter(s)
                    setSelected("")
                    setReviewedArchive(null)
                  }}
                />
              ))}
          </div>
        </details>
        <section className="space-y-2 border-t border-border pt-4">
          <h3 className="text-base font-semibold">
            {label("importAvatarPack", "Import an avatar pack")}
          </h3>
          <label className="block text-sm">
            {label("chooseAvatarPack", "Choose avatar pack")}
            <input
              type="file"
              accept=".tldw-persona-vpack"
              className={fieldClass}
              disabled={busy || reviewingArchive}
              onChange={(e) => {
                const file = e.currentTarget.files?.[0]
                e.currentTarget.value = ""
                if (file) void reviewArchive(file)
              }}
            />
          </label>
          <p className="text-sm text-text-muted">
            {label("importAvatarHelp", "Choose a .tldw-persona-vpack file, review it, then select Apply.")}
          </p>
          {reviewingArchive ? (
            <p role="status">{label("reviewingArchive", "Checking avatar pack…")}</p>
          ) : null}
          {reviewedArchive ? (
            <div className="space-y-2 rounded-md border border-primary bg-primary/10 p-3 text-sm">
              <p role="status" className="break-words font-medium">
                {reviewedArchive.preview.name}
              </p>
              <p>
                {t("buddyManagement.importAvatarContents", {
                  defaultValue: "{{assets}} artwork files · {{states}} states",
                  assets: reviewedArchive.preview.asset_count,
                  states: reviewedArchive.preview.state_count
                })}
              </p>
              {reviewedArchive.preview.artwork ? (
                <dl className="grid gap-x-3 gap-y-1 break-words sm:grid-cols-[max-content_minmax(0,1fr)]">
                  {reviewedArchive.preview.artwork.creator ? (
                    <>
                      <dt>{label("artworkCreator", "Creator")}</dt>
                      <dd>{reviewedArchive.preview.artwork.creator}</dd>
                    </>
                  ) : null}
                  {reviewedArchive.preview.artwork.license ? (
                    <>
                      <dt>{label("artworkLicense", "Licence")}</dt>
                      <dd>{reviewedArchive.preview.artwork.license}</dd>
                    </>
                  ) : null}
                  {reviewedArchive.preview.artwork.source_url ? (
                    <>
                      <dt>{label("artworkSource", "Source")}</dt>
                      <dd className="break-all">{reviewedArchive.preview.artwork.source_url}</dd>
                    </>
                  ) : null}
                </dl>
              ) : null}
              {reviewedArchive.preview.artwork?.notices ? (
                <details>
                  <summary className="cursor-pointer">{label("artworkNotices", "Artwork notices")}</summary>
                  <p className="max-h-40 overflow-y-auto whitespace-pre-wrap break-words">
                    {reviewedArchive.preview.artwork.notices}
                  </p>
                </details>
              ) : null}
              <Button size="small" disabled={busy} onClick={() => setReviewedArchive(null)}>
                {label("clearAvatarPack", "Remove pack selection")}
              </Button>
            </div>
          ) : null}
        </section>
        <section className="space-y-3 border-t border-border pt-4">
          <h3 className="text-base font-semibold">
            {label("attachment", "Attach to")}
          </h3>
          <label className="block text-sm">
            {label("scope", "Scope")}
            <select
              aria-label={label("scope", "Scope")}
              disabled={draftLocked}
              className={fieldClass}
              value={scope}
              onChange={(e) => {
                setScope(e.target.value as typeof scope)
                setScopeId("")
              }}
            >
              <option value="conversation">
                {label("oneConversation", "One conversation")}
              </option>
              <option value="workspace">
                {label("workspace", "Workspace")}
              </option>
            </select>
          </label>
          {scope === "conversation" ? (
            <>
              <label className="block text-sm">
                {label("location", "Conversation location")}
                <select
                  aria-label={label("location", "Conversation location")}
                  disabled={draftLocked}
                  className={fieldClass}
                  value={location}
                  onChange={(e) => {
                    setLocation(e.target.value)
                    setScopeId("")
                    setChats([])
                    setChatOffset(0)
                  }}
                >
                  <option value="">
                    {label("global", "Outside a workspace")}
                  </option>
                  {workspaces.map((w) => (
                    <option key={w.id} value={w.id}>
                      {w.name || w.id}
                    </option>
                  ))}
                </select>
              </label>
              <label className="block text-sm">
                {label("conversation", "Conversation")}
                <select
                  aria-label={label("conversation", "Conversation")}
                  disabled={draftLocked || loadingChats}
                  className={fieldClass}
                  value={scopeId}
                  onChange={(e) => setScopeId(e.target.value)}
                >
                  <option value="">
                    {label("chooseConversation", "Choose a conversation")}
                  </option>
                  {chats.map((c) => (
                    <option key={c.id} value={c.id}>
                      {conversationLabels.get(c.id) ?? c.title}
                    </option>
                  ))}
                </select>
              </label>
              {hasMoreChats ? (
                <Button
                  size="small"
                  loading={loadingChats}
                  onClick={() => setChatOffset((offset) => offset + 100)}
                >
                  {label("moreConversations", "Load more conversations")}
                </Button>
              ) : null}
              {!loadingChats && !chats.length ? (
                <p role="status" className="text-sm text-text-muted">
                  {label(
                    "noConversations",
                    "Start a conversation in Console, then return here to attach your avatar. Workspace attachments can be selected before conversations exist."
                  )}
                </p>
              ) : null}
              <p className="text-sm text-text-muted">
                {label(
                  "conversationPersona",
                  "The conversation keeps its current Persona. Use conversation settings to change its identity or behavior."
                )}
              </p>
            </>
          ) : (
            <>
              <label className="block text-sm">
                {label("workspace", "Workspace")}
                <select
                  aria-label={label("workspace", "Workspace")}
                  disabled={draftLocked}
                  className={fieldClass}
                  value={scopeId}
                  onChange={(e) => setScopeId(e.target.value)}
                >
                  <option value="">
                    {label("chooseWorkspace", "Choose a workspace")}
                  </option>
                  {workspaces.map((w) => (
                    <option key={w.id} value={w.id}>
                      {w.name || w.id}
                    </option>
                  ))}
                </select>
              </label>
              {workspace ? (
                <>
                  <label className="block text-sm">
                    {label(
                      "defaultPersona",
                      "Default Persona for new conversations"
                    )}
                    <select
                      aria-label={label(
                        "defaultPersona",
                        "Default Persona for new conversations"
                      )}
                      className={fieldClass}
                      disabled={draftLocked}
                      value={defaultPersona}
                      onChange={(e) => {
                        setDefaultPersona(e.target.value)
                        setDefaultDirty(true)
                      }}
                    >
                      <option value="">{label("none", "None")}</option>
                      {defaultPersona &&
                      !personas.some((p) => p.id === defaultPersona) ? (
                        <option value={defaultPersona}>
                          {label(
                            "unavailablePersona",
                            "Saved Persona unavailable"
                          )}
                        </option>
                      ) : null}
                      {personas.map((p) => (
                        <option key={p.id} value={p.id}>
                          {p.name}
                        </option>
                      ))}
                    </select>
                  </label>
                  <p className="text-sm text-text-muted">
                    {label(
                      "newOnly",
                      "Applies to new conversations only. Existing conversations and explicit choices keep their identity. Changed defaults use read-only memory; advanced memory settings remain in workspace settings."
                    )}
                  </p>
                </>
              ) : null}
              <p className="text-sm text-text-muted">
                {label(
                  "workspaceVoice",
                  "Workspace mode names each conversation and can read responses aloud. Microphone input is available only for a single conversation."
                )}
              </p>
            </>
          )}
        </section>
      </div>
    </Modal>
  )
}

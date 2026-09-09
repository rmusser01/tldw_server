import React from "react"
import { Button, Modal } from "antd"
import { useTranslation } from "react-i18next"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import type { WorkspaceApiResponse } from "@/services/tldw/domains/workspace-api"
import { listPersonaVisualStarterPacks } from "@/services/persona-visuals"
import {
  buddyAssets,
  createBuddy,
  putBuddyAttachment,
  resolveBuddyConversationTarget,
  type BuddyAttachmentState,
  type BuddyProfile,
  type BuddyConversationSummary
} from "@/services/buddies"
import type { BuddyManagementTarget } from "@/store/buddy-management"
import type { PersonaVisualStarterPackSummary } from "@/types/persona-visuals"
import { BuddyStarterArtwork } from "@/components/PersonaGarden/BuddyStarterArtwork"
import { SpriteFrameRenderer } from "./SpriteFrameRenderer"

const fieldClass =
  "block w-full rounded-md border border-border bg-surface px-3 py-2 text-sm text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-primary"
const StarterChoice = ({
  starter,
  selected,
  onSelect
}: {
  starter: PersonaVisualStarterPackSummary
  selected: boolean
  onSelect: () => void
}) => {
  const { t } = useTranslation("sidepanel")
  const [ready, setReady] = React.useState(false)
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
        <h4 className="font-semibold">{starter.title}</h4>
        <p className="text-sm text-text-muted">{starter.license_label}</p>
        <Button disabled={!ready} onClick={onSelect}>
          {selected
            ? t("buddyManagement.selected", { defaultValue: "Selected" })
            : t("buddyManagement.useBuddy", { defaultValue: "Use this Buddy" })}
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
  const { t } = useTranslation("sidepanel")
  const label = (key: string, text: string) =>
    t(`buddyManagement.${key}`, { defaultValue: text })
  const mounted = React.useRef(true)
  const targetGeneration = React.useRef(0)
  React.useLayoutEffect(() => {
    targetGeneration.current += 1
  }, [target?.scope_type, target?.scope_id])
  React.useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
    }
  }, [])
  const [createdArtwork, setCreatedArtwork] = React.useState<{
    starterId: string
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
  const [busy, setBusy] = React.useState(false)
  const [loading, setLoading] = React.useState(true)
  const [error, setError] = React.useState<string | null>(null)
  const [notice, setNotice] = React.useState<string | null>(null)
  React.useEffect(() => {
    if (!target) return
    setScope(target.scope_type)
    setScopeId(target.scope_id)
    setResolvedTarget(null)
    setLocation(target.scope_type === "workspace" ? target.scope_id : "")
    setChatOffset(0)
    setSearch("")
    setError(null)
    setNotice(null)
    setResolvingTarget(target.scope_type === "conversation")
    if (target.scope_type !== "conversation") return
    let active = true
    setResolvingTarget(true)
    void resolveBuddyConversationTarget(target.scope_id)
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
  }, [target?.scope_type, target?.scope_id])
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
  const apply = async () => {
    if (
      busy ||
      resolvingTarget ||
      !scopeId ||
      (!selected && !starter) ||
      (scope === "conversation" &&
        (loadingChats || !chats.some((c) => c.id === scopeId)))
    )
      return
    setBusy(true)
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
          artworkSaved && label("artworkSaved", "Buddy artwork saved to Your Buddies."),
          defaultSaved && label("previousDefaultSaved", "The previous workspace default was saved."),
          attachmentSaved && label("previousAttachmentSaved", "The previous Buddy attachment was saved."),
          failure && label("earlierApplyFailed", "The earlier Apply could not finish."),
          label("targetChanged", "Target changed. Your new selection is retained; review it before applying.")
        ].filter(Boolean).join(" ")
      )
      return true
    }
    try {
      let buddyId = selected
      if (starter) {
        const created = createdArtwork?.starterId === starter.id
          ? createdArtwork.profile
          : await createBuddy({
              name: starter.title,
              source: { kind: "starter", starter_id: starter.id },
              optional_persona_id: null
            })
        if (!mounted.current) return
        artworkSaved = true
        setCreatedArtwork({ starterId: starter.id, profile: created })
        if (stopForRetarget()) return
        buddyId = created.id
        // Retain the created profile on retry rather than creating duplicates.
        setSelected(created.id)
        setStarter(null)
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
      const savedMessage = attachmentSaved
        ? label("attachmentSaved", "Buddy attachment saved. Refresh failed: ")
        : defaultSaved
          ? label("defaultSaved", "Workspace default saved. Buddy attachment was not confirmed: ")
          : ""
      setError(
        `${savedMessage}${e instanceof Error ? e.message : label("saveFailed", "Could not apply. Your selection is retained; refresh and retry.")}`
      )
    } finally {
      if (mounted.current) setBusy(false)
    }
  }
  const stagedChanges = Boolean(
    starter ||
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
      title={label("title", "Buddy & Persona Management")}
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
              (!selected && !starter) ||
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
            "Choose a Buddy and where it belongs. Artwork is independent of the conversation’s Persona."
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
                      "Apply or cancel your Buddy changes before editing conversation settings."
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
          {label("searchBuddies", "Find a Buddy")}
          <input
            type="search"
            className={fieldClass}
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </label>
        <section aria-label={label("yourBuddies", "Your Buddies")}>
          <h3 className="mb-2 text-base font-semibold">
            {label("yourBuddies", "Your Buddies")}
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
                  aria-pressed={selected === p.id && !starter}
                  aria-label={`${label("use", "Use")} ${p.name}`}
                  onClick={() => {
                    setSelected(p.id)
                    setStarter(null)
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
              {label("loadMoreBuddies", "Load more Buddies")}
            </Button>
          ) : null}
        </section>
        <details open={!availableProfiles.length}>
          <summary className="cursor-pointer text-base font-semibold">
            {label("gallery", "Choose a ready-made Buddy")}
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
                  onSelect={() => {
                    setStarter(s)
                    setSelected("")
                  }}
                />
              ))}
          </div>
        </details>
        <section className="space-y-3 border-t border-border pt-4">
          <h3 className="text-base font-semibold">
            {label("attachment", "Attach to")}
          </h3>
          <label className="block text-sm">
            {label("scope", "Scope")}
            <select
              aria-label={label("scope", "Scope")}
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
                  disabled={loadingChats}
                  className={fieldClass}
                  value={scopeId}
                  onChange={(e) => setScopeId(e.target.value)}
                >
                  <option value="">
                    {label("chooseConversation", "Choose a conversation")}
                  </option>
                  {chats.map((c) => (
                    <option key={c.id} value={c.id}>
                      {c.title}
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
                    "Start a conversation in Console, then return here to attach your Buddy. Workspace attachments can be selected before conversations exist."
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

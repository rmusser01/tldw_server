import React from "react"
import {
  Checkbox,
  Divider,
  Form,
  Input,
  InputNumber,
  Modal,
  Select,
  Tooltip
} from "antd"
import type { FormInstance } from "antd"
import { X } from "lucide-react"
import { useTranslation } from "react-i18next"
import type {
  ActorAspect,
  ActorEditorMode,
  ActorSettings,
  ActorSource,
  ActorTarget
} from "@/types/actor"
import {
  ACTOR_ASPECT_SOFT_LIMIT,
  ACTOR_TOKENS_WARNING_THRESHOLD,
  createDefaultActorSettings,
  normalizeActorAspectKey,
  SIMPLE_MODE_ASPECT_IDS
} from "@/types/actor"
import { useActorWorldBooks } from "@/hooks/useActorWorldBooks"
import { ActorTokens } from "@/components/Common/Settings/ActorTokens"
import { buildActorDictionaryTokens, buildActorSettingsFromForm } from "@/utils/actor"
const Markdown = React.lazy(() => import("@/components/Common/Markdown"))
import { ACTOR_PRESETS, applyActorPresetById } from "@/data/actor-presets"
import type { ActorPresetId } from "@/data/actor-presets"
import { useSelectedCharacter } from "@/hooks/useSelectedCharacter"
import { getSelectedAssistantOperationRevision } from "@/hooks/useSelectedAssistant"
import type { Character } from "@/types/character"
import { captureActorProfileTarget } from "@/utils/actor-settings-target"

const parseWorldBookEntryOptions = (content: string): string[] => {
  const text = (content || "").trim()
  if (!text) return []
  if (text.includes(",")) {
    return text
      .split(",")
      .map((part) => part.trim())
      .filter(Boolean)
  }
  return [text]
}

type Props = {
  form: FormInstance
  settings: ActorSettings | null
  setSettings: (next: ActorSettings) => void
  isCurrent: () => boolean
  captureEditGuard: () => () => boolean
  actorPreview: string
  actorTokenCount: number
  onRecompute: () => void
  newAspectTarget: ActorTarget
  setNewAspectTarget: (target: ActorTarget) => void
  newAspectName: string
  setNewAspectName: (val: string) => void
  actorPositionValue?: string
  editorMode: ActorEditorMode
  onModeChange: (mode: ActorEditorMode) => void
}

type ActorBladeId = "aspects" | "notes" | "placement" | "tokens"

export const ActorEditor: React.FC<Props> = ({
  form,
  settings,
  setSettings: setSettingsState,
  isCurrent,
  captureEditGuard,
  actorPreview,
  actorTokenCount,
  onRecompute,
  newAspectTarget,
  setNewAspectTarget,
  newAspectName,
  setNewAspectName,
  actorPositionValue,
  editorMode,
  onModeChange
}) => {
  const isSimpleMode = editorMode === "simple"
  const { t } = useTranslation(["playground", "common"])
  const [modal, modalContextHolder] = Modal.useModal()
  const {
    worldBooks,
    worldBooksLoading,
    entriesByWorldBook,
    entriesLoading,
    loadEntriesForWorldBook
  } = useActorWorldBooks()
  const [loreWarnings, setLoreWarnings] = React.useState<Record<string, string>>(
    {}
  )
  const [showNotesPreview, setShowNotesPreview] = React.useState(false)
  const [selectedPreset, setSelectedPreset] =
    React.useState<ActorPresetId | null>(null)
  const [activeBlade, setActiveBlade] =
    React.useState<ActorBladeId>("aspects")
  const bladeContentRefs = React.useRef<Record<ActorBladeId, HTMLDivElement | null>>({
    aspects: null,
    notes: null,
    placement: null,
    tokens: null
  })

  const settingsRevisionRef = React.useRef(0)
  const renderedSettingsRevision = settingsRevisionRef.current
  const setSettings = React.useCallback(
    (next: ActorSettings) => {
      if (!isCurrent()) return
      settingsRevisionRef.current++
      setSettingsState(next)
    },
    [isCurrent, setSettingsState]
  )

  // Currently active assistant/character (from CharacterSelect).
  // Used to label per-character defaults and profiles.
  const [selectedCharacter, , characterMeta] =
    useSelectedCharacter<Character>(null)
  const renderedCharacterRevision = getSelectedAssistantOperationRevision()
  const profileActionRef = React.useRef<object | null>(null)
  const profileMountedRef = React.useRef(true)
  const [profileBusy, setProfileBusy] = React.useState(false)
  const [profileFeedback, setProfileFeedback] = React.useState<{
    attempt: object
    pending: boolean
    error: boolean
    message: string
    isCurrent: () => boolean
  } | null>(null)
  React.useEffect(() => {
    profileMountedRef.current = true
    return () => {
      profileMountedRef.current = false
    }
  }, [])

  const runCharacterDefault = async (action: "save" | "apply") => {
    const characterId = selectedCharacter?.id
    if (
      profileActionRef.current ||
      !settings ||
      !characterId ||
      settingsRevisionRef.current !== renderedSettingsRevision
    )
      return
    const sourceIsCurrent = () =>
      profileMountedRef.current &&
      isCurrent() &&
      characterMeta?.isCurrent?.() !== false &&
      getSelectedAssistantOperationRevision() === renderedCharacterRevision
    const editIsCurrent = captureEditGuard()
    const profileTarget = captureActorProfileTarget(characterMeta?.ownerKey, characterId, sourceIsCurrent)
    if (!sourceIsCurrent() || !editIsCurrent()) return
    const attempt = {}
    profileActionRef.current = attempt
    const canPublish = () =>
      profileActionRef.current === attempt && sourceIsCurrent()
    const canCommit = () => canPublish() && editIsCurrent()
    const feedback = (
      message: string,
      error = false,
      pending = false,
      editGuard = editIsCurrent
    ) =>
      setProfileFeedback({
        attempt,
        pending,
        error,
        message,
        isCurrent: () => sourceIsCurrent() && editGuard()
      })
    let staged = false
    let stagedEditIsCurrent = editIsCurrent
    setProfileBusy(true)
    feedback(
      action === "save"
        ? t("playground:actor.defaultSaving", "Saving character default...")
        : t("playground:actor.defaultLoading", "Loading character default..."),
      false,
      true
    )
    try {
      if (!profileTarget) throw new Error("actor_profile_scope_unavailable")
      const proposal =
        action === "save"
          ? buildActorSettingsFromForm(settings, form.getFieldsValue(true))
          : null
      const service = await import("@/services/actor-settings")
      if (!canCommit()) return
      if (proposal) {
        const saved = await service.saveActorProfileForCharacter({
          characterId,
          target: profileTarget,
          settings: proposal,
          validate_lease: canCommit
        })
        if (!canCommit()) return
        if (!saved) throw new Error("actor_profile_save_failed")
        feedback(
          t(
            "playground:actor.defaultSaved",
            "Character default saved on this device."
          )
        )
        return
      }
      const profile = await service.getActorProfileForCharacter(characterId, {
        target: profileTarget,
        throwOnError: true,
        validate_lease: canCommit
      })
      if (!canCommit()) return
      if (!profile) {
        feedback(
          t(
            "playground:actor.defaultAbsent",
            "No saved default exists for this character on this device."
          )
        )
        return
      }
      staged = true
      setSettings(profile)
      if (!canPublish()) return
      stagedEditIsCurrent = captureEditGuard()
      if (!stagedEditIsCurrent()) return
      const fieldValues: Record<string, unknown> = {
        actorEnabled: profile.isEnabled,
        actorNotes: profile.notes ?? "",
        actorNotesGmOnly: profile.notesGmOnly ?? false,
        actorChatPosition: profile.chatPosition,
        actorChatDepth: profile.chatDepth,
        actorChatRole: profile.chatRole,
        actorTemplateMode: profile.templateMode ?? "merge",
        actorAppendable: profile.appendable ?? false
      }
      for (const aspect of profile.aspects || []) {
        fieldValues[`actor_${aspect.id}`] = aspect.value ?? ""
        fieldValues[`actor_key_${aspect.id}`] = aspect.key
      }
      form.setFieldsValue(fieldValues)
      if (!canPublish() || !stagedEditIsCurrent()) return
      onRecompute()
      if (!canPublish() || !stagedEditIsCurrent()) return
      feedback(
        t(
          "playground:actor.defaultStaged",
          "Character default applied to this draft. Save applies it to this chat."
        ),
        false,
        false,
        stagedEditIsCurrent
      )
    } catch {
      if (!(staged ? canPublish() && stagedEditIsCurrent() : canCommit())) return
      feedback(
        action === "save"
          ? t(
              "playground:actor.defaultSaveError",
              "Character default could not be saved. Your staged edits are still here; try Save default again."
            )
          : staged
            ? t(
                "playground:actor.defaultApplyError",
                "Character default could not be applied. Review your staged scene and try Apply default again."
              )
            : t(
                "playground:actor.defaultLoadError",
                "Character default could not be loaded. Your staged edits are still here; try Apply default again."
              ),
        true,
        false,
        staged ? stagedEditIsCurrent : editIsCurrent
      )
    } finally {
      // Keep an issued write's admission until it settles, even after its view retires.
      if (profileActionRef.current === attempt) {
        profileActionRef.current = null
        if (profileMountedRef.current) {
          setProfileBusy(false)
          setProfileFeedback((current) =>
            current?.attempt === attempt && current.pending ? null : current
          )
        }
      }
    }
  }

  // Focus management: focus first interactive element when blade changes
  React.useEffect(() => {
    const contentEl = bladeContentRefs.current[activeBlade]
    if (!contentEl) return
    // Wait for DOM to update, then focus first focusable element
    const frame = requestAnimationFrame(() => {
      if (!isCurrent()) return
      const focusable = contentEl.querySelector<HTMLElement>(
        'input:not([disabled]), select:not([disabled]), button:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
      )
      focusable?.focus()
    })
    return () => cancelAnimationFrame(frame)
  }, [activeBlade, isCurrent])

  React.useEffect(() => {
    if (isSimpleMode && activeBlade === "placement") {
      setActiveBlade("aspects")
    }
  }, [activeBlade, isSimpleMode])

  React.useEffect(() => {
    if (!settings || !isCurrent()) return
    for (const aspect of settings.aspects || []) {
      if (aspect.source === "lore" && aspect.lorebookId) {
        void loadEntriesForWorldBook(aspect.lorebookId)
      }
    }
  }, [isCurrent, loadEntriesForWorldBook, settings])

  React.useEffect(() => {
    if (!settings || !isCurrent()) return
    if (!worldBooks || worldBooksLoading) return

    const worldBookIds = new Set(worldBooks.map((wb) => wb.id))
    const nextWarnings: Record<string, string> = {}
    let changed = false

    const nextAspects: ActorAspect[] = (settings.aspects || []).map(
      (aspect) => {
        if (aspect.source !== "lore") {
          return aspect
        }

        const loreId = aspect.lorebookId
        const entryId = aspect.entryId

        if (!loreId || !worldBookIds.has(loreId)) {
          changed = true
          nextWarnings[aspect.id] = t(
            "playground:composer.actorLoreWorldBookMissing",
            "Previous world book binding was not found; using free text instead."
          )
          return {
            ...aspect,
            source: "free",
            lorebookId: undefined,
            entryId: undefined
          }
        }

        const entries = entriesByWorldBook[loreId]
        if (!entries) {
          return aspect
        }

        const exists = entries.some(
          (entry) => entry.entry_id === String(entryId || "")
        )
        if (!exists && entryId) {
          changed = true
          nextWarnings[aspect.id] = t(
            "playground:composer.actorLoreEntryMissing",
            "Previous world book entry was not found; using free text instead."
          )
          return {
            ...aspect,
            source: "free",
            entryId: undefined
          }
        }

        return aspect
      }
    )

    if (changed) {
      const base = settings ?? createDefaultActorSettings()
      setSettings({
        ...base,
        aspects: nextAspects
      })
      if (!isCurrent()) return
      onRecompute()
    }

    if (!isCurrent()) return
    setLoreWarnings((prev) => {
      const prevKeys = Object.keys(prev)
      const nextKeys = Object.keys(nextWarnings)
      if (prevKeys.length !== nextKeys.length) {
        return nextWarnings
      }
      for (const key of nextKeys) {
        if (prev[key] !== nextWarnings[key]) {
          return nextWarnings
        }
      }
      return prev
    })
  }, [
    entriesByWorldBook,
    isCurrent,
    onRecompute,
    setSettings,
    settings,
    t,
    worldBooks,
    worldBooksLoading
  ])

  const handleRemoveAspect = React.useCallback(
    (aspectId: string) => {
      if (!settings || !isCurrent()) return
      const next: ActorSettings = {
        ...settings,
        aspects: (settings.aspects || []).filter((a) => a.id !== aspectId)
      }
      setSettings(next)
      if (!isCurrent()) return
      form.setFieldsValue({ [`actor_${aspectId}`]: undefined })
      onRecompute()
    },
    [form, isCurrent, onRecompute, setSettings, settings]
  )

  const handleAddAspect = React.useCallback(() => {
    const name = newAspectName.trim()
    if (!settings || !name || !isCurrent()) return

    const target: ActorTarget = newAspectTarget || "user"
    const baseSlug = name
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "_")
      .replace(/^_+|_+$/g, "") || "aspect"
    const baseId = `${target}_${baseSlug}`

    const existingKeys = new Set((settings.aspects || []).map((a) => a.key))
    const existingIds = new Set((settings.aspects || []).map((a) => a.id))
    let key = baseId
    let suffix = 1
    while (existingKeys.has(key) || existingIds.has(key)) {
      suffix += 1
      key = `${baseId}_${suffix}`
    }
    const id = key

    const nextAspect: ActorAspect = {
      id,
      key,
      target,
      name,
      source: "free",
      value: ""
    }

    const base = settings ?? createDefaultActorSettings()
    const next: ActorSettings = {
      ...base,
      aspects: [...(base.aspects || []), nextAspect]
    }

    setSettings(next)
    if (!isCurrent()) return
    form.setFieldsValue({ [`actor_${id}`]: "" })
    form.setFieldsValue({ [`actor_key_${id}`]: key })
    setNewAspectName("")
    onRecompute()
  }, [
    form,
    isCurrent,
    newAspectName,
    newAspectTarget,
    onRecompute,
    setNewAspectName,
    setSettings,
    settings
  ])

  const renderActorAspectField = React.useCallback(
    (aspect: ActorAspect) => {
      const fieldName = `actor_${aspect.id}`
      const keyFieldName = `actor_key_${aspect.id}`
      const currentSource: ActorSource = aspect.source || "free"
      const loreWorldBookId = aspect.lorebookId || ""

      const worldBookOptions =
        worldBooks?.map((wb) => ({
          label: wb.name,
          value: wb.id
        })) || []

      const entries = loreWorldBookId
        ? entriesByWorldBook[loreWorldBookId] || []
        : []

      const entryOptions = entries.map((entry) => ({
        label:
          (entry.keywords && entry.keywords.length
            ? entry.keywords.join(", ")
            : entry.content?.slice(0, 60)) || entry.entry_id,
        value: entry.entry_id
      }))

      const selectedEntry =
        entries.find(
          (entry) => entry.entry_id === String(aspect.entryId || "")
        ) || null

      const valueOptions = selectedEntry
        ? parseWorldBookEntryOptions(selectedEntry.content)
        : []

      const warning = loreWarnings[aspect.id]
      const liveValue = form.getFieldValue(fieldName)
      const hasValue = Boolean(
        String(liveValue ?? aspect.value ?? "").trim()
      )

      const handleSourceChange = (nextSource: ActorSource) => {
        if (!isCurrent()) return
        const base = settings ?? createDefaultActorSettings()
        const aspects = (base.aspects || []).map((a) =>
          a.id === aspect.id
            ? {
                ...a,
                source: nextSource,
                ...(nextSource === "free"
                  ? {
                      lorebookId: undefined,
                      entryId: undefined
                    }
                  : {})
              }
            : a
        )
        setSettings({
          ...base,
          aspects
        })
        if (isCurrent()) onRecompute()
      }

      const handleWorldBookChange = (worldBookId: string) => {
        if (!isCurrent()) return
        const id = worldBookId || ""
        const base = settings ?? createDefaultActorSettings()
        const aspects = (base.aspects || []).map((a) =>
          a.id === aspect.id
            ? {
                ...a,
                lorebookId: id || undefined,
                entryId: undefined
              }
            : a
        )
        setSettings({
          ...base,
          aspects
        })
        if (!isCurrent()) return
        if (id) {
          void loadEntriesForWorldBook(id)
        }
        form.setFieldsValue({
          [fieldName]: undefined
        })
        onRecompute()
      }

      const handleEntryChange = (entryId: string) => {
        if (!isCurrent()) return
        const id = entryId || ""
        const base = settings ?? createDefaultActorSettings()
        const aspects = (base.aspects || []).map((a) =>
          a.id === aspect.id
            ? {
                ...a,
                entryId: id || undefined
              }
            : a
        )
        setSettings({
          ...base,
          aspects
        })
        if (!isCurrent()) return

        if (!loreWorldBookId) {
          return
        }

        const entriesForWorldBook = entriesByWorldBook[loreWorldBookId] || []
        const entry =
          entriesForWorldBook.find((e) => e.entry_id === id) || null
        const options = entry ? parseWorldBookEntryOptions(entry.content) : []

        const currentValue = form.getFieldValue(fieldName)
        let nextValue = currentValue
        if (!options.includes(currentValue)) {
          nextValue = options[0] ?? ""
        }
        form.setFieldsValue({
          [fieldName]: nextValue
        })
        onRecompute()
      }

      const handleKeyChange = (val: string) => {
        const canApply = captureEditGuard()
        if (!canApply()) return
        const currentKey = aspect.key || ""
        const candidate = (val || "").trim() || currentKey
        const fallbackKey = currentKey || `${aspect.target}_aspect`
        const currentNormalized = normalizeActorAspectKey({
          raw: currentKey,
          target: aspect.target,
          fallback: fallbackKey
        })
        const normalized = normalizeActorAspectKey({
          raw: candidate,
          target: aspect.target,
          fallback: fallbackKey
        })

        if (normalized === currentNormalized) {
          if (candidate !== currentNormalized) {
            form.setFieldsValue({ [keyFieldName]: currentNormalized })
          }
          if (currentNormalized !== currentKey) {
            const base = settings ?? createDefaultActorSettings()
            const aspects = (base.aspects || []).map((a) =>
              a.id === aspect.id ? { ...a, key: currentNormalized } : a
            )
            setSettings({
              ...base,
              aspects
            })
          }
          return
        }

        const existingKeys = new Set(
          (settings?.aspects || [])
            .filter((a) => a.id !== aspect.id)
            .map((a) =>
              normalizeActorAspectKey({
                raw: a.key,
                target: a.target,
                fallback: a.key
              })
            )
        )
        if (existingKeys.has(normalized)) {
          modal.error({
            title: t(
              "playground:actor.keyCollisionTitle",
              "Actor key already in use"
            ),
            content: t(
              "playground:actor.keyCollisionBody",
              "Another aspect already uses this key. Keys must be unique per chat."
            ),
            onOk: () => {
              if (canApply()) form.setFieldsValue({ [keyFieldName]: currentKey })
            }
          })
          return
        }

        modal.confirm({
          title: t(
            "playground:actor.keyChangeTitle",
            "Change Actor token key?"
          ),
          content: t(
            "playground:actor.keyChangeBody",
            "This will change the token from {{old}} to {{next}}. Prompts using the old token won't update automatically.",
            {
              old: `[[actor_${currentNormalized}]]`,
              next: `[[actor_${normalized}]]`
            }
          ),
          okText: t("common:confirm", "Confirm"),
          cancelText: t("common:cancel", "Cancel"),
          onOk: () => {
            if (!canApply()) return
            const base = settings ?? createDefaultActorSettings()
            const aspects = (base.aspects || []).map((a) =>
              a.id === aspect.id ? { ...a, key: normalized } : a
            )
            setSettings({
              ...base,
              aspects
            })
            if (isCurrent()) form.setFieldsValue({ [keyFieldName]: normalized })
          },
          onCancel: () => {
            if (canApply()) form.setFieldsValue({ [keyFieldName]: currentNormalized })
          }
        })
      }

      return (
        <Form.Item
          key={aspect.id}
          name={fieldName}
          className={
            hasValue
              ? "border-l-2 border-l-primary pl-2 transition-all"
              : "opacity-60 pl-2 transition-all"
          }
          label={
            <div className="space-y-1">
              <div className="flex flex-col gap-1">
                <div className="flex items-center justify-between">
                  <span>{aspect.name}</span>
                  <div className="flex items-center gap-2">
                    <Select
                      size="small"
                      value={currentSource}
                      style={{ width: 100 }}
                      onChange={(val) =>
                        handleSourceChange(val as ActorSource)
                      }
                      options={[
                        {
                          value: "free",
                          label: t(
                            "playground:composer.actorSourceFree",
                            "Free text"
                          )
                        },
                        {
                          value: "lore",
                          label: t(
                            "playground:composer.actorSourceLore",
                            "World book"
                          )
                        }
                      ]}
                    />
                    <button
                      type="button"
                      onClick={(e) => {
                        e.preventDefault()
                        e.stopPropagation()
                        handleRemoveAspect(aspect.id)
                      }}
                      className="ml-1 text-xs text-text-subtle hover:text-danger">
                      <X className="h-3 w-3" />
                    </button>
                  </div>
                </div>

                {!isSimpleMode && (
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-[11px] text-text-subtle">
                      {t("playground:actor.tokenLabel", "Token")}
                    </span>
                    <div className="flex items-center gap-1">
                      <span className="text-[11px] text-text-subtle">
                        [[actor_
                      </span>
                      <Form.Item
                        name={keyFieldName}
                        noStyle
                        initialValue={aspect.key}>
                        <Input
                          size="small"
                          className="text-[11px] font-mono"
                          onBlur={(e) => handleKeyChange(e.target.value)}
                        />
                      </Form.Item>
                      <span className="text-[11px] text-text-subtle">
                        ]]
                      </span>
                    </div>
                  </div>
                )}
              </div>

              {currentSource === "lore" && (
                <div className="flex flex-col gap-1">
                  <Select
                    size="small"
                    placeholder={t(
                      "playground:composer.actorWorldBookPlaceholder",
                      "World book"
                    )}
                    value={loreWorldBookId || undefined}
                    loading={worldBooksLoading}
                    options={worldBookOptions}
                    onChange={handleWorldBookChange}
                    allowClear
                  />
                  <Select
                    size="small"
                    placeholder={t(
                      "playground:composer.actorEntryPlaceholder",
                      "Entry"
                    )}
                    value={aspect.entryId || undefined}
                    loading={
                      loreWorldBookId
                        ? !!entriesLoading[loreWorldBookId]
                        : false
                    }
                    disabled={!loreWorldBookId || !worldBookOptions.length}
                    options={entryOptions}
                    onChange={handleEntryChange}
                    allowClear
                  />
                </div>
              )}

              {warning && (
                <div className="text-xs text-warn">
                  {warning}
                </div>
              )}
            </div>
          }>
          {currentSource === "lore" ? (
            <Select
              allowClear
              placeholder={t(
                "playground:composer.actorLoreValuePlaceholder",
                "Select an option (optional)"
              )}
              options={valueOptions.map((opt) => ({
                label: opt,
                value: opt
              }))}
              disabled={!selectedEntry || valueOptions.length === 0}
            />
          ) : (
            <Input
              placeholder={t(
                "playground:composer.actorAspectPlaceholder",
                "Describe briefly (optional)"
              )}
            />
          )}
        </Form.Item>
      )
    },
    [
      entriesByWorldBook,
      entriesLoading,
      form,
      captureEditGuard,
      isCurrent,
      modal,
      isSimpleMode,
      loadEntriesForWorldBook,
      loreWarnings,
      onRecompute,
      setSettings,
      settings,
      t,
      worldBooks,
      worldBooksLoading
    ]
  )

  const notesValue: string = Form.useWatch("actorNotes", form) || ""
  const notesGmOnlyValue: boolean =
    Form.useWatch("actorNotesGmOnly", form) ?? settings?.notesGmOnly ?? false
  const chatPositionValue: string =
    Form.useWatch("actorChatPosition", form) || settings?.chatPosition
  const chatDepthValue: number =
    Form.useWatch("actorChatDepth", form) ?? settings?.chatDepth
  const chatRoleValue: string =
    Form.useWatch("actorChatRole", form) || settings?.chatRole

  if (!settings) return null
  const aspectCount = settings.aspects?.length ?? 0
  const tokensOverLimit = actorTokenCount > ACTOR_TOKENS_WARNING_THRESHOLD
  const actorDictionaryTokens = buildActorDictionaryTokens(settings)

  const aspectsSummary =
    aspectCount === 0
      ? t("playground:actor.summaryAspectsEmpty", "No aspects configured yet.")
      : t(
          "playground:actor.summaryAspectsCount",
          "{{count}} aspects configured.",
          { count: aspectCount }
        )

  const notesSummary =
    notesValue && notesValue.length > 0
      ? t(
          "playground:actor.summaryNotes",
          "{{chars}} characters{{gmSuffix}}.",
          {
            chars: notesValue.length,
            gmSuffix:
              notesGmOnlyValue === true
                ? t("playground:actor.summaryNotesGmOnly", " · GM-only")
                : ""
          }
        )
      : t("playground:actor.summaryNotesEmpty", "No scene notes yet.")

  const placementSummary = (() => {
    if (chatPositionValue === "before") {
      return t(
        "playground:actor.summaryPlacementBefore",
        "Before main prompt/story."
      )
    }
    if (chatPositionValue === "after") {
      return t(
        "playground:actor.summaryPlacementAfter",
        "After main prompt/story."
      )
    }
    // depth
    return t(
      "playground:actor.summaryPlacementDepth",
      "Depth {{depth}} · Role: {{role}}.",
      {
        depth: chatDepthValue,
        role:
          chatRoleValue === "user"
            ? t("playground:composer.actorRoleUser", "User")
            : chatRoleValue === "assistant"
            ? t("playground:composer.actorRoleAssistant", "Assistant")
            : t("playground:composer.actorRoleSystem", "System")
      }
    )
  })()

  const tokensSummary =
    actorDictionaryTokens.length === 0
      ? t(
          "playground:actor.summaryTokensEmpty",
          "No tokens defined yet; set aspect values."
        )
      : t(
          "playground:actor.summaryTokensCount",
          "{{tokens}} tokens · ~{{estimate}} tokens in prompt.",
          {
            tokens: actorDictionaryTokens.length,
            estimate: actorTokenCount
          }
        )

  const renderBladeHeader = (
    id: ActorBladeId,
    title: string,
    summary: string
  ) => {
    const isActive = activeBlade === id
    return (
      <button
        type="button"
        onClick={() => setActiveBlade(id)}
        className={`w-full rounded-md border px-3 py-2 text-left transition-colors ${
          isActive
            ? "border-border-strong bg-surface2"
            : "border-border bg-surface hover:bg-surface2"
        }`}
        aria-expanded={isActive}>
        <div className="flex items-center justify-between gap-2">
          <div className="flex flex-col">
            <span className="text-xs font-semibold uppercase text-text">
              {title}
            </span>
            <span className="text-[11px] text-text-subtle">
              {summary}
            </span>
          </div>
          <span className="text-xs text-text-subtle">
            {isActive ? "▾" : "▸"}
          </span>
        </div>
      </button>
    )
  }

  return (
    <div className="space-y-3">
      {modalContextHolder}
      <Form.Item noStyle shouldUpdate>
        {() =>
          profileFeedback?.isCurrent() ? (
            <div
              role={profileFeedback.error ? "alert" : "status"}
              className="text-xs text-text-muted">
              {profileFeedback.message}
            </div>
          ) : profileBusy ? (
            <div role="status" className="text-xs text-text-muted">
              {t(
                "playground:actor.defaultWaiting",
                "Waiting for the previous character default action..."
              )}
            </div>
          ) : null
        }
      </Form.Item>
      <div className="space-y-2">
        {renderBladeHeader(
          "aspects",
          t("playground:actor.bladeAspectsTitle", "Aspects"),
          aspectsSummary
        )}
        {renderBladeHeader(
          "notes",
          t("playground:actor.bladeNotesTitle", "Scene notes"),
          notesSummary
        )}
        {!isSimpleMode && renderBladeHeader(
          "placement",
          t("playground:actor.bladePlacementTitle", "Placement & templates"),
          placementSummary
        )}
        {renderBladeHeader(
          "tokens",
          t("playground:actor.bladeTokensTitle", "Tokens & preview"),
          tokensSummary
        )}
      </div>

      {activeBlade === "aspects" && (
        <div
          ref={(el) => { bladeContentRefs.current.aspects = el }}
          className="space-y-3">
          {!isSimpleMode && (
            <div className="flex flex-col md:flex-row md:items-center gap-2">
              <span className="text-xs text-text-muted">
                {t(
                  "playground:composer.actorAddLabel",
                  "Add or remove aspects to focus the scene."
                )}
              </span>
              <div className="flex flex-col sm:flex-row sm:items-center gap-2 flex-1">
                <Select
                  size="small"
                  value={newAspectTarget}
                  onChange={(val) => setNewAspectTarget(val as ActorTarget)}
                  style={{ width: 140 }}
                  options={[
                    {
                      value: "user",
                      label: t("playground:composer.actorUser", "User")
                    },
                    {
                      value: "char",
                      label: t("playground:composer.actorChar", "Character")
                    },
                    {
                      value: "world",
                      label: t("playground:composer.actorWorld", "World")
                    }
                  ]}
                />
                <Input
                  size="small"
                  value={newAspectName}
                  onChange={(e) => setNewAspectName(e.target.value)}
                  placeholder={t(
                    "playground:composer.actorNewAspectPlaceholder",
                    "Aspect name (e.g., Mood, Role, Location)"
                  )}
                />
                <button
                  type="button"
                  onClick={() => handleAddAspect()}
                  disabled={!newAspectName.trim()}
                  className="inline-flex items-center justify-center rounded-md border border-border-strong px-2 py-1 text-xs text-text hover:bg-surface2 disabled:opacity-50">
                  {t("playground:composer.actorAddButton", "Add aspect")}
                </button>
              </div>
            </div>
          )}

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-text">
                {t("playground:actor.quickSetupLabel", "Quick setup")}
              </span>
              <span className="text-[11px] text-text-subtle">
                {t(
                  "playground:actor.quickSetupHint",
                  "Click a preset to apply"
                )}
              </span>
            </div>
            <div className="grid grid-cols-2 gap-2">
              {ACTOR_PRESETS.map((preset) => (
                <button
                  key={preset.id}
                  type="button"
                  onClick={() => {
                    if (!settings || !isCurrent()) return
                    const base = settings ?? createDefaultActorSettings()
                    const next = applyActorPresetById(base, preset.id)
                    setSettings(next)
                    if (!isCurrent()) return
                    setSelectedPreset(preset.id)
                    const fieldValues: Record<string, any> = {
                      actorNotes: next.notes ?? ""
                    }
                    for (const aspect of next.aspects || []) {
                      fieldValues[`actor_${aspect.id}`] = aspect.value ?? ""
                    }
                    form.setFieldsValue(fieldValues)
                    onRecompute()
                  }}
                  className={`flex flex-col items-start gap-0.5 rounded-md border p-2 text-left transition-colors ${
                    selectedPreset === preset.id
                      ? "border-primary bg-primary/10"
                      : "border-border hover:border-border-strong hover:bg-surface2"
                  }`}>
                  <span className="text-xs font-medium text-text">
                    {t(
                      `playground:actor.preset.${preset.id}` as any,
                      preset.name
                    )}
                  </span>
                  <span className="text-[10px] text-text-subtle line-clamp-2">
                    {preset.description}
                  </span>
                </button>
              ))}
            </div>
            {!isSimpleMode && (
              <>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => {
                      const canApply = captureEditGuard()
                      if (!canApply()) return
                      modal.confirm({
                        title: t(
                          "playground:actor.resetConfirmTitle",
                          "Reset Actor settings?"
                        ),
                        content: t(
                          "playground:actor.resetStagedHelp",
                          "This resets the staged aspects, notes, and placement. Save applies the reset; Cancel keeps your saved scene."
                        ),
                        okText: t("common:confirm", "Confirm"),
                        cancelText: t("common:cancel", "Cancel"),
                        okButtonProps: { danger: true },
                        onOk: () => {
                          if (!canApply()) return
                          const base = createDefaultActorSettings()
                          setSettings(base)
                          if (!isCurrent()) return
                          const fieldValues: Record<string, any> = {
                            actorEnabled: base.isEnabled,
                            actorNotes: base.notes ?? "",
                            actorNotesGmOnly: base.notesGmOnly ?? false,
                            actorChatPosition: base.chatPosition,
                            actorChatDepth: base.chatDepth,
                            actorChatRole: base.chatRole,
                            actorTemplateMode: base.templateMode ?? "merge",
                            actorAppendable: base.appendable ?? false
                          }
                          for (const aspect of base.aspects || []) {
                            fieldValues[`actor_${aspect.id}`] = aspect.value ?? ""
                            fieldValues[`actor_key_${aspect.id}`] = aspect.key
                          }
                          form.setFieldsValue(fieldValues)
                          onRecompute()
                        }
                      })
                    }}
                    className="inline-flex items-center justify-center rounded-md border border-border-strong px-2 py-1 text-xs text-text hover:bg-surface2">
                    {t("playground:actor.resetChat", "Reset for this chat")}
                  </button>
                </div>
                <div className="flex flex-col sm:flex-row sm:items-center gap-2">
                  <button
                    type="button"
                    onClick={() => {
                      void runCharacterDefault("save")
                    }}
                    disabled={profileBusy || !selectedCharacter?.id}
                    className="inline-flex items-center justify-center rounded-md border border-border-strong px-2 py-1 text-xs text-text hover:bg-surface2 disabled:opacity-50">
                    {t(
                      "playground:actor.saveAsCharacterDefault",
                      "Save as {{name}}'s default",
                      { name: selectedCharacter?.name || "character" }
                    )}
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      void runCharacterDefault("apply")
                    }}
                    disabled={profileBusy || !selectedCharacter?.id}
                    className="inline-flex items-center justify-center rounded-md border border-border-strong px-2 py-1 text-xs text-text hover:bg-surface2 disabled:opacity-50">
                    {t("playground:actor.applyCharacterDefault", "Apply {{name}}'s default", {
                      name: selectedCharacter?.name || "character"
                    })}
                  </button>
                </div>
              </>
            )}
          </div>

          {!isSimpleMode && !worldBooksLoading && !worldBooks.length && (
            <div className="rounded-md border border border-warn/30 bg-warn/10 px-3 py-2 text-xs text-warn">
              <div className="font-medium">
                {t(
                  "playground:composer.actorOfflineHintTitle",
                  "World Books not available for Actor"
                )}
              </div>
              <div className="mt-0.5">
                {t(
                  "playground:composer.actorOfflineHintBody",
                  "Connect to your tldw server from Settings to use World Books for Actor aspects, or keep using free-text values."
                )}
              </div>
            </div>
          )}

          {!isSimpleMode && aspectCount > ACTOR_ASPECT_SOFT_LIMIT && (
            <div className="rounded-md border border border-warn/30 bg-warn/10 px-3 py-2 text-xs text-warn">
              <div className="font-medium">
                {t(
                  "playground:actor.aspectLimitWarningTitle",
                  "Many aspects configured for this chat"
                )}
              </div>
              <div className="mt-0.5">
                {t(
                  "playground:actor.aspectLimitWarningBody",
                  "You have {{count}} aspects (soft limit {{limit}}). Consider trimming to focus the scene and save tokens.",
                  {
                    count: aspectCount,
                    limit: ACTOR_ASPECT_SOFT_LIMIT
                  }
                )}
              </div>
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="space-y-2">
              <div className="text-xs font-semibold uppercase text-text-subtle">
                {t("playground:composer.actorUser", "User")}
              </div>
              {settings.aspects
                .filter((a) => a.target === "user")
                .filter((a) => !isSimpleMode || SIMPLE_MODE_ASPECT_IDS.includes(a.id as any))
                .map((aspect) => renderActorAspectField(aspect))}
            </div>

            <div className="space-y-2">
              <div className="text-xs font-semibold uppercase text-text-subtle">
                {t("playground:composer.actorChar", "Character")}
              </div>
              {settings.aspects
                .filter((a) => a.target === "char")
                .filter((a) => !isSimpleMode || SIMPLE_MODE_ASPECT_IDS.includes(a.id as any))
                .map((aspect) => renderActorAspectField(aspect))}
            </div>

            <div className="space-y-2">
              <div className="text-xs font-semibold uppercase text-text-subtle">
                {t("playground:composer.actorWorld", "World")}
              </div>
              {settings.aspects
                .filter((a) => a.target === "world")
                .filter((a) => !isSimpleMode || SIMPLE_MODE_ASPECT_IDS.includes(a.id as any))
                .map((aspect) => renderActorAspectField(aspect))}
            </div>
          </div>

          {isSimpleMode && (
            <div className="text-center pt-2">
              <button
                type="button"
                onClick={() => onModeChange("advanced")}
                className="text-xs text-primary hover:underline">
                {t("playground:actor.showAllOptions", "Show all options")}
              </button>
            </div>
          )}
        </div>
      )}

      {activeBlade === "notes" && (
        <div
          ref={(el) => { bladeContentRefs.current.notes = el }}
          className="space-y-3">
          <Form.Item
            name="actorNotes"
            label={t("playground:composer.actorNotes", "Scene notes")}
            help={t(
              "playground:composer.actorNotesHelp",
              "Optional notes sent to the model. Supports Markdown and LaTeX."
            )}>
            <Input.TextArea rows={3} />
          </Form.Item>

          <Form.Item
            name="actorNotesGmOnly"
            valuePropName="checked"
            className="!-mt-3 mb-2">
            <Checkbox className="text-[11px] text-text-muted">
              {t(
                "playground:actor.notesGmOnly",
                "GM-only: do not send notes to the model"
              )}
            </Checkbox>
          </Form.Item>

          {notesValue && (
            <div className="space-y-1">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold uppercase text-text-subtle">
                  {t(
                    "playground:actor.notesPreviewTitle",
                    "Scene notes preview"
                  )}
                </span>
                <label className="inline-flex items-center gap-2 text-[11px] text-text-subtle">
                  <input
                    type="checkbox"
                    className="h-3 w-3"
                    checked={showNotesPreview}
                    onChange={(e) => setShowNotesPreview(e.target.checked)}
                  />
                  <span>
                    {t(
                      "playground:actor.notesPreviewToggle",
                      "Show Markdown + LaTeX preview"
                    )}
                  </span>
                </label>
              </div>
              {showNotesPreview && (
                <div className="max-h-40 overflow-y-auto rounded-md border border-border px-2 py-1 bg-surface2">
                  <React.Suspense fallback={null}>
                    <Markdown message={notesValue} />
                  </React.Suspense>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {!isSimpleMode && activeBlade === "placement" && (
        <div
          ref={(el) => { bladeContentRefs.current.placement = el }}
          className="space-y-3">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <Form.Item
              name="actorChatPosition"
              label={t(
                "playground:composer.actorPosition",
                "Actor prompt position"
              )}
              help={t(
                "playground:composer.actorPositionHelp",
                'Depth is only used when position is "In-chat at depth". Role controls whether Actor appears as system, user, or assistant.'
              )}>
              <Select
                options={[
                  {
                    value: "before",
                    label: t(
                      "playground:composer.actorBefore",
                      "Before main prompt/story"
                    )
                  },
                  {
                    value: "after",
                    label: t(
                      "playground:composer.actorAfter",
                      "After main prompt/story"
                    )
                  },
                  {
                    value: "depth",
                    label: t(
                      "playground:composer.actorDepth",
                      "In-chat at depth"
                    )
                  }
                ]}
              />
            </Form.Item>

            <Form.Item
              name="actorChatDepth"
              label={
                <span className="inline-flex items-center gap-1">
                  {t(
                    "playground:composer.actorDepthLabel",
                    "Depth (non-system messages)"
                  )}
                  <Tooltip
                    title={t(
                      "playground:composer.actorDepthHelp",
                      'Used only when position is "In-chat at depth". Counts non-system messages from the top of the chat.'
                    )}>
                    <span className="text-xs text-text-subtle cursor-help">
                      ?
                    </span>
                  </Tooltip>
                </span>
              }
              help={t(
                "playground:actor.depthRangeHelp",
                "Depth must be between 0 and 999; values beyond history length are clamped."
              )}>
              <InputNumber
                min={0}
                max={999}
                style={{ width: "100%" }}
                disabled={actorPositionValue !== "depth"}
                className={actorPositionValue !== "depth" ? "opacity-60" : ""}
                placeholder={t(
                  "playground:composer.actorDepthPlaceholder",
                  "0 = top of chat"
                )}
                controls
              />
            </Form.Item>

            <Form.Item
              name="actorChatRole"
              label={t(
                "playground:composer.actorRoleLabel",
                "Actor message role"
              )}>
              <Select
                options={[
                  {
                    value: "system",
                    label: t(
                      "playground:composer.actorRoleSystem",
                      "System"
                    )
                  },
                  {
                    value: "user",
                    label: t("playground:composer.actorRoleUser", "User")
                  },
                  {
                    value: "assistant",
                    label: t(
                      "playground:composer.actorRoleAssistant",
                      "Assistant"
                    )
                  }
                ]}
              />
            </Form.Item>
          </div>

          <Form.Item
            name="actorTemplateMode"
            label={t(
              "playground:actor.templateModeLabel",
              "Scene template interaction"
            )}
            help={t(
              "playground:actor.templateModeHelp",
              "Controls how Actor interacts with active scene templates. Merge keeps both; Override lets Actor replace overlapping fields; Ignore skips Actor when templates are active."
            )}>
            <Select
              options={[
                {
                  value: "merge",
                  label: t(
                    "playground:actor.templateMode.merge",
                    "Merge with templates"
                  )
                },
                {
                  value: "override",
                  label: t(
                    "playground:actor.templateMode.override",
                    "Override templates"
                  )
                },
                {
                  value: "ignore",
                  label: t(
                    "playground:actor.templateMode.ignore",
                    "Ignore when templates active"
                  )
                }
              ]}
            />
          </Form.Item>

          <Form.Item
            name="actorAppendable"
            valuePropName="checked"
            className="!-mt-2"
            help={t(
              "playground:actor.appendableHelp",
              "When enabled, Actor content can be concatenated with other appendable system blocks."
            )}>
            <Checkbox>
              {t("playground:actor.appendableLabel", "Appendable")}
            </Checkbox>
          </Form.Item>
        </div>
      )}

      {activeBlade === "tokens" && (
        <div
          ref={(el) => { bladeContentRefs.current.tokens = el }}
          className="space-y-3">
          <div className="space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase text-text-subtle">
                {t(
                  "playground:composer.actorPreview",
                  "Actor prompt preview"
                )}
              </span>
              <span className="text-xs text-text-subtle">
                {t(
                  "playground:composer.actorTokens",
                  "Tokens: {{count}}",
                  {
                    count: actorTokenCount
                  }
                )}
              </span>
            </div>
            {tokensOverLimit && (
              <div className="mt-0.5 text-[11px] text-warn">
                {t(
                  "playground:actor.tokenWarning",
                  "Actor prompt is long ({{count}} tokens; soft limit {{limit}}). Consider trimming aspects or notes to save context.",
                  {
                    count: actorTokenCount,
                    limit: ACTOR_TOKENS_WARNING_THRESHOLD
                  }
                )}
              </div>
            )}
            <div className="max-h-40 overflow-y-auto rounded-md border border-border px-2 py-1 bg-surface2">
              <pre className="whitespace-pre-wrap text-xs text-text">
                {actorPreview ||
                  t(
                    "playground:composer.actorPreviewEmpty",
                    "Nothing to send yet."
                  )}
              </pre>
            </div>
          </div>

          <ActorTokens settings={settings} />
        </div>
      )}

      <Divider />
    </div>
  )
}

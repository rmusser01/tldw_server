import React from "react"
import { Tooltip } from "antd"
import { ChevronDown, ChevronUp, Plus, Save, Trash2 } from "lucide-react"
import { useTranslation } from "react-i18next"

import {
  updateChatMacroSettings,
  type ChatMacroOutputProfile,
  type ChatMacroSettings
} from "@/services/chat-macros"
import { outputProfilesToSettings } from "./chat-macro-editor-utils"

const PROFILE_KEY = /^[a-z][a-z0-9_]{0,63}$/
const MAX_SECTIONS = 10
const MAX_HEADING_LENGTH = 128

type ProfileDraft = ChatMacroOutputProfile & { sectionHeadings: string[] }
type ProfileDrafts = Record<string, ProfileDraft>

export interface OutputProfileEditorProps {
  settings: ChatMacroSettings
  onSaved: (settings: ChatMacroSettings) => void
}

const iconButtonClassName =
  "inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-md border border-border bg-surface text-text transition-colors hover:bg-surface2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus disabled:cursor-not-allowed disabled:opacity-50"

const fieldClassName =
  "h-9 w-full rounded-md border border-border bg-background px-2.5 text-sm text-text outline-none focus:border-primary focus:ring-2 focus:ring-focus disabled:cursor-not-allowed disabled:opacity-60"

const segmentClassName = (selected: boolean): string =>
  `min-h-9 px-3 text-sm font-medium transition-colors focus-visible:relative focus-visible:z-10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus ${
    selected
      ? "bg-primary text-white"
      : "bg-surface text-text hover:bg-surface2"
  }`

const createProfile = (): ProfileDraft => ({
  format: "structured_sections",
  sections: ["summary"],
  section_titles: {},
  include_branch_outputs: false,
  sectionHeadings: [""]
})

const cloneProfile = (profile: ChatMacroOutputProfile): ProfileDraft => {
  const sections = Array.isArray(profile.sections) ? [...profile.sections] : []
  const section_titles =
    profile.section_titles && typeof profile.section_titles === "object"
      ? { ...profile.section_titles }
      : {}
  const sourceHeadings = (profile as Partial<ProfileDraft>).sectionHeadings

  return {
    format: profile.format,
    sections,
    section_titles,
    include_branch_outputs: Boolean(profile.include_branch_outputs),
    sectionHeadings: Array.isArray(sourceHeadings)
      ? [...sourceHeadings]
      : sections.map((section) => section_titles[section] || "")
  }
}

const cloneProfiles = (profiles: Record<string, ChatMacroOutputProfile>): ProfileDrafts =>
  Object.fromEntries(
    Object.entries(profiles).map(([name, profile]) => [name, cloneProfile(profile)])
  )

const firstProfileName = (profiles: Record<string, ChatMacroOutputProfile>): string =>
  profiles.default ? "default" : Object.keys(profiles)[0] || "default"

export const OutputProfileEditor = ({ settings, onSaved }: OutputProfileEditorProps) => {
  const { t } = useTranslation()
  const [originalSettings, setOriginalSettings] = React.useState<ChatMacroSettings>(settings)
  const [drafts, setDrafts] = React.useState<ProfileDrafts>(() => cloneProfiles(settings.output_profiles))
  const [selectedProfile, setSelectedProfile] = React.useState(() =>
    firstProfileName(settings.output_profiles)
  )
  const [newProfileName, setNewProfileName] = React.useState("")
  const [error, setError] = React.useState<string | null>(null)
  const [status, setStatus] = React.useState<string | null>(null)
  const [saving, setSaving] = React.useState(false)

  const label = React.useCallback(
    (key: string, defaultValue: string) => t(`outputProfileEditor.${key}`, defaultValue),
    [t]
  )
  const indexedLabel = React.useCallback(
    (key: string, defaultValue: string, index: number) =>
      t(`outputProfileEditor.${key}`, { index, defaultValue }),
    [t]
  )

  React.useEffect(() => {
    const nextDrafts = cloneProfiles(settings.output_profiles)
    setOriginalSettings(settings)
    setDrafts(nextDrafts)
    setSelectedProfile((current) => (nextDrafts[current] ? current : firstProfileName(nextDrafts)))
    setNewProfileName("")
    setError(null)
    setStatus(null)
  }, [settings])

  const currentProfile = drafts[selectedProfile] || createProfile()

  const clearFeedback = React.useCallback(() => {
    setError(null)
    setStatus(null)
  }, [])

  const updateCurrentProfile = React.useCallback(
    (update: (profile: ProfileDraft) => ProfileDraft) => {
      clearFeedback()
      setDrafts((current) => ({
        ...current,
        [selectedProfile]: update(cloneProfile(current[selectedProfile] || createProfile()))
      }))
    },
    [clearFeedback, selectedProfile]
  )

  const addProfile = React.useCallback(() => {
    const name = newProfileName.trim()
    if (!PROFILE_KEY.test(name)) {
      setError(label("validation.profileKey", "Profile names must use lowercase letters, numbers, and underscores."))
      setStatus(null)
      return
    }
    if (drafts[name]) {
      setError(label("validation.profileDuplicate", "A profile with this name already exists."))
      setStatus(null)
      return
    }

    setDrafts((current) => ({ ...current, [name]: createProfile() }))
    setSelectedProfile(name)
    setNewProfileName("")
    clearFeedback()
  }, [clearFeedback, drafts, label, newProfileName])

  const deleteProfile = React.useCallback(() => {
    if (selectedProfile === "default") return
    setDrafts((current) => {
      const { [selectedProfile]: _deleted, ...remaining } = current
      return remaining
    })
    setSelectedProfile("default")
    clearFeedback()
  }, [clearFeedback, selectedProfile])

  const updateSection = React.useCallback(
    (index: number, key: "section" | "heading", value: string) => {
      updateCurrentProfile((profile) => {
        if (key === "section") {
          const sections = [...profile.sections]
          sections[index] = value
          return { ...profile, sections }
        }

        const sectionHeadings = [...profile.sectionHeadings]
        sectionHeadings[index] = value
        return {
          ...profile,
          sectionHeadings
        }
      })
    },
    [updateCurrentProfile]
  )

  const moveSection = React.useCallback(
    (index: number, direction: -1 | 1) => {
      updateCurrentProfile((profile) => {
        const destination = index + direction
        if (destination < 0 || destination >= profile.sections.length) return profile
        const sections = [...profile.sections]
        const sectionHeadings = [...profile.sectionHeadings]
        ;[sections[index], sections[destination]] = [sections[destination], sections[index]]
        ;[sectionHeadings[index], sectionHeadings[destination]] = [
          sectionHeadings[destination],
          sectionHeadings[index]
        ]
        return { ...profile, sections, sectionHeadings }
      })
    },
    [updateCurrentProfile]
  )

  const removeSection = React.useCallback(
    (index: number) => {
      updateCurrentProfile((profile) => {
        return {
          ...profile,
          sections: profile.sections.filter((_, sectionIndex) => sectionIndex !== index),
          sectionHeadings: profile.sectionHeadings.filter((_, sectionIndex) => sectionIndex !== index)
        }
      })
    },
    [updateCurrentProfile]
  )

  const addSection = React.useCallback(() => {
    updateCurrentProfile((profile) => ({
      ...profile,
      sections: [...profile.sections, ""],
      sectionHeadings: [...profile.sectionHeadings, ""]
    }))
  }, [updateCurrentProfile])

  const validationErrors = React.useCallback(
    (profiles: ProfileDrafts): string[] => {
      const errors: string[] = []
      for (const [profileName, profile] of Object.entries(profiles)) {
        if (!PROFILE_KEY.test(profileName)) {
          errors.push(label("validation.profileKey", "Profile names must use lowercase letters, numbers, and underscores."))
        }

        const sections = profile.sections.map((section) => section.trim())
        if (sections.length > MAX_SECTIONS) {
          errors.push(label("validation.tooManySections", "A profile can contain at most 10 sections."))
        }
        if (new Set(sections).size !== sections.length) {
          errors.push(label("validation.sectionDuplicate", "Section keys must be unique."))
        }
        if (sections.some((section) => !PROFILE_KEY.test(section))) {
          errors.push(
            label(
              "validation.sectionKey",
              "Section keys must use lowercase letters, numbers, and underscores."
            )
          )
        }
        if (profile.sectionHeadings.some((heading) => heading.trim().length > MAX_HEADING_LENGTH)) {
          errors.push(label("validation.headingLength", "Section headings must be 128 characters or fewer."))
        }
      }
      return [...new Set(errors)]
    },
    [label]
  )

  const normalizedProfiles = React.useCallback(
    (profiles: ProfileDrafts): Record<string, ChatMacroOutputProfile> =>
      Object.fromEntries(
        Object.entries(profiles).map(([name, profile]) => {
          const { sectionHeadings, ...profileWithoutHeadings } = profile
          const sections = profile.sections.map((section) => section.trim())
          const section_titles = sections.reduce<Record<string, string>>((titles, section, index) => {
            const heading = sectionHeadings[index]?.trim()
            if (heading) titles[section] = heading
            return titles
          }, {})
          return [
            name,
            {
              ...profileWithoutHeadings,
              sections,
              section_titles
            }
          ]
        })
      ),
    []
  )

  const saveProfiles = React.useCallback(async () => {
    const errors = validationErrors(drafts)
    if (errors.length > 0) {
      setError(errors.join(" "))
      setStatus(null)
      return
    }

    const profiles = normalizedProfiles(drafts)
    const nextSettings = outputProfilesToSettings(originalSettings, profiles)
    setSaving(true)
    setError(null)
    setStatus(null)

    try {
      const response = await updateChatMacroSettings(nextSettings)
      if (!response.ok || !response.data?.settings) {
        setError(
          response.error ||
            t("outputProfileEditor.requestFailed", {
              status: response.status,
              defaultValue: `Request failed (${response.status})`
            })
        )
        return
      }

      const normalizedSettings = response.data.settings
      const normalizedDrafts = cloneProfiles(normalizedSettings.output_profiles)
      setOriginalSettings(normalizedSettings)
      setDrafts(normalizedDrafts)
      setSelectedProfile((current) =>
        normalizedDrafts[current] ? current : firstProfileName(normalizedDrafts)
      )
      setStatus(label("saved", "Output profiles saved."))
      onSaved(normalizedSettings)
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : label("saveFailed", "Failed to save output profiles."))
    } finally {
      setSaving(false)
    }
  }, [drafts, label, normalizedProfiles, onSaved, originalSettings, t, validationErrors])

  const profileNames = Object.keys(drafts)

  return (
    <section className="flex w-full min-w-0 flex-col gap-4" aria-labelledby="output-profile-editor-title">
      <header className="flex flex-col gap-1">
        <h2 id="output-profile-editor-title" className="text-base font-semibold text-text">
          {label("title", "Output profiles")}
        </h2>
        <p className="max-w-3xl text-sm text-text-muted">
          {label("description", "Choose how macro results are organized and which branch outputs are included.")}
        </p>
      </header>

      <div className="grid gap-3 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] md:items-end">
        <label className="min-w-0 text-sm font-medium text-text" htmlFor="output-profile-select">
          {label("profile", "Profile")}
          <select
            id="output-profile-select"
            className={fieldClassName}
            value={selectedProfile}
            disabled={saving}
            onChange={(event) => {
              setSelectedProfile(event.target.value)
              clearFeedback()
            }}
          >
            {profileNames.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </label>

        <div className="grid min-w-0 grid-cols-[minmax(0,1fr)_auto_auto] items-end gap-2">
          <label className="min-w-0 text-sm font-medium text-text" htmlFor="new-output-profile-name">
            {label("newProfileName", "New profile name")}
            <input
              id="new-output-profile-name"
              className={fieldClassName}
              value={newProfileName}
              disabled={saving}
              onChange={(event) => {
                setNewProfileName(event.target.value)
                clearFeedback()
              }}
            />
          </label>
          <Tooltip title={label("addProfile", "Add profile")}>
            <button
              type="button"
              className={iconButtonClassName}
              aria-label={label("addProfile", "Add profile")}
              disabled={saving}
              onClick={addProfile}
            >
              <Plus className="size-4" aria-hidden="true" />
            </button>
          </Tooltip>
          <Tooltip title={label("deleteProfile", "Delete profile")}>
            <button
              type="button"
              className={iconButtonClassName}
              aria-label={label("deleteProfile", "Delete profile")}
              title={
                selectedProfile === "default"
                  ? label("defaultProtected", "The default profile cannot be deleted.")
                  : label("deleteProfile", "Delete profile")
              }
              disabled={saving || selectedProfile === "default"}
              onClick={deleteProfile}
            >
              <Trash2 className="size-4" aria-hidden="true" />
            </button>
          </Tooltip>
        </div>
      </div>

      <fieldset className="min-w-0">
        <legend className="text-sm font-medium text-text">
          {label("responseFormat", "Response format")}
        </legend>
        <div className="mt-1 inline-flex overflow-hidden rounded-md border border-border" role="group">
          <button
            type="button"
            className={`rounded-l-md ${segmentClassName(currentProfile.format === "structured_sections")}`}
            aria-pressed={currentProfile.format === "structured_sections"}
            disabled={saving}
            onClick={() => updateCurrentProfile((profile) => ({ ...profile, format: "structured_sections" }))}
          >
            {label("structuredSections", "Structured sections")}
          </button>
          <button
            type="button"
            className={`rounded-r-md border-l border-border ${segmentClassName(currentProfile.format === "single_response")}`}
            aria-pressed={currentProfile.format === "single_response"}
            disabled={saving}
            onClick={() => updateCurrentProfile((profile) => ({ ...profile, format: "single_response" }))}
          >
            {label("singleResponse", "Single response")}
          </button>
        </div>
      </fieldset>

      <label className="flex min-h-9 items-center gap-2 text-sm text-text" htmlFor="include-branch-outputs">
        <input
          id="include-branch-outputs"
          type="checkbox"
          className="size-4 rounded border-border text-primary focus:ring-2 focus:ring-focus"
          checked={currentProfile.include_branch_outputs}
          disabled={saving}
          onChange={(event) =>
            updateCurrentProfile((profile) => ({
              ...profile,
              include_branch_outputs: event.target.checked
            }))
          }
        />
        {label("includeBranchOutputs", "Include branch outputs")}
      </label>

      <div className="min-w-0">
        <div className="mb-2 flex items-center justify-between gap-3">
          <h3 className="text-sm font-medium text-text">{label("sections", "Sections")}</h3>
          <button
            type="button"
            className="inline-flex h-9 items-center gap-2 rounded-md border border-border bg-surface px-3 text-sm font-medium text-text transition-colors hover:bg-surface2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus"
            disabled={saving}
            onClick={addSection}
          >
            <Plus className="size-4" aria-hidden="true" />
            {label("addSection", "Add section")}
          </button>
        </div>

        <div className="flex min-w-0 flex-col gap-2">
          {currentProfile.sections.map((section, index) => (
            <div
              key={index}
              data-testid="output-profile-section-row"
              className="grid min-h-[52px] min-w-0 gap-2 rounded-md border border-border bg-surface2/30 p-2 sm:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_auto] sm:items-end"
            >
              <label className="min-w-0 text-xs font-medium text-text-muted" htmlFor={`output-profile-section-key-${index}`}>
                {indexedLabel("sectionKey", `Section key ${index + 1}`, index + 1)}
                <input
                  id={`output-profile-section-key-${index}`}
                  className={`${fieldClassName} mt-1`}
                  value={section}
                  disabled={saving}
                  onChange={(event) => updateSection(index, "section", event.target.value)}
                />
              </label>
              <label className="min-w-0 text-xs font-medium text-text-muted" htmlFor={`output-profile-section-heading-${index}`}>
                {indexedLabel("sectionHeading", `Section heading ${index + 1}`, index + 1)}
                <input
                  id={`output-profile-section-heading-${index}`}
                  className={`${fieldClassName} mt-1`}
                  value={currentProfile.sectionHeadings[index] || ""}
                  disabled={saving}
                  onChange={(event) => updateSection(index, "heading", event.target.value)}
                />
              </label>
              <div className="flex h-9 shrink-0 items-center gap-1">
                <Tooltip title={indexedLabel("moveSectionUp", `Move section ${index + 1} up`, index + 1)}>
                  <button
                    type="button"
                    className={iconButtonClassName}
                    aria-label={indexedLabel("moveSectionUp", `Move section ${index + 1} up`, index + 1)}
                    disabled={saving || index === 0}
                    onClick={() => moveSection(index, -1)}
                  >
                    <ChevronUp className="size-4" aria-hidden="true" />
                  </button>
                </Tooltip>
                <Tooltip title={indexedLabel("moveSectionDown", `Move section ${index + 1} down`, index + 1)}>
                  <button
                    type="button"
                    className={iconButtonClassName}
                    aria-label={indexedLabel("moveSectionDown", `Move section ${index + 1} down`, index + 1)}
                    disabled={saving || index === currentProfile.sections.length - 1}
                    onClick={() => moveSection(index, 1)}
                  >
                    <ChevronDown className="size-4" aria-hidden="true" />
                  </button>
                </Tooltip>
                <Tooltip title={indexedLabel("removeSection", `Remove section ${index + 1}`, index + 1)}>
                  <button
                    type="button"
                    className={iconButtonClassName}
                    aria-label={indexedLabel("removeSection", `Remove section ${index + 1}`, index + 1)}
                    disabled={saving}
                    onClick={() => removeSection(index)}
                  >
                    <Trash2 className="size-4" aria-hidden="true" />
                  </button>
                </Tooltip>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <button
          type="button"
          className="inline-flex h-9 items-center gap-2 rounded-md bg-primary px-3 text-sm font-medium text-white transition-colors hover:bg-primaryStrong focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus disabled:cursor-not-allowed disabled:opacity-50"
          disabled={saving}
          onClick={() => void saveProfiles()}
        >
          <Save className="size-4" aria-hidden="true" />
          {saving ? label("saving", "Saving") : label("save", "Save profiles")}
        </button>
        {status ? (
          <p className="text-sm text-success" role="status" aria-live="polite">
            {status}
          </p>
        ) : null}
      </div>

      {error ? (
        <p className="rounded-md border border-danger/40 bg-danger/10 px-3 py-2 text-sm font-medium text-danger" role="alert">
          {error}
        </p>
      ) : null}
    </section>
  )
}

export default OutputProfileEditor

import React, { useState, useMemo, useRef, useEffect } from "react"
import { useQuery } from "@tanstack/react-query"
import { Avatar, Dropdown, Empty, Input, Select, Tooltip } from "antd"
import type { InputRef } from "antd"
import type { ItemType, MenuItemType } from "antd/es/menu/interface"
import { Star, User2, Search, X } from "lucide-react"
import { useTranslation } from "react-i18next"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { useServerCapabilities } from "@/hooks/useServerCapabilities"
import { useStorage } from "@plasmohq/storage/hook"
import { browser } from "wxt/browser"
import { collectGreetings, pickGreeting } from "@/utils/character-greetings"
import {
  CHARACTER_MOOD_OPTIONS,
  getCharacterMoodImagesFromExtensions,
  removeCharacterMoodImage,
  upsertCharacterMoodImage,
  type CharacterMoodLabel
} from "@/utils/character-mood"
import {
  DEFAULT_MESSAGE_STEERING_PROMPTS,
  MESSAGE_STEERING_PROMPTS_STORAGE_KEY,
  normalizeMessageSteeringPrompts
} from "@/utils/message-steering"
import { MyChatIdentityMenu } from "@/components/Common/MyChatIdentityMenu"
import { IconButton } from "@/components/Common/IconButton"
import { useSetBuddyShellRenderContext } from "@/components/Common/PersonaBuddy"
import { useAntdNotification } from "@/hooks/useAntdNotification"
import { useAntdModal } from "@/hooks/useAntdModal"
import { useConfirmModal } from "@/hooks/useConfirmModal"
import { useSelectedCharacter } from "@/hooks/useSelectedCharacter"
import { useSelectedAssistant } from "@/hooks/useSelectedAssistant"
import { useClearChat } from "@/hooks/chat/useClearChat"
import { useStoreMessageOption } from "@/store/option"
import { getBrowserRuntime, isExtensionRuntime } from "@/utils/browser-runtime"
import {
  buildCharactersHash as buildCharactersHashUrl,
  buildCharactersRoute as buildCharactersRouteUrl,
  resolveCharactersDestinationMode
} from "@/utils/characters-route"
import type { PersonaInfo } from "@/routes/personaTypes"
import type {
  Character as StoredCharacter,
  CharacterApiResponse
} from "@/types/character"
import type { MessageSteeringPromptTemplates } from "@/types/message-steering"
import {
  characterToAssistantSelection,
  personaToAssistantSelection,
  type AssistantSelection
} from "@/types/assistant-selection"

type Props = {
  selectedCharacterId: string | null
  setSelectedCharacterId: (id: string | null) => void
  className?: string
  iconClassName?: string
}

type ImportCharacterResponse = {
  character?: CharacterApiResponse
  message?: string
  character_id?: string | number
  characterId?: string | number
} & Partial<CharacterApiResponse>

type CharacterSortMode = "favorites" | "az"

type FavoriteCharacter = {
  id?: string
  slug?: string
  name: string
}

type ImageOnlyErrorDetail = {
  code?: string
  message?: string
}

const GREETING_RETRY_DELAY_MS = 800
const MAX_PERSONA_IMAGE_BYTES = 5 * 1024 * 1024
const MAX_MOOD_IMAGE_BYTES = 5 * 1024 * 1024

const delayWithAbort = (ms: number, signal?: AbortSignal) =>
  new Promise<void>((resolve) => {
    if (!signal) {
      window.setTimeout(resolve, ms)
      return
    }
    if (signal.aborted) {
      resolve()
      return
    }
    let timeoutId: number | null = null
    const onAbort = () => {
      if (timeoutId !== null) {
        window.clearTimeout(timeoutId)
        timeoutId = null
      }
      signal.removeEventListener("abort", onAbort)
      resolve()
    }
    timeoutId = window.setTimeout(() => {
      signal.removeEventListener("abort", onAbort)
      timeoutId = null
      resolve()
    }, ms)
    signal.addEventListener("abort", onAbort, { once: true })
  })

export const CharacterSelect: React.FC<Props> = ({
  selectedCharacterId,
  setSelectedCharacterId,
  className = "text-text-muted",
  iconClassName = "size-4"
}) => {
  const { t } = useTranslation(["sidepanel", "common", "settings"])
  const notification = useAntdNotification()
  const modal = useAntdModal()
  const confirmWithModal = useConfirmModal()
  const setBuddyShellRenderContext = useSetBuddyShellRenderContext()
  const [menuDensity] = useStorage("menuDensity", "comfortable")
  const [favoriteCharacters, setFavoriteCharacters] = useStorage<FavoriteCharacter[]>(
    "favoriteCharacters",
    []
  )
  const [sortMode, setSortMode] = useStorage<CharacterSortMode>(
    "characterSortMode",
    "favorites"
  )
  const [selectedCharacterStored, setSelectedCharacter] =
    useSelectedCharacter<StoredCharacter | null>(null)
  const [selectedAssistant, setSelectedAssistant] = useSelectedAssistant(null)
  const clearChat = useClearChat()
  const messages = useStoreMessageOption((state) => state.messages)
  const serverChatId = useStoreMessageOption((state) => state.serverChatId)
  const [userDisplayName, setUserDisplayName] = useStorage(
    "chatUserDisplayName",
    ""
  )
  const [userPersonaImage, setUserPersonaImage] = useStorage(
    "chatUserPersonaImage",
    ""
  )
  const [showCharacterPortraits, setShowCharacterPortraits] = useStorage(
    "chatShowCharacterPortraits",
    true
  )
  const [messageSteeringPrompts, setMessageSteeringPrompts] =
    useStorage<MessageSteeringPromptTemplates>(
      MESSAGE_STEERING_PROMPTS_STORAGE_KEY,
      DEFAULT_MESSAGE_STEERING_PROMPTS
    )
  const [searchText, setSearchText] = useState("")
  const [dropdownOpen, setDropdownOpen] = useState(false)
  const [isImporting, setIsImporting] = useState(false)
  const [activeTab, setActiveTab] = useState<"character" | "persona">(
    selectedAssistant?.kind === "persona" ? "persona" : "character"
  )
  const searchInputRef = useRef<InputRef | null>(null)
  const importInputRef = useRef<HTMLInputElement | null>(null)
  const personaImageInputRef = useRef<HTMLInputElement | null>(null)
  const moodImageInputRef = useRef<HTMLInputElement | null>(null)
  const pendingMoodUploadRef = useRef<CharacterMoodLabel | null>(null)
  const isMountedRef = useRef(true)
  const greetingRetryAbortRef = useRef<AbortController | null>(null)
  const selectedCharacterIdRef = useRef<string | null>(
    selectedCharacterId ?? null
  )
  const imageOnlyModalRef = useRef<
    ReturnType<ReturnType<typeof useAntdModal>["confirm"]> | null
  >(null)
  const imageOnlyModalResolveRef = useRef<((value: boolean) => void) | null>(
    null
  )
  const { capabilities } = useServerCapabilities()

  const hasCharacters = capabilities?.hasCharacters
  const hasPersona = capabilities?.hasPersona

  const destroyImageOnlyModal = React.useCallback(() => {
    if (!imageOnlyModalRef.current) return
    imageOnlyModalRef.current.destroy()
    imageOnlyModalRef.current = null
  }, [])

  useEffect(() => {
    return () => {
      isMountedRef.current = false
      greetingRetryAbortRef.current?.abort()
      greetingRetryAbortRef.current = null
      if (imageOnlyModalResolveRef.current) {
        imageOnlyModalResolveRef.current(false)
        imageOnlyModalResolveRef.current = null
      }
      destroyImageOnlyModal()
    }
  }, [destroyImageOnlyModal])

  useEffect(() => {
    selectedCharacterIdRef.current = selectedCharacterId ?? null
  }, [selectedCharacterId])

  useEffect(() => {
    if (selectedAssistant?.kind === "character" || selectedAssistant?.kind === "persona") {
      setActiveTab(selectedAssistant.kind)
    }
  }, [selectedAssistant?.kind])

  useEffect(() => {
    if (activeTab === "persona" && !hasPersona) {
      setActiveTab("character")
    }
  }, [activeTab, hasPersona])

  const { data: characters = [], isLoading, refetch } = useQuery<
    CharacterApiResponse[]
  >({
    queryKey: ["characters-list"],
    queryFn: async () => {
      await tldwClient.initialize().catch(() => null)
      const result = await tldwClient.listAllCharacters()
      return result as CharacterApiResponse[]
    },
    enabled: !!hasCharacters,
    staleTime: 5 * 60 * 1000 // 5 minutes
  })
  const { data: personas = [] } = useQuery<PersonaInfo[]>({
    queryKey: ["persona-profiles", "sidepanel-character-select"],
    queryFn: async () => {
      await tldwClient.initialize().catch(() => null)
      const result = await tldwClient.listPersonaProfiles()
      return Array.isArray(result) ? (result as PersonaInfo[]) : []
    },
    enabled: !!hasPersona,
    staleTime: 5 * 60 * 1000
  })

  // Filter characters based on search
  const filteredCharacters = useMemo<CharacterApiResponse[]>(() => {
    if (!characters) return []
    if (!searchText.trim()) return characters
    const q = searchText.toLowerCase()
    return characters.filter(
      (char) =>
        char.name?.toLowerCase().includes(q) ||
        char.description?.toLowerCase().includes(q) ||
        char.tags?.some((tag) => tag.toLowerCase().includes(q))
    )
  }, [characters, searchText])
  const filteredPersonas = useMemo<PersonaInfo[]>(() => {
    if (!personas) return []
    if (!searchText.trim()) return personas
    const q = searchText.toLowerCase()
    return personas.filter((persona) =>
      String(persona.name || "").toLowerCase().includes(q)
    )
  }, [personas, searchText])

  const favoriteIndex = useMemo(() => {
    const ids = new Set<string>()
    const slugs = new Set<string>()
    const names = new Set<string>()
    ;(favoriteCharacters || []).forEach((fav) => {
      if (fav.id) ids.add(String(fav.id))
      if (fav.slug) slugs.add(String(fav.slug))
      if (fav.name) names.add(String(fav.name))
    })
    return { ids, slugs, names }
  }, [favoriteCharacters])

  const getCharacterDisplayName = React.useCallback(
    (character: CharacterApiResponse) =>
      (character.name ||
        character.title ||
        character.slug ||
        (character.id != null ? String(character.id) : "")).toString(),
    []
  )

  const isFavoriteCharacter = React.useCallback(
    (character: CharacterApiResponse) => {
      const id = character.id != null ? String(character.id) : ""
      const slug = character.slug ? String(character.slug) : ""
      const name = getCharacterDisplayName(character)
      return (
        (id && favoriteIndex.ids.has(id)) ||
        (slug && favoriteIndex.slugs.has(slug)) ||
        (name && favoriteIndex.names.has(name))
      )
    },
    [favoriteIndex, getCharacterDisplayName]
  )

  const toggleFavoriteCharacter = React.useCallback(
    (character: CharacterApiResponse) => {
      const name = getCharacterDisplayName(character).trim()
      const id = character.id != null ? String(character.id) : undefined
      const slug = character.slug ? String(character.slug) : undefined
      if (!name) return
      void setFavoriteCharacters((prev) => {
        const list = Array.isArray(prev) ? prev : []
        const next = list.filter((fav) => {
          if (id && fav.id && fav.id === id) return false
          if (slug && fav.slug && fav.slug === slug) return false
          if (name && fav.name === name) return false
          return true
        })
        if (next.length === list.length) {
          next.push({ id, slug, name })
        }
        return next
      })
    },
    [getCharacterDisplayName, setFavoriteCharacters]
  )

  const sortedCharacters = useMemo(() => {
    const list = filteredCharacters || []
    const byName = (a: CharacterApiResponse, b: CharacterApiResponse) =>
      getCharacterDisplayName(a).localeCompare(getCharacterDisplayName(b))
    if (sortMode === "favorites") {
      const favorites = list.filter(isFavoriteCharacter).sort(byName)
      const others = list.filter((char) => !isFavoriteCharacter(char)).sort(byName)
      return { favorites, others }
    }
    return { favorites: [] as CharacterApiResponse[], others: list.slice().sort(byName) }
  }, [filteredCharacters, getCharacterDisplayName, isFavoriteCharacter, sortMode])

  const selectedCharacter = useMemo(() => {
    if (!selectedCharacterId || !characters) return null
    return characters.find(
      (char) => String(char.id) === String(selectedCharacterId)
    )
  }, [characters, selectedCharacterId])
  const selectedPersona = useMemo(() => {
    if (selectedAssistant?.kind !== "persona") return null
    return (
      personas.find(
        (persona) => String(persona.id ?? "") === String(selectedAssistant.id)
      ) ??
      selectedAssistant
    )
  }, [personas, selectedAssistant])
  const selectedPersonaSource = useMemo(() => {
    if (selectedAssistant?.kind !== "persona") return null
    return personas.some(
      (persona) => String(persona.id ?? "") === String(selectedAssistant.id)
    )
      ? "catalog"
      : "selected-assistant-fallback"
  }, [personas, selectedAssistant])
  const selectedCharacterMoodImages = useMemo(
    () =>
      getCharacterMoodImagesFromExtensions(
        selectedCharacter?.extensions ?? selectedCharacterStored?.extensions
      ),
    [selectedCharacter?.extensions, selectedCharacterStored?.extensions]
  )
  const hasActiveChat = useMemo(() => {
    if (serverChatId) return true
    return messages.some(
      (message) => message.messageType !== "character:greeting"
    )
  }, [messages, serverChatId])
  const hasUserPersonaImage = useMemo(() => {
    return (
      typeof userPersonaImage === "string" &&
      userPersonaImage.trim().length > 0
    )
  }, [userPersonaImage])

  useEffect(() => {
    let context: Parameters<typeof setBuddyShellRenderContext>[0] = null

    if (selectedAssistant?.kind === "persona") {
      const personaId = String(selectedAssistant.id || "").trim()
      if (personaId) {
        context = {
          surface_id: "sidepanel-chat",
          surface_active: true,
          active_persona_id: personaId,
          position_bucket: "sidepanel-desktop",
          buddy_summary: selectedPersona?.buddy_summary ?? null,
          persona_source: selectedPersonaSource
        }
      }
    }

    setBuddyShellRenderContext(context)
    return () => {
      setBuddyShellRenderContext(null)
    }
  }, [
    selectedAssistant,
    selectedPersona,
    selectedPersonaSource,
    setBuddyShellRenderContext
  ])
  const trimmedDisplayName = userDisplayName.trim()
  const displayNameActionLabel = trimmedDisplayName
    ? (t("sidepanel:characterSelect.displayNameCurrent", {
        defaultValue: "Your name: {{name}}",
        name: trimmedDisplayName
      }) as string)
    : (t("sidepanel:characterSelect.displayNameAction", {
        defaultValue: "Set your name"
      }) as string)
  const imageActionLabel = hasUserPersonaImage
    ? (t("sidepanel:characterSelect.identityImageReplace", {
        defaultValue: "Replace your image"
      }) as string)
    : (t("sidepanel:characterSelect.identityImageUpload", {
        defaultValue: "Upload your image"
      }) as string)
  const clearImageActionLabel = hasUserPersonaImage
    ? (t("sidepanel:characterSelect.identityImageClear", {
        defaultValue: "Remove your image"
      }) as string)
    : undefined
  const promptTemplatesActionLabel = t(
    "sidepanel:characterSelect.promptTemplatesAction",
    {
      defaultValue: "Prompt style templates"
    }
  ) as string
  const displayNameInputRef = useRef(trimmedDisplayName)
  const steeringPromptDraftRef = useRef<MessageSteeringPromptTemplates>(
    normalizeMessageSteeringPrompts(messageSteeringPrompts)
  )

  useEffect(() => {
    steeringPromptDraftRef.current = normalizeMessageSteeringPrompts(
      messageSteeringPrompts
    )
  }, [messageSteeringPrompts])

  const buildStoredCharacter = React.useCallback(
    (character: Partial<CharacterApiResponse>): StoredCharacter | null => {
      const id = character?.id
      const name = character?.name
      if (!id || !name) return null
      const avatar =
        character.avatar_url ||
        (character.image_base64
          ? `data:${character.image_mime || "image/png"};base64,${
              character.image_base64
            }`
          : undefined)
      return {
        id: String(id),
        name,
        avatar_url: avatar ?? null,
        image_base64: character.image_base64 ?? null,
        image_mime: character.image_mime ?? null,
        tags: character.tags,
        greeting: pickGreeting(collectGreetings(character)) || null,
        extensions: character.extensions ?? null,
        version:
          typeof character.version === "number" &&
          Number.isFinite(character.version)
            ? character.version
            : undefined
      }
    },
    []
  )

  const confirmCharacterSwitch = React.useCallback(
    (nextName?: string) =>
      confirmWithModal({
        title: t("sidepanel:characterSelect.switchConfirmTitle", {
          defaultValue: "Switch character?"
        }),
        content: t("sidepanel:characterSelect.switchConfirmBody", {
          defaultValue: nextName
            ? "Switching to {{name}} will clear the current chat. Continue?"
            : "Changing the character will clear the current chat. Continue?",
          name: nextName
        }),
        okText: t("sidepanel:characterSelect.switchConfirmOk", {
          defaultValue: "Clear chat & switch"
        }),
        cancelText: t("common:cancel", { defaultValue: "Cancel" }),
        centered: true,
        okButtonProps: { danger: true }
      }),
    [confirmWithModal, t]
  )

  const openDisplayNameModal = React.useCallback(() => {
    displayNameInputRef.current = trimmedDisplayName
    modal.confirm({
      title: t("sidepanel:characterSelect.displayNameTitle", {
        defaultValue: "Set your name"
      }),
      content: (
        <div className="space-y-2">
          <Input
            autoFocus
            defaultValue={trimmedDisplayName}
            placeholder={t("sidepanel:characterSelect.displayNamePlaceholder", {
              defaultValue: "Enter a display name"
            }) as string}
            onChange={(event) => {
              displayNameInputRef.current = event.target.value
            }}
          />
          <div className="text-xs text-text-muted">
            {t("sidepanel:characterSelect.displayNameHelp", {
              defaultValue: "Used to replace {{user}} and similar placeholders."
            })}
          </div>
        </div>
      ),
      okText: t("sidepanel:characterSelect.displayNameSave", {
        defaultValue: "Save"
      }),
      cancelText: t("common:cancel", { defaultValue: "Cancel" }),
      centered: true,
      maskClosable: false,
      onOk: () => {
        setUserDisplayName(displayNameInputRef.current.trim())
      }
    })
  }, [modal, setUserDisplayName, t, trimmedDisplayName])

  const openGenerationPromptsModal = React.useCallback(() => {
    const current = normalizeMessageSteeringPrompts(messageSteeringPrompts)
    steeringPromptDraftRef.current = { ...current }
    React.startTransition(() => {
      modal.confirm({
        title: t("sidepanel:characterSelect.editPromptsTitle", {
          defaultValue: "Prompt style templates"
        }),
        width: 720,
        centered: true,
        maskClosable: false,
        content: (
          <div className="space-y-3">
            <div className="text-xs text-text-muted">
              {t("sidepanel:characterSelect.editPromptsHelp", {
                defaultValue:
                  "These templates are used when running Continue as user, Impersonate user, and Force narrate."
              })}
            </div>
            <div>
              <div className="mb-1 text-xs font-medium text-text">
                {t("sidepanel:characterSelect.continueAsUser", {
                  defaultValue: "Continue as user"
                })}
              </div>
              <Input.TextArea
                rows={3}
                defaultValue={current.continueAsUser}
                onChange={(event) => {
                  steeringPromptDraftRef.current = {
                    ...steeringPromptDraftRef.current,
                    continueAsUser: event.target.value
                  }
                }}
              />
            </div>
            <div>
              <div className="mb-1 text-xs font-medium text-text">
                {t("sidepanel:characterSelect.impersonateUser", {
                  defaultValue: "Impersonate user"
                })}
              </div>
              <Input.TextArea
                rows={3}
                defaultValue={current.impersonateUser}
                onChange={(event) => {
                  steeringPromptDraftRef.current = {
                    ...steeringPromptDraftRef.current,
                    impersonateUser: event.target.value
                  }
                }}
              />
            </div>
            <div>
              <div className="mb-1 text-xs font-medium text-text">
                {t("sidepanel:characterSelect.forceNarrate", {
                  defaultValue: "Force narrate"
                })}
              </div>
              <Input.TextArea
                rows={3}
                defaultValue={current.forceNarrate}
                onChange={(event) => {
                  steeringPromptDraftRef.current = {
                    ...steeringPromptDraftRef.current,
                    forceNarrate: event.target.value
                  }
                }}
              />
            </div>
          </div>
        ),
        okText: t("common:save", { defaultValue: "Save" }),
        cancelText: t("common:cancel", { defaultValue: "Cancel" }),
        onOk: async () => {
          const next = normalizeMessageSteeringPrompts(
            steeringPromptDraftRef.current
          )
          await new Promise<void>((resolve) => {
            React.startTransition(() => {
              Promise.resolve(setMessageSteeringPrompts(next)).finally(() =>
                resolve()
              )
            })
          })
          notification.success({
            message: t("sidepanel:characterSelect.editPromptsSaved", {
              defaultValue: "Prompt style templates saved"
            })
          })
        }
      })
    })
  }, [messageSteeringPrompts, modal, notification, setMessageSteeringPrompts, t])

  const handlePersonaImageUploadClick = React.useCallback(() => {
    if (!personaImageInputRef.current) return
    setDropdownOpen(false)
    personaImageInputRef.current.value = ""
    personaImageInputRef.current.click()
  }, [setDropdownOpen])

  const handlePersonaImageFile = React.useCallback(
    async (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0]
      if (!file) return

      try {
        if (!file.type.startsWith("image/")) {
          notification.error({
            message: t("settings:manageCharacters.notification.error", {
              defaultValue: "Error"
            }),
            description: t("common:upload.imageRequired", {
              defaultValue: "Please select an image file."
            })
          })
          return
        }

        if (file.size > MAX_PERSONA_IMAGE_BYTES) {
          notification.error({
            message: t("settings:manageCharacters.notification.error", {
              defaultValue: "Error"
            }),
            description: t("common:upload.imageTooLarge", {
              defaultValue:
                "Please choose a smaller image (around 5 MB or less)."
            })
          })
          return
        }

        const dataUrl = await new Promise<string>((resolve, reject) => {
          const reader = new FileReader()
          reader.onload = () => {
            const result = reader.result
            if (typeof result === "string" && result.startsWith("data:image/")) {
              resolve(result)
              return
            }
            reject(new Error("Invalid image payload"))
          }
          reader.onerror = () => {
            reject(reader.error || new Error("Failed to read image"))
          }
          reader.readAsDataURL(file)
        })

        await setUserPersonaImage(dataUrl)
        await setShowCharacterPortraits(true)
        notification.success({
          message: t("sidepanel:characterSelect.personaImageSaved", {
            defaultValue: "Persona image updated"
          })
        })
      } catch (error) {
        const messageText =
          error instanceof Error ? error.message : String(error)
        notification.error({
          message: t("settings:manageCharacters.notification.error", {
            defaultValue: "Error"
          }),
          description:
            messageText ||
            t("settings:manageCharacters.notification.someError", {
              defaultValue: "Something went wrong. Please try again later"
            })
        })
      } finally {
        event.target.value = ""
      }
    },
    [notification, setShowCharacterPortraits, setUserPersonaImage, t]
  )

  const clearPersonaImage = React.useCallback(() => {
    void setUserPersonaImage("")
    notification.success({
      message: t("sidepanel:characterSelect.personaImageRemoved", {
        defaultValue: "Persona image removed"
      })
    })
    setDropdownOpen(false)
  }, [notification, setDropdownOpen, setUserPersonaImage, t])

  const openDisplayNameFromIdentityMenu = React.useCallback(() => {
    setDropdownOpen(false)
    openDisplayNameModal()
  }, [openDisplayNameModal, setDropdownOpen])

  const openPromptTemplatesFromIdentityMenu = React.useCallback(() => {
    setDropdownOpen(false)
    openGenerationPromptsModal()
  }, [openGenerationPromptsModal, setDropdownOpen])

  const handleMoodImageUploadClick = React.useCallback(
    (mood: CharacterMoodLabel) => {
      if (!selectedCharacterId) {
        notification.warning({
          message: t("sidepanel:characterSelect.moodPortraitNeedsCharacter", {
            defaultValue: "Select a character first"
          })
        })
        return
      }
      if (!moodImageInputRef.current) return
      pendingMoodUploadRef.current = mood
      setDropdownOpen(false)
      moodImageInputRef.current.value = ""
      moodImageInputRef.current.click()
    },
    [notification, selectedCharacterId, setDropdownOpen, t]
  )

  const handleMoodImageFile = React.useCallback(
    async (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0]
      const pendingMood = pendingMoodUploadRef.current
      const activeCharacterId = selectedCharacterId
      const moodOption = CHARACTER_MOOD_OPTIONS.find(
        (option) => option.key === pendingMood
      )
      const moodLabel = moodOption?.label ?? pendingMood ?? "mood"

      try {
        if (!file || !pendingMood || !activeCharacterId) {
          return
        }

        if (!file.type.startsWith("image/")) {
          notification.error({
            message: t("settings:manageCharacters.notification.error", {
              defaultValue: "Error"
            }),
            description: t("common:upload.imageRequired", {
              defaultValue: "Please select an image file."
            })
          })
          return
        }

        if (file.size > MAX_MOOD_IMAGE_BYTES) {
          notification.error({
            message: t("settings:manageCharacters.notification.error", {
              defaultValue: "Error"
            }),
            description: t("common:upload.imageTooLarge", {
              defaultValue:
                "Please choose a smaller image (around 5 MB or less)."
            })
          })
          return
        }

        const dataUrl = await new Promise<string>((resolve, reject) => {
          const reader = new FileReader()
          reader.onload = () => {
            const result = reader.result
            if (typeof result === "string" && result.startsWith("data:image/")) {
              resolve(result)
              return
            }
            reject(new Error("Invalid image payload"))
          }
          reader.onerror = () => {
            reject(reader.error || new Error("Failed to read image"))
          }
          reader.readAsDataURL(file)
        })

        await tldwClient.initialize().catch(() => null)
        const fetchedCharacter = (await tldwClient.getCharacter(
          activeCharacterId
        )) as CharacterApiResponse | null
        const baseCharacter =
          fetchedCharacter ||
          selectedCharacter ||
          selectedCharacterStored ||
          ({ id: activeCharacterId } as CharacterApiResponse)
        const nextExtensions = upsertCharacterMoodImage(
          baseCharacter?.extensions,
          pendingMood,
          dataUrl
        )
        const expectedVersion =
          typeof baseCharacter?.version === "number" &&
          Number.isFinite(baseCharacter.version)
            ? baseCharacter.version
            : undefined
        const updatedCharacter = (await tldwClient.updateCharacter(
          activeCharacterId,
          { extensions: nextExtensions },
          expectedVersion
        )) as CharacterApiResponse
        const normalized = buildStoredCharacter(
          updatedCharacter || {
            ...(baseCharacter || {}),
            extensions: nextExtensions
          }
        )
        if (normalized && String(normalized.id) === String(activeCharacterId)) {
          await setSelectedCharacter(normalized)
        }
        await setShowCharacterPortraits(true)
        await refetch({ cancelRefetch: true })
        notification.success({
          message: t("sidepanel:characterSelect.moodPortraitSaved", {
            defaultValue: "{{mood}} mood portrait updated",
            mood: moodLabel
          })
        })
      } catch (error) {
        const messageText =
          error instanceof Error ? error.message : String(error)
        notification.error({
          message: t("settings:manageCharacters.notification.error", {
            defaultValue: "Error"
          }),
          description:
            messageText ||
            t("settings:manageCharacters.notification.someError", {
              defaultValue: "Something went wrong. Please try again later"
            })
        })
      } finally {
        pendingMoodUploadRef.current = null
        event.target.value = ""
      }
    },
    [
      buildStoredCharacter,
      notification,
      refetch,
      selectedCharacter,
      selectedCharacterId,
      selectedCharacterStored,
      setSelectedCharacter,
      setShowCharacterPortraits,
      t
    ]
  )

  const clearMoodImage = React.useCallback(
    async (mood: CharacterMoodLabel) => {
      if (!selectedCharacterId) {
        notification.warning({
          message: t("sidepanel:characterSelect.moodPortraitNeedsCharacter", {
            defaultValue: "Select a character first"
          })
        })
        return
      }
      const moodOption = CHARACTER_MOOD_OPTIONS.find(
        (option) => option.key === mood
      )
      const moodLabel = moodOption?.label ?? mood
      try {
        await tldwClient.initialize().catch(() => null)
        const fetchedCharacter = (await tldwClient.getCharacter(
          selectedCharacterId
        )) as CharacterApiResponse | null
        const baseCharacter =
          fetchedCharacter ||
          selectedCharacter ||
          selectedCharacterStored ||
          ({ id: selectedCharacterId } as CharacterApiResponse)
        const currentImages = getCharacterMoodImagesFromExtensions(
          baseCharacter?.extensions
        )
        if (!currentImages[mood]) return

        const nextExtensions = removeCharacterMoodImage(
          baseCharacter?.extensions,
          mood
        )
        const expectedVersion =
          typeof baseCharacter?.version === "number" &&
          Number.isFinite(baseCharacter.version)
            ? baseCharacter.version
            : undefined
        const updatedCharacter = (await tldwClient.updateCharacter(
          selectedCharacterId,
          { extensions: nextExtensions },
          expectedVersion
        )) as CharacterApiResponse
        const normalized = buildStoredCharacter(
          updatedCharacter || {
            ...(baseCharacter || {}),
            extensions: nextExtensions
          }
        )
        if (normalized) {
          await setSelectedCharacter(normalized)
        }
        await refetch({ cancelRefetch: true })
        notification.success({
          message: t("sidepanel:characterSelect.moodPortraitRemoved", {
            defaultValue: "{{mood}} mood portrait removed",
            mood: moodLabel
          })
        })
      } catch (error) {
        const messageText =
          error instanceof Error ? error.message : String(error)
        notification.error({
          message: t("settings:manageCharacters.notification.error", {
            defaultValue: "Error"
          }),
          description:
            messageText ||
            t("settings:manageCharacters.notification.someError", {
              defaultValue: "Something went wrong. Please try again later"
            })
        })
      } finally {
        setDropdownOpen(false)
      }
    },
    [
      buildStoredCharacter,
      notification,
      refetch,
      selectedCharacter,
      selectedCharacterId,
      selectedCharacterStored,
      setDropdownOpen,
      setSelectedCharacter,
      t
    ]
  )

  const applySelection = React.useCallback(
    async (nextId: string | null, stored: StoredCharacter | null) => {
      const currentId = selectedCharacterId ?? null
      if (nextId === currentId) {
        setDropdownOpen(false)
        return
      }

      if (hasActiveChat) {
        const confirmed = await confirmCharacterSwitch(stored?.name)
        if (!confirmed) return
        clearChat()
      }

      greetingRetryAbortRef.current?.abort()
      greetingRetryAbortRef.current = null
      setSelectedCharacterId(nextId)
      if (!nextId) {
        await setSelectedCharacter(null)
        await setSelectedAssistant(null)
        setDropdownOpen(false)
        return
      }

      if (!stored) {
        await setSelectedCharacter(null)
      } else {
        await setSelectedCharacter(stored)
        await setSelectedAssistant(characterToAssistantSelection(stored))
      }
      setDropdownOpen(false)

      const shouldHydrateGreetingOrExtensions =
        Boolean(stored) && (!stored?.greeting || stored?.extensions == null)
      if (!shouldHydrateGreetingOrExtensions) return

      const retryController = new AbortController()
      greetingRetryAbortRef.current = retryController

      const hydrateGreeting = async () => {
        try {
          await tldwClient.initialize().catch(() => null)
          if (!isMountedRef.current) return false
          const full = await tldwClient.getCharacter(nextId)
          if (!isMountedRef.current) return false
          const hydrated = buildStoredCharacter(full || {})
          if (
            hydrated?.id === String(nextId) &&
            hydrated.greeting &&
            selectedCharacterIdRef.current === nextId &&
            isMountedRef.current
          ) {
            await setSelectedCharacter(hydrated)
          }
          return true
        } catch (error) {
          console.warn("Failed to hydrate character greeting:", error)
          return false
        }
      }

      void (async () => {
        try {
          const ok = await hydrateGreeting()
          if (!isMountedRef.current || retryController.signal.aborted) return
          if (!ok && selectedCharacterIdRef.current === nextId) {
            await delayWithAbort(
              GREETING_RETRY_DELAY_MS,
              retryController.signal
            )
            if (!isMountedRef.current || retryController.signal.aborted) return
            const retried = await hydrateGreeting()
            if (!isMountedRef.current || retryController.signal.aborted) return
            if (!retried && selectedCharacterIdRef.current === nextId) {
              notification.warning({
                message: t(
                  "settings:manageCharacters.notification.error",
                  "Error"
                ),
                description: t(
                  "settings:manageCharacters.notification.someError",
                  "Couldn't load the character greeting. Try again later."
                )
              })
            }
          }
        } finally {
          if (greetingRetryAbortRef.current === retryController) {
            greetingRetryAbortRef.current = null
          }
        }
      })()
    },
    [
      buildStoredCharacter,
      clearChat,
      confirmCharacterSwitch,
      hasActiveChat,
      notification,
      selectedCharacterId,
      setSelectedAssistant,
      setSelectedCharacter,
      setSelectedCharacterId,
      t
    ]
  )

  const handleSelect = React.useCallback(
    (id: string | null) => {
      const selected = id
        ? characters?.find((char) => String(char.id) === String(id))
        : null
      const stored = selected ? buildStoredCharacter(selected) : null
      void applySelection(id, stored)
    },
    [applySelection, buildStoredCharacter, characters]
  )

  const handlePersonaSelect = React.useCallback(
    async (persona: PersonaInfo) => {
      const selection = personaToAssistantSelection({
        ...persona,
        id: String(persona.id ?? ""),
        name: String(persona.name ?? "Persona")
      })
      if (!selection) return
      if (
        selectedAssistant?.kind === "persona" &&
        String(selectedAssistant.id) === String(selection.id)
      ) {
        setDropdownOpen(false)
        return
      }
      if (hasActiveChat) {
        const confirmed = await confirmCharacterSwitch(selection.name)
        if (!confirmed) return
        clearChat()
      }
      greetingRetryAbortRef.current?.abort()
      greetingRetryAbortRef.current = null
      setSelectedCharacterId(null)
      await setSelectedCharacter(null)
      await setSelectedAssistant(selection)
      setDropdownOpen(false)
    },
    [
      clearChat,
      confirmCharacterSwitch,
      hasActiveChat,
      selectedAssistant?.id,
      selectedAssistant?.kind,
      setSelectedAssistant,
      setSelectedCharacter,
      setSelectedCharacterId
    ]
  )

  const handleImportClick = React.useCallback(() => {
    if (isImporting) return
    if (!importInputRef.current) return
    setDropdownOpen(false)
    importInputRef.current.value = ""
    importInputRef.current.click()
  }, [isImporting, setDropdownOpen])

  const showImportError = React.useCallback(
    (error: unknown) => {
      const messageText = error instanceof Error ? error.message : String(error)
      notification.error({
        message: t("settings:manageCharacters.notification.error", {
          defaultValue: "Error"
        }),
        description:
          messageText ||
          t("settings:manageCharacters.notification.someError", {
            defaultValue: "Something went wrong. Please try again later"
          })
      })
    },
    [notification, t]
  )

  const handleImportFile = React.useCallback(async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0]
    if (!file) return

    const getImageOnlyDetail = (error: unknown): ImageOnlyErrorDetail | null => {
      const details: unknown = (error as { details?: unknown })?.details
      if (!details || typeof details !== "object") return null
      const detailCandidate =
        "detail" in details ? (details as { detail?: unknown }).detail : details
      if (!detailCandidate || typeof detailCandidate !== "object") return null
      const code = (detailCandidate as { code?: unknown }).code
      if (typeof code !== "string") return null
      if (code === "missing_character_data") {
        return detailCandidate as ImageOnlyErrorDetail
      }
      return null
    }

    const confirmImageOnlyImport = (message?: string) =>
      confirmWithModal(
        {
          title: t("settings:manageCharacters.imageOnlyTitle", {
            defaultValue: "No character data detected"
          }),
          content:
            message ||
            t("settings:manageCharacters.imageOnlyBody", {
              defaultValue:
                "No character data was found in the image metadata. Import this image as a character anyway?"
            }),
          okText: t("settings:manageCharacters.imageOnlyConfirm", {
            defaultValue: "Import image-only"
          }),
          cancelText: t("common:cancel", { defaultValue: "Cancel" }),
          centered: true,
          okButtonProps: { danger: false },
          maskClosable: false
        },
        { instance: imageOnlyModalRef, resolver: imageOnlyModalResolveRef }
      ).finally(() => {
        imageOnlyModalResolveRef.current = null
        destroyImageOnlyModal()
      })

    const importCharacter = async (allowImageOnly = false) =>
      await tldwClient.importCharacterFile(file, {
        allowImageOnly
      })
    const handleImportSuccess = (imported: ImportCharacterResponse) => {
      const importedCharacter =
        imported?.character ??
        (imported?.id && imported?.name
          ? { id: imported.id, name: imported.name }
          : imported)
      const successDetail =
        typeof imported?.message === "string" && imported.message.trim()
          ? imported.message
          : undefined
      notification.success({
        message: t("settings:manageCharacters.notification.addSuccess", {
          defaultValue: "Character created"
        }),
        description: successDetail
      })
      refetch({ cancelRefetch: true })
      const createdId = (() => {
        if (!importedCharacter || typeof importedCharacter !== "object") {
          return null
        }
        const candidate = importedCharacter as Record<string, unknown>
        const resolveId = (value: unknown) =>
          typeof value === "string" || typeof value === "number"
            ? value
            : null
        return (
          resolveId(candidate.id) ??
          resolveId(candidate.character_id) ??
          resolveId(candidate.characterId)
        )
      })()
      if (createdId != null) {
        const stored = buildStoredCharacter(importedCharacter ?? {})
        void applySelection(String(createdId), stored)
      }
    }

    try {
      setIsImporting(true)
      await tldwClient.initialize().catch(() => null)
      const imported = await importCharacter()
      handleImportSuccess(imported)
    } catch (error) {
      const imageOnlyDetail = getImageOnlyDetail(error)
      if (imageOnlyDetail) {
        const confirmed = await confirmImageOnlyImport(
          imageOnlyDetail?.message
        )
        if (confirmed) {
          try {
            const imported = await importCharacter(true)
            handleImportSuccess(imported)
          } catch (retryError) {
            showImportError(retryError)
          }
        }
        return
      }
      showImportError(error)
    } finally {
      setIsImporting(false)
      event.target.value = ""
    }
  }, [
    applySelection,
    buildStoredCharacter,
    confirmWithModal,
    destroyImageOnlyModal,
    notification,
    refetch,
    setIsImporting,
    showImportError,
    t
  ])

  const buildCharactersRoute = React.useCallback((create?: boolean) => {
    return buildCharactersRouteUrl({ from: "sidepanel-character-select", create })
  }, [])

  const buildCharactersHash = React.useCallback((create?: boolean) => {
    return buildCharactersHashUrl({ from: "sidepanel-character-select", create })
  }, [])

  const openCharactersWorkspace = React.useCallback(
    async (options?: { create?: boolean }) => {
      if (typeof window === "undefined") return
      const route = buildCharactersRoute(options?.create)
      const hash = buildCharactersHash(options?.create)
      const pathname = window.location.pathname || ""
      const optionsPath = `/options.html${hash}`
      const runtime = getBrowserRuntime()
      const mode = resolveCharactersDestinationMode({
        pathname,
        extensionRuntime: isExtensionRuntime(runtime)
      })

      if (mode === "options-in-place") {
        const base = window.location.href.replace(/#.*$/, "")
        window.location.href = `${base}${hash}`
        return
      }

      if (mode === "options-tab") {
        const url = runtime?.getURL ? runtime.getURL(optionsPath) : optionsPath
        try {
          if (browser.tabs?.create) {
            await browser.tabs.create({ url })
            return
          }
        } catch (error) {
          console.debug(
            "[CharacterSelect] Failed to open characters workspace tab:",
            error
          )
        }

        window.open(url, "_blank")
        return
      }

      try {
        window.open(route, "_blank")
      } catch (error) {
        console.debug(
          "[CharacterSelect] Failed to open characters workspace route:",
          error
        )
      }
    },
    [buildCharactersHash, buildCharactersRoute]
  )

  const createCharacterItem = React.useCallback(
    (char: CharacterApiResponse): MenuItemType => {
      const isFavorite = isFavoriteCharacter(char)
      const favoriteTitle = isFavorite
        ? t("sidepanel:characterSelect.favoriteRemove", "Remove from favorites")
        : t("sidepanel:characterSelect.favoriteAdd", "Add to favorites")
      return {
        key: String(char.id),
        label: (
          <div className="w-56 py-0.5 flex items-center gap-2">
            {char.avatar_url ? (
              <Avatar src={char.avatar_url} size="small" />
            ) : (
              <Avatar size="small" icon={<User2 className="size-3" />} />
            )}
            <div className="flex-1 min-w-0">
              <div className="truncate font-medium">{char.name}</div>
              {char.description && (
                <p className="text-xs text-text-subtle line-clamp-1">
                  {char.description}
                </p>
              )}
            </div>
            <button
              type="button"
              className="rounded p-0.5 text-text-subtle transition hover:bg-surface2"
              onMouseDown={(event) => {
                event.preventDefault()
                event.stopPropagation()
              }}
              onClick={(event) => {
                event.preventDefault()
                event.stopPropagation()
                toggleFavoriteCharacter(char)
              }}
              aria-label={favoriteTitle}
              title={favoriteTitle}
            >
              <Star
                className={`h-3.5 w-3.5 ${
                  isFavorite ? "fill-warn text-warn" : "text-text-subtle"
                }`}
              />
            </button>
          </div>
        ),
        onClick: () => handleSelect(String(char.id))
      }
    },
    [handleSelect, isFavoriteCharacter, t, toggleFavoriteCharacter]
  )

  const menuItems = useMemo<ItemType[]>(() => {
    const createLabel = t(
      "sidepanel:characterSelect.createNewCharacter",
      "Create a New Character+"
    )
    const importLabel = t(
      "sidepanel:characterSelect.importCharacter",
      "Import Character"
    )
    const createItem: ItemType = {
      key: "create",
      label: (
        <div className="w-56 py-0.5 flex items-center gap-2 text-primary font-medium">
          <span>{createLabel}</span>
        </div>
      ),
      onClick: () => {
        setDropdownOpen(false)
        void openCharactersWorkspace({ create: true })
      }
    }
    const importItem: ItemType = {
      key: "import",
      label: (
        <div className="w-56 py-0.5 flex items-center gap-2 text-primary font-medium">
          <span>{importLabel}</span>
        </div>
      ),
      onClick: handleImportClick
    }

    if (isLoading) {
      return [
        {
          key: "loading",
          label: (
            <div className="text-text-muted text-sm py-2">
              {t("common:loading.title", { defaultValue: "Loading..." })}
            </div>
          ),
          disabled: true
        },
        { type: "divider" },
        createItem,
        importItem
      ]
    }

    if (filteredCharacters.length === 0) {
      return [
        {
          key: "empty",
          label: (
            <Empty
              description={
                searchText
                  ? t("sidepanel:characterSelect.noMatches", "No matching characters")
                  : t("sidepanel:characterSelect.noCharacters", "No characters available")
              }
              image={Empty.PRESENTED_IMAGE_SIMPLE}
            />
          ),
          disabled: true
        },
        { type: "divider" },
        createItem,
        importItem
      ]
    }

    const items: ItemType[] = []

    // Add "None" option to clear selection
    items.push({
      key: "none",
      label: (
        <div className="w-56 py-0.5 flex items-center gap-2 text-text-muted">
          <X className="size-4" />
          <span>{t("sidepanel:characterSelect.none", "No character")}</span>
        </div>
      ),
      onClick: () => handleSelect(null)
    })

    items.push({
      key: "persona-portrait-toggle",
      label: (
        <div className="w-56 py-0.5 flex items-center gap-2 text-text-muted">
          <span>
            {showCharacterPortraits
              ? t("sidepanel:characterSelect.hidePortraits", {
                  defaultValue: "Hide large portraits"
                })
              : t("sidepanel:characterSelect.showPortraits", {
                  defaultValue: "Show large portraits"
                })}
          </span>
        </div>
      ),
      onClick: () => {
        void setShowCharacterPortraits((prev) => !prev)
        setDropdownOpen(false)
      }
    })

    if (selectedCharacterId) {
      items.push({ type: "divider" })
      items.push({
        key: "mood-portraits-heading",
        label: (
          <div className="w-full text-left text-[11px] font-semibold uppercase tracking-wide text-text-subtle">
            {t("sidepanel:characterSelect.moodPortraits", {
              defaultValue: "Mood portraits"
            })}
          </div>
        ),
        disabled: true
      })
      CHARACTER_MOOD_OPTIONS.forEach((moodOption) => {
        const hasMoodImage = Boolean(selectedCharacterMoodImages[moodOption.key])
        items.push({
          key: `mood-upload-${moodOption.key}`,
          label: (
            <div className="w-56 py-0.5 flex items-center gap-2 text-text-muted">
              <span>
                {hasMoodImage
                  ? t("sidepanel:characterSelect.moodPortraitReplace", {
                      defaultValue: "Replace {{mood}} mood portrait",
                      mood: moodOption.label
                    })
                  : t("sidepanel:characterSelect.moodPortraitSet", {
                      defaultValue: "Set {{mood}} mood portrait",
                      mood: moodOption.label
                    })}
              </span>
            </div>
          ),
          onClick: () => handleMoodImageUploadClick(moodOption.key)
        })
        if (hasMoodImage) {
          items.push({
            key: `mood-clear-${moodOption.key}`,
            label: (
              <div className="w-56 py-0.5 flex items-center gap-2 text-text-muted">
                <span>
                  {t("sidepanel:characterSelect.moodPortraitRemove", {
                    defaultValue: "Remove {{mood}} mood portrait",
                    mood: moodOption.label
                  })}
                </span>
              </div>
            ),
            onClick: () => {
              void clearMoodImage(moodOption.key)
            }
          })
        }
      })
    }

    items.push(createItem, importItem)
    items.push({ type: "divider" })

    const favoriteItems = sortedCharacters.favorites.map(createCharacterItem)
    const otherItems = sortedCharacters.others.map(createCharacterItem)

    if (sortMode === "favorites" && favoriteItems.length > 0) {
      items.push({
        type: "group",
        label: t("sidepanel:characterSelect.favorites", "Favorites"),
        children: favoriteItems
      })
    }

    // Add character items
    items.push(...otherItems)

    return items
  }, [
    filteredCharacters,
    createCharacterItem,
    handleImportClick,
    handleSelect,
    isLoading,
    clearMoodImage,
    handleMoodImageUploadClick,
    openCharactersWorkspace,
    searchText,
    selectedCharacterId,
    selectedCharacterMoodImages,
    setShowCharacterPortraits,
    showCharacterPortraits,
    setDropdownOpen,
    sortedCharacters,
    sortMode,
    t
  ])

  const personaPanel = useMemo(() => {
    if (!hasPersona) {
      return (
        <div className="px-3 py-4 text-center text-sm text-text-subtle">
          {t("sidepanel:characterSelect.personaUnavailable", {
            defaultValue: "Personas are not available on this server."
          })}
        </div>
      )
    }

    if (filteredPersonas.length === 0) {
      return (
        <div className="px-3 py-4 text-center text-sm text-text-subtle">
          {searchText
            ? t("sidepanel:characterSelect.noPersonaMatches", {
                defaultValue: "No matching personas."
              })
            : t("sidepanel:characterSelect.noPersonas", {
                defaultValue: "No personas available."
              })}
        </div>
      )
    }

    return (
      <div className="max-h-[320px] overflow-y-auto px-2 py-2">
        <div className="flex flex-col gap-1">
          {filteredPersonas.map((persona) => {
            const personaId = String(persona.id ?? "")
            const isActive =
              selectedAssistant?.kind === "persona" &&
              String(selectedAssistant.id) === personaId
            return (
              <button
                key={personaId}
                type="button"
                className={`w-full rounded-md border px-3 py-2 text-left text-sm transition ${
                  isActive
                    ? "border-primary bg-primary/10 text-text"
                    : "border-border bg-background text-text hover:bg-surface2"
                }`}
                onClick={() => {
                  void handlePersonaSelect(persona)
                }}
              >
                <span className="block truncate font-medium">
                  {String(persona.name ?? personaId ?? "Persona")}
                </span>
                <span className="block truncate text-xs text-text-subtle">
                  {t("sidepanel:characterSelect.personaLabel", {
                    defaultValue: "Persona"
                  })}
                </span>
              </button>
            )
          })}
        </div>
      </div>
    )
  }, [
    filteredPersonas,
    handlePersonaSelect,
    hasPersona,
    searchText,
    selectedAssistant?.id,
    selectedAssistant?.kind,
    t
  ])

  // Focus search input when dropdown opens
  useEffect(() => {
    if (!dropdownOpen) {
      setSearchText("")
      return
    }

    let frameId: number | null = null
    let attempts = 0
    let canceled = false
    const focusWhenReady = () => {
      if (canceled) return
      if (searchInputRef.current) {
        try {
          searchInputRef.current.focus({ preventScroll: true } as any)
        } catch {
          searchInputRef.current.focus()
        }
        return
      }
      if (attempts < 10) {
        attempts += 1
        frameId = window.requestAnimationFrame(focusWhenReady)
      }
    }

    frameId = window.requestAnimationFrame(focusWhenReady)

    return () => {
      canceled = true
      if (frameId !== null) {
        window.cancelAnimationFrame(frameId)
      }
    }
  }, [dropdownOpen])

  // Don't render if neither assistant source is available
  if (!hasCharacters && !hasPersona) {
    return null
  }

  return (
    <div>
      <input
        ref={importInputRef}
        type="file"
        accept=".json,.yaml,.yml,.txt,.md,.png,.webp,.jpg,.jpeg"
        className="hidden"
        onChange={handleImportFile}
      />
      <input
        ref={personaImageInputRef}
        type="file"
        accept="image/png,image/jpeg,image/webp,image/gif"
        className="hidden"
        onChange={handlePersonaImageFile}
      />
      <input
        ref={moodImageInputRef}
        type="file"
        accept="image/png,image/jpeg,image/webp,image/gif"
        className="hidden"
        onChange={handleMoodImageFile}
      />
      <Dropdown
        open={dropdownOpen}
        onOpenChange={setDropdownOpen}
        menu={{
          items: menuItems,
          style: { maxHeight: 400, overflowY: "auto" },
          className: `no-scrollbar ${
            menuDensity === "compact"
              ? "menu-density-compact"
              : "menu-density-comfortable"
          }`,
          activeKey: selectedCharacterId ?? undefined
        }}
        popupRender={(menu) => (
          <div className="bg-surface rounded-lg shadow-lg border border-border">
            <div className="p-2 border-b border-border flex items-center gap-2">
              <Input
                ref={searchInputRef}
                placeholder={
                  activeTab === "persona"
                    ? t(
                        "sidepanel:characterSelect.searchPersonas",
                        "Search personas..."
                      )
                    : t(
                        "sidepanel:characterSelect.search",
                        "Search characters..."
                      )
                }
                prefix={<Search className="size-4 text-text-subtle" />}
                value={searchText}
                onChange={(e) => setSearchText(e.target.value)}
                allowClear
                size="small"
                className="flex-1"
                onKeyDown={(e) => e.stopPropagation()}
              />
              {activeTab === "character" ? (
                <Select
                  size="small"
                  value={sortMode}
                  onChange={(value) => setSortMode(value as CharacterSortMode)}
                  options={[
                    {
                      value: "favorites",
                      label: t("sidepanel:characterSelect.sort.favorites", "Favorites")
                    },
                    { value: "az", label: t("sidepanel:characterSelect.sort.az", "A-Z") }
                  ]}
                  className="min-w-[110px]"
                  onKeyDown={(event) => event.stopPropagation()}
                />
              ) : null}
            </div>
            <MyChatIdentityMenu
              className="border-b border-border"
              displayNameLabel={displayNameActionLabel}
              imageLabel={imageActionLabel}
              clearImageLabel={clearImageActionLabel}
              promptTemplatesLabel={promptTemplatesActionLabel}
              onDisplayName={openDisplayNameFromIdentityMenu}
              onImage={handlePersonaImageUploadClick}
              onPromptTemplates={openPromptTemplatesFromIdentityMenu}
              onClearImage={clearImageActionLabel ? clearPersonaImage : undefined}
            />
            {hasCharacters && hasPersona ? (
              <div className="flex items-center gap-1 border-b border-border px-2 pt-2">
                {(["character", "persona"] as const).map((tab) => {
                  const isActive = activeTab === tab
                  return (
                    <button
                      key={tab}
                      type="button"
                      className={`rounded-t-md px-3 py-2 text-sm transition ${
                        isActive
                          ? "bg-surface2 font-medium text-text"
                          : "text-text-subtle hover:text-text"
                      }`}
                      onClick={() => setActiveTab(tab)}
                    >
                      {tab === "persona"
                        ? t("sidepanel:characterSelect.personasTab", "Personas")
                        : t("sidepanel:characterSelect.charactersTab", "Characters")}
                    </button>
                  )
                })}
              </div>
            ) : null}
            {activeTab === "persona" ? personaPanel : menu}
          </div>
        )}
        placement="topLeft"
        trigger={["click"]}
      >
        <Tooltip
          title={
            hasPersona
              ? t("sidepanel:characterSelect.tooltipAssistant", "Select an assistant")
              : t("sidepanel:characterSelect.tooltip", "Select a character")
          }
        >
          <IconButton
            ariaLabel={
              (hasPersona
                ? t(
                    "sidepanel:characterSelect.tooltipAssistant",
                    "Select an assistant"
                  )
                : t(
                    "sidepanel:characterSelect.tooltip",
                    "Select a character"
                  )) as string
            }
            hasPopup="menu"
            dataTestId="chat-character-select"
            className={className}
          >
            {selectedAssistant?.kind === "persona" &&
            typeof selectedPersona?.avatar_url === "string" &&
            selectedPersona.avatar_url ? (
              <Avatar
                src={selectedPersona.avatar_url}
                size="small"
                className="size-5"
              />
            ) : selectedCharacter?.avatar_url ? (
              <Avatar
                src={selectedCharacter.avatar_url}
                size="small"
                className="size-5"
              />
            ) : (
              <User2 className={iconClassName} />
            )}
            <span className="ml-1 hidden max-w-[100px] truncate text-xs font-medium text-text sm:inline">
              {selectedAssistant?.kind === "persona"
                ? String(
                    selectedPersona?.name ||
                      selectedAssistant.name ||
                      t("sidepanel:characterSelect.personaLabel", "Persona")
                  )
                : selectedCharacter?.name ||
                  t("sidepanel:characterSelect.label", "Character")}
            </span>
          </IconButton>
        </Tooltip>
      </Dropdown>
    </div>
  )
}

export default CharacterSelect

import { useQuery } from "@tanstack/react-query"
import {
  Dropdown,
  Tooltip,
  Input,
  Select,
  type MenuProps,
  type InputRef
} from "antd"
import { Star, UserCircle2 } from "lucide-react"
import React from "react"
import { useStorage } from "@plasmohq/storage/hook"
import { useTranslation } from "react-i18next"
import { browser } from "wxt/browser"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { IconButton } from "./IconButton"
import { MyChatIdentityMenu } from "./MyChatIdentityMenu"
import { useAntdNotification } from "@/hooks/useAntdNotification"
import { useAntdModal } from "@/hooks/useAntdModal"
import { useConfirmModal } from "@/hooks/useConfirmModal"
import { useSelectedCharacter } from "@/hooks/useSelectedCharacter"
import { collectGreetings } from "@/utils/character-greetings"
import { createImageDataUrl } from "@/utils/image-utils"
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
import { useClearChat } from "@/hooks/chat/useClearChat"
import { useStoreMessageOption } from "@/store/option"
import { preserveAssistantSelectionMode } from "@/types/assistant-selection"
import type { MessageSteeringPromptTemplates } from "@/types/message-steering"
import { getBrowserRuntime, isExtensionRuntime } from "@/utils/browser-runtime"
import {
  buildCharactersHash as buildCharactersHashUrl,
  buildCharactersRoute as buildCharactersRouteUrl,
  resolveCharactersDestinationMode
} from "@/utils/characters-route"

type Props = {
  className?: string
  iconClassName?: string
  showLabel?: boolean
}

type CharacterSummary = {
  id?: string | number
  version?: number
  slug?: string
  name?: string
  title?: string
  description?: string
  tags?: string[]
  avatar_url?: string
  image_base64?: string
  image_mime?: string
  system_prompt?: string
  systemPrompt?: string
  instructions?: string
  greeting?: string
  first_message?: string
  firstMessage?: string
  greet?: string
  alternate_greetings?: string[] | string | null
  alternateGreetings?: string[] | string | null
  extensions?: Record<string, unknown> | string | null
}

type ImportedCharacterResponse = {
  character?: CharacterSummary
  id?: string | number
  name?: string
  message?: string
}

type ImageOnlyErrorDetail = {
  code?: string
  message?: string
}

type ImportError = Error & {
  details?: ImageOnlyErrorDetail | { detail?: ImageOnlyErrorDetail }
}

type CharacterSelection = {
  id: string
  version?: number
  name: string
  system_prompt: string
  greeting: string
  alternate_greetings?: string[]
  avatar_url: string
  image_base64?: string | null
  image_mime?: string | null
  extensions?: Record<string, unknown> | string | null
}

type CharacterSortMode = "favorites" | "az"

type FavoriteCharacter = {
  id?: string
  slug?: string
  name: string
}

const MAX_PERSONA_IMAGE_BYTES = 5 * 1024 * 1024
const MAX_MOOD_IMAGE_BYTES = 5 * 1024 * 1024

const normalizeCharacter = (
  character: CharacterSummary
): CharacterSelection | null => {
  const idSource =
    character.id ?? character.slug ?? character.name ?? character.title ?? ""
  const nameSource = character.name ?? character.title ?? character.slug ?? ""

  if (!idSource || !nameSource) {
    return null
  }

  const avatar =
    character.avatar_url ||
    (character.image_base64
      ? createImageDataUrl(character.image_base64) || ""
      : "")

  const greetings = collectGreetings(character)
  const [primaryGreeting, ...alternateGreetings] = greetings

  return {
    id: String(idSource),
    version:
      typeof character.version === "number" && Number.isFinite(character.version)
        ? character.version
        : undefined,
    name: String(nameSource),
    system_prompt:
      character.system_prompt ||
      character.systemPrompt ||
      character.instructions ||
      "",
    // Keep greeting deterministic; choose a randomized greeting at injection time.
    greeting: primaryGreeting ?? "",
    alternate_greetings:
      alternateGreetings.length > 0 ? alternateGreetings : undefined,
    avatar_url: avatar,
    image_base64:
      typeof character.image_base64 === "string"
        ? character.image_base64
        : null,
    image_mime:
      typeof character.image_mime === "string" ? character.image_mime : null,
    extensions: character.extensions ?? null
  }
}

export const CharacterSelect: React.FC<Props> = ({
  className = "text-text-muted",
  iconClassName = "size-5",
  showLabel = true
}) => {
  const { t } = useTranslation(["option", "common", "settings", "playground"])
  const notification = useAntdNotification()
  const modal = useAntdModal()
  const confirmWithModal = useConfirmModal()
  const [selectedCharacter, setSelectedCharacter] =
    useSelectedCharacter<CharacterSelection | null>(null)
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
  const latestSelectionIdRef = React.useRef<string | null>(null)
  const lastErrorRef = React.useRef<unknown | null>(null)
  const importInputRef = React.useRef<HTMLInputElement | null>(null)
  const personaImageInputRef = React.useRef<HTMLInputElement | null>(null)
  const moodImageInputRef = React.useRef<HTMLInputElement | null>(null)
  const pendingMoodUploadRef = React.useRef<CharacterMoodLabel | null>(null)
  const imageOnlyModalRef = React.useRef<ReturnType<typeof modal.confirm> | null>(
    null
  )
  const displayNameModalRef = React.useRef<ReturnType<typeof modal.confirm> | null>(
    null
  )
  const confirmResolveRef = React.useRef<((confirmed: boolean) => void) | null>(
    null
  )
  const [isImporting, setIsImporting] = React.useState(false)
  const steeringPromptDraftRef = React.useRef<MessageSteeringPromptTemplates>(
    normalizeMessageSteeringPrompts(messageSteeringPrompts)
  )

  React.useEffect(() => {
    steeringPromptDraftRef.current = normalizeMessageSteeringPrompts(
      messageSteeringPrompts
    )
  }, [messageSteeringPrompts])

  const preserveSelectionMode = React.useCallback(
    (next: CharacterSelection | null): CharacterSelection | null =>
      preserveAssistantSelectionMode(next, selectedCharacter),
    [selectedCharacter]
  )

  const { data, refetch, isFetching, isLoading, error } = useQuery<CharacterSummary[]>({
    queryKey: ["tldw:listCharacters"],
    queryFn: async () => {
      await tldwClient.initialize()
      const list = await tldwClient.listAllCharacters()
      return Array.isArray(list) ? list : []
    },
    // Cache characters so we don't refetch on every open.
    staleTime: 1000 * 60 * 10,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
    refetchOnMount: false
  })

  const [menuDensity] = useStorage<"comfortable" | "compact">(
    "menuDensity",
    "comfortable"
  )
  const [favoriteCharacters, setFavoriteCharacters] = useStorage<FavoriteCharacter[]>(
    "favoriteCharacters",
    []
  )
  const [sortMode, setSortMode] = useStorage<CharacterSortMode>(
    "characterSortMode",
    "favorites"
  )
  const [searchQuery, setSearchQuery] = React.useState("")
  const [dropdownOpen, setDropdownOpen] = React.useState(false)
  const searchInputRef = React.useRef<InputRef | null>(null)
  const selectLabel = t("option:characters.selectCharacter", {
    defaultValue: "Select character"
  }) as string
  const clearLabel = t("option:characters.clearCharacter", {
    defaultValue: "Clear character"
  }) as string
  const emptyTitle = t("settings:manageCharacters.emptyTitle", {
    defaultValue: "No characters yet"
  }) as string
  const emptyDescription = t("settings:manageCharacters.emptyDescription", {
    defaultValue:
      "Create a reusable character with a name, description, and system prompt you can chat with."
  }) as string
  const emptyCreateLabel = t("settings:manageCharacters.emptyPrimaryCta", {
    defaultValue: "Create character"
  }) as string
  const createNewLabel = t("option:characters.createNewCharacter", {
    defaultValue: "Create a New Character+"
  }) as string
  const importLabel = t("option:characters.importCharacter", {
    defaultValue: "Import Character"
  }) as string
  const openPageLabel = t("option:characters.openCharactersPage", {
    defaultValue: "Characters Page"
  }) as string
  const searchPlaceholder = t("option:characters.searchPlaceholder", {
    defaultValue: "Search characters by name"
  }) as string
  const trimmedDisplayName =
    typeof userDisplayName === "string" ? userDisplayName.trim() : ""
  const displayNameActionLabel = trimmedDisplayName
    ? (t("option:characters.displayNameCurrent", {
        defaultValue: "Your name: {{name}}",
        name: trimmedDisplayName
      }) as string)
    : (t("option:characters.displayNameAction", {
        defaultValue: "Set your name"
      }) as string)
  const hasUserPersonaImage =
    typeof userPersonaImage === "string" && userPersonaImage.trim().length > 0
  const imageActionLabel = hasUserPersonaImage
    ? (t("option:characters.identityImageReplace", {
        defaultValue: "Replace your image"
      }) as string)
    : (t("option:characters.identityImageUpload", {
        defaultValue: "Upload your image"
      }) as string)
  const clearImageActionLabel = hasUserPersonaImage
    ? (t("option:characters.identityImageClear", {
        defaultValue: "Remove your image"
      }) as string)
    : undefined
  const promptTemplatesActionLabel = t(
    "option:characters.promptTemplatesAction",
    {
      defaultValue: "Prompt style templates"
    }
  ) as string
  const selectedCharacterMoodImages = React.useMemo(
    () => getCharacterMoodImagesFromExtensions(selectedCharacter?.extensions),
    [selectedCharacter?.extensions]
  )
  const hasActiveChat = React.useMemo(() => {
    if (serverChatId) return true
    return messages.some(
      (message) => message.messageType !== "character:greeting"
    )
  }, [messages, serverChatId])
  const openDisplayNameModal = React.useCallback(() => {
    let nextValue = trimmedDisplayName
    displayNameModalRef.current = modal.confirm({
      title: t("option:characters.displayNameTitle", {
        defaultValue: "Set your name"
      }),
      content: (
        <div className="space-y-2">
          <Input
            autoFocus
            defaultValue={trimmedDisplayName}
            placeholder={t("option:characters.displayNamePlaceholder", {
              defaultValue: "Enter a display name"
            }) as string}
            onChange={(event) => {
              nextValue = event.target.value
            }}
          />
          <div className="text-xs text-text-muted">
            {t("option:characters.displayNameHelp", {
              defaultValue: "Used to replace {{user}} and similar placeholders."
            })}
          </div>
        </div>
      ),
      okText: t("option:characters.displayNameSave", {
        defaultValue: "Save"
      }),
      cancelText: t("common:cancel", { defaultValue: "Cancel" }),
      centered: true,
      maskClosable: false,
      onOk: () => {
        setUserDisplayName(nextValue.trim())
      },
      afterClose: () => {
        displayNameModalRef.current = null
      }
    })
  }, [modal, setUserDisplayName, t, trimmedDisplayName])

  const openGenerationPromptsModal = React.useCallback(() => {
    const current = normalizeMessageSteeringPrompts(messageSteeringPrompts)
    steeringPromptDraftRef.current = {
      ...current
    }
    React.startTransition(() => {
      modal.confirm({
        title: t("playground:composer.steering.editPromptsTitle", {
          defaultValue: "Prompt style templates"
        }),
        width: 720,
        centered: true,
        maskClosable: false,
        content: (
          <div className="space-y-3">
            <div className="text-xs text-text-muted">
              {t("playground:composer.steering.editPromptsHelp", {
                defaultValue:
                  "These templates are used when running Continue as user, Impersonate user, and Force narrate."
              })}
            </div>
            <div>
              <div className="mb-1 text-xs font-medium text-text">
                {t("playground:composer.steering.continue", {
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
                {t("playground:composer.steering.impersonate", {
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
                {t("playground:composer.steering.narrate", {
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
            message: t("playground:composer.steering.editPromptsSaved", {
              defaultValue: "Prompt style templates saved"
            })
          })
        }
      })
    })
  }, [messageSteeringPrompts, modal, notification, setMessageSteeringPrompts, t])

  const handlePersonaImageUploadClick = React.useCallback(() => {
    if (!personaImageInputRef.current) return
    personaImageInputRef.current.value = ""
    personaImageInputRef.current.click()
  }, [])

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
          message: t("option:characters.personaImageSaved", {
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
      message: t("option:characters.personaImageRemoved", {
        defaultValue: "Persona image removed"
      })
    })
  }, [notification, setUserPersonaImage, t])

  const openDisplayNameFromIdentityMenu = React.useCallback(() => {
    setDropdownOpen(false)
    openDisplayNameModal()
  }, [openDisplayNameModal, setDropdownOpen])

  const openPromptTemplatesFromIdentityMenu = React.useCallback(() => {
    setDropdownOpen(false)
    openGenerationPromptsModal()
  }, [openGenerationPromptsModal, setDropdownOpen])

  const uploadImageFromIdentityMenu = React.useCallback(() => {
    setDropdownOpen(false)
    handlePersonaImageUploadClick()
  }, [handlePersonaImageUploadClick, setDropdownOpen])

  const clearImageFromIdentityMenu = React.useCallback(() => {
    setDropdownOpen(false)
    clearPersonaImage()
  }, [clearPersonaImage, setDropdownOpen])

  const handleMoodImageUploadClick = React.useCallback(
    (mood: CharacterMoodLabel) => {
      if (!selectedCharacter?.id) {
        notification.warning({
          message: t("option:characters.moodPortraitNeedsCharacter", {
            defaultValue: "Select a character first"
          })
        })
        return
      }
      if (!moodImageInputRef.current) return
      pendingMoodUploadRef.current = mood
      moodImageInputRef.current.value = ""
      moodImageInputRef.current.click()
    },
    [notification, selectedCharacter?.id, t]
  )

  const handleMoodImageFile = React.useCallback(
    async (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0]
      const pendingMood = pendingMoodUploadRef.current
      const activeCharacterId = selectedCharacter?.id
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
        )) as CharacterSummary
        const baseCharacter = fetchedCharacter || selectedCharacter
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
          {
            extensions: nextExtensions
          },
          expectedVersion
        )) as CharacterSummary
        const normalized = normalizeCharacter(
          updatedCharacter || {
            ...(baseCharacter || {}),
            extensions: nextExtensions
          }
        )
        if (normalized) {
          await setSelectedCharacter(preserveSelectionMode(normalized))
        }
        await setShowCharacterPortraits(true)
        await refetch({ cancelRefetch: true })

        notification.success({
          message: t("option:characters.moodPortraitSaved", {
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
      notification,
      preserveSelectionMode,
      refetch,
      selectedCharacter,
      setSelectedCharacter,
      setShowCharacterPortraits,
      t
    ]
  )

  const clearMoodImage = React.useCallback(
    async (mood: CharacterMoodLabel) => {
      if (!selectedCharacter?.id) {
        notification.warning({
          message: t("option:characters.moodPortraitNeedsCharacter", {
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
          selectedCharacter.id
        )) as CharacterSummary
        const baseCharacter = fetchedCharacter || selectedCharacter
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
          selectedCharacter.id,
          { extensions: nextExtensions },
          expectedVersion
        )) as CharacterSummary
        const normalized = normalizeCharacter(
          updatedCharacter || {
            ...(baseCharacter || {}),
            extensions: nextExtensions
          }
        )
        if (normalized) {
          await setSelectedCharacter(preserveSelectionMode(normalized))
        }
        await refetch({ cancelRefetch: true })
        notification.success({
          message: t("option:characters.moodPortraitRemoved", {
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
      }
    },
    [
      notification,
      preserveSelectionMode,
      refetch,
      selectedCharacter,
      setSelectedCharacter,
      t
    ]
  )

  const confirmCharacterSwitch = React.useCallback(
    (nextName?: string) =>
      confirmWithModal(
        {
          title: t("option:characters.switchConfirmTitle", {
            defaultValue: "Switch character?"
          }),
          content: t("option:characters.switchConfirmBody", {
            defaultValue: nextName
              ? "Switching to {{name}} will clear the current chat. Continue?"
              : "Changing the character will clear the current chat. Continue?",
            name: nextName
          }),
          okText: t("option:characters.switchConfirmOk", {
            defaultValue: "Clear chat & switch"
          }),
          cancelText: t("common:cancel", { defaultValue: "Cancel" }),
          centered: true,
          okButtonProps: { danger: true }
        },
        { resolver: confirmResolveRef }
      ),
    [confirmWithModal, t]
  )

  const applySelection = React.useCallback(
    async (next: CharacterSelection | null) => {
      const nextId = next?.id ?? null
      const currentId = selectedCharacter?.id ?? null
      if (nextId === currentId) return

      if (hasActiveChat) {
        const confirmed = await confirmCharacterSwitch(next?.name)
        if (!confirmed) return
        clearChat()
      }

      latestSelectionIdRef.current = nextId
      await setSelectedCharacter(next)
      if (next?.name) {
        notification.success({
          message: t("option:characters.chattingAs", {
            defaultValue: "You are chatting with {{name}}.",
            name: next.name
          })
        })
      }

      const shouldHydrateGreetingOrExtensions =
        Boolean(next) &&
        (!next?.greeting || next?.extensions == null)

      if (next && shouldHydrateGreetingOrExtensions) {
        const targetId = next.id
        void tldwClient
          .initialize()
          .catch(() => null)
          .then(() => tldwClient.getCharacter(targetId))
          .then((full) => {
            if (latestSelectionIdRef.current !== targetId) return
            const hydrated = normalizeCharacter(full || {})
            if (hydrated && hydrated.id === targetId && hydrated.greeting) {
              void setSelectedCharacter(preserveSelectionMode(hydrated))
            }
          })
          .catch((error) => {
            if (latestSelectionIdRef.current !== targetId) return
            console.warn("Failed to hydrate character greeting:", error)
            notification.warning({
              message: t("settings:manageCharacters.notification.error", "Error"),
              description: t(
                "settings:manageCharacters.notification.someError",
                "Couldn't load the character greeting. Try again later."
              )
            })
          })
      }
    },
    [
      clearChat,
      confirmCharacterSwitch,
      hasActiveChat,
      notification,
      preserveSelectionMode,
      selectedCharacter?.id,
      setSelectedCharacter,
      t
    ]
  )

  React.useEffect(() => {
    latestSelectionIdRef.current = selectedCharacter?.id ?? null
  }, [selectedCharacter?.id])

  const handleImportSuccess = React.useCallback(
    (imported: ImportedCharacterResponse) => {
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
      const normalized = normalizeCharacter(importedCharacter || {})
      if (normalized) {
        void applySelection(normalized)
      }
    },
    [applySelection, notification, refetch, t]
  )

  React.useEffect(() => {
    return () => {
      if (confirmResolveRef.current) {
        confirmResolveRef.current(false)
        confirmResolveRef.current = null
      }
    }
  }, [])

  React.useEffect(() => {
    if (!error || isFetching) {
      lastErrorRef.current = null
      return
    }

    if (lastErrorRef.current === error) {
      return
    }

    lastErrorRef.current = error

    notification.error({
      message: t(
        "option:characters.fetchErrorTitle",
        "Unable to load characters"
      ),
      description: t(
        "option:characters.fetchErrorBody",
        "Check your connection or server health, then try again."
      ),
      placement: "bottomRight",
      duration: 3
    })
  }, [error, isFetching, notification, t])

  const buildCharactersRoute = React.useCallback((create?: boolean) => {
    return buildCharactersRouteUrl({ from: "header-select", create })
  }, [])

  const buildCharactersHash = React.useCallback((create?: boolean) => {
    return buildCharactersHashUrl({ from: "header-select", create })
  }, [])

  const handleOpenCharacters = React.useCallback((options?: { create?: boolean }) => {
    try {
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
        // If we're already inside options UI, switch routes in-place.
        const base = window.location.href.replace(/#.*$/, "")
        window.location.href = `${base}${hash}`
        return
      }

      if (mode === "options-tab") {
        // Outside options.html in extension runtime: open options page in a new tab.
        try {
          const url = runtime?.getURL ? runtime.getURL(optionsPath) : optionsPath
          if (browser.tabs?.create) {
            browser.tabs.create({ url })
          } else {
            window.open(url, "_blank")
          }
          return
        } catch {
          // fall through to options path fallback
        }

        window.open(optionsPath, "_blank")
        return
      }

      // In web runtime, open the direct Next.js route.
      window.open(route, "_blank")
    } catch {
      // ignore navigation errors
    }
  }, [buildCharactersHash, buildCharactersRoute])

  const handleOpenCreate = React.useCallback(() => {
    handleOpenCharacters({ create: true })
  }, [handleOpenCharacters])

  React.useEffect(() => {
    return () => {
      imageOnlyModalRef.current?.destroy()
      imageOnlyModalRef.current = null
      displayNameModalRef.current = null
    }
  }, [])

  const handleImportClick = React.useCallback(() => {
    if (isImporting) return
    if (!importInputRef.current) return
    importInputRef.current.value = ""
    importInputRef.current.click()
  }, [isImporting])

  const handleImportFile = React.useCallback(
    async (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0]
      if (!file) return

      const getImageOnlyDetail = (error: unknown): ImageOnlyErrorDetail | null => {
        const details = (error as ImportError)?.details
        if (!details || typeof details !== "object") return null
        const candidate =
          "detail" in details ? (details as { detail?: unknown }).detail : details
        if (!candidate || typeof candidate !== "object") return null
        const code = (candidate as { code?: unknown }).code
        if (code === "missing_character_data") return candidate as ImageOnlyErrorDetail
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
          { instance: imageOnlyModalRef }
        )

      const importCharacter = (allowImageOnly = false) =>
        tldwClient.importCharacterFile(file, { allowImageOnly })

      try {
        setIsImporting(true)
        try {
          await tldwClient.initialize()
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : String(error)
          throw new Error(
            `Failed to initialize character service: ${errorMessage}`
          )
        }
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
              const messageText =
                retryError instanceof Error
                  ? retryError.message
                  : String(retryError)
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
            }
          }
          return
        }
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
        setIsImporting(false)
        event.target.value = ""
      }
    },
    [confirmWithModal, handleImportSuccess, notification, t]
  )

  const filteredCharacters = React.useMemo(() => {
    const list = Array.isArray(data) ? data : []
    const q = searchQuery.trim().toLowerCase()
    if (!q) return list
    return list.filter((c) => {
      const name = (c.name || c.title || c.slug || "").toString().toLowerCase()
      const description = (c.description || "").toString().toLowerCase()
      const tags = Array.isArray(c.tags) ? c.tags.join(" ").toLowerCase() : ""
      return (
        name.includes(q) ||
        description.includes(q) ||
        tags.includes(q)
      )
    })
  }, [data, searchQuery])

  const favoriteIndex = React.useMemo(() => {
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

  const getCharacterDisplayName = React.useCallback((character: CharacterSummary) => {
    return (
      character.name ||
      character.title ||
      character.slug ||
      (character.id != null ? String(character.id) : "")
    ).toString()
  }, [])

  const isFavoriteCharacter = React.useCallback(
    (character: CharacterSummary) => {
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
    (character: CharacterSummary) => {
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

  const sortedCharacters = React.useMemo(() => {
    const list = filteredCharacters || []
    const byName = (a: CharacterSummary, b: CharacterSummary) =>
      getCharacterDisplayName(a).localeCompare(getCharacterDisplayName(b))
    if (sortMode === "favorites") {
      const favorites = list.filter(isFavoriteCharacter).sort(byName)
      const others = list.filter((c) => !isFavoriteCharacter(c)).sort(byName)
      return { favorites, others }
    }
    return { favorites: [] as CharacterSummary[], others: list.slice().sort(byName) }
  }, [filteredCharacters, getCharacterDisplayName, isFavoriteCharacter, sortMode])

  const characterItems = React.useMemo<MenuProps["items"]>(() => {
    const buildItem = (character: CharacterSummary, index: number) => {
      const normalized = normalizeCharacter(character)
      if (!normalized) {
        console.debug(
          "[CharacterSelect] Skipping character with invalid id/name",
          {
            id: character.id,
            slug: character.slug,
            name: character.name,
            title: character.title
          }
        )
        return null
      }
      const displayName = getCharacterDisplayName(character)
      const menuKey =
        character.id ??
        character.slug ??
        character.name ??
        character.title ??
        `character-${index}`
      const isFavorite = isFavoriteCharacter(character)
      const favoriteTitle = isFavorite
        ? t("option:characters.favoriteRemove", "Remove from favorites")
        : t("option:characters.favoriteAdd", "Add to favorites")

      return {
        key: String(menuKey),
        label: (
          <div className="w-56 gap-2 text-sm inline-flex items-center leading-5">
            {normalized.avatar_url ? (
              <img
                src={normalized.avatar_url}
                alt={displayName || normalized.id || `Character ${menuKey}`}
                className="w-4 h-4 rounded-full"
              />
            ) : (
              <UserCircle2 className="w-4 h-4" />
            )}
            <span className="truncate flex-1">
              {displayName || normalized.id || String(menuKey)}
            </span>
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
                toggleFavoriteCharacter(character)
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
        onClick: () => {
          void applySelection(normalized)
        }
      } as MenuProps["items"][number]
    }

    const favorites = sortedCharacters.favorites
      .map(buildItem)
      .filter(Boolean) as MenuProps["items"]
    const others = sortedCharacters.others
      .map(buildItem)
      .filter(Boolean) as MenuProps["items"]

    const items: MenuProps["items"] = []
    if (sortMode === "favorites" && favorites.length > 0) {
      items.push({
        type: "group",
        key: "__favorites__",
        label: t("option:characters.favorites", "Favorites"),
        children: favorites
      })
    }
    items.push(...others)
    return items
  }, [
    applySelection,
    getCharacterDisplayName,
    isFavoriteCharacter,
    sortedCharacters,
    sortMode,
    t,
    toggleFavoriteCharacter
  ])

  const clearItem: MenuProps["items"][number] | null =
    selectedCharacter
      ? {
          key: "__clear__",
          label: (
            <button
              type="button"
              className="w-full text-left text-xs font-medium text-text hover:text-text-muted"
            >
              {t(
                "option:characters.clearCharacter",
                "Clear character"
              ) as string}
            </button>
          ),
          onClick: () => {
            void applySelection(null)
          }
        }
      : null

  const refreshItem: MenuProps["items"][number] = {
    key: "__refresh__",
    label: (
      <button
        type="button"
        className="w-full text-left text-xs font-medium text-primary hover:text-primaryStrong"
      >
        {isFetching
          ? t("option:characters.refreshing", "Refreshing characters…")
          : t("option:characters.refresh", "Refresh characters")}
      </button>
    ),
    onClick: () => {
      refetch({ cancelRefetch: true })
    }
  } as const

  const dividerItem = (key: string): MenuProps["items"][number] => ({
    type: "divider",
    key
  })

  const menuItems: MenuProps["items"] = []

  const noneItem: MenuProps["items"][number] = {
    key: "__none__",
    label: (
      <button
        type="button"
        className="w-full text-left text-xs font-medium text-text hover:text-text-muted"
      >
        {t("option:characters.none", "None (no character)") as string}
      </button>
    ),
    onClick: () => {
      void applySelection(null)
    }
  }

  const createItem: MenuProps["items"][number] = {
    key: "__create_character__",
    label: (
      <button
        type="button"
        className="w-full text-left text-xs font-medium text-primary hover:text-primaryStrong"
      >
        {createNewLabel}
      </button>
    ),
    onClick: handleOpenCreate
  }

  const openPageItem: MenuProps["items"][number] = {
    key: "__open_characters_page__",
    label: (
      <button
        type="button"
        className="w-full text-left text-xs font-medium text-primary hover:text-primaryStrong"
      >
        {openPageLabel}
      </button>
    ),
    onClick: () => handleOpenCharacters()
  }

  const importItem: MenuProps["items"][number] = {
    key: "__import_character__",
    label: (
      <button
        type="button"
        className="w-full text-left text-xs font-medium text-primary hover:text-primaryStrong"
      >
        {importLabel}
      </button>
    ),
    onClick: handleImportClick
  }

  const togglePortraitsItem: MenuProps["items"][number] = {
    key: "__toggle_character_portraits__",
    label: (
      <button
        type="button"
        className="w-full text-left text-xs font-medium text-text hover:text-text-muted"
      >
        {showCharacterPortraits
          ? t("option:characters.hidePortraits", {
              defaultValue: "Hide large portraits"
            })
          : t("option:characters.showPortraits", {
              defaultValue: "Show large portraits"
            })}
      </button>
    ),
    onClick: () => {
      void setShowCharacterPortraits((prev) => !prev)
    }
  }

  const moodPortraitItems: MenuProps["items"] = selectedCharacter
    ? CHARACTER_MOOD_OPTIONS.flatMap((moodOption) => {
        const hasMoodImage = Boolean(selectedCharacterMoodImages[moodOption.key])
        const uploadItem: MenuProps["items"][number] = {
          key: `__mood_upload_${moodOption.key}__`,
          label: (
            <button
              type="button"
              className="w-full text-left text-xs font-medium text-text hover:text-text-muted"
            >
              {hasMoodImage
                ? t("option:characters.moodPortraitReplace", {
                    defaultValue: "Replace {{mood}} mood portrait",
                    mood: moodOption.label
                  })
                : t("option:characters.moodPortraitSet", {
                    defaultValue: "Set {{mood}} mood portrait",
                    mood: moodOption.label
                  })}
            </button>
          ),
          onClick: () => {
            handleMoodImageUploadClick(moodOption.key)
          }
        }
        if (!hasMoodImage) {
          return [uploadItem]
        }
        const clearItem: MenuProps["items"][number] = {
          key: `__mood_clear_${moodOption.key}__`,
          label: (
            <button
              type="button"
              className="w-full text-left text-xs font-medium text-text hover:text-text-muted"
            >
              {t("option:characters.moodPortraitRemove", {
                defaultValue: "Remove {{mood}} mood portrait",
                mood: moodOption.label
              })}
            </button>
          ),
          onClick: () => {
            void clearMoodImage(moodOption.key)
          }
        }
        return [uploadItem, clearItem]
      })
    : []

  menuItems.push(noneItem, togglePortraitsItem)
  if (selectedCharacter && moodPortraitItems.length > 0) {
    menuItems.push(
      dividerItem("__divider_mood_portraits__"),
      {
        key: "__mood_portraits_heading__",
        label: (
          <div className="w-full text-left text-[11px] font-semibold uppercase tracking-wide text-text-subtle">
            {t("option:characters.moodPortraits", {
              defaultValue: "Mood portraits"
            })}
          </div>
        ),
        disabled: true
      },
      ...moodPortraitItems
    )
  }
  menuItems.push(openPageItem, createItem, importItem)

  if (characterItems && characterItems.length > 0) {
    menuItems.push(dividerItem("__divider_items__"), ...characterItems)
  } else if (isLoading) {
    menuItems.push({
      key: "__loading__",
      label: (
        <div className="w-56 px-2 py-2 text-xs text-text-muted">
          {t("common:loading.title", "Loading…") as string}
        </div>
      )
    })
  } else if (!data || (Array.isArray(data) && data.length === 0)) {
    menuItems.push(
      dividerItem("__divider_empty__"),
      {
        key: "empty",
        label: (
          <div className="w-56 px-2 py-2 text-xs text-text-muted">
            <div className="font-medium text-text">
              {emptyTitle}
            </div>
            <div className="mt-1 text-[11px] text-text-muted">
              {emptyDescription}
            </div>
            <button
              type="button"
              className="mt-2 inline-flex items-center rounded border border-border bg-surface px-2 py-1 text-xs font-medium text-primary hover:border-primary hover:text-primaryStrong">
              {emptyCreateLabel}
            </button>
          </div>
        ),
        onClick: handleOpenCreate
      }
    )
  } else {
    menuItems.push({
      key: "__no_matches__",
      label: (
      <div className="w-56 px-2 py-2 text-xs text-text-muted">
        {t(
          "option:characters.noMatches",
          "No characters match your search yet."
          ) as string}
        </div>
      )
    })
  }

  if (clearItem) {
    menuItems.push(dividerItem("__divider_clear__"), clearItem)
  }

  const actorItem: MenuProps["items"][number] = {
    key: "__actor__",
    label: (
      <button
        type="button"
        className="w-full text-left text-xs font-medium text-text hover:text-text-muted"
      >
        {t(
          "playground:composer.actorTitle",
          "Optional scene context"
        ) as string}
      </button>
    ),
    onClick: () => {
      try {
        if (typeof window !== "undefined") {
          window.dispatchEvent(new CustomEvent("tldw:open-actor-settings"))
        }
      } catch {
        // no-op
      }
    }
  }

  menuItems.push(dividerItem("__divider_actor__"), actorItem)
  menuItems.push(dividerItem("__divider_refresh__"), refreshItem)

  const menuContainerRef = React.useRef<HTMLDivElement | null>(null)
  const menuListRef = React.useRef<HTMLUListElement | null>(null)

  const attachMenuRef = React.useCallback(
    (node: HTMLUListElement | null, ref?: React.Ref<HTMLUListElement>) => {
      menuListRef.current = node
      if (!ref) return
      if (typeof ref === "function") {
        ref(node)
      } else if ("current" in ref) {
        ;(ref as React.MutableRefObject<HTMLUListElement | null>).current = node
      }
    },
    []
  )

  const renderMenuWithRef = React.useCallback(
    (menuNode: React.ReactNode) => {
      if (!React.isValidElement(menuNode)) return menuNode
      const menuElement = menuNode as React.ReactElement
      // AntD menu nodes don't expose a typed ref; grab the internal ref so we can
      // merge it with our own list ref for keyboard focus management.
      const originalRef = (menuElement as any).ref as React.Ref<HTMLUListElement> | undefined
      return React.cloneElement(menuElement, {
        ref: (node: HTMLUListElement | null) => attachMenuRef(node, originalRef)
      } as any)
    },
    [attachMenuRef]
  )

  const focusFirstMenuItem = React.useCallback(() => {
    const firstItem = menuListRef.current?.querySelector<HTMLElement>(
      '[role="menuitem"]:not([aria-disabled="true"])'
    )
    firstItem?.focus()
  }, [])

  React.useEffect(() => {
    if (!dropdownOpen) return

    let frameId: number | null = null
    let attempts = 0
    let canceled = false
    const focusWhenReady = () => {
      if (canceled) return
      const input = searchInputRef.current
      if (input) {
        try {
          input.focus({ preventScroll: true } as any)
        } catch {
          input.focus()
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

  return (
    <div className="flex items-center gap-2">
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
        onOpenChange={(open) => {
          setDropdownOpen(open)
          if (!open) {
            setSearchQuery("")
          }
        }}
        popupRender={(menu) => (
          <div className="w-64" ref={menuContainerRef}>
            <div className="px-2 py-2 border-b border-border flex items-center gap-2">
              <Input
                ref={searchInputRef}
                size="small"
                placeholder={searchPlaceholder}
                value={searchQuery}
                allowClear
                className="flex-1"
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "ArrowDown") {
                    e.preventDefault()
                    focusFirstMenuItem()
                  }
                }}
              />
              <Select
                size="small"
                value={sortMode}
                onChange={(value) => setSortMode(value as CharacterSortMode)}
                options={[
                  {
                    value: "favorites",
                    label: t("option:characters.sort.favorites", "Favorites")
                  },
                  { value: "az", label: t("option:characters.sort.az", "A-Z") }
                ]}
                className="min-w-[110px]"
                onKeyDown={(event) => event.stopPropagation()}
              />
            </div>
            <MyChatIdentityMenu
              className="border-b border-border"
              displayNameLabel={displayNameActionLabel}
              imageLabel={imageActionLabel}
              clearImageLabel={clearImageActionLabel}
              promptTemplatesLabel={promptTemplatesActionLabel}
              onDisplayName={openDisplayNameFromIdentityMenu}
              onImage={uploadImageFromIdentityMenu}
              onPromptTemplates={openPromptTemplatesFromIdentityMenu}
              onClearImage={clearImageActionLabel ? clearImageFromIdentityMenu : undefined}
            />
            <div className="max-h-[420px] overflow-y-auto no-scrollbar">
              {renderMenuWithRef(menu)}
            </div>
          </div>
        )}
        menu={{
          items: menuItems,
          activeKey: selectedCharacter?.id,
          className: `character-select-menu no-scrollbar ${
            menuDensity === "compact"
              ? "menu-density-compact"
              : "menu-density-comfortable"
          }`
        }}
        placement="topLeft"
        trigger={["click"]}>
        <Tooltip
          title={
            selectedCharacter?.name
              ? `${selectedCharacter.name} — ${clearLabel}`
              : selectLabel
          }>
          <div className="relative inline-flex">
            <IconButton
              ariaLabel={
                (selectedCharacter?.name
                  ? `${selectedCharacter.name} — ${clearLabel}`
                  : selectLabel) as string
              }
              hasPopup="menu"
              className={`h-11 w-11 sm:h-7 sm:w-7 sm:min-w-0 sm:min-h-0 ${className}`}>
              {selectedCharacter?.avatar_url ? (
                <img
                  src={selectedCharacter.avatar_url}
                  alt={selectedCharacter?.name || "Character avatar"}
                  className={"rounded-full " + iconClassName}
                />
              ) : (
                <UserCircle2 className={iconClassName} />
              )}
            </IconButton>
            {selectedCharacter && (
              <button
                type="button"
                onClick={(event) => {
                  event.preventDefault()
                  event.stopPropagation()
                  void applySelection(null)
                }}
                className="absolute -top-2 -right-2 flex h-5 w-5 items-center justify-center rounded-full bg-text text-[10px] font-semibold text-bg shadow-sm hover:bg-text-muted"
                aria-label={clearLabel}
                title={clearLabel}>
                ×
              </button>
            )}
          </div>
        </Tooltip>
      </Dropdown>

      {showLabel && selectedCharacter?.name && (
        <div className="hidden items-center gap-2 rounded-full border border-border bg-surface px-3 py-1 text-xs font-medium text-text shadow-sm sm:inline-flex">
          {selectedCharacter?.avatar_url ? (
            <img
              src={selectedCharacter.avatar_url}
              alt={selectedCharacter.name || "Character avatar"}
              className="h-5 w-5 rounded-full"
            />
          ) : (
            <UserCircle2 className="h-4 w-4" />
          )}
          <span className="max-w-[180px] truncate">{selectedCharacter.name}</span>
        </div>
      )}
    </div>
  )
}

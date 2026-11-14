// @vitest-environment jsdom
import React from "react"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { PlaygroundChat } from "../PlaygroundChat"
import {
  IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE,
  IMAGE_GENERATION_USER_MESSAGE_TYPE
} from "@/utils/image-generation-chat"

const useMessageOptionState = vi.hoisted(() => ({
  value: {
    messages: [] as any[],
    setMessages: vi.fn(),
    streaming: false,
    isProcessing: false,
    regenerateLastMessage: vi.fn(),
    isSearchingInternet: false,
    editMessage: vi.fn(),
    deleteMessage: vi.fn(),
    toggleMessagePinned: vi.fn(),
    ttsEnabled: false,
    onSubmit: vi.fn(),
    actionInfo: null,
    messageSteeringMode: "none",
    setMessageSteeringMode: vi.fn(),
    messageSteeringForceNarrate: false,
    setMessageSteeringForceNarrate: vi.fn(),
    clearMessageSteering: vi.fn(),
    createChatBranch: vi.fn(),
    createCompareBranch: vi.fn(),
    temporaryChat: false,
    serverChatId: "chat-1",
    serverChatCharacterId: null,
    stopStreamingRequest: vi.fn(),
    isEmbedding: false,
    compareMode: false,
    compareFeatureEnabled: false,
    compareSelectionByCluster: {},
    setCompareSelectionForCluster: vi.fn(),
    compareActiveModelsByCluster: {},
    setCompareActiveModelsForCluster: vi.fn(),
    setCompareSelectedModels: vi.fn(),
    historyId: "history-1",
    setSelectedModel: vi.fn(),
    setCompareMode: vi.fn(),
    sendPerModelReply: vi.fn(),
    compareCanonicalByCluster: {},
    setCompareCanonicalForCluster: vi.fn(),
    compareContinuationModeByCluster: {},
    setCompareContinuationModeForCluster: vi.fn(),
    setCompareParentForHistory: vi.fn(),
    compareSplitChats: {},
    setCompareSplitChat: vi.fn(),
    compareMaxModels: 3
  }
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, defaultValue?: string) => defaultValue || key
  })
}))

vi.mock("@tanstack/react-query", async (original) => {
  const actual = await original<typeof import("@tanstack/react-query")>()
  const queryClient = new actual.QueryClient()
  return {
    ...actual,
    useQueryClient: () => queryClient,
    useQuery: () => ({ data: [] })
  }
})

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: () => [false]
}))

vi.mock("@/hooks/useMessageOption", () => ({
  useMessageOption: () => useMessageOptionState.value
}))

vi.mock("@/hooks/useSelectedCharacter", () => ({
  useSelectedCharacter: () => [null]
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({
    success: vi.fn(),
    error: vi.fn(),
    info: vi.fn(),
    warning: vi.fn()
  })
}))

vi.mock("@/components/Common/ChatGreetingPicker", () => ({
  ChatGreetingPicker: () => <div data-testid="chat-greeting-picker" />
}))

vi.mock("./PlaygroundEmpty", () => ({
  PlaygroundEmpty: () => <div data-testid="playground-empty" />
}))

vi.mock("@/components/Common/Playground/Message", () => ({
  PlaygroundMessage: (props: {
    message: string
    messageId?: string
    message_type?: string
    generationInfo?: Record<string, any>
    onSelectImageVariant?: (payload: {
      messageId?: string
      variantIndex: number
    }) => void
    onKeepImageVariant?: (payload: {
      messageId?: string
      variantIndex: number
    }) => void
    onDeleteImageVariant?: (payload: {
      messageId?: string
      variantIndex: number
    }) => void
    onDeleteAllImageVariants?: (payload: { messageId?: string }) => void
    onDeleteImage?: (payload: {
      messageId?: string
      imageIndex: number
      imageUrl: string
    }) => void
    onRegenerateImage?: (payload: {
      messageId?: string
      imageIndex: number
      imageUrl: string
      request: {
        prompt: string
        backend: string
      } | null
    }) => void | Promise<void>
  }) => (
    <div
      data-testid="playground-message-mock"
      data-message-type={props.message_type || ""}
      data-image-source={String(props.generationInfo?.image_generation?.source || "")}
    >
      {props.message}
      <button
        type="button"
        data-testid={`playground-message-select-variant-0-${props.messageId || "unknown"}`}
        onClick={() => {
          props.onSelectImageVariant?.({
            messageId: props.messageId,
            variantIndex: 0
          })
        }}
      >
        Select variant 0
      </button>
      <button
        type="button"
        data-testid={`playground-message-keep-variant-${props.messageId || "unknown"}`}
        onClick={() => {
          props.onKeepImageVariant?.({
            messageId: props.messageId,
            variantIndex: 0
          })
        }}
      >
        Keep variant
      </button>
      <button
        type="button"
        data-testid={`playground-message-delete-variant-1-${props.messageId || "unknown"}`}
        onClick={() => {
          props.onDeleteImageVariant?.({
            messageId: props.messageId,
            variantIndex: 1
          })
        }}
      >
        Delete variant 1
      </button>
      <button
        type="button"
        data-testid={`playground-message-delete-all-variants-${props.messageId || "unknown"}`}
        onClick={() => {
          props.onDeleteAllImageVariants?.({
            messageId: props.messageId
          })
        }}
      >
        Delete all variants
      </button>
      <button
        type="button"
        data-testid={`playground-message-regenerate-image-${props.messageId || "unknown"}`}
        onClick={() => {
          void props.onRegenerateImage?.({
            messageId: props.messageId,
            imageIndex: 0,
            imageUrl: "data:image/png;base64,AAAA",
            request:
              props.generationInfo?.image_generation?.request || null
          })
        }}
      >
        Regenerate image
      </button>
      <button
        type="button"
        data-testid={`playground-message-delete-image-${props.messageId || "unknown"}`}
        onClick={() => {
          props.onDeleteImage?.({
            messageId: props.messageId,
            imageIndex: 0,
            imageUrl: "data:image/png;base64,AAAA"
          })
        }}
      >
        Delete image
      </button>
    </div>
  )
}))

const setImageGenerationTurn = (source: "generate-modal" | "slash-command") => {
  useMessageOptionState.value.messages = [
    {
      id: "image-user",
      role: "user",
      isBot: false,
      name: "You",
      message: source === "generate-modal" ? "sunlit city skyline" : "/generate-image:comfyui sunlit city skyline",
      messageType: IMAGE_GENERATION_USER_MESSAGE_TYPE
    },
    {
      id: "image-assistant",
      role: "assistant",
      isBot: true,
      name: "Image backend",
      message: "",
      images: ["data:image/png;base64,AAAA"],
      messageType: IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE,
      generationInfo: {
        image_generation: {
          request: {
            prompt: "sunlit city skyline",
            backend: "comfyui"
          },
          source
        }
      }
    }
  ]
}

describe("PlaygroundChat image generation event integration", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    useMessageOptionState.value.messages = []
    useMessageOptionState.value.temporaryChat = false
    useMessageOptionState.value.historyId = "history-1"
  })

  it("renders only one assistant event card for modal image generation", () => {
    setImageGenerationTurn("generate-modal")

    render(<PlaygroundChat />)

    const cards = screen.getAllByTestId("playground-message-mock")
    expect(cards).toHaveLength(1)
    expect(cards[0]).toHaveAttribute(
      "data-message-type",
      IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE
    )
    expect(cards[0]).toHaveAttribute("data-image-source", "generate-modal")
  })

  it("applies the same assistant-only event contract for slash image generation", () => {
    setImageGenerationTurn("slash-command")

    render(<PlaygroundChat />)

    const cards = screen.getAllByTestId("playground-message-mock")
    expect(cards).toHaveLength(1)
    expect(cards[0]).toHaveAttribute(
      "data-message-type",
      IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE
    )
    expect(cards[0]).toHaveAttribute("data-image-source", "slash-command")
  })

  it("keeps image event render parity across desktop and mobile widths", () => {
    setImageGenerationTurn("generate-modal")
    Object.defineProperty(window, "innerWidth", {
      configurable: true,
      writable: true,
      value: 1280
    })

    const desktop = render(<PlaygroundChat />)
    const desktopSummary = screen
      .getAllByTestId("playground-message-mock")
      .map((card) => ({
        type: card.getAttribute("data-message-type"),
        source: card.getAttribute("data-image-source")
      }))
    desktop.unmount()

    Object.defineProperty(window, "innerWidth", {
      configurable: true,
      writable: true,
      value: 390
    })
    render(<PlaygroundChat />)
    const mobileSummary = screen
      .getAllByTestId("playground-message-mock")
      .map((card) => ({
        type: card.getAttribute("data-message-type"),
        source: card.getAttribute("data-image-source")
      }))

    expect(mobileSummary).toEqual(desktopSummary)
    expect(mobileSummary).toMatchInlineSnapshot(`
      [
        {
          "source": "generate-modal",
          "type": "image-generation:assistant",
        },
      ]
    `)
  })

  it("routes image regenerate through variant regrouping submit payload", async () => {
    const user = userEvent.setup()
    setImageGenerationTurn("generate-modal")

    render(<PlaygroundChat />)

    await user.click(
      screen.getByTestId("playground-message-regenerate-image-image-assistant")
    )

    expect(useMessageOptionState.value.setMessages).toHaveBeenCalledTimes(1)
    expect(useMessageOptionState.value.onSubmit).toHaveBeenCalledTimes(1)
    const submitPayload = useMessageOptionState.value.onSubmit.mock.calls[0]?.[0]
    expect(submitPayload).toMatchObject({
      message: "sunlit city skyline",
      isRegenerate: true,
      userMessageType: IMAGE_GENERATION_USER_MESSAGE_TYPE,
      assistantMessageType: IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE,
      imageGenerationSource: "message-regen"
    })
    expect(submitPayload.regenerateFromMessage?.id).toBe("image-assistant")
    expect(submitPayload.messages).toHaveLength(1)
    expect(submitPayload.messages[0]?.id).toBe("image-user")
  })

  it("deletes the active image variant and falls back to the previous variant", async () => {
    const user = userEvent.setup()
    useMessageOptionState.value.historyId = "temp"
    useMessageOptionState.value.messages = [
      {
        id: "image-user",
        role: "user",
        isBot: false,
        name: "You",
        message: "sunlit city skyline",
        messageType: IMAGE_GENERATION_USER_MESSAGE_TYPE
      },
      {
        id: "image-assistant",
        role: "assistant",
        isBot: true,
        name: "Image backend",
        message: "",
        images: ["data:image/png;base64,BBBB"],
        messageType: IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE,
        activeVariantIndex: 1,
        variants: [
          {
            id: "variant-1",
            message: "",
            images: ["data:image/png;base64,AAAA"],
            generationInfo: {
              image_generation: {
                request: {
                  prompt: "sunlit city skyline",
                  backend: "comfyui"
                }
              }
            }
          },
          {
            id: "variant-2",
            message: "",
            images: ["data:image/png;base64,BBBB"],
            generationInfo: {
              image_generation: {
                request: {
                  prompt: "sunlit city skyline",
                  backend: "comfyui"
                }
              }
            }
          }
        ],
        generationInfo: {
          image_generation: {
            request: {
              prompt: "sunlit city skyline",
              backend: "comfyui"
            }
          }
        }
      }
    ]

    render(<PlaygroundChat />)

    await user.click(
      screen.getByTestId("playground-message-delete-image-image-assistant")
    )

    expect(useMessageOptionState.value.setMessages).toHaveBeenCalledTimes(1)
    const updater = useMessageOptionState.value.setMessages.mock.calls[0]?.[0]
    expect(typeof updater).toBe("function")

    const nextMessages = updater(useMessageOptionState.value.messages)
    expect(nextMessages).toHaveLength(2)
    const updatedAssistant = nextMessages.find((entry: any) => entry.isBot)
    expect(updatedAssistant).toBeDefined()
    expect(updatedAssistant?.id).toBe("variant-1")
    expect(updatedAssistant?.activeVariantIndex).toBe(0)
    expect(updatedAssistant?.variants).toHaveLength(1)
    expect(updatedAssistant?.images).toEqual(["data:image/png;base64,AAAA"])
    expect(
      updatedAssistant?.generationInfo?.image_generation?.variant_count
    ).toBe(1)
    expect(
      updatedAssistant?.generationInfo?.image_generation?.active_variant_index
    ).toBe(0)
    expect(
      updatedAssistant?.generationInfo?.image_generation?.event_id
    ).toBe("image-assistant")
  })

  it("selects a specific image variant and applies it as active", async () => {
    const user = userEvent.setup()
    useMessageOptionState.value.historyId = "temp"
    useMessageOptionState.value.messages = [
      {
        id: "image-user",
        role: "user",
        isBot: false,
        name: "You",
        message: "sunlit city skyline",
        messageType: IMAGE_GENERATION_USER_MESSAGE_TYPE
      },
      {
        id: "image-assistant",
        role: "assistant",
        isBot: true,
        name: "Image backend",
        message: "",
        images: ["data:image/png;base64,BBBB"],
        messageType: IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE,
        activeVariantIndex: 1,
        variants: [
          {
            id: "variant-1",
            message: "",
            images: ["data:image/png;base64,AAAA"],
            generationInfo: {
              image_generation: {
                request: {
                  prompt: "sunlit city skyline",
                  backend: "comfyui"
                }
              }
            }
          },
          {
            id: "variant-2",
            message: "",
            images: ["data:image/png;base64,BBBB"],
            generationInfo: {
              image_generation: {
                request: {
                  prompt: "sunlit city skyline",
                  backend: "comfyui"
                }
              }
            }
          }
        ],
        generationInfo: {
          image_generation: {
            request: {
              prompt: "sunlit city skyline",
              backend: "comfyui"
            }
          }
        }
      }
    ]

    render(<PlaygroundChat />)

    await user.click(
      screen.getByTestId("playground-message-select-variant-0-image-assistant")
    )

    expect(useMessageOptionState.value.setMessages).toHaveBeenCalledTimes(1)
    const updater = useMessageOptionState.value.setMessages.mock.calls[0]?.[0]
    const nextMessages = updater(useMessageOptionState.value.messages)
    const updatedAssistant = nextMessages.find((entry: any) => entry.isBot)
    expect(updatedAssistant?.activeVariantIndex).toBe(0)
    expect(updatedAssistant?.images).toEqual(["data:image/png;base64,AAAA"])
  })

  it("keeps active image variant by promoting it as the canonical tail variant", async () => {
    const user = userEvent.setup()
    useMessageOptionState.value.historyId = "temp"
    useMessageOptionState.value.messages = [
      {
        id: "image-user",
        role: "user",
        isBot: false,
        name: "You",
        message: "sunlit city skyline",
        messageType: IMAGE_GENERATION_USER_MESSAGE_TYPE
      },
      {
        id: "image-assistant",
        role: "assistant",
        isBot: true,
        name: "Image backend",
        message: "",
        images: ["data:image/png;base64,AAAA"],
        messageType: IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE,
        activeVariantIndex: 0,
        variants: [
          {
            id: "variant-1",
            message: "",
            images: ["data:image/png;base64,AAAA"],
            generationInfo: {
              image_generation: {
                request: {
                  prompt: "sunlit city skyline",
                  backend: "comfyui"
                }
              }
            }
          },
          {
            id: "variant-2",
            message: "",
            images: ["data:image/png;base64,BBBB"],
            generationInfo: {
              image_generation: {
                request: {
                  prompt: "sunlit city skyline",
                  backend: "comfyui"
                }
              }
            }
          }
        ],
        generationInfo: {
          image_generation: {
            request: {
              prompt: "sunlit city skyline",
              backend: "comfyui"
            }
          }
        }
      }
    ]

    render(<PlaygroundChat />)

    await user.click(
      screen.getByTestId("playground-message-keep-variant-image-assistant")
    )

    expect(useMessageOptionState.value.setMessages).toHaveBeenCalledTimes(1)
    const updater = useMessageOptionState.value.setMessages.mock.calls[0]?.[0]
    const nextMessages = updater(useMessageOptionState.value.messages)
    const updatedAssistant = nextMessages.find((entry: any) => entry.isBot)
    expect(updatedAssistant?.activeVariantIndex).toBe(1)
    expect(updatedAssistant?.variants?.[1]?.id).toBe("variant-1")
    expect(updatedAssistant?.images).toEqual(["data:image/png;base64,AAAA"])
    expect(
      updatedAssistant?.generationInfo?.image_generation?.variant_count
    ).toBe(2)
    expect(
      updatedAssistant?.generationInfo?.image_generation?.active_variant_index
    ).toBe(1)
    expect(
      updatedAssistant?.generationInfo?.image_generation?.event_id
    ).toBe("image-assistant")
  })

  it("clears grouped image variants when delete-all is requested", async () => {
    const user = userEvent.setup()
    useMessageOptionState.value.historyId = "temp"
    useMessageOptionState.value.messages = [
      {
        id: "image-user",
        role: "user",
        isBot: false,
        name: "You",
        message: "sunlit city skyline",
        messageType: IMAGE_GENERATION_USER_MESSAGE_TYPE
      },
      {
        id: "image-assistant",
        role: "assistant",
        isBot: true,
        name: "Image backend",
        message: "",
        images: ["data:image/png;base64,AAAA"],
        messageType: IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE,
        activeVariantIndex: 0,
        variants: [
          {
            id: "variant-1",
            message: "",
            images: ["data:image/png;base64,AAAA"]
          },
          {
            id: "variant-2",
            message: "",
            images: ["data:image/png;base64,BBBB"]
          }
        ]
      }
    ]

    render(<PlaygroundChat />)

    await user.click(
      screen.getByTestId("playground-message-delete-all-variants-image-assistant")
    )

    expect(useMessageOptionState.value.setMessages).toHaveBeenCalledTimes(1)
    const updater = useMessageOptionState.value.setMessages.mock.calls[0]?.[0]
    const nextMessages = updater(useMessageOptionState.value.messages)
    const updatedAssistant = nextMessages.find((entry: any) => entry.id === "image-assistant")
    expect(updatedAssistant?.images).toEqual([])
    expect(updatedAssistant?.variants).toEqual([])
    expect(updatedAssistant?.activeVariantIndex).toBe(0)
  })
})

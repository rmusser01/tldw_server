// @vitest-environment jsdom
import fs from "node:fs"
import path from "node:path"
import React from "react"
import { render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { PlaygroundMessage } from "../Message"

const storageState = vi.hoisted(() => ({
  values: new Map<string, unknown>()
}))
const parseReasoningMock = vi.hoisted(() =>
  vi.fn((content: string) => [{ type: "message", content }])
)
const highlightSpy = vi.hoisted(() =>
  vi.fn((text: string) => text)
)
// Per-row render counts for the 50-row memo test, keyed by the row marker we
// smuggle through `generationInfo` (MessageActionsBar receives it verbatim).
const actionBarRenderCounts = vi.hoisted(() => new Map<number, number>())

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback || _key,
    i18n: { language: "en" }
  })
}))

vi.mock("antd", () => {
  const ModalMock = ({
    open,
    children
  }: {
    open?: boolean
    children?: React.ReactNode
  }) => (open ? <div>{children}</div> : null)
  const messageApi = {
    success: vi.fn(),
    error: vi.fn(),
    warning: vi.fn(),
    info: vi.fn()
  }

  return {
    App: {
      useApp: () => ({ message: messageApi })
    },
    Tag: ({ children }: { children: React.ReactNode }) => <span>{children}</span>,
    Image: ({ src, alt }: { src?: string; alt?: string }) => (
      <img src={src || ""} alt={alt || ""} />
    ),
    Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>,
    Collapse: ({
      children,
      items
    }: {
      children?: React.ReactNode
      items?: Array<{ children?: React.ReactNode; label?: React.ReactNode }>
    }) => (
      <div>
        {items?.map((item, index) => (
          <section key={index}>
            <div>{item.label}</div>
            {item.children}
          </section>
        )) ?? children}
      </div>
    ),
    Avatar: ({ src, alt }: { src?: string; alt?: string }) => (
      <img src={src || ""} alt={alt || ""} />
    ),
    Modal: Object.assign(ModalMock, {
      confirm: vi.fn()
    }),
    message: messageApi
  }
})

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (key: string, defaultValue: unknown) => [
    storageState.values.has(key) ? storageState.values.get(key) : defaultValue,
    vi.fn()
  ]
}))

vi.mock("@/components/Common/Markdown", () => ({
  default: (props: Record<string, unknown>) => (
    <div data-testid="mock-markdown">{String(props.message ?? "")}</div>
  )
}))

vi.mock("../ActionInfo", () => ({
  LoadingStatus: () => null
}))

vi.mock("../EditMessageForm", () => ({
  EditMessageForm: () => <div data-testid="edit-form" />
}))

vi.mock("../PlaygroundUserMessage", () => ({
  PlaygroundUserMessageBubble: ({ message }: { message: string }) => (
    <div>{message}</div>
  )
}))

vi.mock("@/components/Sidepanel/Chat/FeedbackModal", () => ({
  FeedbackModal: () => null
}))

vi.mock("@/components/Sidepanel/Chat/SourceFeedback", () => ({
  SourceFeedback: () => null
}))

vi.mock("@/components/Sidepanel/Chat/ToolCallBlock", () => ({
  ToolCallBlock: () => null
}))

// Render spy: MessageActionsBar is rendered synchronously for every
// non-editing row, so counting its invocations counts committed row renders.
vi.mock("../MessageActionsBar", () => ({
  MessageActionsBar: ({ generationInfo }: { generationInfo?: any }) => {
    const rowId = generationInfo?.rowId
    if (typeof rowId === "number") {
      actionBarRenderCounts.set(
        rowId,
        (actionBarRenderCounts.get(rowId) ?? 0) + 1
      )
    }
    return null
  }
}))

vi.mock("../DiscoSkillAnnotation", () => ({
  DiscoSkillAnnotation: () => null
}))

vi.mock("@/hooks/useTTS", () => ({
  useTTS: () => ({
    cancel: vi.fn(),
    isSpeaking: false,
    speak: vi.fn()
  })
}))

vi.mock("@/hooks/useChatMoodBadgePreference", () => ({
  useChatMoodBadgePreference: () => [false, vi.fn()]
}))

vi.mock("@/hooks/useFeedback", () => ({
  useFeedback: () => ({
    thumb: null,
    detail: "",
    sourceFeedback: {},
    canSubmit: false,
    isSubmitting: false,
    showThanks: false,
    submitThumb: vi.fn(),
    submitDetail: vi.fn(),
    submitSourceThumb: vi.fn()
  })
}))

vi.mock("@/hooks/useImplicitFeedback", () => ({
  useImplicitFeedback: () => ({
    trackCopy: vi.fn(),
    trackSourcesExpanded: vi.fn(),
    trackSourceClick: vi.fn(),
    trackCitationUsed: vi.fn(),
    trackDwellTime: vi.fn()
  })
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    capabilities: {
      hasFeedbackExplicit: false,
      hasFeedbackImplicit: false
    }
  })
}))

vi.mock("@/hooks/useTldwAudioStatus", () => ({
  useTldwAudioStatus: () => ({
    healthState: "ready",
    voicesAvailable: true
  })
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: vi.fn(async () => undefined),
    saveChatKnowledge: vi.fn(async () => undefined),
    createChatCompletion: vi.fn(async () => ({}))
  }
}))

vi.mock("@/store/ui-mode", () => ({
  useUiModeStore: (selector: (state: { mode: string }) => unknown) =>
    selector({ mode: "pro" })
}))

vi.mock("@/store/option", () => ({
  useStoreMessageOption: (selector: (state: any) => unknown) =>
    selector({
      setReplyTarget: vi.fn(),
      ragPinnedResults: [],
      setMessages: vi.fn()
    })
}))

vi.mock("@/store/model", () => ({
  useStoreChatModelSettings: (selector: (state: any) => unknown) =>
    selector({
      apiProvider: "openai",
      updateSetting: vi.fn()
    })
}))

vi.mock("@/store", () => ({
  useStoreMessage: (selector: (state: any) => unknown) =>
    selector({
      selectedModel: "gpt-4o-mini"
    })
}))

vi.mock("@/hooks/useDiscoSkills", () => ({
  useDiscoSkills: () => ({
    enabled: false,
    stats: null,
    triggerProbabilityBase: 0,
    persistComments: false
  })
}))

vi.mock("@/libs/reasoning", async importOriginal => ({
  ...await importOriginal<typeof import("@/libs/reasoning")>(),
  parseReasoning: parseReasoningMock
}))

vi.mock("@/utils/chat-error-message", () => ({
  decodeChatErrorPayload: vi.fn(() => null)
}))

vi.mock("@/utils/feedback", () => ({
  getSourceFeedbackKey: () => "feedback-key"
}))

vi.mock("@/utils/clipboard", () => ({
  copyToClipboard: vi.fn(async () => undefined)
}))

vi.mock("@/utils/chat-style", () => ({
  buildChatTextClass: () => ""
}))

vi.mock("@/utils/text-highlight", () => ({
  highlightText: highlightSpy
}))

vi.mock("@/db/dexie/models", () => ({
  removeModelSuffix: (value: string) => value
}))

vi.mock("@/utils/color", () => ({
  tagColors: {}
}))

vi.mock("@/utils/disco-skill-check", () => ({
  attemptSkillTrigger: vi.fn(),
  buildSkillPrompt: vi.fn(() => ""),
  createSkillComment: vi.fn(() => null)
}))

vi.mock("@/utils/character-mood", () => ({
  detectCharacterMood: () => ({ label: null }),
  normalizeCharacterMoodLabel: (value: unknown) => value,
  resolveCharacterBaseAvatarUrl: () => "",
  resolveCharacterMoodImageUrl: () => ""
}))

vi.mock("@/db/dexie/helpers", () => ({
  updateMessageDiscoSkillComment: vi.fn(async () => undefined)
}))

vi.mock("../message-layout", () => ({
  resolveAvatarColumnAlignment: () => "",
  resolveMessageRenderSide: () => "left"
}))

vi.mock("../playground-message-shortcuts", () => ({
  resolvePlaygroundMessageShortcutAction: () => null
}))

vi.mock("../quick-message-actions", () => ({
  buildQuickMessageActionPrompt: vi.fn(() => "prompt")
}))

// Stable row props mirroring the parents' discipline (PlaygroundChat.tsx and
// Sidepanel/Chat/body.tsx): stable handler identities, module-level empty
// array, referentially-stable per-row data except the streaming row.
const STABLE_ON_REGENERATE = vi.fn()
const STABLE_ON_EDIT_SUBMIT = vi.fn()
const EMPTY_IMAGES: string[] = []

type RowProps = React.ComponentProps<typeof PlaygroundMessage>

const buildRowProps = (
  index: number,
  total: number,
  overrides: Partial<RowProps> = {}
): RowProps => ({
  message: `Row ${index} completed content`,
  isBot: true,
  role: "assistant",
  name: "Assistant",
  currentMessageIndex: index,
  totalMessages: total,
  onRegenerate: STABLE_ON_REGENERATE,
  onEditFormSubmit: STABLE_ON_EDIT_SUBMIT,
  isProcessing: false,
  isStreaming: false,
  conversationInstanceId: "conversation-1",
  images: EMPTY_IMAGES,
  generationInfo: { rowId: index },
  ...overrides
})

type RowBundle = { id: string; props: RowProps }

const buildRows = (count: number): RowBundle[] =>
  Array.from({ length: count }, (_unused, index) => ({
    id: `row-${index}`,
    props: buildRowProps(index, count)
  }))

const TranscriptHarness = ({ rows }: { rows: RowBundle[] }) => (
  <div>
    {rows.map((row) => (
      <PlaygroundMessage key={row.id} {...row.props} />
    ))}
  </div>
)

const countRowsRendered = () =>
  Array.from(actionBarRenderCounts.values()).filter((count) => count > 0).length

const totalRowRenders = () =>
  Array.from(actionBarRenderCounts.values()).reduce(
    (sum, count) => sum + count,
    0
  )

describe("PlaygroundMessage memoization (WP-02/WP-03/WP-09)", () => {
  beforeEach(() => {
    storageState.values.clear()
    actionBarRenderCounts.clear()
    parseReasoningMock.mockImplementation((content: string) => [
      { type: "message", content }
    ])
  })

  it("a token flush re-renders only the streaming row, not the other 49", () => {
    const initialRows = buildRows(50)

    const { rerender } = render(<TranscriptHarness rows={initialRows} />)

    expect(screen.getAllByTestId("chat-message")).toHaveLength(50)
    expect(countRowsRendered()).toBe(50)
    expect(totalRowRenders()).toBe(50)

    actionBarRenderCounts.clear()

    // Token flush (Stage 1 semantics): a new messages array where only the
    // last row's entry is replaced; every other row keeps prop identity.
    const flushedRows = initialRows.map((row, index) =>
      index === initialRows.length - 1
        ? {
            ...row,
            props: {
              ...row.props,
              message: `${row.props.message} + streamed token`,
              isStreaming: true
            }
          }
        : row
    )
    rerender(<TranscriptHarness rows={flushedRows} />)

    // Guard against over-memoizing: the last row DID re-render...
    expect(actionBarRenderCounts.get(49)).toBe(1)
    // ...and no sibling row re-rendered.
    for (let index = 0; index < 49; index += 1) {
      expect(actionBarRenderCounts.get(index) ?? 0).toBe(0)
    }
    expect(totalRowRenders()).toBe(1)
  })

  it("memoizes search highlighting: unrelated re-renders do not re-run highlightText", () => {
    const { rerender } = render(
      <PlaygroundMessage
        {...buildRowProps(0, 1, {
          isBot: false,
          role: "user",
          name: "User",
          message: "lorem ipsum dolor sit amet"
        })}
        searchQuery="ipsum"
      />
    )

    expect(highlightSpy).toHaveBeenCalledTimes(1)
    expect(highlightSpy).toHaveBeenCalledWith("lorem ipsum dolor sit amet", "ipsum")

    // Unrelated prop change (search match ring): the row re-renders but the
    // highlight computation must not re-run for unchanged message + query.
    rerender(
      <PlaygroundMessage
        {...buildRowProps(0, 1, {
          isBot: false,
          role: "user",
          name: "User",
          message: "lorem ipsum dolor sit amet"
        })}
        searchQuery="ipsum"
        searchMatch="match"
      />
    )

    // Prove the row itself re-rendered (the memo boundary is the row, not
    // this prop), otherwise the assertion below would be vacuous.
    expect(screen.getByTestId("chat-message")).toHaveAttribute(
      "data-search-match",
      "match"
    )
    expect(highlightSpy).toHaveBeenCalledTimes(1)

    // Changing the query recomputes the highlight for the affected row.
    rerender(
      <PlaygroundMessage
        {...buildRowProps(0, 1, {
          isBot: false,
          role: "user",
          name: "User",
          message: "lorem ipsum dolor sit amet"
        })}
        searchQuery="dolor"
        searchMatch="match"
      />
    )
    expect(highlightSpy).toHaveBeenCalledTimes(2)
    expect(highlightSpy).toHaveBeenLastCalledWith(
      "lorem ipsum dolor sit amet",
      "dolor"
    )
  })
})

describe("PlaygroundMessage row-prop contract (source guard)", () => {
  const readSource = (relativePath: string) =>
    fs.readFileSync(path.resolve(__dirname, relativePath), "utf8")

  it("exports PlaygroundMessage through React.memo", () => {
    const source = readSource("../Message.tsx")
    expect(source).toContain("export const PlaygroundMessage = React.memo(")
  })

  it("both transcript parents pass a stable empty images constant", () => {
    const chatSource = readSource("../../../Option/Playground/PlaygroundChat.tsx")
    const bodySource = readSource("../../../Sidepanel/Chat/body.tsx")

    expect(chatSource).toContain("images={message.images || EMPTY_IMAGES}")
    expect(bodySource).toContain("images={message.images || EMPTY_IMAGES}")
  })

  it("both transcript parents narrow isStreaming to the last row only", () => {
    const chatSource = readSource("../../../Option/Playground/PlaygroundChat.tsx")
    const bodySource = readSource("../../../Sidepanel/Chat/body.tsx")

    // ChatPane formulation (ResearchWorkspace) copied verbatim per plan.
    expect(chatSource).toContain(
      "isStreaming={streaming && block.index === messages.length - 1}"
    )
    expect(bodySource).toContain(
      "isStreaming={streaming && index === messages.length - 1}"
    )
  })
})

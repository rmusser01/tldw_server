import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  fetchWithAuth: vi.fn(),
  resolvedDefaults: {
    sttLanguage: "en-US",
    sttModel: "parakeet",
    ttsProvider: "tldw",
    ttsVoice: "af_heart",
    confirmationMode: "destructive_only" as const,
    voiceChatTriggerPhrases: ["hey helper"],
    wakeBehavior: "one_shot" as const,
    autoResume: true,
    bargeIn: false,
    autoCommitEnabled: true,
    vadThreshold: 0.5,
    minSilenceMs: 250,
    turnStopSecs: 0.2,
    minUtteranceSecs: 0.4
  }
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string
          }
    ) => {
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      if (defaultValueOrOptions?.defaultValue) return defaultValueOrOptions.defaultValue
      return key
    }
  })
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    fetchWithAuth: (...args: unknown[]) =>
      (mocks.fetchWithAuth as (...args: unknown[]) => unknown)(...args)
  }
}))

vi.mock("@/hooks/useResolvedPersonaVoiceDefaults", () => ({
  PERSONA_TURN_DETECTION_BALANCED_DEFAULTS: {
    autoCommitEnabled: true,
    vadThreshold: 0.5,
    minSilenceMs: 250,
    turnStopSecs: 0.2,
    minUtteranceSecs: 0.4
  },
  useResolvedPersonaVoiceDefaults: (voiceDefaults?: Record<string, unknown> | null) => ({
    ...mocks.resolvedDefaults,
    sttLanguage:
      typeof voiceDefaults?.stt_language === "string"
        ? voiceDefaults.stt_language
        : mocks.resolvedDefaults.sttLanguage,
    sttModel:
      typeof voiceDefaults?.stt_model === "string"
        ? voiceDefaults.stt_model
        : mocks.resolvedDefaults.sttModel,
    ttsProvider:
      typeof voiceDefaults?.tts_provider === "string"
        ? voiceDefaults.tts_provider
        : mocks.resolvedDefaults.ttsProvider,
    ttsVoice:
      typeof voiceDefaults?.tts_voice === "string"
        ? voiceDefaults.tts_voice
        : mocks.resolvedDefaults.ttsVoice,
    confirmationMode:
      typeof voiceDefaults?.confirmation_mode === "string"
        ? (voiceDefaults.confirmation_mode as
            | "always"
            | "destructive_only"
            | "never")
        : mocks.resolvedDefaults.confirmationMode,
    voiceChatTriggerPhrases: Array.isArray(voiceDefaults?.voice_chat_trigger_phrases)
      ? voiceDefaults.voice_chat_trigger_phrases.map((phrase) => String(phrase))
      : mocks.resolvedDefaults.voiceChatTriggerPhrases,
    wakeBehavior:
      voiceDefaults?.wake_behavior === "continuous" ||
      voiceDefaults?.wake_behavior === "push_to_talk_after_wake" ||
      voiceDefaults?.wake_behavior === "one_shot"
        ? voiceDefaults.wake_behavior
        : mocks.resolvedDefaults.wakeBehavior,
    autoResume:
      typeof voiceDefaults?.auto_resume === "boolean"
        ? voiceDefaults.auto_resume
        : mocks.resolvedDefaults.autoResume,
    bargeIn:
      typeof voiceDefaults?.barge_in === "boolean"
        ? voiceDefaults.barge_in
        : mocks.resolvedDefaults.bargeIn,
    autoCommitEnabled:
      typeof voiceDefaults?.auto_commit_enabled === "boolean"
        ? voiceDefaults.auto_commit_enabled
        : mocks.resolvedDefaults.autoCommitEnabled,
    vadThreshold:
      typeof voiceDefaults?.vad_threshold === "number"
        ? voiceDefaults.vad_threshold
        : mocks.resolvedDefaults.vadThreshold,
    minSilenceMs:
      typeof voiceDefaults?.min_silence_ms === "number"
        ? voiceDefaults.min_silence_ms
        : mocks.resolvedDefaults.minSilenceMs,
    turnStopSecs:
      typeof voiceDefaults?.turn_stop_secs === "number"
        ? voiceDefaults.turn_stop_secs
        : mocks.resolvedDefaults.turnStopSecs,
    minUtteranceSecs:
      typeof voiceDefaults?.min_utterance_secs === "number"
        ? voiceDefaults.min_utterance_secs
        : mocks.resolvedDefaults.minUtteranceSecs
  })
}))

import { AssistantDefaultsPanel } from "../AssistantDefaultsPanel"

type MockRecentLiveSession = {
  session_id: string
  started_at: string
  ended_at: string
  auto_commit_enabled: boolean
  vad_threshold: number
  min_silence_ms: number
  turn_stop_secs: number
  min_utterance_secs: number
  turn_detection_changed_during_session: boolean
  total_committed_turns: number
  vad_auto_commit_count: number
  manual_commit_count: number
  manual_mode_required_count: number
  text_only_tts_count: number
  listening_recovery_count: number
  thinking_recovery_count: number
}

const buildRecentLiveSession = (
  overrides: Partial<MockRecentLiveSession> = {}
): MockRecentLiveSession => ({
  session_id: "sess-1",
  started_at: "2026-03-13T17:00:00Z",
  ended_at: "2026-03-13T17:05:00Z",
  auto_commit_enabled: true,
  vad_threshold: 0.5,
  min_silence_ms: 250,
  turn_stop_secs: 0.2,
  min_utterance_secs: 0.4,
  turn_detection_changed_during_session: false,
  total_committed_turns: 4,
  vad_auto_commit_count: 4,
  manual_commit_count: 0,
  manual_mode_required_count: 0,
  text_only_tts_count: 0,
  listening_recovery_count: 0,
  thinking_recovery_count: 0,
  ...overrides
})

const buildVoiceAnalytics = (recentSessions: MockRecentLiveSession[]) => ({
  persona_id: "persona-1",
  summary: {
    total_events: 0,
    direct_command_count: 0,
    planner_fallback_count: 0,
    success_rate: 0,
    fallback_rate: 0,
    avg_response_time_ms: 0
  },
  live_voice: {
    total_committed_turns: recentSessions.reduce(
      (total, session) => total + session.total_committed_turns,
      0
    ),
    vad_auto_commit_count: recentSessions.reduce(
      (total, session) => total + session.vad_auto_commit_count,
      0
    ),
    manual_commit_count: recentSessions.reduce(
      (total, session) => total + session.manual_commit_count,
      0
    ),
    vad_auto_rate: 0,
    manual_commit_rate: 0,
    degraded_session_count: recentSessions.filter(
      (session) => session.manual_mode_required_count > 0
    ).length
  },
  commands: [],
  fallbacks: {
    total_invocations: 0,
    success_count: 0,
    error_count: 0,
    avg_response_time_ms: 0,
    last_used: null
  },
  recent_live_sessions: recentSessions
})

describe("AssistantDefaultsPanel", () => {
  beforeEach(() => {
    Object.defineProperty(HTMLElement.prototype, "scrollIntoView", {
      configurable: true,
      value: vi.fn()
    })
    mocks.fetchWithAuth.mockReset()
    mocks.resolvedDefaults = {
      sttLanguage: "en-US",
      sttModel: "parakeet",
      ttsProvider: "tldw",
      ttsVoice: "af_heart",
      confirmationMode: "destructive_only",
      voiceChatTriggerPhrases: ["hey helper"],
      wakeBehavior: "one_shot",
      autoResume: true,
      bargeIn: false,
      autoCommitEnabled: true,
      vadThreshold: 0.5,
      minSilenceMs: 250,
      turnStopSecs: 0.2,
      minUtteranceSecs: 0.4
    }
    mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string; body?: any }) => {
      if (
        path === "/api/v1/persona/profiles/persona-1" &&
        String(init?.method || "GET").toUpperCase() === "GET"
      ) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "persona-1",
            voice_defaults: {
              stt_language: "en-US",
              stt_model: "whisper-1",
              tts_provider: "tldw",
              tts_voice: "af_heart",
              confirmation_mode: "destructive_only",
              voice_chat_trigger_phrases: ["hey helper"],
              wake_behavior: "continuous",
              auto_resume: true,
              barge_in: false,
              auto_commit_enabled: true,
              vad_threshold: 0.35,
              min_silence_ms: 150,
              turn_stop_secs: 0.1,
              min_utterance_secs: 0.25
            }
          })
        })
      }
      if (
        path === "/api/v1/persona/profiles/persona-1" &&
        String(init?.method || "").toUpperCase() === "PATCH"
      ) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "persona-1",
            voice_defaults: init?.body?.voice_defaults
          })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })
  })

  it("loads persona defaults, explains fallback behavior, and saves edits", async () => {
    const onSaved = vi.fn()
    render(
      <AssistantDefaultsPanel
        selectedPersonaId="persona-1"
        selectedPersonaName="Helper"
        isActive
        onSaved={onSaved}
      />
    )

    expect(
      screen.getByText(
        "Persona defaults stay separate from browser-wide fallback settings. The preview below shows the effective values after local fallback is applied."
      )
    ).toBeInTheDocument()

    await waitFor(() => {
      expect(screen.getByLabelText("STT language")).toHaveValue("en-US")
      expect(screen.getByLabelText("STT model")).toHaveValue("whisper-1")
      expect(screen.getByLabelText("Wake behavior")).toHaveValue("continuous")
      expect(screen.getByLabelText("Wake behavior").closest("label")).toHaveAttribute(
        "for",
        "persona-assistant-defaults-wake-behavior"
      )
    })
    expect(screen.getByText("Turn detection defaults")).toBeInTheDocument()
    expect(screen.getByTestId("assistant-defaults-vad-auto-commit")).toBeChecked()

    fireEvent.click(screen.getByTestId("assistant-defaults-vad-advanced-toggle"))
    expect(screen.getByTestId("assistant-defaults-vad-threshold")).toHaveDisplayValue("0.35")
    expect(screen.getByTestId("assistant-defaults-vad-min-silence-ms")).toHaveDisplayValue(
      "150"
    )
    expect(screen.getByTestId("assistant-defaults-vad-turn-stop-secs")).toHaveDisplayValue(
      "0.1"
    )
    expect(screen.getByTestId("assistant-defaults-vad-min-utterance-secs")).toHaveDisplayValue(
      "0.25"
    )

    fireEvent.change(screen.getByLabelText("STT language"), {
      target: { value: "fr-FR" }
    })
    fireEvent.change(screen.getByLabelText("TTS provider"), {
      target: { value: "openai" }
    })
    fireEvent.change(screen.getByLabelText("TTS voice"), {
      target: { value: "nova" }
    })
    fireEvent.change(screen.getByLabelText("Trigger phrases"), {
      target: { value: "bonjour helper\nhey garden, start research" }
    })
    fireEvent.change(screen.getByLabelText("Wake behavior"), {
      target: { value: "push_to_talk_after_wake" }
    })
    fireEvent.change(screen.getByLabelText("Auto-resume"), {
      target: { value: "false" }
    })
    fireEvent.change(screen.getByLabelText("Barge-in"), {
      target: { value: "true" }
    })
    fireEvent.click(screen.getByTestId("assistant-defaults-vad-auto-commit"))
    fireEvent.change(screen.getByTestId("assistant-defaults-vad-threshold"), {
      target: { value: "0.61" }
    })
    fireEvent.blur(screen.getByTestId("assistant-defaults-vad-threshold"))
    fireEvent.change(screen.getByTestId("assistant-defaults-vad-min-silence-ms"), {
      target: { value: "640" }
    })
    fireEvent.blur(screen.getByTestId("assistant-defaults-vad-min-silence-ms"))
    fireEvent.change(screen.getByTestId("assistant-defaults-vad-turn-stop-secs"), {
      target: { value: "0.48" }
    })
    fireEvent.blur(screen.getByTestId("assistant-defaults-vad-turn-stop-secs"))
    fireEvent.change(screen.getByTestId("assistant-defaults-vad-min-utterance-secs"), {
      target: { value: "0.82" }
    })
    fireEvent.blur(screen.getByTestId("assistant-defaults-vad-min-utterance-secs"))
    fireEvent.click(screen.getByRole("button", { name: "Save assistant defaults" }))

    await waitFor(() => {
      expect(mocks.fetchWithAuth).toHaveBeenCalledWith(
        "/api/v1/persona/profiles/persona-1",
        {
          method: "PATCH",
          body: {
            voice_defaults: {
              stt_language: "fr-FR",
              stt_model: "whisper-1",
              tts_provider: "openai",
              tts_voice: "nova",
              confirmation_mode: "destructive_only",
              voice_chat_trigger_phrases: [
                "bonjour helper",
                "hey garden, start research"
              ],
              wake_behavior: "push_to_talk_after_wake",
              auto_resume: false,
              barge_in: true,
              auto_commit_enabled: false,
              vad_threshold: 0.61,
              min_silence_ms: 640,
              turn_stop_secs: 0.48,
              min_utterance_secs: 0.82
            }
          }
        }
      )
    })

    expect(screen.getByText("Assistant defaults saved.")).toBeInTheDocument()
    expect(screen.getByText("Effective Preview")).toBeInTheDocument()
    expect(screen.getByText("whisper-1")).toBeInTheDocument()
    expect(screen.getByText("bonjour helper, hey garden, start research")).toBeInTheDocument()
    expect(onSaved).toHaveBeenCalledWith(
      expect.objectContaining({
        stt_language: "fr-FR"
      })
    )
  })

  it("clears the previous persona defaults before a new persona load resolves", async () => {
    let personaTwoResolve: ((value: unknown) => void) | null = null
    mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string }) => {
      if (
        path === "/api/v1/persona/profiles/persona-1" &&
        String(init?.method || "GET").toUpperCase() === "GET"
      ) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "persona-1",
            voice_defaults: {
              stt_language: "en-US",
              stt_model: "whisper-1"
            }
          })
        })
      }
      if (
        path === "/api/v1/persona/profiles/persona-2" &&
        String(init?.method || "GET").toUpperCase() === "GET"
      ) {
        return new Promise((resolve) => {
          personaTwoResolve = resolve
        })
      }
      return Promise.resolve({
        ok: true,
        json: async () => ({})
      })
    })

    const view = render(
      <AssistantDefaultsPanel
        selectedPersonaId="persona-1"
        selectedPersonaName="Helper"
        isActive
      />
    )

    await waitFor(() => {
      expect(screen.getByLabelText("STT language")).toHaveValue("en-US")
    })

    view.rerender(
      <AssistantDefaultsPanel
        selectedPersonaId="persona-2"
        selectedPersonaName="Scout"
        isActive
      />
    )

    await waitFor(() => {
      expect(screen.getByLabelText("STT language")).toHaveValue("")
    })

    personaTwoResolve?.({
      ok: true,
      json: async () => ({
        id: "persona-2",
        voice_defaults: {
          stt_language: "fr-FR",
          stt_model: "nova"
        }
      })
    })

    await waitFor(() => {
      expect(screen.getByLabelText("STT language")).toHaveValue("fr-FR")
    })
  })

  it("clears stale defaults when loading the next persona fails", async () => {
    mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string }) => {
      if (
        path === "/api/v1/persona/profiles/persona-1" &&
        String(init?.method || "GET").toUpperCase() === "GET"
      ) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "persona-1",
            voice_defaults: {
              stt_language: "en-US",
              stt_model: "whisper-1"
            }
          })
        })
      }
      if (
        path === "/api/v1/persona/profiles/persona-2" &&
        String(init?.method || "GET").toUpperCase() === "GET"
      ) {
        return Promise.resolve({
          ok: false,
          error: "Persona 2 unavailable",
          json: async () => ({})
        })
      }
      return Promise.resolve({
        ok: true,
        json: async () => ({})
      })
    })

    const view = render(
      <AssistantDefaultsPanel
        selectedPersonaId="persona-1"
        selectedPersonaName="Helper"
        isActive
      />
    )

    await waitFor(() => {
      expect(screen.getByLabelText("STT language")).toHaveValue("en-US")
    })

    view.rerender(
      <AssistantDefaultsPanel
        selectedPersonaId="persona-2"
        selectedPersonaName="Scout"
        isActive
      />
    )

    await waitFor(() => {
      expect(screen.getByText("Persona 2 unavailable")).toBeInTheDocument()
    })
    expect(screen.getByLabelText("STT language")).toHaveValue("")
  })

  it("shows custom as the saved preset when advanced turn detection values diverge", async () => {
    mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string; body?: any }) => {
      if (
        path === "/api/v1/persona/profiles/persona-1" &&
        String(init?.method || "GET").toUpperCase() === "GET"
      ) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "persona-1",
            voice_defaults: {
              stt_language: "en-US",
              stt_model: "whisper-1",
              tts_provider: "tldw",
              tts_voice: "af_heart",
              confirmation_mode: "destructive_only",
              voice_chat_trigger_phrases: ["hey helper"],
              auto_resume: true,
              barge_in: false,
              auto_commit_enabled: false,
              vad_threshold: 0.61,
              min_silence_ms: 640,
              turn_stop_secs: 0.48,
              min_utterance_secs: 0.82
            }
          })
        })
      }
      return Promise.resolve({
        ok: true,
        json: async () => ({
          id: "persona-1",
          voice_defaults: init?.body?.voice_defaults
        })
      })
    })

    render(
      <AssistantDefaultsPanel
        selectedPersonaId="persona-1"
        selectedPersonaName="Helper"
        isActive
      />
    )

    await waitFor(() => {
      expect(
        screen.getByTestId("assistant-defaults-vad-preset-custom")
      ).toHaveAttribute("data-active", "true")
    })
  })

  it("shows no tuning suggestion yet when recent eligible data is sparse", async () => {
    render(
      <AssistantDefaultsPanel
        selectedPersonaId="persona-1"
        selectedPersonaName="Helper"
        isActive
        analytics={buildVoiceAnalytics([
          buildRecentLiveSession({ session_id: "sess-sparse-1" }),
          buildRecentLiveSession({ session_id: "sess-sparse-2" })
        ])}
      />
    )

    await waitFor(() => {
      expect(screen.getByText("Recent live tuning feedback")).toBeInTheDocument()
    })

    expect(screen.getByText("Current signal")).toBeInTheDocument()
    expect(screen.getByText("No tuning suggestion yet")).toBeInTheDocument()
    expect(
      screen.getByText("Run a few live sessions to unlock guidance.")
    ).toBeInTheDocument()
  })

  it("shows a healthy-state suggestion when recent sessions look stable", async () => {
    render(
      <AssistantDefaultsPanel
        selectedPersonaId="persona-1"
        selectedPersonaName="Helper"
        isActive
        analytics={buildVoiceAnalytics([
          buildRecentLiveSession({ session_id: "sess-healthy-1" }),
          buildRecentLiveSession({ session_id: "sess-healthy-2" }),
          buildRecentLiveSession({ session_id: "sess-healthy-3" })
        ])}
      />
    )

    await waitFor(() => {
      expect(
        screen.getByText("Suggestion: current settings look healthy")
      ).toBeInTheDocument()
    })
  })

  it("suggests trying Fast when manual sends stay high across eligible sessions", async () => {
    render(
      <AssistantDefaultsPanel
        selectedPersonaId="persona-1"
        selectedPersonaName="Helper"
        isActive
        analytics={buildVoiceAnalytics([
          buildRecentLiveSession({
            session_id: "sess-fast-1",
            total_committed_turns: 4,
            vad_auto_commit_count: 2,
            manual_commit_count: 2
          }),
          buildRecentLiveSession({
            session_id: "sess-fast-2",
            total_committed_turns: 4,
            vad_auto_commit_count: 2,
            manual_commit_count: 2
          }),
          buildRecentLiveSession({
            session_id: "sess-fast-3",
            total_committed_turns: 4,
            vad_auto_commit_count: 2,
            manual_commit_count: 2
          })
        ])}
      />
    )

    await waitFor(() => {
      expect(
        screen.getByText("Suggestion: try Fast for quicker commits")
      ).toBeInTheDocument()
    })
  })

  it("suggests checking auto-commit availability before changing thresholds", async () => {
    render(
      <AssistantDefaultsPanel
        selectedPersonaId="persona-1"
        selectedPersonaName="Helper"
        isActive
        analytics={buildVoiceAnalytics([
          buildRecentLiveSession({
            session_id: "sess-manual-mode-1",
            total_committed_turns: 4,
            vad_auto_commit_count: 1,
            manual_commit_count: 3,
            manual_mode_required_count: 1
          }),
          buildRecentLiveSession({
            session_id: "sess-manual-mode-2",
            total_committed_turns: 4,
            vad_auto_commit_count: 2,
            manual_commit_count: 2,
            manual_mode_required_count: 1
          }),
          buildRecentLiveSession({
            session_id: "sess-manual-mode-3",
            total_committed_turns: 4,
            vad_auto_commit_count: 2,
            manual_commit_count: 2,
            manual_mode_required_count: 1
          })
        ])}
      />
    )

    await waitFor(() => {
      expect(
        screen.getByText("Suggestion: check auto-commit availability first")
      ).toBeInTheDocument()
    })
  })

  it("marks mixed sessions and excludes them from recommendation heuristics", async () => {
    render(
      <AssistantDefaultsPanel
        selectedPersonaId="persona-1"
        selectedPersonaName="Helper"
        isActive
        analytics={buildVoiceAnalytics([
          buildRecentLiveSession({ session_id: "sess-mixed-1" }),
          buildRecentLiveSession({ session_id: "sess-mixed-2" }),
          buildRecentLiveSession({
            session_id: "sess-mixed-3",
            turn_detection_changed_during_session: true,
            total_committed_turns: 4,
            vad_auto_commit_count: 1,
            manual_commit_count: 3,
            manual_mode_required_count: 2
          })
        ])}
      />
    )

    await waitFor(() => {
      expect(screen.getByText("Mixed session")).toBeInTheDocument()
    })

    expect(
      screen.getByTestId("persona-turn-detection-feedback-mixed-note")
    ).toHaveTextContent("1 mixed recent session excluded from suggestions")
    expect(
      screen.getByText("Suggestion: current settings look healthy")
    ).toBeInTheDocument()
  })

  it("focuses confirmation mode for setup handoff requests after defaults load", async () => {
    const onSetupHandoffFocusConsumed = vi.fn()

    render(
      <AssistantDefaultsPanel
        selectedPersonaId="persona-1"
        selectedPersonaName="Helper"
        isActive
        handoffFocusRequest={{ section: "confirmation_mode", token: 1 }}
        onSetupHandoffFocusConsumed={onSetupHandoffFocusConsumed}
      />
    )

    await waitFor(() =>
      expect(screen.getByLabelText("Confirmation mode")).toHaveFocus()
    )
    expect(onSetupHandoffFocusConsumed).toHaveBeenCalledWith(1)
  })

  it("focuses the first assistant defaults control for assistant_defaults handoff requests", async () => {
    const onSetupHandoffFocusConsumed = vi.fn()

    render(
      <AssistantDefaultsPanel
        selectedPersonaId="persona-1"
        selectedPersonaName="Helper"
        isActive
        handoffFocusRequest={{ section: "assistant_defaults", token: 2 }}
        onSetupHandoffFocusConsumed={onSetupHandoffFocusConsumed}
      />
    )

    await waitFor(() =>
      expect(screen.getByLabelText("STT language")).toHaveFocus()
    )
    expect(onSetupHandoffFocusConsumed).toHaveBeenCalledWith(2)
  })
})

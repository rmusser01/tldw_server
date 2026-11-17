import React from "react";
import { fireEvent, render as testingRender, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
const render: typeof testingRender = (ui, options) => testingRender(ui, { wrapper: ({ children }) => <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>{children}</QueryClientProvider>, ...options });

const generationMocks = vi.hoisted(() => ({
  mutateAsync: vi.fn(),
}));

const audioStudioServiceMocks = vi.hoisted(() => ({
  getProviders: vi.fn(),
  fetchAudioStudioArtifactBlob: vi.fn(),
  mintAudioStudioArtifactMediaTicket: vi.fn(),
}));

const urlMocks = vi.hoisted(() => ({
  createObjectURL: vi.fn(),
  revokeObjectURL: vi.fn(),
}));

const projectHookMocks = vi.hoisted(() => ({
  useProjects: vi.fn(),
  useArtifacts: vi.fn(),
  createProject: vi.fn(),
  updateProject: vi.fn(),
  upsertSection: vi.fn(),
  upsertClip: vi.fn(),
  createPending: false,
  updatePending: false,
  upsertSectionPending: false,
  upsertClipPending: false,
}));

vi.mock("@/components/Option/AudioStudio/AudioStudioSession", () => ({ AudioStudioSession: ({children}: React.PropsWithChildren) => <>{children}</>, useAudioStudioSession: () => null }));
vi.mock("@/entries/shared/route-leave-prompt", () => ({ RouteLeavePrompt: () => null }));
vi.mock("@/components/Option/AudioStudio/useAudioStudioJobs", () => ({ useAudioStudioJobs: () => ({ data: [], isLoading: false }), useCancelAudioStudioJob: () => ({ isPending: false }), isAudioStudioJobActive: () => false }));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback ?? _key,
  }),
}));

vi.mock("react-router-dom", () => ({
  useLocation: () => ({
    pathname: "/audio-studio",
    search: "",
    hash: "",
    state: null,
  }),
  useNavigate: () => vi.fn(),
}));

vi.mock("@/components/Common/PageShell", () => ({
  PageShell: ({ children }: { children: React.ReactNode }) => (
    <main>{children}</main>
  ),
}));

vi.mock("@/components/Option/AudiobookStudio/ContentInput/TextEditor", () => ({
  TextEditor: () => <div>Paste or type your content</div>,
}));

vi.mock(
  "@/components/Option/AudiobookStudio/ChapterEditor/ChapterList",
  () => ({
    ChapterList: () => <div>Chapter List</div>,
  }),
);

vi.mock(
  "@/components/Option/AudiobookStudio/Generation/GenerationPanel",
  () => ({
    GenerationPanel: () => <div>Voice Settings</div>,
  }),
);

vi.mock("@/components/Option/AudiobookStudio/Output/OutputPanel", () => ({
  OutputPanel: () => <div>Audiobook Player</div>,
}));

vi.mock("@/hooks/useAudioStudioGeneration", () => ({
  useRetryAudioStudioAdmission: () => ({ intent: undefined, isPending: false }),
  useCreateAudioStudioRender: () => ({ mutateAsync: vi.fn(), isPending: false }),
  useCreateAudioStudioExport: () => ({ mutateAsync: vi.fn(), isPending: false }),
  useCreateAudioStudioGeneration: () => ({
    mutateAsync: generationMocks.mutateAsync,
    isPending: false,
  }),
}));

vi.mock("@/hooks/useAudioStudioProjects", () => ({
  useAudioStudioProjectDetail: () => ({ isLoading: false, isError: false }),
  useAudioStudioProjects: (...args: unknown[]) =>
    projectHookMocks.useProjects(...args),
  useAudioStudioArtifacts: (...args: unknown[]) =>
    projectHookMocks.useArtifacts(...args),
  useCreateAudioStudioProject: () => ({
    mutateAsync: projectHookMocks.createProject,
    isPending: projectHookMocks.createPending,
  }),
  useUpdateAudioStudioProject: () => ({
    mutateAsync: projectHookMocks.updateProject,
    isPending: projectHookMocks.updatePending,
  }),
  useUpsertAudioStudioSection: () => ({
    mutateAsync: projectHookMocks.upsertSection,
    isPending: projectHookMocks.upsertSectionPending,
  }),
  useUpsertAudioStudioClip: () => ({
    mutateAsync: projectHookMocks.upsertClip,
    isPending: projectHookMocks.upsertClipPending,
  }),
}));

vi.mock("@/services/audio-studio", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/services/audio-studio")>();
  return {
    ...actual,
    getAudioStudioProviders: (...args: unknown[]) => audioStudioServiceMocks.getProviders(...args),
    getAudioStudioSpeechProviders: vi.fn().mockResolvedValue({ providers: { pocket_tts: { models: ["pocket_tts"] } } }),
    getAudioStudioVoices: vi.fn().mockResolvedValue([{ id: "reference", name: "Reference" }]),
    fetchAudioStudioArtifactBlob: (...args: unknown[]) =>
      audioStudioServiceMocks.fetchAudioStudioArtifactBlob(...args),
    mintAudioStudioArtifactMediaTicket: (...args: unknown[]) =>
      audioStudioServiceMocks.mintAudioStudioArtifactMediaTicket(...args),
  };
});

import { AudioStudioPage } from "../AudioStudioPage";
import {
  useAudioStudioStore,
  type AudioStudioProject,
} from "@/store/audio-studio";
import { getAudioStudioSpeechProviders, type AudioStudioArtifact } from "@/services/audio-studio";

const SMALL_AUDIO_BLOB_URL = "blob:mock-selected-clip-audio";

const buildArtifact = (
  overrides: Partial<AudioStudioArtifact> = {},
): AudioStudioArtifact => ({
  artifact_id: "artifact-host",
  artifact_type: "audio",
  provider: "tts",
  mime_type: "audio/wav",
  size_bytes: 1024,
  metadata: { filename: "host-intro.wav" },
  created_at: "2026-06-23T12:00:00Z",
  ...overrides,
});

const setActiveProject = (overrides: Partial<AudioStudioProject> = {}) => {
  useAudioStudioStore.getState().setProjects([
    {
      project_id: "project-1",
      title: "Working project",
      workflow: "music",
      status: "draft",
      current_revision_id: "revision-current",
      sections: [
        {
          section_id: "section-1",
          workflow: "narration",
          title: "Intro",
          body_text: "Welcome.",
          order: 0,
        },
      ],
      tracks: [
        {
          track_id: "music-track-1",
          name: "Music",
          kind: "music",
          order: 0,
        },
      ],
      clips: [],
      settings: { speech: { model: "pocket_tts", voice: "reference", provider_hint: "pocket_tts" } },
      ...overrides,
    } satisfies AudioStudioProject,
  ]);
  useAudioStudioStore.getState().setActiveProjectId("project-1");
};

const createDeferred = <T,>() => {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((promiseResolve, promiseReject) => {
    resolve = promiseResolve;
    reject = promiseReject;
  });
  return { promise, resolve, reject };
};

describe("AudioStudioPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(getAudioStudioSpeechProviders).mockResolvedValue({ providers: { pocket_tts: { models: ["pocket_tts"] } }, voices: {} });
    audioStudioServiceMocks.getProviders.mockResolvedValue({ providers: [{ provider_id: "ace_step", supported_kinds: ["music"] }] });
    projectHookMocks.createProject.mockResolvedValue(undefined);
    projectHookMocks.updateProject.mockResolvedValue(undefined);
    generationMocks.mutateAsync.mockResolvedValue({ job_id: "job-test", status: "queued" });
    Object.defineProperty(URL, "createObjectURL", {
      configurable: true,
      writable: true,
      value: urlMocks.createObjectURL,
    });
    Object.defineProperty(URL, "revokeObjectURL", {
      configurable: true,
      writable: true,
      value: urlMocks.revokeObjectURL,
    });
    urlMocks.createObjectURL.mockReturnValue(SMALL_AUDIO_BLOB_URL);
    audioStudioServiceMocks.fetchAudioStudioArtifactBlob.mockResolvedValue(
      new Blob(["fake audio"], { type: "audio/wav" }),
    );
    audioStudioServiceMocks.mintAudioStudioArtifactMediaTicket.mockResolvedValue(
      {
        ticket_path: "/api/v1/audio-studio/media-tickets/ticket-playback",
        ticket_url:
          "http://127.0.0.1:8000/api/v1/audio-studio/media-tickets/ticket-playback",
        expires_at: "2026-06-24T12:00:00Z",
        purpose: "playback",
        artifact_id: "artifact-host",
      },
    );
    projectHookMocks.createPending = false;
    projectHookMocks.updatePending = false;
    projectHookMocks.upsertSectionPending = false;
    projectHookMocks.upsertClipPending = false;
    projectHookMocks.upsertSection.mockResolvedValue({
      section_id: "section_draft",
      workflow: "podcast",
      title: "Podcast script",
      body_text: "Saved script",
      order_index: 0,
      current_revision_id: "revision-section-saved",
    });
    projectHookMocks.upsertClip.mockResolvedValue({
      clip_id: "clip-host",
      track_id: "speech-track-1",
      title: "Host intro",
      clip_type: "speech",
      start_ms: 2500,
      duration_ms: 42000,
      volume: 0.6,
      fade_in_ms: 750,
      fade_out_ms: 1000,
      muted: true,
      current_revision_id: "revision-clip-saved",
    });
    projectHookMocks.useProjects.mockReturnValue({
      isLoading: false,
      isError: false,
      error: null,
    });
    projectHookMocks.useArtifacts.mockReturnValue({
      data: [],
      isLoading: false,
      isError: false,
    });
    useAudioStudioStore.getState().resetAudioStudio();
  });

  it("renders all workflow labels as first-class choices", () => {
    render(<AudioStudioPage />);

    expect(projectHookMocks.useProjects).toHaveBeenCalledWith({
      workflow: "narration",
      includeArchived: false,
    });
    expect(
      screen.getByRole("heading", { name: "Audio Studio" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /Narration/ })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /Podcast/ })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /Briefing/ })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /Music/ })).toBeInTheDocument();
  });

  it("gives workflow tabs keyboard navigation and panel relationships", () => {
    render(<AudioStudioPage />);

    const narrationTab = screen.getByRole("tab", { name: /Narration/ });
    const podcastTab = screen.getByRole("tab", { name: /Podcast/ });
    const panel = screen.getByRole("tabpanel", { name: /Narration/ });

    expect(narrationTab).toHaveAttribute(
      "aria-controls",
      "audio-studio-workflow-panel",
    );
    expect(narrationTab).toHaveAttribute("tabindex", "0");
    expect(podcastTab).toHaveAttribute("tabindex", "-1");
    expect(panel).toHaveAttribute("id", "audio-studio-workflow-panel");

    fireEvent.keyDown(narrationTab, { key: "ArrowRight" });

    expect(useAudioStudioStore.getState().activeWorkflow).toBe("podcast");
  });

  it("shows saved Studio sections without the disconnected legacy editor", () => {
    setActiveProject({ workflow: "narration" });
    useAudioStudioStore.getState().setActiveWorkflow("narration");
    render(<AudioStudioPage />);
    expect(screen.getByLabelText("Section 1 text")).toHaveValue("Welcome.");
    expect(screen.getByRole("button", { name: "Add section" })).toBeEnabled();
    expect(screen.queryByText("Paste or type your content")).not.toBeInTheDocument();
  });

  it("surfaces saved Podcast and Briefing sections", () => {
    setActiveProject({ workflow: "podcast" }); useAudioStudioStore.getState().setActiveWorkflow("podcast");
    const { rerender } = render(<AudioStudioPage />);
    expect(screen.getByLabelText("Section 1 text")).toHaveValue("Welcome.");
    setActiveProject({ workflow: "briefing" }); useAudioStudioStore.getState().setActiveWorkflow("briefing");
    rerender(<AudioStudioPage />);
    expect(screen.getByLabelText("Section 1 text")).toHaveValue("Welcome.");
  });

  it("shows Music prompt, lyrics, style, and provider controls", () => {
    useAudioStudioStore.getState().setActiveWorkflow("music");

    render(<AudioStudioPage />);

    expect(screen.getByLabelText("Prompt")).toBeInTheDocument();
    expect(screen.getByLabelText("Lyrics")).toBeInTheDocument();
    expect(screen.getByLabelText("Style")).toBeInTheDocument();
    expect(screen.getByLabelText("Provider")).toBeInTheDocument();
    expect(
      screen.queryByRole("option", { name: "Server default" }),
    ).not.toBeInTheDocument();
  });

  it("shows non-disruptive project loading and error states", () => {
    projectHookMocks.useProjects.mockReturnValueOnce({
      isLoading: true,
      isError: false,
      error: null,
    });
    const { rerender } = render(<AudioStudioPage />);

    expect(
      screen.getByText("Loading Audio Studio projects..."),
    ).toBeInTheDocument();

    projectHookMocks.useProjects.mockReturnValueOnce({
      isLoading: false,
      isError: true,
      error: new Error("Nope"),
    });
    rerender(<AudioStudioPage />);

    expect(
      screen.getByText("Audio Studio projects could not load."),
    ).toBeInTheDocument();
  });

  it("creates new projects through the server-backed create hook", async () => {
    useAudioStudioStore.getState().setActiveWorkflow("podcast");

    render(<AudioStudioPage />);

    fireEvent.click(
      screen.getByRole("button", { name: "New Audio Studio project" }),
    );

    await waitFor(() =>
      expect(projectHookMocks.createProject).toHaveBeenCalled(),
    );
    expect(projectHookMocks.createProject).toHaveBeenCalledWith({
      title: "Untitled Podcast Project",
      workflow: "podcast",
    });
    expect(
      useAudioStudioStore
        .getState()
        .projects.some((project) => project.revision_id === "local-draft"),
    ).toBe(false);
  });

  it("selects an existing project from the project sidebar", () => {
    useAudioStudioStore.getState().setProjects([
      {
        project_id: "project-1",
        title: "First project",
        workflow: "narration",
        status: "draft",
        current_revision_id: "revision-1",
        sections: [],
        tracks: [],
        clips: [],
        settings: {},
      },
      {
        project_id: "project-2",
        title: "Second project",
        workflow: "narration",
        status: "draft",
        current_revision_id: "revision-2",
        sections: [],
        tracks: [],
        clips: [],
        settings: {},
      },
    ] satisfies AudioStudioProject[]);
    useAudioStudioStore.getState().setActiveProjectId("project-1");

    render(<AudioStudioPage />);

    fireEvent.click(screen.getByRole("button", { name: /Second project/ }));

    expect(useAudioStudioStore.getState().activeProjectId).toBe("project-2");
  });

  it("saves active project edits through the update hook", async () => {
    setActiveProject({
      workflow: "briefing",
      description: "Existing description",
      settings: { voice: "Ava" },
    });
    useAudioStudioStore.getState().setActiveWorkflow("briefing");

    render(<AudioStudioPage />);

    fireEvent.change(screen.getByLabelText("Project title"), {
      target: { value: "Renamed project" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() =>
      expect(projectHookMocks.updateProject).toHaveBeenCalled(),
    );
    expect(projectHookMocks.updateProject).toHaveBeenCalledWith({
      title: "Renamed project",
      description: "Existing description",
      settings: { voice: "Ava" },
      base_revision_id: "revision-current",
    });
  });

  it("disables save and generation for projects without a real revision", () => {
    setActiveProject({
      workflow: "music",
      current_revision_id: undefined,
      revision_id: "local-draft",
    });
    useAudioStudioStore.getState().setActiveWorkflow("music");

    render(<AudioStudioPage />);

    fireEvent.change(screen.getByLabelText("Prompt"), {
      target: { value: "Warm documentary intro" },
    });

    expect(screen.getByRole("button", { name: "Save" })).toBeDisabled();
    expect(
      screen.getByRole("button", { name: "Generate music" }),
    ).toBeDisabled();
    expect(
      screen.getByRole("button", { name: "Queue music generation" }),
    ).toBeDisabled();
  });

  it("does not keep a hidden project active after changing workflows", () => {
    setActiveProject({ workflow: "narration" });
    useAudioStudioStore.getState().setActiveWorkflow("narration");

    render(<AudioStudioPage />);

    fireEvent.click(screen.getByRole("tab", { name: /Music/ }));
    fireEvent.change(screen.getByLabelText("Prompt"), {
      target: { value: "Warm documentary intro" },
    });

    expect(useAudioStudioStore.getState().activeProject).toBeNull();
    expect(screen.getByRole("button", { name: "Save" })).toBeDisabled();
    expect(
      screen.getByRole("button", { name: "Generate music" }),
    ).toBeDisabled();

    fireEvent.click(screen.getByRole("button", { name: "Generate music" }));

    expect(generationMocks.mutateAsync).not.toHaveBeenCalled();
  });

  it("refuses render/export until current section audio is available", () => {
    setActiveProject({ workflow: "narration" }); useAudioStudioStore.getState().setActiveWorkflow("narration");
    render(<AudioStudioPage />);
    expect(screen.getByRole("button", { name: "Create preview render" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Create export" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Create preview render" })).toHaveAttribute("title", "Generate current audio for every section before rendering.");
  });

  it("renders active project tracks and clips in the timeline editor", () => {
    setActiveProject({
      workflow: "podcast",
      tracks: [
        {
          track_id: "speech-track-1",
          name: "Dialogue",
          kind: "speech",
          order: 0,
        },
        {
          track_id: "music-track-1",
          name: "Music bed",
          kind: "music",
          order: 1,
        },
      ],
      clips: [
        {
          clip_id: "clip-host",
          track_id: "speech-track-1",
          section_id: "section-1",
          artifact_id: "artifact-host",
          title: "Host intro",
          clip_type: "speech",
          start_ms: 1000,
          duration_ms: 30000,
          volume: 0.8,
          fade_in_ms: 250,
          fade_out_ms: 500,
        },
      ],
    });
    useAudioStudioStore.getState().setActiveWorkflow("podcast");

    render(<AudioStudioPage />);

    expect(
      screen.getByRole("heading", { name: "Timeline" }),
    ).toBeInTheDocument();
    expect(screen.getByText("Dialogue")).toBeInTheDocument();
    expect(screen.getByText("Music bed")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /Host intro/ }),
    ).toBeInTheDocument();
    expect(screen.getByText("Starts 1.0s")).toBeInTheDocument();
  });

  it("loads a selected clip artifact as a Blob URL preview and safe download", async () => {
    const artifact = buildArtifact();
    projectHookMocks.useArtifacts.mockReturnValue({
      data: [artifact],
      isLoading: false,
      isError: false,
    });
    setActiveProject({
      workflow: "podcast",
      tracks: [
        {
          track_id: "speech-track-1",
          name: "Dialogue",
          kind: "speech",
          order: 0,
        },
      ],
      clips: [
        {
          clip_id: "clip-host",
          track_id: "speech-track-1",
          section_id: "section-1",
          artifact_id: "artifact-host",
          title: "Host intro",
          clip_type: "speech",
          start_ms: 1000,
          duration_ms: 30000,
        },
      ],
    });
    useAudioStudioStore.getState().setActiveWorkflow("podcast");

    const { unmount } = render(<AudioStudioPage />);

    await waitFor(() =>
      expect(
        audioStudioServiceMocks.fetchAudioStudioArtifactBlob,
      ).toHaveBeenCalledWith("project-1", artifact, undefined),
    );
    const audioPreview = await screen.findByLabelText(
      "Selected clip audio preview",
    );
    expect(audioPreview).toHaveAttribute("src", SMALL_AUDIO_BLOB_URL);
    const downloadLink = screen.getByRole("link", {
      name: "Download selected clip audio",
    });
    expect(downloadLink).toHaveAttribute("href", SMALL_AUDIO_BLOB_URL);
    expect(downloadLink).toHaveAttribute("download", "host-intro.wav");
    expect(downloadLink.getAttribute("href")).not.toContain(
      "/api/v1/audio-studio",
    );
    expect(document.body.innerHTML).not.toContain("/api/v1/audio-studio");

    unmount();

    expect(urlMocks.revokeObjectURL).toHaveBeenCalledWith(SMALL_AUDIO_BLOB_URL);
  });

  it("does not create a Blob object URL after the selected clip preview is stale", async () => {
    const artifact = buildArtifact();
    const previewBlob = new Blob(["late audio"], { type: "audio/wav" });
    const previewFetch = createDeferred<Blob>();
    audioStudioServiceMocks.fetchAudioStudioArtifactBlob.mockReturnValue(
      previewFetch.promise,
    );
    projectHookMocks.useArtifacts.mockReturnValue({
      data: [artifact],
      isLoading: false,
      isError: false,
    });
    setActiveProject({
      workflow: "podcast",
      tracks: [
        {
          track_id: "speech-track-1",
          name: "Dialogue",
          kind: "speech",
          order: 0,
        },
      ],
      clips: [
        {
          clip_id: "clip-host",
          track_id: "speech-track-1",
          section_id: "section-1",
          artifact_id: "artifact-host",
          title: "Host intro",
          clip_type: "speech",
          start_ms: 1000,
          duration_ms: 30000,
        },
      ],
    });
    useAudioStudioStore.getState().setActiveWorkflow("podcast");

    const { unmount } = render(<AudioStudioPage />);

    await waitFor(() =>
      expect(
        audioStudioServiceMocks.fetchAudioStudioArtifactBlob,
      ).toHaveBeenCalledWith("project-1", artifact, undefined),
    );

    unmount();
    previewFetch.resolve(previewBlob);
    await Promise.resolve();
    await Promise.resolve();

    expect(urlMocks.createObjectURL).not.toHaveBeenCalled();
  });

  it("loads a selected clip artifact when its artifact type is clip audio", async () => {
    const artifact = buildArtifact({
      artifact_type: "clip_audio",
      mime_type: "audio/wav",
    });
    projectHookMocks.useArtifacts.mockReturnValue({
      data: [artifact],
      isLoading: false,
      isError: false,
    });
    setActiveProject({
      workflow: "podcast",
      tracks: [
        {
          track_id: "speech-track-1",
          name: "Dialogue",
          kind: "speech",
          order: 0,
        },
      ],
      clips: [
        {
          clip_id: "clip-host",
          track_id: "speech-track-1",
          section_id: "section-1",
          artifact_id: "artifact-host",
          title: "Host intro",
          clip_type: "speech",
          start_ms: 1000,
          duration_ms: 30000,
        },
      ],
    });
    useAudioStudioStore.getState().setActiveWorkflow("podcast");

    render(<AudioStudioPage />);

    await waitFor(() =>
      expect(
        audioStudioServiceMocks.fetchAudioStudioArtifactBlob,
      ).toHaveBeenCalledWith("project-1", artifact, undefined),
    );
    expect(
      await screen.findByLabelText("Selected clip audio preview"),
    ).toHaveAttribute("src", SMALL_AUDIO_BLOB_URL);
    expect(
      screen.getByRole("link", { name: "Download selected clip audio" }),
    ).toHaveAttribute("href", SMALL_AUDIO_BLOB_URL);
  });

  it("shows an unavailable state when the selected clip has no artifact id", async () => {
    setActiveProject({
      workflow: "podcast",
      tracks: [
        {
          track_id: "speech-track-1",
          name: "Dialogue",
          kind: "speech",
          order: 0,
        },
      ],
      clips: [
        {
          clip_id: "clip-host",
          track_id: "speech-track-1",
          title: "Host intro",
          clip_type: "speech",
          start_ms: 1000,
          duration_ms: 30000,
        },
      ],
    });
    useAudioStudioStore.getState().setActiveWorkflow("podcast");

    render(<AudioStudioPage />);

    expect(
      await screen.findByText("No audio artifact is attached to this clip."),
    ).toBeInTheDocument();
    expect(
      audioStudioServiceMocks.fetchAudioStudioArtifactBlob,
    ).not.toHaveBeenCalled();
  });

  it("shows a missing state when clip artifact metadata is absent", async () => {
    setActiveProject({
      workflow: "podcast",
      tracks: [
        {
          track_id: "speech-track-1",
          name: "Dialogue",
          kind: "speech",
          order: 0,
        },
      ],
      clips: [
        {
          clip_id: "clip-host",
          track_id: "speech-track-1",
          artifact_id: "artifact-host",
          title: "Host intro",
          clip_type: "speech",
          start_ms: 1000,
          duration_ms: 30000,
        },
      ],
    });
    useAudioStudioStore.getState().setActiveWorkflow("podcast");

    render(<AudioStudioPage />);

    expect(
      await screen.findByText(
        "Selected clip artifact metadata is unavailable.",
      ),
    ).toBeInTheDocument();
    expect(
      audioStudioServiceMocks.fetchAudioStudioArtifactBlob,
    ).not.toHaveBeenCalled();
  });

  it("offers click-only download for non-audio artifacts without rendering an audio element", async () => {
    const artifact = buildArtifact({
      artifact_type: "analysis",
      mime_type: "application/json",
      size_bytes: null,
      metadata: { filename: "host-analysis.json" },
    });
    audioStudioServiceMocks.mintAudioStudioArtifactMediaTicket.mockResolvedValueOnce(
      {
        ticket_path: "/api/v1/audio-studio/media-tickets/ticket-json",
        ticket_url:
          "http://127.0.0.1:8000/api/v1/audio-studio/media-tickets/ticket-json",
        expires_at: "2026-06-24T12:00:00Z",
        purpose: "download",
        artifact_id: "artifact-host",
      },
    );
    projectHookMocks.useArtifacts.mockReturnValue({
      data: [artifact],
      isLoading: false,
      isError: false,
    });
    setActiveProject({
      workflow: "podcast",
      tracks: [
        {
          track_id: "speech-track-1",
          name: "Dialogue",
          kind: "speech",
          order: 0,
        },
      ],
      clips: [
        {
          clip_id: "clip-host",
          track_id: "speech-track-1",
          artifact_id: "artifact-host",
          title: "Host intro",
          clip_type: "speech",
          start_ms: 1000,
          duration_ms: 30000,
        },
      ],
    });
    useAudioStudioStore.getState().setActiveWorkflow("podcast");
    const anchorClick = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => {});

    render(<AudioStudioPage />);

    expect(
      await screen.findByText("Selected clip artifact is download-only."),
    ).toBeInTheDocument();
    expect(
      audioStudioServiceMocks.fetchAudioStudioArtifactBlob,
    ).not.toHaveBeenCalled();
    expect(
      screen.queryByLabelText("Selected clip audio preview"),
    ).not.toBeInTheDocument();

    fireEvent.click(
      screen.getByRole("button", { name: "Download selected artifact" }),
    );

    await waitFor(() =>
      expect(
        audioStudioServiceMocks.mintAudioStudioArtifactMediaTicket,
      ).toHaveBeenCalledWith("project-1", "artifact-host", "download", undefined),
    );
    expect(anchorClick).toHaveBeenCalled();
    anchorClick.mockRestore();
  });

  it("streams an unknown-size selected audio artifact with a playback ticket", async () => {
    const artifact = buildArtifact({
      size_bytes: null,
    });
    projectHookMocks.useArtifacts.mockReturnValue({
      data: [artifact],
      isLoading: false,
      isError: false,
    });
    setActiveProject({
      workflow: "narration",
      tracks: [
        {
          track_id: "speech-track-1",
          name: "Narration",
          kind: "speech",
          order: 0,
        },
      ],
      clips: [
        {
          clip_id: "clip-host",
          track_id: "speech-track-1",
          section_id: "section-1",
          artifact_id: "artifact-host",
          title: "Narration clip",
          clip_type: "speech",
          start_ms: 1000,
          duration_ms: 30000,
        },
      ],
    });
    useAudioStudioStore.getState().setActiveWorkflow("narration");

    render(<AudioStudioPage />);

    await waitFor(() =>
      expect(
        audioStudioServiceMocks.mintAudioStudioArtifactMediaTicket,
      ).toHaveBeenCalledWith("project-1", "artifact-host", "playback", undefined),
    );
    expect(
      audioStudioServiceMocks.fetchAudioStudioArtifactBlob,
    ).not.toHaveBeenCalled();
    expect(
      await screen.findByLabelText("Selected clip audio preview"),
    ).toHaveAttribute(
      "src",
      "http://127.0.0.1:8000/api/v1/audio-studio/media-tickets/ticket-playback",
    );
  });

  it("streams an oversized selected audio artifact with a playback ticket", async () => {
    const artifact = buildArtifact({
      size_bytes: 25 * 1024 * 1024 + 1,
    });
    projectHookMocks.useArtifacts.mockReturnValue({
      data: [artifact],
      isLoading: false,
      isError: false,
    });
    setActiveProject({
      workflow: "podcast",
      tracks: [
        {
          track_id: "speech-track-1",
          name: "Dialogue",
          kind: "speech",
          order: 0,
        },
      ],
      clips: [
        {
          clip_id: "clip-host",
          track_id: "speech-track-1",
          artifact_id: "artifact-host",
          title: "Host intro",
          clip_type: "speech",
          start_ms: 1000,
          duration_ms: 30000,
        },
      ],
    });
    useAudioStudioStore.getState().setActiveWorkflow("podcast");

    render(<AudioStudioPage />);

    await waitFor(() =>
      expect(
        audioStudioServiceMocks.mintAudioStudioArtifactMediaTicket,
      ).toHaveBeenCalledWith("project-1", "artifact-host", "playback", undefined),
    );
    expect(
      audioStudioServiceMocks.fetchAudioStudioArtifactBlob,
    ).not.toHaveBeenCalled();
    expect(
      await screen.findByLabelText("Selected clip audio preview"),
    ).toHaveAttribute(
      "src",
      "http://127.0.0.1:8000/api/v1/audio-studio/media-tickets/ticket-playback",
    );
    expect(screen.getByLabelText("Selected clip audio preview")).toHaveAttribute(
      "referrerpolicy",
      "no-referrer",
    );
  });

  it("does not restore reminted playback time onto a newly selected artifact", async () => {
    const hostTicket = {
      ticket_path: "/api/v1/audio-studio/media-tickets/ticket-host",
      ticket_url:
        "http://127.0.0.1:8000/api/v1/audio-studio/media-tickets/ticket-host",
      expires_at: "2026-06-24T12:00:00Z",
      purpose: "playback",
      artifact_id: "artifact-host",
    };
    const hostRemintTicket = {
      ...hostTicket,
      ticket_path: "/api/v1/audio-studio/media-tickets/ticket-host-remint",
      ticket_url:
        "http://127.0.0.1:8000/api/v1/audio-studio/media-tickets/ticket-host-remint",
    };
    const guestTicket = {
      ticket_path: "/api/v1/audio-studio/media-tickets/ticket-guest",
      ticket_url:
        "http://127.0.0.1:8000/api/v1/audio-studio/media-tickets/ticket-guest",
      expires_at: "2026-06-24T12:00:00Z",
      purpose: "playback",
      artifact_id: "artifact-guest",
    };
    const hostRemint = createDeferred<typeof hostRemintTicket>();
    let hostPlaybackCalls = 0;
    audioStudioServiceMocks.mintAudioStudioArtifactMediaTicket.mockImplementation(
      (
        _projectId: string,
        artifactId: string,
        purpose: "playback" | "download",
      ) => {
        if (purpose === "playback" && artifactId === "artifact-host") {
          hostPlaybackCalls += 1;
          return hostPlaybackCalls === 1
            ? Promise.resolve(hostTicket)
            : hostRemint.promise;
        }
        if (purpose === "playback" && artifactId === "artifact-guest") {
          return Promise.resolve(guestTicket);
        }
        return Promise.reject(new Error("unexpected ticket request"));
      },
    );
    projectHookMocks.useArtifacts.mockReturnValue({
      data: [
        buildArtifact({ size_bytes: null }),
        buildArtifact({
          artifact_id: "artifact-guest",
          size_bytes: null,
          metadata: { filename: "guest-reply.wav" },
        }),
      ],
      isLoading: false,
      isError: false,
    });
    setActiveProject({
      workflow: "podcast",
      tracks: [
        {
          track_id: "speech-track-1",
          name: "Dialogue",
          kind: "speech",
          order: 0,
        },
      ],
      clips: [
        {
          clip_id: "clip-host",
          track_id: "speech-track-1",
          section_id: "section-1",
          artifact_id: "artifact-host",
          title: "Host intro",
          clip_type: "speech",
          start_ms: 1000,
          duration_ms: 30000,
        },
        {
          clip_id: "clip-guest",
          track_id: "speech-track-1",
          section_id: "section-1",
          artifact_id: "artifact-guest",
          title: "Guest reply",
          clip_type: "speech",
          start_ms: 32000,
          duration_ms: 15000,
        },
      ],
    });
    useAudioStudioStore.getState().setActiveWorkflow("podcast");

    render(<AudioStudioPage />);

    const hostAudio = (await screen.findByLabelText(
      "Selected clip audio preview",
    )) as HTMLAudioElement;
    await waitFor(() =>
      expect(hostAudio).toHaveAttribute("src", hostTicket.ticket_url),
    );
    hostAudio.currentTime = 12;

    fireEvent.error(hostAudio);
    await waitFor(() => expect(hostPlaybackCalls).toBe(2));

    const originalSetTimeout = window.setTimeout.bind(window);
    let restoreCurrentTime: (() => void) | null = null;
    const timeoutSpy = vi
      .spyOn(window, "setTimeout")
      .mockImplementation((handler: TimerHandler, timeout?: number) => {
        if (
          !restoreCurrentTime &&
          timeout === 0 &&
          typeof handler === "function"
        ) {
          restoreCurrentTime = handler as () => void;
          return 1;
        }
        return originalSetTimeout(handler, timeout);
      });

    try {
      hostRemint.resolve(hostRemintTicket);
      await waitFor(() => expect(timeoutSpy).toHaveBeenCalled());

      fireEvent.click(screen.getByRole("button", { name: /Guest reply/ }));
      await waitFor(() =>
        expect(
          audioStudioServiceMocks.mintAudioStudioArtifactMediaTicket,
        ).toHaveBeenCalledWith("project-1", "artifact-guest", "playback", undefined),
      );
      const guestAudio = (await screen.findByLabelText(
        "Selected clip audio preview",
      )) as HTMLAudioElement;
      await waitFor(() =>
        expect(guestAudio).toHaveAttribute("src", guestTicket.ticket_url),
      );
      guestAudio.currentTime = 0;

      restoreCurrentTime?.();

      expect(guestAudio).toHaveAttribute("src", guestTicket.ticket_url);
      expect(guestAudio.currentTime).toBe(0);
    } finally {
      timeoutSpy.mockRestore();
    }
  });

  it("keeps a ticket-backed audio download available after playback fails", async () => {
    const artifact = buildArtifact({ size_bytes: 25 * 1024 * 1024 + 1 });
    audioStudioServiceMocks.mintAudioStudioArtifactMediaTicket.mockRejectedValue(
      new Error("playback ticket failed"),
    );
    projectHookMocks.useArtifacts.mockReturnValue({
      data: [artifact],
      isLoading: false,
      isError: false,
    });
    setActiveProject({
      workflow: "podcast",
      tracks: [
        {
          track_id: "speech-track-1",
          name: "Dialogue",
          kind: "speech",
          order: 0,
        },
      ],
      clips: [
        {
          clip_id: "clip-host",
          track_id: "speech-track-1",
          section_id: "section-1",
          artifact_id: "artifact-host",
          title: "Host intro",
          clip_type: "speech",
          start_ms: 1000,
          duration_ms: 30000,
        },
      ],
    });
    useAudioStudioStore.getState().setActiveWorkflow("podcast");

    render(<AudioStudioPage />);

    expect(await screen.findByText("Preview unavailable")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Download selected clip audio" }),
    ).toBeInTheDocument();
  });

  it("mints a download ticket only when the user clicks a large audio download", async () => {
    const artifact = buildArtifact({ size_bytes: 25 * 1024 * 1024 + 1 });
    audioStudioServiceMocks.mintAudioStudioArtifactMediaTicket
      .mockResolvedValueOnce({
        ticket_path: "/api/v1/audio-studio/media-tickets/ticket-playback",
        ticket_url:
          "http://127.0.0.1:8000/api/v1/audio-studio/media-tickets/ticket-playback",
        expires_at: "2026-06-24T12:00:00Z",
        purpose: "playback",
        artifact_id: "artifact-host",
      })
      .mockResolvedValueOnce({
        ticket_path: "/api/v1/audio-studio/media-tickets/ticket-download",
        ticket_url:
          "http://127.0.0.1:8000/api/v1/audio-studio/media-tickets/ticket-download",
        expires_at: "2026-06-24T12:00:00Z",
        purpose: "download",
        artifact_id: "artifact-host",
      });
    projectHookMocks.useArtifacts.mockReturnValue({
      data: [artifact],
      isLoading: false,
      isError: false,
    });
    setActiveProject({
      workflow: "podcast",
      tracks: [
        {
          track_id: "speech-track-1",
          name: "Dialogue",
          kind: "speech",
          order: 0,
        },
      ],
      clips: [
        {
          clip_id: "clip-host",
          track_id: "speech-track-1",
          section_id: "section-1",
          artifact_id: "artifact-host",
          title: "Host intro",
          clip_type: "speech",
          start_ms: 1000,
          duration_ms: 30000,
        },
      ],
    });
    useAudioStudioStore.getState().setActiveWorkflow("podcast");
    const anchorClick = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => {});

    render(<AudioStudioPage />);
    const button = await screen.findByRole("button", {
      name: "Download selected clip audio",
    });
    expect(document.body.innerHTML).not.toContain("ticket-download");
    expect(
      audioStudioServiceMocks.mintAudioStudioArtifactMediaTicket,
    ).not.toHaveBeenCalledWith("project-1", "artifact-host", "download", undefined);

    fireEvent.click(button);

    await waitFor(() =>
      expect(
        audioStudioServiceMocks.mintAudioStudioArtifactMediaTicket,
      ).toHaveBeenLastCalledWith("project-1", "artifact-host", "download", undefined),
    );
    expect(anchorClick).toHaveBeenCalled();
    expect(document.body.innerHTML).not.toContain("ticket-download");
    anchorClick.mockRestore();
  });

  it("does not show a stale download error after selecting another artifact", async () => {
    const hostTicket = {
      ticket_path: "/api/v1/audio-studio/media-tickets/ticket-host",
      ticket_url:
        "http://127.0.0.1:8000/api/v1/audio-studio/media-tickets/ticket-host",
      expires_at: "2026-06-24T12:00:00Z",
      purpose: "playback",
      artifact_id: "artifact-host",
    };
    const guestTicket = {
      ticket_path: "/api/v1/audio-studio/media-tickets/ticket-guest",
      ticket_url:
        "http://127.0.0.1:8000/api/v1/audio-studio/media-tickets/ticket-guest",
      expires_at: "2026-06-24T12:00:00Z",
      purpose: "playback",
      artifact_id: "artifact-guest",
    };
    const hostDownload = createDeferred<typeof hostTicket>();
    audioStudioServiceMocks.mintAudioStudioArtifactMediaTicket.mockImplementation(
      (
        _projectId: string,
        artifactId: string,
        purpose: "playback" | "download",
      ) => {
        if (purpose === "download" && artifactId === "artifact-host") {
          return hostDownload.promise;
        }
        if (purpose === "playback" && artifactId === "artifact-host") {
          return Promise.resolve(hostTicket);
        }
        if (purpose === "playback" && artifactId === "artifact-guest") {
          return Promise.resolve(guestTicket);
        }
        return Promise.reject(new Error("unexpected ticket request"));
      },
    );
    projectHookMocks.useArtifacts.mockReturnValue({
      data: [
        buildArtifact({ size_bytes: null }),
        buildArtifact({
          artifact_id: "artifact-guest",
          size_bytes: null,
          metadata: { filename: "guest-reply.wav" },
        }),
      ],
      isLoading: false,
      isError: false,
    });
    setActiveProject({
      workflow: "podcast",
      tracks: [
        {
          track_id: "speech-track-1",
          name: "Dialogue",
          kind: "speech",
          order: 0,
        },
      ],
      clips: [
        {
          clip_id: "clip-host",
          track_id: "speech-track-1",
          section_id: "section-1",
          artifact_id: "artifact-host",
          title: "Host intro",
          clip_type: "speech",
          start_ms: 1000,
          duration_ms: 30000,
        },
        {
          clip_id: "clip-guest",
          track_id: "speech-track-1",
          section_id: "section-1",
          artifact_id: "artifact-guest",
          title: "Guest reply",
          clip_type: "speech",
          start_ms: 32000,
          duration_ms: 15000,
        },
      ],
    });
    useAudioStudioStore.getState().setActiveWorkflow("podcast");

    render(<AudioStudioPage />);

    fireEvent.click(
      await screen.findByRole("button", {
        name: "Download selected clip audio",
      }),
    );
    fireEvent.click(screen.getByRole("button", { name: /Guest reply/ }));
    await waitFor(() =>
      expect(
        audioStudioServiceMocks.mintAudioStudioArtifactMediaTicket,
      ).toHaveBeenCalledWith("project-1", "artifact-guest", "playback", undefined),
    );

    hostDownload.reject(new Error("download failed"));
    await Promise.resolve();
    await Promise.resolve();

    expect(screen.queryByText("Download unavailable")).not.toBeInTheDocument();
  });

  it("does not trigger a stale successful download after selecting another artifact", async () => {
    const hostTicket = {
      ticket_path: "/api/v1/audio-studio/media-tickets/ticket-host",
      ticket_url:
        "http://127.0.0.1:8000/api/v1/audio-studio/media-tickets/ticket-host",
      expires_at: "2026-06-24T12:00:00Z",
      purpose: "playback",
      artifact_id: "artifact-host",
    };
    const guestTicket = {
      ticket_path: "/api/v1/audio-studio/media-tickets/ticket-guest",
      ticket_url:
        "http://127.0.0.1:8000/api/v1/audio-studio/media-tickets/ticket-guest",
      expires_at: "2026-06-24T12:00:00Z",
      purpose: "playback",
      artifact_id: "artifact-guest",
    };
    const hostDownloadTicket = {
      ticket_path: "/api/v1/audio-studio/media-tickets/ticket-host-download",
      ticket_url:
        "http://127.0.0.1:8000/api/v1/audio-studio/media-tickets/ticket-host-download",
      expires_at: "2026-06-24T12:00:00Z",
      purpose: "download",
      artifact_id: "artifact-host",
    };
    const hostDownload = createDeferred<typeof hostDownloadTicket>();
    audioStudioServiceMocks.mintAudioStudioArtifactMediaTicket.mockImplementation(
      (
        _projectId: string,
        artifactId: string,
        purpose: "playback" | "download",
      ) => {
        if (purpose === "download" && artifactId === "artifact-host") {
          return hostDownload.promise;
        }
        if (purpose === "playback" && artifactId === "artifact-host") {
          return Promise.resolve(hostTicket);
        }
        if (purpose === "playback" && artifactId === "artifact-guest") {
          return Promise.resolve(guestTicket);
        }
        return Promise.reject(new Error("unexpected ticket request"));
      },
    );
    projectHookMocks.useArtifacts.mockReturnValue({
      data: [
        buildArtifact({ size_bytes: null }),
        buildArtifact({
          artifact_id: "artifact-guest",
          size_bytes: null,
          metadata: { filename: "guest-reply.wav" },
        }),
      ],
      isLoading: false,
      isError: false,
    });
    setActiveProject({
      workflow: "podcast",
      tracks: [
        {
          track_id: "speech-track-1",
          name: "Dialogue",
          kind: "speech",
          order: 0,
        },
      ],
      clips: [
        {
          clip_id: "clip-host",
          track_id: "speech-track-1",
          section_id: "section-1",
          artifact_id: "artifact-host",
          title: "Host intro",
          clip_type: "speech",
          start_ms: 1000,
          duration_ms: 30000,
        },
        {
          clip_id: "clip-guest",
          track_id: "speech-track-1",
          section_id: "section-1",
          artifact_id: "artifact-guest",
          title: "Guest reply",
          clip_type: "speech",
          start_ms: 32000,
          duration_ms: 15000,
        },
      ],
    });
    useAudioStudioStore.getState().setActiveWorkflow("podcast");
    const anchorClick = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => {});

    try {
      render(<AudioStudioPage />);

      fireEvent.click(
        await screen.findByRole("button", {
          name: "Download selected clip audio",
        }),
      );
      fireEvent.click(screen.getByRole("button", { name: /Guest reply/ }));
      await waitFor(() =>
        expect(
          audioStudioServiceMocks.mintAudioStudioArtifactMediaTicket,
        ).toHaveBeenCalledWith("project-1", "artifact-guest", "playback", undefined),
      );

      hostDownload.resolve(hostDownloadTicket);
      await Promise.resolve();
      await Promise.resolve();

      expect(anchorClick).not.toHaveBeenCalled();
    } finally {
      anchorClick.mockRestore();
    }
  });

  it("shows a compact error state when artifact Blob preview fetch fails", async () => {
    projectHookMocks.useArtifacts.mockReturnValue({
      data: [buildArtifact()],
      isLoading: false,
      isError: false,
    });
    audioStudioServiceMocks.fetchAudioStudioArtifactBlob.mockRejectedValue(
      new Error("download failed"),
    );
    setActiveProject({
      workflow: "podcast",
      tracks: [
        {
          track_id: "speech-track-1",
          name: "Dialogue",
          kind: "speech",
          order: 0,
        },
      ],
      clips: [
        {
          clip_id: "clip-host",
          track_id: "speech-track-1",
          artifact_id: "artifact-host",
          title: "Host intro",
          clip_type: "speech",
          start_ms: 1000,
          duration_ms: 30000,
        },
      ],
    });
    useAudioStudioStore.getState().setActiveWorkflow("podcast");

    render(<AudioStudioPage />);

    expect(await screen.findByText("Preview unavailable")).toBeInTheDocument();
    expect(
      screen.queryByLabelText("Selected clip audio preview"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("link", { name: "Download selected clip audio" }),
    ).not.toBeInTheDocument();
  });

  it("persists selected clip trim fade and volume edits through the clip hook", async () => {
    setActiveProject({
      workflow: "podcast",
      tracks: [
        {
          track_id: "speech-track-1",
          name: "Dialogue",
          kind: "speech",
          order: 0,
        },
      ],
      clips: [
        {
          clip_id: "clip-host",
          track_id: "speech-track-1",
          section_id: "section-1",
          artifact_id: "artifact-host",
          title: "Host intro",
          clip_type: "speech",
          start_ms: 1000,
          duration_ms: 30000,
          volume: 0.8,
          fade_in_ms: 250,
          fade_out_ms: 500,
        },
      ],
    });
    useAudioStudioStore.getState().setActiveWorkflow("podcast");

    render(<AudioStudioPage />);

    fireEvent.click(screen.getByRole("button", { name: /Host intro/ }));
    fireEvent.change(screen.getByLabelText("Start seconds"), {
      target: { value: "2.5" },
    });
    fireEvent.change(screen.getByLabelText("Duration seconds"), {
      target: { value: "42" },
    });
    fireEvent.change(screen.getByLabelText("Volume percent"), {
      target: { value: "60" },
    });
    fireEvent.change(screen.getByLabelText("Fade in seconds"), {
      target: { value: "0.75" },
    });
    fireEvent.change(screen.getByLabelText("Fade out seconds"), {
      target: { value: "1" },
    });
    fireEvent.click(screen.getByLabelText("Mute clip"));
    fireEvent.click(screen.getByRole("button", { name: "Save clip edits" }));

    await waitFor(() => expect(projectHookMocks.upsertClip).toHaveBeenCalled());
    expect(projectHookMocks.upsertClip).toHaveBeenCalledWith({
      clipId: "clip-host",
      payload: {
        base_revision_id: "revision-current",
        track_id: "speech-track-1",
        section_id: "section-1",
        title: "Host intro",
        clip_type: "speech",
        artifact_id: "artifact-host",
        start_ms: 2500,
        duration_ms: 42000,
        volume: 0.6,
        fade_in_ms: 750,
        fade_out_ms: 1000,
        muted: true,
        settings: {},
      },
    });
  });

  it("updates the selected clip start time when dragged on its timeline lane", () => {
    setActiveProject({
      workflow: "podcast",
      tracks: [
        {
          track_id: "speech-track-1",
          name: "Dialogue",
          kind: "speech",
          order: 0,
        },
      ],
      clips: [
        {
          clip_id: "clip-host",
          track_id: "speech-track-1",
          title: "Host intro",
          clip_type: "speech",
          start_ms: 0,
          duration_ms: 45000,
          volume: 0.8,
        },
      ],
    });
    useAudioStudioStore.getState().setActiveWorkflow("podcast");

    render(<AudioStudioPage />);

    const clipButton = screen.getByRole("button", { name: /Host intro/ });
    const lane = clipButton.parentElement;
    expect(lane).not.toBeNull();
    vi.spyOn(lane as HTMLElement, "getBoundingClientRect").mockReturnValue({
      x: 0,
      y: 0,
      left: 0,
      right: 100,
      top: 0,
      bottom: 56,
      width: 100,
      height: 56,
      toJSON: () => ({}),
    });

    fireEvent.pointerDown(clipButton, { clientX: 0, pointerId: 1 });
    fireEvent.pointerMove(clipButton, { clientX: 20, pointerId: 1 });
    fireEvent.pointerUp(clipButton, { clientX: 20, pointerId: 1 });

    expect(screen.getByLabelText("Start seconds")).toHaveValue("9.0");
  });

  it("supports timeline playhead animation pause and manual scrub state", () => {
    setActiveProject({
      workflow: "music",
      tracks: [
        {
          track_id: "music-track-1",
          name: "Music",
          kind: "music",
          order: 0,
        },
      ],
      clips: [
        {
          clip_id: "clip-bed",
          track_id: "music-track-1",
          title: "Music bed",
          clip_type: "music",
          start_ms: 0,
          duration_ms: 45000,
          volume: 0.7,
        },
      ],
    });
    useAudioStudioStore.getState().setActiveWorkflow("music");

    render(<AudioStudioPage />);

    fireEvent.change(screen.getByLabelText("Timeline playhead seconds"), {
      target: { value: "12" },
    });

    expect(screen.getByText("12.0s / 45.0s")).toBeInTheDocument();

    fireEvent.click(
      screen.getByRole("button", { name: "Animate playhead" }),
    );
    expect(
      screen.getByRole("button", { name: "Pause playhead animation" }),
    ).toBeInTheDocument();
  });

  it("queues music generation with controlled Music workflow inputs", async () => {
    setActiveProject({ workflow: "music" });
    useAudioStudioStore.getState().setActiveWorkflow("music");

    render(<AudioStudioPage />);

    fireEvent.change(screen.getByLabelText("Prompt"), {
      target: { value: "Warm documentary intro" },
    });
    fireEvent.change(screen.getByLabelText("Lyrics"), {
      target: { value: "Hold the first phrase" },
    });
    fireEvent.change(screen.getByLabelText("Style"), {
      target: { value: "cinematic, ambient" },
    });
    projectHookMocks.updateProject.mockImplementationOnce(async () => {
      const store = useAudioStudioStore.getState(); const current = store.activeProject!;
      store.markProjectClean(current.project_id); store.upsertProjectFromServer({ ...current, current_revision_id: "revision-music-saved" });
      return useAudioStudioStore.getState().activeProject;
    });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() => expect(screen.getByRole("button", { name: "Generate music" })).toBeEnabled());
    fireEvent.click(screen.getByRole("button", { name: "Generate music" }));

    await waitFor(() => expect(generationMocks.mutateAsync).toHaveBeenCalled());
    expect(generationMocks.mutateAsync).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: "music",
        provider: "ace_step",
        target_resource_kind: "track",
        target_resource_id: "music-track-1",
        target_revision_id: "revision-music-saved",
        options: {
          prompt: "Warm documentary intro",
          lyrics: "Hold the first phrase",
          style: "cinematic, ambient",
          duration: 45,
        },
      }),
    );
    expect(
      generationMocks.mutateAsync.mock.calls[0][0].idempotency_key.length,
    ).toBeGreaterThanOrEqual(16);
  });

  it("does not submit a blank Music request from the shared panel", () => {
    setActiveProject({ workflow: "music" }); useAudioStudioStore.getState().setActiveWorkflow("music");
    render(<AudioStudioPage />);
    expect(screen.getByRole("button", { name: "Queue music generation" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Queue music generation" }));
    expect(generationMocks.mutateAsync).not.toHaveBeenCalled();
  });

  it("queues shared speech generation for the first available section", async () => {
    setActiveProject({ workflow: "narration" });
    useAudioStudioStore.getState().setActiveWorkflow("narration");

    render(<AudioStudioPage />);

    fireEvent.click(
      screen.getByRole("button", { name: "Queue speech generation" }),
    );

    await waitFor(() => expect(generationMocks.mutateAsync).toHaveBeenCalled());
    expect(generationMocks.mutateAsync).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: "speech",
        provider: "tts",
        target_resource_kind: "section",
        target_resource_id: "section-1",
        target_revision_id: "revision-current",
      }),
    );
  });

  it("queues Podcast and Briefing using saved section identity and the selected voice", async () => {
    for (const workflow of ["podcast", "briefing"] as const) {
      setActiveProject({ workflow }); useAudioStudioStore.getState().setActiveWorkflow(workflow);
      const view = render(<AudioStudioPage />);
      fireEvent.click(screen.getByRole("button", { name: "Generate section 1" }));
      await waitFor(() => expect(generationMocks.mutateAsync).toHaveBeenCalledWith(expect.objectContaining({ kind: "speech", target_resource_id: "section-1", target_revision_id: "revision-current", options: expect.objectContaining({ model: "pocket_tts", voice: "reference", workflow }) })));
      view.unmount(); generationMocks.mutateAsync.mockClear();
    }
  });

  it("retains Podcast and Briefing drafts and refuses generation until an explicit save", () => {
    for (const workflow of ["podcast", "briefing"] as const) {
      useAudioStudioStore.getState().resetAudioStudio(); setActiveProject({ workflow }); useAudioStudioStore.getState().setActiveWorkflow(workflow);
      const view = render(<AudioStudioPage />);
      fireEvent.change(screen.getByLabelText("Section 1 text"), { target: { value: "New spoken text" } });
      fireEvent.change(screen.getByLabelText("Section 1 voice override"), { target: { value: "guest" } });
      expect(screen.getByRole("button", { name: "Generate section 1" })).toBeDisabled();
      expect(generationMocks.mutateAsync).not.toHaveBeenCalled();
      expect(useAudioStudioStore.getState().activeProject?.sections[0]).toMatchObject({ body_text: "New spoken text", settings: { voice: "guest" } });
      fireEvent.click(screen.getByRole("button", { name: "Save" }));
      expect(projectHookMocks.updateProject).toHaveBeenCalled();
      view.unmount();
    }
  });

  it("disables shared speech generation without a usable section target", () => {
    setActiveProject({ workflow: "narration", sections: [] });
    useAudioStudioStore.getState().setActiveWorkflow("narration");

    render(<AudioStudioPage />);

    expect(
      screen.getByRole("button", { name: "Queue speech generation" }),
    ).toBeDisabled();
  });
  it("B06 accepts an explicit model ID when an available provider omits enumeration", async () => {
    vi.mocked(getAudioStudioSpeechProviders).mockResolvedValue({ providers: { pocket_tts: { provider_name: "PocketTTS" } }, voices: {} });
    setActiveProject({ workflow: "narration", settings: { speech: { provider_hint: "pocket_tts", model: "", voice: "reference" } } });
    useAudioStudioStore.getState().setActiveWorkflow("narration");
    render(<AudioStudioPage />);
    const field = await screen.findByRole("textbox", { name: "Speech model", exact: true });
    fireEvent.change(field, { target: { value: "pocket_tts" } });
    expect(useAudioStudioStore.getState().activeProject?.settings.speech).toMatchObject({ model: "pocket_tts", voice: "" });
    expect(screen.getByText(/does not publish a model list/)).toBeInTheDocument();
  });

  it("B06 refuses New Project while a local draft needs saving", () => {
    setActiveProject({ workflow: "narration" }); useAudioStudioStore.getState().setActiveWorkflow("narration");
    useAudioStudioStore.getState().updateProjectLocal("project-1", { title: "Unsaved" });
    render(<AudioStudioPage />);
    expect(screen.getByRole("button", { name: "New Audio Studio project" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "New Audio Studio project" }));
    expect(projectHookMocks.createProject).not.toHaveBeenCalled();
    expect(useAudioStudioStore.getState().activeProject?.title).toBe("Unsaved");
  });

  it("B06 missing music provider cannot be replaced by speech generation", async () => {
    audioStudioServiceMocks.getProviders.mockResolvedValue({ providers: [{ provider_id: "tts", supported_kinds: ["speech"] }] });
    setActiveProject({ workflow: "music", settings: { music: { prompt: "A short instrumental cue" } } }); useAudioStudioStore.getState().setActiveWorkflow("music");
    render(<AudioStudioPage />);
    await waitFor(() => expect(screen.getByRole("button", { name: "Generate music" })).toBeDisabled());
    await waitFor(() => expect(screen.getByRole("button", { name: "Generate music" })).toHaveAttribute("title", "Configure a music service before generating Music."));
    expect(generationMocks.mutateAsync).not.toHaveBeenCalled();
  });

  it("B06 Music draft uses saved project settings and requires Save after edits", () => {
    setActiveProject({ workflow: "music", settings: { music: { prompt: "Saved cue", lyrics: "Saved lyrics", style: "ambient", provider: "ace_step", duration: 30 } } });
    useAudioStudioStore.getState().setActiveWorkflow("music");
    render(<AudioStudioPage />);
    expect(screen.getByLabelText("Prompt")).toHaveValue("Saved cue");
    fireEvent.change(screen.getByLabelText("Prompt"), { target: { value: "Edited cue" } });
    expect(useAudioStudioStore.getState().activeProject?.settings.music).toMatchObject({ prompt: "Edited cue", lyrics: "Saved lyrics", style: "ambient", duration: 30 });
    expect(screen.getByRole("button", { name: "Generate music" })).toBeDisabled();
    expect(generationMocks.mutateAsync).not.toHaveBeenCalled();
  });

});

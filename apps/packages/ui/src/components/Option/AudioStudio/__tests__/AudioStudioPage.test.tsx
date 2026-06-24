import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const generationMocks = vi.hoisted(() => ({
  mutateAsync: vi.fn(),
}));

const audioStudioServiceMocks = vi.hoisted(() => ({
  fetchAudioStudioArtifactBlob: vi.fn(),
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
  useCreateAudioStudioGeneration: () => ({
    mutateAsync: generationMocks.mutateAsync,
    isPending: false,
  }),
}));

vi.mock("@/hooks/useAudioStudioProjects", () => ({
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
    fetchAudioStudioArtifactBlob: (...args: unknown[]) =>
      audioStudioServiceMocks.fetchAudioStudioArtifactBlob(...args),
  };
});

import { AudioStudioPage } from "../AudioStudioPage";
import {
  useAudioStudioStore,
  type AudioStudioProject,
} from "@/store/audio-studio";
import type { AudioStudioArtifact } from "@/services/audio-studio";

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
      settings: {},
      ...overrides,
    } satisfies AudioStudioProject,
  ]);
  useAudioStudioStore.getState().setActiveProjectId("project-1");
};

describe("AudioStudioPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
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

  it("shows imported audiobook controls in Narration without the old top heading", () => {
    useAudioStudioStore.getState().setActiveWorkflow("narration");

    render(<AudioStudioPage />);

    expect(screen.getByText("Paste or type your content")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("tab", { name: "Chapters" }));
    expect(screen.getByText("Chapter List")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("tab", { name: "Voice" }));
    expect(screen.getByText("Voice Settings")).toBeInTheDocument();
    expect(
      screen.queryByRole("heading", { name: "Audiobook Studio" }),
    ).not.toBeInTheDocument();
  });

  it("surfaces Podcast and Briefing as production workflows", () => {
    useAudioStudioStore.getState().setActiveWorkflow("podcast");
    const { rerender } = render(<AudioStudioPage />);

    expect(screen.getByText("Podcast script")).toBeInTheDocument();
    expect(screen.getByText("Speakers")).toBeInTheDocument();

    useAudioStudioStore.getState().setActiveWorkflow("briefing");
    rerender(<AudioStudioPage />);

    expect(screen.getByText("Briefing outline")).toBeInTheDocument();
    expect(screen.getByText("Source notes")).toBeInTheDocument();
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

  it("keeps render and export controls disabled until the render controls slice", () => {
    setActiveProject({ workflow: "narration" });
    useAudioStudioStore.getState().setActiveWorkflow("narration");

    render(<AudioStudioPage />);

    expect(
      screen.getByRole("button", { name: "Create preview render" }),
    ).toBeDisabled();
    expect(
      screen.getByRole("button", { name: "Create export" }),
    ).toBeDisabled();
    expect(
      screen.getByRole("button", { name: "Create preview render" }),
    ).toHaveAttribute(
      "title",
      "Render/export controls need a ready timeline render controls slice.",
    );
    expect(
      screen.getByRole("button", { name: "Create export" }),
    ).toHaveAttribute(
      "title",
      "Render/export controls need a ready timeline render controls slice.",
    );
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
      ).toHaveBeenCalledWith("project-1", artifact),
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
      ).toHaveBeenCalledWith("project-1", artifact),
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

  it("does not fetch or preview a matching non-audio artifact", async () => {
    projectHookMocks.useArtifacts.mockReturnValue({
      data: [
        buildArtifact({
          artifact_type: "analysis",
          mime_type: "application/json",
          metadata: { filename: "host-analysis.json" },
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
      await screen.findByText("Selected clip artifact is not audio."),
    ).toBeInTheDocument();
    expect(
      audioStudioServiceMocks.fetchAudioStudioArtifactBlob,
    ).not.toHaveBeenCalled();
    expect(
      screen.queryByLabelText("Selected clip audio preview"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("link", { name: "Download selected clip audio" }),
    ).not.toBeInTheDocument();
    expect(document.body.innerHTML).not.toContain("/api/v1/audio-studio");
  });

  it("does not fetch or preview artifacts with unknown size", async () => {
    projectHookMocks.useArtifacts.mockReturnValue({
      data: [
        buildArtifact({
          size_bytes: null,
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
        "Artifact size is unavailable for browser preview.",
      ),
    ).toBeInTheDocument();
    expect(
      audioStudioServiceMocks.fetchAudioStudioArtifactBlob,
    ).not.toHaveBeenCalled();
    expect(
      screen.queryByLabelText("Selected clip audio preview"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("link", { name: "Download selected clip audio" }),
    ).not.toBeInTheDocument();
    expect(document.body.innerHTML).not.toContain("/api/v1/audio-studio");
  });

  it("disables preview and download for artifacts larger than the Blob threshold", async () => {
    projectHookMocks.useArtifacts.mockReturnValue({
      data: [
        buildArtifact({
          size_bytes: 25 * 1024 * 1024 + 1,
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
      await screen.findByText("Artifact is too large for browser preview."),
    ).toBeInTheDocument();
    expect(
      audioStudioServiceMocks.fetchAudioStudioArtifactBlob,
    ).not.toHaveBeenCalled();
    expect(
      screen.queryByLabelText("Selected clip audio preview"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("link", { name: "Download selected clip audio" }),
    ).not.toBeInTheDocument();
    expect(document.body.innerHTML).not.toContain("/api/v1/audio-studio");
    expect(document.body.innerHTML).not.toContain("blob:");
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

  it("supports timeline preview play pause and manual scrub state", () => {
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
      screen.getByRole("button", { name: "Play timeline preview" }),
    );
    expect(
      screen.getByRole("button", { name: "Pause timeline preview" }),
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
    fireEvent.click(screen.getByRole("button", { name: "Generate music" }));

    await waitFor(() => expect(generationMocks.mutateAsync).toHaveBeenCalled());
    expect(generationMocks.mutateAsync).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: "music",
        provider: "ace_step",
        target_resource_kind: "track",
        target_resource_id: "music-track-1",
        target_revision_id: "revision-current",
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

  it("queues shared music generation from the side panel", async () => {
    setActiveProject({ workflow: "music" });
    useAudioStudioStore.getState().setActiveWorkflow("music");

    render(<AudioStudioPage />);

    fireEvent.click(
      screen.getByRole("button", { name: "Queue music generation" }),
    );

    await waitFor(() => expect(generationMocks.mutateAsync).toHaveBeenCalled());
    expect(generationMocks.mutateAsync).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: "music",
        provider: "ace_step",
        target_resource_kind: "track",
        target_resource_id: "music-track-1",
        target_revision_id: "revision-current",
      }),
    );
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

  it("queues Podcast and Briefing inline speech actions from their saved draft sections", async () => {
    setActiveProject({ workflow: "podcast" });
    useAudioStudioStore.getState().setActiveWorkflow("podcast");
    const { rerender } = render(<AudioStudioPage />);

    fireEvent.change(screen.getByLabelText("Podcast script"), {
      target: { value: "Host: Welcome." },
    });
    fireEvent.click(
      screen.getByRole("button", { name: "Generate segment speech" }),
    );

    await waitFor(() => expect(generationMocks.mutateAsync).toHaveBeenCalled());
    expect(generationMocks.mutateAsync).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: "speech",
        provider: "tts",
        target_resource_kind: "section",
        target_resource_id: "section_podcast_script",
        target_revision_id: "revision-section-saved",
      }),
    );

    generationMocks.mutateAsync.mockClear();
    projectHookMocks.upsertSection.mockResolvedValueOnce({
      section_id: "section_briefing_outline",
      workflow: "briefing",
      title: "Briefing outline",
      body_text: "Briefing text",
      order_index: 0,
      current_revision_id: "revision-briefing-saved",
    });
    setActiveProject({ workflow: "briefing" });
    useAudioStudioStore.getState().setActiveWorkflow("briefing");
    rerender(<AudioStudioPage />);

    fireEvent.change(screen.getByLabelText("Briefing outline"), {
      target: { value: "Briefing text" },
    });
    fireEvent.click(
      screen.getByRole("button", { name: "Generate briefing sections" }),
    );

    await waitFor(() => expect(generationMocks.mutateAsync).toHaveBeenCalled());
    expect(generationMocks.mutateAsync).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: "speech",
        target_resource_id: "section_briefing_outline",
        target_revision_id: "revision-briefing-saved",
      }),
    );
  });

  it("saves Podcast and Briefing draft text before queuing inline speech generation", async () => {
    setActiveProject({ workflow: "podcast", sections: [] });
    useAudioStudioStore.getState().setActiveWorkflow("podcast");
    const { rerender } = render(<AudioStudioPage />);

    fireEvent.change(screen.getByLabelText("Podcast script"), {
      target: { value: "Host: Welcome.\nGuest: Good to be here." },
    });
    fireEvent.change(screen.getByLabelText("Host speaker"), {
      target: { value: "Ava" },
    });
    fireEvent.change(screen.getByLabelText("Guest speaker"), {
      target: { value: "Noah" },
    });
    fireEvent.click(
      screen.getByRole("button", { name: "Generate segment speech" }),
    );

    await waitFor(() =>
      expect(projectHookMocks.upsertSection).toHaveBeenCalled(),
    );
    expect(projectHookMocks.upsertSection).toHaveBeenCalledWith({
      sectionId: "section_podcast_script",
      payload: {
        base_revision_id: "revision-current",
        title: "Podcast script",
        body_text: "Host: Welcome.\nGuest: Good to be here.",
        order_index: 0,
        settings: {
          hostSpeaker: "Ava",
          guestSpeaker: "Noah",
        },
      },
    });
    await waitFor(() => expect(generationMocks.mutateAsync).toHaveBeenCalled());
    expect(generationMocks.mutateAsync).toHaveBeenCalledWith(
      expect.objectContaining({
        target_resource_id: "section_podcast_script",
        target_revision_id: "revision-section-saved",
      }),
    );

    generationMocks.mutateAsync.mockClear();
    projectHookMocks.upsertSection.mockClear();
    projectHookMocks.upsertSection.mockResolvedValueOnce({
      section_id: "section_briefing_outline",
      workflow: "briefing",
      title: "Briefing outline",
      body_text: "Top story and implications.",
      order_index: 0,
      current_revision_id: "revision-briefing-saved",
    });
    setActiveProject({ workflow: "briefing", sections: [] });
    useAudioStudioStore.getState().setActiveWorkflow("briefing");
    rerender(<AudioStudioPage />);

    fireEvent.change(screen.getByLabelText("Briefing outline"), {
      target: { value: "Top story and implications." },
    });
    fireEvent.change(screen.getByLabelText("Source notes"), {
      target: { value: "Source note A" },
    });
    fireEvent.click(
      screen.getByRole("button", { name: "Generate briefing sections" }),
    );

    await waitFor(() =>
      expect(projectHookMocks.upsertSection).toHaveBeenCalled(),
    );
    expect(projectHookMocks.upsertSection).toHaveBeenCalledWith({
      sectionId: "section_briefing_outline",
      payload: {
        base_revision_id: "revision-current",
        title: "Briefing outline",
        body_text: "Top story and implications.",
        order_index: 0,
        settings: {
          sourceNotes: "Source note A",
        },
      },
    });
    await waitFor(() => expect(generationMocks.mutateAsync).toHaveBeenCalled());
    expect(generationMocks.mutateAsync).toHaveBeenCalledWith(
      expect.objectContaining({
        target_resource_id: "section_briefing_outline",
        target_revision_id: "revision-briefing-saved",
      }),
    );
  });

  it("disables shared speech generation without a usable section target", () => {
    setActiveProject({ workflow: "narration", sections: [] });
    useAudioStudioStore.getState().setActiveWorkflow("narration");

    render(<AudioStudioPage />);

    expect(
      screen.getByRole("button", { name: "Queue speech generation" }),
    ).toBeDisabled();
  });
});

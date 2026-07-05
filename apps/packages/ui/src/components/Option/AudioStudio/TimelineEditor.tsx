import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Button, Checkbox, Divider, InputNumber, Typography } from "antd";
import { Download, Pause, Play, Save } from "lucide-react";
import { useUpsertAudioStudioClip } from "@/hooks/useAudioStudioProjects";
import {
  fetchAudioStudioArtifactBlob,
  mintAudioStudioArtifactMediaTicket,
  type AudioStudioArtifact,
} from "@/services/audio-studio";
import {
  useAudioStudioStore,
  type AudioStudioClip,
  type AudioStudioClipType,
  type AudioStudioProject,
  type AudioStudioTrack,
} from "@/store/audio-studio";

const { Text } = Typography;

type ClipDraft = {
  startSeconds: number;
  durationSeconds: number;
  volumePercent: number;
  fadeInSeconds: number;
  fadeOutSeconds: number;
  muted: boolean;
};

const DEFAULT_TIMELINE_MS = 60_000;
const PREVIEW_TICK_MS = 500;
const MAX_BLOB_PREVIEW_BYTES = 25 * 1024 * 1024;
const EMPTY_CLIPS: AudioStudioClip[] = [];
const EMPTY_TRACKS: AudioStudioTrack[] = [];
const EMPTY_ARTIFACTS: AudioStudioArtifact[] = [];

type TimelineEditorProps = {
  artifacts?: AudioStudioArtifact[];
};

const clamp = (value: number, min: number, max: number) =>
  Math.min(Math.max(value, min), max);

const msToSeconds = (value?: number | null) =>
  Number(((value ?? 0) / 1000).toFixed(3));
const secondsToMs = (value: number) => Math.round(value * 1000);
const formatSeconds = (valueMs: number) => `${(valueMs / 1000).toFixed(1)}s`;

const MIME_EXTENSION_BY_TYPE: Record<string, string> = {
  "audio/aac": "aac",
  "audio/flac": "flac",
  "audio/m4a": "m4a",
  "audio/mp4": "m4a",
  "audio/mpeg": "mp3",
  "audio/ogg": "ogg",
  "audio/wav": "wav",
  "audio/webm": "webm",
  "audio/x-m4a": "m4a",
  "audio/x-wav": "wav",
};

const AUDIO_ARTIFACT_TYPES = new Set([
  "audio",
  "clip_audio",
  "generated_audio",
  "tts_audio",
  "normalized_audio",
  "reference_audio",
  "preview_mix",
  "final_mix",
  "alternate_format",
]);

const readStringSetting = (
  settings: Record<string, unknown> | undefined,
  key: string,
) => {
  const value = settings?.[key];
  return typeof value === "string" && value.trim().length > 0
    ? value
    : undefined;
};

const readStringMetadata = (
  metadata: Record<string, unknown> | undefined,
  key: string,
) => {
  const value = metadata?.[key];
  return typeof value === "string" && value.trim().length > 0
    ? value
    : undefined;
};

const getTrackById = (
  tracks: AudioStudioTrack[],
  trackId: string,
): AudioStudioTrack | undefined =>
  tracks.find((track) => track.track_id === trackId);

const getClipTitle = (clip: AudioStudioClip) =>
  clip.title ?? readStringSetting(clip.settings, "title") ?? clip.clip_id;

const sanitizeDownloadFilename = (value: string) => {
  const trimmed = value.trim().replace(/[\\/]+/g, "-");
  const safe = trimmed
    .replace(/[^a-zA-Z0-9._-]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 96);
  return safe.length > 0 ? safe : "audio-studio-artifact";
};

const getArtifactExtension = (artifact: AudioStudioArtifact) => {
  const mimeType = artifact.mime_type?.toLowerCase();
  return mimeType ? (MIME_EXTENSION_BY_TYPE[mimeType] ?? "bin") : "bin";
};

const isAudioArtifact = (artifact: AudioStudioArtifact | null) => {
  if (!artifact) return false;

  const artifactType = artifact.artifact_type?.trim().toLowerCase();
  const mimeType = artifact.mime_type?.trim().toLowerCase();

  return Boolean(
    artifactType &&
    AUDIO_ARTIFACT_TYPES.has(artifactType) &&
    mimeType?.startsWith("audio/"),
  );
};

const getArtifactDownloadFilename = (
  artifact: AudioStudioArtifact,
  clip: AudioStudioClip,
) => {
  const metadataFilename =
    readStringMetadata(artifact.metadata, "filename") ??
    readStringMetadata(artifact.metadata, "file_name");
  if (metadataFilename) {
    return sanitizeDownloadFilename(metadataFilename);
  }

  const extension = getArtifactExtension(artifact);
  return sanitizeDownloadFilename(
    `${getClipTitle(clip)}-${artifact.artifact_id}.${extension}`,
  );
};

const getClipType = (
  clip: AudioStudioClip,
  track?: AudioStudioTrack,
): AudioStudioClipType =>
  clip.clip_type ??
  (readStringSetting(clip.settings, "clip_type") as
    | AudioStudioClipType
    | undefined) ??
  (track?.kind as AudioStudioClipType | undefined) ??
  "speech";

const getClipDraft = (clip: AudioStudioClip): ClipDraft => ({
  startSeconds: msToSeconds(clip.start_ms),
  durationSeconds: msToSeconds(clip.duration_ms ?? 30_000),
  volumePercent: Math.round((clip.volume ?? 1) * 100),
  fadeInSeconds: msToSeconds(clip.fade_in_ms),
  fadeOutSeconds: msToSeconds(clip.fade_out_ms),
  muted: Boolean(clip.muted),
});

const getTimelineEndMs = (clips: AudioStudioClip[]) =>
  clips.length === 0
    ? DEFAULT_TIMELINE_MS
    : Math.max(
        1_000,
        ...clips.map((clip) => clip.start_ms + (clip.duration_ms ?? 30_000)),
      );

const getRevision = (project: AudioStudioProject) =>
  project.current_revision_id ?? project.revision_id;

const triggerBrowserDownload = (url: string, filename?: string | null) => {
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.rel = "noreferrer";
  anchor.referrerPolicy = "no-referrer";
  if (filename) {
    anchor.download = filename;
  }
  anchor.style.display = "none";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
};

export const TimelineEditor: React.FC<TimelineEditorProps> = ({
  artifacts = EMPTY_ARTIFACTS,
}) => {
  const activeProject = useAudioStudioStore((state) => state.activeProject);
  const clips = activeProject?.clips ?? EMPTY_CLIPS;
  const tracks = activeProject?.tracks ?? EMPTY_TRACKS;
  const [selectedClipId, setSelectedClipId] = useState<string | null>(null);
  const [draft, setDraft] = useState<ClipDraft | null>(null);
  const [draggingClipId, setDraggingClipId] = useState<string | null>(null);
  const [playheadMs, setPlayheadMs] = useState(0);
  const [isPreviewing, setIsPreviewing] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [isPreviewLoading, setIsPreviewLoading] = useState(false);
  const [downloadError, setDownloadError] = useState<string | null>(null);
  const [isDownloadLoading, setIsDownloadLoading] = useState(false);
  const previewStateKeyRef = useRef<string | null>(null);
  const audioElementRef = useRef<HTMLAudioElement | null>(null);
  const setAudioElementRef = useCallback((element: HTMLAudioElement | null) => {
    audioElementRef.current = element;
    element?.setAttribute("referrerpolicy", "no-referrer");
  }, []);
  const ticketRetryKeyRef = useRef<string | null>(null);
  const downloadStateKeyRef = useRef<string | null>(null);
  const upsertClip = useUpsertAudioStudioClip(
    activeProject?.project_id ?? null,
  );

  const artifactById = useMemo(
    () =>
      new Map(
        artifacts.map((artifact) => [artifact.artifact_id, artifact] as const),
      ),
    [artifacts],
  );
  const timelineEndMs = useMemo(() => getTimelineEndMs(clips), [clips]);
  const selectedClip =
    clips.find((clip) => clip.clip_id === selectedClipId) ?? clips[0] ?? null;
  const selectedArtifactId = selectedClip?.artifact_id ?? null;
  const selectedArtifact = selectedArtifactId
    ? (artifactById.get(selectedArtifactId) ?? null)
    : null;
  const selectedArtifactSizeKnown =
    typeof selectedArtifact?.size_bytes === "number" &&
    selectedArtifact.size_bytes >= 0;
  const selectedArtifactIsAudio = isAudioArtifact(selectedArtifact);
  const selectedArtifactCanUseBlob =
    selectedArtifactIsAudio &&
    selectedArtifactSizeKnown &&
    selectedArtifact.size_bytes <= MAX_BLOB_PREVIEW_BYTES;
  const selectedArtifactShouldUseTicketPlayback =
    selectedArtifactIsAudio && !selectedArtifactCanUseBlob;
  const selectedArtifactCanDownload = Boolean(selectedArtifact);
  const selectedPreviewKey =
    activeProject && selectedClip && selectedArtifactId
      ? `${activeProject.project_id}:${selectedClip.clip_id}:${selectedArtifactId}`
      : null;
  const visiblePreviewUrl =
    previewStateKeyRef.current === selectedPreviewKey ? previewUrl : null;
  const visiblePreviewError =
    previewStateKeyRef.current === selectedPreviewKey ? previewError : null;
  const visiblePreviewLoading =
    previewStateKeyRef.current === selectedPreviewKey && isPreviewLoading;
  const visibleDownloadError =
    downloadStateKeyRef.current === selectedPreviewKey ? downloadError : null;
  const visibleDownloadLoading =
    downloadStateKeyRef.current === selectedPreviewKey && isDownloadLoading;
  const downloadFilename =
    selectedClip && selectedArtifact
      ? getArtifactDownloadFilename(selectedArtifact, selectedClip)
      : null;

  useEffect(() => {
    if (!selectedClip) {
      setSelectedClipId(null);
      setDraft(null);
      return;
    }
    if (selectedClip.clip_id !== selectedClipId) {
      setSelectedClipId(selectedClip.clip_id);
    }
    setDraft(getClipDraft(selectedClip));
  }, [selectedClip, selectedClipId]);

  useEffect(() => {
    downloadStateKeyRef.current = selectedPreviewKey;
    setDownloadError(null);
    setIsDownloadLoading(false);
  }, [selectedPreviewKey]);

  useEffect(() => {
    let cancelled = false;
    let objectUrl: string | null = null;

    setPreviewUrl(null);
    setPreviewError(null);
    setIsPreviewLoading(false);

    if (
      !activeProject ||
      !selectedClip ||
      !selectedArtifactId ||
      !selectedArtifact ||
      !selectedArtifactIsAudio ||
      !selectedPreviewKey
    ) {
      previewStateKeyRef.current = null;
      ticketRetryKeyRef.current = null;
      return () => {
        cancelled = true;
      };
    }

    previewStateKeyRef.current = selectedPreviewKey;
    setIsPreviewLoading(true);

    const loadPreview = selectedArtifactCanUseBlob
      ? fetchAudioStudioArtifactBlob(
          activeProject.project_id,
          selectedArtifact,
        ).then((blob) => {
          if (
            cancelled ||
            previewStateKeyRef.current !== selectedPreviewKey
          ) {
            return null;
          }
          objectUrl = URL.createObjectURL(blob);
          return objectUrl;
        })
      : mintAudioStudioArtifactMediaTicket(
          activeProject.project_id,
          selectedArtifact.artifact_id,
          "playback",
        ).then((ticket) => ticket.ticket_url);

    void loadPreview
      .then((url) => {
        if (!url || cancelled) return;
        previewStateKeyRef.current = selectedPreviewKey;
        ticketRetryKeyRef.current = null;
        setPreviewUrl(url);
      })
      .catch(() => {
        if (!cancelled) {
          previewStateKeyRef.current = selectedPreviewKey;
          setPreviewError("Preview unavailable");
        }
      })
      .finally(() => {
        if (!cancelled) {
          setIsPreviewLoading(false);
        }
      });

    return () => {
      cancelled = true;
      if (objectUrl?.startsWith("blob:")) {
        URL.revokeObjectURL(objectUrl);
      }
    };
  }, [
    activeProject?.project_id,
    selectedArtifact,
    selectedArtifactCanUseBlob,
    selectedArtifactId,
    selectedArtifactIsAudio,
    selectedClip?.clip_id,
    selectedPreviewKey,
  ]);

  useEffect(() => {
    if (!isPreviewing) return;

    const interval = window.setInterval(() => {
      setPlayheadMs((current) => {
        const next = current + PREVIEW_TICK_MS;
        if (next >= timelineEndMs) {
          setIsPreviewing(false);
          return timelineEndMs;
        }
        return next;
      });
    }, PREVIEW_TICK_MS);

    return () => window.clearInterval(interval);
  }, [isPreviewing, timelineEndMs]);

  useEffect(() => {
    setPlayheadMs((current) => clamp(current, 0, timelineEndMs));
  }, [timelineEndMs]);

  const updateDraft = (updates: Partial<ClipDraft>) => {
    setDraft((current) => (current ? { ...current, ...updates } : current));
  };

  const updateClipStartFromPointer = (
    event: React.PointerEvent<HTMLButtonElement>,
  ) => {
    const lane = event.currentTarget.parentElement;
    if (!lane) return;
    const bounds = lane.getBoundingClientRect();
    if (bounds.width <= 0) return;
    const ratio = clamp((event.clientX - bounds.left) / bounds.width, 0, 1);
    const snappedStartMs = Math.round((ratio * timelineEndMs) / 500) * 500;
    updateDraft({ startSeconds: msToSeconds(snappedStartMs) });
  };

  const handleSaveClip = async () => {
    if (!activeProject || !selectedClip || !draft) return;
    const baseRevisionId = getRevision(activeProject);
    if (!baseRevisionId) return;

    const track = getTrackById(tracks, selectedClip.track_id);

    await upsertClip.mutateAsync({
      clipId: selectedClip.clip_id,
      payload: {
        base_revision_id: baseRevisionId,
        track_id: selectedClip.track_id,
        section_id: selectedClip.section_id,
        title: getClipTitle(selectedClip),
        clip_type: getClipType(selectedClip, track),
        artifact_id: selectedClip.artifact_id,
        start_ms: secondsToMs(draft.startSeconds),
        duration_ms: secondsToMs(draft.durationSeconds),
        volume: draft.volumePercent / 100,
        fade_in_ms: secondsToMs(draft.fadeInSeconds),
        fade_out_ms: secondsToMs(draft.fadeOutSeconds),
        muted: draft.muted,
        settings: selectedClip.settings ?? {},
      },
    });
  };

  const handleTicketPlaybackError = () => {
    if (
      !activeProject ||
      !selectedArtifact ||
      !selectedArtifactShouldUseTicketPlayback ||
      !selectedPreviewKey ||
      ticketRetryKeyRef.current === selectedPreviewKey
    ) {
      return;
    }
    ticketRetryKeyRef.current = selectedPreviewKey;
    const currentTime = audioElementRef.current?.currentTime ?? 0;
    setIsPreviewLoading(true);
    void mintAudioStudioArtifactMediaTicket(
      activeProject.project_id,
      selectedArtifact.artifact_id,
      "playback",
    )
      .then((ticket) => {
        if (previewStateKeyRef.current !== selectedPreviewKey) return;
        const ticketUrl = ticket.ticket_url;
        setPreviewUrl(ticketUrl);
        window.setTimeout(() => {
          const audioElement = audioElementRef.current;
          if (
            previewStateKeyRef.current === selectedPreviewKey &&
            audioElement?.src === ticketUrl &&
            Number.isFinite(currentTime)
          ) {
            audioElement.currentTime = currentTime;
          }
        }, 0);
      })
      .catch(() => {
        if (previewStateKeyRef.current === selectedPreviewKey) {
          setPreviewError("Preview unavailable");
        }
      })
      .finally(() => {
        if (previewStateKeyRef.current === selectedPreviewKey) {
          setIsPreviewLoading(false);
        }
      });
  };

  const handleDownloadArtifact = async () => {
    if (!activeProject || !selectedArtifact || !selectedPreviewKey) return;
    const downloadStateKey = selectedPreviewKey;
    downloadStateKeyRef.current = downloadStateKey;
    setDownloadError(null);
    setIsDownloadLoading(true);
    try {
      const ticket = await mintAudioStudioArtifactMediaTicket(
        activeProject.project_id,
        selectedArtifact.artifact_id,
        "download",
      );
      if (downloadStateKeyRef.current !== downloadStateKey) {
        return;
      }
      triggerBrowserDownload(ticket.ticket_url, downloadFilename);
    } catch {
      if (downloadStateKeyRef.current === downloadStateKey) {
        setDownloadError("Download unavailable");
      }
    } finally {
      if (downloadStateKeyRef.current === downloadStateKey) {
        setIsDownloadLoading(false);
      }
    }
  };

  const hasUsableRevision = Boolean(
    activeProject && getRevision(activeProject),
  );
  const canSaveClip = Boolean(selectedClip && draft && hasUsableRevision);
  const ticketDownloadButton =
    selectedArtifactCanDownload && !selectedArtifactCanUseBlob ? (
      <Button
        size="small"
        icon={<Download className="h-4 w-4" />}
        onClick={handleDownloadArtifact}
        loading={visibleDownloadLoading}
      >
        {selectedArtifactIsAudio
          ? "Download selected clip audio"
          : "Download selected artifact"}
      </Button>
    ) : null;

  return (
    <section
      aria-labelledby="audio-studio-timeline-heading"
      className="rounded-md border border-border bg-surface p-3"
    >
      <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <Typography.Title
            id="audio-studio-timeline-heading"
            level={2}
            className="!mb-1 !text-lg"
          >
            Timeline
          </Typography.Title>
          <Text type="secondary" className="block text-xs">
            Server-backed track and clip arrangement
          </Text>
        </div>
        <div className="flex items-center gap-2">
          <Button
            icon={
              isPreviewing ? (
                <Pause className="h-4 w-4" />
              ) : (
                <Play className="h-4 w-4" />
              )
            }
            onClick={() => setIsPreviewing((current) => !current)}
            disabled={clips.length === 0}
          >
            {isPreviewing ? "Pause timeline preview" : "Play timeline preview"}
          </Button>
          <Text className="text-xs tabular-nums">
            {formatSeconds(playheadMs)} / {formatSeconds(timelineEndMs)}
          </Text>
        </div>
      </div>

      <label className="mt-3 block">
        <span className="sr-only">Timeline playhead seconds</span>
        <input
          aria-label="Timeline playhead seconds"
          className="h-2 w-full accent-primary"
          type="range"
          min={0}
          max={Math.round(timelineEndMs / 1000)}
          step={0.5}
          value={Number((playheadMs / 1000).toFixed(1))}
          onChange={(event) => {
            setIsPreviewing(false);
            setPlayheadMs(secondsToMs(Number(event.target.value)));
          }}
        />
      </label>

      <div className="mt-3 space-y-2">
        {tracks.length === 0 ? (
          <div className="rounded-md border border-dashed border-border p-4">
            <Text type="secondary">No tracks yet.</Text>
          </div>
        ) : (
          tracks
            .slice()
            .sort((a, b) => a.order - b.order)
            .map((track) => {
              const trackClips = clips.filter(
                (clip) => clip.track_id === track.track_id,
              );
              return (
                <div
                  key={track.track_id}
                  role="group"
                  aria-label={`${track.name} timeline track`}
                  className="grid gap-2 md:grid-cols-[160px_minmax(0,1fr)]"
                >
                  <div className="rounded-md border border-border bg-background px-3 py-2">
                    <Text strong className="block text-sm">
                      {track.name}
                    </Text>
                    <Text type="secondary" className="block text-xs">
                      {track.kind}
                    </Text>
                  </div>
                  <div className="relative min-h-[56px] overflow-hidden rounded-md border border-border bg-background">
                    <div
                      className="absolute bottom-0 top-0 w-px bg-primary/70"
                      style={{
                        left: `${clamp((playheadMs / timelineEndMs) * 100, 0, 100)}%`,
                      }}
                    />
                    {trackClips.length === 0 ? (
                      <Text
                        type="secondary"
                        className="block px-3 py-4 text-xs"
                      >
                        No clips
                      </Text>
                    ) : (
                      trackClips.map((clip) => {
                        const width = clamp(
                          ((clip.duration_ms ?? 30_000) / timelineEndMs) * 100,
                          8,
                          100,
                        );
                        const displayStartMs =
                          selectedClip?.clip_id === clip.clip_id && draft
                            ? secondsToMs(draft.startSeconds)
                            : clip.start_ms;
                        const left = clamp(
                          (displayStartMs / timelineEndMs) * 100,
                          0,
                          96,
                        );
                        const title = getClipTitle(clip);
                        const selected = selectedClip?.clip_id === clip.clip_id;
                        return (
                          <button
                            key={clip.clip_id}
                            type="button"
                            className={`absolute top-2 h-10 rounded-md border px-2 text-left text-xs transition ${
                              selected
                                ? "border-primary bg-primary/10"
                                : "border-border bg-surface hover:border-primary/60"
                            }`}
                            style={{
                              left: `${left}%`,
                              width: `${Math.min(width, 100 - left)}%`,
                            }}
                            onClick={() => setSelectedClipId(clip.clip_id)}
                            onPointerDown={(event) => {
                              setSelectedClipId(clip.clip_id);
                              setDraggingClipId(clip.clip_id);
                              event.currentTarget.setPointerCapture?.(
                                event.pointerId,
                              );
                              updateClipStartFromPointer(event);
                            }}
                            onPointerMove={(event) => {
                              if (draggingClipId === clip.clip_id) {
                                updateClipStartFromPointer(event);
                              }
                            }}
                            onPointerUp={(event) => {
                              if (draggingClipId === clip.clip_id) {
                                updateClipStartFromPointer(event);
                              }
                              event.currentTarget.releasePointerCapture?.(
                                event.pointerId,
                              );
                              setDraggingClipId(null);
                            }}
                            onPointerCancel={() => setDraggingClipId(null)}
                          >
                            <span className="block truncate font-medium">
                              {title}
                            </span>
                            <span className="block truncate text-text-muted">
                              Starts {formatSeconds(displayStartMs)}
                            </span>
                          </button>
                        );
                      })
                    )}
                  </div>
                </div>
              );
            })
        )}
      </div>

      {selectedClip && draft ? (
        <>
          <Divider className="my-3" />
          <div
            aria-label="Selected clip artifact preview"
            className="mb-3 rounded-md border border-border bg-background p-3"
          >
            <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <Text strong className="block text-sm">
                  Selected clip audio
                </Text>
                <Text type="secondary" className="block text-xs">
                  {getClipTitle(selectedClip)}
                </Text>
              </div>
              {!selectedArtifactId ? (
                <Text type="secondary" className="text-xs">
                  No audio artifact is attached to this clip.
                </Text>
              ) : !selectedArtifact ? (
                <Text type="secondary" className="text-xs">
                  Selected clip artifact metadata is unavailable.
                </Text>
              ) : !selectedArtifactIsAudio ? (
                <div className="flex flex-col items-start gap-2 sm:items-end">
                  <Text type="secondary" className="text-xs">
                    Selected clip artifact is download-only.
                  </Text>
                  {ticketDownloadButton}
                </div>
              ) : visiblePreviewError ? (
                <div className="flex flex-col items-start gap-2 sm:items-end">
                  <Text type="danger" className="text-xs">
                    {visiblePreviewError}
                  </Text>
                  {ticketDownloadButton}
                </div>
              ) : visiblePreviewUrl && selectedArtifactCanUseBlob ? (
                <a
                  className="text-xs font-medium text-primary hover:underline"
                  href={visiblePreviewUrl}
                  download={downloadFilename ?? undefined}
                >
                  Download selected clip audio
                </a>
              ) : selectedArtifactCanDownload && !selectedArtifactCanUseBlob ? (
                ticketDownloadButton
              ) : visiblePreviewLoading ? (
                <Text type="secondary" className="text-xs">
                  Loading audio preview...
                </Text>
              ) : (
                <Text type="secondary" className="text-xs">
                  Audio preview unavailable.
                </Text>
              )}
            </div>
            {visibleDownloadError ? (
              <Text type="danger" className="mt-2 block text-xs">
                {visibleDownloadError}
              </Text>
            ) : null}
            {visiblePreviewUrl ? (
              <audio
                ref={setAudioElementRef}
                aria-label="Selected clip audio preview"
                className="mt-3 w-full"
                controls
                src={visiblePreviewUrl}
                onError={handleTicketPlaybackError}
              />
            ) : null}
          </div>
          <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_auto] lg:items-end">
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
              <label className="space-y-1">
                <span className="block text-xs font-medium text-text-muted">
                  Start
                </span>
                <InputNumber
                  aria-label="Start seconds"
                  min={0}
                  step={0.5}
                  value={draft.startSeconds}
                  onChange={(value) =>
                    updateDraft({ startSeconds: Number(value ?? 0) })
                  }
                />
              </label>
              <label className="space-y-1">
                <span className="block text-xs font-medium text-text-muted">
                  Duration
                </span>
                <InputNumber
                  aria-label="Duration seconds"
                  min={0.5}
                  step={0.5}
                  value={draft.durationSeconds}
                  onChange={(value) =>
                    updateDraft({ durationSeconds: Number(value ?? 0.5) })
                  }
                />
              </label>
              <label className="space-y-1">
                <span className="block text-xs font-medium text-text-muted">
                  Volume
                </span>
                <InputNumber
                  aria-label="Volume percent"
                  min={0}
                  max={200}
                  step={5}
                  value={draft.volumePercent}
                  onChange={(value) =>
                    updateDraft({ volumePercent: Number(value ?? 0) })
                  }
                />
              </label>
              <label className="space-y-1">
                <span className="block text-xs font-medium text-text-muted">
                  Fade in
                </span>
                <InputNumber
                  aria-label="Fade in seconds"
                  min={0}
                  step={0.25}
                  value={draft.fadeInSeconds}
                  onChange={(value) =>
                    updateDraft({ fadeInSeconds: Number(value ?? 0) })
                  }
                />
              </label>
              <label className="space-y-1">
                <span className="block text-xs font-medium text-text-muted">
                  Fade out
                </span>
                <InputNumber
                  aria-label="Fade out seconds"
                  min={0}
                  step={0.25}
                  value={draft.fadeOutSeconds}
                  onChange={(value) =>
                    updateDraft({ fadeOutSeconds: Number(value ?? 0) })
                  }
                />
              </label>
            </div>
            <div className="flex items-center gap-3">
              <Checkbox
                aria-label="Mute clip"
                checked={draft.muted}
                onChange={(event) =>
                  updateDraft({ muted: event.target.checked })
                }
              >
                Mute
              </Checkbox>
              <Button
                type="primary"
                icon={<Save className="h-4 w-4" />}
                disabled={!canSaveClip}
                loading={upsertClip.isPending}
                onClick={handleSaveClip}
              >
                Save clip edits
              </Button>
            </div>
          </div>
        </>
      ) : null}
    </section>
  );
};

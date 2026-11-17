import React, { useEffect } from "react";
import { Button, Typography } from "antd";
import { PageShell } from "@/components/Common/PageShell";
import { useLocation, useNavigate } from "react-router-dom";
import {
  useAudioStudioArtifacts,
  useAudioStudioProjects,
  useAudioStudioProjectDetail,
} from "@/hooks/useAudioStudioProjects";
import {
  AUDIO_STUDIO_WORKFLOWS,
  useAudioStudioStore,
  type AudioStudioWorkflow,
} from "@/store/audio-studio";
import { BriefingWorkflow } from "./BriefingWorkflow";
import { GenerationPanel } from "./GenerationPanel";
import { MigrationBanner } from "./MigrationBanner";
import { MusicWorkflow } from "./MusicWorkflow";
import { NarrationWorkflow } from "./NarrationWorkflow";
import { PodcastWorkflow } from "./PodcastWorkflow";
import { ProjectHeader } from "./ProjectHeader";
import { ProjectSidebar } from "./ProjectSidebar";
import { RenderExportPanel } from "./RenderExportPanel";
import { TimelineEditor } from "./TimelineEditor";
import { WorkflowSwitcher } from "./WorkflowSwitcher";

import { AudioStudioSession } from "./AudioStudioSession";
import { RouteLeavePrompt } from "@/entries/shared/route-leave-prompt";

const { Text } = Typography;

const WORKFLOW_IDS = new Set<AudioStudioWorkflow>(
  AUDIO_STUDIO_WORKFLOWS.map((workflow) => workflow.id),
);

const getWorkflowFromSearch = (search: string): AudioStudioWorkflow | null => {
  const workflow = new URLSearchParams(search).get("workflow");
  return WORKFLOW_IDS.has(workflow as AudioStudioWorkflow)
    ? (workflow as AudioStudioWorkflow)
    : null;
};

const WorkflowEditor: React.FC<{ workflow: AudioStudioWorkflow }> = ({
  workflow,
}) => {
  if (workflow === "podcast") return <PodcastWorkflow />;
  if (workflow === "briefing") return <BriefingWorkflow />;
  if (workflow === "music") return <MusicWorkflow />;
  return <NarrationWorkflow />;
};

const AudioStudioContent: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const requestedProject = new URLSearchParams(location.search).get("project");
  const activeWorkflow = useAudioStudioStore((state) => state.activeWorkflow);
  const activeProject = useAudioStudioStore((state) => state.activeProject);
  const projectsQuery = useAudioStudioProjects({
    workflow: activeWorkflow,
    includeArchived: false,
  });
  const detailQuery = useAudioStudioProjectDetail(requestedProject ?? activeProject?.project_id ?? null);
  const hasDrafts = useAudioStudioStore((state) => Object.keys(state.dirtyProjectIds).length > 0);
  const dirty = useAudioStudioStore((state) => activeProject ? Boolean(state.dirtyProjectIds[activeProject.project_id]) : false);
  const conflict = useAudioStudioStore((state) => activeProject ? state.revisionConflicts[activeProject.project_id] : undefined);
  const artifactsQuery = useAudioStudioArtifacts(
    activeProject?.project_id ?? null,
  );

  useEffect(() => {
    const workflow = getWorkflowFromSearch(location.search);
    const store = useAudioStudioStore.getState();
    if (workflow) store.setActiveWorkflow(workflow);
    if (requestedProject && store.projects.some((project) => project.project_id === requestedProject)) store.setActiveProjectId(requestedProject);
  }, [location.search, requestedProject]);

  useEffect(() => {
    if (requestedProject && detailQuery.data?.project_id === requestedProject) {
      const store = useAudioStudioStore.getState();
      store.setActiveWorkflow(detailQuery.data.workflow);
      store.setActiveProjectId(requestedProject);
    }
  }, [requestedProject, detailQuery.data]);

  return (
    <PageShell maxWidthClassName="max-w-7xl" className="py-4">
      <div className="space-y-4">
        <RouteLeavePrompt when={hasDrafts} message="Leave Audio Studio with unsaved changes?" />
        <ProjectHeader />
        {dirty ? <p role="status">Unsaved changes. Save before generating or leaving.</p> : null}
        {conflict ? <p role="alert">The server revision changed; your draft is retained. <Button onClick={() => {
          if (!window.confirm("Discard your local draft and load the server revision?")) return;
          const store = useAudioStudioStore.getState();
          store.markProjectClean(conflict.incoming.project_id); store.upsertProjectFromServer(conflict.incoming);
        }}>Discard local edits and reload</Button></p> : null}
        <WorkflowSwitcher
          activeWorkflow={activeWorkflow}
          onChange={(workflow) => { if (!hasDrafts) useAudioStudioStore.getState().setActiveWorkflow(workflow); navigate(`?workflow=${workflow}`); }}
        />
        {projectsQuery.isLoading ? (
          <Text type="secondary" className="block text-xs">
            Loading Audio Studio projects...
          </Text>
        ) : null}
        {projectsQuery.isError ? (
          <Text type="danger" className="block text-xs">
            Audio Studio projects could not load.
          </Text>
        ) : null}
        <MigrationBanner />
        <div className="grid gap-4 lg:grid-cols-[280px_minmax(0,1fr)_320px]">
          <ProjectSidebar />
          <div
            id="audio-studio-workflow-panel"
            role="tabpanel"
            aria-labelledby={`audio-studio-workflow-tab-${activeWorkflow}`}
          >
            {detailQuery.isError ? <p role="alert">Project details could not load. <Button onClick={() => void detailQuery.refetch()}>Reload project</Button></p> : detailQuery.isLoading ? <p role="status">Loading saved sections…</p> : <WorkflowEditor key={activeProject?.project_id ?? activeWorkflow} workflow={activeWorkflow} />}
          </div>
          <div className="space-y-4">
            <GenerationPanel />
            <RenderExportPanel />
          </div>
        </div>
        {activeProject?.workflow === 'music' || Boolean(activeProject?.clips?.length) ? <TimelineEditor artifacts={artifactsQuery.data ?? []} /> : null}
      </div>
    </PageShell>
  );
};

export const AudioStudioPage: React.FC = () => <AudioStudioSession><AudioStudioContent /></AudioStudioSession>;

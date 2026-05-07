import { beforeEach, describe, expect, it } from "vitest"
import {
  DEFAULT_AUDIO_SETTINGS,
  DEFAULT_WORKSPACE_NOTE
} from "@/types/workspace"
import { WORKSPACE_STORAGE_KEY } from "@/store/workspace-events"
import {
  WORKSPACE_STORAGE_INDEXEDDB_FLAG_STORAGE_KEY,
  WORKSPACE_STORAGE_SPLIT_KEY_FLAG_STORAGE_KEY,
  useWorkspaceStore
} from "../workspace"

const resetWorkspaceStore = () => {
  localStorage.removeItem(WORKSPACE_STORAGE_KEY)
  localStorage.removeItem(WORKSPACE_STORAGE_SPLIT_KEY_FLAG_STORAGE_KEY)
  localStorage.removeItem(WORKSPACE_STORAGE_INDEXEDDB_FLAG_STORAGE_KEY)
  useWorkspaceStore.setState({
    workspaceId: "",
    workspaceName: "",
    workspaceTag: "",
    workspaceCreatedAt: null,
    workspaceChatReferenceId: "",
    sources: [],
    selectedSourceIds: [],
    sourceFolders: [],
    sourceFolderMemberships: [],
    selectedSourceFolderIds: [],
    activeFolderId: null,
    sourceSearchQuery: "",
    sourceFocusTarget: null,
    sourcesLoading: false,
    sourcesError: null,
    generatedArtifacts: [],
    notes: "",
    currentNote: { ...DEFAULT_WORKSPACE_NOTE },
    workspaceBanner: {
      title: "",
      subtitle: "",
      image: null
    },
    isGeneratingOutput: false,
    generatingOutputType: null,
    storeHydrated: false,
    leftPaneCollapsed: false,
    rightPaneCollapsed: false,
    addSourceModalOpen: false,
    addSourceModalTab: "upload",
    addSourceProcessing: false,
    addSourceError: null,
    chatFocusTarget: null,
    noteFocusTarget: null,
    audioSettings: { ...DEFAULT_AUDIO_SETTINGS },
    savedWorkspaces: [],
    archivedWorkspaces: [],
    workspaceCollections: [],
    workspaceSnapshots: {},
    workspaceChatSessions: {}
  })
}

describe("workspace artifact review contract", () => {
  beforeEach(async () => {
    resetWorkspaceStore()
    if (useWorkspaceStore.persist?.clearStorage) {
      await useWorkspaceStore.persist.clearStorage()
    }
  })

  it("preserves review metadata when adding artifacts", () => {
    const artifact = useWorkspaceStore.getState().addArtifact({
      type: "report",
      title: "Executive Brief",
      status: "completed",
      templateId: "executive_brief",
      reviewStatus: "reviewing",
      sourceLineage: [
        {
          sourceId: "source-1",
          mediaId: 101,
          title: "Source One",
          citationCount: 3
        }
      ],
      reviewChecklist: [
        {
          id: "claim-sourcing",
          label: "Every material claim has a source.",
          checked: false
        }
      ],
      exportTargets: ["markdown", "pdf"]
    })

    expect(artifact.templateId).toBe("executive_brief")
    expect(artifact.reviewStatus).toBe("reviewing")
    expect(artifact.sourceLineage).toEqual([
      {
        sourceId: "source-1",
        mediaId: 101,
        title: "Source One",
        citationCount: 3
      }
    ])
    expect(artifact.reviewChecklist).toEqual([
      {
        id: "claim-sourcing",
        label: "Every material claim has a source.",
        checked: false
      }
    ])
    expect(artifact.exportTargets).toEqual(["markdown", "pdf"])
  })

  it("does not erase review metadata when generation status changes", () => {
    const artifact = useWorkspaceStore.getState().addArtifact({
      type: "report",
      title: "Research Dossier",
      status: "pending",
      templateId: "research_dossier",
      reviewStatus: "draft",
      sourceLineage: [{ sourceId: "source-2", citationCount: 1 }],
      reviewChecklist: [
        {
          id: "source-coverage",
          label: "Source coverage is visible.",
          checked: true
        }
      ]
    })

    useWorkspaceStore
      .getState()
      .updateArtifactStatus(artifact.id, "completed", { content: "Done" })

    const updatedArtifact = useWorkspaceStore
      .getState()
      .generatedArtifacts.find((entry) => entry.id === artifact.id)

    expect(updatedArtifact?.status).toBe("completed")
    expect(updatedArtifact?.reviewStatus).toBe("draft")
    expect(updatedArtifact?.templateId).toBe("research_dossier")
    expect(updatedArtifact?.sourceLineage).toEqual([
      { sourceId: "source-2", citationCount: 1 }
    ])
    expect(updatedArtifact?.reviewChecklist).toEqual([
      {
        id: "source-coverage",
        label: "Source coverage is visible.",
        checked: true
      }
    ])
  })

  it("keeps review metadata through persistence sanitize and revive", async () => {
    useWorkspaceStore.getState().initializeWorkspace("Artifact Review Contract")
    const workspaceId = useWorkspaceStore.getState().workspaceId
    useWorkspaceStore.getState().addArtifact({
      type: "report",
      title: "Market Memo",
      status: "completed",
      templateId: "competitive_market_memo",
      reviewStatus: "accepted",
      sourceLineage: [
        {
          sourceId: "source-3",
          mediaId: 303,
          title: "Market Source",
          citationCount: 4
        }
      ],
      reviewChecklist: [
        {
          id: "market-assumptions",
          label: "Market assumptions are separated from observed evidence.",
          checked: true
        }
      ],
      exportTargets: ["docx", "chatbook"]
    })
    useWorkspaceStore.getState().saveCurrentWorkspace()

    await useWorkspaceStore.persist.rehydrate()

    const restoredArtifact =
      useWorkspaceStore.getState().workspaceSnapshots[workspaceId]
        ?.generatedArtifacts[0]

    expect(restoredArtifact?.templateId).toBe("competitive_market_memo")
    expect(restoredArtifact?.reviewStatus).toBe("accepted")
    expect(restoredArtifact?.sourceLineage).toEqual([
      {
        sourceId: "source-3",
        mediaId: 303,
        title: "Market Source",
        citationCount: 4
      }
    ])
    expect(restoredArtifact?.reviewChecklist).toEqual([
      {
        id: "market-assumptions",
        label: "Market assumptions are separated from observed evidence.",
        checked: true
      }
    ])
    expect(restoredArtifact?.exportTargets).toEqual(["docx", "chatbook"])
  })
})

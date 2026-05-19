import { existsSync, readFileSync } from "node:fs"
import { describe, expect, it } from "vitest"

const optionsAppPathCandidates = [
  "../packages/ui/src/entries/options/App.tsx",
  "apps/packages/ui/src/entries/options/App.tsx",
  "packages/ui/src/entries/options/App.tsx",
]

const sidepanelAppPathCandidates = [
  "../packages/ui/src/entries/sidepanel/App.tsx",
  "apps/packages/ui/src/entries/sidepanel/App.tsx",
  "packages/ui/src/entries/sidepanel/App.tsx",
]

const sharedOptionsAppPathCandidates = [
  "../packages/ui/src/entries/shared/options-app.tsx",
  "apps/packages/ui/src/entries/shared/options-app.tsx",
  "packages/ui/src/entries/shared/options-app.tsx",
]

const sharedSidepanelAppPathCandidates = [
  "../packages/ui/src/entries/shared/sidepanel-app.tsx",
  "apps/packages/ui/src/entries/shared/sidepanel-app.tsx",
  "packages/ui/src/entries/shared/sidepanel-app.tsx",
]

const extensionSettingsRouteCandidates = [
  "apps/tldw-frontend/extension/routes/settings-route.tsx",
  "extension/routes/settings-route.tsx",
]
const sharedAppRouteCandidates = [
  "apps/packages/ui/src/routes/app-route.tsx",
  "../packages/ui/src/routes/app-route.tsx",
  "packages/ui/src/routes/app-route.tsx",
]
const sharedSidepanelRouteShellCandidates = [
  "apps/packages/ui/src/routes/sidepanel-route-shell.tsx",
  "../packages/ui/src/routes/sidepanel-route-shell.tsx",
  "packages/ui/src/routes/sidepanel-route-shell.tsx",
]
const sharedOptionsRouteShellCandidates = [
  "apps/packages/ui/src/routes/options-route-shell.tsx",
  "../packages/ui/src/routes/options-route-shell.tsx",
  "packages/ui/src/routes/options-route-shell.tsx",
]
const deferredOptionsRoutePathCandidates = [
  "apps/packages/ui/src/routes/deferred-options-route.tsx",
  "../packages/ui/src/routes/deferred-options-route.tsx",
  "packages/ui/src/routes/deferred-options-route.tsx",
]
const optionStartupRoutesCandidates = [
  "apps/packages/ui/src/routes/option-startup-routes.tsx",
  "../packages/ui/src/routes/option-startup-routes.tsx",
  "packages/ui/src/routes/option-startup-routes.tsx",
]
const optionHomeResolverCandidates = [
  "apps/packages/ui/src/routes/option-home-resolver.tsx",
  "../packages/ui/src/routes/option-home-resolver.tsx",
  "packages/ui/src/routes/option-home-resolver.tsx",
]
const optionIndexPathCandidates = [
  "apps/packages/ui/src/routes/option-index.tsx",
  "../packages/ui/src/routes/option-index.tsx",
  "packages/ui/src/routes/option-index.tsx",
]
const notesDockIndexPathCandidates = [
  "apps/packages/ui/src/components/Common/NotesDock/index.tsx",
  "../packages/ui/src/components/Common/NotesDock/index.tsx",
  "packages/ui/src/components/Common/NotesDock/index.tsx",
]
const sharedLayoutPathCandidates = [
  "apps/packages/ui/src/components/Layouts/Layout.tsx",
  "../packages/ui/src/components/Layouts/Layout.tsx",
  "packages/ui/src/components/Layouts/Layout.tsx",
]
const settingsNavPathCandidates = [
  "apps/packages/ui/src/components/Layouts/settings-nav.ts",
  "../packages/ui/src/components/Layouts/settings-nav.ts",
  "packages/ui/src/components/Layouts/settings-nav.ts",
]
const settingsNavConfigPathCandidates = [
  "apps/packages/ui/src/components/Layouts/settings-nav-config.ts",
  "../packages/ui/src/components/Layouts/settings-nav-config.ts",
  "packages/ui/src/components/Layouts/settings-nav-config.ts",
]
const routeRegistryPathCandidates = [
  "apps/packages/ui/src/routes/route-registry.tsx",
  "../packages/ui/src/routes/route-registry.tsx",
  "packages/ui/src/routes/route-registry.tsx",
]
const sidepanelRouteRegistryPathCandidates = [
  "apps/packages/ui/src/routes/sidepanel-route-registry.tsx",
  "../packages/ui/src/routes/sidepanel-route-registry.tsx",
  "packages/ui/src/routes/sidepanel-route-registry.tsx",
]
const commandPaletteHostPathCandidates = [
  "apps/packages/ui/src/components/Common/CommandPaletteHost.tsx",
  "../packages/ui/src/components/Common/CommandPaletteHost.tsx",
  "packages/ui/src/components/Common/CommandPaletteHost.tsx",
]
const quickIngestButtonPathCandidates = [
  "apps/packages/ui/src/components/Layouts/QuickIngestButton.tsx",
  "../packages/ui/src/components/Layouts/QuickIngestButton.tsx",
  "packages/ui/src/components/Layouts/QuickIngestButton.tsx",
]
const extensionWxtConfigCandidates = [
  "apps/extension/wxt.config.ts",
  "../extension/wxt.config.ts",
  "extension/wxt.config.ts",
]
const i18nIndexPathCandidates = [
  "apps/packages/ui/src/i18n/index.ts",
  "../packages/ui/src/i18n/index.ts",
  "packages/ui/src/i18n/index.ts",
]
const i18nEnglishBundlePathCandidates = [
  "apps/packages/ui/src/i18n/lang/en.ts",
  "../packages/ui/src/i18n/lang/en.ts",
  "packages/ui/src/i18n/lang/en.ts",
]
const workflowContainerPathCandidates = [
  "apps/packages/ui/src/components/Common/Workflow/WorkflowContainer.tsx",
  "../packages/ui/src/components/Common/Workflow/WorkflowContainer.tsx",
  "packages/ui/src/components/Common/Workflow/WorkflowContainer.tsx",
]
const workflowIntegrationHostPathCandidates = [
  "apps/packages/ui/src/components/Common/Workflow/WorkflowIntegrationHost.tsx",
  "../packages/ui/src/components/Common/Workflow/WorkflowIntegrationHost.tsx",
  "packages/ui/src/components/Common/Workflow/WorkflowIntegrationHost.tsx",
]
const appShellPathCandidates = [
  "apps/packages/ui/src/entries/shared/AppShell.tsx",
  "../packages/ui/src/entries/shared/AppShell.tsx",
  "packages/ui/src/entries/shared/AppShell.tsx",
]
const repo2TxtPagePathCandidates = [
  "apps/packages/ui/src/components/Option/Repo2Txt/Repo2TxtPage.tsx",
  "../packages/ui/src/components/Option/Repo2Txt/Repo2TxtPage.tsx",
  "packages/ui/src/components/Option/Repo2Txt/Repo2TxtPage.tsx",
]
const documentWorkspacePagePathCandidates = [
  "apps/packages/ui/src/components/DocumentWorkspace/DocumentWorkspacePage.tsx",
  "../packages/ui/src/components/DocumentWorkspace/DocumentWorkspacePage.tsx",
  "packages/ui/src/components/DocumentWorkspace/DocumentWorkspacePage.tsx",
]
const documentViewerPathCandidates = [
  "apps/packages/ui/src/components/DocumentWorkspace/DocumentViewer/index.tsx",
  "../packages/ui/src/components/DocumentWorkspace/DocumentViewer/index.tsx",
  "packages/ui/src/components/DocumentWorkspace/DocumentViewer/index.tsx",
]
const watchlistsPlaygroundPagePathCandidates = [
  "apps/packages/ui/src/components/Option/Watchlists/WatchlistsPlaygroundPage.tsx",
  "../packages/ui/src/components/Option/Watchlists/WatchlistsPlaygroundPage.tsx",
  "packages/ui/src/components/Option/Watchlists/WatchlistsPlaygroundPage.tsx",
]
const acpPlaygroundPathCandidates = [
  "apps/packages/ui/src/components/Option/ACPPlayground/index.tsx",
  "../packages/ui/src/components/Option/ACPPlayground/index.tsx",
  "packages/ui/src/components/Option/ACPPlayground/index.tsx",
]
const acpWorkspacePanelPathCandidates = [
  "apps/packages/ui/src/components/Option/ACPPlayground/ACPWorkspacePanel.tsx",
  "../packages/ui/src/components/Option/ACPPlayground/ACPWorkspacePanel.tsx",
  "packages/ui/src/components/Option/ACPPlayground/ACPWorkspacePanel.tsx",
]
const promptsBodyPathCandidates = [
  "apps/packages/ui/src/components/Option/Prompt/index.tsx",
  "../packages/ui/src/components/Option/Prompt/index.tsx",
  "packages/ui/src/components/Option/Prompt/index.tsx",
]
const studioTabContainerPathCandidates = [
  "apps/packages/ui/src/components/Option/Prompt/Studio/StudioTabContainer.tsx",
  "../packages/ui/src/components/Option/Prompt/Studio/StudioTabContainer.tsx",
  "packages/ui/src/components/Option/Prompt/Studio/StudioTabContainer.tsx",
]
const workspacePlaygroundPathCandidates = [
  "apps/packages/ui/src/components/Option/WorkspacePlayground/index.tsx",
  "../packages/ui/src/components/Option/WorkspacePlayground/index.tsx",
  "packages/ui/src/components/Option/WorkspacePlayground/index.tsx",
]
const viewMediaPagePathCandidates = [
  "apps/packages/ui/src/components/Review/ViewMediaPage.tsx",
  "../packages/ui/src/components/Review/ViewMediaPage.tsx",
  "packages/ui/src/components/Review/ViewMediaPage.tsx",
]
const contentViewerPathCandidates = [
  "apps/packages/ui/src/components/Media/ContentViewer.tsx",
  "../packages/ui/src/components/Media/ContentViewer.tsx",
  "packages/ui/src/components/Media/ContentViewer.tsx",
]
const studioPanePathCandidates = [
  "apps/packages/ui/src/components/Option/WorkspacePlayground/StudioPane/index.tsx",
  "../packages/ui/src/components/Option/WorkspacePlayground/StudioPane/index.tsx",
  "packages/ui/src/components/Option/WorkspacePlayground/StudioPane/index.tsx",
]
const knowledgeQaIndexPathCandidates = [
  "apps/packages/ui/src/components/Option/KnowledgeQA/index.tsx",
  "../packages/ui/src/components/Option/KnowledgeQA/index.tsx",
  "packages/ui/src/components/Option/KnowledgeQA/index.tsx",
]
const knowledgeQaLayoutPathCandidates = [
  "apps/packages/ui/src/components/Option/KnowledgeQA/layout/KnowledgeQALayout.tsx",
  "../packages/ui/src/components/Option/KnowledgeQA/layout/KnowledgeQALayout.tsx",
  "packages/ui/src/components/Option/KnowledgeQA/layout/KnowledgeQALayout.tsx",
]
const knowledgeQaSourceListPathCandidates = [
  "apps/packages/ui/src/components/Option/KnowledgeQA/SourceList.tsx",
  "../packages/ui/src/components/Option/KnowledgeQA/SourceList.tsx",
  "packages/ui/src/components/Option/KnowledgeQA/SourceList.tsx",
]
const knowledgeQaEvidenceRailPathCandidates = [
  "apps/packages/ui/src/components/Option/KnowledgeQA/evidence/EvidenceRail.tsx",
  "../packages/ui/src/components/Option/KnowledgeQA/evidence/EvidenceRail.tsx",
  "packages/ui/src/components/Option/KnowledgeQA/evidence/EvidenceRail.tsx",
]
const notesManagerPagePathCandidates = [
  "apps/packages/ui/src/components/Notes/NotesManagerPage.tsx",
  "../packages/ui/src/components/Notes/NotesManagerPage.tsx",
  "packages/ui/src/components/Notes/NotesManagerPage.tsx",
]
const notesEditorPanePathCandidates = [
  "apps/packages/ui/src/components/Notes/NotesEditorPane.tsx",
  "../packages/ui/src/components/Notes/NotesEditorPane.tsx",
  "packages/ui/src/components/Notes/NotesEditorPane.tsx",
]
const notesListPanelPathCandidates = [
  "apps/packages/ui/src/components/Notes/NotesListPanel.tsx",
  "../packages/ui/src/components/Notes/NotesListPanel.tsx",
  "packages/ui/src/components/Notes/NotesListPanel.tsx",
]
const charactersManagerPathCandidates = [
  "apps/packages/ui/src/components/Option/Characters/Manager.tsx",
  "../packages/ui/src/components/Option/Characters/Manager.tsx",
  "packages/ui/src/components/Option/Characters/Manager.tsx",
]
const characterListContentPathCandidates = [
  "apps/packages/ui/src/components/Option/Characters/CharacterListContent.tsx",
  "../packages/ui/src/components/Option/Characters/CharacterListContent.tsx",
  "packages/ui/src/components/Option/Characters/CharacterListContent.tsx",
]
const characterEditorFormPathCandidates = [
  "apps/packages/ui/src/components/Option/Characters/CharacterEditorForm.tsx",
  "../packages/ui/src/components/Option/Characters/CharacterEditorForm.tsx",
  "packages/ui/src/components/Option/Characters/CharacterEditorForm.tsx",
]
const playgroundPathCandidates = [
  "apps/packages/ui/src/components/Option/Playground/Playground.tsx",
  "../packages/ui/src/components/Option/Playground/Playground.tsx",
  "packages/ui/src/components/Option/Playground/Playground.tsx",
]
const playgroundFormPathCandidates = [
  "apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx",
  "../packages/ui/src/components/Option/Playground/PlaygroundForm.tsx",
  "packages/ui/src/components/Option/Playground/PlaygroundForm.tsx",
]
const playgroundChatPathCandidates = [
  "apps/packages/ui/src/components/Option/Playground/PlaygroundChat.tsx",
  "../packages/ui/src/components/Option/Playground/PlaygroundChat.tsx",
  "packages/ui/src/components/Option/Playground/PlaygroundChat.tsx",
]
const playgroundCompareClusterPathCandidates = [
  "apps/packages/ui/src/components/Option/Playground/PlaygroundCompareCluster.tsx",
  "../packages/ui/src/components/Option/Playground/PlaygroundCompareCluster.tsx",
  "packages/ui/src/components/Option/Playground/PlaygroundCompareCluster.tsx",
]
const sidepanelChatPathCandidates = [
  "apps/packages/ui/src/routes/sidepanel-chat.tsx",
  "../packages/ui/src/routes/sidepanel-chat.tsx",
  "packages/ui/src/routes/sidepanel-chat.tsx",
]
const workflowEditorPathCandidates = [
  "apps/packages/ui/src/components/WorkflowEditor/WorkflowEditor.tsx",
  "../packages/ui/src/components/WorkflowEditor/WorkflowEditor.tsx",
  "packages/ui/src/components/WorkflowEditor/WorkflowEditor.tsx",
]
const workflowEditorIndexPathCandidates = [
  "apps/packages/ui/src/components/WorkflowEditor/index.ts",
  "../packages/ui/src/components/WorkflowEditor/index.ts",
  "packages/ui/src/components/WorkflowEditor/index.ts",
]
const quizPlaygroundPathCandidates = [
  "apps/packages/ui/src/components/Quiz/QuizPlayground.tsx",
  "../packages/ui/src/components/Quiz/QuizPlayground.tsx",
  "packages/ui/src/components/Quiz/QuizPlayground.tsx",
]
const writingPlaygroundPathCandidates = [
  "apps/packages/ui/src/components/Option/WritingPlayground/index.tsx",
  "../packages/ui/src/components/Option/WritingPlayground/index.tsx",
  "packages/ui/src/components/Option/WritingPlayground/index.tsx",
]

const optionsAppPath = optionsAppPathCandidates.find((candidate) =>
  existsSync(candidate),
)
const sidepanelAppPath = sidepanelAppPathCandidates.find((candidate) =>
  existsSync(candidate),
)
const sharedOptionsAppPath = sharedOptionsAppPathCandidates.find((candidate) =>
  existsSync(candidate),
)
const sharedSidepanelAppPath = sharedSidepanelAppPathCandidates.find((candidate) =>
  existsSync(candidate),
)
const extensionSettingsRoutePath = extensionSettingsRouteCandidates.find(
  (candidate) => existsSync(candidate),
)
const sharedAppRoutePath = sharedAppRouteCandidates.find((candidate) =>
  existsSync(candidate),
)
const sharedSidepanelRouteShellPath = sharedSidepanelRouteShellCandidates.find(
  (candidate) => existsSync(candidate),
)
const sharedOptionsRouteShellPath = sharedOptionsRouteShellCandidates.find(
  (candidate) => existsSync(candidate),
)
const deferredOptionsRoutePath = deferredOptionsRoutePathCandidates.find(
  (candidate) => existsSync(candidate),
)
const optionStartupRoutesPath = optionStartupRoutesCandidates.find((candidate) =>
  existsSync(candidate),
)
const optionHomeResolverPath = optionHomeResolverCandidates.find((candidate) =>
  existsSync(candidate),
)
const optionIndexPath = optionIndexPathCandidates.find((candidate) =>
  existsSync(candidate),
)
const notesDockIndexPath = notesDockIndexPathCandidates.find((candidate) =>
  existsSync(candidate),
)
const sharedLayoutPath = sharedLayoutPathCandidates.find((candidate) =>
  existsSync(candidate),
)
const settingsNavPath = settingsNavPathCandidates.find((candidate) =>
  existsSync(candidate),
)
const settingsNavConfigPath = settingsNavConfigPathCandidates.find((candidate) =>
  existsSync(candidate),
)
const routeRegistryPath = routeRegistryPathCandidates.find((candidate) =>
  existsSync(candidate),
)
const sidepanelRouteRegistryPath = sidepanelRouteRegistryPathCandidates.find(
  (candidate) => existsSync(candidate),
)
const commandPaletteHostPath = commandPaletteHostPathCandidates.find((candidate) =>
  existsSync(candidate),
)
const quickIngestButtonPath = quickIngestButtonPathCandidates.find((candidate) =>
  existsSync(candidate),
)
const extensionWxtConfigPath = extensionWxtConfigCandidates.find((candidate) =>
  existsSync(candidate),
)
const i18nIndexPath = i18nIndexPathCandidates.find((candidate) =>
  existsSync(candidate),
)
const i18nEnglishBundlePath = i18nEnglishBundlePathCandidates.find(
  (candidate) => existsSync(candidate),
)
const workflowContainerPath = workflowContainerPathCandidates.find((candidate) =>
  existsSync(candidate),
)
const workflowIntegrationHostPath = workflowIntegrationHostPathCandidates.find(
  (candidate) => existsSync(candidate),
)
const appShellPath = appShellPathCandidates.find((candidate) =>
  existsSync(candidate),
)
const repo2TxtPagePath = repo2TxtPagePathCandidates.find((candidate) =>
  existsSync(candidate),
)
const documentWorkspacePagePath = documentWorkspacePagePathCandidates.find(
  (candidate) => existsSync(candidate),
)
const documentViewerPath = documentViewerPathCandidates.find((candidate) =>
  existsSync(candidate),
)
const watchlistsPlaygroundPagePath = watchlistsPlaygroundPagePathCandidates.find(
  (candidate) => existsSync(candidate),
)
const acpPlaygroundPath = acpPlaygroundPathCandidates.find((candidate) =>
  existsSync(candidate),
)
const acpWorkspacePanelPath = acpWorkspacePanelPathCandidates.find((candidate) =>
  existsSync(candidate),
)
const promptsBodyPath = promptsBodyPathCandidates.find((candidate) =>
  existsSync(candidate),
)
const studioTabContainerPath = studioTabContainerPathCandidates.find((candidate) =>
  existsSync(candidate),
)
const workspacePlaygroundPath = workspacePlaygroundPathCandidates.find(
  (candidate) => existsSync(candidate),
)
const viewMediaPagePath = viewMediaPagePathCandidates.find((candidate) =>
  existsSync(candidate),
)
const contentViewerPath = contentViewerPathCandidates.find((candidate) =>
  existsSync(candidate),
)
const studioPanePath = studioPanePathCandidates.find((candidate) =>
  existsSync(candidate),
)
const knowledgeQaIndexPath = knowledgeQaIndexPathCandidates.find((candidate) =>
  existsSync(candidate),
)
const knowledgeQaLayoutPath = knowledgeQaLayoutPathCandidates.find((candidate) =>
  existsSync(candidate),
)
const knowledgeQaSourceListPath = knowledgeQaSourceListPathCandidates.find((candidate) =>
  existsSync(candidate),
)
const knowledgeQaEvidenceRailPath = knowledgeQaEvidenceRailPathCandidates.find((candidate) =>
  existsSync(candidate),
)
const notesManagerPagePath = notesManagerPagePathCandidates.find((candidate) =>
  existsSync(candidate),
)
const notesEditorPanePath = notesEditorPanePathCandidates.find((candidate) =>
  existsSync(candidate),
)
const notesListPanelPath = notesListPanelPathCandidates.find((candidate) =>
  existsSync(candidate),
)
const charactersManagerPath = charactersManagerPathCandidates.find((candidate) =>
  existsSync(candidate),
)
const characterListContentPath = characterListContentPathCandidates.find((candidate) =>
  existsSync(candidate),
)
const characterEditorFormPath = characterEditorFormPathCandidates.find((candidate) =>
  existsSync(candidate),
)
const playgroundPath = playgroundPathCandidates.find((candidate) =>
  existsSync(candidate),
)
const playgroundFormPath = playgroundFormPathCandidates.find((candidate) =>
  existsSync(candidate),
)
const playgroundChatPath = playgroundChatPathCandidates.find((candidate) =>
  existsSync(candidate),
)
const sidepanelChatPath = sidepanelChatPathCandidates.find((candidate) =>
  existsSync(candidate),
)
const playgroundCompareClusterPath = playgroundCompareClusterPathCandidates.find(
  (candidate) => existsSync(candidate),
)
const workflowEditorPath = workflowEditorPathCandidates.find((candidate) =>
  existsSync(candidate),
)
const workflowEditorIndexPath = workflowEditorIndexPathCandidates.find((candidate) =>
  existsSync(candidate),
)
const quizPlaygroundPath = quizPlaygroundPathCandidates.find((candidate) =>
  existsSync(candidate),
)
const writingPlaygroundPath = writingPlaygroundPathCandidates.find((candidate) =>
  existsSync(candidate),
)

if (
  !optionsAppPath ||
  !sidepanelAppPath ||
  !sharedOptionsAppPath ||
  !sharedSidepanelAppPath ||
  !extensionSettingsRoutePath ||
  !sharedAppRoutePath ||
  !sharedSidepanelRouteShellPath ||
  !sharedOptionsRouteShellPath ||
  !deferredOptionsRoutePath ||
  !notesDockIndexPath ||
  !sharedLayoutPath ||
  !settingsNavPath ||
  !settingsNavConfigPath ||
  !routeRegistryPath ||
  !commandPaletteHostPath ||
  !quickIngestButtonPath ||
  !optionStartupRoutesPath ||
  !optionHomeResolverPath ||
  !optionIndexPath ||
  !i18nIndexPath ||
  !i18nEnglishBundlePath ||
  !workflowContainerPath ||
  !workflowIntegrationHostPath ||
  !appShellPath ||
  !repo2TxtPagePath ||
  !documentWorkspacePagePath ||
  !documentViewerPath ||
  !watchlistsPlaygroundPagePath ||
  !acpPlaygroundPath ||
  !acpWorkspacePanelPath ||
  !promptsBodyPath ||
  !studioTabContainerPath ||
  !workspacePlaygroundPath ||
  !viewMediaPagePath ||
  !contentViewerPath ||
  !studioPanePath ||
  !knowledgeQaIndexPath ||
  !knowledgeQaLayoutPath ||
  !knowledgeQaSourceListPath ||
  !knowledgeQaEvidenceRailPath ||
  !notesManagerPagePath ||
  !notesEditorPanePath ||
  !notesListPanelPath ||
  !charactersManagerPath ||
  !characterListContentPath ||
  !characterEditorFormPath ||
  !playgroundPath ||
  !playgroundFormPath ||
  !playgroundChatPath ||
  !playgroundCompareClusterPath ||
  !sidepanelChatPath ||
  !workflowEditorPath ||
  !workflowEditorIndexPath ||
  !quizPlaygroundPath ||
  !writingPlaygroundPath
) {
  throw new Error("Unable to locate extension entry or settings route source files")
}

const optionsAppSource = readFileSync(optionsAppPath, "utf8")
const sidepanelAppSource = readFileSync(sidepanelAppPath, "utf8")
const sharedOptionsAppSource = readFileSync(sharedOptionsAppPath, "utf8")
const sharedSidepanelAppSource = readFileSync(sharedSidepanelAppPath, "utf8")
const extensionSettingsRouteSource = readFileSync(
  extensionSettingsRoutePath,
  "utf8",
)
const sharedAppRouteSource = readFileSync(sharedAppRoutePath, "utf8")
const sharedSidepanelRouteShellSource = readFileSync(
  sharedSidepanelRouteShellPath,
  "utf8",
)
const sharedOptionsRouteShellSource = readFileSync(
  sharedOptionsRouteShellPath,
  "utf8",
)
const deferredOptionsRouteSource = readFileSync(
  deferredOptionsRoutePath,
  "utf8",
)
const notesDockIndexSource = readFileSync(notesDockIndexPath, "utf8")
const sharedLayoutSource = readFileSync(sharedLayoutPath, "utf8")
const settingsNavSource = readFileSync(settingsNavPath, "utf8")
const settingsNavConfigSource = readFileSync(settingsNavConfigPath, "utf8")
const routeRegistrySource = readFileSync(routeRegistryPath, "utf8")
const sidepanelRouteRegistrySource = readFileSync(
  sidepanelRouteRegistryPath,
  "utf8",
)
const commandPaletteHostSource = readFileSync(commandPaletteHostPath, "utf8")
const quickIngestButtonSource = readFileSync(quickIngestButtonPath, "utf8")
const optionStartupRoutesSource = readFileSync(optionStartupRoutesPath, "utf8")
const optionHomeResolverSource = readFileSync(optionHomeResolverPath, "utf8")
const optionIndexSource = readFileSync(optionIndexPath, "utf8")
const i18nIndexSource = readFileSync(i18nIndexPath, "utf8")
const i18nEnglishBundleSource = readFileSync(i18nEnglishBundlePath, "utf8")
const workflowContainerSource = readFileSync(workflowContainerPath, "utf8")
const workflowIntegrationHostSource = readFileSync(
  workflowIntegrationHostPath,
  "utf8",
)
const appShellSource = readFileSync(appShellPath, "utf8")
const repo2TxtPageSource = readFileSync(repo2TxtPagePath, "utf8")
const documentWorkspacePageSource = readFileSync(documentWorkspacePagePath, "utf8")
const documentViewerSource = readFileSync(documentViewerPath, "utf8")
const watchlistsPlaygroundPageSource = readFileSync(
  watchlistsPlaygroundPagePath,
  "utf8",
)
const acpPlaygroundSource = readFileSync(acpPlaygroundPath, "utf8")
const acpWorkspacePanelSource = readFileSync(acpWorkspacePanelPath, "utf8")
const promptsBodySource = readFileSync(promptsBodyPath, "utf8")
const studioTabContainerSource = readFileSync(studioTabContainerPath, "utf8")
const workspacePlaygroundSource = readFileSync(workspacePlaygroundPath, "utf8")
const viewMediaPageSource = readFileSync(viewMediaPagePath, "utf8")
const contentViewerSource = readFileSync(contentViewerPath, "utf8")
const studioPaneSource = readFileSync(studioPanePath, "utf8")
const knowledgeQaIndexSource = readFileSync(knowledgeQaIndexPath, "utf8")
const knowledgeQaLayoutSource = readFileSync(knowledgeQaLayoutPath, "utf8")
const knowledgeQaSourceListSource = readFileSync(knowledgeQaSourceListPath, "utf8")
const knowledgeQaEvidenceRailSource = readFileSync(knowledgeQaEvidenceRailPath, "utf8")
const notesManagerPageSource = readFileSync(notesManagerPagePath, "utf8")
const notesEditorPaneSource = readFileSync(notesEditorPanePath, "utf8")
const notesListPanelSource = readFileSync(notesListPanelPath, "utf8")
const charactersManagerSource = readFileSync(charactersManagerPath, "utf8")
const characterListContentSource = readFileSync(characterListContentPath, "utf8")
const characterEditorFormSource = readFileSync(characterEditorFormPath, "utf8")
const playgroundSource = readFileSync(playgroundPath, "utf8")
const playgroundFormSource = readFileSync(playgroundFormPath, "utf8")
const playgroundChatSource = readFileSync(playgroundChatPath, "utf8")
const playgroundCompareClusterSource = readFileSync(
  playgroundCompareClusterPath,
  "utf8",
)
const sidepanelChatSource = readFileSync(sidepanelChatPath, "utf8")
const workflowEditorSource = readFileSync(workflowEditorPath, "utf8")
const workflowEditorIndexSource = readFileSync(workflowEditorIndexPath, "utf8")
const quizPlaygroundSource = readFileSync(quizPlaygroundPath, "utf8")
const writingPlaygroundSource = readFileSync(writingPlaygroundPath, "utf8")

describe("extension entry shell performance contracts", () => {
  it("keeps options and sidepanel entry modules split instead of re-exporting from one shared apps module", () => {
    expect(optionsAppSource).not.toContain('from "@/entries/shared/apps"')
    expect(sidepanelAppSource).not.toContain('from "@/entries/shared/apps"')
  })

  it("avoids wrapping extension settings pages in the full web layout shell", () => {
    expect(extensionSettingsRouteSource).not.toContain(
      '@web/components/layout/WebLayout',
    )
  })

  it("does not keep both option and sidepanel route registries in one shared route shell", () => {
    expect(sharedAppRouteSource).not.toContain("optionRoutes,")
    expect(sharedAppRouteSource).not.toContain("sidepanelRoutes,")
    expect(sharedAppRouteSource).not.toContain(
      'kind === "options" ? optionRoutes : sidepanelRoutes',
    )
  })

  it("loads UI diagnostics lazily instead of bundling them into the shared route shell", () => {
    expect(sharedAppRouteSource).not.toContain('from "@/utils/ui-diagnostics"')
    expect(sharedAppRouteSource).toContain('import("@/utils/ui-diagnostics")')
  })

  it("keeps settings navigation metadata off the deferred route registry chunk", () => {
    expect(settingsNavSource).not.toContain('from "@/routes/route-registry"')
    expect(settingsNavSource).toContain('from "./settings-nav-config"')
    expect(settingsNavConfigSource).toContain('from "lucide-react"')
    expect(routeRegistrySource).not.toContain('from "lucide-react"')
  })

  it("keeps the sidepanel route shell on a dedicated sidepanel registry path", () => {
    expect(sharedSidepanelRouteShellSource).not.toContain(
      'from "./route-registry"',
    )
  })

  it("keeps sidepanel route definitions off the deferred options registry chunk", () => {
    expect(routeRegistrySource).not.toContain("SidepanelChat")
    expect(routeRegistrySource).not.toContain("SidepanelHomeResolver")
    expect(routeRegistrySource).not.toContain("SidepanelCompanion")
    expect(routeRegistrySource).not.toContain("SidepanelPersona")
    expect(routeRegistrySource).not.toContain('kind: "sidepanel"')
    expect(sidepanelRouteRegistrySource).toContain("SidepanelChat")
    expect(sidepanelRouteRegistrySource).toContain("SidepanelHomeResolver")
    expect(sidepanelRouteRegistrySource).toContain("SidepanelCompanion")
    expect(sidepanelRouteRegistrySource).toContain("SidepanelPersona")
  })

  it("keeps the options route shell on a small startup registry and defers the full registry", () => {
    expect(sharedOptionsRouteShellSource).not.toContain(
      'from "./route-registry"',
    )
    expect(sharedOptionsRouteShellSource).toContain(
      'from "./option-startup-routes"',
    )
    expect(sharedOptionsRouteShellSource).toContain(
      'from "./deferred-options-route"',
    )
    expect(sharedOptionsRouteShellSource).toContain("renderUnmatchedRoute")
    expect(optionStartupRoutesSource).toContain(
      'import OptionHomeResolver from "./option-home-resolver"',
    )
    expect(optionStartupRoutesSource).toContain('path: "/"')
    expect(optionStartupRoutesSource).not.toContain('path: "/chat"')
    expect(optionStartupRoutesSource).not.toContain('path: "/media"')
    expect(optionStartupRoutesSource).not.toContain('path: "/media-multi"')
    expect(optionHomeResolverSource).toContain('import("./option-index")')
  })

  it("loads settings deep links through a dedicated settings registry before falling back to the full options registry", () => {
    expect(deferredOptionsRouteSource).toContain(
      'import("./option-settings-route-registry")',
    )
    expect(deferredOptionsRouteSource).toContain('import("./route-registry")')
    expect(deferredOptionsRouteSource).toContain(
      'pathname === "/settings" || pathname.startsWith("/settings/")',
    )
  })

  it("loads chat and media deep links through dedicated route-family registries instead of the full options registry", () => {
    expect(deferredOptionsRouteSource).toContain(
      'import("./option-chat-route-registry")',
    )
    expect(deferredOptionsRouteSource).toContain(
      'import("./option-media-view-route-registry")',
    )
    expect(deferredOptionsRouteSource).toContain(
      'import("./option-media-review-route-registry")',
    )
    expect(deferredOptionsRouteSource).toContain(
      'pathname === "/chat"',
    )
    expect(deferredOptionsRouteSource).toContain(
      'pathname === "/media"',
    )
    expect(deferredOptionsRouteSource).toContain(
      'pathname === "/media-trash"',
    )
    expect(deferredOptionsRouteSource).toContain(
      'pathname === "/media-multi"',
    )
    expect(deferredOptionsRouteSource).toContain(
      'pathname === "/review"',
    )
    expect(routeRegistrySource).not.toContain('import OptionChat from "./option-chat"')
    expect(routeRegistrySource).not.toContain(
      'import OptionMediaMulti from "./option-media-multi"',
    )
    expect(routeRegistrySource).not.toContain('import OptionMedia from "./option-media"')
  })

  it("keeps mutually exclusive option-home branches lazy instead of bundling onboarding and companion home together", () => {
    expect(optionIndexSource).not.toContain(
      'import { OnboardingWizard } from "@/components/Option/Onboarding/OnboardingWizard"',
    )
    expect(optionIndexSource).not.toContain(
      'import { CompanionHomeShell } from "@/components/Option/CompanionHome"',
    )
    expect(optionIndexSource).toContain(
      'import("@/components/Option/Onboarding/OnboardingWizard")',
    )
    expect(optionIndexSource).toContain(
      'import("@/components/Option/CompanionHome")',
    )
  })

  it("keeps the hosted-only home branch behind a dedicated lazy route module", () => {
    expect(optionIndexSource).toContain('import("./option-hosted-home")')
    expect(optionIndexSource).not.toContain(
      "Start with the narrow hosted path, keep self-host when you need full control.",
    )
  })

  it("lazy-loads the workflow integration host from extension entry apps", () => {
    expect(sharedOptionsAppSource).not.toContain(
      'import { WorkflowIntegrationHost } from "@/components/Common/Workflow"',
    )
    expect(sharedSidepanelAppSource).not.toContain(
      'import { WorkflowIntegrationHost } from "@/components/Common/Workflow"',
    )
    expect(sharedOptionsAppSource).not.toContain(
      'import("@/components/Common/Workflow")',
    )
    expect(sharedSidepanelAppSource).not.toContain(
      'import("@/components/Common/Workflow")',
    )
    expect(sharedOptionsAppSource).toContain(
      'import("@/components/Common/Workflow/WorkflowIntegrationHost")',
    )
    expect(sharedSidepanelAppSource).toContain(
      'import("@/components/Common/Workflow/WorkflowIntegrationHost")',
    )
  })

  it("keeps workflow steps off the base workflow container import path", () => {
    expect(workflowContainerSource).not.toContain(
      'import { SummarizePageWorkflow, QuickSaveWorkflow, AnalyzeBookWorkflow } from "./steps"',
    )
    expect(workflowContainerSource).toContain(
      'import("./steps/SummarizePageWorkflow")',
    )
    expect(workflowContainerSource).toContain(
      'import("./steps/QuickSaveWorkflow")',
    )
    expect(workflowContainerSource).toContain(
      'import("./steps/AnalyzeBookWorkflow")',
    )
  })

  it("keeps the workflow integration host on lazy landing and overlay boundaries", () => {
    expect(workflowIntegrationHostSource).not.toContain(
      'import { WorkflowLandingModal } from "./WorkflowLanding"',
    )
    expect(workflowIntegrationHostSource).not.toContain(
      'import { WorkflowOverlay } from "./WorkflowContainer"',
    )
    expect(workflowIntegrationHostSource).toContain(
      'import("./WorkflowLanding")',
    )
    expect(workflowIntegrationHostSource).toContain(
      'import("./WorkflowContainer")',
    )
  })

  it("does not statically re-export the notes dock panel from the shared notes dock barrel", () => {
    expect(notesDockIndexSource).not.toContain(
      'export { NotesDockPanel } from "./NotesDockPanel"',
    )
  })

  it("loads the global command palette behind a lightweight host instead of importing it directly into layout", () => {
    expect(sharedLayoutSource).not.toContain(
      'from "@/components/Common/CommandPalette"',
    )
    expect(sharedLayoutSource).toContain(
      'from "@/components/Common/CommandPaletteHost"',
    )
    expect(commandPaletteHostSource).toContain(
      'import("./CommandPalette")',
    )
    expect(commandPaletteHostSource).toContain("registerGlobalOpenShortcut={false}")
    expect(commandPaletteHostSource).toContain("listenForOpenEvents={false}")
  })

  it("lazy-loads the quick ingest wizard modal instead of importing it into the layout shell", () => {
    expect(quickIngestButtonSource).not.toContain(
      'from "../Common/QuickIngestWizardModal"',
    )
    expect(quickIngestButtonSource).toContain(
      'import("../Common/QuickIngestWizardModal")',
    )
  })

  it("lazy-loads current chat model settings from the shared options layout", () => {
    expect(sharedLayoutSource).not.toContain(
      'import { CurrentChatModelSettings } from "../Common/Settings/CurrentChatModelSettings"',
    )
    expect(sharedLayoutSource).toContain(
      'import("../Common/Settings/CurrentChatModelSettings")',
    )
  })

  it("lazy-loads even the shared english startup namespace while the route shell bootstraps required namespaces", () => {
    expect(i18nIndexSource).not.toContain('resources: { en }')
    expect(i18nIndexSource).not.toContain('from "./lang/en"')
    expect(i18nIndexSource).toMatch(
      /const BASE_NAMESPACES: Namespace\[\] = \[\s*"common"\s*\]/,
    )
    expect(i18nEnglishBundleSource).not.toContain(
      'import common from "@/assets/locale/en/common.json"',
    )
    expect(sharedAppRouteSource).toContain(
      'const routeNamespaces = getRouteBootstrapNamespaces(kind, location.pathname)',
    )
    expect(sharedAppRouteSource).toContain('await ensureI18nNamespaces(routeNamespaces, "en")')
    expect(sharedAppRouteSource).toContain("setRouteNamespacesReady(false)")
    expect(sharedOptionsAppSource).toContain('defaultValue: "No data"')
    expect(sharedSidepanelAppSource).toContain('defaultValue: "No data"')
  })

  it("keeps locale JSON diagnostics off the production app shell import path", () => {
    expect(appShellSource).not.toContain(
      'from "@/components/Common/LocaleJsonDiagnostics"',
    )
    expect(appShellSource).toContain(
      'import("@/components/Common/LocaleJsonDiagnostics")',
    )
  })

  it("defers repo2txt providers and formatter until the user actually loads or generates content", () => {
    expect(repo2TxtPageSource).not.toContain(
      'import { Formatter } from "./formatter/Formatter"',
    )
    expect(repo2TxtPageSource).not.toContain(
      'import { GitHubProvider } from "./providers/GitHubProvider"',
    )
    expect(repo2TxtPageSource).not.toContain(
      'import { LocalProvider } from "./providers/LocalProvider"',
    )
    expect(repo2TxtPageSource).toContain('import("./formatter/Formatter")')
    expect(repo2TxtPageSource).toContain('import("./providers/GitHubProvider")')
    expect(repo2TxtPageSource).toContain('import("./providers/LocalProvider")')
  })

  it("defers document workspace viewer internals and tab panels behind route-local lazy boundaries", () => {
    expect(documentWorkspacePageSource).not.toContain(
      'import { DocumentViewer } from "./DocumentViewer"',
    )
    expect(documentWorkspacePageSource).not.toContain(
      'from "./LeftSidebar"',
    )
    expect(documentWorkspacePageSource).not.toContain(
      'from "./RightPanel"',
    )
    expect(documentWorkspacePageSource).toContain('import("./DocumentViewer")')
    expect(documentWorkspacePageSource).toContain(
      'import("./LeftSidebar/PagesTab")',
    )
    expect(documentWorkspacePageSource).toContain(
      'import("./LeftSidebar/ReferencesTab")',
    )
    expect(documentWorkspacePageSource).toContain(
      'import("./RightPanel/DocumentChat")',
    )
    expect(documentViewerSource).not.toContain(
      'import { PdfDocument } from "./PdfViewer/PdfDocument"',
    )
    expect(documentViewerSource).not.toContain(
      'import { PdfSearch } from "./PdfSearch"',
    )
    expect(documentViewerSource).not.toContain(
      'import { EpubViewer } from "./EpubViewer"',
    )
    expect(documentViewerSource).toContain('import("./PdfViewer/PdfDocument")')
    expect(documentViewerSource).toContain('import("./PdfSearch")')
    expect(documentViewerSource).toContain('import("./EpubViewer")')
  })

  it("defers watchlists tabs behind route-local lazy boundaries instead of importing every pane into startup", () => {
    expect(watchlistsPlaygroundPageSource).not.toContain(
      'import { OverviewTab } from "./OverviewTab/OverviewTab"',
    )
    expect(watchlistsPlaygroundPageSource).not.toContain(
      'import { SourcesTab } from "./SourcesTab/SourcesTab"',
    )
    expect(watchlistsPlaygroundPageSource).not.toContain(
      'import { JobsTab } from "./JobsTab/JobsTab"',
    )
    expect(watchlistsPlaygroundPageSource).not.toContain(
      'import { RunsTab } from "./RunsTab/RunsTab"',
    )
    expect(watchlistsPlaygroundPageSource).not.toContain(
      'import { OutputsTab } from "./OutputsTab/OutputsTab"',
    )
    expect(watchlistsPlaygroundPageSource).not.toContain(
      'import { TemplatesTab } from "./TemplatesTab/TemplatesTab"',
    )
    expect(watchlistsPlaygroundPageSource).not.toContain(
      'import { SettingsTab } from "./SettingsTab/SettingsTab"',
    )
    expect(watchlistsPlaygroundPageSource).not.toContain(
      'import { ItemsTab } from "./ItemsTab/ItemsTab"',
    )
    expect(watchlistsPlaygroundPageSource).toContain(
      'import("./OverviewTab/OverviewTab")',
    )
    expect(watchlistsPlaygroundPageSource).toContain(
      'import("./TemplatesTab/TemplatesTab")',
    )
    expect(watchlistsPlaygroundPageSource).toContain("renderWatchlistsTab")
    expect(watchlistsPlaygroundPageSource).toContain("destroyOnHidden")
  })

  it("defers ACP side panels and permission modal behind route-local lazy boundaries", () => {
    expect(acpPlaygroundSource).not.toContain(
      'import { ACPSessionPanel } from "./ACPSessionPanel"',
    )
    expect(acpPlaygroundSource).not.toContain(
      'import { ACPToolsPanel } from "./ACPToolsPanel"',
    )
    expect(acpPlaygroundSource).not.toContain(
      'import { ACPPermissionModal } from "./ACPPermissionModal"',
    )
    expect(acpPlaygroundSource).not.toContain(
      'import { ACPWorkspacePanel } from "./ACPWorkspacePanel"',
    )
    expect(acpPlaygroundSource).toContain('import("./ACPSessionPanel")')
    expect(acpPlaygroundSource).toContain('import("./ACPToolsPanel")')
    expect(acpPlaygroundSource).toContain('import("./ACPPermissionModal")')
    expect(acpPlaygroundSource).toContain('import("./ACPWorkspacePanel")')
    expect(acpPlaygroundSource).toContain("renderAcpToolsPanel")
    expect(acpPlaygroundSource).toContain("destroyOnHidden")
  })

  it("defers ACP workspace terminal runtime behind the active SSH session path", () => {
    expect(acpWorkspacePanelSource).not.toContain('from "xterm"')
    expect(acpWorkspacePanelSource).not.toContain('from "@xterm/addon-fit"')
    expect(acpWorkspacePanelSource).not.toContain('import "xterm/css/xterm.css"')
    expect(acpWorkspacePanelSource).toContain('import("xterm")')
    expect(acpWorkspacePanelSource).toContain('import("@xterm/addon-fit")')
    expect(acpWorkspacePanelSource).toContain('import("xterm/css/xterm.css")')
  })

  it("defers the chat artifacts panel behind click-time lazy boundaries across web and sidepanel chat", () => {
    expect(playgroundSource).not.toContain(
      'import { ArtifactsPanel } from "@/components/Sidepanel/Chat/ArtifactsPanel"',
    )
    expect(sidepanelChatSource).not.toContain(
      'import { ArtifactsPanel } from "@/components/Sidepanel/Chat/ArtifactsPanel"',
    )
    expect(playgroundSource).toContain(
      'import("@/components/Sidepanel/Chat/ArtifactsPanel")',
    )
    expect(sidepanelChatSource).toContain(
      'import("@/components/Sidepanel/Chat/ArtifactsPanel")',
    )
  })

  it("defers closed-default chat form overlays behind lazy boundaries", () => {
    expect(playgroundFormSource).not.toContain(
      'import { CurrentChatModelSettings } from "@/components/Common/Settings/CurrentChatModelSettings"',
    )
    expect(playgroundFormSource).not.toContain(
      'import { ActorPopout } from "@/components/Common/Settings/ActorPopout"',
    )
    expect(playgroundFormSource).not.toContain(
      'import { DocumentGeneratorDrawer } from "@/components/Common/Playground/DocumentGeneratorDrawer"',
    )
    expect(playgroundFormSource).not.toContain(
      'import { PlaygroundImageGenModal } from "./PlaygroundImageGenModal"',
    )
    expect(playgroundFormSource).not.toContain(
      'import { PlaygroundRawRequestModal } from "./PlaygroundRawRequestModal"',
    )
    expect(playgroundFormSource).not.toContain(
      'import { PlaygroundStartupTemplateModal } from "./PlaygroundStartupTemplateModal"',
    )
    expect(playgroundFormSource).not.toContain(
      'import { PlaygroundContextWindowModal } from "./PlaygroundContextWindowModal"',
    )
    expect(playgroundFormSource).not.toContain(
      'import { PlaygroundMcpSettingsModal } from "./PlaygroundMcpSettingsModal"',
    )
    expect(playgroundFormSource).not.toContain(
      'import { VoiceModeSelector } from "./VoiceModeSelector"',
    )
    expect(playgroundFormSource).toContain(
      'import("@/components/Common/Settings/CurrentChatModelSettings")',
    )
    expect(playgroundFormSource).toContain(
      'import("@/components/Common/Settings/ActorPopout")',
    )
    expect(playgroundFormSource).toContain(
      'import("@/components/Common/Playground/DocumentGeneratorDrawer")',
    )
    expect(playgroundFormSource).toContain(
      'import("./PlaygroundImageGenModal")',
    )
    expect(playgroundFormSource).toContain(
      'import("./PlaygroundRawRequestModal")',
    )
    expect(playgroundFormSource).toContain(
      'import("./PlaygroundStartupTemplateModal")',
    )
    expect(playgroundFormSource).toContain(
      'import("./PlaygroundContextWindowModal")',
    )
    expect(playgroundFormSource).toContain(
      'import("./PlaygroundMcpSettingsModal")',
    )
    expect(playgroundFormSource).toContain(
      'import("./VoiceModeSelector")',
    )
  })

  it("defers compare-only chat rendering behind a chat-local lazy boundary", () => {
    expect(playgroundChatSource).not.toContain(
      'import { ProviderIcons } from "@/components/Common/ProviderIcon"',
    )
    expect(playgroundChatSource).not.toContain(
      'import { Clock, DollarSign, Hash } from "lucide-react"',
    )
    expect(playgroundChatSource).not.toContain(
      'import { decodeChatErrorPayload } from "@/utils/chat-error-message"',
    )
    expect(playgroundChatSource).not.toContain(
      'import { humanizeMilliseconds } from "@/utils/humanize-milliseconds"',
    )
    expect(playgroundChatSource).not.toContain(
      'import { resolveMessageCostUsd } from "@/components/Common/Playground/message-usage"',
    )
    expect(playgroundChatSource).not.toContain(
      'import { formatCost } from "@/utils/model-pricing"',
    )
    expect(playgroundChatSource).not.toContain(
      'import { tldwModels } from "@/services/tldw"',
    )
    expect(playgroundChatSource).not.toContain(
      'import {\n  buildNormalizedPreview,\n  computeNormalizedPreviewBudget\n} from "./compare-normalized-preview"',
    )
    expect(playgroundChatSource).not.toContain(
      'import { computeResponseDiffPreview } from "./compare-response-diff"',
    )
    expect(playgroundChatSource).toContain(
      'import("./PlaygroundCompareCluster")',
    )
    expect(playgroundCompareClusterSource).toContain(
      'import { ProviderIcons } from "@/components/Common/ProviderIcon"',
    )
    expect(playgroundCompareClusterSource).toContain(
      'import { Clock, DollarSign, Hash } from "lucide-react"',
    )
    expect(playgroundCompareClusterSource).toContain(
      'import { decodeChatErrorPayload } from "@/utils/chat-error-message"',
    )
    expect(playgroundCompareClusterSource).toContain(
      'import { humanizeMilliseconds } from "@/utils/humanize-milliseconds"',
    )
    expect(playgroundCompareClusterSource).toContain(
      'import { resolveMessageCostUsd } from "@/components/Common/Playground/message-usage"',
    )
    expect(playgroundCompareClusterSource).toContain(
      'import { formatCost } from "@/utils/model-pricing"',
    )
    expect(playgroundCompareClusterSource).toContain(
      'import { tldwModels } from "@/services/tldw"',
    )
    expect(playgroundCompareClusterSource).toContain(
      'computeNormalizedPreviewBudget',
    )
    expect(playgroundCompareClusterSource).toContain(
      'computeResponseDiffPreview',
    )
  })

  it("keeps linked research status chrome off the default chat transcript path", () => {
    expect(playgroundChatSource).not.toContain(
      'import { ResearchRunStatusStack } from "./ResearchRunStatusStack"',
    )
    expect(playgroundChatSource).toContain('import("./ResearchRunStatusStack")')
  })

  it("keeps character-greeting picking off the default chat transcript path", () => {
    expect(playgroundChatSource).not.toContain(
      'import { ChatGreetingPicker } from "@/components/Common/ChatGreetingPicker"',
    )
    expect(playgroundChatSource).toContain(
      'import("@/components/Common/ChatGreetingPicker")',
    )
  })

  it("defers non-default workflow editor side panels behind sidebar selection boundaries", () => {
    expect(workflowEditorSource).not.toContain(
      'import { NodeConfigPanel } from "./NodeConfigPanel"',
    )
    expect(workflowEditorSource).not.toContain(
      'import { ExecutionPanel } from "./ExecutionPanel"',
    )
    expect(workflowEditorSource).toContain('import("./NodeConfigPanel")')
    expect(workflowEditorSource).toContain('import("./ExecutionPanel")')
    expect(workflowEditorIndexSource).not.toContain(
      'export { NodeConfigPanel } from "./NodeConfigPanel"',
    )
    expect(workflowEditorIndexSource).not.toContain(
      'export { ExecutionPanel } from "./ExecutionPanel"',
    )
  })

  it("defers closed-default media route overlays and library tools behind lazy boundaries", () => {
    expect(viewMediaPageSource).not.toContain(
      "import { JumpToNavigator } from '@/components/Media/JumpToNavigator'",
    )
    expect(viewMediaPageSource).not.toContain(
      "import { KeyboardShortcutsOverlay } from '@/components/Media/KeyboardShortcutsOverlay'",
    )
    expect(viewMediaPageSource).not.toContain(
      "import { MediaIngestJobsPanel } from '@/components/Media/MediaIngestJobsPanel'",
    )
    expect(viewMediaPageSource).not.toContain(
      "import { MediaLibraryStatsPanel } from '@/components/Media/MediaLibraryStatsPanel'",
    )
    expect(viewMediaPageSource).toContain(
      "import('@/components/Media/JumpToNavigator')",
    )
    expect(viewMediaPageSource).toContain(
      "import('@/components/Media/KeyboardShortcutsOverlay')",
    )
    expect(viewMediaPageSource).toContain(
      "import('@/components/Media/MediaIngestJobsPanel')",
    )
    expect(viewMediaPageSource).toContain(
      "import('@/components/Media/MediaLibraryStatsPanel')",
    )
  })

  it("keeps closed-default media viewer modals and dev-only tools behind lazy boundaries", () => {
    expect(contentViewerSource).not.toContain(
      "import { AnalysisModal } from './AnalysisModal'",
    )
    expect(contentViewerSource).not.toContain(
      "import { AnalysisEditModal } from './AnalysisEditModal'",
    )
    expect(contentViewerSource).not.toContain(
      "import { DeveloperToolsSection } from './DeveloperToolsSection'",
    )
    expect(contentViewerSource).not.toContain(
      "import { DiffViewModal } from './DiffViewModal'",
    )
    expect(contentViewerSource).toContain("import('./AnalysisModal')")
    expect(contentViewerSource).toContain("import('./AnalysisEditModal')")
    expect(contentViewerSource).toContain("import('./DeveloperToolsSection')")
    expect(contentViewerSource).toContain("import('./DiffViewModal')")
  })

  it("keeps collapsed-by-default version history off the base media viewer path", () => {
    expect(contentViewerSource).not.toContain(
      "import { VersionHistoryPanel } from './VersionHistoryPanel'",
    )
    expect(contentViewerSource).toContain("import('./VersionHistoryPanel')")
  })

  it("keeps collapsed-by-default document intelligence off the base media viewer path", () => {
    expect(contentViewerSource).not.toContain(
      "const renderDocumentIntelligencePanel = () => {",
    )
    expect(contentViewerSource).toContain(
      "import('./ContentViewerDocumentIntelligenceSection')",
    )
  })

  it("keeps closed-default export and schedule-refresh modals off the base media viewer path", () => {
    expect(contentViewerSource).not.toContain(
      'data-testid="media-export-modal"',
    )
    expect(contentViewerSource).not.toContain(
      'data-testid="media-schedule-refresh-modal"',
    )
    expect(contentViewerSource).toContain(
      "import('./ContentViewerActionModals')",
    )
  })

  it("keeps the collapsed metadata section body off the base media viewer path", () => {
    expect(contentViewerSource).not.toContain(
      "review:mediaPage.idLabel",
    )
    expect(contentViewerSource).toContain(
      "import('./ContentViewerMetadataSectionBody')",
    )
  })

  it("keeps the closed content edit modal off the base media viewer path", () => {
    expect(contentViewerSource).not.toContain(
      '{selectedMedia && !isNote && (\n        <Suspense fallback={null}>\n          <ContentEditModal',
    )
    expect(contentViewerSource).toContain(
      'selectedMedia && !isNote && editState.contentEditModalOpen ? (',
    )
  })

  it("keeps markdown rendering off the base media viewer path until markdown mode is active", () => {
    expect(contentViewerSource).not.toContain(
      'import { MarkdownPreview } from \'@/components/Common/MarkdownPreview\'',
    )
    expect(contentViewerSource).toContain(
      'import(\'@/components/Common/MarkdownPreview\')',
    )
  })

  it("keeps the closed-default bulk toolbar off the base media route body", () => {
    expect(viewMediaPageSource).not.toContain(
      'data-testid="media-bulk-toolbar"',
    )
    expect(viewMediaPageSource).toContain(
      "import('./MediaBulkToolbar')",
    )
  })

  it("keeps the conditional media section navigator off the base media route body", () => {
    expect(viewMediaPageSource).not.toContain(
      'import { MediaSectionNavigator } from \'@/components/Media/MediaSectionNavigator\'',
    )
    expect(viewMediaPageSource).toContain(
      "import('@/components/Media/MediaSectionNavigator')",
    )
  })

  it("keeps sidepanel-only drawers and modals behind lazy boundaries instead of importing them into the chat route", () => {
    expect(sidepanelChatSource).not.toContain(
      'import { SidepanelChatSidebar } from "~/components/Sidepanel/Chat/Sidebar"',
    )
    expect(sidepanelChatSource).not.toContain(
      'import NoteQuickSaveModal from "~/components/Sidepanel/Notes/NoteQuickSaveModal"',
    )
    expect(sidepanelChatSource).toContain(
      'import("~/components/Sidepanel/Chat/Sidebar")',
    )
    expect(sidepanelChatSource).toContain(
      'import("~/components/Sidepanel/Notes/NoteQuickSaveModal")',
    )
  })

  it("mounts the sidepanel command palette through the lazy host instead of importing it into the route", () => {
    expect(sidepanelChatSource).not.toContain(
      'import { CommandPalette } from "@/components/Common/CommandPalette"',
    )
    expect(sidepanelChatSource).toContain(
      'from "@/components/Common/CommandPaletteHost"',
    )
    expect(sidepanelChatSource).toContain("<CommandPaletteHost")
  })

  it("defers prompt studio and closed prompt surfaces behind lazy boundaries", () => {
    expect(promptsBodySource).not.toContain(
      'import { PromptDrawer } from "./PromptDrawer"',
    )
    expect(promptsBodySource).not.toContain(
      'import { ConflictResolutionModal } from "./ConflictResolutionModal"',
    )
    expect(promptsBodySource).not.toContain(
      'import { PromptInspectorPanel } from "./PromptInspectorPanel"',
    )
    expect(promptsBodySource).not.toContain(
      'import { PromptFullPageEditor } from "./PromptFullPageEditor"',
    )
    expect(promptsBodySource).not.toContain(
      'import { ProjectSelector } from "./ProjectSelector"',
    )
    expect(promptsBodySource).not.toContain(
      'import { StudioTabContainer } from "./Studio/StudioTabContainer"',
    )
    expect(promptsBodySource).toContain('import("./PromptDrawer")')
    expect(promptsBodySource).toContain('import("./ConflictResolutionModal")')
    expect(promptsBodySource).toContain('import("./PromptInspectorPanel")')
    expect(promptsBodySource).toContain('import("./PromptFullPageEditor")')
    expect(promptsBodySource).toContain('import("./ProjectSelector")')
    expect(promptsBodySource).toContain('import("./Studio/StudioTabContainer")')
    expect(promptsBodySource).toContain("renderPromptDrawer")
    expect(promptsBodySource).toContain("renderStudioTabContainer")
  })

  it("keeps gallery-only and empty-state prompt surfaces off the default prompt route body", () => {
    expect(promptsBodySource).not.toContain(
      'import {\n  PromptGalleryCard,',
    )
    expect(promptsBodySource).not.toContain(
      'import { PromptStarterCards } from "./PromptStarterCards"',
    )
    expect(promptsBodySource).not.toContain(
      'import { ContextualHint } from "./ContextualHint"',
    )
    expect(promptsBodySource).toContain('import("./PromptGalleryCard")')
    expect(promptsBodySource).toContain('import("./PromptStarterCards")')
    expect(promptsBodySource).toContain('import("./ContextualHint")')
  })

  it("keeps non-default prompt studio tabs behind lazy sub-tab boundaries", () => {
    expect(studioTabContainerSource).not.toContain(
      'import { StudioPromptsTab } from "./Prompts/StudioPromptsTab"',
    )
    expect(studioTabContainerSource).not.toContain(
      'import { TestCasesTab } from "./TestCases/TestCasesTab"',
    )
    expect(studioTabContainerSource).not.toContain(
      'import { EvaluationsTab } from "./Evaluations/EvaluationsTab"',
    )
    expect(studioTabContainerSource).not.toContain(
      'import { OptimizationsTab } from "./Optimizations/OptimizationsTab"',
    )
    expect(studioTabContainerSource).toContain(
      'import("./Prompts/StudioPromptsTab")',
    )
    expect(studioTabContainerSource).toContain(
      'import("./TestCases/TestCasesTab")',
    )
    expect(studioTabContainerSource).toContain(
      'import("./Evaluations/EvaluationsTab")',
    )
    expect(studioTabContainerSource).toContain(
      'import("./Optimizations/OptimizationsTab")',
    )
    expect(studioTabContainerSource).toContain("renderStudioSubTab")
  })

  it("defers secondary workspace playground panes behind route-local lazy boundaries", () => {
    expect(workspacePlaygroundSource).not.toContain(
      'import { SourcesPane } from "./SourcesPane"',
    )
    expect(workspacePlaygroundSource).not.toContain(
      'import { StudioPane } from "./StudioPane"',
    )
    expect(workspacePlaygroundSource).toContain('import("./SourcesPane")')
    expect(workspacePlaygroundSource).toContain('import("./StudioPane")')
    expect(workspacePlaygroundSource).toContain("renderSourcesPane")
    expect(workspacePlaygroundSource).toContain("renderStudioPane")
  })

  it("defers workspace studio artifact modal surfaces behind a lazy modal boundary", () => {
    expect(studioPaneSource).not.toContain('const MindMapArtifactViewer: React.FC<')
    expect(studioPaneSource).not.toContain('const DataTableArtifactViewer: React.FC<')
    expect(studioPaneSource).not.toContain('const FlashcardArtifactEditor: React.FC<')
    expect(studioPaneSource).not.toContain('const QuizArtifactEditor: React.FC<')
    expect(studioPaneSource).toContain(
      'import("./ArtifactModalContent")',
    )
    expect(studioPaneSource).toContain("renderArtifactModalContent")
  })

  it("keeps quick notes behind a closed-default lazy boundary in the workspace studio pane", () => {
    expect(studioPaneSource).not.toContain(
      'import { QuickNotesSection } from "./QuickNotesSection"',
    )
    expect(studioPaneSource).toContain('import("./QuickNotesSection")')
    expect(studioPaneSource).toContain("renderQuickNotesSection")
  })

  it("defers closed-default knowledge route surfaces behind lazy boundaries", () => {
    expect(knowledgeQaIndexSource).not.toContain(
      'import { SettingsPanel } from "./SettingsPanel"',
    )
    expect(knowledgeQaIndexSource).not.toContain(
      'import { ExportDialog } from "./ExportDialog"',
    )
    expect(knowledgeQaIndexSource).toContain('import("./SettingsPanel")')
    expect(knowledgeQaIndexSource).toContain('import("./ExportDialog")')
  })

  it("keeps non-primary knowledge layout surfaces behind lazy boundaries", () => {
    expect(knowledgeQaLayoutSource).not.toContain(
      'import { HistoryPane } from "../history/HistoryPane"',
    )
    expect(knowledgeQaLayoutSource).not.toContain(
      'import { InlineRecentSessions } from "../empty/InlineRecentSessions"',
    )
    expect(knowledgeQaLayoutSource).not.toContain(
      'import { NoResultsRecovery } from "../panels/NoResultsRecovery"',
    )
    expect(knowledgeQaLayoutSource).not.toContain(
      'import { EvidenceRail } from "../evidence/EvidenceRail"',
    )
    expect(knowledgeQaLayoutSource).toContain('import("../history/HistoryPane")')
    expect(knowledgeQaLayoutSource).toContain('import("../empty/InlineRecentSessions")')
    expect(knowledgeQaLayoutSource).toContain('import("../panels/NoResultsRecovery")')
    expect(knowledgeQaLayoutSource).toContain('import("../evidence/EvidenceRail")')
  })

  it("keeps the full-source viewer modal off the base knowledge source list path", () => {
    expect(knowledgeQaSourceListSource).not.toContain(
      'import { SourceViewerModal } from "./SourceViewerModal"',
    )
    expect(knowledgeQaSourceListSource).toContain('import("./SourceViewerModal")')
  })

  it("keeps details-only evidence inspector content behind a lazy boundary", () => {
    expect(knowledgeQaEvidenceRailSource).not.toContain(
      'import { SearchDetailsPanel } from "../SearchDetailsPanel"',
    )
    expect(knowledgeQaEvidenceRailSource).toContain('import("../SearchDetailsPanel")')
  })

  it("keeps notes manager overlay-only modals behind a lazy boundary", () => {
    expect(notesManagerPageSource).not.toContain('data-testid="notes-keyword-manager-modal"')
    expect(notesManagerPageSource).not.toContain('data-testid="notes-import-modal"')
    expect(notesManagerPageSource).toContain('import("./NotesManagerOverlays")')
  })

  it("keeps markdown preview rendering off the default notes editor path until preview modes are active", () => {
    expect(notesEditorPaneSource).not.toContain(
      'import { MarkdownPreview } from \'@/components/Common/MarkdownPreview\'',
    )
    expect(notesEditorPaneSource).toContain(
      'import(\'@/components/Common/MarkdownPreview\')',
    )
  })

  it("keeps notes list empty and offline helper surfaces off the default list path", () => {
    expect(notesListPanelSource).not.toContain(
      "import FeatureEmptyState from '@/components/Common/FeatureEmptyState'",
    )
    expect(notesListPanelSource).not.toContain(
      "import ConnectionProblemBanner from '@/components/Common/ConnectionProblemBanner'",
    )
    expect(notesListPanelSource).toContain("import('./NotesListPanelEmptyStates')")
  })

  it("defers closed-default character dialogs behind a lazy boundary", () => {
    expect(charactersManagerSource).not.toContain(
      'import { CharacterDialogs } from "./CharacterDialogs"',
    )
    expect(charactersManagerSource).toContain('import("./CharacterDialogs")')
  })

  it("keeps the heavy character editor form off the default manager route body", () => {
    expect(charactersManagerSource).not.toContain(
      'import { CharacterPreview } from "./CharacterPreview"',
    )
    expect(charactersManagerSource).not.toContain(
      'import { GenerateFieldButton } from "./GenerateFieldButton"',
    )
    expect(charactersManagerSource).not.toContain(
      "const renderNameField = React.useCallback(",
    )
    expect(charactersManagerSource).not.toContain(
      "const renderAdvancedFields = (",
    )
    expect(charactersManagerSource).toContain('import("./CharacterEditorForm")')
  })

  it("keeps character editor-only prompt preset and example payload off the default manager route body", () => {
    expect(charactersManagerSource).not.toContain("CHARACTER_PROMPT_PRESETS")
    expect(charactersManagerSource).not.toContain("SYSTEM_PROMPT_EXAMPLE")
    expect(characterEditorFormSource).toContain("CHARACTER_PROMPT_PRESETS")
    expect(characterEditorFormSource).toContain("SYSTEM_PROMPT_EXAMPLE")
  })

  it("keeps gallery-only character surfaces off the default list route body", () => {
    expect(characterListContentSource).not.toContain(
      'import { CharacterGalleryCard',
    )
    expect(characterListContentSource).not.toContain(
      'import { CharacterPreviewPopup } from "./CharacterPreviewPopup"',
    )
    expect(characterListContentSource).toContain('import("./CharacterGalleryCard")')
    expect(characterListContentSource).toContain('import("./CharacterPreviewPopup")')
  })

  it("keeps non-default quiz tabs behind lazy tab-selection boundaries", () => {
    expect(quizPlaygroundSource).toContain(
      'import { TakeQuizTab } from "./tabs/TakeQuizTab"',
    )
    expect(quizPlaygroundSource).not.toContain(
      'import { TakeQuizTab, GenerateTab, CreateTab, ManageTab, ResultsTab } from "./tabs"',
    )
    expect(quizPlaygroundSource).not.toContain(
      'import { GenerateTab } from "./tabs/GenerateTab"',
    )
    expect(quizPlaygroundSource).not.toContain(
      'import { CreateTab } from "./tabs/CreateTab"',
    )
    expect(quizPlaygroundSource).not.toContain(
      'import { ManageTab } from "./tabs/ManageTab"',
    )
    expect(quizPlaygroundSource).not.toContain(
      'import { ResultsTab } from "./tabs/ResultsTab"',
    )
    expect(quizPlaygroundSource).toContain('import("./tabs/GenerateTab")')
    expect(quizPlaygroundSource).toContain('import("./tabs/CreateTab")')
    expect(quizPlaygroundSource).toContain('import("./tabs/ManageTab")')
    expect(quizPlaygroundSource).toContain('import("./tabs/ResultsTab")')
  })

  it("keeps closed-default writing workspace modals behind a lazy modal host", () => {
    expect(writingPlaygroundSource).not.toContain(
      'title={t("option:writingPlayground.extraBodyJsonModalTitle"',
    )
    expect(writingPlaygroundSource).not.toContain(
      'title={t("option:writingPlayground.contextPreviewTitle"',
    )
    expect(writingPlaygroundSource).not.toContain(
      'title={t("option:writingPlayground.templatesModalTitle"',
    )
    expect(writingPlaygroundSource).not.toContain(
      'title={t("option:writingPlayground.themesModalTitle"',
    )
    expect(writingPlaygroundSource).not.toContain(
      'title={t("option:writingPlayground.createSessionTitle"',
    )
    expect(writingPlaygroundSource).not.toContain(
      'title={t("option:writingPlayground.renameSessionTitle"',
    )
    expect(writingPlaygroundSource).toContain(
      'import("./WritingPlaygroundModalHost")',
    )
  })
})

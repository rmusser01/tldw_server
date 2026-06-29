import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import {
  expectInsideDesignSystemAlert,
  expectInsideDesignSystemAlertAsync
} from "@/test-utils/designSystemAlert"

const dexieMocks = vi.hoisted(() => ({
  listProjects: vi.fn(),
  serializeProject: vi.fn(),
  markMigrated: vi.fn()
}))

const migrationMocks = vi.hoisted(() => ({
  previewMutateAsync: vi.fn(),
  commitMutateAsync: vi.fn(),
  previewPending: false,
  commitPending: false
}))

const routerMocks = vi.hoisted(() => ({
  navigate: vi.fn()
}))

vi.mock("@/db/dexie/audiobook-projects", () => ({
  listLegacyAudiobookProjectsForMigration: (...args: unknown[]) =>
    dexieMocks.listProjects(...args),
  serializeLegacyAudiobookProjectForMigration: (...args: unknown[]) =>
    dexieMocks.serializeProject(...args),
  markLegacyAudiobookProjectMigrated: (...args: unknown[]) =>
    dexieMocks.markMigrated(...args)
}))

vi.mock("@/hooks/useAudioStudioMigration", () => ({
  usePreviewAudiobookMigration: () => ({
    mutateAsync: migrationMocks.previewMutateAsync,
    isPending: migrationMocks.previewPending
  }),
  useCommitAudiobookMigration: () => ({
    mutateAsync: migrationMocks.commitMutateAsync,
    isPending: migrationMocks.commitPending
  })
}))

vi.mock("react-router-dom", () => ({
  useNavigate: () => routerMocks.navigate
}))

import {
  AUDIOBOOK_COMPATIBILITY_TARGET,
  CompatibilityRedirect
} from "../CompatibilityRedirect"
import { MigrationBanner } from "../MigrationBanner"

const legacyProject = {
  id: "legacy-1",
  title: "Local Audiobook",
  chapters: [{ id: "chapter-1" }, { id: "chapter-2" }],
  updatedAt: 100
}

const serializedProject = {
  migration_schema_version: 1,
  legacy_project_id: "legacy-1",
  title: "Local Audiobook",
  chapters: [{ legacy_chapter_id: "chapter-1" }],
  audio_assets: []
}

describe("CompatibilityRedirect", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    migrationMocks.previewPending = false
    migrationMocks.commitPending = false
    dexieMocks.listProjects.mockResolvedValue([legacyProject])
    dexieMocks.serializeProject.mockResolvedValue(serializedProject)
    migrationMocks.previewMutateAsync.mockResolvedValue({
      preview_id: "preview-1",
      fingerprint: "fingerprint-1",
      workflow: "narration",
      project_count: 1,
      section_count: 2,
      audio_reference_count: 0,
      needs_regeneration_count: 1,
      warnings: []
    })
    migrationMocks.commitMutateAsync.mockResolvedValue({
      project: {
        project_id: "server-project-1",
        title: "Local Audiobook",
        workflow: "narration",
        status: "draft"
      },
      imported_section_count: 2,
      audio_reference_count: 0,
      needs_regeneration_count: 1,
      fingerprint: "fingerprint-1",
      replayed: false
    })
  })

  it("preserves the legacy route redirect when no local projects exist", async () => {
    dexieMocks.listProjects.mockResolvedValueOnce([])

    render(<CompatibilityRedirect />)

    await waitFor(() =>
      expect(routerMocks.navigate).toHaveBeenCalledWith(
        AUDIOBOOK_COMPATIBILITY_TARGET,
        { replace: true }
      )
    )
  })

  it("renders static migration guidance through the design-system Alert primitive", () => {
    render(<MigrationBanner />)

    expectInsideDesignSystemAlert(
      "Audiobook projects can move into Audio Studio Narration"
    )
  })

  it("renders legacy project load errors through the design-system Alert primitive", async () => {
    dexieMocks.listProjects.mockRejectedValueOnce(new Error("Dexie unavailable"))

    render(<CompatibilityRedirect />)

    await expectInsideDesignSystemAlertAsync("Dexie unavailable")
  })

  it("shows a migration banner with local project preview when Dexie projects exist", async () => {
    render(<CompatibilityRedirect />)

    expect(
      await screen.findByRole("heading", {
        name: "Move local Audiobook projects into Audio Studio"
      })
    ).toBeInTheDocument()
    expect(screen.getByText("Local Audiobook")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Preview migration" })).toBeEnabled()
    expect(routerMocks.navigate).not.toHaveBeenCalled()
  })

  it("previews selected migrations through the service without marking local rows", async () => {
    render(<CompatibilityRedirect />)

    fireEvent.click(await screen.findByRole("button", { name: "Preview migration" }))

    await waitFor(() =>
      expect(migrationMocks.previewMutateAsync).toHaveBeenCalledWith({
        legacy_project_id: "legacy-1",
        project_payload: serializedProject
      })
    )
    expect(dexieMocks.markMigrated).not.toHaveBeenCalled()
    expect(
      await screen.findByText("1 project, 2 chapters, 0 audio assets")
    ).toBeInTheDocument()
    expectInsideDesignSystemAlert("Migration preview")
  })

  it("marks local projects only after commit succeeds and redirects to the migrated project", async () => {
    render(<CompatibilityRedirect />)

    fireEvent.click(await screen.findByRole("button", { name: "Preview migration" }))
    await screen.findByText("1 project, 2 chapters, 0 audio assets")
    fireEvent.click(screen.getByRole("button", { name: "Migrate selected" }))

    await waitFor(() =>
      expect(migrationMocks.commitMutateAsync).toHaveBeenCalledWith({
        idempotency_key: expect.stringMatching(/^audiobook-migration-/),
        legacy_project_id: "legacy-1",
        project_payload: serializedProject
      })
    )
    expect(dexieMocks.markMigrated).toHaveBeenCalledWith("legacy-1", {
      migrationId: "fingerprint-1",
      projectId: "server-project-1"
    })
    expect(routerMocks.navigate).toHaveBeenCalledWith(
      "/audio-studio?workflow=narration&project=server-project-1",
      { replace: true }
    )
  })

  it("does not mark local projects when commit fails", async () => {
    migrationMocks.commitMutateAsync.mockRejectedValueOnce(new Error("commit failed"))

    render(<CompatibilityRedirect />)

    fireEvent.click(await screen.findByRole("button", { name: "Preview migration" }))
    await screen.findByText("1 project, 2 chapters, 0 audio assets")
    fireEvent.click(screen.getByRole("button", { name: "Migrate selected" }))

    expect(
      await screen.findByText("commit failed")
    ).toBeInTheDocument()
    expectInsideDesignSystemAlert("commit failed")
    expect(dexieMocks.markMigrated).not.toHaveBeenCalled()
    expect(routerMocks.navigate).not.toHaveBeenCalled()
  })
})

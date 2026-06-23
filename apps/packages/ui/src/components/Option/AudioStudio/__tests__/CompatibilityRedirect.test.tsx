import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

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
      status: "preview",
      counts: { projects: 1, chapters: 2, audio_assets: 0 }
    })
    migrationMocks.commitMutateAsync.mockResolvedValue({
      status: "completed",
      migration_id: "migration-1",
      projects: [
        {
          legacy_project_id: "legacy-1",
          project_id: "server-project-1",
          status: "migrated"
        }
      ]
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
        projects: [serializedProject]
      })
    )
    expect(dexieMocks.markMigrated).not.toHaveBeenCalled()
    expect(
      await screen.findByText("1 project, 2 chapters, 0 audio assets")
    ).toBeInTheDocument()
  })

  it("marks local projects only after commit succeeds and redirects to the migrated project", async () => {
    render(<CompatibilityRedirect />)

    fireEvent.click(await screen.findByRole("button", { name: "Preview migration" }))
    await screen.findByText("1 project, 2 chapters, 0 audio assets")
    fireEvent.click(screen.getByRole("button", { name: "Migrate selected" }))

    await waitFor(() =>
      expect(migrationMocks.commitMutateAsync).toHaveBeenCalledWith({
        idempotency_key: expect.stringMatching(/^audiobook-migration-/),
        projects: [serializedProject]
      })
    )
    expect(dexieMocks.markMigrated).toHaveBeenCalledWith("legacy-1", {
      migrationId: "migration-1",
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
    expect(dexieMocks.markMigrated).not.toHaveBeenCalled()
    expect(routerMocks.navigate).not.toHaveBeenCalled()
  })
})

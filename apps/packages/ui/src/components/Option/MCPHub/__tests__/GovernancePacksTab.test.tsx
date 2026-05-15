// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"

const mocks = vi.hoisted(() => ({
  listGovernancePacks: vi.fn(),
  getGovernancePackDetail: vi.fn(),
  dryRunGovernancePack: vi.fn(),
  importGovernancePack: vi.fn(),
  prepareGovernancePackSourceCandidate: vi.fn(),
  dryRunGovernancePackSourceCandidate: vi.fn(),
  importGovernancePackSourceCandidate: vi.fn(),
  checkGovernancePackUpdates: vi.fn(),
  prepareGovernancePackUpgradeCandidate: vi.fn(),
  dryRunGovernancePackSourceUpgrade: vi.fn(),
  executeGovernancePackSourceUpgrade: vi.fn(),
  dryRunGovernancePackUpgrade: vi.fn(),
  executeGovernancePackUpgrade: vi.fn(),
  listGovernancePackUpgradeHistory: vi.fn()
}))

const designSystemLabels = vi.hoisted(() => ({
  blocked: "Registry Blocked"
}))

vi.mock("@/services/tldw/mcp-hub", () => ({
  listGovernancePacks: (...args: unknown[]) => mocks.listGovernancePacks(...args),
  getGovernancePackDetail: (...args: unknown[]) => mocks.getGovernancePackDetail(...args),
  dryRunGovernancePack: (...args: unknown[]) => mocks.dryRunGovernancePack(...args),
  importGovernancePack: (...args: unknown[]) => mocks.importGovernancePack(...args),
  prepareGovernancePackSourceCandidate: (...args: unknown[]) =>
    mocks.prepareGovernancePackSourceCandidate(...args),
  dryRunGovernancePackSourceCandidate: (...args: unknown[]) =>
    mocks.dryRunGovernancePackSourceCandidate(...args),
  importGovernancePackSourceCandidate: (...args: unknown[]) =>
    mocks.importGovernancePackSourceCandidate(...args),
  checkGovernancePackUpdates: (...args: unknown[]) => mocks.checkGovernancePackUpdates(...args),
  prepareGovernancePackUpgradeCandidate: (...args: unknown[]) =>
    mocks.prepareGovernancePackUpgradeCandidate(...args),
  dryRunGovernancePackSourceUpgrade: (...args: unknown[]) =>
    mocks.dryRunGovernancePackSourceUpgrade(...args),
  executeGovernancePackSourceUpgrade: (...args: unknown[]) =>
    mocks.executeGovernancePackSourceUpgrade(...args),
  dryRunGovernancePackUpgrade: (...args: unknown[]) => mocks.dryRunGovernancePackUpgrade(...args),
  executeGovernancePackUpgrade: (...args: unknown[]) => mocks.executeGovernancePackUpgrade(...args),
  listGovernancePackUpgradeHistory: (...args: unknown[]) => mocks.listGovernancePackUpgradeHistory(...args)
}))

vi.mock("@/design-system", async (importActual) => {
  const actual = await importActual<typeof import("@/design-system")>()

  return {
    ...actual,
    BLOCKED_STATE_LABEL: designSystemLabels.blocked,
    getDesignSystemState: vi.fn(
      (key: Parameters<typeof actual.getDesignSystemState>[0]) => {
        const state = actual.getDesignSystemState(key)

        if (key === "blocked") {
          return { ...state, label: designSystemLabels.blocked }
        }

        return state
      }
    )
  }
})

import { GovernancePacksTab } from "../GovernancePacksTab"

describe("GovernancePacksTab", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.listGovernancePacks.mockResolvedValue([
      {
        id: 81,
        pack_id: "researcher-pack",
        pack_version: "1.0.0",
        title: "Researcher Pack",
        description: "Portable research governance pack",
        owner_scope_type: "user",
        owner_scope_id: 7,
        bundle_digest: "a".repeat(64),
        source_type: "git",
        source_location: "https://github.com/example/researcher-pack.git",
        source_ref_requested: "main",
        source_subpath: "packs/researcher",
        source_commit_resolved: "abc123",
        pack_content_digest: "c".repeat(64),
        source_verified: true,
        source_verification_mode: "git_signature",
        signer_fingerprint: "ABCD1234",
        signer_identity: "Release Bot <bot@example.com>",
        verified_object_type: "commit",
        verification_result_code: "verified_and_trusted",
        verification_warning_code: null,
        is_active_install: true,
        manifest: {
          pack_id: "researcher-pack",
          pack_version: "1.0.0",
          title: "Researcher Pack"
        }
      },
      {
        id: 80,
        pack_id: "researcher-pack",
        pack_version: "0.9.0",
        title: "Researcher Pack",
        description: "Superseded pack",
        owner_scope_type: "user",
        owner_scope_id: 7,
        bundle_digest: "b".repeat(64),
        source_type: "git",
        source_location: "https://github.com/example/researcher-pack.git",
        is_active_install: false,
        superseded_by_governance_pack_id: 81,
        manifest: {
          pack_id: "researcher-pack",
          pack_version: "0.9.0",
          title: "Researcher Pack"
        }
      }
    ])
    mocks.getGovernancePackDetail.mockResolvedValue({
      id: 81,
      pack_id: "researcher-pack",
      pack_version: "1.0.0",
      title: "Researcher Pack",
      description: "Portable research governance pack",
      owner_scope_type: "user",
      owner_scope_id: 7,
      bundle_digest: "a".repeat(64),
      source_type: "git",
      source_location: "https://github.com/example/researcher-pack.git",
      source_ref_requested: "main",
      source_subpath: "packs/researcher",
      source_commit_resolved: "abc123",
      pack_content_digest: "c".repeat(64),
      source_verified: true,
      source_verification_mode: "git_signature",
      signer_fingerprint: "ABCD1234",
      signer_identity: "Release Bot <bot@example.com>",
      verified_object_type: "commit",
      verification_result_code: "verified_and_trusted",
      verification_warning_code: "signer_revoked",
      is_active_install: true,
      manifest: {
        pack_id: "researcher-pack",
        pack_version: "1.0.0",
        title: "Researcher Pack"
      },
      normalized_ir: {
        data: {
          profiles: [{ profile_id: "researcher.profile" }]
        }
      },
      imported_objects: [
        {
          object_type: "permission_profile",
          object_id: "5",
          source_object_id: "researcher.profile"
        }
      ]
    })
    mocks.listGovernancePackUpgradeHistory.mockResolvedValue([
      {
        id: 12,
        pack_id: "researcher-pack",
        owner_scope_type: "user",
        owner_scope_id: 7,
        from_governance_pack_id: 80,
        to_governance_pack_id: 81,
        from_pack_version: "0.9.0",
        to_pack_version: "1.0.0",
        status: "executed",
        plan_summary: {
          object_diff_count: 2,
          dependency_impact_count: 1
        },
        accepted_resolutions: {},
        planned_at: "2026-03-14T00:00:00Z",
        executed_at: "2026-03-14T00:05:00Z"
      }
    ])
    mocks.dryRunGovernancePackUpgrade.mockResolvedValue({
      plan: {
        source_governance_pack_id: 81,
        source_manifest: {
          pack_id: "researcher-pack",
          pack_version: "1.0.0",
          title: "Researcher Pack"
        },
        target_manifest: {
          pack_id: "researcher-pack",
          pack_version: "1.1.0",
          title: "Researcher Pack"
        },
        object_diff: [
          {
            object_type: "permission_profile",
            source_object_id: "researcher.profile",
            change_type: "modified",
            previous_digest: "1".repeat(64),
            next_digest: "2".repeat(64)
          }
        ],
        dependency_impact: [],
        structural_conflicts: [],
        behavioral_conflicts: [],
        warnings: [],
        planner_inputs_fingerprint: "plan-fingerprint",
        adapter_state_fingerprint: "adapter-fingerprint",
        upgradeable: true
      }
    })
    mocks.executeGovernancePackUpgrade.mockResolvedValue({
      upgrade_id: 13,
      source_governance_pack_id: 81,
      target_governance_pack_id: 82,
      from_pack_version: "1.0.0",
      to_pack_version: "1.1.0",
      planner_inputs_fingerprint: "plan-fingerprint",
      adapter_state_fingerprint: "adapter-fingerprint",
      imported_object_ids: {
        approval_policies: [31],
        permission_profiles: [41],
        policy_assignments: [51]
      },
      imported_object_counts: {
        approval_policies: 1,
        permission_profiles: 1,
        policy_assignments: 1
      }
    })
    mocks.dryRunGovernancePack.mockResolvedValue({
      report: {
        manifest: {
          pack_id: "researcher-pack",
          pack_version: "1.0.0",
          title: "Researcher Pack",
          description: "Portable research governance pack"
        },
        digest: "a".repeat(64),
        resolved_capabilities: ["filesystem.read", "tool.invoke.research"],
        unresolved_capabilities: [],
        warnings: [],
        blocked_objects: [],
        verdict: "importable"
      }
    })
    mocks.importGovernancePack.mockResolvedValue({
      governance_pack_id: 81,
      imported_object_counts: {
        approval_policies: 1,
        permission_profiles: 1,
        policy_assignments: 1
      },
      blocked_objects: [],
      report: {
        manifest: {
          pack_id: "researcher-pack",
          pack_version: "1.0.0",
          title: "Researcher Pack",
          description: "Portable research governance pack"
        },
        digest: "a".repeat(64),
        resolved_capabilities: ["filesystem.read", "tool.invoke.research"],
        unresolved_capabilities: [],
        warnings: [],
        blocked_objects: [],
        verdict: "importable"
      }
    })
    mocks.prepareGovernancePackSourceCandidate.mockResolvedValue({
      candidate: {
        id: 501,
        source_type: "local_path",
        source_location: "/srv/packs/researcher-pack",
        source_commit_resolved: null,
        pack_content_digest: "c".repeat(64)
      },
      manifest: {
        pack_id: "researcher-pack",
        pack_version: "1.0.0",
        title: "Researcher Pack",
        description: "Portable research governance pack"
      }
    })
    mocks.dryRunGovernancePackSourceCandidate.mockResolvedValue({
      report: {
        manifest: {
          pack_id: "researcher-pack",
          pack_version: "1.0.0",
          title: "Researcher Pack",
          description: "Portable research governance pack"
        },
        digest: "c".repeat(64),
        resolved_capabilities: ["filesystem.read", "tool.invoke.research"],
        unresolved_capabilities: [],
        warnings: [],
        blocked_objects: [],
        verdict: "importable"
      }
    })
    mocks.importGovernancePackSourceCandidate.mockResolvedValue({
      governance_pack_id: 81,
      imported_object_counts: {
        approval_policies: 1,
        permission_profiles: 1,
        policy_assignments: 1
      },
      blocked_objects: [],
      report: {
        manifest: {
          pack_id: "researcher-pack",
          pack_version: "1.0.0",
          title: "Researcher Pack",
          description: "Portable research governance pack"
        },
        digest: "c".repeat(64),
        resolved_capabilities: ["filesystem.read", "tool.invoke.research"],
        unresolved_capabilities: [],
        warnings: [],
        blocked_objects: [],
        verdict: "importable"
      }
    })
    mocks.checkGovernancePackUpdates.mockResolvedValue({
      governance_pack_id: 81,
      status: "newer_version_available",
      installed_manifest: {
        pack_id: "researcher-pack",
        pack_version: "1.0.0",
        title: "Researcher Pack",
        description: "Portable research governance pack"
      },
      candidate_manifest: {
        pack_id: "researcher-pack",
        pack_version: "1.1.0",
        title: "Researcher Pack",
        description: "Portable research governance pack"
      },
      source_commit_resolved: "def456",
      pack_content_digest: "d".repeat(64),
      signer_fingerprint: "BBBB2222",
      signer_identity: "Backup Bot <backup@example.com>",
      verified_object_type: "commit",
      verification_result_code: "verified_and_trusted",
      verification_warning_code: "signer_rotated_trusted"
    })
    mocks.prepareGovernancePackUpgradeCandidate.mockResolvedValue({
      status: "newer_version_available",
      installed_manifest: {
        pack_id: "researcher-pack",
        pack_version: "1.0.0",
        title: "Researcher Pack",
        description: "Portable research governance pack"
      },
      candidate_manifest: {
        pack_id: "researcher-pack",
        pack_version: "1.1.0",
        title: "Researcher Pack",
        description: "Portable research governance pack"
      },
      candidate: {
        id: 502,
        source_type: "git",
        source_location: "https://github.com/example/researcher-pack.git",
        source_ref_requested: "main",
        source_subpath: "packs/researcher",
        source_commit_resolved: "def456",
        pack_content_digest: "d".repeat(64),
        source_verified: true,
        source_verification_mode: "git_signature",
        signer_fingerprint: "BBBB2222",
        signer_identity: "Backup Bot <backup@example.com>",
        verified_object_type: "commit",
        verification_result_code: "verified_and_trusted",
        verification_warning_code: "signer_rotated_trusted"
      },
      manifest: {
        pack_id: "researcher-pack",
        pack_version: "1.1.0",
        title: "Researcher Pack",
        description: "Portable research governance pack"
      }
    })
    mocks.dryRunGovernancePackSourceUpgrade.mockResolvedValue({
      plan: {
        source_governance_pack_id: 81,
        source_manifest: {
          pack_id: "researcher-pack",
          pack_version: "1.0.0",
          title: "Researcher Pack"
        },
        target_manifest: {
          pack_id: "researcher-pack",
          pack_version: "1.1.0",
          title: "Researcher Pack"
        },
        object_diff: [
          {
            object_type: "permission_profile",
            source_object_id: "researcher.profile",
            change_type: "modified",
            previous_digest: "1".repeat(64),
            next_digest: "2".repeat(64)
          }
        ],
        dependency_impact: [],
        structural_conflicts: [],
        behavioral_conflicts: [],
        warnings: [],
        planner_inputs_fingerprint: "plan-fingerprint",
        adapter_state_fingerprint: "adapter-fingerprint",
        upgradeable: true
      }
    })
    mocks.executeGovernancePackSourceUpgrade.mockResolvedValue({
      upgrade_id: 13,
      source_governance_pack_id: 81,
      target_governance_pack_id: 82,
      from_pack_version: "1.0.0",
      to_pack_version: "1.1.0",
      planner_inputs_fingerprint: "plan-fingerprint",
      adapter_state_fingerprint: "adapter-fingerprint",
      imported_object_ids: {
        approval_policies: [31],
        permission_profiles: [41],
        policy_assignments: [51]
      },
      imported_object_counts: {
        approval_policies: 1,
        permission_profiles: 1,
        policy_assignments: 1
      }
    })
  })

  it("renders dry-run compatibility findings before import", async () => {
    const user = userEvent.setup()
    render(<GovernancePacksTab />)

    expect((await screen.findAllByText("Researcher Pack")).length).toBeGreaterThan(0)

    fireEvent.change(screen.getByLabelText(/governance pack json/i), {
      target: {
        value: JSON.stringify({
          manifest: {
            pack_id: "researcher-pack",
            pack_version: "1.0.0",
            pack_schema_version: 1,
            capability_taxonomy_version: 1,
            adapter_contract_version: 1,
            title: "Researcher Pack"
          },
          profiles: [
            {
              profile_id: "researcher.profile",
              name: "Researcher",
              capabilities: { allow: ["filesystem.read", "tool.invoke.research"] },
              approval_intent: "ask",
              environment_requirements: ["workspace_bounded_read"]
            }
          ],
          approvals: [
            { approval_template_id: "researcher.ask", name: "Ask Before Use", mode: "ask" }
          ],
          personas: [],
          assignments: []
        })
      }
    })
    await user.click(screen.getByRole("button", { name: /preview pack/i }))

    expect(await screen.findByText("Resolved capabilities")).toBeTruthy()
    expect(screen.getByText("tool.invoke.research")).toBeTruthy()
    expect(screen.getByText("Importable")).toBeTruthy()
  })

  it("renders blocked dry-run and upgrade statuses through design-system state labels", async () => {
    mocks.dryRunGovernancePack.mockResolvedValueOnce({
      report: {
        manifest: {
          pack_id: "researcher-pack",
          pack_version: "1.0.0",
          title: "Researcher Pack",
          description: "Portable research governance pack"
        },
        digest: "a".repeat(64),
        resolved_capabilities: ["filesystem.read"],
        unresolved_capabilities: ["tool.invoke.research"],
        warnings: [],
        blocked_objects: ["permission_profile:researcher.profile"],
        verdict: "blocked"
      }
    })
    mocks.dryRunGovernancePackUpgrade.mockResolvedValueOnce({
      plan: {
        source_governance_pack_id: 81,
        source_manifest: {
          pack_id: "researcher-pack",
          pack_version: "1.0.0",
          title: "Researcher Pack"
        },
        target_manifest: {
          pack_id: "researcher-pack",
          pack_version: "1.1.0",
          title: "Researcher Pack"
        },
        object_diff: [],
        dependency_impact: [],
        structural_conflicts: [],
        behavioral_conflicts: ["permission_profile:researcher.profile changed"],
        warnings: [],
        planner_inputs_fingerprint: "plan-fingerprint",
        adapter_state_fingerprint: "adapter-fingerprint",
        upgradeable: false
      }
    })
    const user = userEvent.setup()
    render(<GovernancePacksTab />)

    expect((await screen.findAllByText("Researcher Pack")).length).toBeGreaterThan(0)
    fireEvent.change(screen.getByLabelText(/governance pack json/i), {
      target: {
        value: JSON.stringify({
          manifest: {
            pack_id: "researcher-pack",
            pack_version: "1.0.0",
            pack_schema_version: 1,
            capability_taxonomy_version: 1,
            adapter_contract_version: 1,
            title: "Researcher Pack"
          },
          profiles: [],
          approvals: [],
          personas: [],
          assignments: []
        })
      }
    })

    await user.click(screen.getByRole("button", { name: /preview pack/i }))

    expect(await screen.findByText(designSystemLabels.blocked)).toBeTruthy()
    expect(screen.queryByText("Blocked")).toBeNull()

    fireEvent.change(screen.getByLabelText(/governance pack json/i), {
      target: {
        value: JSON.stringify({
          manifest: {
            pack_id: "researcher-pack",
            pack_version: "1.1.0",
            pack_schema_version: 1,
            capability_taxonomy_version: 1,
            adapter_contract_version: 1,
            title: "Researcher Pack"
          },
          profiles: [],
          approvals: [],
          personas: [],
          assignments: []
        })
      }
    })

    await user.click(screen.getByRole("button", { name: /preview upgrade/i }))

    expect(await screen.findByRole("dialog")).toBeTruthy()
    expect(await screen.findAllByText(designSystemLabels.blocked)).toHaveLength(2)
    expect(screen.queryByText("Blocked")).toBeNull()
  })

  it("surfaces detail load failures without clearing the inventory", async () => {
    mocks.getGovernancePackDetail.mockRejectedValueOnce(new Error("boom"))

    render(<GovernancePacksTab />)

    expect((await screen.findAllByText("Researcher Pack")).length).toBeGreaterThan(0)
    expect(await screen.findByText("Failed to load pack details.")).toBeTruthy()
    expect(screen.getByText("Installed Packs")).toBeTruthy()
  })

  it("handles non-array inventory responses without crashing", async () => {
    mocks.listGovernancePacks.mockResolvedValueOnce({ rows: [] })

    render(<GovernancePacksTab />)

    expect(await screen.findByText("No governance packs imported yet")).toBeTruthy()
    expect(screen.getByText("Installed Packs")).toBeTruthy()
  })

  it("renders install state badges and upgrade history for the selected pack", async () => {
    const user = userEvent.setup()
    render(<GovernancePacksTab />)

    expect(await screen.findByText("Active install")).toBeTruthy()
    expect(await screen.findByText("Inactive install")).toBeTruthy()
    expect(await screen.findByText("Git Source")).toBeTruthy()
    expect(await screen.findByText("Verified Commit")).toBeTruthy()
    await user.click(await screen.findByText("Verification details"))
    expect(await screen.findByText("ABCD1234")).toBeTruthy()
    expect(await screen.findByText("verified_and_trusted")).toBeTruthy()
    expect(await screen.findByText(/signer revoked/i)).toBeTruthy()
    expect(await screen.findByText("Upgrade History")).toBeTruthy()
    expect(screen.getByText("0.9.0 -> 1.0.0")).toBeTruthy()
  })

  it("renders local-path and git source install forms", async () => {
    render(<GovernancePacksTab />)

    expect(await screen.findByLabelText(/local path/i)).toBeTruthy()
    expect(screen.getByLabelText(/git repository url/i)).toBeTruthy()
    expect(screen.getByLabelText(/git ref/i)).toBeTruthy()
    expect(screen.getByLabelText(/git subpath/i)).toBeTruthy()
  })

  it("previews and imports a local source candidate", async () => {
    const user = userEvent.setup()
    render(<GovernancePacksTab />)

    await screen.findAllByText("Researcher Pack")
    await user.clear(screen.getByLabelText(/local path/i))
    await user.type(screen.getByLabelText(/local path/i), "/srv/packs/researcher-pack")
    await user.click(screen.getByRole("button", { name: /preview local source/i }))

    expect(await screen.findByText("Prepared candidate")).toBeTruthy()
    expect(screen.getByText("/srv/packs/researcher-pack")).toBeTruthy()

    await user.click(screen.getByRole("button", { name: /import local source/i }))

    expect(mocks.prepareGovernancePackSourceCandidate).toHaveBeenCalledTimes(1)
    expect(mocks.importGovernancePackSourceCandidate).toHaveBeenCalledWith({
      owner_scope_type: "user",
      candidate_id: 501
    })
  })

  it("checks for updates and previews a prepared source upgrade", async () => {
    const user = userEvent.setup()
    render(<GovernancePacksTab />)

    await screen.findAllByText("Researcher Pack")
    await user.click(screen.getByRole("button", { name: /check for updates/i }))

    expect(await screen.findByText(/newer version available/i)).toBeTruthy()
    expect(await screen.findByText(/signer rotated/i)).toBeTruthy()
    expect(screen.getByRole("button", { name: /preview source upgrade/i })).toBeTruthy()

    await user.click(screen.getByRole("button", { name: /preview source upgrade/i }))

    expect(await screen.findByRole("dialog")).toBeTruthy()
    expect(await screen.findByText("Prepared source candidate")).toBeTruthy()
    expect(screen.getByText("def456")).toBeTruthy()
    const verificationDetails = await screen.findAllByText("Verification details")
    await user.click(verificationDetails[verificationDetails.length - 1]!)
    expect(screen.getByText("BBBB2222")).toBeTruthy()
    expect(mocks.prepareGovernancePackUpgradeCandidate).toHaveBeenCalledWith(81)
    expect(mocks.dryRunGovernancePackSourceUpgrade).toHaveBeenCalledWith({
      source_governance_pack_id: 81,
      owner_scope_type: "user",
      candidate_id: 502
    })
  })

  it("renders no-update and source-drift statuses", async () => {
    mocks.checkGovernancePackUpdates
      .mockResolvedValueOnce({
        governance_pack_id: 81,
        status: "no_update",
        installed_manifest: {
          pack_id: "researcher-pack",
          pack_version: "1.0.0",
          title: "Researcher Pack",
          description: "Portable research governance pack"
        },
        candidate_manifest: null,
        source_commit_resolved: "abc123",
        pack_content_digest: "c".repeat(64)
      })
      .mockResolvedValueOnce({
        governance_pack_id: 81,
        status: "source_drift_same_version",
        installed_manifest: {
          pack_id: "researcher-pack",
          pack_version: "1.0.0",
          title: "Researcher Pack",
          description: "Portable research governance pack"
        },
        candidate_manifest: {
          pack_id: "researcher-pack",
          pack_version: "1.0.0",
          title: "Researcher Pack",
          description: "Portable research governance pack"
        },
        source_commit_resolved: "fedcba",
        pack_content_digest: "e".repeat(64),
        verification_warning_code: "unknown_previous_signer"
      })

    const user = userEvent.setup()
    render(<GovernancePacksTab />)

    await screen.findAllByText("Researcher Pack")
    await user.click(screen.getByRole("button", { name: /check for updates/i }))
    expect(await screen.findByText(/no update available/i)).toBeTruthy()

    await user.click(screen.getByRole("button", { name: /check for updates/i }))
    expect(await screen.findByText(/source drift detected at the same version/i)).toBeTruthy()
    expect(await screen.findByText(/previous signer unknown/i)).toBeTruthy()
  })

  it("opens the upgrade modal and renders dry-run object diffs", async () => {
    const user = userEvent.setup()
    render(<GovernancePacksTab />)

    expect((await screen.findAllByText("Researcher Pack")).length).toBeGreaterThan(0)
    fireEvent.change(screen.getByLabelText(/governance pack json/i), {
      target: {
        value: JSON.stringify({
          manifest: {
            pack_id: "researcher-pack",
            pack_version: "1.1.0",
            pack_schema_version: 1,
            capability_taxonomy_version: 1,
            adapter_contract_version: 1,
            title: "Researcher Pack"
          },
          profiles: [],
          approvals: [],
          personas: [],
          assignments: []
        })
      }
    })

    await user.click(screen.getByRole("button", { name: /preview upgrade/i }))

    expect(await screen.findByRole("dialog")).toBeTruthy()
    expect(await screen.findByText("Modified objects")).toBeTruthy()
    expect((await screen.findAllByText("permission_profile:researcher.profile")).length).toBeGreaterThan(0)
    expect(screen.getByRole("button", { name: /execute upgrade/i })).toBeEnabled()
  })

  it("blocks execute upgrade when the dry-run reports conflicts", async () => {
    mocks.dryRunGovernancePackUpgrade.mockResolvedValueOnce({
      plan: {
        source_governance_pack_id: 81,
        source_manifest: {
          pack_id: "researcher-pack",
          pack_version: "1.0.0",
          title: "Researcher Pack"
        },
        target_manifest: {
          pack_id: "researcher-pack",
          pack_version: "1.1.0",
          title: "Researcher Pack"
        },
        object_diff: [],
        dependency_impact: [],
        structural_conflicts: [],
        behavioral_conflicts: ["permission_profile:researcher.profile changed"],
        warnings: [],
        planner_inputs_fingerprint: "plan-fingerprint",
        adapter_state_fingerprint: "adapter-fingerprint",
        upgradeable: false
      }
    })
    const user = userEvent.setup()
    render(<GovernancePacksTab />)

    expect((await screen.findAllByText("Researcher Pack")).length).toBeGreaterThan(0)
    fireEvent.change(screen.getByLabelText(/governance pack json/i), {
      target: {
        value: JSON.stringify({
          manifest: {
            pack_id: "researcher-pack",
            pack_version: "1.1.0",
            pack_schema_version: 1,
            capability_taxonomy_version: 1,
            adapter_contract_version: 1,
            title: "Researcher Pack"
          },
          profiles: [],
          approvals: [],
          personas: [],
          assignments: []
        })
      }
    })

    await user.click(screen.getByRole("button", { name: /preview upgrade/i }))

    expect(await screen.findByText("Blocking conflicts")).toBeTruthy()
    expect(screen.getByText("permission_profile:researcher.profile changed")).toBeTruthy()
    expect(screen.getByRole("button", { name: /execute upgrade/i })).toBeDisabled()
  })
})

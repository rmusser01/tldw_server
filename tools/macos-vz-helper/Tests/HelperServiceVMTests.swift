import Foundation
import Testing
@testable import MacOSVZHelperDaemon

@Test func helperServiceCreateVMReturnsRunningState() throws {
    let registry = VMRegistry()
    let manager = VZLinuxVMManager(
        registry: registry,
        bootDriver: RecordingBootDriver(),
        guestBridge: ReadyGuestBridge()
    )
    let service = HelperService(
        hostFacts: HostFacts(isMacOS: true, isAppleSilicon: true),
        registry: registry,
        vmManager: manager,
        helperInstanceID: "helper-test-1",
        helperStartedAt: "2026-05-09T00:00:00Z"
    )

    let response = try service.createVM(
        vmID: "vm-service",
        templatePath: "/tmp/template.img",
        workspacePath: "/tmp/workspace",
        readinessTimeoutSeconds: 5
    )

    #expect(response.vmID == "vm-service")
    #expect(response.state == "running")
    #expect(response.details["transport"] == "vsock")
    #expect(response.details["helper_instance_id"] == "helper-test-1")
    #expect(response.details["helper_started_at"] == "2026-05-09T00:00:00Z")
}

@Test func helperServiceCreateVMClearsRegistryWhenBootDriverFails() throws {
    let registry = VMRegistry()
    let bootDriver = FailingBootDriver()
    let manager = VZLinuxVMManager(
        registry: registry,
        bootDriver: bootDriver,
        guestBridge: ReadyGuestBridge()
    )
    let service = HelperService(
        hostFacts: HostFacts(isMacOS: true, isAppleSilicon: true),
        registry: registry,
        vmManager: manager
    )

    #expect(throws: TestBootDriverError.self) {
        _ = try service.createVM(
            vmID: "vm-service-boot-failed",
            templatePath: "/tmp/template.img",
            workspacePath: "/tmp/workspace",
            readinessTimeoutSeconds: 5
        )
    }

    #expect(service.getVMStatus(vmID: "vm-service-boot-failed") == nil)
    #expect(service.listVMs().vms.isEmpty)
    #expect(bootDriver.stoppedVMIDs == ["vm-service-boot-failed"])
}

@Test func helperServiceCreateVMClearsRegistryWhenReadinessFails() throws {
    let registry = VMRegistry()
    let bootDriver = RecordingBootDriver()
    let manager = VZLinuxVMManager(
        registry: registry,
        bootDriver: bootDriver,
        guestBridge: FailingGuestBridge(error: VSockSessionError.requestTimedOut("vm-service-readiness-failed"))
    )
    let service = HelperService(
        hostFacts: HostFacts(isMacOS: true, isAppleSilicon: true),
        registry: registry,
        vmManager: manager
    )

    #expect(throws: VSockSessionError.self) {
        _ = try service.createVM(
            vmID: "vm-service-readiness-failed",
            templatePath: "/tmp/template.img",
            workspacePath: "/tmp/workspace",
            readinessTimeoutSeconds: 5
        )
    }

    #expect(service.getVMStatus(vmID: "vm-service-readiness-failed") == nil)
    #expect(service.listVMs().vms.isEmpty)
    #expect(bootDriver.stoppedVMIDs == ["vm-service-readiness-failed"])
}

@Test func helperServiceSurfacesResourceSnapshotInCreateStatusAndList() throws {
    let registry = VMRegistry()
    let manager = VZLinuxVMManager(
        registry: registry,
        bootDriver: RecordingBootDriver(),
        guestBridge: ReadyGuestBridge()
    )
    let service = HelperService(
        hostFacts: HostFacts(isMacOS: true, isAppleSilicon: true),
        registry: registry,
        vmManager: manager,
        helperInstanceID: "helper-test-1",
        helperStartedAt: "2026-05-09T00:00:00Z"
    )

    let response = try service.createVM(
        vmID: "vm-resource-details",
        templatePath: "/tmp/template.img",
        workspacePath: "/tmp/workspace",
        readinessTimeoutSeconds: 5,
        metadata: VMOwnershipMetadata(
            owner: "tldw",
            runtime: "vz_linux",
            runID: "run-resource-details",
            sessionID: "",
            sessionMode: false,
            templateID: "",
            templatePath: "",
            runManifestPath: "",
            planningSource: "",
            workspacePath: "",
            createdAt: "2026-05-09T00:00:00Z",
            networkPolicy: "deny_all"
        )
    )
    let status = service.getVMStatus(vmID: "vm-resource-details")
    let listed = service.listVMs().vms.first

    for details in [response.details, status?.details, listed?.details] {
        #expect(details?["cpu_count"] == "2")
        #expect(details?["memory_size_mb"] == "1024")
        let wallTime = Int(details?["wall_time_sec"] ?? "")
        #expect(wallTime != nil)
        #expect((wallTime ?? -1) >= 0)
    }
}

@Test func helperServiceSurfacesGuestAgentDetailsInCreateStatusAndList() throws {
    let registry = VMRegistry()
    let guestInfo = GuestAgentInfo(
        guestVersion: "1.0.0",
        workspaceRoot: "/workspace",
        capabilities: ["exec", "output_cap_v1"],
        capabilitiesKnown: true
    )
    let manager = VZLinuxVMManager(
        registry: registry,
        bootDriver: RecordingBootDriver(),
        guestBridge: ReadyGuestBridge(info: guestInfo)
    )
    let service = HelperService(
        hostFacts: HostFacts(isMacOS: true, isAppleSilicon: true),
        registry: registry,
        vmManager: manager,
        helperInstanceID: "helper-test-1",
        helperStartedAt: "2026-05-09T00:00:00Z"
    )

    let response = try service.createVM(
        vmID: "vm-guest-details",
        templatePath: "/tmp/template.img",
        workspacePath: "/tmp/workspace",
        readinessTimeoutSeconds: 5
    )
    let status = service.getVMStatus(vmID: "vm-guest-details")
    let listed = service.listVMs().vms.first

    for details in [response.details, status?.details, listed?.details] {
        #expect(details?["helper_instance_id"] == "helper-test-1")
        #expect(details?["helper_started_at"] == "2026-05-09T00:00:00Z")
        #expect(details?["guest_version"] == "1.0.0")
        #expect(details?["guest_workspace_root"] == "/workspace")
        #expect(details?["guest_capabilities_known"] == "true")
        #expect(details?["guest_capabilities"] == "exec,output_cap_v1")
    }
}

@Test func helperServiceSurfacesUnknownCapabilitiesForOlderGuests() throws {
    let registry = VMRegistry()
    let guestInfo = GuestAgentInfo(
        guestVersion: "0.9.0",
        workspaceRoot: "/workspace",
        capabilities: [],
        capabilitiesKnown: false
    )
    let manager = VZLinuxVMManager(
        registry: registry,
        bootDriver: RecordingBootDriver(),
        guestBridge: ReadyGuestBridge(info: guestInfo)
    )
    let service = HelperService(
        hostFacts: HostFacts(isMacOS: true, isAppleSilicon: true),
        registry: registry,
        vmManager: manager
    )

    let response = try service.createVM(
        vmID: "vm-old-guest-details",
        templatePath: "/tmp/template.img",
        workspacePath: "/tmp/workspace",
        readinessTimeoutSeconds: 5
    )
    let status = service.getVMStatus(vmID: "vm-old-guest-details")
    let listed = service.listVMs().vms.first

    for details in [response.details, status?.details, listed?.details] {
        #expect(details?["guest_version"] == "0.9.0")
        #expect(details?["guest_workspace_root"] == "/workspace")
        #expect(details?["guest_capabilities_known"] == "false")
        #expect(details?["guest_capabilities"] == nil)
    }
}

@Test func helperServiceClearsStaleGuestDetailsWhenVMIDIsReusedWithoutInfo() throws {
    let registry = VMRegistry()
    registry.upsert(
        vmID: "vm-reused",
        state: "running",
        healthy: true,
        guestInfo: GuestAgentInfo(
            guestVersion: "stale",
            workspaceRoot: "/stale",
            capabilities: ["stale_capability"],
            capabilitiesKnown: true
        )
    )
    let manager = VZLinuxVMManager(
        registry: registry,
        bootDriver: RecordingBootDriver(),
        guestBridge: ReadyGuestBridge()
    )
    let service = HelperService(
        hostFacts: HostFacts(isMacOS: true, isAppleSilicon: true),
        registry: registry,
        vmManager: manager
    )

    let response = try service.createVM(
        vmID: "vm-reused",
        templatePath: "/tmp/template.img",
        workspacePath: "/tmp/workspace",
        readinessTimeoutSeconds: 5
    )

    #expect(response.details["guest_version"] == nil)
    #expect(response.details["guest_workspace_root"] == nil)
    #expect(response.details["guest_capabilities_known"] == nil)
    #expect(response.details["guest_capabilities"] == nil)
}

@Test func helperServiceCreateVMReturnsOwnershipMetadata() throws {
    let registry = VMRegistry()
    let manager = VZLinuxVMManager(
        registry: registry,
        bootDriver: RecordingBootDriver(),
        guestBridge: ReadyGuestBridge()
    )
    let service = HelperService(
        hostFacts: HostFacts(isMacOS: true, isAppleSilicon: true),
        registry: registry,
        vmManager: manager
    )
    let metadata = VMOwnershipMetadata(
        owner: "tldw",
        runtime: " VZ_LINUX ",
        runID: "run-owned",
        sessionID: "session-owned",
        sessionMode: true,
        templateID: "vz_linux:template-owned",
        templatePath: "/tmp/template.img",
        runManifestPath: "/tmp/image-store/runs/run-owned/manifest.json",
        planningSource: "image_store",
        workspacePath: "/tmp/workspace",
        createdAt: "",
        networkPolicy: "allowlist"
    )

    let response = try service.createVM(
        vmID: "vm-owned",
        templatePath: "/tmp/template.img",
        workspacePath: "/tmp/workspace",
        readinessTimeoutSeconds: 5,
        metadata: metadata
    )
    let status = service.getVMStatus(vmID: "vm-owned")
    let listed = service.listVMs().vms.first

    #expect(response.metadata.owner == "tldw")
    #expect(response.metadata.runtime == "vz_linux")
    #expect(response.metadata.runID == "run-owned")
    #expect(response.metadata.sessionID == "session-owned")
    #expect(response.metadata.sessionMode == true)
    #expect(response.metadata.templateID == "vz_linux:template-owned")
    #expect(response.metadata.templatePath == "/tmp/template.img")
    #expect(response.metadata.runManifestPath == "/tmp/image-store/runs/run-owned/manifest.json")
    #expect(response.metadata.planningSource == "image_store")
    #expect(response.metadata.workspacePath == "/tmp/workspace")
    #expect(response.metadata.networkPolicy == "deny_all")
    #expect(response.metadata.createdAt.isEmpty == false)
    #expect(status?.metadata.runID == "run-owned")
    #expect(status?.details["network_policy"] == "deny_all")
    #expect(listed?.metadata.runID == "run-owned")
    #expect(listed?.details["network_policy"] == "deny_all")
}

@Test func helperServiceCreateVMRejectsUnsupportedNetworkPolicy() throws {
    let registry = VMRegistry()
    let manager = VZLinuxVMManager(
        registry: registry,
        bootDriver: RecordingBootDriver(),
        guestBridge: ReadyGuestBridge()
    )
    let service = HelperService(
        hostFacts: HostFacts(isMacOS: true, isAppleSilicon: true),
        registry: registry,
        vmManager: manager
    )

    do {
        _ = try service.createVM(
            vmID: "vm-allowlist",
            templatePath: "/tmp/template.img",
            workspacePath: "/tmp/workspace",
            readinessTimeoutSeconds: 5,
            networkPolicy: "allowlist"
        )
        Issue.record("expected unsupported network policy")
    } catch HelperServiceError.unsupportedNetworkPolicy(let policy) {
        #expect(policy == "allowlist")
    } catch {
        Issue.record("expected unsupportedNetworkPolicy, got \(error)")
    }
    #expect(registry.status(vmID: "vm-allowlist") == nil)
}

@Test func helperServiceCreateVMRejectsInvalidContractBeforeRegistryMutation() throws {
    let registry = VMRegistry()
    let manager = VZLinuxVMManager(
        registry: registry,
        bootDriver: RecordingBootDriver(),
        guestBridge: ReadyGuestBridge()
    )
    let service = HelperService(
        hostFacts: HostFacts(isMacOS: true, isAppleSilicon: true),
        registry: registry,
        vmManager: manager
    )

    #expect(throws: HelperServiceError.self) {
        _ = try service.createVM(
            vmID: "bad/name",
            templatePath: "/tmp/template.img",
            workspacePath: "/tmp/workspace",
            readinessTimeoutSeconds: 5
        )
    }
    #expect(registry.status(vmID: "bad/name") == nil)

    #expect(throws: HelperServiceError.self) {
        _ = try service.createVM(
            vmID: "vm-relative-workspace",
            templatePath: "/tmp/template.img",
            workspacePath: "workspace",
            readinessTimeoutSeconds: 5
        )
    }
    #expect(registry.status(vmID: "vm-relative-workspace") == nil)

    #expect(throws: HelperServiceError.self) {
        _ = try service.createVM(
            vmID: "vm-timeout",
            templatePath: "/tmp/template.img",
            workspacePath: "/tmp/workspace",
            readinessTimeoutSeconds: 0
        )
    }
    #expect(registry.status(vmID: "vm-timeout") == nil)
}

@Test func helperServiceCreateVMRejectsExistingSymlinkWorkspaceBeforeRegistryMutation() throws {
    let root = URL(fileURLWithPath: NSTemporaryDirectory())
        .appendingPathComponent("macos-vz-create-vm-\(UUID().uuidString)")
    let target = root.appendingPathComponent("target", isDirectory: true)
    let link = root.appendingPathComponent("workspace-link", isDirectory: true)
    try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
    try FileManager.default.createSymbolicLink(atPath: link.path, withDestinationPath: target.path)
    defer { try? FileManager.default.removeItem(at: root) }

    let registry = VMRegistry()
    let manager = VZLinuxVMManager(
        registry: registry,
        bootDriver: RecordingBootDriver(),
        guestBridge: ReadyGuestBridge()
    )
    let service = HelperService(
        hostFacts: HostFacts(isMacOS: true, isAppleSilicon: true),
        registry: registry,
        vmManager: manager
    )

    #expect(throws: HelperServiceError.self) {
        _ = try service.createVM(
            vmID: "vm-symlink-workspace",
            templatePath: "/tmp/template.img",
            workspacePath: link.path,
            readinessTimeoutSeconds: 5
        )
    }
    #expect(registry.status(vmID: "vm-symlink-workspace") == nil)
}

@Test func helperServiceCreateVMRejectsSymlinkParentWorkspaceBeforeRegistryMutation() throws {
    let root = URL(fileURLWithPath: NSTemporaryDirectory())
        .appendingPathComponent("macos-vz-create-vm-\(UUID().uuidString)")
    let target = root.appendingPathComponent("target", isDirectory: true)
    let link = root.appendingPathComponent("workspace-link", isDirectory: true)
    try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
    try FileManager.default.createSymbolicLink(atPath: link.path, withDestinationPath: target.path)
    defer { try? FileManager.default.removeItem(at: root) }

    let registry = VMRegistry()
    let manager = VZLinuxVMManager(
        registry: registry,
        bootDriver: RecordingBootDriver(),
        guestBridge: ReadyGuestBridge()
    )
    let service = HelperService(
        hostFacts: HostFacts(isMacOS: true, isAppleSilicon: true),
        registry: registry,
        vmManager: manager
    )

    #expect(throws: HelperServiceError.self) {
        _ = try service.createVM(
            vmID: "vm-symlink-parent-workspace",
            templatePath: "/tmp/template.img",
            workspacePath: link.appendingPathComponent("nested-workspace").path,
            readinessTimeoutSeconds: 5
        )
    }
    #expect(registry.status(vmID: "vm-symlink-parent-workspace") == nil)
}

@Test func helperServiceCreateVMDefaultsMissingOwnershipMetadataToUnknown() throws {
    let registry = VMRegistry()
    let manager = VZLinuxVMManager(
        registry: registry,
        bootDriver: RecordingBootDriver(),
        guestBridge: ReadyGuestBridge()
    )
    let service = HelperService(
        hostFacts: HostFacts(isMacOS: true, isAppleSilicon: true),
        registry: registry,
        vmManager: manager
    )

    let response = try service.createVM(
        vmID: "vm-unknown",
        templatePath: "/tmp/template.img",
        workspacePath: "/tmp/workspace",
        readinessTimeoutSeconds: 5
    )

    #expect(response.metadata.owner == "unknown")
    #expect(response.metadata.runtime == "vz_linux")
    #expect(response.metadata.runID == "")
    #expect(response.metadata.templateID == "")
    #expect(response.metadata.createdAt.isEmpty == false)
}

@Test func helperServiceListVMsReturnsKnownVMs() throws {
    let registry = VMRegistry()
    let manager = VZLinuxVMManager(
        registry: registry,
        bootDriver: RecordingBootDriver(),
        guestBridge: ReadyGuestBridge()
    )
    let service = HelperService(
        hostFacts: HostFacts(isMacOS: true, isAppleSilicon: true),
        registry: registry,
        vmManager: manager
    )

    _ = try service.createVM(
        vmID: "vm-list",
        templatePath: "/tmp/template.img",
        workspacePath: "/tmp/workspace",
        readinessTimeoutSeconds: 5
    )

    let response = service.listVMs()

    #expect(response.vms.count == 1)
    #expect(response.vms.first?.vmID == "vm-list")
    #expect(response.vms.first?.state == "running")
    #expect(response.vms.first?.healthy == true)
}

@Test func helperServiceTerminateVMReturnsTrueAndClearsRegistry() throws {
    let registry = VMRegistry()
    let manager = VZLinuxVMManager(
        registry: registry,
        bootDriver: RecordingBootDriver(),
        guestBridge: ReadyGuestBridge()
    )
    let service = HelperService(
        hostFacts: HostFacts(isMacOS: true, isAppleSilicon: true),
        registry: registry,
        vmManager: manager
    )

    _ = try service.createVM(
        vmID: "vm-stop",
        templatePath: "/tmp/template.img",
        workspacePath: "/tmp/workspace",
        readinessTimeoutSeconds: 5
    )

    let terminated = try service.terminateVM(vmID: "vm-stop")

    #expect(terminated == true)
    #expect(registry.status(vmID: "vm-stop") == nil)
}

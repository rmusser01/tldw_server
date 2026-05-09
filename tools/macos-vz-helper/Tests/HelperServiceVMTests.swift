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
        runtime: "vz_linux",
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

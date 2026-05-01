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
        vmManager: manager
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

    #expect(throws: HelperServiceError.self) {
        try service.createVM(
            vmID: "vm-allowlist",
            templatePath: "/tmp/template.img",
            workspacePath: "/tmp/workspace",
            readinessTimeoutSeconds: 5,
            networkPolicy: "allowlist"
        )
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

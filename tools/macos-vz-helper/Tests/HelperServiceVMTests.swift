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
        templatePath: "/tmp/template.img",
        workspacePath: "/tmp/workspace",
        createdAt: ""
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
    #expect(response.metadata.templatePath == "/tmp/template.img")
    #expect(response.metadata.workspacePath == "/tmp/workspace")
    #expect(response.metadata.createdAt.isEmpty == false)
    #expect(status?.metadata.runID == "run-owned")
    #expect(listed?.metadata.runID == "run-owned")
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

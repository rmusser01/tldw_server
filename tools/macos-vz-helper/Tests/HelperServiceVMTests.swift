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

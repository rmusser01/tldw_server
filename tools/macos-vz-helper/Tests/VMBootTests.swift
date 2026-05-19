import Foundation
import Testing
@testable import MacOSVZHelperDaemon

@Test func createVMRegistersBootingState() throws {
    let registry = VMRegistry()
    let bootDriver = RecordingBootDriver { vmID in
        let status = registry.status(vmID: vmID)
        #expect(status?.state == "booting")
        #expect(status?.healthy == false)
    }
    let manager = VZLinuxVMManager(
        registry: registry,
        bootDriver: bootDriver,
        guestBridge: ReadyGuestBridge()
    )

    _ = try manager.createVM(
        vmID: "vm-booting",
        templatePath: "/tmp/template.img",
        workspacePath: "/tmp/workspace",
        readinessTimeoutSeconds: 5
    )
}

@Test func guestReadinessTransitionsVMToRunning() throws {
    let registry = VMRegistry()
    let manager = VZLinuxVMManager(
        registry: registry,
        bootDriver: RecordingBootDriver(),
        guestBridge: ReadyGuestBridge()
    )

    let result = try manager.createVM(
        vmID: "vm-ready",
        templatePath: "/tmp/template.img",
        workspacePath: "/tmp/workspace",
        readinessTimeoutSeconds: 5
    )

    #expect(result.vmID == "vm-ready")
    #expect(result.state == "running")
    #expect(result.healthy == true)
    let status = registry.status(vmID: "vm-ready")
    #expect(status?.state == "running")
    #expect(status?.healthy == true)
}

@Test func createVMPreservesBootResourceSnapshotWhenRunning() throws {
    let registry = VMRegistry()
    let resourceSnapshot = VMResourceSnapshot(cpuCount: 4, memorySizeBytes: 2_147_483_648)
    let manager = VZLinuxVMManager(
        registry: registry,
        bootDriver: RecordingBootDriver(resourceSnapshot: resourceSnapshot),
        guestBridge: ReadyGuestBridge()
    )

    let result = try manager.createVM(
        vmID: "vm-resource-snapshot",
        templatePath: "/tmp/template.img",
        workspacePath: "/tmp/workspace",
        readinessTimeoutSeconds: 5
    )
    let status = registry.status(vmID: "vm-resource-snapshot")

    #expect(result.resourceSnapshot == resourceSnapshot)
    #expect(status?.resourceSnapshot == resourceSnapshot)
}

@Test func createVMClearsStaleResourceSnapshotWhileBootingReusedVMID() throws {
    let registry = VMRegistry()
    registry.upsert(
        vmID: "vm-reused-resource-snapshot",
        state: "running",
        healthy: true,
        resourceSnapshot: VMResourceSnapshot(cpuCount: 8, memorySizeBytes: 4_294_967_296)
    )
    let bootDriver = RecordingBootDriver { vmID in
        let status = registry.status(vmID: vmID)
        #expect(status?.state == "booting")
        #expect(status?.resourceSnapshot == nil)
    }
    let manager = VZLinuxVMManager(
        registry: registry,
        bootDriver: bootDriver,
        guestBridge: ReadyGuestBridge()
    )

    _ = try manager.createVM(
        vmID: "vm-reused-resource-snapshot",
        templatePath: "/tmp/template.img",
        workspacePath: "/tmp/workspace",
        readinessTimeoutSeconds: 5
    )
}

@Test func createVMPassesReadinessTimeoutToBootDriver() throws {
    let bootDriver = RecordingBootDriver()
    let manager = VZLinuxVMManager(
        registry: VMRegistry(),
        bootDriver: bootDriver,
        guestBridge: ReadyGuestBridge()
    )

    _ = try manager.createVM(
        vmID: "vm-timeout",
        templatePath: "/tmp/template.img",
        workspacePath: "/tmp/workspace",
        readinessTimeoutSeconds: 12
    )

    #expect(bootDriver.lastReadinessTimeoutSeconds == 12)
}

@Test func createVMRemovesBootingRecordWhenBootDriverFails() throws {
    let registry = VMRegistry()
    let bootDriver = FailingBootDriver()
    let manager = VZLinuxVMManager(
        registry: registry,
        bootDriver: bootDriver,
        guestBridge: ReadyGuestBridge()
    )

    #expect(throws: TestBootDriverError.self) {
        _ = try manager.createVM(
            vmID: "vm-boot-failed",
            templatePath: "/tmp/template.img",
            workspacePath: "/tmp/workspace",
            readinessTimeoutSeconds: 7
        )
    }

    #expect(registry.status(vmID: "vm-boot-failed") == nil)
    #expect(bootDriver.stoppedVMIDs == ["vm-boot-failed"])
    #expect(bootDriver.lastReadinessTimeoutSeconds == 7)
}

@Test func createVMStopsBootedMachineWhenReadinessFails() throws {
    let registry = VMRegistry()
    let bootDriver = RecordingBootDriver()
    let manager = VZLinuxVMManager(
        registry: registry,
        bootDriver: bootDriver,
        guestBridge: FailingGuestBridge(error: VSockSessionError.requestTimedOut("vm-readiness-failed"))
    )

    #expect(throws: VSockSessionError.self) {
        _ = try manager.createVM(
            vmID: "vm-readiness-failed",
            templatePath: "/tmp/template.img",
            workspacePath: "/tmp/workspace",
            readinessTimeoutSeconds: 5
        )
    }

    #expect(registry.status(vmID: "vm-readiness-failed") == nil)
    #expect(bootDriver.stoppedVMIDs == ["vm-readiness-failed"])
}

@Test func terminateVMRemovesRegistryState() throws {
    let registry = VMRegistry()
    let bootDriver = RecordingBootDriver()
    let manager = VZLinuxVMManager(
        registry: registry,
        bootDriver: bootDriver,
        guestBridge: ReadyGuestBridge()
    )

    _ = try manager.createVM(
        vmID: "vm-terminate",
        templatePath: "/tmp/template.img",
        workspacePath: "/tmp/workspace",
        readinessTimeoutSeconds: 5
    )

    let terminated = try manager.terminateVM(vmID: "vm-terminate")

    #expect(terminated == true)
    #expect(registry.status(vmID: "vm-terminate") == nil)
    #expect(bootDriver.stoppedVMIDs == ["vm-terminate"])
}

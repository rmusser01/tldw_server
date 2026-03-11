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

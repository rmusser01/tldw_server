import Foundation
import Testing
@testable import MacOSVZHelperDaemon

final class RecordingGuestBridge: GuestBridging {
    private(set) var lastExec: (vmID: String, argv: [String], cwd: String, env: [String: String], timeout: TimeInterval)?

    func waitUntilReady(vmID: String, timeoutSeconds: TimeInterval) throws {}

    func exec(
        vmID: String,
        argv: [String],
        cwd: String,
        env: [String: String],
        timeoutSeconds: TimeInterval
    ) throws -> GuestExecResult {
        lastExec = (vmID, argv, cwd, env, timeoutSeconds)
        return GuestExecResult(exitCode: 0, stdout: "ok\n", stderr: "")
    }
}

@Test func helperServiceExecGuestBridgesThroughGuestAgent() throws {
    let registry = VMRegistry()
    let guestBridge = RecordingGuestBridge()
    let manager = VZLinuxVMManager(
        registry: registry,
        bootDriver: RecordingBootDriver(),
        guestBridge: guestBridge
    )
    let service = HelperService(
        hostFacts: HostFacts(isMacOS: true, isAppleSilicon: true),
        registry: registry,
        vmManager: manager
    )

    _ = try service.createVM(
        vmID: "vm-exec",
        templatePath: "/tmp/template.img",
        workspacePath: "/workspace",
        readinessTimeoutSeconds: 5
    )

    let response = try service.execGuest(
        vmID: "vm-exec",
        argv: ["/bin/echo", "ok"],
        cwd: "/workspace",
        env: ["FOO": "1"],
        timeoutSeconds: 15
    )

    #expect(response.exitCode == 0)
    #expect(response.stdout == "ok\n")
    #expect(response.stderr == "")
    #expect(guestBridge.lastExec?.vmID == "vm-exec")
    #expect(guestBridge.lastExec?.argv == ["/bin/echo", "ok"])
    #expect(guestBridge.lastExec?.cwd == "/workspace")
    #expect(guestBridge.lastExec?.env == ["FOO": "1"])
    #expect(guestBridge.lastExec?.timeout == 15)
}

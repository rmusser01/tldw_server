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

@Test func helperServiceExecGuestRejectsInvalidCommandContract() throws {
    let registry = VMRegistry()
    let manager = VZLinuxVMManager(
        registry: registry,
        bootDriver: RecordingBootDriver(),
        guestBridge: RecordingGuestBridge()
    )
    let service = HelperService(
        hostFacts: HostFacts(isMacOS: true, isAppleSilicon: true),
        registry: registry,
        vmManager: manager
    )

    _ = try service.createVM(
        vmID: "vm-exec-contract",
        templatePath: "/tmp/template.img",
        workspacePath: "/workspace",
        readinessTimeoutSeconds: 5
    )

    do {
        _ = try service.execGuest(
            vmID: "vm-exec-contract",
            argv: [],
            cwd: "/workspace",
            env: [:],
            timeoutSeconds: 15
        )
        Issue.record("expected empty argv to be rejected")
    } catch HelperServiceError.invalidExecArgv(let reason) {
        #expect(reason == "argv_required")
    } catch {
        Issue.record("expected invalidExecArgv, got \(error)")
    }

    do {
        _ = try service.execGuest(
            vmID: "vm-exec-contract",
            argv: ["/bin/echo"],
            cwd: "/tmp",
            env: [:],
            timeoutSeconds: 15
        )
        Issue.record("expected cwd outside workspace to be rejected")
    } catch HelperServiceError.invalidExecCwd(let reason) {
        #expect(reason == "cwd_outside_workspace")
    } catch {
        Issue.record("expected invalidExecCwd, got \(error)")
    }

    do {
        _ = try service.execGuest(
            vmID: "vm-exec-contract",
            argv: ["/bin/echo"],
            cwd: "/workspace",
            env: ["BAD=KEY": "1"],
            timeoutSeconds: 15
        )
        Issue.record("expected invalid env key to be rejected")
    } catch HelperServiceError.invalidExecEnv(let reason) {
        #expect(reason == "env_key_invalid")
    } catch {
        Issue.record("expected invalidExecEnv, got \(error)")
    }

    do {
        _ = try service.execGuest(
            vmID: "vm-exec-contract",
            argv: ["/bin/echo"],
            cwd: "/workspace",
            env: [:],
            timeoutSeconds: 0
        )
        Issue.record("expected non-positive timeout to be rejected")
    } catch HelperServiceError.invalidExecTimeout(let reason) {
        #expect(reason == "timeout_out_of_range")
    } catch {
        Issue.record("expected invalidExecTimeout, got \(error)")
    }
}

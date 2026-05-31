import Foundation
import Testing
@testable import MacOSVZHelperDaemon

final class RecordingGuestBridge: GuestBridging {
    private(set) var lastExec: (vmID: String, argv: [String], cwd: String, env: [String: String], timeout: TimeInterval, maxOutputBytes: Int?)?

    func waitUntilReady(vmID: String, timeoutSeconds: TimeInterval) throws {}

    func guestInfo(vmID: String) -> GuestAgentInfo? {
        nil
    }

    func exec(
        vmID: String,
        argv: [String],
        cwd: String,
        env: [String: String],
        timeoutSeconds: TimeInterval,
        maxOutputBytes: Int?
    ) throws -> GuestExecResult {
        lastExec = (vmID, argv, cwd, env, timeoutSeconds, maxOutputBytes)
        return GuestExecResult(exitCode: 0, stdout: "ok\n", stderr: "", details: [:])
    }
}

final class StaticOutputGuestBridge: GuestBridging {
    let stdout: String
    let stderr: String
    let details: [String: String]

    init(stdout: String, stderr: String, details: [String: String] = [:]) {
        self.stdout = stdout
        self.stderr = stderr
        self.details = details
    }

    func waitUntilReady(vmID: String, timeoutSeconds: TimeInterval) throws {}

    func guestInfo(vmID: String) -> GuestAgentInfo? {
        nil
    }

    func exec(
        vmID: String,
        argv: [String],
        cwd: String,
        env: [String: String],
        timeoutSeconds: TimeInterval,
        maxOutputBytes: Int?
    ) throws -> GuestExecResult {
        return GuestExecResult(exitCode: 0, stdout: stdout, stderr: stderr, details: details)
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
    #expect(guestBridge.lastExec?.maxOutputBytes == nil)
}

@Test func helperServiceExecGuestPassesOutputCapToGuestBridge() throws {
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
        vmID: "vm-exec-cap-forward",
        templatePath: "/tmp/template.img",
        workspacePath: "/workspace",
        readinessTimeoutSeconds: 5
    )

    _ = try service.execGuest(
        vmID: "vm-exec-cap-forward",
        argv: ["/bin/echo", "ok"],
        cwd: "/workspace",
        env: [:],
        timeoutSeconds: 15,
        maxOutputBytes: 10
    )

    #expect(guestBridge.lastExec?.maxOutputBytes == 10)
}

@Test func helperServiceExecGuestCapsReturnedOutputAndRecordsDetails() throws {
    let registry = VMRegistry()
    let manager = VZLinuxVMManager(
        registry: registry,
        bootDriver: RecordingBootDriver(),
        guestBridge: StaticOutputGuestBridge(stdout: String(repeating: "o", count: 100), stderr: String(repeating: "e", count: 100))
    )
    let service = HelperService(
        hostFacts: HostFacts(isMacOS: true, isAppleSilicon: true),
        registry: registry,
        vmManager: manager
    )

    _ = try service.createVM(
        vmID: "vm-exec-output-cap",
        templatePath: "/tmp/template.img",
        workspacePath: "/workspace",
        readinessTimeoutSeconds: 5
    )

    let response = try service.execGuest(
        vmID: "vm-exec-output-cap",
        argv: ["/bin/echo", "ok"],
        cwd: "/workspace",
        env: [:],
        timeoutSeconds: 15,
        maxOutputBytes: 10
    )

    #expect(Data(response.stdout.utf8).count + Data(response.stderr.utf8).count <= 10)
    #expect(response.stdout.isEmpty == false)
    #expect(response.stderr.isEmpty == false)
    #expect(response.details["output_limit_bytes"] == "10")
    #expect(response.details["stdout_bytes_original"] == "100")
    #expect(response.details["stderr_bytes_original"] == "100")
    #expect(response.details["stdout_truncated"] == "true")
    #expect(response.details["stderr_truncated"] == "true")
}

@Test func helperServiceExecGuestMergesGuestDetailsWithoutOverwritingHostCounters() throws {
    let registry = VMRegistry()
    let manager = VZLinuxVMManager(
        registry: registry,
        bootDriver: RecordingBootDriver(),
        guestBridge: StaticOutputGuestBridge(
            stdout: String(repeating: "o", count: 100),
            stderr: "",
            details: [
                "guest_output_limit_exceeded": "true",
                "guest_stdout_bytes_observed": "101",
                "stdout_bytes_original": "wrong",
            ]
        )
    )
    let service = HelperService(
        hostFacts: HostFacts(isMacOS: true, isAppleSilicon: true),
        registry: registry,
        vmManager: manager
    )

    _ = try service.createVM(
        vmID: "vm-exec-guest-details",
        templatePath: "/tmp/template.img",
        workspacePath: "/workspace",
        readinessTimeoutSeconds: 5
    )

    let response = try service.execGuest(
        vmID: "vm-exec-guest-details",
        argv: ["/bin/echo", "ok"],
        cwd: "/workspace",
        env: [:],
        timeoutSeconds: 15,
        maxOutputBytes: 10
    )

    #expect(response.details["guest_output_limit_exceeded"] == "true")
    #expect(response.details["guest_stdout_bytes_observed"] == "101")
    #expect(response.details["stdout_bytes_original"] == "100")
    #expect(response.details["stdout_bytes_returned"] == "10")
    #expect(response.details["stdout_truncated"] == "true")
}

@Test func helperServiceExecGuestCapsOutputAtUtf8Boundary() throws {
    let registry = VMRegistry()
    let manager = VZLinuxVMManager(
        registry: registry,
        bootDriver: RecordingBootDriver(),
        guestBridge: StaticOutputGuestBridge(stdout: "ééé", stderr: "")
    )
    let service = HelperService(
        hostFacts: HostFacts(isMacOS: true, isAppleSilicon: true),
        registry: registry,
        vmManager: manager
    )

    _ = try service.createVM(
        vmID: "vm-exec-utf8-cap",
        templatePath: "/tmp/template.img",
        workspacePath: "/workspace",
        readinessTimeoutSeconds: 5
    )

    let response = try service.execGuest(
        vmID: "vm-exec-utf8-cap",
        argv: ["/bin/echo", "ok"],
        cwd: "/workspace",
        env: [:],
        timeoutSeconds: 15,
        maxOutputBytes: 5
    )

    #expect(Data(response.stdout.utf8).count <= 5)
    #expect(response.stdout == "éé")
    #expect(response.details["stdout_bytes_original"] == "6")
    #expect(response.details["stdout_bytes_returned"] == "4")
    #expect(response.details["stdout_truncated"] == "true")
}

@Test func helperServiceExecGuestRebalancesUtf8TrimmedBudget() throws {
    let registry = VMRegistry()
    let manager = VZLinuxVMManager(
        registry: registry,
        bootDriver: RecordingBootDriver(),
        guestBridge: StaticOutputGuestBridge(stdout: "a", stderr: "é")
    )
    let service = HelperService(
        hostFacts: HostFacts(isMacOS: true, isAppleSilicon: true),
        registry: registry,
        vmManager: manager
    )

    _ = try service.createVM(
        vmID: "vm-exec-utf8-rebalance",
        templatePath: "/tmp/template.img",
        workspacePath: "/workspace",
        readinessTimeoutSeconds: 5
    )

    let response = try service.execGuest(
        vmID: "vm-exec-utf8-rebalance",
        argv: ["/bin/echo", "ok"],
        cwd: "/workspace",
        env: [:],
        timeoutSeconds: 15,
        maxOutputBytes: 2
    )

    #expect(Data(response.stdout.utf8).count + Data(response.stderr.utf8).count == 2)
    #expect(response.stderr == "é")
    #expect(response.details["stderr_bytes_returned"] == "2")
}

@Test func helperServiceExecGuestRejectsInvalidOutputLimit() throws {
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
        vmID: "vm-exec-invalid-output-cap",
        templatePath: "/tmp/template.img",
        workspacePath: "/workspace",
        readinessTimeoutSeconds: 5
    )

    do {
        _ = try service.execGuest(
            vmID: "vm-exec-invalid-output-cap",
            argv: ["/bin/echo"],
            cwd: "/workspace",
            env: [:],
            timeoutSeconds: 15,
            maxOutputBytes: 0
        )
        Issue.record("expected invalid output cap to be rejected")
    } catch HelperServiceError.invalidExecOutputLimit(let reason) {
        #expect(reason == "output_limit_out_of_range")
    } catch {
        Issue.record("expected invalidExecOutputLimit, got \(error)")
    }
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

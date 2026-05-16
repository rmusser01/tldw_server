import Foundation
@testable import MacOSVZHelperDaemon

final class RecordingBootDriver: VZBootDriving {
    private let onBoot: (String) -> Void
    private let resourceSnapshot: VMResourceSnapshot
    private(set) var lastReadinessTimeoutSeconds: TimeInterval?
    private(set) var stoppedVMIDs: [String] = []

    init(
        resourceSnapshot: VMResourceSnapshot = VMResourceSnapshot(cpuCount: 2, memorySizeBytes: 1_073_741_824),
        onBoot: @escaping (String) -> Void = { _ in }
    ) {
        self.resourceSnapshot = resourceSnapshot
        self.onBoot = onBoot
    }

    @discardableResult
    func boot(vmID: String, templatePath: String, workspacePath: String, startupTimeoutSeconds: TimeInterval) throws -> VMResourceSnapshot {
        lastReadinessTimeoutSeconds = startupTimeoutSeconds
        onBoot(vmID)
        return resourceSnapshot
    }

    func stop(vmID: String) throws {
        stoppedVMIDs.append(vmID)
    }
}

final class ReadyGuestBridge: GuestBridging {
    private let info: GuestAgentInfo?

    init(info: GuestAgentInfo? = nil) {
        self.info = info
    }

    func waitUntilReady(vmID: String, timeoutSeconds: TimeInterval) throws {}

    func guestInfo(vmID: String) -> GuestAgentInfo? {
        info
    }

    func exec(
        vmID: String,
        argv: [String],
        cwd: String,
        env: [String : String],
        timeoutSeconds: TimeInterval,
        maxOutputBytes: Int?
    ) throws -> GuestExecResult {
        GuestExecResult(exitCode: 0, stdout: "", stderr: "", details: [:])
    }
}

final class FailingGuestBridge: GuestBridging {
    private let error: Error

    init(error: Error) {
        self.error = error
    }

    func waitUntilReady(vmID: String, timeoutSeconds: TimeInterval) throws {
        throw error
    }

    func guestInfo(vmID: String) -> GuestAgentInfo? {
        nil
    }

    func exec(
        vmID: String,
        argv: [String],
        cwd: String,
        env: [String : String],
        timeoutSeconds: TimeInterval,
        maxOutputBytes: Int?
    ) throws -> GuestExecResult {
        throw error
    }
}

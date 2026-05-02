import Foundation
@testable import MacOSVZHelperDaemon

final class RecordingBootDriver: VZBootDriving {
    private let onBoot: (String) -> Void
    private(set) var lastReadinessTimeoutSeconds: TimeInterval?
    private(set) var stoppedVMIDs: [String] = []

    init(onBoot: @escaping (String) -> Void = { _ in }) {
        self.onBoot = onBoot
    }

    func boot(vmID: String, templatePath: String, workspacePath: String, startupTimeoutSeconds: TimeInterval) throws {
        lastReadinessTimeoutSeconds = startupTimeoutSeconds
        onBoot(vmID)
    }

    func stop(vmID: String) throws {
        stoppedVMIDs.append(vmID)
    }
}

final class ReadyGuestBridge: GuestBridging {
    func waitUntilReady(vmID: String, timeoutSeconds: TimeInterval) throws {}

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

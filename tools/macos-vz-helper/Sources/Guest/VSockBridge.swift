import Foundation

protocol GuestReadinessBridging {
    func waitUntilReady(vmID: String, timeoutSeconds: TimeInterval) throws
}

struct GuestExecResult {
    let exitCode: Int
    let stdout: String
    let stderr: String
}

protocol GuestBridging: GuestReadinessBridging {
    func exec(
        vmID: String,
        argv: [String],
        cwd: String,
        env: [String: String],
        timeoutSeconds: TimeInterval
    ) throws -> GuestExecResult
}

enum GuestBridgeError: Error {
    case guestReadinessNotImplemented
    case guestExecNotImplemented
}

final class VSockBridge: GuestBridging {
    func waitUntilReady(vmID: String, timeoutSeconds: TimeInterval) throws {
        throw GuestBridgeError.guestReadinessNotImplemented
    }

    func exec(
        vmID: String,
        argv: [String],
        cwd: String,
        env: [String: String],
        timeoutSeconds: TimeInterval
    ) throws -> GuestExecResult {
        throw GuestBridgeError.guestExecNotImplemented
    }
}

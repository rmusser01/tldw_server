import Foundation

protocol GuestReadinessBridging {
    func waitUntilReady(vmID: String, timeoutSeconds: TimeInterval) throws
}

enum GuestBridgeError: Error {
    case guestReadinessNotImplemented
}

final class VSockBridge: GuestReadinessBridging {
    func waitUntilReady(vmID: String, timeoutSeconds: TimeInterval) throws {
        throw GuestBridgeError.guestReadinessNotImplemented
    }
}

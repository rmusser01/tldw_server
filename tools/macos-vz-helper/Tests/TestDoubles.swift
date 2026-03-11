import Foundation
@testable import MacOSVZHelperDaemon

final class RecordingBootDriver: VZBootDriving {
    private let onBoot: (String) -> Void
    private(set) var stoppedVMIDs: [String] = []

    init(onBoot: @escaping (String) -> Void = { _ in }) {
        self.onBoot = onBoot
    }

    func boot(vmID: String, templatePath: String, workspacePath: String) throws {
        onBoot(vmID)
    }

    func stop(vmID: String) throws {
        stoppedVMIDs.append(vmID)
    }
}

final class ReadyGuestBridge: GuestReadinessBridging {
    func waitUntilReady(vmID: String, timeoutSeconds: TimeInterval) throws {}
}

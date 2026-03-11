import Foundation

protocol VZBootDriving {
    func boot(vmID: String, templatePath: String, workspacePath: String) throws
    func stop(vmID: String) throws
}

enum VZLinuxVMManagerError: Error {
    case bootNotImplemented
}

final class PlaceholderVZBootDriver: VZBootDriving {
    func boot(vmID: String, templatePath: String, workspacePath: String) throws {
        throw VZLinuxVMManagerError.bootNotImplemented
    }

    func stop(vmID: String) throws {}
}

final class VZLinuxVMManager {
    private let registry: VMRegistry
    private let bootDriver: VZBootDriving
    private let guestBridge: GuestReadinessBridging

    init(
        registry: VMRegistry = VMRegistry(),
        bootDriver: VZBootDriving = PlaceholderVZBootDriver(),
        guestBridge: GuestReadinessBridging = VSockBridge()
    ) {
        self.registry = registry
        self.bootDriver = bootDriver
        self.guestBridge = guestBridge
    }

    @discardableResult
    func createVM(
        vmID: String,
        templatePath: String,
        workspacePath: String,
        readinessTimeoutSeconds: TimeInterval
    ) throws -> VMRecord {
        registry.upsert(vmID: vmID, state: "booting", healthy: false)
        do {
            try bootDriver.boot(vmID: vmID, templatePath: templatePath, workspacePath: workspacePath)
            try guestBridge.waitUntilReady(vmID: vmID, timeoutSeconds: readinessTimeoutSeconds)
            registry.upsert(vmID: vmID, state: "running", healthy: true)
            return VMRecord(vmID: vmID, state: "running", healthy: true)
        } catch {
            registry.remove(vmID: vmID)
            throw error
        }
    }

    func terminateVM(vmID: String) throws -> Bool {
        guard registry.status(vmID: vmID) != nil else {
            return false
        }
        try bootDriver.stop(vmID: vmID)
        registry.remove(vmID: vmID)
        return true
    }
}

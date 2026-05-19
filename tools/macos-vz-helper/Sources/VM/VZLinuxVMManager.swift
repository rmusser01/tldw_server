import Foundation

protocol VZBootDriving {
    @discardableResult
    func boot(vmID: String, templatePath: String, workspacePath: String, startupTimeoutSeconds: TimeInterval) throws -> VMResourceSnapshot
    func stop(vmID: String) throws
}

enum VZLinuxVMManagerError: Error {
    case bootNotImplemented
}

final class PlaceholderVZBootDriver: VZBootDriving {
    @discardableResult
    func boot(vmID: String, templatePath: String, workspacePath: String, startupTimeoutSeconds: TimeInterval) throws -> VMResourceSnapshot {
        throw VZLinuxVMManagerError.bootNotImplemented
    }

    func stop(vmID: String) throws {}
}

final class VZLinuxVMManager {
    private let registry: VMRegistry
    private let bootDriver: VZBootDriving
    private let guestBridge: GuestBridging

    init(
        registry: VMRegistry = VMRegistry(),
        bootDriver: VZBootDriving? = nil,
        guestBridge: GuestBridging? = nil,
        sessionManager: VSockSessionManager = VSockSessionManager()
    ) {
        self.registry = registry
        self.bootDriver = bootDriver ?? VirtualizationLinuxBootDriver(sessionManager: sessionManager)
        self.guestBridge = guestBridge ?? VSockBridge(transport: sessionManager)
    }

    @discardableResult
    func createVM(
        vmID: String,
        templatePath: String,
        workspacePath: String,
        readinessTimeoutSeconds: TimeInterval,
        metadata: VMOwnershipMetadata = .unknown
    ) throws -> VMRecord {
        registry.upsert(
            vmID: vmID,
            state: "booting",
            healthy: false,
            metadata: metadata,
            preserveGuestInfo: false,
            preserveResourceSnapshot: false
        )
        do {
            let resourceSnapshot = try bootDriver.boot(
                vmID: vmID,
                templatePath: templatePath,
                workspacePath: workspacePath,
                startupTimeoutSeconds: readinessTimeoutSeconds
            )
            try guestBridge.waitUntilReady(vmID: vmID, timeoutSeconds: readinessTimeoutSeconds)
            let guestInfo = guestBridge.guestInfo(vmID: vmID)
            registry.upsert(
                vmID: vmID,
                state: "running",
                healthy: true,
                guestInfo: guestInfo,
                resourceSnapshot: resourceSnapshot,
                preserveGuestInfo: false
            )
            return registry.status(vmID: vmID) ?? VMRecord(
                vmID: vmID,
                state: "running",
                healthy: true,
                metadata: metadata,
                guestInfo: guestInfo,
                resourceSnapshot: resourceSnapshot
            )
        } catch {
            try? bootDriver.stop(vmID: vmID)
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

    func execGuest(
        vmID: String,
        argv: [String],
        cwd: String,
        env: [String: String],
        timeoutSeconds: TimeInterval,
        maxOutputBytes: Int? = nil
    ) throws -> GuestExecResult {
        guard let record = registry.status(vmID: vmID), record.healthy else {
            throw GuestBridgeError.guestExecNotImplemented
        }
        return try guestBridge.exec(
            vmID: vmID,
            argv: argv,
            cwd: cwd,
            env: env,
            timeoutSeconds: timeoutSeconds,
            maxOutputBytes: maxOutputBytes
        )
    }
}

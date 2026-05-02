import Foundation

final class VSockSessionManager: GuestTransporting {
    private let lock = NSLock()
    private var sessions: [String: VSockSession] = [:]
    private var listeners: [String: VSockListener] = [:]

    @discardableResult
    func prepareSession(
        vmID: String,
        connectionToken: String,
        port: UInt32,
        workspaceRoot: String
    ) -> VSockListener {
        let session = VSockSession(
            vmID: vmID,
            connectionToken: connectionToken,
            workspaceRoot: workspaceRoot,
            port: port
        )
        let listener = VSockListener(vmID: vmID, manager: self)
        lock.withLock {
            sessions[vmID] = session
            listeners[vmID] = listener
        }
        return listener
    }

    func waitUntilGuestReady(vmID: String, timeoutSeconds: TimeInterval) throws {
        let session = try requireSession(vmID: vmID)
        try session.waitUntilReady(timeoutSeconds: timeoutSeconds)
    }

    func sendExecRequest(vmID: String, requestData: Data, timeoutSeconds: TimeInterval) throws -> Data {
        let session = try requireSession(vmID: vmID)
        return try session.sendExecRequest(requestData, timeoutSeconds: timeoutSeconds)
    }

    func guestInfo(vmID: String) -> GuestAgentInfo? {
        guard let session = try? requireSession(vmID: vmID) else {
            return nil
        }
        return session.currentGuestInfo()
    }

    func accept(channel: VSockChanneling, for vmID: String) -> Bool {
        do {
            let session = try requireSession(vmID: vmID)
            session.attach(channel: channel)
            return true
        } catch {
            channel.close()
            return false
        }
    }

    func removeSession(vmID: String) {
        let removed: VSockSession? = lock.withLock {
            listeners.removeValue(forKey: vmID)
            return sessions.removeValue(forKey: vmID)
        }
        removed?.attach(channel: ClosedVSockChannel())
    }

    func hasPreparedSession(vmID: String) -> Bool {
        lock.withLock {
            sessions[vmID] != nil
        }
    }

    private func requireSession(vmID: String) throws -> VSockSession {
        try lock.withLock {
            guard let session = sessions[vmID] else {
                throw VSockSessionError.sessionNotFound(vmID)
            }
            return session
        }
    }
}

private final class ClosedVSockChannel: VSockChanneling {
    func startReading(_ handler: @escaping (Result<Data, Error>) -> Void) {
        handler(.failure(VSockSessionError.closed))
    }

    func writeLine(_ data: Data) throws {
        throw VSockSessionError.closed
    }

    func close() {}
}

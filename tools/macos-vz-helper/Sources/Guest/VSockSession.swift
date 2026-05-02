import Foundation

enum VSockSessionError: Error {
    case sessionNotFound(String)
    case connectionNotReady(String)
    case connectionRejected(String)
    case requestAlreadyInFlight(String)
    case requestTimedOut(String)
    case invalidMessage(String)
    case closed
}

protocol VSockChanneling: AnyObject {
    func startReading(_ handler: @escaping (Result<Data, Error>) -> Void)
    func writeLine(_ data: Data) throws
    func close()
}

private struct VSockPendingRequest {
    let requestID: String
    let semaphore: DispatchSemaphore
    var result: Result<Data, Error>?
}

private let maxAbandonedExecRequestIDs = 128
private let maxGuestCapabilityCount = 128
private let maxGuestCapabilityBytes = 256

final class VSockSession {
    let vmID: String
    let connectionToken: String
    let workspaceRoot: String
    let port: UInt32

    private let lock = NSLock()
    private var channel: VSockChanneling?
    private var ready = false
    private var readinessError: Error?
    private var pendingRequest: VSockPendingRequest?
    private var abandonedRequestIDs: Set<String> = []
    private var abandonedRequestIDOrder: [String] = []
    private var readinessWaiters: [DispatchSemaphore] = []
    private var guestInfo: GuestAgentInfo?

    init(
        vmID: String,
        connectionToken: String,
        workspaceRoot: String,
        port: UInt32
    ) {
        self.vmID = vmID
        self.connectionToken = connectionToken
        self.workspaceRoot = workspaceRoot
        self.port = port
    }

    func attach(channel newChannel: VSockChanneling) {
        lock.lock()
        let previousChannel = channel
        channel = newChannel
        ready = false
        readinessError = nil
        lock.unlock()

        previousChannel?.close()
        newChannel.startReading { [weak self] result in
            self?.handleIncoming(result)
        }
    }

    func waitUntilReady(timeoutSeconds: TimeInterval) throws {
        if let result = currentReadinessResult() {
            switch result {
            case .success:
                return
            case let .failure(error):
                throw error
            }
        }

        let waiter = DispatchSemaphore(value: 0)
        lock.lock()
        readinessWaiters.append(waiter)
        lock.unlock()

        let timedOut = waiter.wait(timeout: .now() + timeoutSeconds) == .timedOut
        if timedOut {
            throw VSockSessionError.requestTimedOut(vmID)
        }

        if let result = currentReadinessResult() {
            switch result {
            case .success:
                return
            case let .failure(error):
                throw error
            }
        }

        throw VSockSessionError.connectionNotReady(vmID)
    }

    func sendExecRequest(_ requestData: Data, timeoutSeconds: TimeInterval) throws -> Data {
        let requestID = try extractRequestID(from: requestData)
        let waiter = DispatchSemaphore(value: 0)

        let activeChannel: VSockChanneling = try lock.withLock {
            guard ready, readinessError == nil else {
                throw VSockSessionError.connectionNotReady(vmID)
            }
            guard pendingRequest == nil else {
                throw VSockSessionError.requestAlreadyInFlight(vmID)
            }
            guard let channel else {
                throw VSockSessionError.connectionNotReady(vmID)
            }
            pendingRequest = VSockPendingRequest(
                requestID: requestID,
                semaphore: waiter,
                result: nil
            )
            return channel
        }

        do {
            try activeChannel.writeLine(requestData)
        } catch {
            completePendingRequest(with: .failure(error))
            throw error
        }

        let timedOut = waiter.wait(timeout: .now() + timeoutSeconds) == .timedOut
        if timedOut {
            markPendingRequestTimedOut(requestID)
            throw VSockSessionError.requestTimedOut(requestID)
        }

        let result = try lock.withLock { () throws -> Result<Data, Error> in
            guard let pendingRequest else {
                throw VSockSessionError.connectionNotReady(vmID)
            }
            if let result = pendingRequest.result {
                self.pendingRequest = nil
                return result
            }
            self.pendingRequest = nil
            throw VSockSessionError.requestTimedOut(requestID)
        }

        switch result {
        case let .success(data):
            return data
        case let .failure(error):
            throw error
        }
    }

    func currentGuestInfo() -> GuestAgentInfo? {
        lock.withLock {
            guestInfo
        }
    }

    private func currentReadinessResult() -> Result<Void, Error>? {
        lock.withLock {
            if ready {
                return .success(())
            }
            if let readinessError {
                return .failure(readinessError)
            }
            return nil
        }
    }

    private func handleIncoming(_ result: Result<Data, Error>) {
        switch result {
        case let .failure(error):
            lock.lock()
            readinessError = error
            ready = false
            let waiters = readinessWaiters
            readinessWaiters.removeAll()
            let pending = pendingRequest
            pendingRequest = nil
            lock.unlock()

            waiters.forEach { $0.signal() }
            pending?.semaphore.signal()
        case let .success(line):
            do {
                try handleLine(line)
            } catch {
                handleIncoming(.failure(error))
            }
        }
    }

    private func handleLine(_ line: Data) throws {
        let payload = try decodeJSONObject(from: line)
        if let type = payload["type"] as? String {
            switch type {
            case "handshake":
                try handleHandshake(payload)
            case "reconnect":
                try handleReconnect(payload)
            case "ready":
                try handleReady(payload)
            case "heartbeat":
                try handleHeartbeat(payload)
            default:
                throw VSockSessionError.invalidMessage(type)
            }
            return
        }

        if payload["request_id"] != nil, payload["exit_code"] != nil || payload["error_code"] != nil {
            try handleExecResponse(line, payload: payload)
            return
        }

        throw VSockSessionError.invalidMessage("missing message type")
    }

    private func handleHandshake(_ payload: [String: Any]) throws {
        try validateControlMessage(payload, requireToken: true)
        let parsedGuestInfo = parseGuestInfo(from: payload)
        try writeJSONObject([
            "protocol_version": guestProtocolVersion,
            "request_id": try requireString("request_id", in: payload),
            "type": "handshake_ack",
            "status": "accepted",
            "vm_id": vmID,
        ])
        lock.withLock {
            guestInfo = parsedGuestInfo
        }
    }

    private func parseGuestInfo(from payload: [String: Any]) -> GuestAgentInfo {
        let capabilities = parseGuestCapabilities(from: payload)
        return GuestAgentInfo(
            guestVersion: optionalString("guest_version", in: payload),
            workspaceRoot: optionalString("workspace_root", in: payload),
            capabilities: capabilities.values,
            capabilitiesKnown: capabilities.known
        )
    }

    private func parseGuestCapabilities(from payload: [String: Any]) -> (known: Bool, values: [String]) {
        guard let rawCapabilities = payload["capabilities"] else {
            return (false, [])
        }
        guard let values = rawCapabilities as? [Any], values.count <= maxGuestCapabilityCount else {
            return (false, [])
        }

        var capabilities = Set<String>()
        capabilities.reserveCapacity(values.count)
        for value in values {
            guard let capability = value as? String,
                  !capability.isEmpty,
                  capability.utf8.count <= maxGuestCapabilityBytes else {
                return (false, [])
            }
            capabilities.insert(capability)
        }
        return (true, capabilities.sorted())
    }

    private func handleReconnect(_ payload: [String: Any]) throws {
        try validateControlMessage(payload, requireToken: true)
        try writeJSONObject([
            "protocol_version": guestProtocolVersion,
            "request_id": try requireString("request_id", in: payload),
            "type": "reconnect_ack",
            "status": "accepted",
            "vm_id": vmID,
        ])
    }

    private func handleReady(_ payload: [String: Any]) throws {
        try validateProtocolVersion(payload)
        try writeJSONObject([
            "protocol_version": guestProtocolVersion,
            "request_id": try requireString("request_id", in: payload),
            "status": "ready",
            "workspace_root": workspaceRoot,
        ])

        lock.lock()
        ready = true
        readinessError = nil
        let waiters = readinessWaiters
        readinessWaiters.removeAll()
        lock.unlock()

        waiters.forEach { $0.signal() }
    }

    private func handleHeartbeat(_ payload: [String: Any]) throws {
        try validateProtocolVersion(payload)
        if let remoteVMID = payload["vm_id"] as? String, remoteVMID != vmID {
            throw VSockSessionError.connectionRejected(remoteVMID)
        }
        try writeJSONObject([
            "protocol_version": guestProtocolVersion,
            "request_id": try requireString("request_id", in: payload),
            "type": "heartbeat",
            "status": "alive",
            "vm_id": vmID,
        ])
    }

    private func handleExecResponse(_ line: Data, payload: [String: Any]) throws {
        let requestID = try requireString("request_id", in: payload)

        var pending: VSockPendingRequest?
        var shouldIgnoreLateResponse = false
        lock.lock()
        if let current = pendingRequest, current.requestID == requestID {
            var updated = current
            updated.result = .success(line)
            pendingRequest = updated
            pending = updated
        } else if abandonedRequestIDs.remove(requestID) != nil {
            if let index = abandonedRequestIDOrder.firstIndex(of: requestID) {
                abandonedRequestIDOrder.remove(at: index)
            }
            shouldIgnoreLateResponse = true
        }
        lock.unlock()

        if shouldIgnoreLateResponse {
            return
        }

        guard let pending else {
            throw VSockSessionError.invalidMessage(requestID)
        }

        pending.semaphore.signal()
    }

    private func completePendingRequest(with result: Result<Data, Error>) {
        var pending: VSockPendingRequest?
        lock.lock()
        if var current = pendingRequest {
            current.result = result
            pendingRequest = current
            pending = current
        }
        lock.unlock()
        pending?.semaphore.signal()
    }

    private func markPendingRequestTimedOut(_ requestID: String) {
        lock.lock()
        if let current = pendingRequest, current.requestID == requestID {
            pendingRequest = nil
            rememberAbandonedRequestID(requestID)
        }
        lock.unlock()
    }

    private func rememberAbandonedRequestID(_ requestID: String) {
        if abandonedRequestIDs.insert(requestID).inserted {
            abandonedRequestIDOrder.append(requestID)
        }
        while abandonedRequestIDOrder.count > maxAbandonedExecRequestIDs {
            let evictedRequestID = abandonedRequestIDOrder.removeFirst()
            abandonedRequestIDs.remove(evictedRequestID)
        }
    }

    private func validateControlMessage(_ payload: [String: Any], requireToken: Bool) throws {
        try validateProtocolVersion(payload)
        let remoteVMID = try requireString("vm_id", in: payload)
        guard remoteVMID == vmID else {
            throw VSockSessionError.connectionRejected(remoteVMID)
        }
        if requireToken {
            let token = try requireString("connection_token", in: payload)
            guard token == connectionToken else {
                throw VSockSessionError.connectionRejected(remoteVMID)
            }
        }
    }

    private func validateProtocolVersion(_ payload: [String: Any]) throws {
        let protocolVersion = try requireString("protocol_version", in: payload)
        guard protocolVersion == guestProtocolVersion else {
            throw VSockSessionError.invalidMessage("protocol_mismatch")
        }
    }

    private func decodeJSONObject(from data: Data) throws -> [String: Any] {
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw VSockSessionError.invalidMessage("invalid_json")
        }
        return object
    }

    private func writeJSONObject(_ object: [String: Any]) throws {
        let data = try JSONSerialization.data(withJSONObject: object)
        try lock.withLock {
            guard let channel else {
                throw VSockSessionError.connectionNotReady(vmID)
            }
            try channel.writeLine(data)
        }
    }

    private func extractRequestID(from data: Data) throws -> String {
        let object = try decodeJSONObject(from: data)
        return try requireString("request_id", in: object)
    }

    private func requireString(_ key: String, in object: [String: Any]) throws -> String {
        guard let value = object[key] as? String, !value.isEmpty else {
            throw VSockSessionError.invalidMessage(key)
        }
        return value
    }

    private func optionalString(_ key: String, in object: [String: Any]) -> String? {
        guard let value = object[key] as? String, !value.isEmpty else {
            return nil
        }
        return value
    }
}

private extension NSLock {
    func withLock<T>(_ body: () throws -> T) rethrows -> T {
        lock()
        defer { unlock() }
        return try body()
    }
}

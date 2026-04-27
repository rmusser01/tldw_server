import Darwin
import Foundation
import Testing
@testable import MacOSVZHelperDaemon

@Test func unixSocketServerHandlesPingRequest() throws {
    let server = UnixSocketServer(
        socketPath: "/tmp/macos-vz-helper.sock",
        service: HelperService()
    )

    let request = """
    {"operation":"ping","protocol_version":"1","request":{}}
    """.data(using: .utf8)!

    let responseData = try server.handleRequestData(request)
    let json = try JSONSerialization.jsonObject(with: responseData) as? [String: Any]

    #expect(json?["protocol_version"] as? String == "1")
    #expect(json?["helper_version"] as? String == "0.1.0")
    #expect(json?["status"] as? String == "ok")
}

@Test func unixSocketServerHandlesCreateVMAndExecGuestRequests() throws {
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
    let server = UnixSocketServer(
        socketPath: "/tmp/macos-vz-helper.sock",
        service: service
    )

    let createRequest = """
    {"operation":"create_vm","protocol_version":"1","request":{"vm_name":"vm-route","template":"/tmp/template.img","workspace_path":"/workspace"}}
    """.data(using: .utf8)!
    let createResponseData = try server.handleRequestData(createRequest)
    let createJSON = try JSONSerialization.jsonObject(with: createResponseData) as? [String: Any]
    #expect(createJSON?["protocol_version"] as? String == "1")
    #expect(createJSON?["helper_version"] as? String == "0.1.0")
    #expect(createJSON?["vm_id"] as? String == "vm-route")
    #expect(createJSON?["state"] as? String == "running")

    let execRequest = """
    {"operation":"exec_guest","protocol_version":"1","request":{"vm_id":"vm-route","argv":["/bin/echo","ok"],"cwd":"/workspace","timeout_sec":15}}
    """.data(using: .utf8)!
    let execResponseData = try server.handleRequestData(execRequest)
    let execJSON = try JSONSerialization.jsonObject(with: execResponseData) as? [String: Any]
    #expect(execJSON?["protocol_version"] as? String == "1")
    #expect(execJSON?["helper_version"] as? String == "0.1.0")
    #expect(execJSON?["exit_code"] as? Int == 0)
    #expect(execJSON?["stdout"] as? String == "ok\n")
    #expect(guestBridge.lastExec?.vmID == "vm-route")
}

@Test func unixSocketServerServesPingOverRealSocket() throws {
    let socketPath = "/tmp/macos-vz-helper-\(UUID().uuidString.prefix(8)).sock"
    let server = UnixSocketServer(
        socketPath: socketPath,
        service: HelperService()
    )

    let queue = DispatchQueue(label: "macos-vz-helper-test-server")
    queue.async {
        try? server.run()
    }
    defer {
        server.stop()
        unlink(socketPath)
    }

    try waitForSocket(at: socketPath)

    let responseData = try sendSocketRequest(
        socketPath: socketPath,
        payload: """
        {"operation":"ping","protocol_version":"1","request":{}}
        """ + "\n"
    )
    let responseJSON = try JSONSerialization.jsonObject(with: responseData) as? [String: Any]

    #expect(responseJSON?["protocol_version"] as? String == "1")
    #expect(responseJSON?["helper_version"] as? String == "0.1.0")
    #expect(responseJSON?["status"] as? String == "ok")
}

private func waitForSocket(at path: String, timeoutSeconds: TimeInterval = 2.0) throws {
    let deadline = Date().addingTimeInterval(timeoutSeconds)
    while Date() < deadline {
        if FileManager.default.fileExists(atPath: path) {
            return
        }
        Thread.sleep(forTimeInterval: 0.05)
    }
    throw TestFailure.socketNotReady
}

private func sendSocketRequest(socketPath: String, payload: String) throws -> Data {
    let clientFD = Darwin.socket(AF_UNIX, SOCK_STREAM, 0)
    guard clientFD >= 0 else {
        throw TestFailure.clientSocketUnavailable
    }
    defer {
        close(clientFD)
    }

    var address = sockaddr_un()
    address.sun_family = sa_family_t(AF_UNIX)
    let pathBytes = Array(socketPath.utf8CString)
    let maxLength = MemoryLayout.size(ofValue: address.sun_path)
    if pathBytes.count > maxLength {
        throw TestFailure.socketPathTooLong
    }
    withUnsafeMutablePointer(to: &address.sun_path.0) { pointer in
        pointer.initialize(repeating: 0, count: maxLength)
        for (index, byte) in pathBytes.enumerated() {
            pointer.advanced(by: index).pointee = byte
        }
    }

    let addressLength = socklen_t(MemoryLayout<sa_family_t>.size + pathBytes.count)
    let connectResult = withUnsafePointer(to: &address) { pointer in
        pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockaddrPointer in
            connect(clientFD, sockaddrPointer, addressLength)
        }
    }
    guard connectResult == 0 else {
        throw TestFailure.socketConnectFailed
    }

    let payloadData = Data(payload.utf8)
    try payloadData.withUnsafeBytes { rawBuffer in
        guard let baseAddress = rawBuffer.baseAddress else {
            return
        }
        if Darwin.write(clientFD, baseAddress, rawBuffer.count) < 0 {
            throw TestFailure.socketWriteFailed
        }
    }

    var buffer = [UInt8](repeating: 0, count: 4096)
    let readCount = recv(clientFD, &buffer, buffer.count, 0)
    guard readCount > 0 else {
        throw TestFailure.socketReadFailed
    }
    return Data(buffer.prefix(readCount)).trimmingTrailingNewlines()
}

private enum TestFailure: Error {
    case clientSocketUnavailable
    case socketConnectFailed
    case socketNotReady
    case socketPathTooLong
    case socketReadFailed
    case socketWriteFailed
}

private extension Data {
    func trimmingTrailingNewlines() -> Data {
        var trimmed = self
        while let last = trimmed.last, last == 0x0A || last == 0x0D {
            trimmed.removeLast()
        }
        return trimmed
    }
}

import Foundation
import Testing
@testable import MacOSVZHelperDaemon

final class InMemoryVSockChannel: VSockChanneling {
    private var handler: ((Result<Data, Error>) -> Void)?
    private(set) var writes: [[String: Any]] = []
    var onWrite: (([String: Any]) -> Void)?

    func startReading(_ handler: @escaping (Result<Data, Error>) -> Void) {
        self.handler = handler
    }

    func writeLine(_ data: Data) throws {
        let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
        writes.append(object)
        onWrite?(object)
    }

    func close() {}

    func push(json: String) {
        handler?(.success(Data(json.utf8)))
    }

    func fail(_ error: Error) {
        handler?(.failure(error))
    }
}

@Test func vsockSessionManagerBindsHandshakeToExpectedVMID() throws {
    let manager = VSockSessionManager()
    _ = manager.prepareSession(
        vmID: "vm-1",
        connectionToken: "token-1",
        port: 1024,
        workspaceRoot: "/workspace"
    )
    let channel = InMemoryVSockChannel()

    #expect(manager.accept(channel: channel, for: "vm-1") == true)

    channel.push(
        json: #"{"protocol_version":"1","request_id":"req-handshake","type":"handshake","vm_id":"vm-1","connection_token":"token-1","guest_version":"1.0.0","workspace_root":"/workspace"}"#
    )
    channel.push(
        json: #"{"protocol_version":"1","request_id":"req-ready","type":"ready"}"#
    )

    try manager.waitUntilGuestReady(vmID: "vm-1", timeoutSeconds: 0.1)

    #expect(channel.writes.count == 2)
    #expect(channel.writes.first?["type"] as? String == "handshake_ack")
    #expect(channel.writes.first?["vm_id"] as? String == "vm-1")
    #expect(channel.writes.last?["status"] as? String == "ready")
    #expect(channel.writes.last?["workspace_root"] as? String == "/workspace")
}

@Test func vsockSessionManagerRejectsWrongConnectionToken() throws {
    let manager = VSockSessionManager()
    _ = manager.prepareSession(
        vmID: "vm-2",
        connectionToken: "token-expected",
        port: 1024,
        workspaceRoot: "/workspace"
    )
    let channel = InMemoryVSockChannel()

    #expect(manager.accept(channel: channel, for: "vm-2") == true)
    channel.push(
        json: #"{"protocol_version":"1","request_id":"req-handshake","type":"handshake","vm_id":"vm-2","connection_token":"token-wrong"}"#
    )

    do {
        try manager.waitUntilGuestReady(vmID: "vm-2", timeoutSeconds: 0.1)
        Issue.record("expected wrong token to fail readiness")
    } catch {
        #expect(channel.writes.isEmpty)
    }
}

@Test func vsockBridgeExecUsesBoundSessionTransport() throws {
    let manager = VSockSessionManager()
    _ = manager.prepareSession(
        vmID: "vm-exec",
        connectionToken: "token-exec",
        port: 1024,
        workspaceRoot: "/workspace"
    )
    let channel = InMemoryVSockChannel()
    channel.onWrite = { payload in
        guard payload["type"] as? String == "exec" else {
            return
        }
        let requestID = payload["request_id"] as? String ?? ""
        channel.push(
            json: #"{"protocol_version":"1","request_id":"\#(requestID)","exit_code":0,"stdout":"ok\n","stderr":""}"#
        )
    }

    #expect(manager.accept(channel: channel, for: "vm-exec") == true)
    channel.push(
        json: #"{"protocol_version":"1","request_id":"req-handshake","type":"handshake","vm_id":"vm-exec","connection_token":"token-exec"}"#
    )
    channel.push(
        json: #"{"protocol_version":"1","request_id":"req-ready","type":"ready"}"#
    )

    let bridge = VSockBridge(transport: manager)
    try bridge.waitUntilReady(vmID: "vm-exec", timeoutSeconds: 0.1)

    let result = try bridge.exec(
        vmID: "vm-exec",
        argv: ["/bin/echo", "ok"],
        cwd: "/workspace",
        env: ["FOO": "1"],
        timeoutSeconds: 1
    )

    #expect(result.exitCode == 0)
    #expect(result.stdout == "ok\n")
    #expect(channel.writes.count == 3)
    #expect(channel.writes.last?["type"] as? String == "exec")
}

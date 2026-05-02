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

@Test func vsockSessionManagerReportsGuestInfoFromHandshakeCapabilities() throws {
    let manager = VSockSessionManager()
    _ = manager.prepareSession(
        vmID: "vm-capabilities",
        connectionToken: "token-capabilities",
        port: 1024,
        workspaceRoot: "/workspace"
    )
    let channel = InMemoryVSockChannel()

    #expect(manager.accept(channel: channel, for: "vm-capabilities") == true)

    channel.push(
        json: #"{"protocol_version":"1","request_id":"req-handshake","type":"handshake","vm_id":"vm-capabilities","connection_token":"token-capabilities","guest_version":"1.0.0","workspace_root":"/workspace","capabilities":["output_cap_v1","exec","exec"]}"#
    )
    channel.push(
        json: #"{"protocol_version":"1","request_id":"req-ready","type":"ready"}"#
    )

    try manager.waitUntilGuestReady(vmID: "vm-capabilities", timeoutSeconds: 0.1)

    let info = manager.guestInfo(vmID: "vm-capabilities")
    #expect(info?.guestVersion == "1.0.0")
    #expect(info?.workspaceRoot == "/workspace")
    #expect(info?.capabilitiesKnown == true)
    #expect(info?.capabilities == ["exec", "output_cap_v1"])
}

@Test func vsockSessionManagerReportsUnknownCapabilitiesForOlderGuests() throws {
    let manager = VSockSessionManager()
    _ = manager.prepareSession(
        vmID: "vm-old-guest",
        connectionToken: "token-old-guest",
        port: 1024,
        workspaceRoot: "/workspace"
    )
    let channel = InMemoryVSockChannel()

    #expect(manager.accept(channel: channel, for: "vm-old-guest") == true)

    channel.push(
        json: #"{"protocol_version":"1","request_id":"req-handshake","type":"handshake","vm_id":"vm-old-guest","connection_token":"token-old-guest","guest_version":"0.9.0","workspace_root":"/workspace"}"#
    )
    channel.push(
        json: #"{"protocol_version":"1","request_id":"req-ready","type":"ready"}"#
    )

    try manager.waitUntilGuestReady(vmID: "vm-old-guest", timeoutSeconds: 0.1)

    let info = manager.guestInfo(vmID: "vm-old-guest")
    #expect(info?.guestVersion == "0.9.0")
    #expect(info?.workspaceRoot == "/workspace")
    #expect(info?.capabilitiesKnown == false)
    #expect(info?.capabilities == [String]())
}

@Test func vsockSessionManagerTreatsMalformedCapabilitiesAsUnknown() throws {
    let manager = VSockSessionManager()
    _ = manager.prepareSession(
        vmID: "vm-malformed-capabilities",
        connectionToken: "token-malformed-capabilities",
        port: 1024,
        workspaceRoot: "/workspace"
    )
    let channel = InMemoryVSockChannel()

    #expect(manager.accept(channel: channel, for: "vm-malformed-capabilities") == true)

    channel.push(
        json: #"{"protocol_version":"1","request_id":"req-handshake","type":"handshake","vm_id":"vm-malformed-capabilities","connection_token":"token-malformed-capabilities","guest_version":"1.0.0","workspace_root":"/workspace","capabilities":["exec",1]}"#
    )
    channel.push(
        json: #"{"protocol_version":"1","request_id":"req-ready","type":"ready"}"#
    )

    try manager.waitUntilGuestReady(vmID: "vm-malformed-capabilities", timeoutSeconds: 0.1)

    let info = manager.guestInfo(vmID: "vm-malformed-capabilities")
    #expect(info?.guestVersion == "1.0.0")
    #expect(info?.workspaceRoot == "/workspace")
    #expect(info?.capabilitiesKnown == false)
    #expect(info?.capabilities == [String]())
}

@Test func vsockSessionManagerTreatsTooManyCapabilitiesAsUnknown() throws {
    let manager = VSockSessionManager()
    _ = manager.prepareSession(
        vmID: "vm-too-many-capabilities",
        connectionToken: "token-too-many-capabilities",
        port: 1024,
        workspaceRoot: "/workspace"
    )
    let channel = InMemoryVSockChannel()
    let capabilities = (0..<129).map { #""cap-\#($0)""# }.joined(separator: ",")

    #expect(manager.accept(channel: channel, for: "vm-too-many-capabilities") == true)

    channel.push(
        json: #"{"protocol_version":"1","request_id":"req-handshake","type":"handshake","vm_id":"vm-too-many-capabilities","connection_token":"token-too-many-capabilities","guest_version":"1.0.0","workspace_root":"/workspace","capabilities":[\#(capabilities)]}"#
    )
    channel.push(
        json: #"{"protocol_version":"1","request_id":"req-ready","type":"ready"}"#
    )

    try manager.waitUntilGuestReady(vmID: "vm-too-many-capabilities", timeoutSeconds: 0.1)

    let info = manager.guestInfo(vmID: "vm-too-many-capabilities")
    #expect(info?.capabilitiesKnown == false)
    #expect(info?.capabilities == [String]())
}

@Test func vsockSessionManagerTreatsOversizedCapabilitiesAsUnknown() throws {
    let manager = VSockSessionManager()
    _ = manager.prepareSession(
        vmID: "vm-oversized-capabilities",
        connectionToken: "token-oversized-capabilities",
        port: 1024,
        workspaceRoot: "/workspace"
    )
    let channel = InMemoryVSockChannel()
    let capability = String(repeating: "x", count: 257)

    #expect(manager.accept(channel: channel, for: "vm-oversized-capabilities") == true)

    channel.push(
        json: #"{"protocol_version":"1","request_id":"req-handshake","type":"handshake","vm_id":"vm-oversized-capabilities","connection_token":"token-oversized-capabilities","guest_version":"1.0.0","workspace_root":"/workspace","capabilities":["\#(capability)"]}"#
    )
    channel.push(
        json: #"{"protocol_version":"1","request_id":"req-ready","type":"ready"}"#
    )

    try manager.waitUntilGuestReady(vmID: "vm-oversized-capabilities", timeoutSeconds: 0.1)

    let info = manager.guestInfo(vmID: "vm-oversized-capabilities")
    #expect(info?.capabilitiesKnown == false)
    #expect(info?.capabilities == [String]())
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

@Test func vsockSessionIgnoresLateResponseAfterExecTimeout() throws {
    let manager = VSockSessionManager()
    _ = manager.prepareSession(
        vmID: "vm-timeout",
        connectionToken: "token-timeout",
        port: 1024,
        workspaceRoot: "/workspace"
    )
    let channel = InMemoryVSockChannel()
    var respondToExec = false
    channel.onWrite = { payload in
        guard payload["type"] as? String == "exec", respondToExec else {
            return
        }
        let requestID = payload["request_id"] as? String ?? ""
        channel.push(
            json: #"{"protocol_version":"1","request_id":"\#(requestID)","exit_code":0,"stdout":"ok\n","stderr":""}"#
        )
    }

    #expect(manager.accept(channel: channel, for: "vm-timeout") == true)
    channel.push(
        json: #"{"protocol_version":"1","request_id":"req-handshake","type":"handshake","vm_id":"vm-timeout","connection_token":"token-timeout"}"#
    )
    channel.push(
        json: #"{"protocol_version":"1","request_id":"req-ready","type":"ready"}"#
    )

    let bridge = VSockBridge(transport: manager)
    try bridge.waitUntilReady(vmID: "vm-timeout", timeoutSeconds: 0.1)

    #expect(throws: VSockSessionError.self) {
        _ = try bridge.exec(
            vmID: "vm-timeout",
            argv: ["/bin/sh", "-c", "sleep 2"],
            cwd: "/workspace",
            env: [:],
            timeoutSeconds: 0.01
        )
    }
    let timedOutRequestID = channel.writes.last?["request_id"] as? String ?? ""
    channel.push(
        json: #"{"protocol_version":"1","request_id":"\#(timedOutRequestID)","exit_code":0,"stdout":"late\n","stderr":""}"#
    )

    respondToExec = true
    let result = try bridge.exec(
        vmID: "vm-timeout",
        argv: ["/bin/echo", "ok"],
        cwd: "/workspace",
        env: [:],
        timeoutSeconds: 0.1
    )

    #expect(result.exitCode == 0)
    #expect(result.stdout == "ok\n")
}

@Test func vsockSessionIgnoresLateResponseAfterReconnectFollowingExecTimeout() throws {
    let manager = VSockSessionManager()
    _ = manager.prepareSession(
        vmID: "vm-timeout-reconnect",
        connectionToken: "token-timeout-reconnect",
        port: 1024,
        workspaceRoot: "/workspace"
    )
    let firstChannel = InMemoryVSockChannel()

    #expect(manager.accept(channel: firstChannel, for: "vm-timeout-reconnect") == true)
    firstChannel.push(
        json: #"{"protocol_version":"1","request_id":"req-handshake-1","type":"handshake","vm_id":"vm-timeout-reconnect","connection_token":"token-timeout-reconnect"}"#
    )
    firstChannel.push(
        json: #"{"protocol_version":"1","request_id":"req-ready-1","type":"ready"}"#
    )

    let bridge = VSockBridge(transport: manager)
    try bridge.waitUntilReady(vmID: "vm-timeout-reconnect", timeoutSeconds: 0.1)

    #expect(throws: VSockSessionError.self) {
        _ = try bridge.exec(
            vmID: "vm-timeout-reconnect",
            argv: ["/bin/sh", "-c", "sleep 2"],
            cwd: "/workspace",
            env: [:],
            timeoutSeconds: 0.01
        )
    }
    let timedOutRequestID = firstChannel.writes.last?["request_id"] as? String ?? ""

    let secondChannel = InMemoryVSockChannel()
    secondChannel.onWrite = { payload in
        guard payload["type"] as? String == "exec" else {
            return
        }
        let requestID = payload["request_id"] as? String ?? ""
        secondChannel.push(
            json: #"{"protocol_version":"1","request_id":"\#(requestID)","exit_code":0,"stdout":"ok\n","stderr":""}"#
        )
    }
    #expect(manager.accept(channel: secondChannel, for: "vm-timeout-reconnect") == true)
    secondChannel.push(
        json: #"{"protocol_version":"1","request_id":"req-handshake-2","type":"handshake","vm_id":"vm-timeout-reconnect","connection_token":"token-timeout-reconnect"}"#
    )
    secondChannel.push(
        json: #"{"protocol_version":"1","request_id":"req-ready-2","type":"ready"}"#
    )
    try bridge.waitUntilReady(vmID: "vm-timeout-reconnect", timeoutSeconds: 0.1)

    firstChannel.push(
        json: #"{"protocol_version":"1","request_id":"\#(timedOutRequestID)","exit_code":0,"stdout":"late\n","stderr":""}"#
    )

    let result = try bridge.exec(
        vmID: "vm-timeout-reconnect",
        argv: ["/bin/echo", "ok"],
        cwd: "/workspace",
        env: [:],
        timeoutSeconds: 0.1
    )

    #expect(result.exitCode == 0)
    #expect(result.stdout == "ok\n")
}

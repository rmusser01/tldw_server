import Foundation
import Testing
@testable import MacOSVZHelperDaemon

final class RecordingGuestTransport: GuestTransporting {
    private(set) var execPayload: [String: Any]?
    private(set) var readyVMID: String?
    private(set) var execVMID: String?
    private(set) var readyTimeout: TimeInterval?
    private(set) var execTimeout: TimeInterval?
    var readyError: Error?
    var execResponseFactory: (([String: Any]) -> Data)?

    func waitUntilGuestReady(vmID: String, timeoutSeconds: TimeInterval) throws {
        readyVMID = vmID
        readyTimeout = timeoutSeconds
        if let readyError {
            throw readyError
        }
    }

    func sendExecRequest(vmID: String, requestData: Data, timeoutSeconds: TimeInterval) throws -> Data {
        execVMID = vmID
        execTimeout = timeoutSeconds
        let payload = try JSONSerialization.jsonObject(with: requestData) as? [String: Any] ?? [:]
        execPayload = payload
        return execResponseFactory?(payload) ?? Data()
    }

    func guestInfo(vmID: String) -> GuestAgentInfo? {
        nil
    }
}

@Test func vsockBridgeEncodesReadyRequestsAndAcceptsReadyResponses() throws {
    let transport = RecordingGuestTransport()
    let bridge = VSockBridge(transport: transport)

    try bridge.waitUntilReady(vmID: "vm-ready", timeoutSeconds: 9)

    #expect(transport.readyVMID == "vm-ready")
    #expect(transport.readyTimeout == 9)
}

@Test func vsockBridgeEncodesExecRequestsAndReturnsGuestExecResult() throws {
    let transport = RecordingGuestTransport()
    transport.execResponseFactory = { payload in
        let requestID = payload["request_id"] as? String ?? ""
        return Data(
            """
            {"protocol_version":"1","request_id":"\(requestID)","exit_code":0,"stdout":"ok\\n","stderr":""}
            """.utf8
        )
    }
    let bridge = VSockBridge(transport: transport)

    let result = try bridge.exec(
        vmID: "vm-exec",
        argv: ["/bin/echo", "ok"],
        cwd: "/workspace",
        env: ["FOO": "1"],
        timeoutSeconds: 15,
        maxOutputBytes: 10
    )

    #expect(result.exitCode == 0)
    #expect(result.stdout == "ok\n")
    #expect(result.stderr == "")
    #expect(result.details.isEmpty)
    #expect(transport.execVMID == "vm-exec")
    #expect(transport.execTimeout == 15)
    #expect(transport.execPayload?["type"] as? String == "exec")
    #expect(transport.execPayload?["max_output_bytes"] as? Int == 10)
    let argv = transport.execPayload?["argv"] as? [String]
    #expect(argv == ["/bin/echo", "ok"])
}

@Test func vsockBridgeDecodesGuestPrefixedDetailsDefensively() throws {
    let transport = RecordingGuestTransport()
    transport.execResponseFactory = { payload in
        let requestID = payload["request_id"] as? String ?? ""
        return Data(
            """
            {"protocol_version":"1","request_id":"\(requestID)","exit_code":137,"stdout":"x","stderr":"","details":{"guest_output_limit_exceeded":"true","guest_stdout_bytes_observed":"17","ignored_number":1,"host_key":"wrong"}}
            """.utf8
        )
    }
    let bridge = VSockBridge(transport: transport)

    let result = try bridge.exec(
        vmID: "vm-exec-details",
        argv: ["/bin/echo", "ok"],
        cwd: "/workspace",
        env: [:],
        timeoutSeconds: 15,
        maxOutputBytes: 10
    )

    #expect(result.exitCode == 137)
    #expect(result.details["guest_output_limit_exceeded"] == "true")
    #expect(result.details["guest_stdout_bytes_observed"] == "17")
    #expect(result.details["ignored_number"] == nil)
    #expect(result.details["host_key"] == nil)
}

@Test func vsockBridgeRejectsNegativeMaxOutputBytesBeforeEncoding() throws {
    let transport = RecordingGuestTransport()
    let bridge = VSockBridge(transport: transport)

    do {
        _ = try bridge.exec(
            vmID: "vm-negative-cap",
            argv: ["/bin/echo", "ok"],
            cwd: "/workspace",
            env: [:],
            timeoutSeconds: 15,
            maxOutputBytes: -1
        )
        Issue.record("expected negative maxOutputBytes to be rejected")
    } catch GuestBridgeError.guestProtocolError(let reason) {
        #expect(reason == "invalid_max_output_bytes")
    } catch {
        Issue.record("expected guestProtocolError, got \(error)")
    }

    #expect(transport.execPayload == nil)
}

@Test func vsockBridgeTreatsMissingExecStreamsAsEmptyStrings() throws {
    let transport = RecordingGuestTransport()
    transport.execResponseFactory = { payload in
        let requestID = payload["request_id"] as? String ?? ""
        return Data(
            """
            {"protocol_version":"1","request_id":"\(requestID)","exit_code":0}
            """.utf8
        )
    }
    let bridge = VSockBridge(transport: transport)

    let result = try bridge.exec(
        vmID: "vm-exec",
        argv: ["/usr/bin/true"],
        cwd: "/workspace",
        env: [:],
        timeoutSeconds: 15
    )

    #expect(result.exitCode == 0)
    #expect(result.stdout == "")
    #expect(result.stderr == "")
}

@Test func vsockBridgeMapsGuestErrorResponsesToStructuredFailure() throws {
    let transport = RecordingGuestTransport()
    transport.execResponseFactory = { payload in
        let requestID = payload["request_id"] as? String ?? ""
        return Data(
            """
            {"protocol_version":"1","request_id":"\(requestID)","error_code":"exec_failed","message":"guest command failed"}
            """.utf8
        )
    }
    let bridge = VSockBridge(transport: transport)

    do {
        _ = try bridge.exec(
            vmID: "vm-fail",
            argv: ["/bin/false"],
            cwd: "/workspace",
            env: [:],
            timeoutSeconds: 5
        )
        Issue.record("expected guest operation failure")
    } catch let GuestBridgeError.guestOperationFailed(code, message) {
        #expect(code == "exec_failed")
        #expect(message == "guest command failed")
    }
}

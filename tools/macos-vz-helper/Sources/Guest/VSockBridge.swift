import Foundation

let guestProtocolVersion = "1"

protocol GuestReadinessBridging {
    func waitUntilReady(vmID: String, timeoutSeconds: TimeInterval) throws
}

struct GuestExecResult {
    let exitCode: Int
    let stdout: String
    let stderr: String
}

protocol GuestBridging: GuestReadinessBridging {
    func exec(
        vmID: String,
        argv: [String],
        cwd: String,
        env: [String: String],
        timeoutSeconds: TimeInterval
    ) throws -> GuestExecResult
}

enum GuestBridgeError: Error {
    case guestReadinessNotImplemented
    case guestExecNotImplemented
    case guestNotReady(String)
    case guestProtocolError(String)
    case guestOperationFailed(code: String, message: String)
}

private struct GuestReadyRequest: Encodable {
    let protocolVersion: String
    let requestID: String
    let type: String

    private enum CodingKeys: String, CodingKey {
        case protocolVersion = "protocol_version"
        case requestID = "request_id"
        case type
    }
}

private struct GuestExecRequest: Encodable {
    let protocolVersion: String
    let requestID: String
    let type: String
    let argv: [String]
    let cwd: String
    let env: [String: String]
    let timeoutSeconds: Int

    private enum CodingKeys: String, CodingKey {
        case protocolVersion = "protocol_version"
        case requestID = "request_id"
        case type
        case argv
        case cwd
        case env
        case timeoutSeconds = "timeout_sec"
    }
}

private struct GuestReadyResponse: Decodable {
    let protocolVersion: String
    let requestID: String
    let status: String
    let workspaceRoot: String?

    private enum CodingKeys: String, CodingKey {
        case protocolVersion = "protocol_version"
        case requestID = "request_id"
        case status
        case workspaceRoot = "workspace_root"
    }
}

private struct GuestExecResponse: Decodable {
    let protocolVersion: String
    let requestID: String
    let exitCode: Int
    let stdout: String
    let stderr: String

    private enum CodingKeys: String, CodingKey {
        case protocolVersion = "protocol_version"
        case requestID = "request_id"
        case exitCode = "exit_code"
        case stdout
        case stderr
    }
}

private struct GuestErrorResponse: Decodable {
    let protocolVersion: String
    let requestID: String
    let errorCode: String
    let message: String

    private enum CodingKeys: String, CodingKey {
        case protocolVersion = "protocol_version"
        case requestID = "request_id"
        case errorCode = "error_code"
        case message
    }
}

protocol GuestTransporting {
    func waitUntilGuestReady(vmID: String, timeoutSeconds: TimeInterval) throws
    func sendExecRequest(vmID: String, requestData: Data, timeoutSeconds: TimeInterval) throws -> Data
}

private final class UnimplementedGuestTransport: GuestTransporting {
    func waitUntilGuestReady(vmID: String, timeoutSeconds: TimeInterval) throws {
        throw GuestBridgeError.guestReadinessNotImplemented
    }

    func sendExecRequest(vmID: String, requestData: Data, timeoutSeconds: TimeInterval) throws -> Data {
        throw GuestBridgeError.guestExecNotImplemented
    }
}

final class VSockBridge: GuestBridging {
    private let transport: GuestTransporting
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()

    init(transport: GuestTransporting = UnimplementedGuestTransport()) {
        self.transport = transport
    }

    func waitUntilReady(vmID: String, timeoutSeconds: TimeInterval) throws {
        try transport.waitUntilGuestReady(vmID: vmID, timeoutSeconds: timeoutSeconds)
    }

    func exec(
        vmID: String,
        argv: [String],
        cwd: String,
        env: [String: String],
        timeoutSeconds: TimeInterval
    ) throws -> GuestExecResult {
        let requestID = UUID().uuidString
        let requestData = try encoder.encode(
            GuestExecRequest(
                protocolVersion: guestProtocolVersion,
                requestID: requestID,
                type: "exec",
                argv: argv,
                cwd: cwd,
                env: env,
                timeoutSeconds: Int(timeoutSeconds.rounded(.up))
            )
        )
        let responseData = try transport.sendExecRequest(
            vmID: vmID,
            requestData: requestData,
            timeoutSeconds: timeoutSeconds
        )
        if let error = try? decoder.decode(GuestErrorResponse.self, from: responseData) {
            throw GuestBridgeError.guestOperationFailed(code: error.errorCode, message: error.message)
        }
        let response = try decoder.decode(GuestExecResponse.self, from: responseData)
        try validateProtocol(responseProtocolVersion: response.protocolVersion, responseRequestID: response.requestID, requestID: requestID)
        return GuestExecResult(
            exitCode: response.exitCode,
            stdout: response.stdout,
            stderr: response.stderr
        )
    }

    private func validateProtocol(
        responseProtocolVersion: String,
        responseRequestID: String,
        requestID: String
    ) throws {
        guard responseProtocolVersion == guestProtocolVersion else {
            throw GuestBridgeError.guestProtocolError("protocol_mismatch")
        }
        guard responseRequestID == requestID else {
            throw GuestBridgeError.guestProtocolError("request_id_mismatch")
        }
    }
}

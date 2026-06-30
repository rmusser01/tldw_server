import Foundation

let guestProtocolVersion = "1"

protocol GuestReadinessBridging {
    func waitUntilReady(vmID: String, timeoutSeconds: TimeInterval) throws
    func guestInfo(vmID: String) -> GuestAgentInfo?
}

struct GuestExecResult {
    let exitCode: Int
    let stdout: String
    let stderr: String
    let details: [String: String]
}

protocol GuestBridging: GuestReadinessBridging {
    func exec(
        vmID: String,
        argv: [String],
        cwd: String,
        env: [String: String],
        timeoutSeconds: TimeInterval,
        maxOutputBytes: Int?
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
    let maxOutputBytes: Int?

    private enum CodingKeys: String, CodingKey {
        case protocolVersion = "protocol_version"
        case requestID = "request_id"
        case type
        case argv
        case cwd
        case env
        case timeoutSeconds = "timeout_sec"
        case maxOutputBytes = "max_output_bytes"
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
    let details: [String: String]

    private enum CodingKeys: String, CodingKey {
        case protocolVersion = "protocol_version"
        case requestID = "request_id"
        case exitCode = "exit_code"
        case stdout
        case stderr
        case details
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        protocolVersion = try container.decode(String.self, forKey: .protocolVersion)
        requestID = try container.decode(String.self, forKey: .requestID)
        exitCode = try container.decode(Int.self, forKey: .exitCode)
        stdout = try container.decodeIfPresent(String.self, forKey: .stdout) ?? ""
        stderr = try container.decodeIfPresent(String.self, forKey: .stderr) ?? ""
        let decodedDetails = (try? container.decodeIfPresent([String: StringOnlyDetailValue].self, forKey: .details)) ?? [:]
        details = decodedDetails.reduce(into: [String: String]()) { result, item in
            guard item.key.hasPrefix("guest_"), let value = item.value.value else {
                return
            }
            result[item.key] = value
        }
    }
}

private struct StringOnlyDetailValue: Decodable {
    let value: String?

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        value = try? container.decode(String.self)
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
    func guestInfo(vmID: String) -> GuestAgentInfo?
}

private final class UnimplementedGuestTransport: GuestTransporting {
    func waitUntilGuestReady(vmID: String, timeoutSeconds: TimeInterval) throws {
        throw GuestBridgeError.guestReadinessNotImplemented
    }

    func sendExecRequest(vmID: String, requestData: Data, timeoutSeconds: TimeInterval) throws -> Data {
        throw GuestBridgeError.guestExecNotImplemented
    }

    func guestInfo(vmID: String) -> GuestAgentInfo? {
        nil
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

    func guestInfo(vmID: String) -> GuestAgentInfo? {
        transport.guestInfo(vmID: vmID)
    }

    func exec(
        vmID: String,
        argv: [String],
        cwd: String,
        env: [String: String],
        timeoutSeconds: TimeInterval,
        maxOutputBytes: Int? = nil
    ) throws -> GuestExecResult {
        if let maxOutputBytes, maxOutputBytes < 0 {
            throw GuestBridgeError.guestProtocolError("invalid_max_output_bytes")
        }
        let requestID = UUID().uuidString
        let requestData = try encoder.encode(
            GuestExecRequest(
                protocolVersion: guestProtocolVersion,
                requestID: requestID,
                type: "exec",
                argv: argv,
                cwd: cwd,
                env: env,
                timeoutSeconds: Int(timeoutSeconds.rounded(.up)),
                maxOutputBytes: maxOutputBytes
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
            stderr: response.stderr,
            details: response.details
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

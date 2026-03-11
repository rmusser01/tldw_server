import Foundation

struct HelperResponse: Encodable {
    let protocolVersion: String
    let helperVersion: String
    let status: String?
    let available: Bool?
    let executionMode: String?
    let transport: String?
    let reasons: [String]
    let details: [String: String]
    let errorCode: String?
    let message: String?

    private enum CodingKeys: String, CodingKey {
        case protocolVersion = "protocol_version"
        case helperVersion = "helper_version"
        case status
        case available
        case executionMode = "execution_mode"
        case transport
        case reasons
        case details
        case errorCode = "error_code"
        case message
    }
}

struct TemplateValidationResponse {
    let protocolVersion: String
    let helperVersion: String
    let templateID: String
    let source: String
    let ready: Bool
    let reasons: [String]
}

struct VMRecord {
    let vmID: String
    let state: String
    let healthy: Bool
}

struct HelperVMResponse {
    let vmID: String
    let state: String
    let details: [String: String]
}

struct HelperVMStatusResponse {
    let protocolVersion: String
    let helperVersion: String
    let vmID: String
    let state: String
    let healthy: Bool
    let details: [String: String]
}

struct HelperVMListResponse {
    let protocolVersion: String
    let helperVersion: String
    let vms: [HelperVMStatusResponse]
}

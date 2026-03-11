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

import Foundation

struct HelperRequest: Decodable {
    let operation: String
    let protocolVersion: String
    let request: [String: String]

    private enum CodingKeys: String, CodingKey {
        case operation
        case protocolVersion = "protocol_version"
        case request
    }
}

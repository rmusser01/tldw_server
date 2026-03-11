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

struct TemplateValidationResponse: Encodable {
    let protocolVersion: String
    let helperVersion: String
    let templateID: String
    let source: String
    let ready: Bool
    let bootMode: String?
    let validationStrength: String?
    let reasons: [String]

    private enum CodingKeys: String, CodingKey {
        case protocolVersion = "protocol_version"
        case helperVersion = "helper_version"
        case templateID = "template_id"
        case source
        case ready
        case bootMode = "boot_mode"
        case validationStrength = "validation_strength"
        case reasons
    }
}

struct VMRecord {
    let vmID: String
    let state: String
    let healthy: Bool
}

struct HelperVMResponse: Encodable {
    let protocolVersion: String
    let helperVersion: String
    let vmID: String
    let state: String
    let details: [String: String]

    private enum CodingKeys: String, CodingKey {
        case protocolVersion = "protocol_version"
        case helperVersion = "helper_version"
        case vmID = "vm_id"
        case state
        case details
    }
}

struct HelperVMStatusResponse: Encodable {
    let protocolVersion: String
    let helperVersion: String
    let vmID: String
    let state: String
    let healthy: Bool
    let details: [String: String]

    private enum CodingKeys: String, CodingKey {
        case protocolVersion = "protocol_version"
        case helperVersion = "helper_version"
        case vmID = "vm_id"
        case state
        case healthy
        case details
    }
}

struct HelperVMListResponse: Encodable {
    let protocolVersion: String
    let helperVersion: String
    let vms: [HelperVMStatusResponse]

    private enum CodingKeys: String, CodingKey {
        case protocolVersion = "protocol_version"
        case helperVersion = "helper_version"
        case vms
    }
}

struct HelperExecResponse: Encodable {
    let protocolVersion: String
    let helperVersion: String
    let exitCode: Int
    let stdout: String
    let stderr: String
    let details: [String: String]

    private enum CodingKeys: String, CodingKey {
        case protocolVersion = "protocol_version"
        case helperVersion = "helper_version"
        case exitCode = "exit_code"
        case stdout
        case stderr
        case details
    }
}

struct HelperTerminateResponse: Encodable {
    let protocolVersion: String
    let helperVersion: String
    let terminated: Bool

    private enum CodingKeys: String, CodingKey {
        case protocolVersion = "protocol_version"
        case helperVersion = "helper_version"
        case terminated
    }
}

struct HelperErrorResponse: Encodable {
    let protocolVersion: String
    let helperVersion: String
    let errorCode: String
    let message: String

    private enum CodingKeys: String, CodingKey {
        case protocolVersion = "protocol_version"
        case helperVersion = "helper_version"
        case errorCode = "error_code"
        case message
    }
}

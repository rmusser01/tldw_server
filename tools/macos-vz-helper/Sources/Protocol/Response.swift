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

struct VMOwnershipMetadata: Codable, Equatable {
    let owner: String
    let runtime: String
    let runID: String
    let sessionID: String
    let sessionMode: Bool
    let templateID: String
    let templatePath: String
    let runManifestPath: String
    let planningSource: String
    let workspacePath: String
    let createdAt: String
    let networkPolicy: String

    static let unknown = VMOwnershipMetadata(
        owner: "unknown",
        runtime: "",
        runID: "",
        sessionID: "",
        sessionMode: false,
        templateID: "",
        templatePath: "",
        runManifestPath: "",
        planningSource: "",
        workspacePath: "",
        createdAt: "",
        networkPolicy: ""
    )

    private enum CodingKeys: String, CodingKey {
        case owner
        case runtime
        case runID = "run_id"
        case sessionID = "session_id"
        case sessionMode = "session_mode"
        case templateID = "template_id"
        case templatePath = "template_path"
        case runManifestPath = "run_manifest_path"
        case planningSource = "planning_source"
        case workspacePath = "workspace_path"
        case createdAt = "created_at"
        case networkPolicy = "network_policy"
    }

    init(
        owner: String,
        runtime: String,
        runID: String,
        sessionID: String,
        sessionMode: Bool,
        templateID: String,
        templatePath: String,
        runManifestPath: String,
        planningSource: String,
        workspacePath: String,
        createdAt: String,
        networkPolicy: String = ""
    ) {
        self.owner = owner
        self.runtime = runtime
        self.runID = runID
        self.sessionID = sessionID
        self.sessionMode = sessionMode
        self.templateID = templateID
        self.templatePath = templatePath
        self.runManifestPath = runManifestPath
        self.planningSource = planningSource
        self.workspacePath = workspacePath
        self.createdAt = createdAt
        self.networkPolicy = networkPolicy
    }
}

struct VMRecord {
    let vmID: String
    let state: String
    let healthy: Bool
    let metadata: VMOwnershipMetadata
    let guestInfo: GuestAgentInfo?
    let resourceSnapshot: VMResourceSnapshot?

    init(
        vmID: String,
        state: String,
        healthy: Bool,
        metadata: VMOwnershipMetadata = .unknown,
        guestInfo: GuestAgentInfo? = nil,
        resourceSnapshot: VMResourceSnapshot? = nil
    ) {
        self.vmID = vmID
        self.state = state
        self.healthy = healthy
        self.metadata = metadata
        self.guestInfo = guestInfo
        self.resourceSnapshot = resourceSnapshot
    }
}

struct VMResourceSnapshot: Equatable {
    let cpuCount: Int
    let memorySizeBytes: UInt64

    var memorySizeMB: UInt64 {
        memorySizeBytes / 1_048_576
    }
}

struct HelperVMResponse: Encodable {
    let protocolVersion: String
    let helperVersion: String
    let vmID: String
    let state: String
    let metadata: VMOwnershipMetadata
    let details: [String: String]

    private enum CodingKeys: String, CodingKey {
        case protocolVersion = "protocol_version"
        case helperVersion = "helper_version"
        case vmID = "vm_id"
        case state
        case metadata
        case details
    }
}

struct HelperVMStatusResponse: Encodable {
    let protocolVersion: String
    let helperVersion: String
    let vmID: String
    let state: String
    let healthy: Bool
    let metadata: VMOwnershipMetadata
    let details: [String: String]

    private enum CodingKeys: String, CodingKey {
        case protocolVersion = "protocol_version"
        case helperVersion = "helper_version"
        case vmID = "vm_id"
        case state
        case healthy
        case metadata
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

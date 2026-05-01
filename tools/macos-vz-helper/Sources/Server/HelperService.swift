import Foundation

struct HostFacts {
    let isMacOS: Bool
    let isAppleSilicon: Bool
}

enum HelperServiceError: Error {
    case unsupportedNetworkPolicy(String)
}

final class HelperService {
    private let protocolVersion = "1"
    private let helperVersion = "0.1.0"
    private let hostFacts: HostFacts
    private let templateValidator: TemplateValidator
    private let registry: VMRegistry
    private let vmManager: VZLinuxVMManager
    private let metadataDateFormatter = ISO8601DateFormatter()

    init(
        hostFacts: HostFacts = HostFacts(isMacOS: true, isAppleSilicon: true),
        templateValidator: TemplateValidator = TemplateValidator(),
        registry: VMRegistry = VMRegistry(),
        vmManager: VZLinuxVMManager? = nil
    ) {
        self.hostFacts = hostFacts
        self.templateValidator = templateValidator
        self.registry = registry
        self.vmManager = vmManager ?? VZLinuxVMManager(registry: registry)
    }

    func ping() -> HelperResponse {
        HelperResponse(
            protocolVersion: protocolVersion,
            helperVersion: helperVersion,
            status: "ok",
            available: nil,
            executionMode: nil,
            transport: nil,
            reasons: [],
            details: ["transport": "unix"],
            errorCode: nil,
            message: nil
        )
    }

    func validateHost(runtime: String, networkPolicy: String) -> HelperResponse {
        var reasons: [String] = []
        let normalizedNetworkPolicy = normalizeNetworkPolicy(networkPolicy)
        if runtime != "vz_linux" {
            reasons.append("runtime_unsupported")
        }
        if !hostFacts.isMacOS {
            reasons.append("macos_host_required")
        }
        if !hostFacts.isAppleSilicon {
            reasons.append("apple_silicon_required")
        }
        if normalizedNetworkPolicy != "deny_all" {
            reasons.append(networkPolicyErrorCode(normalizedNetworkPolicy))
        }

        let available = reasons.isEmpty
        return HelperResponse(
            protocolVersion: protocolVersion,
            helperVersion: helperVersion,
            status: nil,
            available: available,
            executionMode: available ? "real" : "none",
            transport: available ? "vsock" : nil,
            reasons: reasons,
            details: ["runtime": runtime, "network_policy": normalizedNetworkPolicy],
            errorCode: nil,
            message: nil
        )
    }

    func validateTemplate(runtime: String, templatePath: String) -> TemplateValidationResponse {
        templateValidator.validate(runtime: runtime, templatePath: templatePath)
    }

    func createVM(
        vmID: String,
        templatePath: String,
        workspacePath: String,
        readinessTimeoutSeconds: TimeInterval,
        metadata: VMOwnershipMetadata = .unknown,
        networkPolicy: String = "deny_all"
    ) throws -> HelperVMResponse {
        let normalizedNetworkPolicy = try requireSupportedNetworkPolicy(networkPolicy)
        let normalizedMetadata = normalizeMetadata(
            metadata,
            templatePath: templatePath,
            workspacePath: workspacePath,
            networkPolicy: normalizedNetworkPolicy
        )
        let record = try vmManager.createVM(
            vmID: vmID,
            templatePath: templatePath,
            workspacePath: workspacePath,
            readinessTimeoutSeconds: readinessTimeoutSeconds,
            metadata: normalizedMetadata
        )
        return HelperVMResponse(
            protocolVersion: protocolVersion,
            helperVersion: helperVersion,
            vmID: record.vmID,
            state: record.state,
            metadata: record.metadata,
            details: vmDetails(for: record)
        )
    }

    func getVMStatus(vmID: String) -> HelperVMStatusResponse? {
        guard let record = registry.status(vmID: vmID) else {
            return nil
        }
        return HelperVMStatusResponse(
            protocolVersion: protocolVersion,
            helperVersion: helperVersion,
            vmID: record.vmID,
            state: record.state,
            healthy: record.healthy,
            metadata: record.metadata,
            details: vmDetails(for: record)
        )
    }

    func listVMs() -> HelperVMListResponse {
        let vms = registry.list().map { record in
            HelperVMStatusResponse(
                protocolVersion: protocolVersion,
                helperVersion: helperVersion,
                vmID: record.vmID,
                state: record.state,
                healthy: record.healthy,
                metadata: record.metadata,
                details: vmDetails(for: record)
            )
        }
        return HelperVMListResponse(
            protocolVersion: protocolVersion,
            helperVersion: helperVersion,
            vms: vms
        )
    }

    func terminateVM(vmID: String) throws -> Bool {
        try vmManager.terminateVM(vmID: vmID)
    }

    func execGuest(
        vmID: String,
        argv: [String],
        cwd: String,
        env: [String: String],
        timeoutSeconds: TimeInterval
    ) throws -> HelperExecResponse {
        let result = try vmManager.execGuest(
            vmID: vmID,
            argv: argv,
            cwd: cwd,
            env: env,
            timeoutSeconds: timeoutSeconds
        )
        return HelperExecResponse(
            protocolVersion: protocolVersion,
            helperVersion: helperVersion,
            exitCode: result.exitCode,
            stdout: result.stdout,
            stderr: result.stderr,
            details: ["transport": "vsock", "vm_id": vmID]
        )
    }

    private func normalizeMetadata(
        _ metadata: VMOwnershipMetadata,
        templatePath: String,
        workspacePath: String,
        networkPolicy: String
    ) -> VMOwnershipMetadata {
        VMOwnershipMetadata(
            owner: metadata.owner.isEmpty ? "unknown" : metadata.owner,
            runtime: metadata.runtime.isEmpty ? "vz_linux" : metadata.runtime,
            runID: metadata.runID,
            sessionID: metadata.sessionID,
            sessionMode: metadata.sessionMode,
            templateID: metadata.templateID,
            templatePath: metadata.templatePath.isEmpty ? templatePath : metadata.templatePath,
            runManifestPath: metadata.runManifestPath,
            planningSource: metadata.planningSource,
            workspacePath: metadata.workspacePath.isEmpty ? workspacePath : metadata.workspacePath,
            createdAt: metadata.createdAt.isEmpty ? metadataDateFormatter.string(from: Date()) : metadata.createdAt,
            networkPolicy: networkPolicy
        )
    }

    private func requireSupportedNetworkPolicy(_ networkPolicy: String) throws -> String {
        let normalized = normalizeNetworkPolicy(networkPolicy)
        guard normalized == "deny_all" else {
            throw HelperServiceError.unsupportedNetworkPolicy(normalized)
        }
        return normalized
    }

    private func normalizeNetworkPolicy(_ networkPolicy: String) -> String {
        let normalized = networkPolicy.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return normalized.isEmpty ? "deny_all" : normalized
    }

    private func networkPolicyErrorCode(_ networkPolicy: String) -> String {
        networkPolicy == "allowlist" ? "strict_allowlist_not_supported" : "unsupported_network_policy"
    }

    private func vmDetails(for record: VMRecord) -> [String: String] {
        [
            "transport": "vsock",
            "network_policy": record.metadata.networkPolicy.isEmpty ? "deny_all" : record.metadata.networkPolicy,
        ]
    }
}

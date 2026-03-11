import Foundation

struct HostFacts {
    let isMacOS: Bool
    let isAppleSilicon: Bool
}

final class HelperService {
    private let protocolVersion = "1"
    private let helperVersion = "0.1.0"
    private let hostFacts: HostFacts
    private let templateValidator: TemplateValidator
    private let registry: VMRegistry
    private let vmManager: VZLinuxVMManager

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
        if runtime != "vz_linux" {
            reasons.append("runtime_unsupported")
        }
        if !hostFacts.isMacOS {
            reasons.append("macos_host_required")
        }
        if !hostFacts.isAppleSilicon {
            reasons.append("apple_silicon_required")
        }
        if networkPolicy != "deny_all" {
            reasons.append("strict_allowlist_not_supported")
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
            details: ["runtime": runtime],
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
        readinessTimeoutSeconds: TimeInterval
    ) throws -> HelperVMResponse {
        let record = try vmManager.createVM(
            vmID: vmID,
            templatePath: templatePath,
            workspacePath: workspacePath,
            readinessTimeoutSeconds: readinessTimeoutSeconds
        )
        return HelperVMResponse(
            vmID: record.vmID,
            state: record.state,
            details: ["transport": "vsock"]
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
            details: ["transport": "vsock"]
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
                details: ["transport": "vsock"]
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
}

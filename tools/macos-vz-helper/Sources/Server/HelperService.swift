import Foundation

struct HostFacts {
    let isMacOS: Bool
    let isAppleSilicon: Bool
}

final class HelperService {
    private let protocolVersion = "1"
    private let helperVersion = "0.1.0"
    private let hostFacts: HostFacts

    init(hostFacts: HostFacts = HostFacts(isMacOS: true, isAppleSilicon: true)) {
        self.hostFacts = hostFacts
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
}

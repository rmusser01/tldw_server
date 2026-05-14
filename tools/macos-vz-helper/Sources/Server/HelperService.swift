import Darwin
import Foundation

struct HostFacts {
    let isMacOS: Bool
    let isAppleSilicon: Bool
}

enum HelperServiceError: Error {
    case unsupportedRuntime(String)
    case invalidCreateVMRequest(String)
    case invalidCreateVMTimeout(String)
    case unsupportedNetworkPolicy(String)
    case invalidExecArgv(String)
    case invalidExecCwd(String)
    case invalidExecEnv(String)
    case invalidExecTimeout(String)
    case invalidExecOutputLimit(String)
}

final class HelperService {
    private static let execWorkspaceRoot = "/workspace"
    private static let execWorkspaceRootPrefix = execWorkspaceRoot + "/"
    private static let maxExecArgvCount = 128
    private static let maxExecArgvBytes = 32 * 1024
    private static let maxExecEnvCount = 128
    private static let maxExecEnvBytes = 32 * 1024
    private static let maxExecTimeoutSeconds: TimeInterval = 3_600
    private static let maxExecOutputBytes = 256 * 1024 * 1024
    private static let maxCreateVMIDBytes = 128
    private static let maxCreateVMTextBytes = 1024
    private static let maxCreateVMPathBytes = 4096
    private static let maxCreateVMTimeoutSeconds: TimeInterval = 3_600

    private let protocolVersion = "1"
    private let helperVersion = "0.1.0"
    private let helperInstanceID: String
    private let helperStartedAt: String
    private let hostFacts: HostFacts
    private let templateValidator: TemplateValidator
    private let registry: VMRegistry
    private let vmManager: VZLinuxVMManager
    private let metadataDateFormatter = ISO8601DateFormatter()

    init(
        hostFacts: HostFacts = HostFacts(isMacOS: true, isAppleSilicon: true),
        templateValidator: TemplateValidator = TemplateValidator(),
        registry: VMRegistry = VMRegistry(),
        vmManager: VZLinuxVMManager? = nil,
        helperInstanceID: String = UUID().uuidString,
        helperStartedAt: String = ISO8601DateFormatter().string(from: Date())
    ) {
        self.helperInstanceID = helperInstanceID
        self.helperStartedAt = helperStartedAt
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
            details: withHelperGeneration(["transport": "unix"]),
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
        try validateCreateVMContract(
            vmID: vmID,
            templatePath: templatePath,
            workspacePath: workspacePath,
            readinessTimeoutSeconds: readinessTimeoutSeconds,
            metadata: metadata
        )
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
        timeoutSeconds: TimeInterval,
        maxOutputBytes: Int? = nil
    ) throws -> HelperExecResponse {
        try validateExecGuestContract(
            argv: argv,
            cwd: cwd,
            env: env,
            timeoutSeconds: timeoutSeconds,
            maxOutputBytes: maxOutputBytes
        )
        let result = try vmManager.execGuest(
            vmID: vmID,
            argv: argv,
            cwd: cwd,
            env: env,
            timeoutSeconds: timeoutSeconds,
            maxOutputBytes: maxOutputBytes
        )
        let cappedOutput = capExecOutput(
            stdout: result.stdout,
            stderr: result.stderr,
            maxOutputBytes: maxOutputBytes
        )
        var details = ["transport": "vsock", "vm_id": vmID]
        for (key, value) in result.details where key.hasPrefix("guest_") {
            details[key] = value
        }
        for (key, value) in cappedOutput.details {
            details[key] = value
        }
        return HelperExecResponse(
            protocolVersion: protocolVersion,
            helperVersion: helperVersion,
            exitCode: result.exitCode,
            stdout: cappedOutput.stdout,
            stderr: cappedOutput.stderr,
            details: details
        )
    }

    private func normalizeMetadata(
        _ metadata: VMOwnershipMetadata,
        templatePath: String,
        workspacePath: String,
        networkPolicy: String
    ) -> VMOwnershipMetadata {
        let normalizedRuntime = normalizeRuntime(metadata.runtime)
        return VMOwnershipMetadata(
            owner: metadata.owner.isEmpty ? "unknown" : metadata.owner,
            runtime: normalizedRuntime.isEmpty ? "vz_linux" : normalizedRuntime,
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

    private func normalizeRuntime(_ runtime: String) -> String {
        runtime.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    }

    private func networkPolicyErrorCode(_ networkPolicy: String) -> String {
        networkPolicy == "allowlist" ? "strict_allowlist_not_supported" : "unsupported_network_policy"
    }

    private func validateCreateVMContract(
        vmID: String,
        templatePath: String,
        workspacePath: String,
        readinessTimeoutSeconds: TimeInterval,
        metadata: VMOwnershipMetadata
    ) throws {
        try validateCreateVMID(vmID)
        try validateCreateVMPath(templatePath, reason: "template_path_invalid", required: true)
        try validateCreateVMPath(workspacePath, reason: "workspace_path_invalid", required: true)
        try validateCreateVMTimeout(readinessTimeoutSeconds)

        let metadataRuntime = normalizeRuntime(metadata.runtime)
        if !metadataRuntime.isEmpty, metadataRuntime != "vz_linux" {
            throw HelperServiceError.unsupportedRuntime(metadataRuntime)
        }

        for (value, reason) in [
            (metadata.owner, "owner_invalid"),
            (metadata.runID, "run_id_invalid"),
            (metadata.sessionID, "session_id_invalid"),
            (metadata.templateID, "template_id_invalid"),
            (metadata.planningSource, "planning_source_invalid"),
            (metadata.createdAt, "created_at_invalid"),
        ] {
            try validateCreateVMText(value, reason: reason)
        }
        try validateCreateVMPath(metadata.templatePath, reason: "template_path_invalid", required: false)
        try validateCreateVMPath(metadata.workspacePath, reason: "workspace_path_invalid", required: false)
        try validateCreateVMPath(metadata.runManifestPath, reason: "run_manifest_path_invalid", required: false)
    }

    private func validateCreateVMID(_ vmID: String) throws {
        guard !vmID.isEmpty,
              vmID.trimmingCharacters(in: .whitespacesAndNewlines) == vmID,
              !containsNUL(vmID),
              vmID.utf8.count <= Self.maxCreateVMIDBytes else {
            throw HelperServiceError.invalidCreateVMRequest("vm_id_invalid")
        }
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "._-"))
        guard vmID.unicodeScalars.allSatisfy({ allowed.contains($0) }) else {
            throw HelperServiceError.invalidCreateVMRequest("vm_id_invalid")
        }
    }

    private func validateCreateVMText(_ value: String, reason: String) throws {
        guard !containsNUL(value),
              !containsControlCharacter(value),
              value.utf8.count <= Self.maxCreateVMTextBytes else {
            throw HelperServiceError.invalidCreateVMRequest(reason)
        }
    }

    private func validateCreateVMPath(_ value: String, reason: String, required: Bool) throws {
        if value.isEmpty {
            if required {
                throw HelperServiceError.invalidCreateVMRequest(reason)
            }
            return
        }
        guard !containsNUL(value),
              value.utf8.count <= Self.maxCreateVMPathBytes,
              value.hasPrefix("/") else {
            throw HelperServiceError.invalidCreateVMRequest(reason)
        }

        var pathStat = stat()
        let result = lstat(value, &pathStat)
        if result == 0 {
            let type = pathStat.st_mode & S_IFMT
            guard type != S_IFLNK else {
                throw HelperServiceError.invalidCreateVMRequest(reason)
            }
            return
        }
        guard errno == ENOENT else {
            throw HelperServiceError.invalidCreateVMRequest(reason)
        }
    }

    private func validateCreateVMTimeout(_ timeoutSeconds: TimeInterval) throws {
        guard timeoutSeconds.isFinite,
              timeoutSeconds > 0,
              timeoutSeconds <= Self.maxCreateVMTimeoutSeconds else {
            throw HelperServiceError.invalidCreateVMTimeout("timeout_out_of_range")
        }
    }

    private func vmDetails(for record: VMRecord) -> [String: String] {
        var details = [
            "transport": "vsock",
            "network_policy": record.metadata.networkPolicy.isEmpty ? "deny_all" : record.metadata.networkPolicy,
        ]
        if let guestInfo = record.guestInfo {
            details["guest_capabilities_known"] = guestInfo.capabilitiesKnown ? "true" : "false"
            if let guestVersion = guestInfo.guestVersion {
                details["guest_version"] = guestVersion
            }
            if let workspaceRoot = guestInfo.workspaceRoot {
                details["guest_workspace_root"] = workspaceRoot
            }
            if guestInfo.capabilitiesKnown {
                details["guest_capabilities"] = guestInfo.capabilities.joined(separator: ",")
            }
        }
        return withHelperGeneration(details)
    }

    private func helperGenerationDetails() -> [String: String] {
        [
            "helper_instance_id": helperInstanceID,
            "helper_started_at": helperStartedAt,
        ]
    }

    private func withHelperGeneration(_ details: [String: String]) -> [String: String] {
        var merged = details
        for (key, value) in helperGenerationDetails() {
            merged[key] = value
        }
        return merged
    }

    private func validateExecGuestContract(
        argv: [String],
        cwd: String,
        env: [String: String],
        timeoutSeconds: TimeInterval,
        maxOutputBytes: Int?
    ) throws {
        try validateExecArgv(argv)
        try validateExecCwd(cwd)
        try validateExecEnv(env)
        try validateExecTimeout(timeoutSeconds)
        try validateExecOutputLimit(maxOutputBytes)
    }

    private func validateExecArgv(_ argv: [String]) throws {
        guard !argv.isEmpty else {
            throw HelperServiceError.invalidExecArgv("argv_required")
        }
        guard argv.count <= Self.maxExecArgvCount else {
            throw HelperServiceError.invalidExecArgv("argv_too_large")
        }
        var totalBytes = 0
        for argument in argv {
            guard !argument.isEmpty else {
                throw HelperServiceError.invalidExecArgv("argv_empty_argument")
            }
            guard !containsNUL(argument) else {
                throw HelperServiceError.invalidExecArgv("argv_invalid")
            }
            totalBytes += argument.utf8.count
            guard totalBytes <= Self.maxExecArgvBytes else {
                throw HelperServiceError.invalidExecArgv("argv_too_large")
            }
        }
    }

    private func validateExecCwd(_ cwd: String) throws {
        guard !cwd.isEmpty,
              !containsNUL(cwd),
              cwd == Self.execWorkspaceRoot || cwd.hasPrefix(Self.execWorkspaceRootPrefix) else {
            throw HelperServiceError.invalidExecCwd("cwd_outside_workspace")
        }
        let components = cwd.split(separator: "/", omittingEmptySubsequences: true)
        guard !components.contains("..") else {
            throw HelperServiceError.invalidExecCwd("cwd_outside_workspace")
        }
    }

    private func validateExecEnv(_ env: [String: String]) throws {
        guard env.count <= Self.maxExecEnvCount else {
            throw HelperServiceError.invalidExecEnv("env_too_large")
        }
        var totalBytes = 0
        for (key, value) in env {
            guard !key.isEmpty,
                  !key.contains("="),
                  !containsNUL(key),
                  !containsControlCharacter(key) else {
                throw HelperServiceError.invalidExecEnv("env_key_invalid")
            }
            guard !containsNUL(value) else {
                throw HelperServiceError.invalidExecEnv("env_value_invalid")
            }
            totalBytes += key.utf8.count + value.utf8.count
            guard totalBytes <= Self.maxExecEnvBytes else {
                throw HelperServiceError.invalidExecEnv("env_too_large")
            }
        }
    }

    private func validateExecTimeout(_ timeoutSeconds: TimeInterval) throws {
        guard timeoutSeconds.isFinite,
              timeoutSeconds > 0,
              timeoutSeconds <= Self.maxExecTimeoutSeconds else {
            throw HelperServiceError.invalidExecTimeout("timeout_out_of_range")
        }
    }

    private func validateExecOutputLimit(_ maxOutputBytes: Int?) throws {
        guard let maxOutputBytes else {
            return
        }
        guard maxOutputBytes > 0, maxOutputBytes <= Self.maxExecOutputBytes else {
            throw HelperServiceError.invalidExecOutputLimit("output_limit_out_of_range")
        }
    }

    private func capExecOutput(
        stdout: String,
        stderr: String,
        maxOutputBytes: Int?
    ) -> (stdout: String, stderr: String, details: [String: String]) {
        guard let cap = maxOutputBytes else {
            return (stdout, stderr, [:])
        }

        let stdoutOriginal = stdout.utf8.count
        let stderrOriginal = stderr.utf8.count
        if stdoutOriginal + stderrOriginal <= cap {
            return (
                stdout,
                stderr,
                outputLimitDetails(
                    cap: cap,
                    stdoutOriginal: stdoutOriginal,
                    stderrOriginal: stderrOriginal,
                    stdoutReturned: stdoutOriginal,
                    stderrReturned: stderrOriginal
                )
            )
        }

        let budgets = outputBudgets(
            stdoutBytes: stdoutOriginal,
            stderrBytes: stderrOriginal,
            cap: cap
        )
        let returned = cappedUTF8Output(
            stdout: stdout,
            stderr: stderr,
            stdoutBytes: stdoutOriginal,
            stderrBytes: stderrOriginal,
            stdoutBudget: budgets.stdout,
            stderrBudget: budgets.stderr,
            cap: cap
        )
        let returnedStdout = returned.stdout
        let returnedStderr = returned.stderr
        let stdoutReturned = returned.stdoutBytes
        let stderrReturned = returned.stderrBytes
        return (
            returnedStdout,
            returnedStderr,
            outputLimitDetails(
                cap: cap,
                stdoutOriginal: stdoutOriginal,
                stderrOriginal: stderrOriginal,
                stdoutReturned: stdoutReturned,
                stderrReturned: stderrReturned
            )
        )
    }

    private func outputBudgets(
        stdoutBytes: Int,
        stderrBytes: Int,
        cap: Int
    ) -> (stdout: Int, stderr: Int) {
        if stdoutBytes > 0, stderrBytes > 0, cap >= 2 {
            var stdoutBudget = min(stdoutBytes, max(1, cap / 2))
            var stderrBudget = min(stderrBytes, max(1, cap - stdoutBudget))
            var unused = cap - stdoutBudget - stderrBudget
            if unused > 0, stdoutBytes > stdoutBudget {
                let extra = min(unused, stdoutBytes - stdoutBudget)
                stdoutBudget += extra
                unused -= extra
            }
            if unused > 0, stderrBytes > stderrBudget {
                let extra = min(unused, stderrBytes - stderrBudget)
                stderrBudget += extra
            }
            return (stdoutBudget, stderrBudget)
        }
        if stdoutBytes > 0 {
            return (min(stdoutBytes, cap), 0)
        }
        return (0, min(stderrBytes, cap))
    }

    private func utf8Prefix(_ value: String, maxBytes: Int) -> String {
        guard maxBytes > 0 else {
            return ""
        }
        guard value.utf8.count > maxBytes else {
            return value
        }
        var end = value.utf8.index(value.utf8.startIndex, offsetBy: maxBytes)
        while end > value.utf8.startIndex {
            if let stringEnd = String.Index(end, within: value) {
                return String(value[..<stringEnd])
            }
            end = value.utf8.index(before: end)
        }
        return ""
    }

    private func cappedUTF8Output(
        stdout: String,
        stderr: String,
        stdoutBytes: Int,
        stderrBytes: Int,
        stdoutBudget: Int,
        stderrBudget: Int,
        cap: Int
    ) -> (stdout: String, stderr: String, stdoutBytes: Int, stderrBytes: Int) {
        var candidates: [(stdout: Int, stderr: Int)] = [
            (stdoutBudget, stderrBudget),
            (min(stdoutBytes, cap), 0),
            (0, min(stderrBytes, cap)),
        ]

        let initialStdout = utf8Prefix(stdout, maxBytes: stdoutBudget)
        let initialStderr = utf8Prefix(stderr, maxBytes: stderrBudget)
        let initialUsed = initialStdout.utf8.count + initialStderr.utf8.count
        let unused = cap - initialUsed
        if unused > 0 {
            candidates.append((min(stdoutBytes, stdoutBudget + unused), stderrBudget))
            candidates.append((stdoutBudget, min(stderrBytes, stderrBudget + unused)))
        }

        var best = (stdout: "", stderr: "", stdoutBytes: 0, stderrBytes: 0)
        for candidate in candidates {
            let candidateStdout = utf8Prefix(stdout, maxBytes: candidate.stdout)
            let candidateStderr = utf8Prefix(stderr, maxBytes: candidate.stderr)
            let candidateStdoutBytes = candidateStdout.utf8.count
            let candidateStderrBytes = candidateStderr.utf8.count
            let used = candidateStdoutBytes + candidateStderrBytes
            guard used <= cap else {
                continue
            }
            if isBetterOutputCandidate(
                used: used,
                stderrBytes: candidateStderrBytes,
                stdoutBytes: candidateStdoutBytes,
                best: best
            ) {
                best = (
                    stdout: candidateStdout,
                    stderr: candidateStderr,
                    stdoutBytes: candidateStdoutBytes,
                    stderrBytes: candidateStderrBytes
                )
            }
        }
        return best
    }

    private func isBetterOutputCandidate(
        used: Int,
        stderrBytes: Int,
        stdoutBytes: Int,
        best: (stdout: String, stderr: String, stdoutBytes: Int, stderrBytes: Int)
    ) -> Bool {
        let bestUsed = best.stdoutBytes + best.stderrBytes
        if used != bestUsed {
            return used > bestUsed
        }
        let bothStreams = stdoutBytes > 0 && stderrBytes > 0
        let bestBothStreams = best.stdoutBytes > 0 && best.stderrBytes > 0
        if bothStreams != bestBothStreams {
            return bothStreams
        }
        if stderrBytes != best.stderrBytes {
            return stderrBytes > best.stderrBytes
        }
        return stdoutBytes > best.stdoutBytes
    }

    private func outputLimitDetails(
        cap: Int,
        stdoutOriginal: Int,
        stderrOriginal: Int,
        stdoutReturned: Int,
        stderrReturned: Int
    ) -> [String: String] {
        [
            "output_limit_bytes": "\(cap)",
            "stdout_bytes_original": "\(stdoutOriginal)",
            "stderr_bytes_original": "\(stderrOriginal)",
            "stdout_bytes_returned": "\(stdoutReturned)",
            "stderr_bytes_returned": "\(stderrReturned)",
            "stdout_truncated": stdoutReturned < stdoutOriginal ? "true" : "false",
            "stderr_truncated": stderrReturned < stderrOriginal ? "true" : "false",
        ]
    }

    private func containsNUL(_ value: String) -> Bool {
        value.unicodeScalars.contains { $0.value == 0 }
    }

    private func containsControlCharacter(_ value: String) -> Bool {
        value.unicodeScalars.contains { scalar in
            scalar.value < 0x20 || scalar.value == 0x7F
        }
    }
}

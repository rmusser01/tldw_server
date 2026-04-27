import Foundation

final class TemplateValidator {
    private let protocolVersion = "1"
    private let helperVersion = "0.1.0"
    private let bundleResolver: BundleTemplateResolver
    private let rawDiskResolver: RawDiskTemplateResolver

    init(
        bundleResolver: BundleTemplateResolver = BundleTemplateResolver(),
        rawDiskResolver: RawDiskTemplateResolver = RawDiskTemplateResolver()
    ) {
        self.bundleResolver = bundleResolver
        self.rawDiskResolver = rawDiskResolver
    }

    func validate(runtime: String, templatePath: String) -> TemplateValidationResponse {
        let trimmedPath = templatePath.trimmingCharacters(in: .whitespacesAndNewlines)
        let templateName = URL(fileURLWithPath: trimmedPath).lastPathComponent
        let templateID = "\(runtime):\(templateName)"

        do {
            let spec = try resolve(runtime: runtime, templatePath: trimmedPath)
            return TemplateValidationResponse(
                protocolVersion: protocolVersion,
                helperVersion: helperVersion,
                templateID: templateID,
                source: trimmedPath,
                ready: true,
                bootMode: spec.bootMode.rawValue,
                validationStrength: spec.validationStrength.rawValue,
                reasons: []
            )
        } catch let error as TemplateResolutionError {
            return TemplateValidationResponse(
                protocolVersion: protocolVersion,
                helperVersion: helperVersion,
                templateID: templateID,
                source: trimmedPath,
                ready: false,
                bootMode: nil,
                validationStrength: nil,
                reasons: [error.reason(for: runtime)]
            )
        } catch {
            return TemplateValidationResponse(
                protocolVersion: protocolVersion,
                helperVersion: helperVersion,
                templateID: templateID,
                source: trimmedPath,
                ready: false,
                bootMode: nil,
                validationStrength: nil,
                reasons: [missingReason(for: runtime)]
            )
        }
    }

    func resolve(runtime: String, templatePath: String) throws -> TemplateBootSpec {
        let trimmedPath = templatePath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedPath.isEmpty else {
            throw TemplateResolutionError.templateUnconfigured
        }

        var isDirectory = ObjCBool(false)
        guard FileManager.default.fileExists(atPath: trimmedPath, isDirectory: &isDirectory) else {
            throw TemplateResolutionError.templateMissing
        }

        if isDirectory.boolValue {
            return try bundleResolver.resolve(templatePath: trimmedPath)
        }
        return try rawDiskResolver.resolve(templatePath: trimmedPath)
    }

    private func missingReason(for runtime: String) -> String {
        runtime == "vz_linux" ? "vz_linux_template_missing" : "template_missing"
    }
}

import Foundation

final class TemplateValidator {
    private let protocolVersion = "1"
    private let helperVersion = "0.1.0"

    func validate(runtime: String, templatePath: String) -> TemplateValidationResponse {
        let trimmedPath = templatePath.trimmingCharacters(in: .whitespacesAndNewlines)
        let templateName = URL(fileURLWithPath: trimmedPath).lastPathComponent
        let templateID = "\(runtime):\(templateName)"

        guard !trimmedPath.isEmpty else {
            return TemplateValidationResponse(
                protocolVersion: protocolVersion,
                helperVersion: helperVersion,
                templateID: templateID,
                source: trimmedPath,
                ready: false,
                reasons: ["template_unconfigured"]
            )
        }

        let exists = FileManager.default.fileExists(atPath: trimmedPath)
        return TemplateValidationResponse(
            protocolVersion: protocolVersion,
            helperVersion: helperVersion,
            templateID: templateID,
            source: trimmedPath,
            ready: exists,
            reasons: exists ? [] : [missingReason(for: runtime)]
        )
    }

    private func missingReason(for runtime: String) -> String {
        runtime == "vz_linux" ? "vz_linux_template_missing" : "template_missing"
    }
}

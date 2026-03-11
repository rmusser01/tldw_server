import Foundation
import Testing
@testable import MacOSVZHelperDaemon

@Test func validateTemplateRejectsMissingImagePath() throws {
    let validator = TemplateValidator()
    let missingPath = NSTemporaryDirectory() + "/definitely-missing-vz-linux.img"

    let response = validator.validate(runtime: "vz_linux", templatePath: missingPath)

    #expect(response.protocolVersion == "1")
    #expect(response.helperVersion == "0.1.0")
    #expect(response.ready == false)
    #expect(response.templateID == "vz_linux:definitely-missing-vz-linux.img")
    #expect(response.source == missingPath)
    #expect(response.reasons.contains("vz_linux_template_missing"))
}

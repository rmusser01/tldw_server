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

@Test func validateTemplateIncludesBootMetadataForCanonicalBundle() throws {
    let validator = TemplateValidator()
    let bundleDirectory = try temporaryTemplateBundleDirectory()

    defer {
        try? FileManager.default.removeItem(at: bundleDirectory)
    }

    let response = validator.validate(runtime: "vz_linux", templatePath: bundleDirectory.path())

    #expect(response.ready == true)
    #expect(response.bootMode == "bundle")
    #expect(response.validationStrength == "strong")
}

private func temporaryTemplateBundleDirectory() throws -> URL {
    let root = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    FileManager.default.createFile(atPath: root.appendingPathComponent("kernel").path(), contents: Data("kernel".utf8))
    FileManager.default.createFile(atPath: root.appendingPathComponent("rootfs.img").path(), contents: Data("rootfs".utf8))
    let manifest = """
    {
      "bundle_version": "1",
      "boot_mode": "bundle",
      "kernel": "kernel",
      "rootfs": "rootfs.img",
      "guest_agent_path": "/usr/local/bin/tldw-agent-guest",
      "workspace_mount_tag": "workspace",
      "vsock_port": 1024
    }
    """
    try manifest.write(to: root.appendingPathComponent("manifest.json"), atomically: true, encoding: .utf8)
    return root
}

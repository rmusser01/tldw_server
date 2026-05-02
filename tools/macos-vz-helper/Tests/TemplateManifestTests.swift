import Foundation
import Testing
@testable import MacOSVZHelperDaemon

private func templateFixtureURL(_ relativePath: String, file: StaticString = #filePath) -> URL {
    let testsDirectory = URL(fileURLWithPath: "\(file)")
        .deletingLastPathComponent()
    return testsDirectory
        .appendingPathComponent("TemplateFixtures", isDirectory: true)
        .appendingPathComponent(relativePath, isDirectory: false)
}

@Test func templateManifestDecodesCanonicalBundleFields() throws {
    let data = try Data(contentsOf: templateFixtureURL("bundle/manifest.json"))

    let manifest = try JSONDecoder().decode(TemplateManifest.self, from: data)

    #expect(manifest.bundleVersion == "1")
    #expect(manifest.bootMode == .bundle)
    #expect(manifest.kernel == "kernel")
    #expect(manifest.rootfs == "rootfs.img")
    #expect(manifest.vsockPort == 1024)
    #expect(manifest.workspaceMountTag == "workspace")
    #expect(manifest.guestAgentPath == "/usr/local/bin/tldw-agent-guest")
}

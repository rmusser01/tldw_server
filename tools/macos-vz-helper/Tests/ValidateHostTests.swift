import Testing
@testable import MacOSVZHelperDaemon

@Test func validateHostReturnsUnavailableOnUnsupportedConfig() throws {
    let service = HelperService(hostFacts: HostFacts(isMacOS: false, isAppleSilicon: false))

    let response = service.validateHost(runtime: "vz_linux", networkPolicy: "deny_all")

    #expect(response.protocolVersion == "1")
    #expect(response.helperVersion == "0.1.0")
    #expect(response.available == false)
    #expect(response.executionMode == "none")
    #expect(response.reasons.contains("macos_host_required"))
    #expect(response.reasons.contains("apple_silicon_required"))
}

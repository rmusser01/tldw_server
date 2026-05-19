import Testing
@testable import MacOSVZHelperDaemon

@Test func pingIncludesProtocolAndHelperVersion() throws {
    let service = HelperService(
        helperInstanceID: "helper-test-1",
        helperStartedAt: "2026-05-09T00:00:00Z"
    )

    let response = service.ping()

    #expect(response.protocolVersion == "1")
    #expect(response.helperVersion == "0.1.0")
    #expect(response.status == "ok")
    #expect(response.details["transport"] == "unix")
    #expect(response.details["helper_instance_id"] == "helper-test-1")
    #expect(response.details["helper_started_at"] == "2026-05-09T00:00:00Z")
}

import Testing
@testable import MacOSVZHelperDaemon

@Test func pingIncludesProtocolAndHelperVersion() throws {
    let service = HelperService()

    let response = service.ping()

    #expect(response.protocolVersion == "1")
    #expect(response.helperVersion == "0.1.0")
    #expect(response.status == "ok")
    #expect(response.details["transport"] == "unix")
}

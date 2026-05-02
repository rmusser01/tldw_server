import Foundation

struct GuestAgentInfo: Equatable {
    let guestVersion: String?
    let workspaceRoot: String?
    let capabilities: [String]
    let capabilitiesKnown: Bool
}

import Foundation

public struct ProfileSummary: Codable, Equatable, Hashable, Identifiable {
    public var name: String
    public var description: String
    public var defaultAcousticProfile: String?

    public var id: String { name }

    enum CodingKeys: String, CodingKey {
        case name
        case description
        case defaultAcousticProfile = "default_acoustic_profile"
    }

    public init(name: String, description: String, defaultAcousticProfile: String? = nil) {
        self.name = name
        self.description = description
        self.defaultAcousticProfile = defaultAcousticProfile
    }
}

public struct ProfileCatalog: Codable, Equatable {
    public var driveProfiles: [ProfileSummary]
    public var acousticProfiles: [ProfileSummary]

    enum CodingKeys: String, CodingKey {
        case driveProfiles = "drive_profiles"
        case acousticProfiles = "acoustic_profiles"
    }

    public init(driveProfiles: [ProfileSummary], acousticProfiles: [ProfileSummary]) {
        self.driveProfiles = driveProfiles
        self.acousticProfiles = acousticProfiles
    }

    public static let fallback = ProfileCatalog(
        driveProfiles: [
            ProfileSummary(
                name: "seagate_ironwolf_pro_16tb",
                description: "Seagate IronWolf Pro 16TB physical-prior profile.",
                defaultAcousticProfile: "mounted_in_case"
            ),
            ProfileSummary(
                name: "desktop_7200_internal",
                description: "Desktop 7200 RPM internal HDD.",
                defaultAcousticProfile: "mounted_in_case"
            ),
        ],
        acousticProfiles: [
            ProfileSummary(name: "mounted_in_case", description: "Mounted in a PC case."),
            ProfileSummary(name: "drive_on_desk", description: "Bare drive on a desk."),
        ]
    )
}

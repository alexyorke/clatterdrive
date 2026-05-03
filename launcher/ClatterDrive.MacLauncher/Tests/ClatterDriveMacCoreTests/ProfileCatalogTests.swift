import XCTest
@testable import ClatterDriveMacCore

final class ProfileCatalogTests: XCTestCase {
    func testDecodesBackendProfileJson() throws {
        let data = """
        {
          "drive_profiles": [
            {
              "name": "seagate_ironwolf_pro_16tb",
              "description": "IronWolf",
              "default_acoustic_profile": "mounted_in_case"
            }
          ],
          "acoustic_profiles": [
            {
              "name": "mounted_in_case",
              "description": "Case"
            }
          ]
        }
        """.data(using: .utf8)!

        let catalog = try JSONDecoder().decode(ProfileCatalog.self, from: data)

        XCTAssertEqual(catalog.driveProfiles.first?.name, "seagate_ironwolf_pro_16tb")
        XCTAssertEqual(catalog.driveProfiles.first?.defaultAcousticProfile, "mounted_in_case")
        XCTAssertEqual(catalog.acousticProfiles.first?.name, "mounted_in_case")
    }
}

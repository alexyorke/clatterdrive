import Foundation
import XCTest
@testable import ClatterDriveMacCore

final class BackendOutputLinesTests: XCTestCase {
    func testSplitReadyRecordAndUTF8ArePreserved() {
        let parser = BackendOutputLines()
        let expected = "{\"event\":\"ready\",\"path\":\"caf\u{00e9}\"}"
        let bytes = Array((expected + "\nsecond\n").utf8)
        var lines: [String] = []
        for byte in bytes {
            lines.append(contentsOf: parser.append(Data([byte])))
        }
        XCTAssertEqual(lines, [expected, "second"])
    }

    func testPartialRecordWaitsForNewline() {
        let parser = BackendOutputLines()
        XCTAssertEqual(parser.append(Data("{\"event\":".utf8)), [])
        XCTAssertEqual(parser.append(Data("\"ready\"}\n".utf8)), ["{\"event\":\"ready\"}"])
    }
}

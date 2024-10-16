from meeko import utils

def test_begin_res_parsing():
    assert utils.parse_begin_res("SE8 C 23") == "C:23" 
    assert utils.parse_begin_res("SE8  23") == ":23"
    assert utils.parse_begin_res("SE8  23A") == ":23A"
    assert utils.parse_begin_res("SER A1234A") == "A:1234A"
    assert utils.parse_begin_res("    A 999") == "A:999"
    assert utils.parse_begin_res("  1") == ":1"
    assert utils.parse_begin_res(" A1234A") == "A:1234A"
    assert utils.parse_begin_res(" B1234") == "B:1234"
    assert utils.parse_begin_res("S  23A") == "S:23A"

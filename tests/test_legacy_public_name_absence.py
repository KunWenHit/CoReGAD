import re
from pathlib import Path


def test_legacy_public_name_absence() -> None:
    root = Path(__file__).parents[1]
    production = [root / "coregad", root / "configs", root / "scripts", root / "README.md"]
    forbidden = [
        *("G" + suffix for suffix in ("12", "14", "15", "16", "17", "18")),
        *("G" + suffix for suffix in ("18R", "18F", "18F1", "18F2", "18F2C", "18R1", "18T0", "18T1")),
        *(letter + digit for letter, digit in (("A", "0"), ("A", "1"), ("H", "1"))),
        "K" + "1X",
        "Trial" + "4",
        "M" + "000",
        "M" + "100",
        *("V" + str(index) for index in range(1, 8)),
        "E" + "075",
        "E" + "G075",
        "CP" + "-SRSE-075",
        "res" + "cue",
        "clo" + "sure",
        "candidate" + "_v",
        "seed0" + "_champion",
    ]
    forbidden_prefix = "phase" + "_"
    for path in production:
        files = [path] if path.is_file() else list(path.rglob("*"))
        for file in files:
            if file.is_file() and file.suffix in {".py", ".yaml", ".yml", ".md", ".json", ".toml"}:
                text = file.read_text(encoding="utf-8")
                assert forbidden_prefix not in text, (file, forbidden_prefix)
                assert all(
                    re.search(rf"(?<!\w){re.escape(token)}(?!\w)", text) is None
                    for token in forbidden
                ), (file, forbidden)

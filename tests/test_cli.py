from pathlib import Path

import pytest

from moxhimt_runner import cli


class FakeTranslator:
    def __init__(self, *args, **kwargs):
        pass

    def translate_document(self, text: str) -> str:
        return f"translated:{text}"


def test_cli_translates_file(monkeypatch, tmp_path: Path):
    source = tmp_path / "source.txt"
    target = tmp_path / "target.txt"
    source.write_text("你好", encoding="utf-8")
    monkeypatch.setattr(cli, "Translator", FakeTranslator)

    args = cli.build_parser().parse_args(["-i", str(source), "-o", str(target)])

    assert cli.run(args) == 0
    assert target.read_text(encoding="utf-8") == "translated:你好"


def test_cli_rejects_empty_input(monkeypatch):
    monkeypatch.setattr(cli, "Translator", FakeTranslator)
    args = cli.build_parser().parse_args(["--text", "   "])

    with pytest.raises(ValueError, match="rong"):
        cli.run(args)

from moxhimt_runner.translator import TranslationOptions, Translator, _find_ct2_variant


class FakeBackend:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def translate_batch(self, texts, options):
        self.calls.append(list(texts))
        return [f"VI:{text}" for text in texts]


def test_translate_document_preserves_blank_lines_and_line_endings():
    backend = FakeBackend()
    translator = Translator(_backend=backend, options=TranslationOptions(batch_size=2))

    result = translator.translate_document("第一章\r\n\r\n你好\n世界")

    assert result == "VI:第一章\r\n\r\nVI:你好\nVI:世界"
    assert backend.calls == [["第一章", "你好"], ["世界"]]


def test_translate_empty_document_does_not_call_backend():
    backend = FakeBackend()
    translator = Translator(_backend=backend)

    assert translator.translate_document("") == ""
    assert backend.calls == []


def test_options_must_be_positive():
    try:
        TranslationOptions(batch_size=0)
    except ValueError as exc:
        assert "batch_size" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_finds_ct2_variant_in_local_model(tmp_path):
    model_file = tmp_path / "ct2-int8" / "model.bin"
    model_file.parent.mkdir()
    model_file.touch()

    assert _find_ct2_variant(tmp_path, str(tmp_path)) == ("ct2-int8", "int8")

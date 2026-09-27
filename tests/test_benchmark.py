import importlib.util
import sys
from pathlib import Path

import pytest

BENCHMARK_PATH = Path(__file__).parents[1] / "benchmark.py"
SPEC = importlib.util.spec_from_file_location("benchmark", BENCHMARK_PATH)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


class FakeTokenizer:
    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": list(text)}


def test_counts_tokens_and_one_eos_per_nonempty_line():
    assert benchmark.count_generated_tokens(FakeTokenizer(), "abc\n\ndef") == 8


def test_percentile_uses_nearest_rank_index():
    values = [0.5, 0.1, 0.3, 0.2, 0.4]
    assert benchmark.percentile(values, 0.50) == 0.3
    assert benchmark.percentile(values, 0.95) == 0.5


def test_rejects_empty_sample(tmp_path):
    sample = tmp_path / "sample-input.txt"
    sample.write_text(" \n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="rong"):
        benchmark.read_sample(sample)

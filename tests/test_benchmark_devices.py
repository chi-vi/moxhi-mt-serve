import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parents[1] / "benchmark-devices.py"
SPEC = importlib.util.spec_from_file_location("benchmark_devices", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
benchmark_devices = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark_devices
SPEC.loader.exec_module(benchmark_devices)


def test_cpu_always_has_one_device():
    assert benchmark_devices.parse_devices("cpu", "0,1") == ["cpu"]


def test_parses_multiple_gpu_devices():
    assert benchmark_devices.parse_devices("cuda", "0, 2,3") == ["0", "2", "3"]


@pytest.mark.parametrize("value", ["", "0,0", "zero"])
def test_rejects_invalid_gpu_devices(value):
    with pytest.raises(ValueError):
        benchmark_devices.parse_devices("rocm", value)


def test_reads_only_nonempty_input_lines(tmp_path):
    sample = tmp_path / "sample.txt"
    sample.write_text("第一行\n\n 第二行 \n", encoding="utf-8")

    assert benchmark_devices.read_input_lines(sample) == ["第一行", " 第二行 "]


class FakeBackend:
    def __init__(self):
        self.generated = [10, 20, 15]

    def generate(self):
        return self.generated.pop(0)


def test_warmup_is_excluded_from_measurements(monkeypatch):
    times = iter([0.0, 1.0, 2.0, 4.0])
    monkeypatch.setattr(benchmark_devices.time, "perf_counter", lambda: next(times))

    best, measurements = benchmark_devices.benchmark_backend(FakeBackend(), 1, 2)

    assert len(measurements) == 2
    assert best.generated_tokens == 20

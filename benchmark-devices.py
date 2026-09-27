#!/usr/bin/env python3
"""Benchmark direct MoxhiMT generation without API server or HTTP overhead."""

from __future__ import annotations

import argparse
import gc
import os
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from huggingface_hub import snapshot_download

from moxhimt_runner.translator import (
    MOXHI_30_MODEL_ID,
    MOXHI_60_MODEL_ID,
    _find_ct2_variant,
)

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_DIR / "sample-input.txt"
MODEL_IDS = {"30": MOXHI_30_MODEL_ID, "60": MOXHI_60_MODEL_ID}


@dataclass(frozen=True)
class Measurement:
    seconds: float
    generated_tokens: int

    @property
    def tokens_per_second(self) -> float:
        return self.generated_tokens / self.seconds


class DirectBackend(Protocol):
    def generate(self) -> int: ...

    def close(self) -> None: ...


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Do truc tiep max generated tokens/giay, khong qua API server.",
    )
    parser.add_argument("runtime", choices=("cpu", "cuda", "rocm"))
    parser.add_argument("--model", choices=MODEL_IDS, default="30")
    parser.add_argument(
        "--dtype",
        choices=("f32", "f16"),
        default="f32",
        help="Kieu so hoc PyTorch tren GPU; CPU dung CTranslate2 INT8",
    )
    parser.add_argument(
        "--devices",
        default="0",
        help="Danh sach GPU, vi du 0,1,2; bi bo qua voi CPU (mac dinh: 0)",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument(
        "--repeat",
        type=int,
        default=16,
        help="Lap lai cac dong input de tao workload du lon (mac dinh: 16)",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-source-length", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=512)
    return parser


def parse_devices(runtime: str, value: str) -> list[str]:
    if runtime == "cpu":
        return ["cpu"]
    devices = [item.strip() for item in value.split(",") if item.strip()]
    if not devices:
        raise ValueError("--devices phai co it nhat mot GPU index")
    if len(set(devices)) != len(devices):
        raise ValueError("--devices khong duoc trung lap")
    if any(not device.isdigit() for device in devices):
        raise ValueError("Moi gia tri trong --devices phai la GPU index >= 0")
    return devices


def validate_args(args: argparse.Namespace) -> None:
    if args.warmup < 0:
        raise ValueError("--warmup phai >= 0")
    if args.runtime == "cpu" and args.dtype != "f32":
        raise ValueError("--dtype chi ap dung cho CUDA/ROCm; CPU luon dung INT8")
    for name in (
        "runs",
        "repeat",
        "batch_size",
        "num_beams",
        "max_source_length",
        "max_length",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} phai >= 1")


def read_input_lines(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"Input {path} phai la UTF-8") from exc
    except OSError as exc:
        raise RuntimeError(f"Khong doc duoc input {path}: {exc}") from exc
    lines = [line.rstrip("\r\n") for line in text.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"Input {path} rong")
    return lines


def batches(items: Sequence, batch_size: int) -> list[Sequence]:
    return [items[start : start + batch_size] for start in range(0, len(items), batch_size)]


def validate_token_lengths(tokenizer, lines: Sequence[str], max_length: int) -> None:
    encoded = tokenizer(list(lines), padding=False, truncation=False)["input_ids"]
    for index, token_ids in enumerate(encoded, start=1):
        if len(token_ids) > max_length:
            raise ValueError(f"Dong {index} dai {len(token_ids)} token, vuot gioi han {max_length}")


class CTranslate2DirectBackend:
    def __init__(self, model_id: str, lines: Sequence[str], args: argparse.Namespace) -> None:
        os.environ.setdefault("USE_TF", "0")
        try:
            import ctranslate2
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Hay cai CPU dependencies bang scripts/install.sh cpu") from exc

        model_path = Path(
            snapshot_download(
                model_id,
                allow_patterns=[
                    "config.json",
                    "source.spm",
                    "target.spm",
                    "vocab.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "ct2-int8_float32/*",
                    "ct2-int8/*",
                ],
            )
        )
        subdir, compute_type = _find_ct2_variant(model_path, model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        validate_token_lengths(self._tokenizer, lines, args.max_source_length)
        tokenized = []
        for line in lines:
            ids = self._tokenizer(line, truncation=False)["input_ids"]
            pad_id = self._tokenizer.pad_token_id
            if pad_id is not None:
                ids = [token_id for token_id in ids if token_id != pad_id]
            tokenized.append(self._tokenizer.convert_ids_to_tokens(ids))
        self._batches = batches(tokenized, args.batch_size)
        self._num_beams = args.num_beams
        self._max_length = args.max_length
        self._translator = ctranslate2.Translator(
            str(model_path / subdir),
            device="cpu",
            compute_type=compute_type,
        )

    def generate(self) -> int:
        generated_tokens = 0
        for batch in self._batches:
            results = self._translator.translate_batch(
                batch,
                beam_size=self._num_beams,
                max_decoding_length=self._max_length,
                no_repeat_ngram_size=3,
            )
            generated_tokens += sum(len(result.hypotheses[0]) for result in results)
        return generated_tokens

    def close(self) -> None:
        self._translator.unload_model()


class PyTorchDirectBackend:
    def __init__(
        self,
        model_id: str,
        lines: Sequence[str],
        args: argparse.Namespace,
        device_index: int,
    ) -> None:
        os.environ.setdefault("USE_TF", "0")
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                f"Hay cai dependencies bang scripts/install.sh {args.runtime}"
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch khong tim thay GPU")
        if args.runtime == "cuda" and (torch.version.cuda is None or torch.version.hip is not None):
            raise RuntimeError("PyTorch hien tai khong phai CUDA build")
        if args.runtime == "rocm" and torch.version.hip is None:
            raise RuntimeError("PyTorch hien tai khong phai ROCm/HIP build")
        if device_index >= torch.cuda.device_count():
            raise RuntimeError(
                f"GPU index {device_index} khong ton tai; chi co {torch.cuda.device_count()} device"
            )

        self._torch = torch
        self._device = torch.device(f"cuda:{device_index}")
        self.device_name = torch.cuda.get_device_name(device_index)
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        validate_token_lengths(self._tokenizer, lines, args.max_source_length)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        dtype = torch.float16 if args.dtype == "f16" else torch.float32
        self._model.to(device=self._device, dtype=dtype)
        self._model.eval()

        self._batches = [
            self._tokenizer(list(batch), return_tensors="pt", padding=True, truncation=False).to(
                self._device
            )
            for batch in batches(lines, args.batch_size)
        ]
        self._num_beams = args.num_beams
        self._max_length = args.max_length
        self._pad_token_id = self._model.config.pad_token_id

    def generate(self) -> int:
        generated_tokens = 0
        with self._torch.inference_mode():
            for batch in self._batches:
                generated = self._model.generate(
                    **batch,
                    num_beams=self._num_beams,
                    max_length=self._max_length,
                )
                generated_tokens += int(generated.ne(self._pad_token_id).sum().item())
        self._torch.cuda.synchronize(self._device)
        return generated_tokens

    def close(self) -> None:
        self._model.to("cpu")
        del self._model
        self._torch.cuda.empty_cache()
        gc.collect()


def measure_once(backend: DirectBackend) -> Measurement:
    started = time.perf_counter()
    generated_tokens = backend.generate()
    seconds = time.perf_counter() - started
    if generated_tokens < 1:
        raise RuntimeError("Model khong generate token nao")
    return Measurement(seconds, generated_tokens)


def benchmark_backend(
    backend: DirectBackend, warmup: int, runs: int
) -> tuple[Measurement, list[Measurement]]:
    for _ in range(warmup):
        backend.generate()
    measurements = [measure_once(backend) for _ in range(runs)]
    return max(measurements, key=lambda item: item.tokens_per_second), measurements


def run(args: argparse.Namespace) -> int:
    validate_args(args)
    devices = parse_devices(args.runtime, args.devices)
    lines = read_input_lines(args.input) * args.repeat
    model_id = MODEL_IDS[args.model]
    results: list[tuple[str, Measurement]] = []

    for device in devices:
        if args.runtime == "cpu":
            label = "CPU"
            backend: DirectBackend = CTranslate2DirectBackend(model_id, lines, args)
        else:
            gpu_backend = PyTorchDirectBackend(model_id, lines, args, int(device))
            backend = gpu_backend
            label = f"{args.runtime.upper()}:{device} ({gpu_backend.device_name})"

        try:
            best, measurements = benchmark_backend(backend, args.warmup, args.runs)
        finally:
            backend.close()
        label = f"{label} {args.dtype.upper()}" if args.runtime != "cpu" else f"{label} INT8"
        results.append((label, best))
        rates = [item.tokens_per_second for item in measurements]
        seconds = [item.seconds for item in measurements]
        print(
            f"{label}: MAX {best.tokens_per_second:.2f} tokens/s "
            f"({best.generated_tokens} generated tokens)"
        )
        print(
            f"  runs: {len(measurements)} | latency: {', '.join(f'{x:.3f}' for x in seconds)} s "
            f"| median: {statistics.median(seconds):.3f} s, "
            f"{statistics.median(rates):.2f} tokens/s"
        )

    best_label, best_measurement = max(results, key=lambda item: item[1].tokens_per_second)
    print(f"MAX {best_label}: {best_measurement.tokens_per_second:.2f} tokens/s")
    return 0


def main() -> None:
    parser = build_parser()
    try:
        raise SystemExit(run(parser.parse_args()))
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Loi: {exc}\n")


if __name__ == "__main__":
    main()

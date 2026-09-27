#!/usr/bin/env python3
"""Benchmark end-to-end generated-token throughput of a running MoxhiMT API."""

from __future__ import annotations

import argparse
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from transformers import AutoTokenizer

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_DIR / "sample-input.txt"
MODEL_IDS = {
    "30": "DanVP/MoxhiMT-30",
    "60": "DanVP/MoxhiMT-60",
}


@dataclass(frozen=True)
class Measurement:
    elapsed_seconds: float
    generated_tokens: int

    @property
    def tokens_per_second(self) -> float:
        return self.generated_tokens / self.elapsed_seconds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Do generated tokens/giay cua MoxhiMT API.",
    )
    parser.add_argument("--model", choices=MODEL_IDS, default="30")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.warmup < 0:
        raise ValueError("--warmup phai >= 0")
    if args.runs < 1:
        raise ValueError("--runs phai >= 1")
    if args.timeout <= 0:
        raise ValueError("--timeout phai > 0")


def read_sample(path: Path) -> bytes:
    try:
        sample = path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"Khong doc duoc input {path}: {exc}") from exc
    try:
        text = sample.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"Input {path} phai la UTF-8") from exc
    if not text.strip():
        raise RuntimeError(f"Input {path} rong")
    return sample


def request_translation(url: str, body: bytes, timeout: float) -> tuple[str, float]:
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "text/plain; charset=utf-8"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            output = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API tra HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Khong goi duoc API {url}: {exc.reason}") from exc
    elapsed = time.perf_counter() - started
    try:
        return output.decode("utf-8"), elapsed
    except UnicodeDecodeError as exc:
        raise RuntimeError("API khong tra ve UTF-8") from exc


def count_generated_tokens(tokenizer, output: str) -> int:
    """Count decoded tokens plus one generated EOS token per non-empty output line."""
    count = 0
    for line in output.splitlines():
        if not line.strip():
            continue
        count += len(tokenizer(line, add_special_tokens=False)["input_ids"]) + 1
    return count


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def print_summary(measurements: list[Measurement]) -> None:
    latencies = [item.elapsed_seconds for item in measurements]
    rates = [item.tokens_per_second for item in measurements]
    total_tokens = sum(item.generated_tokens for item in measurements)
    total_seconds = sum(latencies)

    print("\nTong ket")
    print(f"  Runs:                 {len(measurements)}")
    print(f"  Generated tokens:     {total_tokens}")
    print(f"  Throughput tong:      {total_tokens / total_seconds:.2f} tokens/s")
    print(f"  Throughput trung binh:{statistics.mean(rates):9.2f} tokens/s")
    print(f"  Throughput min/max:   {min(rates):.2f} / {max(rates):.2f} tokens/s")
    print(f"  Latency trung binh:   {statistics.mean(latencies) * 1000:.2f} ms")
    print(
        f"  Latency p50/p95:      {percentile(latencies, 0.50) * 1000:.2f} / "
        f"{percentile(latencies, 0.95) * 1000:.2f} ms"
    )


def run(args: argparse.Namespace) -> int:
    validate_args(args)
    body = read_sample(args.input)
    model_id = MODEL_IDS[args.model]
    endpoint = f"{args.url.rstrip('/')}/moxhi-{args.model}"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f"Model:    {model_id}")
    print(f"Endpoint: {endpoint}")
    print(f"Input:    {args.input} ({len(body)} bytes)")
    print(f"Warmup:   {args.warmup}; measured runs: {args.runs}")

    for index in range(args.warmup):
        output, _ = request_translation(endpoint, body, args.timeout)
        if not output.strip():
            raise RuntimeError("API tra ve output rong trong warmup")
        print(f"Warmup {index + 1}/{args.warmup} xong")

    measurements: list[Measurement] = []
    for index in range(args.runs):
        output, elapsed = request_translation(endpoint, body, args.timeout)
        tokens = count_generated_tokens(tokenizer, output)
        if tokens == 0:
            raise RuntimeError("API tra ve output khong co token")
        measurement = Measurement(elapsed, tokens)
        measurements.append(measurement)
        print(
            f"Run {index + 1:>2}/{args.runs}: {elapsed * 1000:>9.2f} ms | "
            f"{tokens:>6} tokens | {measurement.tokens_per_second:>9.2f} tokens/s"
        )

    print_summary(measurements)
    return 0


def main() -> None:
    parser = build_parser()
    try:
        raise SystemExit(run(parser.parse_args()))
    except (RuntimeError, ValueError) as exc:
        parser.exit(1, f"Loi: {exc}\n")


if __name__ == "__main__":
    main()

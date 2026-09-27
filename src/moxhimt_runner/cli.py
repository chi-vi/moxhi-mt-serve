from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .translator import DEFAULT_MODEL_ID, TranslationOptions, Translator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="moxhimt",
        description="Dich van ban Trung -> Viet bang DanVP/MoxhiMT-30.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--text", help="Van ban tieng Trung can dich")
    source.add_argument("--input", "-i", type=Path, help="Tep UTF-8 dau vao (mac dinh: stdin)")
    parser.add_argument("--output", "-o", type=Path, help="Tep UTF-8 dau ra (mac dinh: stdout)")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="Model ID hoac duong dan model")
    parser.add_argument("--revision", help="Nhan/phien ban model, vi du v4.0.3")
    parser.add_argument(
        "--backend",
        choices=("transformers", "ct2-int8"),
        default="transformers",
        help="Backend suy luan (mac dinh: transformers)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Thiet bi cho Transformers (mac dinh: auto)",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-source-length", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=512)
    return parser


def _read_input(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text
    if args.input is not None:
        return args.input.read_text(encoding="utf-8")
    if sys.stdin.isatty():
        raise ValueError("Hay dung --text, --input, hoac truyen van ban qua stdin")
    return sys.stdin.read()


def run(args: argparse.Namespace) -> int:
    source = _read_input(args)
    if not source.strip():
        raise ValueError("Van ban dau vao rong")

    options = TranslationOptions(
        num_beams=args.num_beams,
        max_source_length=args.max_source_length,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    translator = Translator(
        args.model,
        revision=args.revision,
        backend=args.backend,
        device=args.device,
        options=options,
    )
    result = translator.translate_document(source)
    if args.output:
        args.output.write_text(result, encoding="utf-8")
    else:
        sys.stdout.write(result)
        if not result.endswith("\n"):
            sys.stdout.write("\n")
    return 0


def main() -> None:
    parser = build_parser()
    try:
        raise SystemExit(run(parser.parse_args()))
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Loi: {exc}\n")


if __name__ == "__main__":
    main()

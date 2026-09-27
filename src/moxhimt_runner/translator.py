from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

MOXHI_30_MODEL_ID = "DanVP/MoxhiMT-30"
MOXHI_60_MODEL_ID = "DanVP/MoxhiMT-60"
DEFAULT_MODEL_ID = MOXHI_30_MODEL_ID

CT2_CONFIGS = {
    MOXHI_30_MODEL_ID: ("ct2-int8_float32", "int8_float32"),
    MOXHI_60_MODEL_ID: ("ct2-int8", "int8"),
}


def _find_ct2_variant(model_path: Path, model_id: str) -> tuple[str, str]:
    preferred = CT2_CONFIGS.get(model_id)
    candidates = [preferred] if preferred else []
    candidates.extend(config for config in CT2_CONFIGS.values() if config != preferred)
    for subdir, compute_type in candidates:
        if (model_path / subdir / "model.bin").is_file():
            return subdir, compute_type
    expected = " hoac ".join(config[0] for config in CT2_CONFIGS.values())
    raise ValueError(f"Khong tim thay ban CTranslate2 trong {model_path}; can {expected}")


@dataclass(frozen=True)
class TranslationOptions:
    num_beams: int = 1
    max_source_length: int = 512
    max_length: int = 512
    batch_size: int = 8

    def __post_init__(self) -> None:
        for name, value in (
            ("num_beams", self.num_beams),
            ("max_source_length", self.max_source_length),
            ("max_length", self.max_length),
            ("batch_size", self.batch_size),
        ):
            if value < 1:
                raise ValueError(f"{name} phai lon hon 0")


class Backend(Protocol):
    def translate_batch(self, texts: Sequence[str], options: TranslationOptions) -> list[str]: ...

    def close(self) -> None: ...


class TransformersBackend:
    def __init__(self, model_id: str, revision: str | None, device: str) -> None:
        try:
            # Transformers may otherwise import an installed TensorFlow runtime and emit
            # unrelated CUDA warnings, even though this runner only uses PyTorch.
            os.environ.setdefault("USE_TF", "0")
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on the user's environment
            raise RuntimeError(
                "Thieu backend PyTorch. Hay chay: pip install -e '.[pytorch]'"
            ) from exc

        self._torch = torch
        self._device = self._resolve_device(device)
        self.runs_on_gpu = self._device.startswith("cuda") or self._device == "mps"
        self._tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_id, revision=revision)
        self._model.to(self._device)
        self._model.eval()

    def _resolve_device(self, requested: str) -> str:
        if requested != "auto":
            if requested == "cuda" and not self._torch.cuda.is_available():
                raise RuntimeError("Da chon CUDA nhung PyTorch khong tim thay GPU CUDA")
            if requested == "mps" and not self._torch.backends.mps.is_available():
                raise RuntimeError("Da chon MPS nhung PyTorch khong ho tro MPS")
            return requested
        if self._torch.cuda.is_available():
            return "cuda"
        if self._torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def translate_batch(self, texts: Sequence[str], options: TranslationOptions) -> list[str]:
        raw = self._tokenizer(list(texts), padding=False, truncation=False)["input_ids"]
        for index, token_ids in enumerate(raw, start=1):
            if len(token_ids) > options.max_source_length:
                raise ValueError(
                    f"Dong {index} trong batch dai {len(token_ids)} token, vuot gioi han "
                    f"{options.max_source_length}; hay tach dong do thanh cac cau ngan hon"
                )
        encoded = self._tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=options.max_source_length,
        ).to(self._device)
        with self._torch.inference_mode():
            generated = self._model.generate(
                **encoded,
                num_beams=options.num_beams,
                max_length=options.max_length,
            )
        return self._tokenizer.batch_decode(generated, skip_special_tokens=True)

    def close(self) -> None:
        self._model.to("cpu")
        if self._device == "cuda":
            self._torch.cuda.empty_cache()


class CTranslate2Backend:
    def __init__(self, model_id: str, revision: str | None, device: str) -> None:
        try:
            os.environ.setdefault("USE_TF", "0")
            import ctranslate2
            from huggingface_hub import snapshot_download
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on the user's environment
            raise RuntimeError(
                "Thieu backend CTranslate2. Hay chay: pip install -e '.[cpu]'"
            ) from exc

        if device not in {"auto", "cpu"}:
            raise ValueError("Backend ct2-int8 chi ho tro --device auto hoac cpu")

        local_path = Path(model_id).expanduser()
        if local_path.is_dir():
            model_path = local_path
        else:
            model_path = Path(
                snapshot_download(
                    model_id,
                    revision=revision,
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
        ct2_subdir, compute_type = _find_ct2_variant(model_path, model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._translator = ctranslate2.Translator(
            str(model_path / ct2_subdir),
            device="cpu",
            compute_type=compute_type,
        )

    def _source_token_ids(self, text: str, max_length: int) -> list[int]:
        token_ids = self._tokenizer(text, truncation=False)["input_ids"]
        if len(token_ids) > max_length:
            raise ValueError(
                f"Dong dai {len(token_ids)} token, vuot gioi han {max_length}; "
                "hay tach dong do thanh cac cau ngan hon"
            )
        pad_token_id = self._tokenizer.pad_token_id
        if pad_token_id is not None:
            token_ids = [token_id for token_id in token_ids if token_id != pad_token_id]
        return token_ids

    def translate_batch(self, texts: Sequence[str], options: TranslationOptions) -> list[str]:
        token_batches = [
            self._tokenizer.convert_ids_to_tokens(
                self._source_token_ids(text, options.max_source_length)
            )
            for text in texts
        ]
        results = self._translator.translate_batch(
            token_batches,
            beam_size=options.num_beams,
            max_decoding_length=options.max_length,
            no_repeat_ngram_size=3,
        )
        return [
            self._tokenizer.decode(
                self._tokenizer.convert_tokens_to_ids(result.hypotheses[0]),
                skip_special_tokens=True,
            ).strip()
            for result in results
        ]

    def close(self) -> None:
        self._translator.unload_model()


class Translator:
    """Translate Chinese text while preserving line boundaries and blank lines."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        revision: str | None = None,
        backend: str = "transformers",
        device: str = "auto",
        options: TranslationOptions | None = None,
        _backend: Backend | None = None,
    ) -> None:
        self.options = options or TranslationOptions()
        if _backend is not None:
            self._backend = _backend
        elif backend == "transformers":
            self._backend = TransformersBackend(model_id, revision, device)
        elif backend == "ct2-int8":
            self._backend = CTranslate2Backend(model_id, revision, device)
        else:
            raise ValueError(f"Backend khong duoc ho tro: {backend}")
        self.runs_on_gpu = bool(getattr(self._backend, "runs_on_gpu", False))

    def translate_lines(self, lines: Sequence[str]) -> list[str]:
        if not lines:
            return []
        if self.runs_on_gpu:
            return self._backend.translate_batch(lines, self.options)

        output: list[str] = []
        batch_size = self.options.batch_size
        for start in range(0, len(lines), batch_size):
            output.extend(
                self._backend.translate_batch(lines[start : start + batch_size], self.options)
            )
        return output

    def translate_document(self, text: str) -> str:
        return self.translate_documents([text])[0]

    def translate_documents(self, texts: Sequence[str]) -> list[str]:
        """Translate independent documents while retaining their line endings."""
        rendered = [text.splitlines(keepends=True) for text in texts]
        translatable: list[str] = []
        positions: list[tuple[int, int]] = []
        for document_index, pieces in enumerate(rendered):
            for line_index, piece in enumerate(pieces):
                content = piece.rstrip("\r\n")
                if content.strip():
                    positions.append((document_index, line_index))
                    translatable.append(content)

        translated = self.translate_lines(translatable)
        for (document_index, line_index), value in zip(positions, translated, strict=True):
            piece = rendered[document_index][line_index]
            ending = piece[len(piece.rstrip("\r\n")) :]
            rendered[document_index][line_index] = value + ending
        return ["".join(pieces) for pieces in rendered]

    def close(self) -> None:
        close = getattr(self._backend, "close", None)
        if close is not None:
            close()

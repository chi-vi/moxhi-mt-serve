from __future__ import annotations

import argparse
import asyncio
import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Protocol

import anyio
from fastapi import FastAPI, HTTPException, Request, Response, status

from .translator import (
    MOXHI_30_MODEL_ID,
    MOXHI_60_MODEL_ID,
    TranslationOptions,
    Translator,
)


class TranslatorLike(Protocol):
    def translate_document(self, text: str) -> str: ...

    def translate_documents(self, texts: list[str]) -> list[str]: ...

    def close(self) -> None: ...


TranslatorFactory = Callable[..., TranslatorLike]
StartupValidator = Callable[[], None]


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} phai la so nguyen, nhan duoc {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} phai >= {minimum}, nhan duoc {value}")
    return value


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = os.getenv(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} phai la so, nhan duoc {raw!r}") from exc
    if value <= minimum:
        raise ValueError(f"{name} phai > {minimum}, nhan duoc {value}")
    return value


@dataclass(frozen=True)
class ServerSettings:
    model_30: str = MOXHI_30_MODEL_ID
    model_60: str = MOXHI_60_MODEL_ID
    revision_30: str | None = None
    revision_60: str | None = None
    backend: str = "ct2-int8"
    device: str = "auto"
    batch_size: int = 8
    num_beams: int = 1
    max_source_length: int = 512
    max_length: int = 512
    max_body_bytes: int = 2 * 1024 * 1024
    concurrency_per_model: int = 1
    queue_timeout_seconds: float = 30.0
    gpu_batch_max_lines: int = 100
    gpu_batch_wait_ms: int = 300

    @classmethod
    def from_env(cls) -> ServerSettings:
        return cls(
            model_30=os.getenv("MOXHI_30_MODEL", MOXHI_30_MODEL_ID),
            model_60=os.getenv("MOXHI_60_MODEL", MOXHI_60_MODEL_ID),
            revision_30=os.getenv("MOXHI_30_REVISION") or None,
            revision_60=os.getenv("MOXHI_60_REVISION") or None,
            backend=os.getenv("MOXHI_BACKEND", "ct2-int8"),
            device=os.getenv("MOXHI_DEVICE", "auto"),
            batch_size=_env_int("MOXHI_BATCH_SIZE", 8),
            num_beams=_env_int("MOXHI_NUM_BEAMS", 1),
            max_source_length=_env_int("MOXHI_MAX_SOURCE_LENGTH", 512),
            max_length=_env_int("MOXHI_MAX_LENGTH", 512),
            max_body_bytes=_env_int("MOXHI_MAX_BODY_BYTES", 2 * 1024 * 1024),
            concurrency_per_model=_env_int("MOXHI_CONCURRENCY_PER_MODEL", 1),
            queue_timeout_seconds=_env_float("MOXHI_QUEUE_TIMEOUT_SECONDS", 30.0),
            gpu_batch_max_lines=_env_int("MOXHI_GPU_BATCH_MAX_LINES", 100),
            gpu_batch_wait_ms=_env_int("MOXHI_GPU_BATCH_WAIT_MS", 300, minimum=0),
        )


@dataclass
class PendingTranslation:
    text: str
    line_count: int
    queued_at: float
    started: asyncio.Future[None]
    result: asyncio.Future[str]


class ModelService:
    def __init__(
        self,
        translator: TranslatorLike,
        *,
        concurrency: int,
        queue_timeout_seconds: float,
        batch_gpu: bool = False,
        gpu_batch_max_lines: int = 100,
        gpu_batch_wait_ms: int = 300,
    ) -> None:
        self.translator = translator
        self._semaphore = asyncio.Semaphore(concurrency)
        self._queue_timeout_seconds = queue_timeout_seconds
        self._batch_gpu = batch_gpu
        self._gpu_batch_max_lines = gpu_batch_max_lines
        self._gpu_batch_wait_seconds = gpu_batch_wait_ms / 1000
        self._queue: asyncio.Queue[PendingTranslation | None] | None = None
        self._worker: asyncio.Task[None] | None = None
        if batch_gpu:
            if gpu_batch_max_lines < 1:
                raise ValueError("MOXHI_GPU_BATCH_MAX_LINES phai >= 1")
            if not 0 <= gpu_batch_wait_ms <= 300:
                raise ValueError("MOXHI_GPU_BATCH_WAIT_MS phai trong khoang 0..300")
            self._queue = asyncio.Queue()
            self._worker = asyncio.create_task(self._run_gpu_batches())

    async def translate(self, text: str) -> str:
        if self._batch_gpu:
            return await self._translate_gpu(text)
        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=self._queue_timeout_seconds)
        except TimeoutError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="May chu dang ban; vui long thu lai",
                headers={"Retry-After": "1"},
            ) from exc

        try:
            return await anyio.to_thread.run_sync(self.translator.translate_document, text)
        finally:
            self._semaphore.release()

    async def _translate_gpu(self, text: str) -> str:
        assert self._queue is not None
        loop = asyncio.get_running_loop()
        item = PendingTranslation(
            text=text,
            line_count=sum(bool(line.strip()) for line in text.splitlines()),
            queued_at=loop.time(),
            started=loop.create_future(),
            result=loop.create_future(),
        )
        self._queue.put_nowait(item)
        try:
            try:
                await asyncio.wait_for(item.started, timeout=self._queue_timeout_seconds)
            except TimeoutError as exc:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="May chu dang ban; vui long thu lai",
                    headers={"Retry-After": "1"},
                ) from exc
            return await item.result
        finally:
            if not item.result.done():
                item.result.cancel()

    async def _run_gpu_batches(self) -> None:
        assert self._queue is not None
        carry: PendingTranslation | None = None
        while True:
            first = carry if carry is not None else await self._queue.get()
            carry = None
            if first is None:
                return
            if first.result.done():
                continue
            items = [first]
            total_lines = first.line_count
            closing = False
            deadline = first.queued_at + self._gpu_batch_wait_seconds
            while total_lines < self._gpu_batch_max_lines:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    next_item = await asyncio.wait_for(self._queue.get(), remaining)
                except TimeoutError:
                    break
                if next_item is None:
                    closing = True
                    break
                if next_item.result.done():
                    continue
                if total_lines + next_item.line_count > self._gpu_batch_max_lines:
                    carry = next_item
                    break
                items.append(next_item)
                total_lines += next_item.line_count

            active = [item for item in items if not item.result.done()]
            for item in active:
                if not item.started.done():
                    item.started.set_result(None)
            if active:
                try:
                    outputs = await anyio.to_thread.run_sync(
                        self.translator.translate_documents, [item.text for item in active]
                    )
                    if len(outputs) != len(active):
                        raise RuntimeError("Backend tra sai so luong ban dich")
                except ValueError as exc:
                    # A bad line in one request must not fail the other requests.
                    if len(active) == 1:
                        if not active[0].result.done():
                            active[0].result.set_exception(exc)
                    else:
                        for item in active:
                            try:
                                output = await anyio.to_thread.run_sync(
                                    self.translator.translate_document, item.text
                                )
                            except Exception as individual_exc:
                                if not item.result.done():
                                    item.result.set_exception(individual_exc)
                            else:
                                if not item.result.done():
                                    item.result.set_result(output)
                except Exception as exc:
                    for item in active:
                        if not item.result.done():
                            item.result.set_exception(exc)
                else:
                    for item, output in zip(active, outputs, strict=True):
                        if not item.result.done():
                            item.result.set_result(output)
            if closing:
                return

    async def close(self) -> None:
        if self._queue is not None and self._worker is not None:
            self._queue.put_nowait(None)
            await self._worker


def create_app(
    settings: ServerSettings | None = None,
    *,
    translator_factory: TranslatorFactory = Translator,
    startup_validator: StartupValidator | None = None,
) -> FastAPI:
    configured = settings or ServerSettings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if startup_validator is not None:
            await anyio.to_thread.run_sync(startup_validator)
        options = TranslationOptions(
            num_beams=configured.num_beams,
            max_source_length=configured.max_source_length,
            max_length=configured.max_length,
            batch_size=configured.batch_size,
        )
        services: dict[str, ModelService] = {}
        translators: list[TranslatorLike] = []
        try:
            for route, model_id, revision in (
                ("moxhi-30", configured.model_30, configured.revision_30),
                ("moxhi-60", configured.model_60, configured.revision_60),
            ):
                translator = await anyio.to_thread.run_sync(
                    lambda model_id=model_id, revision=revision: translator_factory(
                        model_id,
                        revision=revision,
                        backend=configured.backend,
                        device=configured.device,
                        options=options,
                    )
                )
                translators.append(translator)
                services[route] = ModelService(
                    translator,
                    concurrency=configured.concurrency_per_model,
                    queue_timeout_seconds=configured.queue_timeout_seconds,
                    batch_gpu=configured.backend == "transformers"
                    and (configured.device == "cuda" or getattr(translator, "runs_on_gpu", False)),
                    gpu_batch_max_lines=configured.gpu_batch_max_lines,
                    gpu_batch_wait_ms=configured.gpu_batch_wait_ms,
                )
            app.state.services = services
            app.state.ready = True
            yield
        finally:
            app.state.ready = False
            for service in services.values():
                await service.close()
            for translator in reversed(translators):
                await anyio.to_thread.run_sync(translator.close)

    app = FastAPI(
        title="MoxhiMT API",
        version="0.2.0",
        lifespan=lifespan,
    )
    app.state.ready = False

    async def translate(request: Request, service_name: str) -> Response:
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                declared_size = int(content_length)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="Content-Length khong hop le") from exc
            if declared_size > configured.max_body_bytes:
                raise HTTPException(status_code=413, detail="Request body qua lon")

        body = await request.body()
        if len(body) > configured.max_body_bytes:
            raise HTTPException(status_code=413, detail="Request body qua lon")
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(status_code=400, detail="Request body phai la UTF-8") from exc
        if not text.strip():
            raise HTTPException(status_code=400, detail="Request body rong")

        service: ModelService = request.app.state.services[service_name]
        try:
            result = await service.translate(text)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return Response(content=result, media_type="text/plain; charset=utf-8")

    @app.post("/moxhi-30", response_class=Response)
    async def moxhi_30(request: Request) -> Response:
        return await translate(request, "moxhi-30")

    @app.post("/moxhi-60", response_class=Response)
    async def moxhi_60(request: Request) -> Response:
        return await translate(request, "moxhi-60")

    @app.get("/health/live", include_in_schema=False)
    async def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready", include_in_schema=False)
    async def ready(request: Request) -> dict[str, str]:
        if not request.app.state.ready:
            raise HTTPException(status_code=503, detail="Models are not ready")
        return {"status": "ready"}

    return app


app = create_app()


def run_server(application: FastAPI) -> None:
    parser = argparse.ArgumentParser(description="Run the MoxhiMT API server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Hay cai server dependencies: pip install -e '.[server]'") from exc
    uvicorn.run(application, host=args.host, port=args.port, log_level=args.log_level)


def main() -> None:
    run_server(app)


if __name__ == "__main__":
    main()

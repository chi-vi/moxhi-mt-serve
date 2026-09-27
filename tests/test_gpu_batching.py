import asyncio
import threading
import unittest

from fastapi import HTTPException
from fastapi.testclient import TestClient

from moxhimt_runner.server import ModelService, ServerSettings, create_app
from moxhimt_runner.translator import TranslationOptions, Translator


class RecordingBackend:
    runs_on_gpu = True

    def __init__(self):
        self.calls = []

    def translate_batch(self, texts, options):
        self.calls.append(list(texts))
        return [f"VI:{text}" for text in texts]


class RecordingTranslator:
    def __init__(self):
        self.calls = []

    def translate_documents(self, texts):
        self.calls.append(list(texts))
        if any("bad" in text for text in texts):
            raise ValueError("bad input")
        return [f"VI:{text}" for text in texts]

    def translate_document(self, text):
        return self.translate_documents([text])[0]

    def close(self):
        pass


class BlockingTranslator(RecordingTranslator):
    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def translate_documents(self, texts):
        self.started.set()
        self.release.wait(2)
        return super().translate_documents(texts)


class GpuBatchingTests(unittest.TestCase):
    def test_cpu_keeps_configured_batch_size(self):
        backend = RecordingBackend()
        backend.runs_on_gpu = False
        translator = Translator(_backend=backend, options=TranslationOptions(batch_size=2))

        output = translator.translate_document("a\nb\nc")

        self.assertEqual(output, "VI:a\nVI:b\nVI:c")
        self.assertEqual(backend.calls, [["a", "b"], ["c"]])

    def test_gpu_route_sends_long_request_to_one_model_call(self):
        app = create_app(
            ServerSettings(
                backend="transformers", device="cuda",
                gpu_batch_max_lines=2, gpu_batch_wait_ms=300,
            ),
            translator_factory=lambda model_id, **kwargs: RecordingTranslator(),
        )
        with TestClient(app) as client:
            response = client.post("/moxhi-30", content="a\nb\nc")
            calls = client.app.state.services["moxhi-30"].translator.calls
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text, "VI:a\nb\nc")
        self.assertEqual(calls, [["a\nb\nc"]])

    def test_gpu_translates_entire_request_and_preserves_document_boundaries(self):
        backend = RecordingBackend()
        translator = Translator(_backend=backend, options=TranslationOptions(batch_size=8))
        first = "\n".join(f"句子{index}" for index in range(201)) + "\n"

        output = translator.translate_documents([first, "\r\n短句\r\n"])

        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(len(backend.calls[0]), 202)
        self.assertTrue(output[0].startswith("VI:句子0\n"))
        self.assertTrue(output[0].endswith("VI:句子200\n"))
        self.assertEqual(output[1], "\r\nVI:短句\r\n")

    def test_short_requests_share_one_gpu_call_at_line_limit(self):
        async def run():
            translator = RecordingTranslator()
            service = ModelService(
                translator, concurrency=1, queue_timeout_seconds=1,
                batch_gpu=True, gpu_batch_max_lines=5, gpu_batch_wait_ms=300,
            )
            try:
                outputs = await asyncio.gather(
                    service.translate("a\nb"),
                    service.translate("c\nd"),
                    service.translate("e"),
                )
                return outputs, translator.calls
            finally:
                await service.close()

        outputs, calls = asyncio.run(run())
        self.assertEqual(outputs, ["VI:a\nb", "VI:c\nd", "VI:e"])
        self.assertEqual(calls, [["a\nb", "c\nd", "e"]])

    def test_queue_keeps_batches_below_limit_and_isolates_bad_request(self):
        async def run():
            translator = RecordingTranslator()
            service = ModelService(
                translator, concurrency=1, queue_timeout_seconds=1,
                batch_gpu=True, gpu_batch_max_lines=5, gpu_batch_wait_ms=10,
            )
            try:
                first = await asyncio.gather(
                    service.translate("a\nb\nc"), service.translate("d\ne\nf")
                )
                second = await asyncio.gather(
                    service.translate("good"), service.translate("bad"),
                    return_exceptions=True,
                )
                return first, second, translator.calls
            finally:
                await service.close()

        first, second, calls = asyncio.run(run())
        self.assertEqual(first, ["VI:a\nb\nc", "VI:d\ne\nf"])
        self.assertEqual(second[0], "VI:good")
        self.assertIsInstance(second[1], ValueError)
        self.assertEqual(calls[:2], [["a\nb\nc"], ["d\ne\nf"]])

    def test_queue_timeout_does_not_stop_running_translation(self):
        async def run():
            translator = BlockingTranslator()
            service = ModelService(
                translator, concurrency=1, queue_timeout_seconds=0.03,
                batch_gpu=True, gpu_batch_max_lines=1, gpu_batch_wait_ms=300,
            )
            try:
                first = asyncio.create_task(service.translate("first"))
                self.assertTrue(await asyncio.to_thread(translator.started.wait, 1))
                with self.assertRaises(HTTPException) as error:
                    await service.translate("second")
                self.assertEqual(error.exception.status_code, 503)
                translator.release.set()
                self.assertEqual(await first, "VI:first")
                return translator.calls
            finally:
                translator.release.set()
                await service.close()

        self.assertEqual(asyncio.run(run()), [["first"]])


if __name__ == "__main__":
    unittest.main()

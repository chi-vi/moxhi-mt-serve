from typing import ClassVar

from fastapi.testclient import TestClient

from moxhimt_runner.server import ServerSettings, create_app
from moxhimt_runner.translator import MOXHI_30_MODEL_ID


class FakeTranslator:
    instances: ClassVar[list["FakeTranslator"]] = []

    def __init__(self, model_id, **kwargs):
        self.model_id = model_id
        self.closed = False
        self.__class__.instances.append(self)

    def translate_document(self, text: str) -> str:
        model = "30" if self.model_id == MOXHI_30_MODEL_ID else "60"
        return f"{model}:{text}"

    def close(self) -> None:
        self.closed = True


def make_client(*, max_body_bytes: int = 1024):
    FakeTranslator.instances = []
    app = create_app(
        ServerSettings(max_body_bytes=max_body_bytes),
        translator_factory=FakeTranslator,
    )
    return TestClient(app)


def test_raw_text_endpoints_and_health_checks():
    with make_client() as client:
        assert client.get("/health/live").json() == {"status": "ok"}
        assert client.get("/health/ready").json() == {"status": "ready"}

        response_30 = client.post("/moxhi-30", content="你好\n世界")
        response_60 = client.post("/moxhi-60", content="修仙")

        assert response_30.status_code == 200
        assert response_30.headers["content-type"].startswith("text/plain")
        assert response_30.text == "30:你好\n世界"
        assert response_60.text == "60:修仙"

    assert all(instance.closed for instance in FakeTranslator.instances)


def test_rejects_empty_invalid_utf8_and_oversized_bodies():
    with make_client(max_body_bytes=4) as client:
        assert client.post("/moxhi-30", content="   ").status_code == 400
        assert client.post("/moxhi-30", content=b"\xff").status_code == 400
        assert client.post("/moxhi-30", content="12345").status_code == 413

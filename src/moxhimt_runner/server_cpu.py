"""CPU entrypoint using the bundled CTranslate2 INT8 models."""

from dataclasses import replace

from .server import ServerSettings, create_app, run_server


def create_cpu_app():
    settings = replace(
        ServerSettings.from_env(),
        backend="ct2-int8",
        device="cpu",
    )
    return create_app(settings)


app = create_cpu_app()


def main() -> None:
    run_server(app)


if __name__ == "__main__":
    main()

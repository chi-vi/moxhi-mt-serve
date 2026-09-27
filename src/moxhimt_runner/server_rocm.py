"""AMD ROCm entrypoint using the shared PyTorch GPU implementation."""

from dataclasses import replace

from .server import ServerSettings, create_app, run_server


def validate_rocm_runtime() -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Chua cai PyTorch ROCm; hay chay scripts/install.sh rocm") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch khong tim thay GPU AMD ROCm/HIP")
    if torch.version.hip is None:
        raise RuntimeError("PyTorch hien tai khong duoc build voi ROCm/HIP")


def create_rocm_app():
    settings = replace(
        ServerSettings.from_env(),
        backend="transformers",
        device="cuda",
    )
    return create_app(settings, startup_validator=validate_rocm_runtime)


app = create_rocm_app()


def main() -> None:
    run_server(app)


if __name__ == "__main__":
    main()

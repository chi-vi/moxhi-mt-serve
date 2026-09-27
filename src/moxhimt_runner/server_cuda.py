"""NVIDIA CUDA entrypoint using the shared PyTorch GPU implementation."""

from dataclasses import replace

from .server import ServerSettings, create_app, run_server


def validate_cuda_runtime() -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Chua cai PyTorch CUDA; hay chay scripts/install.sh cuda") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch khong tim thay GPU NVIDIA CUDA")
    if torch.version.hip is not None:
        raise RuntimeError("Da cai PyTorch ROCm thay vi CUDA")
    if torch.version.cuda is None:
        raise RuntimeError("PyTorch hien tai khong duoc build voi CUDA")


def create_cuda_app():
    settings = replace(
        ServerSettings.from_env(),
        backend="transformers",
        device="cuda",
    )
    return create_app(settings, startup_validator=validate_cuda_runtime)


app = create_cuda_app()


def main() -> None:
    run_server(app)


if __name__ == "__main__":
    main()

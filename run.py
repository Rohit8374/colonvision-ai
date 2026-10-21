#!/usr/bin/env python3
"""
ColonVision AI — Unified Platform Runner
=========================================
Runs the FastAPI medical research backend and serves the compiled Stitch
frontend with automatic port detection and clean startup logging.
"""

import os
import socket
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def find_available_port(preferred: int = 8000, max_attempts: int = 10) -> int:
    for port in range(preferred, preferred + max_attempts):
        if not is_port_in_use(port):
            return port
    return preferred


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Start ColonVision AI Platform")

    # Support Render / Cloud PaaS port injection
    env_port = int(os.environ["PORT"]) if "PORT" in os.environ and os.environ["PORT"].isdigit() else None
    env_host = os.environ.get("HOST", "0.0.0.0")

    parser.add_argument("--port", type=int, default=env_port, help="Port to run on (default: $PORT or auto-detect 8000+)")
    parser.add_argument("--host", type=str, default=env_host, help="Host interface (default: $HOST or 0.0.0.0)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    port = args.port if args.port else find_available_port(8000)

    print("=" * 65)
    print("COLONVISION AI — Clinical Research Intelligence Platform")
    print("Differential Classification: Colorectal Cancer vs Colon Diverticula")
    print("=" * 65)
    print(f" * Backend API:   http://{args.host}:{port}/api/health")
    print(f" * Web Interface: http://{args.host}:{port}/")
    print(f" * Documentation: http://{args.host}:{port}/docs")
    print("-" * 65)

    # Configure persistent TORCH_HOME inside project repository so weights are retained on Render
    torch_cache_dir = PROJECT_ROOT / ".torch_cache"
    torch_cache_dir.mkdir(parents=True, exist_ok=True)
    if not os.environ.get("TORCH_HOME"):
        os.environ["TORCH_HOME"] = str(torch_cache_dir.resolve())

    # Pre-cache ImageNet-1K ResNet-50 weights before binding port to prevent request-time timeouts
    try:
        checkpoints_dir = Path(os.environ["TORCH_HOME"]) / "hub" / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        ckpt_file = checkpoints_dir / "resnet50-0676ba61.pth"
        if not ckpt_file.exists() or ckpt_file.stat().st_size < 100000000:
            print(f"Downloading ImageNet-1K weights to {ckpt_file}...")
            import urllib.request
            urllib.request.urlretrieve(
                "https://download.pytorch.org/models/resnet50-0676ba61.pth",
                str(ckpt_file),
            )
            print("Download finished.")
        import gc
        import torchvision.models as tv_models
        m = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1)
        del m
        gc.collect()
        print(f"ImageNet-1K validator weights verified and ready on disk ({ckpt_file.stat().st_size} bytes).")
    except Exception as e:
        print(f"Backbone pre-cache notice: {e}")

    print("Press CTRL+C to stop the platform server.")
    print("=" * 65 + "\n")

    import uvicorn
    uvicorn.run("backend.api:app", host=args.host, port=port, reload=args.reload)


if __name__ == "__main__":
    main()

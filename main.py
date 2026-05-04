"""Convenience launcher for the final project deliverables."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "20210808053_Final_Project.ipynb"
APP_PATH = PROJECT_ROOT / "app.py"


def run_command(command: list[str]) -> int:
    """Run one foreground command from the project root."""
    print("Running:", " ".join(command))
    return subprocess.call(command, cwd=PROJECT_ROOT)


def start_command(command: list[str]) -> subprocess.Popen:
    """Start one background command from the project root."""
    print("Starting:", " ".join(command))
    return subprocess.Popen(command, cwd=PROJECT_ROOT)


def streamlit_command(port: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_PATH),
        "--server.port",
        str(port),
    ]


def jupyter_command(port: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "notebook",
        str(NOTEBOOK_PATH),
        "--notebook-dir",
        str(PROJECT_ROOT),
        "--port",
        str(port),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Streamlit app and/or final Jupyter notebook."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--streamlit", action="store_true", help="Run only Streamlit.")
    mode.add_argument("--jupyter", action="store_true", help="Run only Jupyter Notebook.")
    mode.add_argument("--both", action="store_true", help="Run Streamlit and Jupyter.")
    parser.add_argument("--streamlit-port", type=int, default=8501)
    parser.add_argument("--jupyter-port", type=int, default=8888)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.jupyter:
        return run_command(jupyter_command(args.jupyter_port))

    if args.both:
        processes = [
            start_command(streamlit_command(args.streamlit_port)),
            start_command(jupyter_command(args.jupyter_port)),
        ]
        print(f"Streamlit: http://localhost:{args.streamlit_port}")
        print(f"Jupyter:   http://localhost:{args.jupyter_port}")
        print("Press Ctrl+C to stop both servers.")
        try:
            for process in processes:
                process.wait()
        except KeyboardInterrupt:
            for process in processes:
                process.terminate()
            return 130
        return max(process.returncode or 0 for process in processes)

    return run_command(streamlit_command(args.streamlit_port))


if __name__ == "__main__":
    raise SystemExit(main())

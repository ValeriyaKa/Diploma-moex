from __future__ import annotations

import os
import signal
import subprocess
import sys
import time


def start(cmd: list[str], env: dict[str, str]) -> subprocess.Popen:
    return subprocess.Popen(cmd, env=env)


def main() -> int:
    env = os.environ.copy()
    env.setdefault("API_BASE_URL", "http://127.0.0.1:8000")
    env.setdefault("MODELS_DIR", "./models")
    env.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
    env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

    api = start(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "backend.api.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ],
        env,
    )

    site = start(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "frontend/app.py",
            "--server.address=0.0.0.0",
            "--server.port=8501",
            "--server.headless=true",
            "--browser.gatherUsageStats=false",
        ],
        env,
    )

    children = [api, site]

    def stop_children(*_: object) -> None:
        for proc in children:
            if proc.poll() is None:
                proc.terminate()

    signal.signal(signal.SIGTERM, stop_children)
    signal.signal(signal.SIGINT, stop_children)

    while True:
        for proc in children:
            code = proc.poll()
            if code is not None:
                stop_children()
                return code
        time.sleep(1)


if __name__ == "__main__":
    raise SystemExit(main())

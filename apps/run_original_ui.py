"""Launcher for the original VieNeu-TTS Gradio UI with Windows-safe platform hooks."""

from __future__ import annotations

import os
import platform
import sys

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

if sys.platform == "win32":
    platform.system = lambda: "Windows"
    platform.machine = lambda: "AMD64"
    platform.release = lambda: "10"
    platform.version = lambda: "10.0.19045"

from apps.gradio_main import main


if __name__ == "__main__":
    main()

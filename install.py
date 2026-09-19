"""
LC MaskMaker -- install.py
-----------------------------------
Auto-run by ComfyUI-Manager after clone/update. The model-backed nodes need
`transformers` and `huggingface_hub`, which almost every ComfyUI install
already has. This script only installs a package that is missing, never
upgrades or pins one that is already importable, and never touches torch,
numpy or pillow.

Safe to re-run on every update: does nothing when everything is importable.
Model weights are not installed here. Each node downloads the model it needs
the first time you use it (see README.md, "Models and licensing").
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys

PACKAGES = [
    # (import name, pip name)
    ("transformers", "transformers"),
    ("huggingface_hub", "huggingface_hub"),
    ("timm", "timm"),  # only LC Normal Map (BAE) needs it
]


def _importable(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _pip_install(pip_name: str) -> bool:
    print(f"[LC MaskMaker] installing {pip_name} ...")
    cmd = [sys.executable, "-m", "pip", "install", pip_name]
    return subprocess.call(cmd) == 0


def main() -> int:
    missing = [(imp, pip) for imp, pip in PACKAGES if not _importable(imp)]
    if not missing:
        print("[LC MaskMaker] dependencies already present, nothing to do.")
        return 0

    failed = []
    for _imp, pip_name in missing:
        if not _pip_install(pip_name):
            failed.append(pip_name)

    if failed:
        print(
            "[LC MaskMaker] could not install: " + ", ".join(failed)
            + ". Install them manually with: python -m pip install " + " ".join(failed)
        )
        return 1
    print("[LC MaskMaker] dependencies installed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

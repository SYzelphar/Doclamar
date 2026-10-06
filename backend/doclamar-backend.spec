# -*- mode: python ; coding: utf-8 -*-
# PyInstaller build for the DocLAMAR backend sidecar.
#
#   cd backend
#   .venv\Scripts\python scripts\download_models.py   # bundle models so the app works offline
#   .venv\Scripts\pyinstaller doclamar-backend.spec --noconfirm
#
# Output: backend/dist/doclamar-backend/ (picked up by frontend/package.json -> extraResources)
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

datas, binaries, hiddenimports = [], [], []

# Conda-based Pythons keep OpenSSL in Library/bin, where PyInstaller doesn't look;
# without it _ssl fails to load and every HTTPS call to the LLM provider breaks.
conda_bin = Path(sys.base_prefix) / "Library" / "bin"
for pattern in ("libssl*.dll", "libcrypto*.dll"):
    binaries += [(str(dll), ".") for dll in conda_bin.glob(pattern)]
for package in ("fastembed", "pypdfium2", "pypdfium2_raw", "docx", "onnxruntime", "tokenizers", "rapidocr"):
    d, b, h = collect_all(package)
    datas += d
    binaries += b
    hiddenimports += h
hiddenimports += collect_submodules("uvicorn")

models = Path("models")
if not any(models.glob("models--*")):
    raise SystemExit("Models missing: run `python scripts/download_models.py` first.")
datas.append((str(models), "models"))

a = Analysis(
    ["api.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["torch", "tensorflow", "matplotlib", "tkinter", "IPython", "notebook", "pytest"],
    noarchive=False,
)
# OpenCV's video I/O plugin (~30 MB of FFmpeg) is never used for OCR.
a.binaries = [b for b in a.binaries if "opencv_videoio_ffmpeg" not in b[0]]
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="doclamar-backend",
    console=True,  # Electron spawns it hidden (windowsHide) and reads its stdout
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="doclamar-backend")

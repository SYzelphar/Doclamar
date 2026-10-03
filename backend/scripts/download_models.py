"""Download the embedding + reranker models into backend/models (bundled into the installer).

    python scripts/download_models.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from doclamar.config import Settings  # noqa: E402
from doclamar.embeddings import Embedder, Reranker  # noqa: E402


def main() -> None:
    settings = Settings()
    print(f"Downloading models into {settings.model_dir} …")
    Embedder(settings).embed_documents(["warm up"])
    reranker = Reranker(settings)
    if not reranker.available:
        raise SystemExit("Reranker download failed")
    reranker.score("warm up", ["warm up"])
    size = sum(f.stat().st_size for f in settings.model_dir.rglob("*") if f.is_file())
    print(f"Done ({size / 1e6:.0f} MB).")


if __name__ == "__main__":
    main()

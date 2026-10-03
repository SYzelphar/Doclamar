"""Central settings. Everything tunable lives here and can be overridden by env vars."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _bundle_dir() -> Path | None:
    """Directory PyInstaller unpacked the app into, if running frozen."""
    base = getattr(sys, "_MEIPASS", None)
    return Path(base) if base else None


def default_model_dir() -> Path:
    if os.environ.get("DOCLAMAR_MODEL_DIR"):
        return Path(os.environ["DOCLAMAR_MODEL_DIR"])
    bundle = _bundle_dir()
    if bundle and (bundle / "models").is_dir():
        return bundle / "models"
    # Running from source: use backend/models (filled by scripts/download_models.py
    # or on first use).
    return Path(__file__).resolve().parent.parent / "models"


@dataclass
class Settings:
    home: Path = field(default_factory=lambda: Path(os.environ.get("DOCLAMAR_HOME", Path.home() / ".doclamar")))
    model_dir: Path = field(default_factory=default_model_dir)

    embed_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    # BGE v1.5 retrieves better when short queries carry this instruction.
    query_instruction: str = "Represent this sentence for searching relevant passages: "

    chunk_chars: int = 1000
    chunk_overlap_chars: int = 150
    min_chunk_chars: int = 300
    max_file_mb: int = field(default_factory=lambda: _env_int("DOCLAMAR_MAX_FILE_MB", 100))

    # Retrieval
    candidate_pool: int = 50     # per retriever (vector + keyword) before fusion
    rerank_pool: int = 30        # fused candidates sent to the cross-encoder
    top_k: int = 6               # chunks given to the LLM
    # Relevance = sigmoid(cross-encoder logit). Calibrated with eval/run_eval.py: answerable
    # questions can score as low as ~1e-3 when phrased differently from the text, while
    # unrelated questions score < 1e-4.
    min_relevance: float = 0.1          # below this, try one rewritten query
    not_found_relevance: float = 1e-4   # below this, answer "not found" without calling the LLM

    @property
    def index_db(self) -> Path:
        return self.home / "index.db"

    @property
    def chats_db(self) -> Path:
        return self.home / "chats.db"

    @property
    def log_file(self) -> Path:
        return self.home / "doclamar.log"

    def ensure_dirs(self) -> None:
        self.home.mkdir(parents=True, exist_ok=True)

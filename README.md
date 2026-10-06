# DocLamar

Ask questions about the documents on your own computer. Point DocLamar at a folder of
PDFs (including scanned ones), Word files, text, Markdown or photos of pages, and it answers
with citations to the exact file, page and section — or open a single document and chat with
just that file.

Indexing, OCR and search run locally (no GPU, no PyTorch). Only the question and the few
passages needed to answer it are sent to the LLM provider you choose (Groq or Google Gemini).

## How it works

```
INGEST  (background, incremental — only new or changed files are re-read)
  folder ─► parse (pypdfium2 / python-docx / text)  ─►  sentence-aligned chunks
           pages, sections, tables, soft hyphens        ~1000 chars, section + page range
           scanned pages + images ─► OCR (RapidOCR, ONNX) with column-aware reading order
        ─► embed (bge-small-en-v1.5, ONNX)  ─►  SQLite: files · chunks + vectors · FTS5 keyword index

QUERY   (LangGraph)
  condense follow-up ─► hybrid search (vector + BM25, fused with RRF, scoped to the folder/file)
                     ─► cross-encoder rerank (ms-marco-MiniLM)
                     ─► nothing relevant? rewrite the query once and search again
                     ─► one LLM call: answer with [n] citations
```

A typical question costs **one** LLM call and returns in about 2 seconds.

| | |
|---|---|
| `backend/doclamar/parsing.py` | File readers shared by folder search and document chat |
| `backend/doclamar/ocr.py` | OCR engine and reading-order reconstruction for scans |
| `backend/doclamar/chunking.py` | Sentence-aligned chunks that keep section and page numbers |
| `backend/doclamar/store.py` | SQLite index (files, chunks, embeddings, FTS5) and folder scoping |
| `backend/doclamar/indexer.py` | Incremental folder sync on a background worker with progress |
| `backend/doclamar/retrieval.py` | Hybrid search + reranking |
| `backend/doclamar/pipeline.py` | The LangGraph question-answering graph |
| `backend/doclamar/llm.py` | Groq / Gemini clients with timeouts and retries |
| `backend/doclamar/history.py` | Chat sessions and messages (with citations) |
| `backend/api.py` | FastAPI sidecar used by the desktop app |
| `backend/cli.py` | Same engine from the terminal |
| `frontend/` | Electron + React desktop app |

## Running from source

Requirements: Python 3.11+, Node 20+.

```powershell
# Backend
cd backend
python -m venv .venv
.venv\Scripts\activate              # macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
python scripts/download_models.py   # ~150 MB, once
copy .env.example .env              # optional for dev: add a Groq or Gemini key
pytest

# Desktop app (starts the backend from backend/.venv automatically)
cd ../frontend
npm install
npm run electron-dev
```

The app asks for your LLM API key in **Settings**. It is checked with the provider and stored
encrypted on your machine (Electron `safeStorage`); it is never part of the installer.
Free keys: [Groq](https://console.groq.com/keys) · [Gemini](https://aistudio.google.com/apikey).

### Command line

```bash
cd backend
python cli.py index  "C:\Users\me\Documents\papers"
python cli.py ask    "C:\Users\me\Documents\papers" "What loss function was used?"
python cli.py chat   "C:\Users\me\Documents\papers\paper.pdf"
python cli.py status "C:\Users\me\Documents\papers"
```

## Building the installer

```powershell
cd backend
.venv\Scripts\python scripts\download_models.py
.venv\Scripts\pyinstaller doclamar-backend.spec --noconfirm   # -> backend/dist/doclamar-backend
cd ../frontend
npm run dist                                                 # -> frontend/release/DocLamar Setup x.y.z.exe
```

The models (search, reranking and OCR) are bundled, so the installed app indexes, reads
scans and searches offline.

## OCR

PDF pages that contain images but no text layer, and image files (PNG, JPG, TIFF including
multi-page, BMP, WebP), are read with [RapidOCR](https://github.com/RapidAI/RapidOCR)
(PP-OCR models on ONNX Runtime, Apache-2.0). Pages with a real text layer are never OCRed, so
digital PDFs cost nothing extra.

- **Reading order:** OCR engines return lines top to bottom, which interleaves the columns of
  a two-column scan. `ocr.py` finds the column gutter (recursively, so 3 columns work), treats
  full-width titles and figures as band separators, and rebuilds lines in reading order.
- **Accuracy:** on a 150-DPI greyscale JPEG scan of the evaluation paper, 99.6% of words are
  recovered, reading order matches the original text at 0.94, and section headings are still
  detected.
- **Speed:** about 2.5–4 s per page on a 16-thread desktop CPU (pages are OCRed in parallel),
  slower on small laptops. It happens once, in the background, with page-level progress in the
  sidebar.
- **Transparency:** passages that came from OCR carry an "OCR" badge in the sources, and the
  LLM is told the text may contain recognition errors.
- Photos are auto-rotated using their EXIF orientation. Turn OCR off with `DOCLAMAR_OCR=0`;
  scanned files are then listed as "needs OCR" and picked up automatically once it is back on.

## Evaluation

`backend/eval/run_eval.py` hides the LIP2AUDSPEC paper (`eval/data/machinelearning.pdf`) in a
folder with distractor documents — some deliberately close in vocabulary (speech enhancement
with PESQ and LSTMs, CNN training with Adam and dropout) plus a scanned PDF with no text — and
asks 15 questions with reference answers.

```bash
python -m eval.run_eval                       # retrieval only, offline
python -m eval.run_eval --scanned             # same, with the paper replaced by an image-only scan
python -m eval.run_eval --answers --pause 6   # also generate answers (needs an API key)
```

Results with the default settings (top-6 chunks, Groq `openai/gpt-oss-120b`):

| Metric | Digital paper | Scanned paper (OCR) |
|---|---|---|
| Paper retrieved in top-6 (file hit rate) | 15 / 15 | 15 / 15 |
| MRR of the first chunk from the paper | 0.97 | 0.97 |
| Share of top-6 chunks from the paper | 0.90 | 0.89 |
| Reference-answer key terms found in retrieved text | 0.86 | 0.87 |
| Unanswerable questions | "not found" | "not found" |
| Median retrieval time | ~0.9 s | ~1.0 s |
| Indexing the folder | 2 s | 19 s (5 pages of OCR) |
| Answers that cite their sources | 15 / 15 | — |
| Median answer time (incl. LLM) | ~1.9 s | — |

## Configuration

Environment variables (the desktop app sets the LLM ones from Settings):

| Variable | Default | |
|---|---|---|
| `LLM_PROVIDER` | first provider with a key | `groq` or `gemini` |
| `GROQ_API_KEY` / `GROQ_MODEL` | — / `openai/gpt-oss-120b` | |
| `GEMINI_API_KEY` / `GEMINI_MODEL` | — / `gemini-2.5-flash` | |
| `DOCLAMAR_HOME` | `~/.doclamar` | index (`index.db`), chats (`chats.db`), log |
| `DOCLAMAR_MAX_FILE_MB` | `100` | larger files are skipped |
| `DOCLAMAR_OCR` | `1` | `0` turns OCR off |

Changing the embedding model or chunking rebuilds the index automatically.

## Security

- The backend listens on `127.0.0.1` only, on a random port chosen per launch.
- Every request needs a random per-launch token that only the app window receives, so other
  websites open in your browser cannot read your files or chats through it. Host headers other
  than localhost are rejected (DNS rebinding), and CORS is limited to the app.
- The renderer is sandboxed with a Content Security Policy; the preload exposes only named
  functions, and "open file" only opens documents.
- Profiles ("Who's using DocLamar?") only separate chat history on a shared computer; they are
  not accounts.

## Limitations

- OCR is tuned for printed text. Handwriting, very low-resolution photos and pages scanned
  sideways (without EXIF orientation) read poorly. The bundled recognition model covers English
  and Chinese.
- A PDF that mixes text pages and scanned pages, indexed while OCR was off, keeps its text pages
  searchable but is not re-OCRed automatically later; edit or re-save it to re-index.
- Vector search is exact (brute force over all chunks in memory), which is fast up to a few
  hundred thousand chunks. Beyond that, swap `IndexStore.vector_search` for an ANN index.
- The installer is not code-signed, so Windows SmartScreen will warn on first run.

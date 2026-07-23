# PDF OCR Webpage using Unlimited-OCR

A simple web app to upload a PDF (or image) and extract its contents as markdown using
[baidu/Unlimited-OCR](https://github.com/baidu/Unlimited-OCR).

- **Backend:** FastAPI (single process). Converts PDF pages to images with PyMuPDF, then runs the
  model's documented `infer_multi(...)` multi-page parsing path via `transformers`
  (`trust_remote_code=True`).
- **Frontend:** Plain HTML/JS. Upload a file, wait for OCR, view the rendered markdown, download it.

## Requirements

- Python 3.12 recommended.
- **An NVIDIA CUDA GPU is required to actually run OCR.** The model is loaded in `bfloat16` and
  moved to CUDA, per the official usage. ~6.7 GB of VRAM is needed for weights alone; a 12 GB card
  (e.g. desktop RTX 4070) is comfortable.
- Disk space + internet on first run: the `baidu/Unlimited-OCR` weights (~6.7 GB) are downloaded
  automatically from Hugging Face.

## Local development (no GPU, e.g. macOS)

You can run the app without a GPU to work on the UI. OCR requests will return a clear
"CUDA GPU not available" error instead of results.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # CPU torch wheel is fine for UI-only dev
uvicorn app:app --reload
```

Open http://127.0.0.1:8000.

## GPU deployment

1. Copy/clone this project to the CUDA machine.
2. Install a CUDA-matched torch build first, then the rest:

```bash
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu129
pip install -r requirements.txt
```

3. Run the server:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

4. Check `http://<host>:8000/api/health` — it should report `"cuda_available": true`.
   The model is loaded lazily on the first OCR request (expect the first request to take a while:
   weight download + load).

## API

- `GET /` — upload page.
- `POST /api/ocr` — multipart upload (`file` field, `.pdf/.png/.jpg/.jpeg`). Returns
  `{job_id, markdown, page_count, token_count, elapsed_seconds}`.
- `GET /api/health` — `{cuda_available, model_loaded}`.
- `GET /results/{job_id}/...` — static output files (extracted figure images, `result.md`).

## Notes / limitations

- Inference is serialized with a lock (one OCR job at a time on the GPU); concurrent uploads queue.
- Job output directories under `data/jobs/` are not auto-cleaned (possible follow-up).
- No auth/rate limiting — intended as a simple personal tool.

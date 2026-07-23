"""FastAPI app: upload a PDF/image, run Unlimited-OCR, return markdown."""

import os
import time
import uuid

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

import ocr_engine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JOBS_DIR = os.path.join(BASE_DIR, "data", "jobs")
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}
MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB

os.makedirs(JOBS_DIR, exist_ok=True)

app = FastAPI(title="Unlimited-OCR PDF Reader")

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
app.mount("/results", StaticFiles(directory=JOBS_DIR), name="results")

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/api/health")
async def health():
    return {
        "cuda_available": ocr_engine.cuda_available(),
        "model_loaded": ocr_engine.is_model_loaded(),
    }


def _process_job(upload_path: str, ext: str, job_dir: str, job_id: str) -> dict:
    """Blocking pipeline: PDF -> page images -> infer_multi -> markdown. Runs in a worker thread."""
    started = time.time()

    if ext == ".pdf":
        pages_dir = os.path.join(job_dir, "pages")
        image_paths = ocr_engine.pdf_to_images(upload_path, pages_dir)
        if not image_paths:
            raise ValueError("The PDF appears to have no pages.")
    else:
        image_paths = [upload_path]

    markdown, token_count = ocr_engine.run_ocr(image_paths, job_dir)
    markdown = ocr_engine.rewrite_image_links(markdown, job_id)

    return {
        "job_id": job_id,
        "markdown": markdown,
        "page_count": len(image_paths),
        "token_count": token_count,
        "elapsed_seconds": round(time.time() - started, 1),
    }


@app.post("/api/ocr")
async def ocr(file: UploadFile = File(...)):
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext or 'unknown'}'. Allowed: PDF, PNG, JPG.",
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 100 MB).")

    job_id = uuid.uuid4().hex[:12]
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    upload_path = os.path.join(job_dir, f"upload{ext}")
    with open(upload_path, "wb") as f:
        f.write(contents)

    try:
        result = await run_in_threadpool(_process_job, upload_path, ext, job_dir, job_id)
    except RuntimeError as e:
        # e.g. "CUDA GPU not available" from the model loader
        return JSONResponse(status_code=503, content={"detail": str(e)})
    except ValueError as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"OCR failed: {e}"})

    return result

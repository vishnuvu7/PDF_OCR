"""Unlimited-OCR inference wrapper.

Loads baidu/Unlimited-OCR lazily (once) via transformers and exposes:
- pdf_to_images(): PDF -> per-page PNGs (PyMuPDF)
- run_ocr(): multi-page parsing via the model's documented infer_multi() path
"""

import os
import threading

MODEL_NAME = os.environ.get("UNLIMITED_OCR_MODEL", "baidu/Unlimited-OCR")

# infer_multi mutates shared model/config state (e.g. config.sliding_window),
# so all inference on the single GPU is serialized through this lock.
_GPU_LOCK = threading.Lock()

_model = None
_tokenizer = None
_load_lock = threading.Lock()


def is_model_loaded() -> bool:
    return _model is not None


def cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def load_model():
    """Load model + tokenizer once. Raises RuntimeError with a clear message if no CUDA GPU."""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    with _load_lock:
        if _model is not None:
            return _model, _tokenizer

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPU not available. This app must run on a machine with an NVIDIA GPU "
                "to perform OCR. Deploy it to a CUDA host and try again."
            )

        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
        )
        _model = model.eval().cuda()
        _tokenizer = tokenizer

    return _model, _tokenizer


def pdf_to_images(pdf_path: str, out_dir: str, dpi: int = 300) -> list[str]:
    """Convert each PDF page to a PNG in out_dir. Returns the ordered list of image paths."""
    import fitz  # PyMuPDF

    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    image_paths = []
    try:
        for i, page in enumerate(doc):
            out_path = os.path.join(out_dir, f"page_{i + 1:04d}.png")
            page.get_pixmap(matrix=mat).save(out_path)
            image_paths.append(out_path)
    finally:
        doc.close()
    return image_paths


def run_ocr(image_paths: list[str], job_dir: str) -> tuple[str, int]:
    """Run multi-page parsing on the given page images.

    Returns (markdown, output_token_count). Also writes result.md and any extracted
    figure images under job_dir (save_results=True).
    """
    model, tokenizer = load_model()

    with _GPU_LOCK:
        outputs, output_tokens = model.infer_multi(
            tokenizer,
            prompt="<image>Multi page parsing.",
            image_files=image_paths,
            output_path=job_dir,
            image_size=1024,
            max_length=32768,
            no_repeat_ngram_size=35,
            ngram_window=1024,
            save_results=True,
        )

    return outputs, output_tokens


def rewrite_image_links(markdown: str, job_id: str) -> str:
    """Point the model's relative figure links (images/...) at the served job directory."""
    return markdown.replace("](images/", f"](/results/{job_id}/images/")

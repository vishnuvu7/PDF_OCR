const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const selectedFileEl = document.getElementById("selected-file");
const selectedFileName = document.getElementById("selected-file-name");
const clearFileBtn = document.getElementById("clear-file");
const runBtn = document.getElementById("run-btn");
const spinner = document.getElementById("spinner");
const errorBanner = document.getElementById("error-banner");
const resultSection = document.getElementById("result-section");
const resultRendered = document.getElementById("result-rendered");
const resultRaw = document.getElementById("result-raw");
const resultMeta = document.getElementById("result-meta");
const toggleRawBtn = document.getElementById("toggle-raw");
const downloadBtn = document.getElementById("download-btn");
const gpuStatus = document.getElementById("gpu-status");

const ALLOWED = [".pdf", ".png", ".jpg", ".jpeg"];
let selectedFile = null;
let lastMarkdown = "";

// --- GPU status badge -------------------------------------------------------
fetch("/api/health")
  .then((r) => r.json())
  .then((h) => {
    gpuStatus.hidden = false;
    if (h.cuda_available) {
      gpuStatus.textContent = h.model_loaded ? "GPU ready · model loaded" : "GPU ready";
      gpuStatus.classList.add("ok");
    } else {
      gpuStatus.textContent = "No CUDA GPU detected — OCR requests will fail on this machine";
      gpuStatus.classList.add("warn");
    }
  })
  .catch(() => {});

// --- File selection ----------------------------------------------------------
function setFile(file) {
  if (!file) return;
  const ext = "." + file.name.split(".").pop().toLowerCase();
  if (!ALLOWED.includes(ext)) {
    showError(`Unsupported file type "${ext}". Please choose a PDF, PNG or JPG.`);
    return;
  }
  hideError();
  selectedFile = file;
  selectedFileName.textContent = `${file.name} (${formatSize(file.size)})`;
  selectedFileEl.hidden = false;
  runBtn.disabled = false;
}

function clearFile() {
  selectedFile = null;
  fileInput.value = "";
  selectedFileEl.hidden = true;
  runBtn.disabled = true;
}

function formatSize(bytes) {
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

dropZone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => setFile(fileInput.files[0]));
clearFileBtn.addEventListener("click", clearFile);

["dragover", "dragenter"].forEach((evt) =>
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((evt) =>
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
  })
);
dropZone.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));

// --- Error handling ----------------------------------------------------------
function showError(message) {
  errorBanner.textContent = message;
  errorBanner.hidden = false;
}

function hideError() {
  errorBanner.hidden = true;
}

// --- OCR request -------------------------------------------------------------
runBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  hideError();
  resultSection.hidden = true;
  spinner.hidden = false;
  runBtn.disabled = true;

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const resp = await fetch("/api/ocr", { method: "POST", body: formData });
    const data = await resp.json().catch(() => ({}));

    if (!resp.ok) {
      throw new Error(data.detail || `Request failed with status ${resp.status}`);
    }

    showResult(data);
  } catch (err) {
    showError(err.message || "OCR request failed.");
  } finally {
    spinner.hidden = true;
    runBtn.disabled = false;
  }
});

// --- Result rendering ---------------------------------------------------------
function showResult(data) {
  lastMarkdown = data.markdown || "";

  const html = DOMPurify.sanitize(marked.parse(lastMarkdown));
  resultRendered.innerHTML = html;
  resultRaw.textContent = lastMarkdown;

  const pages = data.page_count === 1 ? "1 page" : `${data.page_count} pages`;
  resultMeta.textContent = `${pages} · ${data.token_count} tokens · ${data.elapsed_seconds}s`;

  resultRendered.hidden = false;
  resultRaw.hidden = true;
  toggleRawBtn.textContent = "View raw";
  resultSection.hidden = false;
  resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

toggleRawBtn.addEventListener("click", () => {
  const showRaw = resultRaw.hidden;
  resultRaw.hidden = !showRaw;
  resultRendered.hidden = showRaw;
  toggleRawBtn.textContent = showRaw ? "View rendered" : "View raw";
});

downloadBtn.addEventListener("click", () => {
  const blob = new Blob([lastMarkdown], { type: "text/markdown" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "ocr-result.md";
  a.click();
  URL.revokeObjectURL(url);
});

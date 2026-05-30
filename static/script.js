// ── DOM refs ──
const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const queryPreview = document.getElementById("queryPreview");
const queryImg = document.getElementById("queryImg");
const queryName = document.getElementById("queryName");
const resultsContainer = document.getElementById("resultsContainer");
const loadingOverlay = document.getElementById("loadingOverlay");
const loadingText = document.getElementById("loadingText");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const searchInfo = document.getElementById("searchInfo");
const dbImageList = document.getElementById("dbImageList");
const btnBuild = document.getElementById("btnBuild");

let selectedFile = null;

// ── Initialise ───────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  checkStatus();
  loadDbImages();

  // Click to select file
  dropZone.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", (e) => handleFile(e.target.files[0]));

  // Drag & drop
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  });
  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("drag-over");
  });
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  });
});

// ── File handling ────────────────────────────────────────────
function handleFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    alert("请选择图片文件");
    return;
  }
  selectedFile = file;
  const url = URL.createObjectURL(file);
  queryImg.src = url;
  queryName.textContent = file.name;
  queryPreview.classList.remove("hidden");
  searchInfo.classList.add("hidden");
}

// ── Search ───────────────────────────────────────────────────
async function doSearch() {
  if (!selectedFile) {
    alert("请先选择查询图片");
    return;
  }

  showLoading("正在检索…");
  resultsContainer.innerHTML = "";

  const form = new FormData();
  form.append("image", selectedFile);

  try {
    const resp = await fetch("/search", { method: "POST", body: form });
    const data = await resp.json();
    hideLoading();

    if (data.error) {
      alert("检索失败: " + data.error);
      return;
    }

    searchInfo.classList.remove("hidden");
    searchInfo.textContent = `⏱ 检索用时 ${data.elapsed_ms} ms`;
    renderResults(data.results);
  } catch (err) {
    hideLoading();
    alert("网络错误: " + err.message);
  }
}

// ── Render results ────────────────────────────────────────────
function renderResults(results) {
  if (!results || results.length === 0) {
    resultsContainer.innerHTML =
      '<div class="placeholder"><span>未找到相似图片</span></div>';
    return;
  }

  let html = "";
  results.forEach((r, i) => {
    html += `
      <div class="result-card" style="animation-delay:${i * 0.08}s">
        <img src="${r.url}" alt="${r.name}" loading="lazy">
        <div class="card-info">
          <span class="card-rank">${i + 1}</span>
          <div class="card-name" title="${r.name}">${r.name}</div>
          <div class="card-score">相似度: ${(r.score * 100).toFixed(1)}%</div>
        </div>
      </div>`;
  });
  resultsContainer.innerHTML = html;
}

// ── Build index ──────────────────────────────────────────────
async function buildIndex() {
  if (!confirm("将重新构建索引，确认？")) return;
  showLoading("正在构建索引，请耐心等待…");
  btnBuild.disabled = true;
  try {
    const resp = await fetch("/build", { method: "POST" });
    const data = await resp.json();
    hideLoading();
    btnBuild.disabled = false;
    if (data.error) {
      alert("构建失败: " + data.error);
    } else {
      alert(`索引构建完成！共 ${data.count} 张图片`);
      checkStatus();
      loadDbImages();
    }
  } catch (err) {
    hideLoading();
    btnBuild.disabled = false;
    alert("错误: " + err.message);
  }
}

// ── Status ───────────────────────────────────────────────────
async function checkStatus() {
  try {
    const resp = await fetch("/status");
    const data = await resp.json();
    if (data.ready) {
      statusDot.className = "dot on";
      statusText.textContent = `索引就绪 – ${data.indexed} 张图片`;
    } else {
      statusDot.className = "dot off";
      statusText.textContent =
        "未构建索引 – 请将图片放入 data/database/ 后点击「构建索引」";
    }
  } catch {
    statusDot.className = "dot off";
    statusText.textContent = "无法连接服务";
  }
}

// ── Load database images ─────────────────────────────────────
async function loadDbImages() {
  try {
    const resp = await fetch("/database-images");
    const data = await resp.json();
    if (data.images.length === 0) {
      dbImageList.innerHTML =
        '<span style="color:#999;font-size:13px;">数据库为空，请添加图片到 data/database/</span>';
      return;
    }
    let html = "";
    data.images.forEach((name) => {
      html += `<img class="db-thumb" src="/images/database/${encodeURIComponent(
        name
      )}" title="${name}" onclick="searchDbImage('${name.replace(/'/g, "\\'")}')">`;
    });
    dbImageList.innerHTML = html;
  } catch {
    dbImageList.innerHTML =
      '<span style="color:#999;font-size:13px;">加载失败</span>';
  }
}

// ── Search by clicking a DB thumbnail ─────────────────────────
async function searchDbImage(name) {
  showLoading("正在检索…");
  resultsContainer.innerHTML = "";
  try {
    const resp = await fetch("/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ db_image: name }),
    });
    const data = await resp.json();
    hideLoading();
    if (data.error) {
      alert("检索失败: " + data.error);
      return;
    }
    searchInfo.classList.remove("hidden");
    searchInfo.textContent = `⏱ 检索用时 ${data.elapsed_ms} ms`;
    renderResults(data.results);

    // Show query image
    queryImg.src = `/images/database/${encodeURIComponent(name)}`;
    queryName.textContent = name;
    queryPreview.classList.remove("hidden");
  } catch (err) {
    hideLoading();
    alert("错误: " + err.message);
  }
}

// ── Loading helpers ──────────────────────────────────────────
function showLoading(text) {
  loadingText.textContent = text || "处理中…";
  loadingOverlay.classList.remove("hidden");
}

function hideLoading() {
  loadingOverlay.classList.add("hidden");
}

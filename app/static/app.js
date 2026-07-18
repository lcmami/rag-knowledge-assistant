"use strict";

// 前端純呼叫後端 API：不含金鑰、不生成回答、不偽造來源。
// 與後端固定拒答訊息一致（app/core/constants.py REFUSAL_MESSAGE）。
const REFUSAL_MESSAGE = "目前提供的文件中沒有足夠資訊可以回答此問題。";
// LLM 未設定（503 LLM_NOT_CONFIGURED）時的固定提示。
const LLM_NOT_CONFIGURED_MSG =
  "LLM 服務尚未設定，目前仍可使用文件上傳與向量檢索功能。";

const $ = (sel) => document.querySelector(sel);

// ---- 共用工具 ----
function fmtSize(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

function setMsg(el, text, kind) {
  el.textContent = text || "";
  el.className = "msg" + (kind ? " " + kind : "");
}

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s == null ? "" : String(s);
  return div.innerHTML;
}

// 高亮 [S1] 這類 citation，方便人工對照。
function highlightCitations(text) {
  return escapeHtml(text).replace(/\[S(\d+)\]/g, '<span class="cite">[S$1]</span>');
}

// 從後端回應解析出可讀錯誤；回傳 {code, message}。
async function parseError(res) {
  let code = "HTTP_" + res.status;
  let message = "請求失敗（HTTP " + res.status + "）。";
  try {
    const body = await res.json();
    if (body && body.error) {
      code = body.error.code || code;
      message = body.error.message || message;
    } else if (body && body.detail) {
      // FastAPI 驗證錯誤（422）
      message =
        typeof body.detail === "string"
          ? body.detail
          : JSON.stringify(body.detail);
    }
  } catch (_) {
    /* 非 JSON 回應，沿用預設訊息 */
  }
  return { code, message };
}

// ---- Tab 切換 ----
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
    tab.classList.add("active");
    $("#panel-" + tab.dataset.tab).classList.add("active");
  });
});

// ---- 服務就緒狀態 + LLM banner ----
async function refreshReady() {
  const statusEl = $("#ready-status");
  const dot = statusEl.querySelector(".dot");
  const label = statusEl.querySelector(".ready-text");
  try {
    const res = await fetch("/ready");
    const body = await res.json();
    const checks = body.checks || {};
    const llmOk = checks.llm_config === "ok";
    // llm_config 以外皆 ok/loaded → 核心功能就緒。
    const coreOk = ["qdrant", "collection", "embedding", "metadata_db"].every(
      (k) => checks[k] === "ok" || checks[k] === "loaded"
    );

    // LLM 未設定屬預期行為：顯示提示，不當成錯誤。
    $("#llm-banner").hidden = llmOk;

    if (coreOk && llmOk) {
      dot.className = "dot dot-ok";
      label.textContent = "全部服務就緒";
    } else if (coreOk && !llmOk) {
      dot.className = "dot dot-partial";
      label.textContent = "核心就緒 · LLM 未設定";
    } else {
      dot.className = "dot dot-error";
      label.textContent = "服務未就緒";
    }
  } catch (_) {
    dot.className = "dot dot-error";
    label.textContent = "無法連線後端";
  }
}

// ---- 區塊一：文件管理 ----
async function loadDocuments() {
  const tbody = $("#docs-tbody");
  setMsg($("#docs-msg"), "載入中…", "loading");
  try {
    const res = await fetch("/api/documents");
    if (!res.ok) {
      const { message } = await parseError(res);
      throw new Error(message);
    }
    const data = await res.json();
    setMsg($("#docs-msg"), "共 " + data.total + " 份文件。", "");
    if (!data.documents.length) {
      tbody.innerHTML = '<tr><td colspan="7" class="empty">尚無文件，請先上傳。</td></tr>';
      return;
    }
    tbody.innerHTML = data.documents
      .map((d) => {
        const cls =
          d.status === "indexed"
            ? "status-indexed"
            : d.status === "failed"
            ? "status-failed"
            : "status-other";
        const title = d.error_message ? ' title="' + escapeHtml(d.error_message) + '"' : "";
        return (
          "<tr>" +
          "<td>" + escapeHtml(d.filename) + "</td>" +
          "<td>" + escapeHtml(d.file_type) + "</td>" +
          "<td>" + fmtSize(d.file_size) + "</td>" +
          '<td><span class="status-pill ' + cls + '"' + title + ">" + escapeHtml(d.status) + "</span></td>" +
          "<td>" + d.chunk_count + "</td>" +
          "<td>" + escapeHtml(d.created_at) + "</td>" +
          '<td><button class="danger" data-id="' + escapeHtml(d.document_id) +
          '" data-name="' + escapeHtml(d.filename) + '">刪除</button></td>' +
          "</tr>"
        );
      })
      .join("");
  } catch (err) {
    setMsg($("#docs-msg"), "載入失敗：" + err.message, "error");
    tbody.innerHTML = '<tr><td colspan="7" class="empty">載入失敗。</td></tr>';
  }
}

// 事件委派：刪除
$("#docs-tbody").addEventListener("click", async (e) => {
  const btn = e.target.closest("button.danger");
  if (!btn) return;
  const id = btn.dataset.id;
  const name = btn.dataset.name;
  if (!confirm('確定刪除文件「' + name + '」及其所有向量嗎？')) return;
  btn.disabled = true;
  setMsg($("#docs-msg"), "刪除中…", "loading");
  try {
    const res = await fetch("/api/documents/" + encodeURIComponent(id), {
      method: "DELETE",
    });
    if (res.status !== 204) {
      const { message } = await parseError(res);
      throw new Error(message);
    }
    setMsg($("#docs-msg"), "已刪除「" + name + "」。", "ok");
    await loadDocuments();
  } catch (err) {
    setMsg($("#docs-msg"), "刪除失敗：" + err.message, "error");
    btn.disabled = false;
  }
});

$("#upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = $("#file-input");
  const btn = $("#upload-btn");
  if (!input.files.length) return;
  const fd = new FormData();
  fd.append("file", input.files[0]);
  btn.disabled = true;
  setMsg($("#upload-msg"), "上傳與索引中…", "loading");
  try {
    const res = await fetch("/api/documents", { method: "POST", body: fd });
    if (res.status !== 201) {
      const { message } = await parseError(res);
      throw new Error(message);
    }
    const data = await res.json();
    setMsg(
      $("#upload-msg"),
      "已索引「" + data.filename + "」，產生 " + data.chunk_count + " 個 chunk。",
      "ok"
    );
    input.value = "";
    await loadDocuments();
  } catch (err) {
    setMsg($("#upload-msg"), "上傳失敗：" + err.message, "error");
  } finally {
    btn.disabled = false;
  }
});

$("#refresh-docs").addEventListener("click", loadDocuments);

// ---- 區塊二：向量檢索 ----
function parseTopK(el) {
  const v = el.value.trim();
  return v === "" ? null : parseInt(v, 10);
}

$("#retrieve-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = $("#retrieve-q").value.trim();
  const btn = $("#retrieve-btn");
  const out = $("#retrieve-results");
  if (!q) return;
  const payload = { question: q };
  const topk = parseTopK($("#retrieve-topk"));
  if (topk !== null) payload.top_k = topk;

  btn.disabled = true;
  out.innerHTML = "";
  setMsg($("#retrieve-msg"), "檢索中…", "loading");
  try {
    const res = await fetch("/api/retrieve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const { message } = await parseError(res);
      throw new Error(message);
    }
    const data = await res.json();
    setMsg(
      $("#retrieve-msg"),
      "命中 " + data.results.length + " 筆 · 檢索耗時 " + data.retrieval_time_ms + " ms",
      "ok"
    );
    if (!data.results.length) {
      out.innerHTML = '<div class="result-card">沒有通過 threshold 的結果。</div>';
      return;
    }
    out.innerHTML = data.results.map(renderSourceCard).join("");
  } catch (err) {
    setMsg($("#retrieve-msg"), "檢索失敗：" + err.message, "error");
  } finally {
    btn.disabled = false;
  }
});

function renderSourceCard(r) {
  const loc = r.page != null ? "第 " + r.page + " 頁" : r.section ? escapeHtml(r.section) : "—";
  return (
    '<div class="result-card">' +
    '<div class="result-head">' +
    '<span class="sid">[' + escapeHtml(r.source_id) + "]</span>" +
    '<span class="score">score ' + r.score.toFixed(4) + "</span>" +
    "</div>" +
    '<div class="result-meta">' +
    escapeHtml(r.filename) + " · " + loc + " · chunk #" + r.chunk_index +
    "</div>" +
    '<div class="result-text">' + escapeHtml(r.text) + "</div>" +
    "</div>"
  );
}

// ---- 區塊三：RAG 問答 ----
$("#query-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = $("#query-q").value.trim();
  const btn = $("#query-btn");
  const answerEl = $("#query-answer");
  const sourcesEl = $("#query-sources");
  if (!q) return;
  const payload = { question: q };
  const topk = parseTopK($("#query-topk"));
  if (topk !== null) payload.top_k = topk;

  btn.disabled = true;
  answerEl.hidden = true;
  sourcesEl.innerHTML = "";
  setMsg($("#query-msg"), "查詢中（檢索 + LLM 生成）…", "loading");
  try {
    const res = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const { code, message } = await parseError(res);
      // LLM 未設定：顯示明確提示，不當成一般未知錯誤。
      if (res.status === 503 && code === "LLM_NOT_CONFIGURED") {
        setMsg($("#query-msg"), LLM_NOT_CONFIGURED_MSG, "warn");
        return;
      }
      throw new Error(message);
    }

    const data = await res.json();
    const isFallback = data.answer === REFUSAL_MESSAGE;
    setMsg(
      $("#query-msg"),
      "檢索 " + data.retrieval_time_ms + " ms · 生成 " + data.generation_time_ms + " ms" +
        (isFallback ? " · 系統拒答" : ""),
      "ok"
    );
    answerEl.hidden = false;
    answerEl.className = "answer " + (isFallback ? "fallback" : "ok");
    answerEl.innerHTML =
      '<div class="answer-text">' + highlightCitations(data.answer) + "</div>" +
      '<div class="answer-meta">' +
      (isFallback
        ? "此為系統拒答（來源為空）。"
        : "引用 " + data.sources.length + " 個來源；來源 metadata 由後端檢索結果組裝。") +
      "</div>";
    sourcesEl.innerHTML = data.sources.map(renderSourceCard).join("");
  } catch (err) {
    setMsg($("#query-msg"), "查詢失敗：" + err.message, "error");
  } finally {
    btn.disabled = false;
  }
});

// ---- 初始化 ----
refreshReady();
loadDocuments();

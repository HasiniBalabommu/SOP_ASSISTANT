// script.js — talks to the FastAPI backend at API_BASE.
// Change this if your backend runs on a different host/port.
const API_BASE = "http://127.0.0.1:8000";
const SESSION_ID = "web_" + Math.random().toString(36).slice(2, 10);
let currentSessionId = SESSION_ID; // reuse the randomly generated one as the initial session

const messagesEl = document.getElementById("messages");
const emptyStateEl = document.getElementById("emptyState");
const composerEl = document.getElementById("composer");
const queryInput = document.getElementById("queryInput");
const sendBtn = document.getElementById("sendBtn");

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const uploadLog = document.getElementById("uploadLog");

const apiStatusDot = document.getElementById("apiStatusDot");
const apiStatusText = document.getElementById("apiStatusText");

const statNodes = document.getElementById("statNodes");
const statEdges = document.getElementById("statEdges");
const machineTags = document.getElementById("machineTags");
const faultTags = document.getElementById("faultTags");
const refreshKgBtn = document.getElementById("refreshKg");

const composerAttachBtn = document.getElementById("composerAttachBtn");
const composerFileInput = document.getElementById("composerFileInput");
const filePreviewRow = document.getElementById("filePreviewRow");

// ---------- Tabs ----------
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
    if (btn.dataset.tab === "chats") loadSessionList();
  });
});

// ---------- Chat history sidebar ----------
const chatListEl = document.getElementById("chatList");
const newChatBtn = document.getElementById("newChatBtn");

async function loadSessionList() {
  try {
    const res = await fetch(`${API_BASE}/sessions`);
    const data = await res.json();
    renderSessionList(data.sessions || []);
  } catch {
    chatListEl.innerHTML = `<p class="panel-hint">Could not load chat history.</p>`;
  }
}

function renderSessionList(sessionIds) {
  if (!sessionIds.length) {
    chatListEl.innerHTML = `<p class="panel-hint">No past chats yet.</p>`;
    return;
  }
  chatListEl.innerHTML = "";
  sessionIds.forEach((sid) => {
    const item = document.createElement("div");
    item.className = "chat-list-item" + (sid === currentSessionId ? " active" : "");
    item.innerHTML = `
      <span class="chat-list-item-title">Loading…</span>
      <button type="button" class="chat-delete-btn" title="Delete chat">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M3 6h18M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2m3 0v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6h14z" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </button>`;

    item.addEventListener("click", (e) => {
      if (e.target.closest(".chat-delete-btn")) return; // don't switch chats when clicking delete
      switchToSession(sid);
    });

    item.querySelector(".chat-delete-btn").addEventListener("click", (e) => {
      e.stopPropagation();
      deleteSession(sid, item);
    });

    chatListEl.appendChild(item);

    // fetch a preview (first question) for the title
    fetch(`${API_BASE}/sessions/${sid}/history`)
      .then((r) => r.json())
      .then((data) => {
        const history = data.history || [];
        const title = history.length ? history[0].query : sid;
        const time = history.length ? new Date(history[history.length - 1].timestamp).toLocaleString() : "";
        const titleEl = item.querySelector(".chat-list-item-title");
        if (titleEl) {
          titleEl.textContent = title;
          const timeEl = document.createElement("span");
          timeEl.className = "chat-list-item-time";
          timeEl.textContent = time;
          item.insertBefore(timeEl, item.querySelector(".chat-delete-btn"));
        }
      })
      .catch(() => {});
  });
}

async function deleteSession(sessionId, itemEl) {
  try {
    const res = await fetch(`${API_BASE}/sessions/${sessionId}`, { method: "DELETE" });
    if (!res.ok) throw new Error(`Server returned ${res.status}`);
    itemEl.remove();

    // if the deleted chat was the one currently open, reset to a fresh chat
    if (sessionId === currentSessionId) {
      newChatBtn.click();
    }

    // if the list is now empty, show the empty-state hint
    if (!chatListEl.children.length) {
      chatListEl.innerHTML = `<p class="panel-hint">No past chats yet.</p>`;
    }
  } catch (err) {
    console.error("Failed to delete session", err);
  }
}

async function switchToSession(sessionId) {
  currentSessionId = sessionId;
  document.querySelectorAll(".chat-list-item").forEach((el) => el.classList.remove("active"));
  loadSessionList();

  messagesEl.innerHTML = "";
  try {
    const res = await fetch(`${API_BASE}/sessions/${sessionId}/history`);
    const data = await res.json();
    const history = data.history || [];
    if (!history.length) {
      messagesEl.innerHTML = `<div class="empty-state"><p class="empty-title">Ask about your equipment</p></div>`;
      return;
    }
    history.forEach((turn) => {
      addUserMessage(turn.query);
      addAnswerMessage({ answer: turn.answer, verification: {}, citations: [] });
    });
  } catch (err) {
    console.error("Failed to load session history", err);
  }
}

// "+ New Chat" — starts a fresh session ID and clears the message pane.
// Note: a brand-new chat won't appear in the sidebar list until the
// first question is actually asked in it — the backend only creates
// a conversation file once memory.add_turn() is called, so an empty
// chat has nothing to save yet. This is expected, not a bug.
newChatBtn.addEventListener("click", () => {
  currentSessionId = "web_" + Math.random().toString(36).slice(2, 10);
  messagesEl.innerHTML = `<div class="empty-state" id="emptyState"><p class="empty-title">Ask about your equipment</p><p class="empty-hint">e.g. "What should I do if I see error code E205?"</p></div>`;
  document.querySelectorAll(".chat-list-item").forEach((el) => el.classList.remove("active"));
});

// ---------- Health check ----------
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error();
    apiStatusDot.className = "status-dot online";
    apiStatusText.textContent = "API connected";
  } catch {
    apiStatusDot.className = "status-dot offline";
    apiStatusText.textContent = "API unreachable — is uvicorn running?";
  }
}

// ---------- Knowledge graph summary ----------
async function loadKgSummary() {
  try {
    const res = await fetch(`${API_BASE}/kg/summary`);
    const data = await res.json();
    statNodes.textContent = data.total_nodes ?? "0";
    statEdges.textContent = data.total_edges ?? "0";
    renderTagList(machineTags, data.machines || []);
    renderTagList(faultTags, data.fault_codes || []);
  } catch {
    statNodes.textContent = "–";
    statEdges.textContent = "–";
  }
}

function renderTagList(container, items) {
  container.innerHTML = "";
  if (!items.length) {
    container.classList.add("empty");
    container.textContent = "None yet";
    return;
  }
  container.classList.remove("empty");
  items.forEach((item) => {
    const span = document.createElement("span");
    span.className = "tag";
    span.textContent = item;
    container.appendChild(span);
  });
}

// ---------- Chat ----------
function clearEmptyState() {
  if (emptyStateEl) emptyStateEl.remove();
  const currentEmpty = document.getElementById("emptyState");
  if (currentEmpty) currentEmpty.remove();
}

function addUserMessage(text) {
  clearEmptyState();
  const el = document.createElement("div");
  el.className = "msg-query";
  el.textContent = text;
  messagesEl.appendChild(el);
  scrollToBottom();
}

function addThinkingIndicator() {
  const el = document.createElement("div");
  el.className = "thinking";
  el.id = "thinkingIndicator";
  el.innerHTML = `<span class="thinking-dots"><span></span><span></span><span></span></span> Retrieving evidence, reasoning, verifying…`;
  messagesEl.appendChild(el);
  scrollToBottom();
}

function removeThinkingIndicator() {
  const el = document.getElementById("thinkingIndicator");
  if (el) el.remove();
}

function verdictClass(verdict) {
  const v = (verdict || "").toLowerCase();
  if (v === "supported") return "supported";
  if (v === "unsupported") return "unsupported";
  if (v === "partially_supported") return "partially_supported";
  return "no_evidence";
}

function addAnswerMessage(result) {
  clearEmptyState();
  const wrap = document.createElement("div");
  wrap.className = "msg-answer";

  const answerP = document.createElement("div");
  answerP.className = "answer-text";
  answerP.textContent = result.answer;
  wrap.appendChild(answerP);

  const verification = result.verification || {};
  const tag = document.createElement("div");
  tag.className = `verdict-tag ${verdictClass(verification.verdict)}`;
  tag.textContent = verification.verdict || "UNKNOWN";
  wrap.appendChild(tag);

  if (verification.reason) {
    const reasonP = document.createElement("div");
    reasonP.className = "verdict-reason";
    reasonP.textContent = verification.reason;
    wrap.appendChild(reasonP);
  }

  if (result.citations && result.citations.length) {
    const citationsWrap = document.createElement("div");
    citationsWrap.className = "citations";
    result.citations.forEach((c) => {
      const citeEl = document.createElement("div");
      citeEl.className = "citation";
      citeEl.innerHTML = `<span class="citation-source">${escapeHtml(c.source)}</span> · page ${c.page}
        <span class="citation-excerpt">"${escapeHtml(c.excerpt)}"</span>`;
      citationsWrap.appendChild(citeEl);
    });
    wrap.appendChild(citationsWrap);
  }

  messagesEl.appendChild(wrap);
  scrollToBottom();
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

async function askQuestion(query) {
  addUserMessage(query);
  addThinkingIndicator();
  sendBtn.disabled = true;

  try {
    const res = await fetch(`${API_BASE}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, session_id: currentSessionId }),
    });
    if (!res.ok) throw new Error(`Server returned ${res.status}`);
    const data = await res.json();
    removeThinkingIndicator();
    addAnswerMessage(data);
  } catch (err) {
    removeThinkingIndicator();
    addAnswerMessage({
      answer: `Could not reach the assistant: ${err.message}`,
      verification: { verdict: "NO_EVIDENCE", reason: "" },
      citations: [],
    });
  } finally {
    sendBtn.disabled = false;
    // Refresh the sidebar list so a brand-new chat (or an updated one)
    // shows up immediately after its first exchange completes.
    if (document.getElementById("tab-chats").classList.contains("active")) {
      loadSessionList();
    }
  }
}

composerEl.addEventListener("submit", (e) => {
  e.preventDefault();
  const query = queryInput.value.trim();
  if (!query) return;
  queryInput.value = "";
  askQuestion(query);
});

// ---------- Upload ----------
dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  if (e.dataTransfer.files.length) showFilePreview(e.dataTransfer.files[0], null);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) showFilePreview(fileInput.files[0], fileInput);
});

// Composer attach button — a second entry point into the same upload flow.
composerAttachBtn.addEventListener("click", () => composerFileInput.click());
composerFileInput.addEventListener("change", () => {
  if (composerFileInput.files.length) showFilePreview(composerFileInput.files[0], composerFileInput);
});

// Shows a small chip with the file name above the composer the moment a
// file is selected (via drag-drop, sidebar picker, or the paperclip
// button), then uploads it and updates the chip's state in place.
function showFilePreview(file, sourceInput) {
  const chip = document.createElement("div");
  chip.className = "file-chip";
  chip.innerHTML = `
    <svg class="file-chip-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/>
      <path d="M14 2v6h6" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
    <span class="file-chip-name">${escapeHtml(file.name)}</span>
    <button type="button" class="file-chip-remove" title="Remove">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 6L6 18M6 6l12 12" stroke-linecap="round"/></svg>
    </button>
  `;
  filePreviewRow.appendChild(chip);

  chip.querySelector(".file-chip-remove").addEventListener("click", () => {
    chip.remove();
    if (sourceInput) sourceInput.value = ""; // reset so re-selecting the same file still fires "change"
  });

  uploadFileWithChip(file, chip, sourceInput);
}

async function uploadFileWithChip(file, chip, sourceInput) {
  chip.classList.add("uploading");

  // keep the sidebar's upload log in sync too, same as before
  const li = document.createElement("li");
  li.textContent = `Uploading ${file.name}…`;
  uploadLog.prepend(li);

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch(`${API_BASE}/upload`, {
      method: "POST",
      headers: { "X-Username": "admin", "X-Role": "admin" },
      body: formData,
    });
    if (!res.ok) throw new Error(`Server returned ${res.status}`);

    chip.classList.remove("uploading");
    chip.classList.add("done");
    li.textContent = `${file.name} — saved, watcher will index it`;
    li.className = "ok";

    setTimeout(loadKgSummary, 7000);
    setTimeout(() => chip.remove(), 4000); // auto-clear the chip once confirmed
  } catch (err) {
    chip.classList.remove("uploading");
    chip.classList.add("error");
    chip.querySelector(".file-chip-name").title = `Upload failed: ${err.message}`;
    li.textContent = `${file.name} — upload failed: ${err.message}`;
    li.className = "err";
  } finally {
    if (sourceInput) sourceInput.value = "";
  }
}

refreshKgBtn.addEventListener("click", loadKgSummary);

// ---------- Init ----------
checkHealth();
loadKgSummary();
loadSessionList();
setInterval(checkHealth, 15000);
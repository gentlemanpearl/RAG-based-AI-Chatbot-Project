/**
 * AGENTIC AI RAG CHATBOT - FRONTEND APPLICATION LOGIC
 * Connects with FastAPI /chat and /health endpoints.
 */

// Determine base API endpoint
const API_BASE = window.location.origin.includes("http") ? window.location.origin : "http://127.0.0.1:8000";

// DOM Elements
const queryInput = document.getElementById("queryInput");
const chatForm = document.getElementById("chatForm");
const messagesContainer = document.getElementById("messagesContainer");
const welcomeHero = document.getElementById("welcomeHero");
const chatScrollArea = document.getElementById("chatScrollArea");
const typingIndicator = document.getElementById("typingIndicator");
const sendBtn = document.getElementById("sendBtn");
const btnClearChat = document.getElementById("btnClearChat");
const btnExportChat = document.getElementById("btnExportChat");
const btnToggleSide = document.getElementById("btnToggleSide");
const evidencePanel = document.getElementById("evidencePanel");

// Telemetry & Evidence Elements
const meterFill = document.getElementById("meterFill");
const meterVal = document.getElementById("meterVal");
const groundingStatusBadge = document.getElementById("groundingStatusBadge");
const statLatency = document.getElementById("statLatency");
const statChunks = document.getElementById("statChunks");
const chunksCountBadge = document.getElementById("chunksCountBadge");
const chunksList = document.getElementById("chunksList");
const citationBox = document.getElementById("citationBox");
const citationText = document.getElementById("citationText");
const rawJsonCode = document.getElementById("rawJsonCode");
const btnCopyJson = document.getElementById("btnCopyJson");
const backendStatusText = document.getElementById("backendStatusText");
const statusPulseDot = document.getElementById("statusPulseDot");

// Tabs
const tabBtnChunks = document.getElementById("tabBtnChunks");
const tabBtnJson = document.getElementById("tabBtnJson");
const chunksTab = document.getElementById("chunksTab");
const jsonTab = document.getElementById("jsonTab");

// State
let conversationHistory = [];
let latestTelemetry = null;

// Initialize
document.addEventListener("DOMContentLoaded", () => {
  checkBackendHealth();
  setupEventListeners();
  autoResizeTextarea();
});

// 1. Health Check
async function checkBackendHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`, { method: "GET" });
    if (res.ok) {
      const data = await res.json();
      backendStatusText.textContent = "Online • LangGraph RAG";
      statusPulseDot.style.backgroundColor = "#10B981";
      statusPulseDot.style.boxShadow = "0 0 8px #10B981";
    } else {
      backendStatusText.textContent = "API Error";
      statusPulseDot.style.backgroundColor = "#F59E0B";
    }
  } catch (err) {
    backendStatusText.textContent = "Local Server";
    statusPulseDot.style.backgroundColor = "#06B6D4";
  }
}

// 2. Setup Event Listeners
function setupEventListeners() {
  // Chat form submit
  chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const query = queryInput.value.trim();
    if (!query) return;
    submitQuery(query);
  });

  // Enter to send, Shift+Enter for newline
  queryInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      chatForm.dispatchEvent(new Event("submit"));
    }
  });

  // Auto-resize textarea on input
  queryInput.addEventListener("input", autoResizeTextarea);

  // Quick benchmark query buttons
  document.querySelectorAll(".prompt-chip").forEach((btn) => {
    btn.addEventListener("click", () => {
      const query = btn.getAttribute("data-query");
      if (query) submitQuery(query);
    });
  });

  // Clear chat
  btnClearChat.addEventListener("click", () => {
    conversationHistory = [];
    messagesContainer.innerHTML = "";
    welcomeHero.style.display = "block";
    resetTelemetry();
  });

  // Export conversation
  btnExportChat.addEventListener("click", exportConversation);

  // Toggle side drawer on mobile
  if (btnToggleSide) {
    btnToggleSide.addEventListener("click", () => {
      evidencePanel.classList.toggle("drawer-open");
    });
  }

  // Panel Tabs
  tabBtnChunks.addEventListener("click", () => {
    tabBtnChunks.classList.add("active");
    tabBtnJson.classList.remove("active");
    chunksTab.classList.add("active");
    jsonTab.classList.remove("active");
  });

  tabBtnJson.addEventListener("click", () => {
    tabBtnJson.classList.add("active");
    tabBtnChunks.classList.remove("active");
    jsonTab.classList.add("active");
    chunksTab.classList.remove("active");
  });

  // Copy JSON button
  btnCopyJson.addEventListener("click", () => {
    if (!latestTelemetry) return;
    navigator.clipboard.writeText(JSON.stringify(latestTelemetry, null, 2));
    btnCopyJson.textContent = "Copied!";
    setTimeout(() => (btnCopyJson.textContent = "Copy JSON"), 2000);
  });
}

function autoResizeTextarea() {
  queryInput.style.height = "auto";
  queryInput.style.height = Math.min(queryInput.scrollHeight, 120) + "px";
}

// 3. Submit Query Flow
async function submitQuery(queryText) {
  // Hide welcome hero on first message
  welcomeHero.style.display = "none";
  queryInput.value = "";
  autoResizeTextarea();

  // Add User Message
  appendMessage("user", queryText);

  // Show typing indicator
  typingIndicator.style.display = "flex";
  scrollToBottom();
  setLoadingState(true);

  const startTime = performance.now();

  try {
    const response = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: queryText }),
    });

    const elapsed = Math.round(performance.now() - startTime);

    if (!response.ok) {
      throw new Error(`Server returned HTTP ${response.status}`);
    }

    const data = await response.json();
    if (!data.latency_ms) data.latency_ms = elapsed;

    latestTelemetry = data;

    // Hide typing indicator
    typingIndicator.style.display = "none";

    // Append Assistant Message
    appendMessage("assistant", data.answer, data.refusal, data);

    // Update Telemetry & Chunks Inspector
    updateTelemetry(data);
  } catch (error) {
    typingIndicator.style.display = "none";
    appendMessage(
      "assistant",
      `⚠️ Could not connect to API (${error.message}). Ensure the backend server is running on http://127.0.0.1:8000.`,
      true
    );
  } finally {
    setLoadingState(false);
    scrollToBottom();
  }
}

// 4. Render Message Bubble
function appendMessage(role, text, isRefusal = false, payload = null) {
  const row = document.createElement("div");
  row.className = `message-row ${role} ${isRefusal ? "refused" : ""}`;

  const avatar = document.createElement("div");
  avatar.className = "msg-avatar";
  if (role === "user") {
    avatar.innerHTML = `<div class="user-avatar-pill">U</div>`;
  } else {
    avatar.innerHTML = `<img src="assets/logo.jpg" alt="Agentic AI" />`;
  }

  const bubbleWrap = document.createElement("div");
  bubbleWrap.className = "msg-bubble-wrap";

  const timeStr = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  const meta = document.createElement("div");
  meta.className = "msg-meta";
  meta.textContent = role === "user" ? `You • ${timeStr}` : `Agentic AI • ${timeStr}`;

  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";

  // Parse markdown
  bubble.innerHTML = formatMarkdown(text);

  bubbleWrap.appendChild(meta);
  bubbleWrap.appendChild(bubble);

  // Message Actions for assistant
  if (role === "assistant") {
    const actions = document.createElement("div");
    actions.className = "msg-actions";
    
    // Copy button
    const copyBtn = document.createElement("button");
    copyBtn.className = "msg-action-btn";
    copyBtn.innerHTML = `📋 <span>Copy</span>`;
    copyBtn.addEventListener("click", () => {
      navigator.clipboard.writeText(text);
      copyBtn.innerHTML = `✅ <span>Copied</span>`;
      setTimeout(() => (copyBtn.innerHTML = `📋 <span>Copy</span>`), 2000);
    });

    // Speak button (TTS)
    const speakBtn = document.createElement("button");
    speakBtn.className = "msg-action-btn";
    speakBtn.innerHTML = `🔊 <span>Listen</span>`;
    speakBtn.addEventListener("click", () => {
      speakText(text);
    });

    actions.appendChild(copyBtn);
    if ("speechSynthesis" in window) actions.appendChild(speakBtn);
    bubbleWrap.appendChild(actions);
  }

  row.appendChild(avatar);
  row.appendChild(bubbleWrap);

  messagesContainer.appendChild(row);

  conversationHistory.push({ role, text, timestamp: timeStr, payload });
}

// 5. Update Telemetry & Evidence Inspector
function updateTelemetry(data) {
  const score = data.confidence_score !== undefined ? data.confidence_score : 0.0;
  const isRefusal = data.refusal || false;
  const chunks = data.retrieved_chunks || [];
  const latency = data.latency_ms || 0;
  const citation = data.citation_notes || "";

  // 1. Radial Meter Animation
  const percent = Math.round(score * 100);
  meterVal.textContent = isRefusal ? "0%" : `${percent}%`;

  // Circumference = 2 * PI * 32 ~= 201
  const circumference = 201;
  const offset = isRefusal ? circumference : circumference - (percent / 100) * circumference;
  meterFill.style.strokeDashoffset = offset;

  if (isRefusal) {
    meterFill.style.stroke = "var(--accent-rose)";
    groundingStatusBadge.textContent = "Refused (Out-of-Domain)";
    groundingStatusBadge.style.color = "var(--accent-rose)";
  } else if (score >= 0.8) {
    meterFill.style.stroke = "var(--accent-emerald)";
    groundingStatusBadge.textContent = "High Grounded Support";
    groundingStatusBadge.style.color = "#34D399";
  } else {
    meterFill.style.stroke = "var(--accent-amber)";
    groundingStatusBadge.textContent = "Moderate Support";
    groundingStatusBadge.style.color = "var(--accent-amber)";
  }

  // 2. Stats
  statLatency.textContent = `${latency} ms`;
  statChunks.textContent = chunks.length;
  chunksCountBadge.textContent = `${chunks.length} Chunks`;

  // 3. Citations
  if (citation) {
    citationBox.style.display = "flex";
    citationText.textContent = citation;
  } else {
    citationBox.style.display = "none";
  }

  // 4. Chunks List
  chunksList.innerHTML = "";
  if (chunks.length === 0) {
    chunksList.innerHTML = `<div class="empty-chunks-placeholder"><p>No context chunks returned.</p></div>`;
  } else {
    chunks.forEach((c, idx) => {
      const card = document.createElement("div");
      card.className = "chunk-card";

      const pageText = c.page ? `Page ${c.page}` : "eBook";
      const scoreVal = c.relevance_score !== undefined ? c.relevance_score.toFixed(3) : "N/A";

      card.innerHTML = `
        <div class="chunk-card-header">
          <span class="chunk-page-pill">${pageText}</span>
          <span class="chunk-score-pill">Score: ${scoreVal}</span>
        </div>
        <div class="chunk-text" id="chunkText_${idx}">${escapeHtml(c.content)}</div>
        <div class="chunk-card-footer">
          <span>Source: ${c.source || "Ebook-Agentic-AI.pdf"}</span>
          <button class="btn-toggle-snippet" onclick="toggleSnippet(${idx})">Expand</button>
        </div>
      `;
      chunksList.appendChild(card);
    });
  }

  // 5. Raw JSON
  rawJsonCode.textContent = JSON.stringify(data, null, 2);
}

function resetTelemetry() {
  meterVal.textContent = "--%";
  meterFill.style.strokeDashoffset = "201";
  meterFill.style.stroke = "var(--accent-cyan)";
  groundingStatusBadge.textContent = "Awaiting Query";
  groundingStatusBadge.style.color = "var(--text-secondary)";
  statLatency.textContent = "-- ms";
  chunksCountBadge.textContent = "0 Chunks";
  citationBox.style.display = "none";
  chunksList.innerHTML = `
    <div class="empty-chunks-placeholder">
      <div class="radar-scan"></div>
      <p>Execute a query to inspect the top-k retrieved semantic chunks from Pinecone with cosine relevance and page numbers.</p>
    </div>
  `;
  rawJsonCode.textContent = "// JSON telemetry will appear here after query execution";
}

// 6. Helpers
window.toggleSnippet = function (idx) {
  const el = document.getElementById(`chunkText_${idx}`);
  if (el) {
    const card = el.closest(".chunk-card");
    card.classList.toggle("expanded");
    const btn = card.querySelector(".btn-toggle-snippet");
    btn.textContent = card.classList.contains("expanded") ? "Collapse" : "Expand";
  }
};

function scrollToBottom() {
  chatScrollArea.scrollTop = chatScrollArea.scrollHeight;
}

function setLoadingState(loading) {
  sendBtn.disabled = loading;
  queryInput.disabled = loading;
}

function formatMarkdown(text) {
  if (!text) return "";
  let html = escapeHtml(text);
  
  // Bold **text**
  html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
  
  // Lists
  html = html.replace(/^\s*[-•]\s+(.*)$/gm, "<li>$1</li>");
  html = html.replace(/(<li>.*<\/li>)/gs, "<ul>$1</ul>");

  // Newlines into paragraphs
  html = html.split("\n\n").map(p => `<p>${p.trim()}</p>`).join("");
  return html;
}

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function speakText(text) {
  if (!("speechSynthesis" in window)) return;
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text.replace(/[*_#]/g, ""));
  utterance.rate = 1.0;
  window.speechSynthesis.speak(utterance);
}

function exportConversation() {
  if (conversationHistory.length === 0) {
    alert("No messages to export.");
    return;
  }
  const blob = new Blob([JSON.stringify(conversationHistory, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `agentic-rag-chat-${new Date().toISOString().slice(0, 10)}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

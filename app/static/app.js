// ── Global tooltip ──
(function() {
  const tip = document.createElement('div');
  tip.id = 'global-tip';
  document.body.appendChild(tip);
  document.addEventListener('mouseover', function(e) {
    const el = e.target.closest('[data-tooltip]');
    if (!el) return;
    tip.textContent = el.getAttribute('data-tooltip');
    const r = el.getBoundingClientRect();
    let left = r.right - 230;
    let top = r.bottom + 8;
    left = Math.max(8, Math.min(left, window.innerWidth - 238));
    top  = Math.max(8, Math.min(top,  window.innerHeight - 120));
    tip.style.left = left + 'px';
    tip.style.top  = top  + 'px';
    tip.style.opacity = '1';
  });
  document.addEventListener('mouseout', function(e) {
    if (e.target.closest('[data-tooltip]')) tip.style.opacity = '0';
  });
})();

// ── State ──
let currentSessionId = localStorage.getItem('plasmid_session_id') || null;
let sessions = [];
let isStreaming = false;
let abortController = null;
let _userLibraryAvailable = false;

async function _checkUserLibrary() {
  try {
    const r = await fetch('/api/config/user-library');
    const d = await r.json();
    _userLibraryAvailable = d.available || false;
    const btn = document.getElementById('import-lib-btn');
    if (btn) btn.style.display = _userLibraryAvailable ? '' : 'none';
  } catch(e) { _userLibraryAvailable = false; }
}

// ── Token indicator ──
function updateTokenIndicator(inputTokens, contextWindow) {
  const indicator = document.getElementById('token-indicator');
  const bar = document.getElementById('token-bar');
  const label = document.getElementById('token-label');
  if (!indicator || !bar || !label) return;
  const pct = Math.min(inputTokens / contextWindow, 1);
  const remaining = contextWindow - inputTokens;
  const remainingK = remaining >= 1000
    ? (remaining / 1000).toFixed(0) + 'k'
    : remaining.toString();
  bar.style.width = (pct * 100).toFixed(1) + '%';
  bar.className = 'token-bar-fill' + (pct >= 0.9 ? ' alert' : pct >= 0.7 ? ' warn' : '');
  label.textContent = remainingK + ' context window tokens left';
  indicator.classList.add('visible');
}

function saveSessionId(id) {
  currentSessionId = id;
  if (id) {
    localStorage.setItem('plasmid_session_id', id);
  } else {
    localStorage.removeItem('plasmid_session_id');
  }
}

// ── DOM refs ──
const messagesEl = document.getElementById('messages');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('send-btn');
const stopBtn = document.getElementById('stop-btn');
const modelSelect = document.getElementById('model-select');
const sidebarEl = document.getElementById('sidebar');
const sessionsListEl = document.getElementById('sessions-list');
const reopenBtn = document.getElementById('sidebar-reopen-btn');
const healthBadge = document.getElementById('health-badge');
const healthText = document.getElementById('health-text');

// ── Helpers ──
function escapeHtml(text) {
  const d = document.createElement('div');
  d.textContent = text;
  return d.innerHTML;
}

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 200) + 'px';
}

function scrollToBottom() {
  // Only auto-scroll if we're viewing the session that's streaming
  if (streamingSessionId && currentSessionId !== streamingSessionId) return;
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

// ── Health check ──
async function checkHealth() {
  try {
    const r = await fetch('/api/health', { signal: AbortSignal.timeout(3000) });
    const ok = r.ok;
    healthBadge.className = 'health-badge ' + (ok ? 'online' : 'offline');
    healthText.textContent = ok ? 'Agent Online' : 'Agent Offline';
  } catch {
    healthBadge.className = 'health-badge offline';
    healthText.textContent = 'Agent Offline';
  }
}

// ── Sessions ──
async function loadSessions() {
  try {
    const r = await fetch('/api/sessions');
    sessions = await r.json();
    renderSessions();
  } catch {}
}

function renderSessions() {
  if (sessions.length === 0) {
    sessionsListEl.innerHTML = '<p class="no-sessions">No conversations yet</p>';
    return;
  }
  sessionsListEl.innerHTML = sessions.map(function(s) {
    const active = s.session_id === currentSessionId ? ' active' : '';
    const name = escapeHtml((s.first_message || 'New conversation').slice(0, 40));
    const batchIcon = s.batch_job_id
      ? '<svg width="11" height="11" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24" style="flex-shrink:0;opacity:0.6"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/></svg>'
      : '';
    return '<div class="session-item' + active + '" onclick="selectSession(\'' + s.session_id + '\')">' +
      '<span class="session-name" style="display:flex;align-items:center;gap:5px;">' + batchIcon + name + '</span>' +
      '<button class="delete-btn" onclick="event.stopPropagation(); deleteSessionById(\'' + s.session_id + '\')" title="Delete">' +
        '<svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24">' +
          '<path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/>' +
        '</svg>' +
      '</button>' +
    '</div>';
  }).join('');
}

async function selectSession(sessionId) {
  // If streaming, detach from the SSE connection but let the backend keep running
  if (isStreaming) {
    _detachStream();
    // Reset streaming UI state
    isStreaming = false;
    abortController = null;
    streamingInner = null;
    streamingSessionId = null;
    sendBtn.style.display = 'flex';
    stopBtn.style.display = 'none';
    inputEl.disabled = false;
  }

  // Pause DOM-update polling for the session we're leaving (batch keeps running backend)
  if (currentSessionId && _batchPollTimers[currentSessionId]) {
    clearInterval(_batchPollTimers[currentSessionId]);
    delete _batchPollTimers[currentSessionId];
  }

  saveSessionId(sessionId);
  renderSessions();

  try {
    const r = await fetch('/api/sessions/' + sessionId + '/messages');
    const msgs = await r.json();
    // Guard: if user switched to another session while fetch was in flight, discard
    if (currentSessionId !== sessionId) return;
    renderStoredMessages(msgs);
    // If the agent is still running in the background, reconnect to the live
    // stream so the user sees the response streaming in real time.
    _reconnectToStream(sessionId);
  } catch {
    // Don't clear messages on fetch failure (e.g., during server reload)
    // — leave the current display intact rather than showing empty state
  }
}

async function _reconnectToStream(sessionId) {
  if (isStreaming) return;

  // Try to open the live replay stream. Returns 404 if the run already ended.
  abortController = new AbortController();
  let resp;
  try {
    resp = await fetch('/api/sessions/' + sessionId + '/stream', { signal: abortController.signal });
  } catch (err) {
    abortController = null;
    if (err.name !== 'AbortError') {
      // Network error — fall back to a one-shot message reload
      try {
        const msgs = await fetch('/api/sessions/' + sessionId + '/messages').then(function(r) { return r.json(); });
        if (currentSessionId === sessionId) renderStoredMessages(msgs);
      } catch {}
    }
    return;
  }

  if (!resp.ok) {
    abortController = null;
    // Run already finished — just reload stored messages
    try {
      const msgs = await fetch('/api/sessions/' + sessionId + '/messages').then(function(r) { return r.json(); });
      if (currentSessionId === sessionId) renderStoredMessages(msgs);
    } catch {}
    return;
  }

  // Run is in progress — set up streaming UI and replay/continue the event stream
  isStreaming = true;
  streamingSessionId = sessionId;
  streamingInner = messagesEl.querySelector('.messages-inner');
  if (!streamingInner) {
    streamingInner = document.createElement('div');
    streamingInner.className = 'messages-inner';
    messagesEl.innerHTML = '';
    messagesEl.appendChild(streamingInner);
  }
  sendBtn.style.display = 'none';
  stopBtn.style.display = 'flex';
  inputEl.disabled = true;
  showPendingCursor();

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split('\n\n');
      buffer = parts.pop();
      let streamDone = false;
      for (const part of parts) {
        const trimmed = part.trim();
        if (!trimmed.startsWith('data: ')) continue;
        const jsonStr = trimmed.slice(6);
        if (!jsonStr) continue;
        let event;
        try { event = JSON.parse(jsonStr); } catch { continue; }
        switch (event.type) {
          case 'thinking_start': clearPendingCursor(); startThinkingBlock(); break;
          case 'thinking_delta': appendThinkingDelta(event.content); break;
          case 'thinking_end': endThinkingBlock(); break;
          case 'text_start': clearPendingCursor(); flushTextBuffer(); startTextBlock(); break;
          case 'text_delta': bufferTextDelta(event.content); break;
          case 'text_end': endTextBlock(); break;
          case 'tool_use_start': clearPendingCursor(); startToolBlock(event.tool); break;
          case 'tool_result': finishToolBlock(event.tool, event.input || {}, event.content, event.download_content, event.download_filename); break;
          case 'plot_data': addPlasmidPlot(event.plot_json); break;
          case 'token_usage': updateTokenIndicator(event.input_tokens, event.context_window); break;
          case 'error': clearPendingCursor(); startTextBlock(); appendTextDelta('Error: ' + event.content); endTextBlock(); break;
          case 'bulk_design_rows':
            streamDone = true;
            if (currentToolId) {
              var _pulse = document.getElementById(currentToolId + '-pulse');
              if (_pulse) _pulse.remove();
              var _body = document.getElementById(currentToolId + '-body');
              if (_body) _body.innerHTML = '<div class="section"><div class="label">Result</div>Submitted ' + (event.rows || []).length + ' design(s) to bulk planner.</div>';
              currentToolId = null;
            }
            requestBulkPlanFromRows(event.rows || [], modelSelect.value);
            break;
          case 'bulk_designs_registered':
            showBulkPreviewModelCard(event);
            break;
          case 'bulk_preview_export':
            _bulkPreviewExports.push({filename: event.filename, content: event.content});
            break;
          case 'bulk_preview_complete':
            // Agent finished construct 1 — close the progress card, show approval card.
            // Do NOT set streamDone; the agent's turn ends naturally with 'done'.
            { var _mc = document.getElementById('bulk-preview-model-card'); if (_mc) _mc.remove(); }
            showBulkPreviewApprovalCard(event);
            break;
          case 'done': streamDone = true; break;
        }
        if (streamDone) break;
      }
      if (streamDone) break;
    }
  } catch (err) {
    if (err.name !== 'AbortError') {
      clearPendingCursor(); startTextBlock(); appendTextDelta('Connection error: ' + err.message); endTextBlock();
    }
  }

  clearPendingCursor();
  isStreaming = false;
  abortController = null;
  streamingInner = null;
  streamingSessionId = null;
  sendBtn.style.display = 'flex';
  stopBtn.style.display = 'none';
  inputEl.disabled = false;
  const cursor = messagesEl.querySelector('.streaming-cursor');
  if (cursor) cursor.remove();
  loadUserLibrary();

  // Reload stored messages so the view is stable (not dependent on DOM built during streaming)
  if (currentSessionId === sessionId) {
    try {
      const msgs = await fetch('/api/sessions/' + sessionId + '/messages').then(function(r) { return r.json(); });
      if (currentSessionId === sessionId) renderStoredMessages(msgs);
    } catch {}
  }
}

function renderStoredBlock(block, container) {
  const uid = 'stored-' + Date.now() + '-' + Math.random().toString(36).slice(2,6);
  if (block.type === 'thinking') {
    const wc = (block.content || '').trim().split(/\s+/).length;
    const div = document.createElement('div');
    div.className = 'thinking-block';
    div.innerHTML = '<div class="block-card">' +
      '<div class="block-header" onclick="toggleBlock(\'' + uid + '\')">' +
        '<svg class="block-icon" viewBox="0 0 24 24" stroke="var(--brand-fig)" fill="none" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">' +
          '<path d="M12 2a7 7 0 017 7c0 2.38-1.19 4.47-3 5.74V17a1 1 0 01-1 1h-6a1 1 0 01-1-1v-2.26C6.19 13.47 5 11.38 5 9a7 7 0 017-7zM9 21h6M10 21v-1h4v1"/>' +
        '</svg>' +
        '<span class="block-label">Thought process</span>' +
        '<span class="block-meta">' + wc + ' words</span>' +
        '<svg class="block-chevron" id="' + uid + '-chevron" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M9 18l6-6-6-6"/></svg>' +
      '</div>' +
      '<div class="block-body" id="' + uid + '-body">' + escapeHtml(block.content || '') + '</div>' +
    '</div>';
    container.appendChild(div);
  } else if (block.type === 'tool_use') {
    const div = document.createElement('div');
    div.className = 'tool-block';
    const inputStr = JSON.stringify(block.input || {}, null, 2);
    const bodyHtml = '<div class="section"><div class="label">Input</div>' + escapeHtml(inputStr) + '</div>' +
      '<div class="section"><div class="label">Result</div>' + escapeHtml(block.result || '') + '</div>';
    div.innerHTML = '<div class="block-card">' +
      '<div class="block-header" onclick="toggleBlock(\'' + uid + '\')">' +
        '<svg class="block-icon" viewBox="0 0 24 24" stroke="var(--brand-fig)" fill="none" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">' +
          '<path d="M14.7 6.3a1 1 0 000 1.4l1.6 1.6a1 1 0 001.4 0l3.77-3.77a6 6 0 01-7.94 7.94l-6.91 6.91a2.12 2.12 0 01-3-3l6.91-6.91a6 6 0 017.94-7.94l-3.76 3.76z"/>' +
        '</svg>' +
        '<span class="block-label">' + escapeHtml(block.name || 'tool') + '</span>' +
        '<svg class="block-chevron" id="' + uid + '-chevron" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M9 18l6-6-6-6"/></svg>' +
      '</div>' +
      '<div class="block-body" id="' + uid + '-body">' + bodyHtml + '</div>' +
    '</div>';
    container.appendChild(div);
    if (block.download_content && block.download_filename) {
      const isGb = block.name === 'export_construct' &&
          ['genbank', 'gb'].includes((block.input && block.input.output_format || '').toLowerCase());
      if (isGb) {
        addExportButtons(container, block.input || {}, block.download_content, block.download_filename);
      } else {
        addDownloadButton(container, block.download_content, block.download_filename);
      }
    }
  } else if (block.type === 'text') {
    const div = document.createElement('div');
    div.className = 'msg assistant';
    div.innerHTML = '<div class="msg-bubble-assistant">' + renderContent(block.content || '') + '</div>';
    makeTablesResizable(div);
    container.appendChild(div);
  }
}

function renderStoredMessages(msgs) {
  if (msgs.length === 0) {
    showWelcome();
    return;
  }
  // A trailing batch_session marker means this session has a bulk job attached.
  // Render any prior chat history first, then append the batch cards — don't discard it.
  var batchMeta = null;
  if (msgs.length && msgs[msgs.length - 1].type === 'batch_session') {
    batchMeta = msgs[msgs.length - 1];
    msgs = msgs.slice(0, -1);
  }
  if (msgs.length === 0 && batchMeta) {
    restoreBatchSession(batchMeta, false);
    return;
  }
  if (msgs.length > 0) {
    hideWelcome();
    const inner = document.createElement('div');
    inner.className = 'messages-inner';
    msgs.forEach(function(m) {
      if (m.role === 'user') {
        const div = document.createElement('div');
        div.className = 'msg user';
        const dateStr = m.timestamp ? new Date(m.timestamp * 1000).toLocaleDateString(undefined, {month:'short',day:'numeric',year:'numeric'}) : '';
        div.innerHTML = '<div><div class="msg-bubble-user">' + escapeHtml(m.content) + '</div>' + (dateStr ? '<div class="msg-date">' + dateStr + '</div>' : '') + '</div>';
        inner.appendChild(div);
      } else if (m.blocks && m.blocks.length > 0) {
        m.blocks.forEach(function(block) { renderStoredBlock(block, inner); });
      } else {
        const div = document.createElement('div');
        div.className = 'msg assistant';
        div.innerHTML = '<div class="msg-bubble-assistant">' + renderContent(m.content || '') + '</div>';
        makeTablesResizable(div);
        inner.appendChild(div);
      }
    });
    messagesEl.innerHTML = '';
    messagesEl.appendChild(inner);
  }
  if (batchMeta) {
    restoreBatchSession(batchMeta, msgs.length > 0);
  } else {
    scrollToBottom();
  }
}

async function restoreBatchSession(meta, keepExisting) {
  var jobId = meta.batch_job_id;
  var filename = meta.batch_filename || '';
  var model = meta.batch_model || '';
  var rowCount = meta.batch_row_count || 0;
  var sessionId = currentSessionId;

  // Clear stale content and set up a fresh container, unless we just rendered
  // this session's real chat history above and want to append after it.
  if (!keepExisting) messagesEl.innerHTML = '';

  try {
    const r = await fetch('/api/batch/' + jobId);
    const data = await r.json();
    if (currentSessionId !== sessionId) return;
    if (data.error) {
      var errHtml = '<div class="msg assistant"><div class="msg-bubble-assistant" style="color:var(--sand-400);font-size:13px;">Could not load batch results.</div></div>';
      if (keepExisting) {
        getInner().insertAdjacentHTML('beforeend', errHtml);
      } else {
        messagesEl.innerHTML = '<div class="messages-inner">' + errHtml + '</div>';
      }
      return;
    }

    // Render the batch label + placeholder cards, then immediately update with real state
    initBatchCards(jobId, rowCount, filename, model);
    updateBatchCards(jobId, data.rows);

    var anyRunning = data.rows && data.rows.some(function(r) {
      return r.status === 'running' || r.status === 'pending';
    });

    _batchSessions[sessionId] = jobId;

    if (data.status !== 'done' || anyRunning) {
      // Batch is still in progress — resume polling
      if (_batchPollTimers[sessionId]) clearInterval(_batchPollTimers[sessionId]);
      _batchPollTimers[sessionId] = setInterval(function() { pollBatchForSession(sessionId); }, 2000);
    } else {
      // Batch finished — show the download-all button if not already there
      var ctrlEl = document.getElementById('batch-ctrl-' + jobId);
      if (ctrlEl) ctrlEl.style.display = 'none';
      var labelEl = document.getElementById('batch-label-' + jobId);
      if (labelEl && !labelEl.querySelector('.batch-dl-all-btn')) {
        var bubble = labelEl.querySelector('.msg-bubble-assistant');
        if (bubble) {
          var wrap = document.createElement('div');
          wrap.className = 'dl-split-wrap batch-dl-all-btn';
          wrap.style.cssText = 'margin-top:10px;';
          var allMenuId = 'dlmenu-all-' + jobId;
          wrap.innerHTML =
            '<button class="download-btn" onclick="downloadAllBatch(\'' + jobId + '\')">' + _DL_SVG + ' Download All (.zip)</button>' +
            '<button class="dl-chevron-btn" onclick="toggleDlMenu(event,\'' + allMenuId + '\')" title="More options">' + _CHEV_DOWN_SVG + '</button>' +
            '<div class="dl-menu" id="' + allMenuId + '">' +
              '<button class="dl-menu-item" onclick="downloadAllBatch(\'' + jobId + '\')">' + _DL_SVG + ' Download All (.zip)</button>' +
              (_userLibraryAvailable ? '<button class="dl-menu-item" id="savall-local-' + jobId + '" onclick="event.stopPropagation();saveAllBatchToLocal(\'' + jobId + '\',document.getElementById(\'savall-local-' + jobId + '\'))">' +
                '<svg width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M3 15v4c0 1.1.9 2 2 2h14a2 2 0 002-2v-4M17 8l-5-5-5 5M12 3v12"/></svg> Save All to Local Library</button>' : '') +
              '<button class="dl-menu-item" id="savall-con-' + jobId + '" onclick="event.stopPropagation();saveAllBatchConstructs(\'' + jobId + '\',document.getElementById(\'savall-con-' + jobId + '\'))">' + _SAVE_SVG + ' Save All Constructs</button>' +
            '</div>';
          bubble.appendChild(document.createElement('br'));
          bubble.appendChild(wrap);
        }
      }
    }
  } catch(e) {
    var failHtml = '<div class="msg assistant"><div class="msg-bubble-assistant" style="color:var(--sand-400);font-size:13px;">Could not reach the server to load batch status.</div></div>';
    if (keepExisting) {
      getInner().insertAdjacentHTML('beforeend', failHtml);
    } else {
      messagesEl.innerHTML = '<div class="messages-inner">' + failHtml + '</div>';
    }
  }
}

async function deleteSessionById(sessionId) {
  try {
    await fetch('/api/sessions/' + sessionId, { method: 'DELETE' });
    if (currentSessionId === sessionId) {
      saveSessionId(null);
      showWelcome();
    }
    loadSessions();
  } catch {}
}

function newChat() {
  if (isStreaming) {
    _detachStream();
    isStreaming = false;
    abortController = null;
    streamingInner = null;
    streamingSessionId = null;
    sendBtn.style.display = 'flex';
    stopBtn.style.display = 'none';
    inputEl.disabled = false;
  }
  // Pause DOM polling for the session being left
  if (currentSessionId && _batchPollTimers[currentSessionId]) {
    clearInterval(_batchPollTimers[currentSessionId]);
    delete _batchPollTimers[currentSessionId];
  }
  saveSessionId(null);
  renderSessions();
  showWelcome();
  inputEl.focus();
}

function showWelcome() {
  messagesEl.innerHTML = '';
  const w = document.createElement('div');
  w.className = 'welcome';
  w.id = 'welcome';
  w.innerHTML = '<div>' +
    '<div class="welcome-icon">' +
      '<svg viewBox="0 0 24 24" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">' +
        '<path d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714a2.25 2.25 0 00.659 1.591L19 14.5M14.25 3.104c.251.023.501.05.75.082M19 14.5l-2.47 2.47a2.25 2.25 0 01-1.591.659H9.061a2.25 2.25 0 01-1.591-.659L5 14.5m14 0V17a2 2 0 01-2 2H7a2 2 0 01-2-2v-2.5"/>' +
      '</svg>' +
    '</div>' +
    '<h2>Design an expression plasmid</h2>' +
    '<p>Describe what you want to build. Claude will retrieve verified sequences,<br>' +
    'assemble your construct, validate it, and export the result.</p>' +
    '<p style="font-size:12px;color:var(--sand-300);margin-top:4px;">Drag &amp; drop a CSV file here to batch design multiple plasmids at once.</p>' +
    '<div class="examples">' +
      '<button onclick="sendExample(this)">Design an EGFP expression plasmid using pcDNA3.1(+)</button>' +
      '<button onclick="sendExample(this)">Put mCherry into a mammalian expression vector</button>' +
      '<button onclick="sendExample(this)">What backbones are available?</button>' +
      '<button onclick="sendExample(this)">Assemble tdTomato in pcDNA3.1(+) and export as GenBank</button>' +
    '</div>' +
  '</div>';
  messagesEl.appendChild(w);
}

function hideWelcome() {
  const w = document.getElementById('welcome');
  if (w) w.style.display = 'none';
}

// ── Sidebar toggle ──
function toggleSidebar() {
  sidebarEl.classList.toggle('collapsed');
  reopenBtn.classList.toggle('visible', sidebarEl.classList.contains('collapsed'));
}

// ── User Library Panel ──
const _ulEntries = {};   // id → full entry object

function _ulDetailRows(entry, isInsert) {
  const fields = isInsert ? [
    ['ID', entry.id],
    ['Category', entry.category],
    ['Enzyme', entry.assembly_enzyme],
    ['Overhangs', entry.overhang_l && entry.overhang_r ? entry.overhang_l + ' / ' + entry.overhang_r : null],
    ['Insert size', entry.insert_size_bp ? entry.insert_size_bp + ' bp' : null],
    ['Vector size', entry.size_bp ? entry.size_bp + ' bp' : null],
    ['Resistance', entry.bacterial_resistance],
    ['Description', entry.description],
  ] : [
    ['ID', entry.id],
    ['Company', entry.company],
    ['Enzyme', entry.assembly_enzyme],
    ['Overhangs 1', entry.overhang_left && entry.overhang_right ? entry.overhang_left + ' / ' + entry.overhang_right : null],
    ['Overhangs 2', entry.overhang_left_2 && entry.overhang_right_2 ? entry.overhang_left_2 + ' / ' + entry.overhang_right_2 : null],
    ['Next enzyme', entry.next_step_enzyme],
    ['E. coli', entry.ecoli_strain],
    ['Resistance', entry.bacterial_resistance],
    ['Mammalian', entry.mammalian_selection],
    ['Size', entry.size_bp ? entry.size_bp + ' bp' : null],
    ['Description', entry.description],
  ];
  return fields.filter(function(f) { return f[1]; }).map(function(f) {
    return '<div class="entry-detail-row"><span class="dk">' + escapeHtml(f[0]) + '</span><span class="dv">' + escapeHtml(String(f[1])) + '</span></div>';
  }).join('');
}

function _ulBuildEntries(items, isInsert) {
  return items.map(function(entry) {
    const eid = entry.id.replace(/[^a-zA-Z0-9_-]/g, '_');
    _ulEntries[eid] = {entry: entry, isInsert: isInsert};
    const meta = isInsert
      ? [entry.category, entry.assembly_enzyme, entry.insert_size_bp ? entry.insert_size_bp + ' bp' : null].filter(Boolean).join(' · ')
      : [entry.company, entry.assembly_enzyme, entry.bacterial_resistance].filter(Boolean).join(' · ');
    return '<div class="user-library-entry" id="ule-' + eid + '" onclick="toggleULEntry(\'' + eid + '\')">' +
      '<div class="entry-header">' +
        '<div><div class="entry-name">' + escapeHtml(entry.name || entry.id) + '</div>' +
        (meta ? '<div class="entry-meta">' + escapeHtml(meta) + '</div>' : '') + '</div>' +
        '<em class="entry-chevron">&#8964;</em>' +
      '</div>' +
      '<div class="entry-detail" id="uld-' + eid + '"></div>' +
    '</div>';
  }).join('');
}

async function loadUserLibrary() {
  try {
    const r = await fetch('/api/user-library');
    const data = await r.json();
    const panel = document.getElementById('user-library-panel');
    const hasContent = data.configured || (data.vendor_backbones && data.vendor_backbones.length);
    if (!hasContent) return;
    panel.style.display = '';
    const body = document.getElementById('user-library-body');
    let html = '';
    if (data.backbones && data.backbones.length) {
      html += '<div class="user-library-section"><div class="user-library-section-title">Backbones</div>' +
        _ulBuildEntries(data.backbones, false) + '</div>';
    }
    if (data.inserts && data.inserts.length) {
      html += '<div class="user-library-section"><div class="user-library-section-title">Inserts</div>' +
        _ulBuildEntries(data.inserts, true) + '</div>';
    }
    if (data.vendor_backbones && data.vendor_backbones.length) {
      html += '<div class="user-library-section"><div class="user-library-section-title">Vendor Backbones</div>' +
        _ulBuildEntries(data.vendor_backbones, false) + '</div>';
    }
    if (data.designed_constructs && data.designed_constructs.length) {
      html += '<div class="user-library-section"><div class="user-library-section-title">Designed Constructs</div>' +
        _ulBuildEntries(data.designed_constructs, false) + '</div>';
    }
    if (!html) html = '<div class="user-library-empty">No entries loaded.</div>';
    body.innerHTML = html;
  } catch {}
}

function toggleULEntry(eid) {
  const row = document.getElementById('ule-' + eid);
  const detail = document.getElementById('uld-' + eid);
  const isOpen = detail.classList.contains('open');
  if (!isOpen && !detail.innerHTML) {
    const rec = _ulEntries[eid];
    detail.innerHTML = _ulDetailRows(rec.entry, rec.isInsert);
  }
  detail.classList.toggle('open', !isOpen);
  row.classList.toggle('expanded', !isOpen);
}

function toggleUserLibrary() {
  const btn = document.getElementById('user-library-toggle');
  const body = document.getElementById('user-library-body');
  btn.classList.toggle('open');
  body.classList.toggle('open');
}

// ── Markdown rendering ──
// ── DNA sequence helpers ─────────────────────────────────────────────────────
const _CLIP_SVG = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>';
const _CHECK_SVG = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>';

function isDnaSeq(s) {
  // Strip whitespace, digits, hyphens, prime/apostrophe, brackets, parens, slashes, punctuation
  const clean = s.replace(/[\s\d\-'()\[\]\/\\.,;:*>]/g, '').toUpperCase();
  return clean.length >= 10 && /^[ACGTURYSWKMBDHVN]+$/.test(clean);
}

function mkCopyBtn(seq, extraClass) {
  return '<button class="seq-copy-btn' + (extraClass ? ' ' + extraClass : '') +
    '" data-seq="' + escapeHtml(seq) + '" onclick="copySeq(this.dataset.seq,this)" title="Copy sequence">' +
    _CLIP_SVG + '</button>';
}

function copySeq(seq, btn) {
  seq = seq.replace(/\s/g, '');
  navigator.clipboard.writeText(seq).then(function() {
    var orig = btn.innerHTML;
    btn.innerHTML = _CHECK_SVG;
    btn.classList.add('copied');
    setTimeout(function() { btn.innerHTML = orig; btn.classList.remove('copied'); }, 1500);
  });
}

function copyRaw(text, btn) {
  navigator.clipboard.writeText(text).then(function() {
    var orig = btn.innerHTML;
    btn.innerHTML = _CHECK_SVG;
    btn.classList.add('copied');
    setTimeout(function() { btn.innerHTML = orig; btn.classList.remove('copied'); }, 1500);
  });
}

function mkRawCopyBtn(text, extraClass) {
  return '<button class="seq-copy-btn' + (extraClass ? ' ' + extraClass : '') +
    '" data-raw="' + escapeHtml(text) + '" onclick="copyRaw(this.dataset.raw,this)" title="Copy">' +
    _CLIP_SVG + '</button>';
}

function stripMarkdown(text) {
  return text
    .replace(/\*\*(.+?)\*\*/g, '$1')
    .replace(/\*(.+?)\*/g, '$1')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/^#{1,6}\s+/, '');
}

function makeTablesResizable(root) {
  if (!root) return;
  root.querySelectorAll('table:not([data-resizable])').forEach(function(table) {
    table.setAttribute('data-resizable', '1');
    table.querySelectorAll('th').forEach(function(th) {
      var resizer = document.createElement('div');
      resizer.className = 'col-resizer';
      th.appendChild(resizer);
      var startX, startW;
      resizer.addEventListener('mousedown', function(e) {
        var allThs = table.querySelectorAll('th');
        allThs.forEach(function(t) { t.style.width = t.offsetWidth + 'px'; });
        table.style.tableLayout = 'fixed';
        table.style.width = table.offsetWidth + 'px';
        startX = e.pageX;
        startW = th.offsetWidth;
        function onMove(e) { th.style.width = Math.max(40, startW + e.pageX - startX) + 'px'; }
        function onUp() { document.removeEventListener('mousemove', onMove); document.removeEventListener('mouseup', onUp); }
        document.addEventListener('mousemove', onMove);
        document.addEventListener('mouseup', onUp);
        e.preventDefault();
      });
    });
  });
}

function renderSeqCodeBlock(rawCode) {
  var langMatch = rawCode.match(/^([a-zA-Z][a-zA-Z0-9_]*)\n/);
  var code = langMatch ? rawCode.slice(langMatch[0].length) : rawCode;
  var trimmed = code.trim();
  var lines = trimmed.split('\n').filter(function(l) { return l.trim(); });

  // FASTA format (lines starting with >)
  if (lines.length >= 1 && lines[0].charAt(0) === '>') {
    var entries = [], curName = null, curSeq = '';
    lines.forEach(function(l) {
      if (l.charAt(0) === '>') {
        if (curName !== null) entries.push({name: curName, seq: curSeq});
        curName = l.slice(1).trim(); curSeq = '';
      } else { curSeq += l.trim(); }
    });
    if (curName !== null) entries.push({name: curName, seq: curSeq});
    if (entries.length > 0 && entries.every(function(e) { return isDnaSeq(e.seq); })) {
      if (entries.length === 1) {
        return '<div class="seq-block"><pre><code>&gt;' + escapeHtml(entries[0].name) + '\n' +
          escapeHtml(entries[0].seq) + '</code></pre>' + mkCopyBtn(entries[0].seq, 'block-btn') + '</div>';
      }
      var allTsv = 'Name\tSequence\n' + entries.map(function(e) { return e.name + '\t' + e.seq; }).join('\n');
      var t = '<div class="seq-table"><table><thead><tr><th>Name</th><th>Sequence</th><th></th></tr></thead><tbody>';
      entries.forEach(function(e) {
        t += '<tr><td>' + escapeHtml(e.name) + '</td><td><code class="dna-seq">' +
          escapeHtml(e.seq) + '</code></td><td>' + mkCopyBtn(e.seq) + '</td></tr>';
      });
      return t + '</tbody></table><div class="tbl-copy-row">' + mkRawCopyBtn(allTsv) + '</div></div>';
    }
  }

  // Named sequences: "Label: SEQUENCE" or "Label = SEQUENCE"
  var namedRe = /^(.{1,50}?)\s*[:=]\s*([ACGTUacgtuRYSWKMBDHVNryswkmbdhvn]{10,})\s*$/;
  var namedMs = lines.map(function(l) { return l.match(namedRe); });
  if (lines.length >= 2 && namedMs.every(function(m) { return m; })) {
    var allTsv = 'Name\tSequence\n' + namedMs.map(function(m) { return m[1].trim() + '\t' + m[2]; }).join('\n');
    var t = '<div class="seq-table"><table><thead><tr><th>Name</th><th>Sequence</th><th></th></tr></thead><tbody>';
    namedMs.forEach(function(m) {
      t += '<tr><td>' + escapeHtml(m[1].trim()) + '</td><td><code class="dna-seq">' +
        escapeHtml(m[2]) + '</code></td><td>' + mkCopyBtn(m[2]) + '</td></tr>';
    });
    return t + '</tbody></table><div class="tbl-copy-row">' + mkRawCopyBtn(allTsv) + '</div></div>';
  }

  // Multiple bare sequences (one per line)
  if (lines.length >= 2 && lines.every(function(l) { return isDnaSeq(l.trim()); })) {
    var seqs = lines.map(function(l) { return l.trim(); });
    var allTsv = seqs.join('\n');
    var t = '<div class="seq-table"><table><thead><tr><th>#</th><th>Sequence</th><th></th></tr></thead><tbody>';
    seqs.forEach(function(s, i) {
      t += '<tr><td>' + (i+1) + '</td><td><code class="dna-seq">' +
        escapeHtml(s) + '</code></td><td>' + mkCopyBtn(s) + '</td></tr>';
    });
    return t + '</tbody></table><div class="tbl-copy-row">' + mkRawCopyBtn(allTsv) + '</div></div>';
  }

  // Single DNA sequence
  if (isDnaSeq(trimmed)) {
    return '<div class="seq-block"><pre><code>' + escapeHtml(trimmed) + '</code></pre>' +
      mkCopyBtn(trimmed.replace(/\s/g, ''), 'block-btn') + '</div>';
  }

  // Regular code block
  return '<div class="code-block-wrap"><pre><code>' + escapeHtml(code) + '</code></pre>' +
    mkRawCopyBtn(code, 'block-btn') + '</div>';
}

// Wrap bare DNA sequences in plain text (skips content already inside backticks)
function applyBareDnaInLine(h) {
  var parts = h.split(/(`[^`]*`)/);
  return parts.map(function(part, i) {
    if (i % 2 === 1) return part;
    return part.replace(/\b([ACGTUacgtu][ACGTUacgtuRYSWKMBDHVNryswkmbdhvn]{14,})\b/g, function(m) {
      return isDnaSeq(m) ? ('<code class="dna-seq">' + m + '</code>' + mkCopyBtn(m)) : m;
    });
  }).join('');
}

function inlineMarkdown(text) {
  let h = escapeHtml(text);
  h = applyBareDnaInLine(h);
  h = h.replace(/`([^`]+)`/g, function(match, code) {
    if (isDnaSeq(code)) return '<code class="dna-seq">' + code + '</code>' + mkCopyBtn(code);
    return '<code>' + code + '</code>';
  });
  h = h.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  return h;
}

function renderContent(text) {
  const codeBlocks = [];
  text = text.replace(/```([\s\S]*?)```/g, function(match, code) {
    codeBlocks.push(code);
    return '%%CODEBLOCK' + (codeBlocks.length - 1) + '%%';
  });

  const lines = text.split('\n');
  const outputParts = [];
  let i = 0;
  while (i < lines.length) {
    if (i + 1 < lines.length &&
        lines[i].trim().startsWith('|') &&
        /^\|[\s:]*-+[\s:]*/.test(lines[i + 1].trim())) {
      const headerCells = lines[i].trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map(function(c) { return c.trim(); });
      i += 2;
      const bodyRows = [];
      while (i < lines.length && lines[i].trim().startsWith('|')) {
        const cells = lines[i].trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map(function(c) { return c.trim(); });
        bodyRows.push(cells);
        i++;
      }
      var tsvRows = [headerCells.map(stripMarkdown).join('\t')].concat(bodyRows.map(function(row) { return row.map(stripMarkdown).join('\t'); }));
      var tblTsv = tsvRows.join('\n');
      let t = '<div class="seq-table"><table><thead><tr>';
      headerCells.forEach(function(c) { t += '<th>' + inlineMarkdown(c) + '</th>'; });
      t += '</tr></thead><tbody>';
      bodyRows.forEach(function(row) {
        t += '<tr>';
        row.forEach(function(c) {
          const stripped = c.trim();
          if (isDnaSeq(stripped)) {
            t += '<td><code class="dna-seq">' + escapeHtml(stripped) + '</code>' + mkCopyBtn(stripped) + '</td>';
          } else {
            t += '<td>' + inlineMarkdown(c) + '</td>';
          }
        });
        t += '</tr>';
      });
      t += '</tbody></table><div class="tbl-copy-row">' + mkRawCopyBtn(tblTsv) + '</div></div>';
      outputParts.push(t);
    } else {
      const trimmed = lines[i].trim();
      // Horizontal rule
      if (/^-{3,}$/.test(trimmed) || /^\*{3,}$/.test(trimmed)) {
        outputParts.push('<hr style="border:none;border-top:1px solid var(--sand-200);margin:12px 0">');
        i++;
        continue;
      }
      let h = escapeHtml(lines[i]);
      h = applyBareDnaInLine(h);
      h = h.replace(/`([^`]+)`/g, function(match, code) {
        if (isDnaSeq(code)) return '<code class="dna-seq">' + code + '</code>' + mkCopyBtn(code);
        return '<code>' + code + '</code>';
      });
      h = h.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
      h = h.replace(/^### (.+)$/, '<strong style="font-size:14px">$1</strong>');
      h = h.replace(/^## (.+)$/, '<strong style="font-size:15px">$1</strong>');
      h = h.replace(/^# (.+)$/, '<strong style="font-size:16px">$1</strong>');
      outputParts.push(h);
      i++;
    }
  }

  let html = outputParts.join('<br>\n');
  codeBlocks.forEach(function(code, idx) {
    html = html.replace('%%CODEBLOCK' + idx + '%%', renderSeqCodeBlock(code));
  });
  return html;
}

// ── Streaming blocks ──
let currentTextDiv = null;
let currentTextRaw = '';
let currentThinkingId = null;
let currentThinkingBody = null;
let currentToolId = null;
// Pinned reference to the .messages-inner container for the active stream.
// Ensures streaming writes go to the correct session even if the user
// clicks a different session in the sidebar mid-stream.
let streamingInner = null;
let streamingSessionId = null;

function getInner() {
  // While streaming, always write to the pinned container
  if (streamingInner) return streamingInner;
  let inner = messagesEl.querySelector('.messages-inner');
  if (!inner) {
    inner = document.createElement('div');
    inner.className = 'messages-inner';
    messagesEl.innerHTML = '';
    messagesEl.appendChild(inner);
  }
  return inner;
}

function toggleBlock(id) {
  const body = document.getElementById(id + '-body');
  const chevron = document.getElementById(id + '-chevron');
  if (body && chevron) {
    body.classList.toggle('open');
    chevron.classList.toggle('open');
  }
}

function startThinkingBlock() {
  currentThinkingId = 'think-' + Date.now();
  const div = document.createElement('div');
  div.className = 'thinking-block';
  div.innerHTML = '<div class="block-card">' +
    '<div class="block-header" onclick="toggleBlock(\'' + currentThinkingId + '\')">' +
      '<svg class="block-icon" viewBox="0 0 24 24" stroke="var(--brand-fig)" fill="none" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">' +
        '<path d="M12 2a7 7 0 017 7c0 2.38-1.19 4.47-3 5.74V17a1 1 0 01-1 1h-6a1 1 0 01-1-1v-2.26C6.19 13.47 5 11.38 5 9a7 7 0 017-7zM9 21h6M10 21v-1h4v1"/>' +
      '</svg>' +
      '<span class="block-label">Thinking...</span>' +
      '<span class="block-meta" id="' + currentThinkingId + '-meta"></span>' +
      '<svg class="block-chevron" id="' + currentThinkingId + '-chevron" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M9 18l6-6-6-6"/></svg>' +
    '</div>' +
    '<div class="block-body" id="' + currentThinkingId + '-body"></div>' +
  '</div>';
  getInner().appendChild(div);
  currentThinkingBody = document.getElementById(currentThinkingId + '-body');
  scrollToBottom();
}

function appendThinkingDelta(text) {
  if (currentThinkingBody) {
    currentThinkingBody.textContent += text;
    if (currentThinkingBody.classList.contains('open')) {
      currentThinkingBody.scrollTop = currentThinkingBody.scrollHeight;
    }
    scrollToBottom();
  }
}

function endThinkingBlock() {
  if (currentThinkingId) {
    const card = currentThinkingBody.closest('.block-card');
    const label = card.querySelector('.block-label');
    if (label) label.textContent = 'Thought process';
    const meta = document.getElementById(currentThinkingId + '-meta');
    if (meta && currentThinkingBody) {
      const wc = currentThinkingBody.textContent.trim().split(/\s+/).length;
      meta.textContent = wc + ' words';
    }
  }
  currentThinkingBody = null;
  currentThinkingId = null;
}

function startTextBlock() {
  const div = document.createElement('div');
  div.className = 'msg assistant';
  div.innerHTML = '<div class="msg-bubble-assistant"><span class="text-content"></span></div>';
  getInner().appendChild(div);
  currentTextDiv = div.querySelector('.text-content');
  currentTextRaw = '';
  scrollToBottom();
}

function appendTextDelta(text) {
  if (currentTextDiv) {
    currentTextRaw += text;
    currentTextDiv.innerHTML = renderContent(currentTextRaw);
    // Add streaming cursor
    let cursor = currentTextDiv.querySelector('.streaming-cursor');
    if (!cursor) {
      cursor = document.createElement('span');
      cursor.className = 'streaming-cursor';
      currentTextDiv.appendChild(cursor);
    }
    scrollToBottom();
  }
}

// ── Smooth streaming ──────────────────────────────────────────────────
// API text_delta events arrive in ~40-100 char bursts every ~400-500ms.
// Dumping each burst at once looks choppy. Buffer incoming chars and
// drain via requestAnimationFrame so text "types" smoothly. The drain
// rate adapts (1/8 of buffer per frame, clamped [2,12] chars) so it
// never lags far behind the model (~180 ch/s) but stays smooth.
let textBuffer = '';
let drainHandle = null;

function bufferTextDelta(text) {
  textBuffer += text;
  if (drainHandle === null) drainHandle = requestAnimationFrame(drainText);
}

function drainText() {
  drainHandle = null;
  if (textBuffer.length === 0) return;
  const n = Math.max(2, Math.min(12, Math.ceil(textBuffer.length / 8)));
  appendTextDelta(textBuffer.slice(0, n));
  textBuffer = textBuffer.slice(n);
  if (textBuffer.length > 0) drainHandle = requestAnimationFrame(drainText);
}

function flushTextBuffer() {
  if (drainHandle !== null) { cancelAnimationFrame(drainHandle); drainHandle = null; }
  if (textBuffer) { appendTextDelta(textBuffer); textBuffer = ''; }
}

// Show a blinking cursor immediately on send so the user sees activity
// during TTFT, before any text/thinking/tool event arrives.
let pendingCursorEl = null;
let pendingCursorTimer = null;
const SLOW_THRESHOLD_MS = 7000;
const SLOW_NOTE = 'Complex designs can take a minute or two — hang tight.';

const WORKING_LABELS = [
  'Pipetting…',
  'Running a gel…',
  'Miniprepping…',
  'Consulting the literature…',
  'Transforming bacteria…',
  'Checking the freezer…',
  'Growing colonies…',
  'Spinning down…',
  'Thawing reagents…',
  'Asking a grad student…',
  'Reading the manual…',
  'Autoclaving…',
  'Counting clones…',
  'Labeling tubes…',
  'Calibrating the pipette…',
  'Staring at the gel…',
  'Refilling tip boxes…',
  'Making competent cells…',
  'Waiting for the PCR…',
  'Pouring a plate…',
  'Streaking for singles…',
  'Checking the incubator…',
  'Ordering reagents…',
  'Waiting for the centrifuge…',
  'Defrosting the -20°C…',
  'Wiping down the bench…',
  'Preparing the buffer…',
  'Changing gloves…',
  'Signing the safety form…',
  'Finding the right tube…',
  'Checking the OD600…',
  'Setting up the water bath…',
  'Asking the PI…',
  'Asking a postdoc…',
  'Reprinting the label…',
  'Hunting for the protocol binder…',
  'Waiting for the autoclave…',
  'Mixing by pipetting up and down…',
  'Pipetting…',
  'Vortexing…',
  'Flash freezing…',
  'Filling out the order form…',
  'Checking if the kit expired…',
  'Making 10× buffer…',
  'Weighing out the powder…',
  'pH-ing the solution…',
  'Waiting for the gel to set…',
  'Realizing the gel ran backwards…',
  'Borrowing tips from the next lab…',
  'Searching for the marker…',
  'Checking the thermocycler program…',
  'Waiting for the overnight culture…',
  'Diluting the sample…',
  'Aliquoting…',
  'Topping off the liquid nitrogen…',
  'De-icing the freezer…',
  'Wiping down the hood…',
  'Running the Western…',
  'Blocking the membrane…',
  'Developing the blot…',
  'Staining with EtBr…',
  'Destaining the gel…',
  'Imaging the blot…',
  'Spinning the columns…',
  'Eluting the DNA…',
  'Measuring the absorbance…',
  'Plating the cells…',
  'Trypsinizing…',
  'Counting cells…',
  'Checking confluency…',
  'Changing the media…',
  'Spinning down the pellet…',
  'Resuspending in buffer…',
  'Snap freezing…',
  'Running the SDS-PAGE…',
  'Loading the samples…',
  'Casting the gel…',
  'Transferring to membrane…',
  'Probing with antibody…',
  'Washing the membrane…',
  'Exposing the film…',
  'Scraping the cells…',
  'Lysing the cells…',
  'Sonicating…',
  'Clarifying the lysate…',
  'Checking the protein concentration…',
  'Setting up the ligation…',
  'Running the digest…',
  'Gel extracting…',
  'Incubating on ice…',
  'Heat shocking…',
  'Recovering in SOC…',
  'Spreading the plates…',
  'Picking colonies…',
  'Inoculating the culture…',
  'Doing a colony PCR…',
  'Checking the growth curve…',
  'Inducing expression…',
  'Harvesting cells…',
  'Resuspending the pellet…',
  'Filtering the solution…',
  'Running the FPLC…',
  'Collecting fractions…',
  'Pooling the peaks…',
  'Concentrating the sample…',
  'Running a Bradford…',
  'Preparing the cryovials…',
  'Labeling the boxes…',
  'Checking the balance…',
];

function randomWorkingLabel() {
  return WORKING_LABELS[Math.floor(Math.random() * WORKING_LABELS.length)];
}

function showPendingCursor(label) {
  clearPendingCursor();
  const div = document.createElement('div');
  div.className = 'msg assistant';
  div.innerHTML =
    '<div class="msg-bubble-assistant">' +
      '<div class="working-indicator">' +
        '<span class="working-dots"><span></span><span></span><span></span></span>' +
        '<span class="working-label">' + (label || randomWorkingLabel()) + '</span>' +
      '</div>' +
      '<div class="slow-note" style="display:none"></div>' +
    '</div>';
  getInner().appendChild(div);
  pendingCursorEl = div;
  pendingCursorTimer = setTimeout(function() {
    if (!pendingCursorEl) return;
    const note = pendingCursorEl.querySelector('.slow-note');
    if (note) { note.textContent = SLOW_NOTE; note.style.display = ''; }
  }, SLOW_THRESHOLD_MS);
  scrollToBottom();
}

function clearPendingCursor() {
  if (pendingCursorTimer) { clearTimeout(pendingCursorTimer); pendingCursorTimer = null; }
  if (pendingCursorEl) { pendingCursorEl.remove(); pendingCursorEl = null; }
}

function endTextBlock() {
  flushTextBuffer();
  if (currentTextDiv) {
    const cursor = currentTextDiv.querySelector('.streaming-cursor');
    if (cursor) cursor.remove();
    makeTablesResizable(currentTextDiv);
  }
  currentTextDiv = null;
  currentTextRaw = '';
}

function startToolBlock(toolName) {
  currentToolId = 'tool-' + Date.now() + '-' + Math.random().toString(36).slice(2,6);
  const div = document.createElement('div');
  div.className = 'tool-block';
  div.innerHTML = '<div class="block-card">' +
    '<div class="block-header" onclick="toggleBlock(\'' + currentToolId + '\')">' +
      '<svg class="block-icon" viewBox="0 0 24 24" stroke="var(--brand-fig)" fill="none" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">' +
        '<path d="M14.7 6.3a1 1 0 000 1.4l1.6 1.6a1 1 0 001.4 0l3.77-3.77a6 6 0 01-7.94 7.94l-6.91 6.91a2.12 2.12 0 01-3-3l6.91-6.91a6 6 0 017.94-7.94l-3.76 3.76z"/>' +
      '</svg>' +
      '<span class="block-label">' + escapeHtml(toolName) + '</span>' +
      '<span class="pulse-dot" id="' + currentToolId + '-pulse"></span>' +
      '<svg class="block-chevron" id="' + currentToolId + '-chevron" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M9 18l6-6-6-6"/></svg>' +
    '</div>' +
    '<div class="block-body" id="' + currentToolId + '-body"><div class="section"><div class="label">Running...</div></div></div>' +
  '</div>';
  getInner().appendChild(div);
  scrollToBottom();
}

function addPlasmidPlot(plotJson) {
  var bokehItem = plotJson.plot !== undefined ? plotJson.plot : plotJson;
  var isLinear = plotJson.linear === true;
  var label = isLinear ? 'Linear DNA Map' : 'Plasmid Map';
  const plotId = 'plot-' + Date.now() + '-' + Math.random().toString(36).slice(2,6);
  const div = document.createElement('div');
  div.className = 'msg assistant';
  div.innerHTML = '<div class="msg-bubble-assistant" style="margin-top:8px;padding:12px;width:100%;max-width:640px;">' +
    '<div style="font-size:11px;font-weight:600;color:var(--sand-500);text-transform:uppercase;letter-spacing:0.05em;margin-bottom:8px;">' + label + '</div>' +
    '<div id="' + plotId + '" style="width:100%;"></div>' +
  '</div>';
  getInner().appendChild(div);
  Bokeh.embed.embed_item(bokehItem, plotId);
  scrollToBottom();
}

function addDownloadButton(container, content, filename) {
  const div = document.createElement('div');
  div.className = 'msg assistant';
  div.innerHTML = '<div class="msg-bubble-assistant" style="margin-top:8px">' +
    '<button class="download-btn">' +
      '<svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24">' +
        '<path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3"/>' +
      '</svg>' +
      ' Download ' + escapeHtml(filename) +
    '</button></div>';
  container.appendChild(div);
  div.querySelector('.download-btn').addEventListener('click', function() {
    const blob = new Blob([content], {type: 'application/octet-stream'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = filename;
    document.body.appendChild(a); a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  });
}

function finishToolBlock(toolName, toolInput, toolResult, downloadContent, downloadFilename) {
  if (currentToolId) {
    const pulse = document.getElementById(currentToolId + '-pulse');
    if (pulse) pulse.remove();
    const body = document.getElementById(currentToolId + '-body');
    if (body) {
      const inputStr = JSON.stringify(toolInput, null, 2);
      let html = '<div class="section"><div class="label">Input</div>' + escapeHtml(inputStr) + '</div>' +
        '<div class="section"><div class="label">Result</div>' + escapeHtml(toolResult) + '</div>';
      body.innerHTML = html;
    }
  }
  // Surface action buttons in the main chat (not just inside the collapsed tool block)
  if (downloadContent && downloadFilename) {
    const isGenbank = toolName === 'export_construct' &&
        ['genbank', 'gb'].includes((toolInput.output_format || '').toLowerCase());
    if (isGenbank) {
      addExportButtons(getInner(), toolInput, downloadContent, downloadFilename);
    } else {
      addDownloadButton(getInner(), downloadContent, downloadFilename);
    }
  }
  currentToolId = null;
  showPendingCursor();
  scrollToBottom();
}

// ── Send / Stop ──
async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text || isStreaming) return;

  // If a CSV is attached, run the template-batch flow instead of normal chat
  if (_pendingCSV && _pendingCSV.rows && _pendingCSV.rows.length) {
    var pendingRows = _pendingCSV.rows;
    var pendingFilename = _pendingCSV.filename;
    clearPendingCSV();
    inputEl.value = '';
    autoResize(inputEl);
    hideWelcome();
    var inner = getInner();
    // Show user message bubble
    var nowStr = new Date().toLocaleDateString(undefined, {month:'short',day:'numeric',year:'numeric'});
    var userDiv = document.createElement('div');
    userDiv.className = 'msg user';
    userDiv.innerHTML = '<div><div class="msg-bubble-user">' + escapeHtml(text) + '</div>' +
      '<div class="msg-date" style="display:flex;align-items:center;gap:6px">' +
        '<svg width="10" height="10" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>' +
        escapeHtml(pendingFilename) + ' · ' + pendingRows.length + ' rows' +
        ' &nbsp;' + nowStr +
      '</div></div>';
    inner.appendChild(userDiv);
    // Show "merging" spinner
    var loadId = 'tpl-loading-' + Date.now();
    var lc = document.createElement('div');
    lc.className = 'msg assistant'; lc.id = loadId;
    lc.innerHTML = '<div class="msg-bubble-assistant" style="color:var(--sand-500);font-size:13px">' +
      '<span class="streaming-cursor"></span> Merging template with ' + pendingRows.length + ' row' + (pendingRows.length === 1 ? '' : 's') + '&hellip;</div>';
    inner.appendChild(lc);
    scrollToBottom();
    fetch('/api/bulk/template-run', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({template: text, csv_rows: pendingRows, model: modelSelect.value}),
    })
    .then(function(r) { return r.json(); })
    .then(function(data) {
      var l = document.getElementById(loadId); if (l) l.remove();
      if (data.error) { alert('Error: ' + data.error); return; }
      var sid = data.session_id; var jobId = data.job_id;
      saveSessionId(sid);
      loadSessions();
      messagesEl.innerHTML = '';
      initBatchCards(jobId, data.row_count, pendingFilename, modelSelect.value);
      _batchSessions[sid] = jobId;
      if (_batchPollTimers[sid]) clearInterval(_batchPollTimers[sid]);
      _batchPollTimers[sid] = setInterval(function() { pollBatchForSession(sid); }, 2000);
      pollBatchForSession(sid);
    })
    .catch(function(e) {
      var l = document.getElementById(loadId); if (l) l.remove();
      alert('Template run failed: ' + e);
    });
    return;
  }

  // Data CSV context — append raw CSV to the message sent to the agent, but display only the user's text.
  var _csvDataAttachment = null;
  if (_pendingDataCSV) {
    _csvDataAttachment = _pendingDataCSV;
    _pendingDataCSV = null;
    // Remove the pending summary card and badge.
    var pendingCard = document.getElementById(_csvDataAttachment.pendingCardId);
    if (pendingCard) pendingCard.remove();
    var badge = document.getElementById('csv-badge');
    if (badge) badge.style.display = 'none';
  }
  var apiText = text;
  if (_csvDataAttachment) {
    apiText = text + '\n\n[CSV data from: ' + _csvDataAttachment.filename + ']\n```csv\n' + _csvDataAttachment.rawText + '\n```';
  }

  isStreaming = true;
  streamingSessionId = currentSessionId;
  sendBtn.style.display = 'none';
  stopBtn.style.display = 'flex';
  inputEl.value = '';
  inputEl.disabled = true;
  autoResize(inputEl);
  hideWelcome();

  var inner = getInner();
  // Pin this container so stream events write here even if user switches sessions
  streamingInner = inner;
  var userDiv = document.createElement('div');
  userDiv.className = 'msg user';
  var nowStr = new Date().toLocaleDateString(undefined, {month:'short',day:'numeric',year:'numeric'});
  var attachNote = _csvDataAttachment
    ? '<div class="msg-date" style="display:flex;align-items:center;gap:6px;margin-top:4px">' +
        '<svg width="10" height="10" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>' +
        escapeHtml(_csvDataAttachment.filename) + ' (data)' +
      '</div>'
    : '';
  userDiv.innerHTML = '<div><div class="msg-bubble-user">' + escapeHtml(text) + '</div>' + attachNote + '<div class="msg-date">' + nowStr + '</div></div>';
  inner.appendChild(userDiv);
  scrollToBottom();
  showPendingCursor();

  abortController = new AbortController();

  try {
    const reqBody = { message: apiText, model: modelSelect.value };
    if (currentSessionId) reqBody.session_id = currentSessionId;

    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(reqBody),
      signal: abortController.signal,
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, {stream: true});
      const parts = buffer.split('\n\n');
      buffer = parts.pop();

      let streamDone = false;
      for (const part of parts) {
        const trimmed = part.trim();
        if (!trimmed.startsWith('data: ')) continue;
        const jsonStr = trimmed.slice(6);
        if (!jsonStr) continue;

        let event;
        try { event = JSON.parse(jsonStr); } catch { continue; }

        switch (event.type) {
          case 'session_id':
            saveSessionId(event.session_id);
            loadSessions();
            break;
          case 'thinking_start': clearPendingCursor(); startThinkingBlock(); break;
          case 'thinking_delta': appendThinkingDelta(event.content); break;
          case 'thinking_end': endThinkingBlock(); break;
          case 'text_start': clearPendingCursor(); flushTextBuffer(); startTextBlock(); break;
          case 'text_delta': bufferTextDelta(event.content); break;
          case 'text_end': endTextBlock(); break;
          case 'tool_use_start': clearPendingCursor(); startToolBlock(event.tool); break;
          case 'tool_result': finishToolBlock(event.tool, event.input || {}, event.content, event.download_content, event.download_filename); break;
          case 'plot_data': addPlasmidPlot(event.plot_json); break;
          case 'token_usage': updateTokenIndicator(event.input_tokens, event.context_window); break;
          case 'error':
            clearPendingCursor();
            startTextBlock();
            appendTextDelta('Error: ' + event.content);
            endTextBlock();
            break;
          case 'bulk_design_rows':
            streamDone = true;
            if (currentToolId) {
              var _pulse = document.getElementById(currentToolId + '-pulse');
              if (_pulse) _pulse.remove();
              var _body = document.getElementById(currentToolId + '-body');
              if (_body) _body.innerHTML = '<div class="section"><div class="label">Result</div>Submitted ' + (event.rows || []).length + ' design(s) to bulk planner.</div>';
              currentToolId = null;
            }
            requestBulkPlanFromRows(event.rows || [], modelSelect.value);
            break;
          case 'bulk_designs_registered':
            showBulkPreviewModelCard(event);
            break;
          case 'bulk_preview_export':
            _bulkPreviewExports.push({filename: event.filename, content: event.content});
            break;
          case 'bulk_preview_complete':
            { var _mc2 = document.getElementById('bulk-preview-model-card'); if (_mc2) _mc2.remove(); }
            showBulkPreviewApprovalCard(event);
            break;
          case 'done': streamDone = true; break;
        }
        if (streamDone) break;
      }
      if (streamDone) break;
    }
  } catch (err) {
    if (err.name !== 'AbortError') {
      clearPendingCursor();
      startTextBlock();
      appendTextDelta('Connection error: ' + err.message);
      endTextBlock();
    }
  }

  clearPendingCursor();
  isStreaming = false;
  abortController = null;
  streamingInner = null;
  streamingSessionId = null;
  sendBtn.style.display = 'flex';
  stopBtn.style.display = 'none';
  inputEl.disabled = false;
  inputEl.focus();
  // Remove any leftover streaming cursor
  const cursor = messagesEl.querySelector('.streaming-cursor');
  if (cursor) cursor.remove();
  // Refresh library panel in case the agent saved a new backbone or construct
  loadUserLibrary();
}

function _detachStream() {
  // Drop the SSE connection without cancelling the server-side run.
  // Use this when navigating away — the agent keeps going in the background.
  if (abortController) abortController.abort();
}

function stopGeneration() {
  // Explicitly cancel the run (Stop button). Aborts client AND tells server to stop.
  if (abortController) abortController.abort();
  if (currentSessionId) {
    fetch('/api/sessions/' + currentSessionId + '/cancel', { method: 'POST' }).catch(function(){});
  }
}

function sendExample(btn) {
  inputEl.value = btn.textContent;
  sendMessage();
}

// ── Keyboard ──
inputEl.addEventListener('keydown', function(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// ── Batch state (must be declared before init so selectSession can reference them) ──
var _batchSessions = {};    // sessionId → jobId
var _batchPollTimers = {};  // sessionId → interval timer
var _batchConfirmData = {};
var _bulkPlanContext = {};  // cardId → {csvText, filename, model}
const chatPanelEl = document.getElementById('chat-panel');
const dropOverlayEl = document.getElementById('drop-overlay');

// ── Init ──
checkHealth();
setInterval(checkHealth, 5000);
loadSessions();
loadUserLibrary();
_checkUserLibrary();
setInterval(loadSessions, 5000);
// Restore active session on page load
if (currentSessionId) {
  selectSession(currentSessionId);
}
inputEl.focus();

// ── Drag & drop onto the chat area (CSV batch or plasmid file) ──
var dragCounter = 0;

var PLASMID_EXTS = ['.gb', '.gbk', '.genbank', '.fasta', '.fa', '.seq'];

function isFileDrag(e) {
  var types = e.dataTransfer && e.dataTransfer.types;
  return types && Array.from(types).indexOf('Files') !== -1;
}

function isPlasmidFile(file) {
  var name = (file.name || '').toLowerCase();
  return PLASMID_EXTS.some(function(ext) { return name.endsWith(ext); });
}

chatPanelEl.addEventListener('dragenter', function(e) {
  if (!isFileDrag(e)) return;
  e.preventDefault();
  dragCounter++;
  dropOverlayEl.classList.add('active');
});

chatPanelEl.addEventListener('dragleave', function(e) {
  if (!isFileDrag(e)) return;
  dragCounter--;
  if (dragCounter <= 0) { dragCounter = 0; dropOverlayEl.classList.remove('active'); }
});

chatPanelEl.addEventListener('dragover', function(e) {
  if (!isFileDrag(e)) return;
  e.preventDefault();
  e.dataTransfer.dropEffect = 'copy';
});

chatPanelEl.addEventListener('drop', function(e) {
  e.preventDefault();
  dragCounter = 0;
  dropOverlayEl.classList.remove('active');
  var file = e.dataTransfer.files[0];
  if (!file) return;
  var reader = new FileReader();
  if (isPlasmidFile(file)) {
    reader.onload = function(ev) { uploadPlasmidFile(ev.target.result, file.name); };
    reader.readAsText(file);
  } else if (file.name.endsWith('.csv') || file.type === 'text/csv') {
    reader.onload = function(ev) { handleCSVUpload(ev.target.result, file.name); };
    reader.readAsText(file);
  } else {
    alert('Supported file types: .gb, .gbk, .fasta (plasmid files) or .csv (data or bulk design).');
  }
});

function onBatchFileChosen(input) {
  var file = input.files[0];
  if (!file) return;
  var reader = new FileReader();
  reader.onload = function(e) { handleCSVUpload(e.target.result, file.name); };
  reader.readAsText(file);
  input.value = '';
}

var _pendingCSV = null; // {rows, filename, rawText}
var _pendingDataCSV = null; // {rawText, filename} — CSV attached as data context, not bulk design

function onCombinedFileChosen(input) {
  var file = input.files[0];
  if (!file) return;
  input.value = '';
  var reader = new FileReader();
  if (file.name.endsWith('.csv') || file.type === 'text/csv') {
    reader.onload = function(e) { handleCSVUpload(e.target.result, file.name); };
  } else if (isPlasmidFile(file)) {
    reader.onload = function(e) { uploadPlasmidFile(e.target.result, file.name); };
  } else {
    alert('Supported: .gb, .gbk, .fasta (plasmid files) or .csv (data or bulk design)');
    return;
  }
  reader.readAsText(file);
}

function attachPendingCSV(csvText, filename) {
  var parsed = _parseCSVRows(csvText);
  if (!parsed.rows.length) {
    alert('No rows with a "description" column found. If you just have Name/Oligo columns, that\'s fine — just attach and send your template message.');
    // Still attach even if no description column — raw rows carry the data
  }
  // Store raw rows from csv (all columns, not just description)
  var rawRows = [];
  var lines = csvText.split('\n').filter(function(l) { return l.trim(); });
  if (lines.length > 1) {
    var headers = _splitCSVLine(lines[0]);
    for (var i = 1; i < lines.length; i++) {
      var fields = _splitCSVLine(lines[i]);
      if (!fields.some(function(f) { return f.trim(); })) continue;
      var row = {};
      headers.forEach(function(h, j) { row[h.trim()] = (fields[j] || '').trim(); });
      rawRows.push(row);
    }
  }
  _pendingCSV = {rows: rawRows, filename: filename, rawText: csvText};
  var badge = document.getElementById('csv-badge');
  var label = document.getElementById('csv-badge-name');
  if (badge) badge.style.display = 'flex';
  if (label) label.textContent = filename + ' — ' + rawRows.length + ' row' + (rawRows.length === 1 ? '' : 's');
}

function clearPendingCSV() {
  _pendingCSV = null;
  _pendingDataCSV = null;
  var badge = document.getElementById('csv-badge');
  if (badge) badge.style.display = 'none';
}

// Detect whether a CSV looks like a bulk design request (has a 'description' column).
function detectCSVIntent(csvText) {
  var lines = csvText.split('\n').filter(function(l) { return l.trim(); });
  if (!lines.length) return {isBulkDesign: false, columns: [], rowCount: 0};
  var headers = _splitCSVLine(lines[0]).map(function(h) { return h.trim(); });
  var hasDescription = headers.some(function(h) { return h.toLowerCase() === 'description'; });
  // Count non-empty data rows (rough — doesn't re-parse each line fully)
  var rowCount = lines.length - 1;
  // Only call it bulk-design if there's a description col AND at least one data row
  return {isBulkDesign: hasDescription && rowCount > 0, columns: headers, rowCount: rowCount};
}

// Central CSV upload router — replaces direct calls to requestBulkPlan / attachPendingCSV.
function handleCSVUpload(csvText, filename) {
  var intent = detectCSVIntent(csvText);
  if (intent.isBulkDesign) {
    requestBulkPlan(csvText, filename);
  } else {
    showCSVIntentCard(csvText, filename, intent.columns, intent.rowCount);
  }
}

var _csvIntentStore = {}; // temporary store keyed by cardId to avoid embedding CSV text in onclick HTML

// Show an inline card when the uploaded CSV doesn't look like a bulk design file.
function showCSVIntentCard(csvText, filename, columns, rowCount) {
  hideWelcome();
  var inner = getInner();
  var cardId = 'csv-intent-' + Date.now();
  _csvIntentStore[cardId] = {csvText: csvText, filename: filename, columns: columns, rowCount: rowCount};

  // Build a short preview: header + up to 3 data rows.
  var lines = csvText.split('\n').filter(function(l) { return l.trim(); });
  var previewLines = lines.slice(0, 4);
  var previewText = previewLines.map(escapeHtml).join('\n');
  if (lines.length > 4) previewText += '\n<span style="color:var(--sand-400)">&hellip; ' + (lines.length - 4) + ' more rows</span>';

  var rowLabel = rowCount + ' row' + (rowCount === 1 ? '' : 's');

  var card = document.createElement('div');
  card.className = 'msg assistant';
  card.id = cardId;
  card.innerHTML =
    '<div class="msg-bubble-assistant" style="font-size:13px">' +
      '<div style="display:flex;align-items:center;gap:6px;margin-bottom:10px">' +
        '<svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>' +
        '<strong>' + escapeHtml(filename) + '</strong>' +
        '<span style="color:var(--sand-400);font-weight:400;font-size:12px">' + rowLabel + '</span>' +
      '</div>' +
      '<pre style="background:var(--sand-100,#f5f0eb);border-radius:6px;padding:10px 12px;font-size:11.5px;margin:0 0 12px;overflow-x:auto;white-space:pre-wrap;word-break:break-all">' + previewText + '</pre>' +
      '<p style="margin:0 0 12px;color:var(--sand-500);font-size:12px">This doesn\'t look like a bulk design file — those need a <code>description</code> column with one design instruction per row.</p>' +
      '<div style="display:flex;gap:8px;flex-wrap:wrap">' +
        '<button onclick="_csvIntentUseAsData(\'' + cardId + '\')" ' +
          'style="padding:6px 14px;background:var(--accent,#7c6fcd);color:#fff;border:none;border-radius:6px;font-size:12px;cursor:pointer">Use as data</button>' +
        '<button onclick="_csvIntentUseBulkFormat(\'' + cardId + '\')" ' +
          'style="padding:6px 14px;background:transparent;color:var(--accent,#7c6fcd);border:1px solid var(--accent,#7c6fcd);border-radius:6px;font-size:12px;cursor:pointer">Use for bulk design</button>' +
      '</div>' +
    '</div>';
  inner.appendChild(card);
  scrollToBottom();
}

function _csvIntentUseAsData(cardId) {
  var data = _csvIntentStore[cardId];
  if (!data) return;
  delete _csvIntentStore[cardId];
  var el = document.getElementById(cardId);
  if (el) el.remove();
  attachDataCSV(data.csvText, data.filename, data.columns, data.rowCount);
}

function _csvIntentUseBulkFormat(cardId) {
  delete _csvIntentStore[cardId];
  var el = document.getElementById(cardId);
  if (el) el.remove();
  var tip = document.createElement('div');
  tip.className = 'msg assistant';
  tip.innerHTML = '<div class="msg-bubble-assistant" style="font-size:13px">To use bulk design, add a <code>description</code> column to your CSV with one full design instruction per row, then re-upload it.</div>';
  getInner().appendChild(tip);
  scrollToBottom();
}

// Attach a CSV as plain data context (included in the next message sent to the agent).
function attachDataCSV(csvText, filename, columns, rowCount) {
  // Show a compact summary card in the chat so the user can see what's pending.
  hideWelcome();
  var pendingCardId = 'csv-data-pending-' + Date.now();
  var colSnippet = (columns && columns.length)
    ? columns.slice(0, 5).map(escapeHtml).join(', ') + (columns.length > 5 ? ', …' : '')
    : '';
  var rowLabel = rowCount != null ? rowCount + ' row' + (rowCount === 1 ? '' : 's') : '';
  var metaLine = [rowLabel, colSnippet].filter(Boolean).join('  ·  ');

  var card = document.createElement('div');
  card.className = 'msg assistant';
  card.id = pendingCardId;
  card.innerHTML =
    '<div class="msg-bubble-assistant" style="font-size:12.5px;padding:10px 14px">' +
      '<div style="display:flex;align-items:center;justify-content:space-between;gap:8px">' +
        '<div style="display:flex;align-items:center;gap:7px;font-weight:500;min-width:0">' +
          '<svg width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24" style="flex-shrink:0"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>' +
          '<span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">' + escapeHtml(filename) + '</span>' +
        '</div>' +
        '<button onclick="_csvDataRemove(\'' + pendingCardId + '\')" title="Remove" ' +
          'style="flex-shrink:0;background:none;border:none;cursor:pointer;color:var(--sand-400,#a09080);font-size:15px;line-height:1;padding:0 2px">&times;</button>' +
      '</div>' +
      (metaLine ? '<div style="margin-top:4px;color:var(--sand-500,#9a8a7a);font-size:11.5px">' + metaLine + '</div>' : '') +
      '<div style="margin-top:6px;color:var(--sand-500,#9a8a7a);font-size:11.5px;font-style:italic">Attached as data — will be included in your next message.</div>' +
    '</div>';
  getInner().appendChild(card);
  scrollToBottom();

  _pendingDataCSV = {rawText: csvText, filename: filename, pendingCardId: pendingCardId};

  // Also update the input-area badge as a secondary indicator.
  var badge = document.getElementById('csv-badge');
  var label = document.getElementById('csv-badge-name');
  if (badge) badge.style.display = 'flex';
  if (label) label.textContent = filename + ' — data';
}

function _csvDataRemove(pendingCardId) {
  var el = document.getElementById(pendingCardId);
  if (el) el.remove();
  _pendingDataCSV = null;
  var badge = document.getElementById('csv-badge');
  if (badge) badge.style.display = 'none';
}

function onPlasmidFileChosen(input) {
  var file = input.files[0];
  if (!file) return;
  var reader = new FileReader();
  reader.onload = function(e) { uploadPlasmidFile(e.target.result, file.name); };
  reader.readAsText(file);
  input.value = '';
}

// ── Plasmid file upload ──
var plasmidBadgeEl = document.getElementById('plasmid-badge');
var plasmidBadgeNameEl = document.getElementById('plasmid-badge-name');
var plasmidBadgeStatusEl = document.getElementById('plasmid-badge-status');

function uploadPlasmidFile(text, filename) {
  // Show "analyzing" badge in the input area
  plasmidBadgeNameEl.textContent = filename;
  plasmidBadgeStatusEl.textContent = 'analyzing with plannotate…';
  plasmidBadgeEl.style.display = 'flex';

  var model = modelSelect.value;
  fetch('/api/upload-plasmid', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({content: text, filename: filename}),
  })
  .then(function(r) { return r.json(); })
  .then(function(data) {
    plasmidBadgeEl.style.display = 'none';
    if (data.error) {
      alert('Could not process plasmid file: ' + data.error);
      return;
    }
    // Auto-send the intake message as if the user typed it
    hideWelcome();
    sendPlasmidIntakeMessage(data.message, model, filename, data.size_bp, data.feature_count);
  })
  .catch(function(e) {
    plasmidBadgeEl.style.display = 'none';
    alert('Upload failed: ' + e);
  });
}

async function sendPlasmidIntakeMessage(apiMessage, model, filename, sizeBp, featureCount) {
  if (isStreaming) return;
  isStreaming = true;
  streamingSessionId = currentSessionId;
  sendBtn.style.display = 'none';
  stopBtn.style.display = 'flex';
  inputEl.disabled = true;
  hideWelcome();

  const inner = getInner();
  streamingInner = inner;

  // Show short summary as user bubble (not the full sequence)
  var summary = '📎 ' + filename;
  if (sizeBp) summary += ' — ' + sizeBp.toLocaleString() + ' bp';
  if (featureCount) summary += ', ' + featureCount + ' plannotate feature(s)';
  const nowStr = new Date().toLocaleDateString(undefined, {month:'short',day:'numeric',year:'numeric'});
  const userDiv = document.createElement('div');
  userDiv.className = 'msg user';
  userDiv.innerHTML = '<div><div class="msg-bubble-user">' + escapeHtml(summary) + '</div><div class="msg-date">' + nowStr + '</div></div>';
  inner.appendChild(userDiv);
  scrollToBottom();
  showPendingCursor();

  abortController = new AbortController();
  try {
    const reqBody = {message: apiMessage, model: model};
    if (currentSessionId) reqBody.session_id = currentSessionId;
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(reqBody),
      signal: abortController.signal,
    });
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true});
      const parts = buffer.split('\n\n');
      buffer = parts.pop();
      let streamDone = false;
      for (const part of parts) {
        const trimmed = part.trim();
        if (!trimmed.startsWith('data: ')) continue;
        let event;
        try { event = JSON.parse(trimmed.slice(6)); } catch { continue; }
        switch (event.type) {
          case 'session_id': saveSessionId(event.session_id); loadSessions(); break;
          case 'thinking_start': clearPendingCursor(); startThinkingBlock(); break;
          case 'thinking_delta': appendThinkingDelta(event.content); break;
          case 'thinking_end': endThinkingBlock(); break;
          case 'text_start': clearPendingCursor(); flushTextBuffer(); startTextBlock(); break;
          case 'text_delta': bufferTextDelta(event.content); break;
          case 'text_end': endTextBlock(); break;
          case 'tool_use_start': clearPendingCursor(); startToolBlock(event.tool); break;
          case 'tool_result': finishToolBlock(event.tool, event.input || {}, event.content, event.download_content, event.download_filename); break;
          case 'plot_data': addPlasmidPlot(event.plot_json); break;
          case 'token_usage': updateTokenIndicator(event.input_tokens, event.context_window); break;
          case 'error': clearPendingCursor(); startTextBlock(); appendTextDelta('Error: ' + event.content); endTextBlock(); break;
          case 'bulk_designs_registered':
            showBulkPreviewModelCard(event);
            break;
          case 'bulk_preview_export':
            _bulkPreviewExports.push({filename: event.filename, content: event.content});
            break;
          case 'bulk_preview_complete':
            { var _mc3 = document.getElementById('bulk-preview-model-card'); if (_mc3) _mc3.remove(); }
            showBulkPreviewApprovalCard(event);
            break;
          case 'done': streamDone = true; break;
        }
        if (streamDone) break;
      }
      if (streamDone) break;
    }
  } catch(err) {
    if (err.name !== 'AbortError') {
      clearPendingCursor(); startTextBlock(); appendTextDelta('Connection error: ' + err.message); endTextBlock();
    }
  }
  clearPendingCursor();
  isStreaming = false; abortController = null; streamingInner = null; streamingSessionId = null;
  sendBtn.style.display = 'flex'; stopBtn.style.display = 'none';
  inputEl.disabled = false; inputEl.focus();
  const cursor = messagesEl.querySelector('.streaming-cursor');
  if (cursor) cursor.remove();
  // Refresh library panel — agent may have saved a new backbone during this conversation turn
  loadUserLibrary();
}

function _splitCSVLine(line) {
  var result = [], field = '', inQuote = false;
  for (var i = 0; i < line.length; i++) {
    var c = line[i];
    if (c === '"') { inQuote = !inQuote; }
    else if (c === ',' && !inQuote) { result.push(field.trim()); field = ''; }
    else { field += c; }
  }
  result.push(field.trim());
  return result;
}

function _parseCSVRows(csvText) {
  var rawLines = csvText.split('\n');
  var headerLine = '';
  var headerFields = [];
  for (var i = 0; i < rawLines.length; i++) {
    if (rawLines[i].trim()) { headerLine = rawLines[i]; headerFields = _splitCSVLine(headerLine); break; }
  }
  var descIdx = headerFields.findIndex(function(h) { return h.trim().toLowerCase() === 'description'; });
  var nameIdx = headerFields.findIndex(function(h) { return h.trim().toLowerCase() === 'name'; });
  if (descIdx < 0) return {header: headerLine, rows: []};
  var rows = [];
  for (var j = i + 1; j < rawLines.length; j++) {
    var line = rawLines[j];
    if (!line.trim()) continue;
    var fields = _splitCSVLine(line);
    var desc = fields[descIdx] || '';
    if (!desc.trim()) continue;
    rows.push({
      description: desc,
      name: nameIdx >= 0 ? (fields[nameIdx] || '') : '',
      originalLine: line,
    });
  }
  return {header: headerLine, rows: rows};
}

function showBatchConfirm(csvText, filename) {
  var parsed = _parseCSVRows(csvText);
  var rows = parsed.rows;
  var confirmId = 'batch-confirm-' + Date.now();
  _batchConfirmData[confirmId] = {csvText: csvText, filename: filename, header: parsed.header, rows: rows};
  hideWelcome();
  var inner = getInner();
  var card = document.createElement('div');
  card.className = 'msg assistant';
  card.id = confirmId;
  var curModel = 'claude-sonnet-4-6';
  var modelOpts = [
    ['claude-opus-4-7', 'Opus 4.7 — most capable'],
    ['claude-opus-4-6', 'Opus 4.6'],
    ['claude-sonnet-4-6', 'Sonnet 4.6 — recommended for bulk assembly'],
    ['claude-haiku-4-5-20251001', 'Haiku 4.5 — fastest'],
  ].map(function(o) {
    return '<option value="' + o[0] + '"' + (curModel === o[0] ? ' selected' : '') + '>' + o[1] + '</option>';
  }).join('');

  var rowsHtml = rows.map(function(r, i) {
    var nameHtml = r.name ? '<span class="batch-confirm-row-name">' + escapeHtml(r.name) + '</span>' : '';
    return '<label class="batch-confirm-row">' +
      '<input type="checkbox" id="' + confirmId + '-row-' + i + '" checked onchange="updateBatchConfirmCount(\'' + confirmId + '\')">' +
      '<span class="batch-confirm-row-num">' + (i + 1) + '</span>' +
      '<span class="batch-confirm-row-desc">' + escapeHtml(r.description) + '</span>' +
      nameHtml +
    '</label>';
  }).join('');

  card.innerHTML = '<div class="msg-bubble-assistant"><div class="batch-confirm-card">' +
    '<div style="font-size:14px;font-weight:600;color:var(--sand-800);margin-bottom:10px">' +
      escapeHtml(filename) +
    '</div>' +
    '<div class="batch-advisory">' +
      '<strong>Before you run a bulk design</strong>' +
      '<ul>' +
        '<li><strong>Test first.</strong> Run a few representative prompts as individual chats and confirm they succeed before scaling up. Failures in bulk are harder to debug and still cost tokens.</li>' +
        '<li><strong>Design by parts works best.</strong> Bulk mode is optimized for assembly-style designs (backbone + insert combinations). Bespoke or highly custom designs often need back-and-forth that bulk mode can\'t do.</li>' +
        '<li><strong>Use a cheaper model.</strong> Sonnet 4.6 handles most bulk assembly tasks well at a fraction of the cost of Opus. Switch the model below before starting.</li>' +
      '</ul>' +
    '</div>' +
    '<label class="batch-confirm-select-all">' +
      '<input type="checkbox" id="' + confirmId + '-selectall" checked onchange="toggleBatchSelectAll(\'' + confirmId + '\',' + rows.length + ')">' +
      '<span>Select All — <span id="' + confirmId + '-selcount">' + rows.length + '</span> of ' + rows.length + ' selected</span>' +
    '</label>' +
    '<div class="batch-confirm-rows">' + rowsHtml + '</div>' +
    '<div style="margin-bottom:14px">' +
      '<label style="font-size:12px;font-weight:500;color:var(--sand-500);display:block;margin-bottom:5px">Model</label>' +
      '<select id="' + confirmId + '-model" class="model-select" style="font-size:12px;max-width:100%">' + modelOpts + '</select>' +
      '<div style="font-size:11px;color:var(--sand-400);margin-top:5px">Tip: Sonnet 4.6 handles bulk assembly by parts well at lower cost than Opus.</div>' +
    '</div>' +
    '<div style="display:flex;gap:8px;align-items:center">' +
      '<button id="' + confirmId + '-startbtn" class="send-btn" style="width:auto;padding:0 18px;height:32px;font-size:13px;border-radius:10px" ' +
        'onclick="startBatchFromConfirm(\'' + confirmId + '\',' + rows.length + ')">' +
        'Start <span id="' + confirmId + '-btncount">' + rows.length + '</span> design' + (rows.length === 1 ? '' : 's') +
      '</button>' +
      '<button onclick="cancelBatchConfirm(\'' + confirmId + '\')" ' +
        'style="padding:0 14px;height:32px;font-size:13px;background:transparent;border:1px solid var(--sand-200);border-radius:10px;cursor:pointer;color:var(--sand-600);font-family:inherit">Cancel</button>' +
    '</div>' +
  '</div></div>';
  inner.appendChild(card);
  scrollToBottom();
}

function updateBatchConfirmCount(confirmId) {
  var data = _batchConfirmData[confirmId];
  if (!data) return;
  var n = data.rows.length;
  var checked = 0;
  for (var i = 0; i < n; i++) {
    var cb = document.getElementById(confirmId + '-row-' + i);
    if (cb && cb.checked) checked++;
  }
  var selCount = document.getElementById(confirmId + '-selcount');
  var btnCount = document.getElementById(confirmId + '-btncount');
  var startBtn = document.getElementById(confirmId + '-startbtn');
  var selectAll = document.getElementById(confirmId + '-selectall');
  if (selCount) selCount.textContent = checked;
  if (btnCount) btnCount.textContent = checked;
  if (startBtn) startBtn.disabled = checked === 0;
  if (selectAll) selectAll.indeterminate = (checked > 0 && checked < n);
  if (selectAll && !selectAll.indeterminate) selectAll.checked = (checked === n);
}

function toggleBatchSelectAll(confirmId, total) {
  var selectAllEl = document.getElementById(confirmId + '-selectall');
  if (!selectAllEl) return;
  var checked = selectAllEl.checked;
  for (var i = 0; i < total; i++) {
    var cb = document.getElementById(confirmId + '-row-' + i);
    if (cb) cb.checked = checked;
  }
  updateBatchConfirmCount(confirmId);
}

function startBatchFromConfirm(confirmId, total) {
  var data = _batchConfirmData[confirmId];
  if (!data) return;
  delete _batchConfirmData[confirmId];
  // Build filtered CSV from checked rows
  var selectedLines = [];
  for (var i = 0; i < (data.rows || []).length; i++) {
    var cb = document.getElementById(confirmId + '-row-' + i);
    if (cb && cb.checked) selectedLines.push(data.rows[i].originalLine);
  }
  var card = document.getElementById(confirmId);
  if (card) card.remove();
  if (!selectedLines.length) return;
  var filteredCSV = data.header + '\n' + selectedLines.join('\n');
  // Route through chat so the agent handles it like typed input
  requestBulkPlan(filteredCSV, data.filename);
}

function cancelBatchConfirm(confirmId) {
  delete _batchConfirmData[confirmId];
  var card = document.getElementById(confirmId);
  if (card) card.remove();
}

// ── Bulk preview approval (agent-driven new flow) ──────────────────────────

// Stores {rows, sharedCtx} keyed by cardId for approveBulkPreview()
var _bulkPreviewData    = {};
var _bulkPreviewTokens  = {in: 0, out: 0};  // actual token counts from preview run (full cost, for display)
var _bulkPreviewMarginalTokens = {in: 0, out: 0};  // projected per-row cost, excludes one-time setup
var _bulkPreviewModel   = 'claude-sonnet-4-6';  // model the user picked for the bulk run
var _bulkPreviewExports = [];  // export files captured during the preview run

function buildEnrichedPrompt(description, sharedCtx) {
  var lines = ['<!-- bulk-enriched-row -->',
               'SHARED CONTEXT (already resolved — skip these tool calls):'];
  if (sharedCtx.backbone_id) {
    lines.push('- Backbone: "' + (sharedCtx.backbone_name || sharedCtx.backbone_id) +
               '" — use backbone_id="' + sharedCtx.backbone_id + '" directly, do NOT search or fetch again');
  }
  if (sharedCtx.insertion_site_start !== undefined && sharedCtx.insertion_site_start !== null) {
    var siteRange = sharedCtx.insertion_site_end
      ? sharedCtx.insertion_site_start + '–' + sharedCtx.insertion_site_end
      : String(sharedCtx.insertion_site_start);
    lines.push('- Insertion site: position ' + siteRange +
               ' — use insertion_position=' + sharedCtx.insertion_site_start + ' directly');
  }
  if (sharedCtx.enzyme) lines.push('- Assembly enzyme: ' + sharedCtx.enzyme);
  if (sharedCtx.assembly_method) lines.push('- Assembly method: ' + sharedCtx.assembly_method);
  if (sharedCtx.extra) lines.push('- Additional context: ' + sharedCtx.extra);
  lines.push('', 'YOUR TASK:', description);
  return lines.join('\n');
}

function _bulkModelOpts(defaultModel) {
  return [
    ['claude-sonnet-4-6',        'Sonnet 4.6 — recommended'],
    ['claude-opus-4-7',          'Opus 4.7 — most capable'],
    ['claude-haiku-4-5-20251001','Haiku 4.5 — fastest / cheapest'],
  ].map(function(o) {
    return '<option value="' + o[0] + '"' + (o[0] === defaultModel ? ' selected' : '') + '>' + o[1] + '</option>';
  }).join('');
}

// Shown when the agent registers bulk designs — user picks model and confirms before preview starts.
function showBulkPreviewModelCard(event) {
  var n       = event.n_constructs || 0;  // remaining constructs after the preview
  var defMdl  = event.preview_model || 'claude-sonnet-4-6';
  _bulkPreviewModel = defMdl;

  var existing = document.getElementById('bulk-preview-model-card');
  if (existing) existing.remove();

  // Reset export capture for this (re)run
  _bulkPreviewExports = [];

  var card = document.createElement('div');
  card.className = 'msg assistant';
  card.id = 'bulk-preview-model-card';
  card.innerHTML = '<div class="msg-bubble-assistant"><div class="bulk-plan-card">' +
    '<div class="bulk-plan-title">Ready to build bulk preview</div>' +
    '<div class="bulk-plan-summary">' + (n + 1) + ' construct' + ((n + 1) === 1 ? '' : 's') + ' queued. ' +
      'Construct #1 will be built here as a preview before committing to the rest.</div>' +
    '<div style="margin-bottom:14px">' +
      '<label style="font-size:12px;font-weight:500;color:var(--sand-500);display:block;margin-bottom:6px">' +
        'Model for preview and subsequent constructs:</label>' +
      '<select class="model-select" style="font-size:12px;max-width:280px" id="bulk-preview-model-sel" ' +
        'onchange="_bulkPreviewModel = this.value">' +
        _bulkModelOpts(defMdl) +
      '</select>' +
    '</div>' +
    '<div style="display:flex;gap:8px;align-items:center">' +
      '<button class="send-btn" style="width:auto;padding:0 18px;height:32px;font-size:13px;border-radius:10px" ' +
        'onclick="startBulkPreviewRun(this)">Start Preview</button>' +
      '<button onclick="document.getElementById(\'bulk-preview-model-card\').remove()" ' +
        'style="padding:0 14px;height:32px;font-size:13px;background:transparent;border:1px solid var(--sand-200);border-radius:10px;cursor:pointer;color:var(--sand-600);font-family:inherit">Cancel</button>' +
    '</div>' +
  '</div></div>';

  getInner().appendChild(card);
  scrollToBottom();
}

function startBulkPreviewRun(btn) {
  if (btn) { btn.disabled = true; btn.textContent = 'Starting…'; }
  // Switch the global model selector so the follow-up API call uses the chosen model
  var modelSel = document.getElementById('model-select');
  if (modelSel) modelSel.value = _bulkPreviewModel;
  // Replace card content with a progress indicator
  var card = document.getElementById('bulk-preview-model-card');
  if (card) {
    var bc = card.querySelector('.bulk-plan-card');
    if (bc) bc.innerHTML =
      '<div style="display:flex;align-items:center;gap:8px;font-size:13px;color:var(--sand-500)">' +
      '<span class="streaming-cursor"></span> Building preview with ' +
      '<strong style="color:var(--sand-700)">' + _bulkPreviewModel + '</strong>&hellip;</div>';
  }
  // Use rerun message if set (from rerunBulkPreview), otherwise start fresh
  var msg = window._bulkRerunMsg || 'Please start the bulk preview.';
  window._bulkRerunMsg = null;
  inputEl.value = msg;
  sendMessage();
}

function showBulkPreviewApprovalCard(event) {
  var remainingRows = event.remaining_rows || [];
  var sharedCtx     = event.shared_context || {};
  var summary       = event.preview_summary || '';
  var n             = remainingRows.length;
  var cardId        = 'bulk-preview-' + Date.now();

  // Use raw preview token counts for both the banner and the cost estimate.
  _bulkPreviewTokens = {
    in:  event.preview_tokens_in  || 0,
    out: event.preview_tokens_out || 0,
  };
  _bulkPreviewMarginalTokens = {
    in:  event.preview_tokens_in  || 0,
    out: event.preview_tokens_out || 0,
  };
  // Use the model the user picked in the model-picker card (or the session default).
  if (event.preview_model) _bulkPreviewModel = event.preview_model;
  var defaultModel = _bulkPreviewModel || 'claude-sonnet-4-6';

  _bulkPreviewData[cardId] = {rows: remainingRows, sharedCtx: sharedCtx};

  // Build shared context summary lines
  var ctxLines = [];
  if (sharedCtx.backbone_name || sharedCtx.backbone_id)
    ctxLines.push('Backbone: ' + (sharedCtx.backbone_name || sharedCtx.backbone_id));
  if (sharedCtx.insertion_site_start !== undefined && sharedCtx.insertion_site_start !== null)
    ctxLines.push('Insertion site: ' + sharedCtx.insertion_site_start +
                  (sharedCtx.insertion_site_end ? '–' + sharedCtx.insertion_site_end : ''));
  if (sharedCtx.enzyme) ctxLines.push('Enzyme: ' + sharedCtx.enzyme);
  if (sharedCtx.assembly_method) ctxLines.push('Method: ' + sharedCtx.assembly_method);
  if (sharedCtx.extra) ctxLines.push(sharedCtx.extra);

  var ctxHtml = ctxLines.length
    ? '<div style="font-size:12px;color:var(--sand-600);background:var(--sand-50,#fafaf9);border:1px solid var(--sand-100);border-radius:6px;padding:8px 10px;margin-bottom:12px">' +
        '<div style="font-weight:500;margin-bottom:3px;color:var(--sand-500)">Shared context — already resolved, will be reused:</div>' +
        ctxLines.map(function(l) { return '<div>· ' + escapeHtml(l) + '</div>'; }).join('') +
      '</div>'
    : '';

  // Build preview construct download section
  var prevExports = event.preview_exports || [];
  var exportsHtml = '';
  if (prevExports.length) {
    exportsHtml = '<div style="margin-bottom:12px;padding:8px 10px;background:var(--sand-50,#fafaf9);border:1px solid var(--sand-100);border-radius:6px">' +
      '<div style="font-size:12px;font-weight:500;color:var(--sand-500);margin-bottom:6px">Preview construct (#1):</div>' +
      prevExports.map(function(exp, ei) {
        var key = '_previewExp_' + cardId + '_' + ei;
        window[key] = exp;
        return '<div style="display:flex;align-items:center;gap:8px;font-size:12px">' +
          '<svg width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">' +
            '<path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/>' +
            '<polyline points="14 2 14 8 20 8"/>' +
          '</svg>' +
          '<a href="#" style="color:var(--brand-fig);text-decoration:none" ' +
            'onclick="(function(e){e.preventDefault();var d=window[\'' + key + '\'];if(!d)return;' +
              'var b=new Blob([d.content||\'\'],{type:\'text/plain\'});var u=URL.createObjectURL(b);' +
              'var a=document.createElement(\'a\');a.href=u;a.download=d.filename;a.click();URL.revokeObjectURL(u);})(event)">' +
            escapeHtml(exp.filename) +
          '</a>' +
        '</div>';
      }).join('') +
    '</div>';
  }

  // Build row list — each item is a data attribute row for live filter
  var rowChecks = remainingRows.map(function(r, i) {
    var num  = i + 2;  // construct number (1 was the preview)
    var name = r.name || '';
    var desc = (r.description || '').slice(0, 80);
    return '<label class="bulk-sel-row" data-num="' + num + '" data-name="' + escapeHtml(name).toLowerCase() + '" ' +
      'style="cursor:pointer;display:flex;align-items:center;gap:8px;padding:5px 8px;border-bottom:1px solid var(--sand-100)">' +
      '<input type="checkbox" class="bulk-preview-chk" data-idx="' + i + '" checked ' +
        'style="accent-color:var(--brand-fig);flex-shrink:0" onchange="onBulkPreviewChk(\'' + cardId + '\')">' +
      '<span style="font-size:11px;font-weight:600;color:var(--brand-fig);width:28px;flex-shrink:0;text-align:right">#' + num + '</span>' +
      '<span style="font-size:11px;color:var(--sand-500);max-width:100px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex-shrink:0">' + escapeHtml(name) + '</span>' +
      '<span style="font-size:12px;color:var(--sand-700);flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">' + escapeHtml(desc) + '</span>' +
    '</label>';
  }).join('');

  var modelOpts = _bulkModelOpts(defaultModel);

  // Token stats banner (shown when we have real data from the preview run)
  var tokIn  = _bulkPreviewTokens.in;
  var tokOut = _bulkPreviewTokens.out;
  var tokenStatHtml = (tokIn > 0)
    ? '<div style="font-size:11px;color:var(--sand-400);background:var(--sand-50,#fafaf9);border:1px solid var(--sand-100);border-radius:6px;padding:6px 10px;margin-bottom:10px;display:flex;gap:12px;align-items:center">' +
        '<svg width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>' +
        '<span>Preview used <strong>' + Math.round(tokIn / 1000) + 'k</strong> input + ' +
          '<strong>' + Math.round(tokOut / 1000) + 'k</strong> output tokens &mdash; cost estimates below use these actual counts</span>' +
      '</div>'
    : '';

  var rowsHtml = n > 0
    ? '<div id="' + cardId + '-rows-section" style="margin-bottom:14px">' +
        // Header: label + search + all/none
        '<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;flex-wrap:wrap">' +
          '<span style="font-size:12px;font-weight:500;color:var(--sand-500)">Constructs to run:</span>' +
          '<span id="' + cardId + '-selcount" style="font-size:12px;color:var(--brand-fig);font-weight:500">' + n + ' of ' + n + ' selected</span>' +
          '<div style="flex:1"></div>' +
          '<input id="' + cardId + '-filter" type="text" placeholder="Filter by # or name…" ' +
            'style="height:26px;padding:0 8px;border:1px solid var(--sand-200);border-radius:6px;font-size:12px;font-family:inherit;width:130px;box-sizing:border-box" ' +
            'oninput="filterBulkPreviewRows(\'' + cardId + '\')">' +
          '<button onclick="setBulkPreviewAll(\'' + cardId + '\',' + n + ',true)" ' +
            'style="height:26px;padding:0 10px;font-size:11px;font-weight:500;border:1px solid var(--sand-200);border-radius:6px;cursor:pointer;background:transparent;color:var(--sand-600);font-family:inherit">All</button>' +
          '<button onclick="setBulkPreviewAll(\'' + cardId + '\',' + n + ',false)" ' +
            'style="height:26px;padding:0 10px;font-size:11px;font-weight:500;border:1px solid var(--sand-200);border-radius:6px;cursor:pointer;background:transparent;color:var(--sand-600);font-family:inherit">None</button>' +
        '</div>' +
        // Construct #1 done badge
        '<div style="display:flex;align-items:center;gap:8px;padding:5px 8px;background:var(--sand-50,#fafaf9);border:1px solid var(--sand-100);border-radius:6px;margin-bottom:4px;font-size:12px;color:var(--sand-500)">' +
          '<svg width="14" height="14" fill="none" stroke="var(--brand-aqua)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>' +
          '<span style="font-weight:600;color:var(--brand-fig)">#1</span>' +
          '<span>Preview — already built</span>' +
        '</div>' +
        // Scrollable list
        '<div id="' + cardId + '-list" style="border:1px solid var(--sand-200);border-radius:8px;max-height:280px;overflow-y:auto">' + rowChecks + '</div>' +
      '</div>'
    : '<div style="font-size:13px;color:var(--sand-500);margin-bottom:14px">No remaining constructs.</div>';

  var bottomHtml = n > 0
    ? '<div id="' + cardId + '-tokenstat">' + tokenStatHtml + '</div>' +
      '<div id="' + cardId + '-controls" style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:12px">' +
        '<div>' +
          '<label style="font-size:12px;font-weight:500;color:var(--sand-500);display:block;margin-bottom:4px">Model</label>' +
          '<select id="' + cardId + '-model" class="model-select" style="font-size:12px;max-width:220px" ' +
            'onchange="onBulkPreviewChk(\'' + cardId + '\')">' + modelOpts + '</select>' +
        '</div>' +
        '<div id="' + cardId + '-cost" style="font-size:12px;margin-top:18px"></div>' +
      '</div>'
    : '';

  var actionsHtml = '<div id="' + cardId + '-actions" style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">' +
    (n > 0
      ? '<button class="send-btn" id="' + cardId + '-runbtn" style="width:auto;padding:0 18px;height:32px;font-size:13px;border-radius:10px" onclick="approveBulkPreview(\'' + cardId + '\')">' +
          'Run ' + n + ' construct' + (n === 1 ? '' : 's') +
        '</button>'
      : '') +
    '<button onclick="rerunBulkPreview(\'' + cardId + '\')" ' +
      'style="padding:0 14px;height:32px;font-size:13px;background:transparent;border:1px solid var(--sand-200);border-radius:10px;cursor:pointer;color:var(--sand-600);font-family:inherit" ' +
      'title="Something wrong? Fix it in chat then click here to rebuild construct #1">Rerun Preview</button>' +
    '<button onclick="delete _bulkPreviewData[\'' + cardId + '\']; document.getElementById(\'' + cardId + '\').remove()" ' +
      'style="padding:0 14px;height:32px;font-size:13px;background:transparent;border:1px solid var(--sand-200);border-radius:10px;cursor:pointer;color:var(--sand-600);font-family:inherit">Cancel</button>' +
  '</div>';

  var inner = getInner();
  var card  = document.createElement('div');
  card.className = 'msg assistant';
  card.id = cardId;
  card.innerHTML = '<div class="msg-bubble-assistant"><div class="bulk-plan-card">' +
    '<div class="bulk-plan-title">Preview complete' + (n > 0 ? ' — choose constructs to run' : '') + '</div>' +
    (summary ? '<div class="bulk-plan-summary">' + escapeHtml(summary) + '</div>' : '') +
    exportsHtml +
    ctxHtml +
    rowsHtml +
    bottomHtml +
    actionsHtml +
  '</div></div>';
  inner.appendChild(card);
  scrollToBottom();
  if (n > 0) onBulkPreviewChk(cardId);
}

// Shared cost projection: actual preview token totals when available, else a rough complexity-based guess.
function _bulkComputeCost(n, model) {
  var cost;
  var basedOnActual = _bulkPreviewMarginalTokens.in > 0;
  if (basedOnActual) {
    var pricing = _BULK_MODEL_PRICING[model] || _BULK_MODEL_PRICING['claude-sonnet-4-6'];
    var cpr = (_bulkPreviewMarginalTokens.in * pricing[0] + _bulkPreviewMarginalTokens.out * pricing[1]) / 1000000;
    cost = Math.round(cpr * n * 10000) / 10000;
  } else {
    cost = _estimateBulkCost(n, model, 'standard');
  }
  var cls = cost >= BULK_COST_SPLIT ? 'orange' : cost >= BULK_COST_WARN ? 'yellow' : 'ok';
  var constructWord = n === 1 ? 'construct' : 'constructs';
  var lbl = n === 0
    ? 'No constructs selected'
    : (cost < 0.01 ? '< $0.01' : '~$' + cost.toFixed(2)) +
      ' for ' + n + ' ' + constructWord +
      (basedOnActual ? ' (based on preview)' : ' (rough estimate)');
  return {cost: cost, cls: cls, label: lbl, basedOnActual: basedOnActual};
}

// Called whenever a checkbox changes or model changes — updates count, cost, run button
function onBulkPreviewChk(cardId) {
  var card = document.getElementById(cardId);
  if (!card) return;
  var checked = card.querySelectorAll('.bulk-preview-chk:checked').length;
  var total   = card.querySelectorAll('.bulk-preview-chk').length;
  var countEl = document.getElementById(cardId + '-selcount');
  if (countEl) countEl.textContent = checked + ' of ' + total + ' selected';
  var runBtn = document.getElementById(cardId + '-runbtn');
  if (runBtn) {
    runBtn.textContent = 'Run ' + checked + ' construct' + (checked === 1 ? '' : 's');
    runBtn.disabled = (checked === 0);
    runBtn.style.opacity = checked === 0 ? '0.4' : '1';
  }
  var sel   = document.getElementById(cardId + '-model');
  var model = sel ? sel.value : (_bulkPreviewModel || 'claude-sonnet-4-6');
  var result = _bulkComputeCost(checked, model);
  var costEl = document.getElementById(cardId + '-cost');
  if (costEl) {
    costEl.innerHTML = '<div class="bulk-plan-cost ' + (checked > 0 ? result.cls : '') + '">' +
      (checked > 0 ? '<svg width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg> ' : '') +
      result.label + '</div>';
  }
}

// Filter the construct list by number or name
function filterBulkPreviewRows(cardId) {
  var input = document.getElementById(cardId + '-filter');
  if (!input) return;
  var q = input.value.trim().toLowerCase();
  var list = document.getElementById(cardId + '-list');
  if (!list) return;
  list.querySelectorAll('.bulk-sel-row').forEach(function(row) {
    var num  = row.getAttribute('data-num') || '';
    var name = row.getAttribute('data-name') || '';
    var show = !q || ('#' + num).includes(q) || num.includes(q) || name.includes(q);
    row.style.display = show ? '' : 'none';
  });
}

// Set all VISIBLE rows checked or unchecked
function setBulkPreviewAll(cardId, total, checked) {
  var card = document.getElementById(cardId);
  if (!card) return;
  card.querySelectorAll('.bulk-sel-row').forEach(function(row) {
    if (row.style.display === 'none') return;
    var chk = row.querySelector('.bulk-preview-chk');
    if (chk) chk.checked = checked;
  });
  onBulkPreviewChk(cardId);
}

// Legacy alias kept for CSV-upload flow
function toggleBulkPreviewAll(cardId, checked) { setBulkPreviewAll(cardId, 0, checked); }

function approveBulkPreview(cardId) {
  var data = _bulkPreviewData[cardId];
  if (!data) return;
  var modelEl = document.getElementById(cardId + '-model');
  var model   = modelEl ? modelEl.value : 'claude-sonnet-4-6';
  var card    = document.getElementById(cardId);

  // Collect selected rows
  var selectedRows = [];
  if (card) {
    card.querySelectorAll('.bulk-preview-chk:checked').forEach(function(chk) {
      var idx = parseInt(chk.getAttribute('data-idx'), 10);
      if (data.rows[idx]) selectedRows.push(data.rows[idx]);
    });
  }

  if (!selectedRows.length) {
    alert('No constructs selected.');
    return;
  }

  // Swap the selection list / model picker / action buttons for a submitting
  // spinner, but leave the summary, shared-context, and token-usage sections
  // in place so the card still shows what was reused once the rest of the
  // batch is running in the background.
  var statusId = cardId + '-status';
  if (card) {
    ['rows-section', 'controls', 'actions'].forEach(function(suffix) {
      var el = document.getElementById(cardId + '-' + suffix);
      if (el) el.remove();
    });
    var bc = card.querySelector('.bulk-plan-card');
    if (bc) {
      var status = document.createElement('div');
      status.id = statusId;
      status.style.cssText = 'color:var(--sand-500);font-size:13px';
      status.innerHTML = '<span class="streaming-cursor"></span> Submitting ' + selectedRows.length + ' construct' + (selectedRows.length === 1 ? '' : 's') + '&hellip;';
      bc.appendChild(status);
    }
    var titleEl = card.querySelector('.bulk-plan-title');
    if (titleEl) titleEl.textContent = 'Preview complete';
  }

  // Build enriched prompts embedding the shared context
  var enrichedRows = selectedRows.map(function(r) {
    return {
      name:          r.name || '',
      description:   buildEnrichedPrompt(r.description, data.sharedCtx),
      output_format: 'genbank',
    };
  });

  delete _bulkPreviewData[cardId];

  fetch('/api/bulk/run', {
    method:  'POST',
    headers: {'Content-Type': 'application/json'},
    body:    JSON.stringify({
      enriched_rows:   enrichedRows,
      model:           model,
      filename:        'bulk_design.csv',
      preview_exports: _bulkPreviewExports,
      session_id:      currentSessionId,
    }),
  })
  .then(function(r) { return r.json(); })
  .then(function(data) {
    if (data.error) { alert('Error: ' + data.error); return; }
    var statusEl = document.getElementById(statusId);
    if (statusEl) {
      var result = _bulkComputeCost(selectedRows.length, model);
      statusEl.innerHTML =
        '<div style="display:flex;align-items:center;gap:6px;color:var(--brand-aqua);font-size:13px;margin-bottom:6px">' +
          '<svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>' +
          'Submitted ' + selectedRows.length + ' construct' + (selectedRows.length === 1 ? '' : 's') + ' — running in background' +
        '</div>' +
        '<div class="bulk-plan-cost ' + result.cls + '">' + result.label + '</div>';
    }
    var jobId = data.job_id;
    var activeSid = currentSessionId || ('bulk-' + jobId);
    initBatchCards(jobId, data.row_count, data.filename || 'bulk_design.csv', model);
    loadSessions();
    _batchSessions[activeSid] = jobId;
    if (_batchPollTimers[activeSid]) clearInterval(_batchPollTimers[activeSid]);
    _batchPollTimers[activeSid] = setInterval(function() { pollBatchForSession(activeSid); }, 2000);
    pollBatchForSession(activeSid);
  })
  .catch(function(e) { alert('Failed to submit bulk run: ' + e); });
}

// Called when user clicks "Rerun Preview" — removes the approval card and asks
// the agent to rebuild construct #1 with any corrections already discussed.
function rerunBulkPreview(cardId) {
  var data = _bulkPreviewData[cardId];
  if (!data) return;

  // Build a compact summary of the shared context already in history so the
  // agent can fast-path (no re-fetching backbone/insertion site).
  var ctx = data.sharedCtx || {};
  var ctxParts = [];
  if (ctx.backbone_id)   ctxParts.push('backbone_id=' + ctx.backbone_id);
  if (ctx.backbone_name) ctxParts.push('backbone=' + ctx.backbone_name);
  if (ctx.insertion_site_start != null)
    ctxParts.push('insertion_site=' + ctx.insertion_site_start + (ctx.insertion_site_end ? '-' + ctx.insertion_site_end : ''));
  if (ctx.enzyme)           ctxParts.push('enzyme=' + ctx.enzyme);
  if (ctx.assembly_method)  ctxParts.push('method=' + ctx.assembly_method);

  // Store remaining rows globally so the agent's complete_bulk_preview can
  // reference them; we pass them via the rerun message too.
  var remainingRows = data.rows || [];

  // Remove the approval card and reset export buffer for the fresh run
  var card = document.getElementById(cardId);
  if (card) card.remove();
  delete _bulkPreviewData[cardId];
  _bulkPreviewExports = [];

  // Show the model card again (user can change model for the rerun)
  showBulkPreviewModelCard({n_constructs: remainingRows.length, preview_model: _bulkPreviewModel});

  // Pre-fill a rerun message (startBulkPreviewRun will send it)
  window._bulkRerunMsg = 'Please rebuild the preview construct incorporating the corrections we just discussed.\n' +
    (ctxParts.length ? 'Use the shared context already found (do not re-fetch): ' + ctxParts.join(', ') + '.\n' : '') +
    'After rebuilding and exporting, call complete_bulk_preview again with the same ' + remainingRows.length + ' remaining construct(s).';
}

// ── Bulk design planning flow ────────────────────────────────────────────

// Client-side pricing mirrors bulk_planner.py so cost updates instantly on model change
var _BULK_MODEL_PRICING = {
  'claude-haiku-4-5-20251001': [1.00,  5.00],
  'claude-sonnet-4-6':          [3.00, 15.00],
  'claude-opus-4-6':            [5.00, 25.00],
  'claude-opus-4-7':            [5.00, 25.00],
};
var _BULK_TOKENS_BY_COMPLEXITY = {
  'simple':   [200000,  1400],
  'standard': [300000,  8000],
  'complex':  [450000, 17000],
};
var BULK_COST_WARN  = 5.0;
var BULK_COST_SPLIT = 20.0;

function _estimateBulkCost(nRows, model, complexity) {
  var pricing = _BULK_MODEL_PRICING[model] || _BULK_MODEL_PRICING['claude-sonnet-4-6'];
  var tokens  = _BULK_TOKENS_BY_COMPLEXITY[complexity] || _BULK_TOKENS_BY_COMPLEXITY['standard'];
  var cpr = (tokens[0] * pricing[0] + tokens[1] * pricing[1]) / 1000000;
  return Math.round(cpr * nRows * 10000) / 10000;
}

// plan_id → [{name, description}, ...] for subset selection
var _bulkPlanRows = {};

var _bulkEntryMenuOpen = false;

function showBulkEntryMenu(e) {
  e.stopPropagation();
  var existing = document.getElementById('bulk-entry-menu');
  if (existing) { existing.remove(); _bulkEntryMenuOpen = false; return; }
  _bulkEntryMenuOpen = true;
  var btn = document.getElementById('bulk-design-btn');
  var wrap = btn.closest('.input-buttons') || btn.parentElement;
  var menu = document.createElement('div');
  menu.className = 'bulk-entry-menu';
  menu.id = 'bulk-entry-menu';
  menu.innerHTML =
    '<button onclick="closeBulkEntryMenu();document.getElementById(\'batch-csv-input\').click()">' +
      '<svg width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>' +
      'Upload CSV' +
    '</button>' +
    '<button onclick="closeBulkEntryMenu();startBulkChatMode()">' +
      '<svg width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>' +
      'Describe in chat' +
    '</button>';
  wrap.style.position = 'relative';
  wrap.appendChild(menu);
  setTimeout(function() {
    document.addEventListener('click', closeBulkEntryMenuOnBlur);
  }, 0);
}

function closeBulkEntryMenu() {
  var m = document.getElementById('bulk-entry-menu');
  if (m) m.remove();
  _bulkEntryMenuOpen = false;
  document.removeEventListener('click', closeBulkEntryMenuOnBlur);
}

function closeBulkEntryMenuOnBlur() { closeBulkEntryMenu(); }

function startBulkChatMode() {
  // Inject a primer into the chat textarea so the user knows what to do
  var ta = document.getElementById('user-input');
  if (ta) {
    ta.value = 'I want to design multiple constructs in bulk. ';
    ta.focus();
    ta.setSelectionRange(ta.value.length, ta.value.length);
    if (typeof autoResize === 'function') autoResize(ta);
  }
}

// Legacy SSE path: agent fired the old submit_bulk_designs ([BULK_DESIGNS_READY]).
// Route through chat so the agent handles it like typed input on the next turn.
function requestBulkPlanFromRows(rows, model) {
  if (!rows || !rows.length) return;
  var lines = ['Please design the following ' + rows.length + ' construct' +
    (rows.length === 1 ? '' : 's') + ' in bulk:'];
  rows.forEach(function(r, i) {
    lines.push((i + 1) + '. ' + (r.name ? r.name + ': ' : '') + (r.description || ''));
  });
  inputEl.value = lines.join('\n');
  autoResize(inputEl);
  sendMessage();
}

// Core planning flow — called when a CSV is uploaded via drag-drop or button.
// Routes through /api/bulk/plan instead of showing the old batch confirm dialog.
function requestBulkPlan(csvText, filename) {
  var parsed = _parseCSVRows(csvText);
  if (!parsed.rows.length) {
    hideWelcome();
    var errCard = document.createElement('div');
    errCard.className = 'msg assistant';
    errCard.innerHTML = '<div class="msg-bubble-assistant" style="color:var(--sand-500);font-size:13px">' +
      'Could not parse any rows from <strong>' + escapeHtml(filename) + '</strong>. ' +
      'Make sure the CSV has a <code>description</code> column (and optionally a <code>name</code> column).' +
    '</div>';
    getInner().appendChild(errCard);
    scrollToBottom();
    return;
  }
  // Send as a chat message so the agent handles it like typed input
  var lines = ['Please design the following ' + parsed.rows.length + ' construct' +
    (parsed.rows.length === 1 ? '' : 's') + ' in bulk (from ' + filename + '):'];
  parsed.rows.forEach(function(r, i) {
    lines.push((i + 1) + '. ' + (r.name ? r.name + ': ' : '') + r.description);
  });
  inputEl.value = lines.join('\n');
  autoResize(inputEl);
  sendMessage();
}

function showBulkPlanCard(plan, csvText, filename, rows) {
  hideWelcome();
  var inner = getInner();
  var cardId = 'bulk-plan-' + Date.now();
  _bulkPlanContext[cardId] = {csvText: csvText || null, filename: filename || 'bulk_design.csv', rows: rows || null};
  // Store rows keyed by plan_id for subset selection later
  _bulkPlanRows[plan.plan_id] = plan.rows || [];
  var n = plan.rows.length;
  // Default to Sonnet even if the planner suggested Haiku
  var defaultModel = (plan.model_suggestion === 'claude-haiku-4-5-20251001')
    ? 'claude-sonnet-4-6' : (plan.model_suggestion || 'claude-sonnet-4-6');
  var complexity = plan.complexity || 'standard';

  var modelOpts = [
    ['claude-opus-4-7',          'Opus 4.7 — most capable'],
    ['claude-opus-4-6',          'Opus 4.6'],
    ['claude-sonnet-4-6',        'Sonnet 4.6 — recommended for bulk'],
    ['claude-haiku-4-5-20251001','Haiku 4.5 — fastest / cheapest'],
  ].map(function(o) {
    var sel = o[0] === defaultModel ? ' selected' : '';
    return '<option value="' + o[0] + '"' + sel + '>' + o[1] + '</option>';
  }).join('');

  function buildCostHtml(model) {
    var cost = _estimateBulkCost(n, model, complexity);
    var cls  = cost >= BULK_COST_SPLIT ? 'orange' : cost >= BULK_COST_WARN ? 'yellow' : 'ok';
    var lbl  = cost < 0.01 ? '< $0.01 estimated' : '~$' + cost.toFixed(2) + ' estimated';
    var warn = '';
    if (cls === 'orange') {
      warn = '<div id="' + cardId + '-cost-warn" style="font-size:12px;color:#9a3412;margin-top:6px;margin-bottom:10px">⚠ High cost estimate.</div>';
    } else if (cls === 'yellow') {
      warn = '<div id="' + cardId + '-cost-warn" style="font-size:12px;color:#854d0e;margin-top:6px;margin-bottom:10px">⚠ Cost is above $5 — consider using Haiku to reduce costs.</div>';
    } else {
      warn = '<div id="' + cardId + '-cost-warn"></div>';
    }
    return { cls: cls, lbl: lbl, warn: warn };
  }

  var costInfo = buildCostHtml(defaultModel);
  var batchBadge = plan.batch_eligible
    ? '<div style="display:inline-flex;align-items:center;gap:5px;font-size:11px;font-weight:500;padding:3px 9px;background:#f0fdf4;color:#166534;border-radius:6px;margin-bottom:10px">' +
        '<svg width="11" height="11" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>' +
        'Batch mode — all ' + n + ' constructs run in one agent call' +
      '</div><div style="font-size:12px;color:var(--sand-500);margin-bottom:12px">' + escapeHtml(plan.shared_context || '') + '</div>'
    : '';

  var rowsHtml = plan.rows.map(function(r, i) {
    return '<div class="bulk-plan-row">' +
      '<span class="bulk-plan-row-num">' + (i + 1) + '</span>' +
      '<span class="bulk-plan-row-name">' + escapeHtml(r.name || '') + '</span>' +
      '<span class="bulk-plan-row-desc">' + escapeHtml(r.description || '') + '</span>' +
    '</div>';
  }).join('');

  var card = document.createElement('div');
  card.className = 'msg assistant';
  card.id = cardId;
  card.innerHTML = '<div class="msg-bubble-assistant"><div class="bulk-plan-card">' +
    '<div class="bulk-plan-title">' + escapeHtml(filename) + ' · ' + n + ' design' + (n === 1 ? '' : 's') + '</div>' +
    '<div class="bulk-plan-summary">' + escapeHtml(plan.summary) + '</div>' +
    batchBadge +
    '<div id="' + cardId + '-cost" class="bulk-plan-cost ' + costInfo.cls + '">' +
      '<svg width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>' +
      '<span id="' + cardId + '-cost-lbl">' + costInfo.lbl + '</span>' +
    '</div>' +
    costInfo.warn +
    '<div class="bulk-plan-rows-wrap">' + rowsHtml + '</div>' +
    '<div style="margin-bottom:14px">' +
      '<label style="font-size:12px;font-weight:500;color:var(--sand-500);display:block;margin-bottom:5px">Model for full run</label>' +
      '<select id="' + cardId + '-model" class="model-select" style="font-size:12px;max-width:100%" onchange="updateBulkPlanCost(\'' + cardId + '\',' + n + ',\'' + complexity + '\')">' + modelOpts + '</select>' +
    '</div>' +
    '<div id="' + cardId + '-context-wrap" class="bulk-context-area">' +
      '<label style="font-size:12px;font-weight:500;color:var(--sand-500);display:block;margin-bottom:5px">Additional context</label>' +
      '<textarea id="' + cardId + '-context-text" placeholder="e.g. &quot;All constructs need a WPRE terminator&quot; or &quot;Use Haiku 4.5 for cost savings&quot;"></textarea>' +
      '<div style="display:flex;gap:8px;margin-top:8px">' +
        '<button class="send-btn" style="width:auto;padding:0 14px;height:30px;font-size:12px;border-radius:8px" onclick="replanWithContext(\'' + cardId + '\',\'' + plan.plan_id + '\')">Re-analyse</button>' +
        '<button onclick="document.getElementById(\'' + cardId + '-context-wrap\').style.display=\'none\'" style="padding:0 12px;height:30px;font-size:12px;background:transparent;border:1px solid var(--sand-200);border-radius:8px;cursor:pointer;color:var(--sand-600);font-family:inherit">Cancel</button>' +
      '</div>' +
    '</div>' +
    '<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">' +
      '<button class="send-btn" style="width:auto;padding:0 16px;height:32px;font-size:13px;border-radius:10px" onclick="approveBulkPlan(\'' + cardId + '\',\'' + plan.plan_id + '\')">' +
        'Run sample design' +
      '</button>' +
      '<button onclick="document.getElementById(\'' + cardId + '-context-wrap\').style.display=\'block\'" style="padding:0 14px;height:32px;font-size:13px;background:transparent;border:1px solid var(--sand-200);border-radius:10px;cursor:pointer;color:var(--sand-600);font-family:inherit">Add context / Modify</button>' +
      '<button onclick="cancelBulkPlan(\'' + cardId + '\')" style="padding:0 10px;height:32px;font-size:13px;background:transparent;border:none;cursor:pointer;color:var(--sand-400);font-family:inherit">Cancel</button>' +
    '</div>' +
  '</div></div>';
  inner.appendChild(card);
  scrollToBottom();
}

function updateBulkPlanCost(cardId, nRows, complexity) {
  var sel = document.getElementById(cardId + '-model');
  if (!sel) return;
  var model = sel.value;
  var cost  = _estimateBulkCost(nRows, model, complexity);
  var cls   = cost >= BULK_COST_SPLIT ? 'orange' : cost >= BULK_COST_WARN ? 'yellow' : 'ok';
  var lbl   = cost < 0.01 ? '< $0.01 estimated' : '~$' + cost.toFixed(2) + ' estimated';
  var costEl = document.getElementById(cardId + '-cost');
  var lblEl  = document.getElementById(cardId + '-cost-lbl');
  var warnEl = document.getElementById(cardId + '-cost-warn');
  if (costEl) { costEl.className = 'bulk-plan-cost ' + cls; }
  if (lblEl)  { lblEl.textContent = lbl; }
  if (warnEl) {
    if (cls === 'orange') warnEl.textContent = '⚠ High cost estimate.';
    else if (cls === 'yellow') warnEl.textContent = '⚠ Cost is above $5 — consider using Haiku to reduce costs.';
    else warnEl.textContent = '';
  }
}

function cancelBulkPlan(cardId) {
  delete _bulkPlanContext[cardId];
  var card = document.getElementById(cardId);
  if (card) card.remove();
}

function replanWithContext(cardId, oldPlanId) {
  var contextText = (document.getElementById(cardId + '-context-text') || {}).value || '';
  var ctx         = _bulkPlanContext[cardId] || {};
  var csvText     = ctx.csvText || null;
  var rows        = ctx.rows || null;
  var filename    = ctx.filename || 'bulk_design.csv';
  var model       = (document.getElementById(cardId + '-model') || {}).value || 'claude-sonnet-4-6';

  delete _bulkPlanContext[cardId];
  var card = document.getElementById(cardId);
  if (card) card.remove();

  // Show loading
  hideWelcome();
  var inner = getInner();
  var loadingId = 'bulk-loading-' + Date.now();
  var lc = document.createElement('div');
  lc.className = 'msg assistant';
  lc.id = loadingId;
  lc.innerHTML = '<div class="msg-bubble-assistant" style="color:var(--sand-500);font-size:13px"><span class="streaming-cursor"></span> Re-analysing with updated context&hellip;</div>';
  inner.appendChild(lc);
  scrollToBottom();

  var replanBody = {user_context: contextText, model: model, filename: filename};
  if (csvText) replanBody.csv_content = csvText;
  else if (rows) replanBody.rows = rows;

  fetch('/api/bulk/plan', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(replanBody),
  })
  .then(function(r) { return r.json(); })
  .then(function(plan) {
    var l = document.getElementById(loadingId); if (l) l.remove();
    if (plan.error) { alert('Planning error: ' + plan.error); return; }
    showBulkPlanCard(plan, csvText, filename, rows);
  })
  .catch(function(e) {
    var l = document.getElementById(loadingId); if (l) l.remove();
    alert('Re-analysis failed: ' + e);
  });
}

function approveBulkPlan(cardId, planId) {
  var modelEl = document.getElementById(cardId + '-model');
  var model = modelEl ? modelEl.value : 'claude-sonnet-4-6';
  var card = document.getElementById(cardId);

  // Replace plan card content with "running sample" state
  var planCard = card ? card.querySelector('.bulk-plan-card') : null;
  if (planCard) {
    planCard.innerHTML = '<div style="color:var(--sand-500);font-size:13px"><span class="streaming-cursor"></span> Running sample design&hellip;</div>';
  }

  fetch('/api/bulk/sample', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({plan_id: planId, model: model}),
  })
  .then(function(r) { return r.json(); })
  .then(function(data) {
    if (data.error) {
      if (planCard) planCard.innerHTML = '<div style="color:red;font-size:13px">Error: ' + escapeHtml(data.error) + '</div>';
      return;
    }
    // Re-ID the spinner card so updateBulkSampleProgress can find it by job ID
    if (card) card.id = 'bulk-sample-' + data.job_id;
    pollBulkSample(data.job_id, planId, model, data.session_id);
  })
  .catch(function(e) {
    if (planCard) planCard.innerHTML = '<div style="color:red;font-size:13px">Request failed: ' + e + '</div>';
  });
}

// Poll the sample job (single-row batch) and render/update result inline in chat
function pollBulkSample(sampleJobId, planId, model, sessionId) {
  var pollTimer = null;

  function checkSample() {
    fetch('/api/batch/' + sampleJobId)
      .then(function(r) { return r.json(); })
      .then(function(job) {
        if (!job.rows || !job.rows.length) return;
        var row = job.rows[0];
        if (row.status === 'done' || row.status === 'error' || row.status === 'no_export') {
          clearInterval(pollTimer);
          renderBulkSampleResult(row, sampleJobId, planId, model);
        } else if (row.status === 'running') {
          // Show live activity so the user knows the agent is working
          updateBulkSampleProgress(sampleJobId, row.log || []);
        }
      })
      .catch(function() {});
  }
  pollTimer = setInterval(checkSample, 2000);
  checkSample();
}

function updateBulkSampleProgress(sampleJobId, log) {
  var card = document.getElementById('bulk-sample-' + sampleJobId);
  if (!card) return;
  var pc = card.querySelector('.bulk-plan-card');
  if (!pc) return;

  // Show rich batch-card log so users see tool calls as they happen
  pc.innerHTML =
    '<div class="bulk-plan-title" style="margin-bottom:6px">Sample design running…</div>' +
    '<div style="color:var(--sand-500);font-size:13px;margin-bottom:8px"><span class="streaming-cursor"></span> Working on your design</div>' +
    '<div class="batch-row-log open" style="max-height:280px;overflow-y:auto;border:1px solid var(--sand-100);border-radius:8px;padding:6px 8px">' +
      renderBatchLog(log) +
    '</div>';
  // Auto-scroll to bottom of the log
  var logEl = pc.querySelector('.batch-row-log');
  if (logEl) logEl.scrollTop = logEl.scrollHeight;
}

function renderBulkSampleResult(row, sampleJobId, planId, model) {
  hideWelcome();
  var inner = getInner();
  // Reuse existing card if present (re-render after continuation)
  var existingCard = document.getElementById('bulk-sample-' + sampleJobId);
  var sampleCardId = existingCard ? existingCard.id : 'bulk-sample-' + sampleJobId;

  var succeeded = row.status === 'done' && row.exports && row.exports.length > 0;
  var needsInput = row.status === 'no_export';

  var exportsHtml = '';
  if (row.exports && row.exports.length) {
    exportsHtml = '<div style="margin:10px 0 4px;font-size:12px;font-weight:500;color:var(--sand-500)">Sample output</div>' +
      row.exports.map(function(exp, ei) {
        return '<div style="display:flex;align-items:center;gap:8px;font-size:12px;margin-bottom:6px">' +
          '<svg width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>' +
          '<a href="/api/batch/' + sampleJobId + '/download/0/' + ei + '" style="color:var(--brand-fig);text-decoration:none" download>' + escapeHtml(exp.filename) + '</a>' +
        '</div>';
      }).join('');
  } else if (row.status === 'error') {
    exportsHtml = '<div style="color:#b91c1c;font-size:12px;margin-top:8px;padding:8px;background:#fef2f2;border-radius:6px">' +
      '⚠ Sample failed: ' + escapeHtml(row.error || 'unknown error') + '</div>';
  }

  // Show the last text the agent produced (usually a question when no_export)
  var lastText = '';
  if (row.log) {
    for (var i = row.log.length - 1; i >= 0; i--) {
      if (row.log[i].type === 'text') { lastText = row.log[i].content; break; }
    }
  }
  var summaryHtml = lastText
    ? '<div style="font-size:13px;color:var(--sand-700);margin-bottom:10px;white-space:pre-wrap;line-height:1.5;border-left:3px solid var(--sand-200);padding-left:10px">' +
        escapeHtml(lastText.slice(0, 800)) + (lastText.length > 800 ? '…' : '') +
      '</div>'
    : '';

  // Continuation input — shown when agent asked a question but didn't finish
  var continuationHtml = '';
  if (needsInput) {
    continuationHtml =
      '<div style="margin-top:12px;padding:10px;background:var(--sand-50,#fafaf9);border:1px solid var(--sand-200);border-radius:8px">' +
        '<div style="font-size:12px;font-weight:500;color:var(--sand-500);margin-bottom:6px">The agent needs more information to complete this design:</div>' +
        '<textarea id="bulk-sample-input-' + sampleJobId + '" rows="2" ' +
          'style="width:100%;box-sizing:border-box;border:1px solid var(--sand-200);border-radius:6px;padding:7px 10px;font-size:13px;font-family:inherit;resize:vertical" ' +
          'placeholder="Type your answer here…" ' +
          'onkeydown="if(event.key===\'Enter\'&&!event.shiftKey){event.preventDefault();continueBulkSample(\'' + sampleJobId + '\',\'' + planId + '\',\'' + model + '\')}"></textarea>' +
        '<div style="display:flex;gap:8px;margin-top:6px">' +
          '<button class="send-btn" style="width:auto;padding:0 14px;height:28px;font-size:12px;border-radius:7px" ' +
            'onclick="continueBulkSample(\'' + sampleJobId + '\',\'' + planId + '\',\'' + model + '\')">' +
            'Send' +
          '</button>' +
        '</div>' +
      '</div>';
  }

  // Build subset selection for rows 1..n (row 0 was the sample)
  var subsetHtml = '';
  if (succeeded) {
    var planRows = _bulkPlanRows[planId] || [];
    var remaining = planRows.slice(1);  // rows 1..n
    if (remaining.length > 0) {
      var checkboxId = sampleCardId + '-subset';
      var rowChecks = remaining.map(function(r, i) {
        var idx = i + 1;  // original index
        return '<label class="batch-confirm-row" style="cursor:pointer;display:flex;align-items:center;gap:8px;padding:6px 8px;border-bottom:1px solid var(--sand-100)">' +
          '<input type="checkbox" class="bulk-subset-chk" data-idx="' + idx + '" checked style="accent-color:var(--brand-fig)">' +
          '<span class="batch-confirm-row-num" style="font-size:11px;color:var(--sand-400);width:20px">' + (idx + 1) + '</span>' +
          '<span class="batch-confirm-row-name" style="font-size:11px;color:var(--sand-400);max-width:100px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">' + escapeHtml(r.name || '') + '</span>' +
          '<span class="batch-confirm-row-desc" style="font-size:12px;color:var(--sand-700);flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">' + escapeHtml((r.description || '').slice(0, 80)) + '</span>' +
        '</label>';
      }).join('');
      subsetHtml =
        '<div style="margin-top:14px">' +
          '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px">' +
            '<span style="font-size:12px;font-weight:500;color:var(--sand-500)">Remaining constructs to run:</span>' +
            '<label style="font-size:11px;color:var(--sand-400);cursor:pointer;display:flex;align-items:center;gap:4px">' +
              '<input type="checkbox" id="' + checkboxId + '-all" checked onchange="toggleBulkSubsetAll(\'' + sampleCardId + '\',this.checked)" style="accent-color:var(--brand-fig)"> Select all' +
            '</label>' +
          '</div>' +
          '<div class="batch-confirm-rows" style="border:1px solid var(--sand-200);border-radius:8px;max-height:200px;overflow-y:auto" id="' + checkboxId + '-list">' +
            rowChecks +
          '</div>' +
        '</div>';
    }
  }

  var actionsHtml = '<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:14px">' +
    (succeeded
      ? '<button class="send-btn" id="' + sampleCardId + '-run-btn" style="width:auto;padding:0 16px;height:32px;font-size:13px;border-radius:10px" onclick="approveBulkSample(\'' + sampleCardId + '\',\'' + planId + '\',\'' + sampleJobId + '\',\'' + model + '\')">' +
          'Run selected' +
        '</button>'
      : '') +
    '<button onclick="document.getElementById(\'' + sampleCardId + '\').remove()" style="padding:0 14px;height:32px;font-size:13px;background:transparent;border:1px solid var(--sand-200);border-radius:10px;cursor:pointer;color:var(--sand-600);font-family:inherit">Cancel</button>' +
  '</div>';

  var innerHtml = '<div class="msg-bubble-assistant"><div class="bulk-plan-card">' +
    '<div class="bulk-plan-title">Sample design: ' + escapeHtml(row.name || 'construct 1') + '</div>' +
    summaryHtml +
    exportsHtml +
    continuationHtml +
    subsetHtml +
    actionsHtml +
  '</div></div>';

  if (existingCard) {
    existingCard.innerHTML = innerHtml;
  } else {
    var card = document.createElement('div');
    card.className = 'msg assistant';
    card.id = sampleCardId;
    card.innerHTML = innerHtml;
    inner.appendChild(card);
  }
  scrollToBottom();
}

function toggleBulkSubsetAll(sampleCardId, checked) {
  var card = document.getElementById(sampleCardId);
  if (!card) return;
  card.querySelectorAll('.bulk-subset-chk').forEach(function(chk) { chk.checked = checked; });
}

function continueBulkSample(sampleJobId, planId, model) {
  var inputEl = document.getElementById('bulk-sample-input-' + sampleJobId);
  if (!inputEl) return;
  var message = inputEl.value.trim();
  if (!message) return;

  // Disable input while waiting
  inputEl.disabled = true;
  inputEl.value = '';

  fetch('/api/batch/' + sampleJobId + '/rows/0/continue', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({message: message}),
  })
  .then(function(r) { return r.json(); })
  .then(function(data) {
    if (data.error) { alert('Error: ' + data.error); if (inputEl) inputEl.disabled = false; return; }
    // Clear the card to a base spinner; pollBulkSample will update it with live activity
    var card = document.getElementById('bulk-sample-' + sampleJobId);
    var pc = card ? card.querySelector('.bulk-plan-card') : null;
    if (pc) {
      pc.innerHTML =
        '<div class="bulk-plan-title" style="margin-bottom:6px">Sample design running…</div>' +
        '<div style="color:var(--sand-500);font-size:13px"><span class="streaming-cursor"></span> Continuing design — tool activity will appear here</div>';
    }
    // Resume polling — live log updates every 2 s
    pollBulkSample(sampleJobId, planId, model, null);
  })
  .catch(function(e) { alert('Failed: ' + e); if (inputEl) inputEl.disabled = false; });
}

function approveBulkSample(sampleCardId, planId, sampleJobId, model) {
  // Collect which remaining rows the user selected (0-based original indices)
  var card = document.getElementById(sampleCardId);
  var selectedIndices = [0];  // row 0 is always included (the sample)
  if (card) {
    card.querySelectorAll('.bulk-subset-chk:checked').forEach(function(chk) {
      selectedIndices.push(parseInt(chk.getAttribute('data-idx'), 10));
    });
    var bc = card.querySelector('.bulk-plan-card');
    if (bc) bc.innerHTML = '<div style="color:var(--sand-500);font-size:13px"><span class="streaming-cursor"></span> Submitting full run&hellip;</div>';
  }

  fetch('/api/bulk/run', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      plan_id: planId,
      model: model,
      sample_job_id: sampleJobId,
      selected_indices: selectedIndices,
    }),
  })
  .then(function(r) { return r.json(); })
  .then(function(data) {
    if (data.error) { alert('Error: ' + data.error); return; }
    if (card) card.remove();
    var jobId = data.job_id;
    // Render batch cards inline in the CURRENT session (no session switch)
    var activeSid = currentSessionId || ('bulk-' + jobId);
    initBatchCards(jobId, data.row_count, data.filename || 'bulk_design.csv', model);
    // Refresh sidebar to show the new persisted batch entry without switching sessions
    loadSessions();
    _batchSessions[activeSid] = jobId;
    if (_batchPollTimers[activeSid]) clearInterval(_batchPollTimers[activeSid]);
    _batchPollTimers[activeSid] = setInterval(function() { pollBatchForSession(activeSid); }, 2000);
    pollBatchForSession(activeSid);
  })
  .catch(function(e) { alert('Full run failed: ' + e); });
}

function uploadBatchCSV(csvText, filename, model) {
  model = model || modelSelect.value;
  fetch('/api/batch', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({csv_content: csvText, model: model, filename: filename}),
  })
  .then(function(r) { return r.json(); })
  .then(function(data) {
    if (data.error) { alert('Error: ' + data.error); return; }
    var sid = data.session_id;
    var jobId = data.job_id;
    // Switch to the dedicated batch session
    saveSessionId(sid);
    loadSessions();
    // Clear any stale DOM content from the previous session
    messagesEl.innerHTML = '';
    // Render batch cards into the new session's container
    initBatchCards(jobId, data.row_count, filename, model);
    // Track and start polling per-session
    _batchSessions[sid] = jobId;
    if (_batchPollTimers[sid]) clearInterval(_batchPollTimers[sid]);
    _batchPollTimers[sid] = setInterval(function() { pollBatchForSession(sid); }, 2000);
    pollBatchForSession(sid);
  })
  .catch(function(e) { alert('Upload failed: ' + e); });
}

var _MODEL_LABELS = {
  'claude-opus-4-7': 'Opus 4.7',
  'claude-opus-4-6': 'Opus 4.6',
  'claude-sonnet-4-6': 'Sonnet 4.6',
  'claude-haiku-4-5-20251001': 'Haiku 4.5',
};

function initBatchCards(jobId, count, filename, model) {
  hideWelcome();
  var inner = getInner();
  var modelLabel = _MODEL_LABELS[model] || model || '';
  // Label with Pause All / Resume All controls
  var label = document.createElement('div');
  label.className = 'msg assistant';
  label.id = 'batch-label-' + jobId;
  label.innerHTML = '<div class="msg-bubble-assistant" style="color:var(--sand-500);font-size:13px;">' +
    'Batch designing <strong>' + count + ' plasmid' + (count === 1 ? '' : 's') + '</strong> from <em>' + escapeHtml(filename) + '</em>' +
    (modelLabel ? ' \u00b7 <span style="color:var(--sand-400)">' + escapeHtml(modelLabel) + '</span>' : '') + '. ' +
    'Click any row to expand and see what\u2019s happening.' +
    '<div style="display:flex;gap:8px;margin-top:10px;align-items:center" id="batch-ctrl-' + jobId + '">' +
      '<button class="batch-row-pause-btn" id="batch-pause-all-' + jobId + '" title="Pause all" style="width:auto;padding:0 10px;height:26px;font-size:12px;border-radius:6px;border:1px solid var(--sand-200);gap:5px;color:var(--sand-600)" onclick="pauseAllBatch(\'' + jobId + '\')">' + PAUSE_SVG + ' Pause all</button>' +
      '<button class="batch-row-pause-btn" id="batch-resume-all-' + jobId + '" title="Resume all" style="width:auto;padding:0 10px;height:26px;font-size:12px;border-radius:6px;border:1px solid var(--sand-200);gap:5px;color:var(--sand-600);display:none" onclick="resumeAllBatch(\'' + jobId + '\')">' + RESUME_SVG + ' Resume all</button>' +
    '</div>' +
    '</div>';
  inner.appendChild(label);
  // Placeholder cards
  for (var i = 0; i < count; i++) {
    var card = document.createElement('div');
    card.className = 'msg assistant';
    card.id = 'batch-card-' + jobId + '-' + i;
    card.innerHTML = buildBatchCardHtml(jobId, i, {
      status: 'pending', description: '\u2026', exports: [], error: null, log: [], paused: false
    }, false);
    inner.appendChild(card);
  }
  scrollToBottom();
}

function pollBatchForSession(sessionId) {
  var jobId = _batchSessions[sessionId];
  if (!jobId) return;
  fetch('/api/batch/' + jobId)
  .then(function(r) { return r.json(); })
  .then(function(data) {
    if (data.error) return;
    // Only update the DOM if this session is still active
    if (currentSessionId === sessionId) {
      updateBatchCards(jobId, data.rows);
    }
    var anyRunning = data.rows && data.rows.some(function(r) {
      return r.status === 'running' || r.status === 'pending' || r.status === 'waiting';
    });
    if (data.status === 'done' && !anyRunning) {
      clearInterval(_batchPollTimers[sessionId]);
      delete _batchPollTimers[sessionId];
      if (currentSessionId !== sessionId) return;
      // Hide pause controls when batch is fully done
      var ctrlEl = document.getElementById('batch-ctrl-' + jobId);
      if (ctrlEl) ctrlEl.style.display = 'none';
      // Add Download All split button to label message
      var labelEl = document.getElementById('batch-label-' + jobId);
      if (labelEl && !labelEl.querySelector('.batch-dl-all-btn')) {
        var bubble = labelEl.querySelector('.msg-bubble-assistant');
        if (bubble) {
          var wrap = document.createElement('div');
          wrap.className = 'dl-split-wrap batch-dl-all-btn';
          wrap.style.cssText = 'margin-top:10px;';
          var jid = jobId;
          var allMenuId = 'dlmenu-all-' + jid;
          wrap.innerHTML =
            '<button class="download-btn" onclick="downloadAllBatch(\'' + jid + '\')">' + _DL_SVG + ' Download All (.zip)</button>' +
            '<button class="dl-chevron-btn" onclick="toggleDlMenu(event,\'' + allMenuId + '\')" title="More options">' + _CHEV_DOWN_SVG + '</button>' +
            '<div class="dl-menu" id="' + allMenuId + '">' +
              '<button class="dl-menu-item" onclick="downloadAllBatch(\'' + jid + '\')">' + _DL_SVG + ' Download All (.zip)</button>' +
              (_userLibraryAvailable ? '<button class="dl-menu-item" id="savall-local-' + jid + '" onclick="event.stopPropagation();saveAllBatchToLocal(\'' + jid + '\',document.getElementById(\'savall-local-' + jid + '\'))">' +
                '<svg width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M3 15v4c0 1.1.9 2 2 2h14a2 2 0 002-2v-4M17 8l-5-5-5 5M12 3v12"/></svg> Save All to Local Library</button>' : '') +
              '<button class="dl-menu-item" id="savall-con-' + jid + '" onclick="event.stopPropagation();saveAllBatchConstructs(\'' + jid + '\',document.getElementById(\'savall-con-' + jid + '\'))">' + _SAVE_SVG + ' Save All Constructs</button>' +
            '</div>';
          bubble.appendChild(document.createElement('br'));
          bubble.appendChild(wrap);
        }
      }
    }
  })
  .catch(function() {});
}

var STATUS_ICONS = {
  pending: '<svg width="18" height="18" fill="none" stroke="var(--sand-300)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/></svg>',
  running: '<svg width="18" height="18" fill="none" stroke="var(--brand-fig)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24" class="spin"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>',
  done: '<svg width="18" height="18" fill="none" stroke="var(--brand-aqua)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>',
  no_export: '<svg width="18" height="18" fill="none" stroke="var(--sand-400)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>',
  error: '<svg width="18" height="18" fill="none" stroke="var(--brand-orange)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><path d="M12 8v4m0 4h.01"/></svg>',
  paused: '<svg width="18" height="18" fill="none" stroke="var(--sand-400)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>',
};
STATUS_ICONS['waiting'] = '<svg width="18" height="18" fill="none" stroke="var(--brand-fig)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><path d="M12 8v4l3 3"/></svg>';
var STATUS_LABELS = {pending: 'Pending', running: 'Running\u2026', done: 'Done', no_export: 'No export produced', error: 'Error', paused: 'Paused', waiting: 'Waiting for approval\u2026'};
var CHEV_SVG = '<svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M9 18l6-6-6-6"/></svg>';
var PAUSE_SVG = '<svg width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>';
var RESUME_SVG = '<svg width="12" height="12" fill="currentColor" viewBox="0 0 24 24"><polygon points="5 3 19 12 5 21 5 3"/></svg>';

function renderBatchLog(log) {
  if (!log || !log.length) return '<div style="font-size:12px;color:var(--sand-400);padding:4px 0;">No activity yet.</div>';
  return log.map(function(entry) {
    if (entry.type === 'tool') {
      return '<div class="batch-log-entry batch-log-tool">' +
        '<div class="batch-log-tool-header">' +
          '<svg width="11" height="11" fill="none" stroke="var(--brand-fig)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M14.7 6.3a1 1 0 000 1.4l1.6 1.6a1 1 0 001.4 0l3-3a1 1 0 000-1.4l-1.6-1.6a1 1 0 00-1.4 0l-3 3z"/><path d="M20.26 2.26L9 13.5l-5 1 1-5L16.5 3.74"/></svg>' +
          escapeHtml(entry.name) +
        '</div>' +
        '<div class="batch-log-tool-result">' + escapeHtml(entry.result || '') + '</div>' +
      '</div>';
    } else if (entry.type === 'text') {
      return '<div class="batch-log-entry batch-log-text">' + renderContent(entry.content || '') + '</div>';
    } else if (entry.type === 'user') {
      return '<div class="batch-log-entry batch-log-user">' + escapeHtml(entry.content || '') + '</div>';
    } else if (entry.type === 'error') {
      return '<div class="batch-log-entry batch-log-error">\u26a0 ' + escapeHtml(entry.content || '') + '</div>';
    }
    return '';
  }).join('');
}

var _DL_SVG = '<svg width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>';
var _CHEV_DOWN_SVG = '<svg width="10" height="10" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><polyline points="6 9 12 15 18 9"/></svg>';
var _SAVE_SVG = '<svg width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"/></svg>';

function toggleDlMenu(event, menuId) {
  event.stopPropagation();
  var menu = document.getElementById(menuId);
  if (!menu) return;
  var isOpen = menu.classList.toggle('open');
  if (isOpen) {
    document.querySelectorAll('.dl-menu.open').forEach(function(m) {
      if (m.id !== menuId) m.classList.remove('open');
    });
    function closeMenu(e) {
      if (!menu.contains(e.target)) {
        menu.classList.remove('open');
        document.removeEventListener('click', closeMenu, true);
      }
    }
    document.addEventListener('click', closeMenu, true);
  }
}

function buildDownloadsHtml(jobId, idx, exports) {
  if (!exports || !exports.length) return '';
  var html = '<div class="batch-row-downloads">';
  exports.forEach(function(exp, eidx) {
    var fname = escapeHtml(exp.filename);
    var isGbk = /\.(gb|gbk|genbank)$/i.test(exp.filename);
    var menuId = 'dlmenu-' + jobId + '-' + idx + '-' + eidx;
    var dlCall = 'event.stopPropagation();downloadBatchFile(\'' + jobId + '\',' + idx + ',' + eidx + ',\'' + fname + '\')';
    if (isGbk) {
      // Split button: download primary + chevron dropdown
      html += '<div class="dl-split-wrap" onclick="event.stopPropagation()">' +
        '<button class="download-btn" onclick="' + dlCall + '">' + _DL_SVG + ' ' + fname + '</button>' +
        '<button class="dl-chevron-btn" onclick="toggleDlMenu(event,\'' + menuId + '\')" title="More options">' + _CHEV_DOWN_SVG + '</button>' +
        '<div class="dl-menu" id="' + menuId + '">' +
          '<button class="dl-menu-item" onclick="event.stopPropagation();downloadBatchFile(\'' + jobId + '\',' + idx + ',' + eidx + ',\'' + fname + '\')">' + _DL_SVG + ' Download to computer</button>' +
          (_userLibraryAvailable ? '<button class="dl-menu-item" id="savlocal-' + menuId + '" onclick="event.stopPropagation();saveBatchToLocal(\'' + jobId + '\',' + idx + ',' + eidx + ',document.getElementById(\'savlocal-' + menuId + '\'))">' +
            '<svg width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M3 15v4c0 1.1.9 2 2 2h14a2 2 0 002-2v-4M17 8l-5-5-5 5M12 3v12"/></svg> Save to Local Library</button>' : '') +
        '</div>' +
      '</div>';
      // Save Construct button (to DB)
      html += '<button class="save-btn" id="savcon-' + menuId + '" style="margin-left:4px" onclick="event.stopPropagation();saveBatchConstruct(\'' + jobId + '\',' + idx + ',' + eidx + ',document.getElementById(\'savcon-' + menuId + '\'))">' +
        _SAVE_SVG + ' Save Construct</button>';
    } else {
      html += '<button class="download-btn" onclick="' + dlCall + '">' + _DL_SVG + ' ' + fname + '</button>';
    }
    if (exp.has_plot) {
      html += '<button class="download-btn" style="border-color:var(--brand-fig-30);color:var(--brand-fig);background:var(--brand-fig-10);margin-left:4px" ' +
        'onclick="event.stopPropagation();openBatchPlot(\'' + jobId + '\',' + idx + ',' + eidx + ')">' +
        '<svg width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="3"/></svg>' +
        'View Map</button>';
    }
  });
  return html + '</div>';
}

function buildFollowupHtml(jobId, idx, status) {
  if (status === 'running' || status === 'pending' || status === 'waiting') return '';
  var fid = 'batch-finput-' + jobId + '-' + idx;
  return '<div class="batch-followup">' +
    '<textarea class="batch-followup-input" id="' + fid + '" rows="1" ' +
      'placeholder="Follow up with the agent about this design\u2026" ' +
      'onkeydown="batchFollowupKey(event,\'' + jobId + '\',' + idx + ')" ' +
      'oninput="this.style.height=\'auto\';this.style.height=Math.min(this.scrollHeight,100)+\'px\'"></textarea>' +
    '<button class="batch-followup-send" onclick="sendBatchFollowup(\'' + jobId + '\',' + idx + ')" title="Send">' +
      '<svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M12 19V5M5 12l7-7 7 7"/></svg>' +
    '</button>' +
  '</div>';
}

function buildBatchCardHtml(jobId, idx, row, isOpen) {
  var isPaused = row.paused && row.status === 'running';
  var icon = isPaused ? STATUS_ICONS.paused : (STATUS_ICONS[row.status] || STATUS_ICONS.pending);
  var label = isPaused ? STATUS_LABELS.paused : (STATUS_LABELS[row.status] || row.status);
  var desc = escapeHtml((row.description || '').slice(0, 120) + ((row.description || '').length > 120 ? '\u2026' : ''));
  var downloads = buildDownloadsHtml(jobId, idx, row.exports);
  var logId = 'batch-log-' + jobId + '-' + idx;
  var chevId = 'batch-chev-' + jobId + '-' + idx;
  var pauseBtn = '';
  if (row.status === 'running') {
    if (isPaused) {
      pauseBtn = '<button class="batch-row-pause-btn" title="Resume" onclick="event.stopPropagation();resumeBatchRow(\'' + jobId + '\',' + idx + ')">' + RESUME_SVG + '</button>';
    } else {
      pauseBtn = '<button class="batch-row-pause-btn" title="Pause" onclick="event.stopPropagation();pauseBatchRow(\'' + jobId + '\',' + idx + ')">' + PAUSE_SVG + '</button>';
    }
  }
  var proceedBtn = (row.status === 'waiting')
    ? '<button class="send-btn" style="width:auto;padding:0 14px;height:28px;font-size:12px;border-radius:7px;margin:10px 0 4px" ' +
        'onclick="event.stopPropagation();proceedToBatchRow(\'' + jobId + '\',' + idx + ')">' +
        'Proceed to design ' + (idx + 1) +
      '</button>'
    : '';

  return '<div class="batch-card">' +
    '<div class="batch-row-header" onclick="toggleBatchCard(\'' + jobId + '\',' + idx + ')">' +
      '<div class="batch-row-status">' + icon + '</div>' +
      '<div class="batch-row-body">' +
        '<div class="batch-row-desc">' + desc + '</div>' +
        '<div class="batch-row-meta">' + (idx + 1) + ' \xb7 ' + label + '</div>' +
        downloads +
        proceedBtn +
      '</div>' +
      pauseBtn +
      '<span id="' + chevId + '" class="batch-row-chevron' + (isOpen ? ' open' : '') + '">' + CHEV_SVG + '</span>' +
    '</div>' +
    '<div id="' + logId + '" class="batch-row-log' + (isOpen ? ' open' : '') + '">' +
      renderBatchLog(row.log) +
      buildFollowupHtml(jobId, idx, row.status) +
    '</div>' +
  '</div>';
}

function updateBatchCards(jobId, rows) {
  rows.forEach(function(row, idx) {
    var cardEl = document.getElementById('batch-card-' + jobId + '-' + idx);
    if (!cardEl) return;
    // Preserve expanded state
    var logEl = document.getElementById('batch-log-' + jobId + '-' + idx);
    var isOpen = logEl ? logEl.classList.contains('open') : false;
    cardEl.innerHTML = buildBatchCardHtml(jobId, idx, row, isOpen);
  });
}

function toggleBatchCard(jobId, idx) {
  var log = document.getElementById('batch-log-' + jobId + '-' + idx);
  var chev = document.getElementById('batch-chev-' + jobId + '-' + idx);
  if (!log) return;
  var open = log.classList.toggle('open');
  if (chev) chev.classList.toggle('open', open);
}

function proceedToBatchRow(jobId, rowIdx) {
  fetch('/api/batch/' + jobId + '/proceed/' + rowIdx, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}' })
    .then(function(r) { return r.json(); })
    .catch(function() {});
}

function batchFollowupKey(e, jobId, rowIdx) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendBatchFollowup(jobId, rowIdx); }
}

function sendBatchFollowup(jobId, rowIdx) {
  var inputEl = document.getElementById('batch-finput-' + jobId + '-' + rowIdx);
  if (!inputEl) return;
  var message = inputEl.value.trim();
  if (!message) return;
  inputEl.value = '';
  inputEl.style.height = 'auto';
  // Optimistically show the user message in the log
  var logEl = document.getElementById('batch-log-' + jobId + '-' + rowIdx);
  if (logEl) {
    var followup = logEl.querySelector('.batch-followup');
    var userDiv = document.createElement('div');
    userDiv.className = 'batch-log-entry batch-log-user';
    userDiv.textContent = message;
    if (followup) logEl.insertBefore(userDiv, followup);
    else logEl.appendChild(userDiv);
    // Disable input while running
    if (followup) {
      var btn = followup.querySelector('.batch-followup-send');
      if (inputEl) inputEl.disabled = true;
      if (btn) btn.disabled = true;
    }
  }
  fetch('/api/batch/' + jobId + '/rows/' + rowIdx + '/continue', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({message: message}),
  })
  .then(function(r) { return r.json(); })
  .then(function(data) {
    if (data.error) { alert('Error: ' + data.error); return; }
    // Restart polling for whichever session owns this job
    var ownerSid = Object.keys(_batchSessions).find(function(k) { return _batchSessions[k] === jobId; });
    if (ownerSid && !_batchPollTimers[ownerSid]) {
      _batchPollTimers[ownerSid] = setInterval(function() { pollBatchForSession(ownerSid); }, 2000);
    }
  })
  .catch(function(e) { alert('Failed to send: ' + e); });
}

function openBatchPlot(jobId, rowIdx, expIdx) {
  // Expand the card if collapsed
  var log = document.getElementById('batch-log-' + jobId + '-' + rowIdx);
  var chev = document.getElementById('batch-chev-' + jobId + '-' + rowIdx);
  if (log && !log.classList.contains('open')) {
    log.classList.add('open');
    if (chev) chev.classList.add('open');
  }
  // Don't render twice
  var plotWrapperId = 'bplotwrap-' + jobId + '-' + rowIdx + '-' + expIdx;
  if (document.getElementById(plotWrapperId)) return;
  var plotId = 'bplot-' + jobId + '-' + rowIdx + '-' + expIdx;
  // Insert plot container before the follow-up input
  var wrapper = document.createElement('div');
  wrapper.id = plotWrapperId;
  wrapper.className = 'batch-plot-wrapper';
  wrapper.style.cssText = 'padding:12px 16px;border-top:1px solid var(--sand-100);max-width:640px;';
  wrapper.innerHTML =
    '<div style="font-size:11px;font-weight:600;color:var(--sand-500);text-transform:uppercase;letter-spacing:0.05em;margin-bottom:10px;">Plasmid Map</div>' +
    '<div id="' + plotId + '" style="width:600px;height:600px;">Loading\u2026</div>';
  if (log) {
    var followup = log.querySelector('.batch-followup');
    if (followup) log.insertBefore(wrapper, followup);
    else log.appendChild(wrapper);
  }
  // Fetch the plot JSON then wait one animation frame so the browser has
  // laid out the container before Bokeh reads its dimensions.
  fetch('/api/batch/' + jobId + '/rows/' + rowIdx + '/plot/' + expIdx)
  .then(function(r) { return r.json(); })
  .then(function(data) {
    var el = document.getElementById(plotId);
    if (!el) return;
    if (data.error) { el.textContent = 'No map available.'; el.style.minHeight = ''; return; }
    el.innerHTML = '';
    // Double rAF ensures the element is fully painted before Bokeh measures it
    requestAnimationFrame(function() {
      requestAnimationFrame(function() {
        Bokeh.embed.embed_item(data, plotId);
      });
    });
  })
  .catch(function() {
    var el = document.getElementById(plotId);
    if (el) { el.textContent = 'Failed to load map.'; el.style.minHeight = ''; }
  });
}

function downloadAllBatch(jobId) {
  var a = document.createElement('a');
  a.href = '/api/batch/' + jobId + '/download-all';
  a.download = 'batch_designs.zip';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

function pauseBatchRow(jobId, rowIdx) {
  fetch('/api/batch/' + jobId + '/rows/' + rowIdx + '/pause', {method: 'POST'})
  .then(function(r) { return r.json(); })
  .catch(function() {});
}

function resumeBatchRow(jobId, rowIdx) {
  fetch('/api/batch/' + jobId + '/rows/' + rowIdx + '/resume', {method: 'POST'})
  .then(function(r) { return r.json(); })
  .catch(function() {});
}

function pauseAllBatch(jobId) {
  fetch('/api/batch/' + jobId + '/pause-all', {method: 'POST'})
  .then(function(r) { return r.json(); })
  .then(function() {
    // Swap button visibility
    var p = document.getElementById('batch-pause-all-' + jobId);
    var r = document.getElementById('batch-resume-all-' + jobId);
    if (p) p.style.display = 'none';
    if (r) r.style.display = '';
  })
  .catch(function() {});
}

function resumeAllBatch(jobId) {
  fetch('/api/batch/' + jobId + '/resume-all', {method: 'POST'})
  .then(function(r) { return r.json(); })
  .then(function() {
    var p = document.getElementById('batch-pause-all-' + jobId);
    var r = document.getElementById('batch-resume-all-' + jobId);
    if (p) p.style.display = '';
    if (r) r.style.display = 'none';
  })
  .catch(function() {});
}

// ── Saved Constructs ─────────────────────────────────────────────────────────

const _SVG_CHECK = '<svg width="13" height="13" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><polyline points="20 6 9 17 4 12"/></svg>';
const _SVG_DL = '<svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3"/></svg>';
const _SVG_FOLDER = '<svg width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>';

function _triggerDownload(content, filename) {
  const blob = new Blob([content], {type: 'application/octet-stream'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function addExportButtons(container, toolInput, genbankContent, filename) {
  const outer = document.createElement('div');
  outer.className = 'msg assistant';

  const _SVG_CHOOSE = '<svg width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/></svg>';
  const localLibItem = _userLibraryAvailable
    ? '<button class="dl-menu-item" data-role="dl-library">' + _SVG_FOLDER + ' Save to Local Library</button>'
    : '';

  outer.innerHTML =
    '<div class="msg-bubble-assistant" style="margin-top:8px">' +
      '<div style="display:flex;flex-wrap:wrap;gap:8px;align-items:center">' +
        '<div class="dl-split-wrap">' +
          '<button class="download-btn" data-role="dl" data-tooltip="Download this file to your computer">' + _SVG_DL + ' Download</button>' +
          '<button class="dl-chevron-btn" data-role="dl-chevron" aria-label="More save options">' +
            '<svg width="10" height="10" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><polyline points="6 9 12 15 18 9"/></svg>' +
          '</button>' +
          '<div class="dl-menu">' +
            '<button class="dl-menu-item" data-role="dl-computer">' + _SVG_DL + ' Download to computer</button>' +
            '<button class="dl-menu-item" data-role="dl-choosepath">' + _SVG_CHOOSE + ' Save to…</button>' +
            localLibItem +
          '</div>' +
        '</div>' +
        '<button class="save-btn" data-role="save" data-tooltip="Save to local database">' +
          '<svg width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"/></svg>' +
          ' Save Construct' +
        '</button>' +
        '<button class="viewer-btn" data-role="viewer" data-tooltip="Open an inline Ori-style sequence viewer">' +
          '<svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7z"/><circle cx="12" cy="12" r="3"/></svg>' +
          '<span class="viewer-btn-label">Open in Viewer</span>' +
        '</button>' +
      '</div>' +
    '</div>';
  container.appendChild(outer);

  // Download main button — direct download to computer
  outer.querySelector('[data-role="dl"]').addEventListener('click', function() {
    _triggerDownload(genbankContent, filename);
  });

  // Chevron toggles dropdown
  const chevronBtn = outer.querySelector('[data-role="dl-chevron"]');
  const menu = outer.querySelector('.dl-menu');
  chevronBtn.addEventListener('click', function(e) {
    e.stopPropagation();
    var isOpen = menu.classList.toggle('open');
    if (isOpen) {
      document.querySelectorAll('.dl-menu.open').forEach(function(m) {
        if (m !== menu) m.classList.remove('open');
      });
      function closeMenu(e) {
        if (!menu.contains(e.target)) {
          menu.classList.remove('open');
          document.removeEventListener('click', closeMenu, true);
        }
      }
      document.addEventListener('click', closeMenu, true);
    }
  });

  outer.querySelector('[data-role="dl-computer"]').addEventListener('click', function() {
    menu.classList.remove('open');
    _triggerDownload(genbankContent, filename);
  });

  outer.querySelector('[data-role="dl-choosepath"]').addEventListener('click', async function() {
    menu.classList.remove('open');
    if (window.showSaveFilePicker) {
      try {
        const handle = await window.showSaveFilePicker({
          suggestedName: filename,
          types: [{ description: 'GenBank file', accept: {'text/plain': ['.gb', '.gbk']} }],
        });
        const writable = await handle.createWritable();
        await writable.write(genbankContent);
        await writable.close();
      } catch(e) {
        if (e.name !== 'AbortError') _triggerDownload(genbankContent, filename);
      }
    }
  });

  const libItem = outer.querySelector('[data-role="dl-library"]');
  if (libItem) {
    async function _saveToLib(name, overwrite) {
      const dlBtn = outer.querySelector('[data-role="dl"]');
      const origHtml = dlBtn.innerHTML;
      dlBtn.disabled = true; dlBtn.textContent = 'Saving…';
      try {
        const r = await fetch('/api/local-library/save', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({name, content: genbankContent, overwrite}),
        });
        const data = await r.json();
        if (data.saved_to) {
          dlBtn.innerHTML = _SVG_CHECK + ' Saved: ' + escapeHtml(data.saved_to.split('/').pop());
          dlBtn.style.opacity = '0.75';
          // Remove any rename row
          const rr = outer.querySelector('.dl-rename-row');
          if (rr) rr.remove();
        } else if (data.exists) {
          dlBtn.innerHTML = origHtml; dlBtn.disabled = false;
          // Show inline rename form
          let rr = outer.querySelector('.dl-rename-row');
          if (!rr) {
            rr = document.createElement('div');
            rr.className = 'dl-rename-row';
            outer.querySelector('.msg-bubble-assistant > div').after(rr);
          }
          rr.innerHTML =
            '<span style="font-size:11px;color:var(--sand-500)">A file with that name already exists. Save as:</span>' +
            '<input type="text" value="' + escapeHtml(data.suggested_name) + '">.gb' +
            '<button class="dl-rename-confirm">Save</button>' +
            '<button class="dl-rename-cancel">Cancel</button>';
          const inp = rr.querySelector('input');
          inp.focus(); inp.select();
          rr.querySelector('.dl-rename-confirm').addEventListener('click', function() {
            _saveToLib(inp.value.trim() || data.suggested_name, true);
          });
          inp.addEventListener('keydown', function(e) {
            if (e.key === 'Enter') _saveToLib(inp.value.trim() || data.suggested_name, true);
            if (e.key === 'Escape') rr.remove();
          });
          rr.querySelector('.dl-rename-cancel').addEventListener('click', function() { rr.remove(); });
        } else { dlBtn.innerHTML = origHtml; dlBtn.disabled = false; }
      } catch(e) { dlBtn.innerHTML = origHtml; dlBtn.disabled = false; }
    }
    libItem.addEventListener('click', function() {
      menu.classList.remove('open');
      _saveToLib(toolInput.construct_name || 'construct', false);
    });
  }

  // Save Construct → DB
  outer.querySelector('[data-role="save"]').addEventListener('click', async function() {
    const btn = this;
    btn.disabled = true; btn.textContent = 'Saving…';
    const body = {
      construct_name: toolInput.construct_name || 'construct',
      genbank_content: genbankContent,
      total_size_bp: null,
      session_id: currentSessionId,
      backbone_name: toolInput.backbone_name || '',
      insert_name: toolInput.insert_name || '',
      sequence_cache_key: toolInput.sequence_cache_key || '',
    };
    try {
      const r = await fetch('/api/db/constructs', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
      const data = await r.json();
      if (data.id) { btn.innerHTML = _SVG_CHECK + ' Saved'; btn.style.opacity = '0.75'; refreshLibraryData(); }
      else { btn.textContent = 'Save failed'; btn.disabled = false; }
    } catch(e) { btn.textContent = 'Save failed'; btn.disabled = false; }
  });

  // Open in Viewer → inline Ori-style sequence panel (toggles open/closed)
  const viewerBtn = outer.querySelector('[data-role="viewer"]');
  if (viewerBtn) {
    viewerBtn.addEventListener('click', function() {
      try {
        const nm = (toolInput.construct_name ||
          (filename || '').replace(/\.(gb|gbk|fasta|fa|txt)$/i, '') || 'construct');
        window.OriViewer.open(genbankContent, nm, viewerBtn);
      } catch(err) {
        console.error('OriViewer error', err);
        const lbl = viewerBtn.querySelector('.viewer-btn-label');
        if (lbl) lbl.textContent = 'Viewer error';
      }
    });
  }
}

// ── Library panel toggle ─────────────────────────────────────────────────────

let _libraryPanelOpen = false;

function toggleLibraryPanel() {
  const panel = document.getElementById('library-panel');
  _libraryPanelOpen = !_libraryPanelOpen;
  panel.style.display = _libraryPanelOpen ? 'flex' : 'none';
  const btn = document.getElementById('lib-panel-btn');
  if (btn) btn.classList.toggle('active', _libraryPanelOpen);
  if (_libraryPanelOpen) {
    _checkUserLibrary();
    // Defer init until after the browser has painted the panel so Tabulator
    // measures the real container width, not 0.
    requestAnimationFrame(function() {
      if (!_constructsTable) {
        _initTabulator();
      } else {
        _constructsTable.setData('/api/db/constructs');
      }
      if (!_cy) _initCytoscape();
    });
  }
}

function showLibraryTab(tab) {
  document.getElementById('lib-table-pane').style.display = tab === 'table' ? '' : 'none';
  document.getElementById('lib-graph-pane').style.display = tab === 'graph' ? 'flex' : 'none';
  document.getElementById('lib-tab-table').classList.toggle('active', tab === 'table');
  document.getElementById('lib-tab-graph').classList.toggle('active', tab === 'graph');
  if (tab === 'graph') {
    _loadGraphData();
    if (_cy) { setTimeout(function() { _cy.resize(); _cy.fit(); }, 50); }
  }
}

function refreshLibraryData() {
  if (_constructsTable) _constructsTable.setData('/api/db/constructs');
  if (_cy && document.getElementById('lib-graph-pane').style.display !== 'none') _loadGraphData();
}

// ── Tabulator ────────────────────────────────────────────────────────────────

let _constructsTable = null;

function _initTabulator() {
  _constructsTable = new Tabulator('#constructs-table', {
    ajaxURL: '/api/db/constructs',
    layout: 'fitColumns',
    height: 'calc(90vh - 58px)',
    placeholder: 'No constructs saved yet. Export a construct as GenBank and click "Save Construct".',
    columns: [
      {formatter: 'rowSelection', titleFormatter: 'rowSelection', width: 42,
       hozAlign: 'center', headerSort: false, frozen: true,
       cellClick: function(e) { e.stopPropagation(); }},
      {title: 'ID', field: 'accession', frozen: true, width: 100,
       sorter: 'string', hozAlign: 'center',
       formatter: function(cell) {
         return '<code style="font-size:11px;color:var(--brand-fig);font-weight:600">' + escapeHtml(cell.getValue() || '') + '</code>';
       }},
      {title: 'Construct Name', field: 'construct_name', frozen: true, width: 190,
       sorter: 'string', tooltip: true},
      {title: 'Source', field: 'origin', width: 110, headerSort: false,
       formatter: function(cell) {
         const v = cell.getValue() || 'designer';
         const cfg = {
           'designer':     {bg:'#3B82F6', label:'Designed'},
           'user_library': {bg:'#10B981', label:'Your Library'},
           'annotation':   {bg:'#8B5CF6', label:'Annotation'},
         };
         const c = cfg[v] || cfg['designer'];
         return '<span style="background:' + c.bg + ';color:#fff;padding:2px 8px;border-radius:12px;font-size:11px;font-family:Inter,sans-serif;white-space:nowrap">' + c.label + '</span>';
       }},
      {title: 'Type', field: 'part_type', width: 80, headerSort: false,
       formatter: function(cell) {
         const v = cell.getValue();
         if (!v) return '';
         return '<span style="font-size:11px;color:var(--text-secondary);font-family:Inter,sans-serif">' + escapeHtml(v) + '</span>';
       }},
      {title: 'User Label', field: 'user_name', editor: 'input', width: 130,
       cellEdited: _onCellEdited, placeholder: 'Add label…'},
      {title: 'bp', field: 'total_size_bp', sorter: 'number', width: 70, hozAlign: 'right'},
      {title: 'Created', field: 'created_at', sorter: 'datetime', width: 140,
       formatter: function(cell) {
         const v = cell.getValue();
         return v ? v.slice(0, 16).replace('T', ' ') : '';
       }},
      {title: '&#10003;', field: 'sequence_verified', formatter: 'tickCross',
       editor: true, width: 42, hozAlign: 'center', cellEdited: _onCellEdited,
       headerTooltip: 'Sequence verified'},
      {title: 'File', width: 52, formatter: function(cell) {
         const id = cell.getRow().getData().id;
         return '<a class="download-btn" style="font-size:11px;padding:3px 7px" href="/api/db/constructs/' + id + '/genbank">GBK</a>';
       }, hozAlign: 'center', headerSort: false, cellClick: function(e) { e.stopPropagation(); }},
      {title: 'Notes', field: 'notes', editor: 'textarea', widthGrow: 1,
       cellEdited: _onCellEdited, formatter: 'plaintext', tooltip: true},
    ],
    rowFormatter: function(row) {
      row.getElement().addEventListener('click', function(e) {
        if (e.target.tagName === 'A' || e.target.tagName === 'INPUT') return;
        _toggleRowDetail(row);
      });
    },
  });
  _constructsTable.on('rowSelectionChanged', function(data, rows) {
    const n = rows.length;
    const btn = document.getElementById('lib-remove-btn');
    const cnt = document.getElementById('lib-remove-count');
    if (btn) btn.style.display = n > 0 ? '' : 'none';
    if (cnt) cnt.textContent = n;
  });
}

async function _onCellEdited(cell) {
  const id = cell.getRow().getData().id;
  const field = cell.getField();
  const value = cell.getValue();
  await fetch('/api/db/constructs/' + id, {
    method: 'PATCH',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({[field]: value}),
  });
}

function _toggleRowDetail(row) {
  const el = row.getElement();
  const existing = el.nextElementSibling;
  if (existing && existing.classList.contains('row-detail-wrap')) {
    existing.remove();
    return;
  }
  const data = row.getData();
  const wrap = document.createElement('div');
  wrap.className = 'row-detail-wrap';

  // Metadata grid (for imported library items)
  let partsHtml = '';
  const meta = data.metadata || {};
  const metaFields = [
    ['Description', meta.description],
    ['Category', meta.category],
    ['Assembly enzyme', meta.assembly_enzyme],
    ['Next step enzyme', meta.next_step_enzyme],
    ['Overhang L', meta.overhang_l],
    ['Overhang R', meta.overhang_r],
    ['Overhang pair 1', (meta.overhang_left && meta.overhang_right) ? meta.overhang_left + ' / ' + meta.overhang_right : null],
    ['Overhang pair 2', (meta.overhang_left_2 && meta.overhang_right_2) ? meta.overhang_left_2 + ' / ' + meta.overhang_right_2 : null],
    ['Insert size', meta.insert_size_bp ? meta.insert_size_bp + ' bp' : null],
    ['Bacterial resistance', meta.bacterial_resistance],
    ['Mammalian selection', meta.mammalian_selection],
    ['E. coli strain', meta.ecoli_strain],
  ].filter(function(r) { return r[1]; });
  if (metaFields.length) {
    partsHtml += '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:6px 16px;padding:10px 0 6px;border-bottom:1px solid var(--sand-200);margin-bottom:8px">';
    metaFields.forEach(function(r) {
      partsHtml += '<div style="font-size:12px;font-family:Inter,sans-serif">' +
        '<span style="color:var(--text-secondary)">' + escapeHtml(r[0]) + ':</span> ' +
        '<strong>' + escapeHtml(String(r[1])) + '</strong></div>';
    });
    partsHtml += '</div>';
  }

  // Parts table
  partsHtml += '<h4>Parts &amp; Provenance</h4>';
  if (data.parts && data.parts.length) {
    partsHtml += '<table class="parts-sub-table"><thead><tr>' +
      '<th>Part</th><th>Type</th><th>Region</th><th>Source</th><th>DOI / Accession</th>' +
      '</tr></thead><tbody>';
    data.parts.forEach(function(p) {
      const srcLink = p.source_url
        ? '<a href="' + escapeHtml(p.source_url) + '" target="_blank" rel="noopener">' + escapeHtml(p.source_system || p.source_url) + '</a>'
        : escapeHtml(p.source_system || '—');
      let ref = '—';
      if (p.source_doi) ref = '<a href="https://doi.org/' + escapeHtml(p.source_doi) + '" target="_blank" rel="noopener">' + escapeHtml(p.source_doi) + '</a>';
      else if (p.genbank_accession) ref = '<a href="https://www.ncbi.nlm.nih.gov/nuccore/' + escapeHtml(p.genbank_accession) + '" target="_blank" rel="noopener">' + escapeHtml(p.genbank_accession) + '</a>';
      else if (p.addgene_id) ref = '<a href="https://www.addgene.org/' + escapeHtml(p.addgene_id) + '/" target="_blank" rel="noopener">Addgene #' + escapeHtml(p.addgene_id) + '</a>';
      partsHtml += '<tr><td>' + escapeHtml(p.part_name) + '</td><td>' + escapeHtml(p.part_type) +
        '</td><td>' + escapeHtml(p.part_region || '—') + '</td><td>' + srcLink + '</td><td>' + ref + '</td></tr>';
    });
    partsHtml += '</tbody></table>';
  } else {
    partsHtml += '<p style="font-size:11px;color:var(--text-secondary)">No part details captured.</p>';
  }

  // Upload verified sequence
  const uploadId = 'upload-seq-' + data.id;
  partsHtml += '<label class="upload-verified-btn" for="' + uploadId + '">' +
    '&#8679; Upload verified sequence</label>' +
    '<input type="file" id="' + uploadId + '" accept=".gb,.fasta,.fa,.txt" style="display:none">';

  // Save to local library (only for designer constructs when user library is configured)
  if ((data.origin || 'designer') === 'designer' && _userLibraryAvailable) {
    partsHtml += '<button id="save-to-lib-' + data.id + '" class="upload-verified-btn" style="cursor:pointer;border:none;background:var(--sand-200)" onclick="saveToLocalLibrary(' + data.id + ', this)">&#8681; Save to Local Library</button>';
    if (data.local_path) {
      partsHtml += '<span style="font-size:11px;color:var(--text-secondary);margin-left:8px">Saved: ' + escapeHtml(data.local_path) + '</span>';
    }
  }

  wrap.innerHTML = partsHtml;
  el.after(wrap);

  // Wire up upload
  const fileInput = wrap.querySelector('#' + uploadId);
  fileInput.addEventListener('change', async function() {
    if (!fileInput.files.length) return;
    const text = await fileInput.files[0].text();
    await fetch('/api/db/constructs/' + data.id, {
      method: 'PATCH',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({verified_sequence: text, sequence_verified: true}),
    });
    // Update the cell in the table
    if (_constructsTable) {
      const r = _constructsTable.getRow(data.id);
      if (r) r.update({sequence_verified: true});
    }
    const lbl = wrap.querySelector('label.upload-verified-btn');
    if (lbl) lbl.textContent = '✓ Verified sequence uploaded';
  });
}

async function saveToLocalLibrary(constructId, btn, name, overwrite) {
  const origText = btn.dataset.origText || btn.textContent;
  btn.dataset.origText = origText;
  btn.disabled = true;
  btn.textContent = 'Saving…';
  try {
    const body = {};
    if (name) body.name = name;
    if (overwrite) body.overwrite = true;
    const r = await fetch('/api/db/constructs/' + constructId + '/save-to-library', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    });
    const data = await r.json();
    if (data.saved_to) {
      btn.textContent = '✓ Saved';
      // Remove any rename row that may be present
      const existing = btn.parentElement && btn.parentElement.querySelector('.lib-rename-row');
      if (existing) existing.remove();
      const span = document.createElement('span');
      span.style.cssText = 'font-size:11px;color:var(--text-secondary);margin-left:8px';
      span.textContent = data.saved_to;
      btn.after(span);
    } else if (data.exists) {
      btn.textContent = origText;
      btn.disabled = false;
      // Remove any stale rename row first
      const stale = btn.parentElement && btn.parentElement.querySelector('.lib-rename-row');
      if (stale) stale.remove();
      // Inline rename form
      const row = document.createElement('div');
      row.className = 'dl-rename-row lib-rename-row';
      row.style.marginTop = '6px';
      row.innerHTML =
        '<span style="font-size:11px;color:var(--sand-500)">File exists. Save as:</span>' +
        '<input type="text" value="' + escapeHtml(data.suggested_name) + '">.gb' +
        '<button class="dl-rename-confirm">Save</button>' +
        '<button class="dl-rename-cancel">Cancel</button>';
      btn.after(row);
      const inp = row.querySelector('input');
      inp.focus(); inp.select();
      row.querySelector('.dl-rename-confirm').addEventListener('click', function() {
        row.remove();
        saveToLocalLibrary(constructId, btn, inp.value.trim() || data.suggested_name, true);
      });
      inp.addEventListener('keydown', function(e) {
        if (e.key === 'Enter') {
          row.remove();
          saveToLocalLibrary(constructId, btn, inp.value.trim() || data.suggested_name, true);
        }
        if (e.key === 'Escape') row.remove();
      });
      row.querySelector('.dl-rename-cancel').addEventListener('click', function() { row.remove(); });
    } else {
      btn.textContent = 'Save failed: ' + (data.error || 'unknown error');
      btn.disabled = false;
    }
  } catch(e) {
    btn.textContent = 'Save failed';
    btn.disabled = false;
  }
}

async function _removeSelected() {
  if (!_constructsTable) return;
  const rows = _constructsTable.getSelectedRows();
  if (!rows.length) return;
  const n = rows.length;
  if (!confirm('Remove ' + n + ' item' + (n > 1 ? 's' : '') + ' from the library?\nSource files on disk are not deleted.')) return;
  for (const row of rows) {
    const id = row.getData().id;
    try {
      await fetch('/api/db/constructs/' + id, {method: 'DELETE'});
      row.delete();
    } catch(e) { console.warn('Failed to delete', id, e); }
  }
}

// ── Import modal ─────────────────────────────────────────────────────────────

let _importItems = [];

async function importUserLibrary() {
  const modal = document.getElementById('import-modal');
  if (!modal) return;
  modal.style.display = 'flex';
  const tbody = document.getElementById('import-preview-body');
  tbody.innerHTML = '<tr><td colspan="7" style="padding:20px;text-align:center;color:var(--text-secondary);font-family:Inter,sans-serif;font-size:12px">Loading…</td></tr>';
  _updateImportCount();
  try {
    const r = await fetch('/api/db/user-library-preview');
    if (!r.ok) { tbody.innerHTML = '<tr><td colspan="7" style="padding:20px;text-align:center;color:#E86235">Failed to load library</td></tr>'; return; }
    _importItems = await r.json();
    _renderImportTable();
  } catch(e) {
    tbody.innerHTML = '<tr><td colspan="7" style="padding:20px;text-align:center;color:#E86235">' + escapeHtml(String(e)) + '</td></tr>';
  }
}

function _renderImportTable() {
  const tbody = document.getElementById('import-preview-body');
  if (!_importItems.length) {
    tbody.innerHTML = '<tr><td colspan="7" style="padding:20px;text-align:center;color:var(--text-secondary);font-family:Inter,sans-serif;font-size:12px">No items found in library directory.</td></tr>';
    return;
  }
  const typeBadge = {'backbone':'#D97757','insert':'#24B283','annotation':'#8B5CF6'};
  tbody.innerHTML = _importItems.map(function(item, i) {
    const already = item.already_imported;
    const badge = typeBadge[item.part_type] || '#888';
    const sizeStr = item.size_bp ? item.size_bp.toLocaleString() + ' bp' : '—';
    const catEn = [item.category, item.assembly_enzyme].filter(Boolean).join(' / ') || '—';
    const res = item.bacterial_resistance || '—';
    return '<tr style="border-bottom:1px solid var(--sand-200);' + (already ? 'opacity:0.55' : '') + '">' +
      '<td style="padding:7px 10px;text-align:center">' +
        '<input type="checkbox" class="import-item-check" data-idx="' + i + '"' +
        (already ? ' disabled' : '') + ' onchange="_updateImportCount()"></td>' +
      '<td style="padding:7px 10px;font-family:Inter,sans-serif;font-size:12px">' + escapeHtml(item.name) + '</td>' +
      '<td style="padding:7px 10px"><span style="background:' + badge + ';color:#fff;padding:1px 7px;border-radius:10px;font-size:10px;font-family:Inter,sans-serif">' + escapeHtml(item.part_type) + '</span></td>' +
      '<td style="padding:7px 10px;font-size:12px;font-family:Inter,sans-serif;color:var(--text-secondary)">' + escapeHtml(sizeStr) + '</td>' +
      '<td style="padding:7px 10px;font-size:12px;font-family:Inter,sans-serif">' + escapeHtml(catEn) + '</td>' +
      '<td style="padding:7px 10px;font-size:12px;font-family:Inter,sans-serif">' + escapeHtml(res) + '</td>' +
      '<td style="padding:7px 10px;font-size:11px;color:var(--text-secondary);font-family:Inter,sans-serif">' +
        (already ? '&#10003; already imported' : '') + '</td>' +
      '</tr>';
  }).join('');
  _updateImportCount();
}

function _updateImportCount() {
  const checks = document.querySelectorAll('.import-item-check:checked');
  const lbl = document.getElementById('import-selected-count');
  if (lbl) lbl.textContent = checks.length + ' selected';
}

function onCheckAllChange(master) {
  document.querySelectorAll('.import-item-check:not(:disabled)').forEach(function(cb) {
    cb.checked = master.checked;
  });
  _updateImportCount();
}

function toggleImportSelectAll() {
  const checks = document.querySelectorAll('.import-item-check:not(:disabled)');
  const allChecked = Array.from(checks).every(function(c) { return c.checked; });
  checks.forEach(function(c) { c.checked = !allChecked; });
  const master = document.getElementById('import-check-all');
  if (master) master.checked = !allChecked;
  _updateImportCount();
}

function closeImportModal() {
  const modal = document.getElementById('import-modal');
  if (modal) modal.style.display = 'none';
}

async function confirmImport() {
  const checks = document.querySelectorAll('.import-item-check:checked');
  if (!checks.length) { alert('No items selected.'); return; }
  const selected = Array.from(checks).map(function(cb) {
    return _importItems[parseInt(cb.dataset.idx)].local_path;
  }).filter(Boolean);
  const btn = document.getElementById('import-confirm-btn');
  if (btn) { btn.disabled = true; btn.textContent = 'Importing…'; }
  try {
    const r = await fetch('/api/db/import-user-library', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({local_paths: selected}),
    });
    const data = await r.json();
    if (data.error) {
      alert('Import failed: ' + data.error);
    } else {
      closeImportModal();
      refreshLibraryData();
    }
  } catch(e) {
    alert('Import failed: ' + e);
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = 'Import Selected'; }
  }
}

// ── Cytoscape ────────────────────────────────────────────────────────────────

let _cy = null;

function _buildTooltipHtml(node) {
  const d = node.data();
  const type = d.nodeType;
  let rows = '';
  const row = (label, val) => val
    ? '<tr><td style="color:#87867F;padding-right:10px;white-space:nowrap">' + label + '</td>' +
      '<td style="font-weight:500">' + val + '</td></tr>'
    : '';

  if (type === 'construct') {
    rows += row('Accession', '<code style="color:#D97757;font-weight:600">' + escapeHtml(d.accession) + '</code>');
    rows += row('Name', escapeHtml(d.label));
    if (d.user_name) rows += row('Label', escapeHtml(d.user_name));
    if (d.size_bp) rows += row('Size', d.size_bp.toLocaleString() + ' bp');
    if (d.backbone_name) rows += row('Backbone', escapeHtml(d.backbone_name));
    if (d.insert_names && d.insert_names.length)
      rows += row('Inserts', escapeHtml(d.insert_names.join(', ')));
    if (d.created_at) rows += row('Created', escapeHtml(d.created_at));
    const originLabels = {designer:'Designed', user_library:'Your Library', annotation:'Annotation'};
    if (d.origin && d.origin !== 'designer') rows += row('Source', escapeHtml(originLabels[d.origin] || d.origin));
    if (d.sequence_verified)
      rows += row('', '<span style="color:#24B283;font-weight:600">&#10003; Sequence verified</span>');
  } else if (type === 'backbone') {
    rows += row('Backbone', '<strong>' + escapeHtml(d.label) + '</strong>');
    if (d.source_system) rows += row('Source', escapeHtml(d.source_system));
    if (d.addgene_id) rows += row('Addgene', '#' + escapeHtml(d.addgene_id));
    if (d.source_doi) rows += row('DOI', '<a href="https://doi.org/' + escapeHtml(d.source_doi) + '" target="_blank" style="color:#D97757">' + escapeHtml(d.source_doi) + '</a>');
    if (d.usage_count > 1) rows += row('Used in', d.usage_count + ' constructs');
  } else if (type === 'insert') {
    rows += row('Insert', '<strong>' + escapeHtml(d.label) + '</strong>');
    if (d.source_system) rows += row('Source', escapeHtml(d.source_system));
    if (d.genbank_accession) rows += row('Accession', escapeHtml(d.genbank_accession));
    if (d.usage_count > 1) rows += row('Used in', d.usage_count + ' constructs');
  }
  return '<table style="border-collapse:collapse;font-size:12px">' + rows + '</table>';
}

function _showCyTooltip(evt) {
  const tip = document.getElementById('cy-tooltip');
  if (!tip) return;
  tip.innerHTML = _buildTooltipHtml(evt.target);
  tip.style.display = 'block';
  _positionCyTooltip(evt);
}

function _positionCyTooltip(evt) {
  const tip = document.getElementById('cy-tooltip');
  if (!tip || tip.style.display === 'none') return;
  const container = document.getElementById('constructs-graph');
  if (!container) return;
  const rect = container.getBoundingClientRect();
  const pos = evt.renderedPosition || evt.target.renderedPosition();
  let x = pos.x + 14;
  let y = pos.y + 14;
  // Clamp to container bounds
  const tw = tip.offsetWidth || 220;
  const th = tip.offsetHeight || 100;
  if (x + tw > rect.width - 10) x = pos.x - tw - 10;
  if (y + th > rect.height - 10) y = pos.y - th - 10;
  tip.style.left = Math.max(4, x) + 'px';
  tip.style.top = Math.max(4, y) + 'px';
}

function _hideCyTooltip() {
  const tip = document.getElementById('cy-tooltip');
  if (tip) tip.style.display = 'none';
}

function _initCytoscape() {
  const container = document.getElementById('constructs-graph');
  if (!container || typeof cytoscape === 'undefined') return;
  _cy = cytoscape({
    container: container,
    elements: [],
    style: [
      {selector: 'node[nodeType="construct"]', style: {
        'background-color': '#3B82F6',
        'label': 'data(label)',
        'font-size': '10px',
        'color': '#3D3D3A',
        'text-wrap': 'wrap',
        'text-max-width': '70px',
        'width': '65px', 'height': '65px',
        'border-width': '2px', 'border-color': '#E8E6DC',
        'font-family': 'Inter, sans-serif',
        'cursor': 'pointer',
      }},
      {selector: 'node[nodeType="construct"][origin="user_library"]', style: {
        'background-color': '#10B981',
      }},
      {selector: 'node[nodeType="construct"][origin="annotation"]', style: {
        'background-color': '#8B5CF6',
      }},
      {selector: 'node[nodeType="backbone"]', style: {
        'background-color': '#D97757',
        'shape': 'diamond',
        'label': 'data(label)',
        'font-size': '10px',
        'color': '#3D3D3A',
        'text-wrap': 'wrap',
        'text-max-width': '70px',
        'width': '64px', 'height': '64px',
        'font-family': 'Inter, sans-serif',
        'cursor': 'pointer',
      }},
      {selector: 'node[nodeType="insert"]', style: {
        'background-color': '#5C5B56',
        'shape': 'round-rectangle',
        'label': 'data(label)',
        'font-size': '10px',
        'color': '#3D3D3A',
        'text-wrap': 'wrap',
        'text-max-width': '65px',
        'width': '65px', 'height': '40px',
        'font-family': 'Inter, sans-serif',
        'cursor': 'pointer',
      }},
      {selector: 'node:selected', style: {
        'border-color': '#E86235', 'border-width': '3px',
      }},
      {selector: 'node:active', style: {
        'overlay-opacity': 0.1,
      }},
      {selector: 'edge', style: {
        'width': 1.5,
        'line-color': '#ADAAA0',
        'curve-style': 'bezier',
        'opacity': 0.7,
      }},
      {selector: 'edge:selected', style: {
        'line-color': '#D97757', 'opacity': 1, 'width': 2.5,
      }},
    ],
    layout: {name: 'klay', klay: {spacing: 55, direction: 'RIGHT'}},
  });

  _cy.on('tap', 'node[nodeType="construct"]', function(evt) {
    _hideCyTooltip();
    const rawId = evt.target.id().replace('c_', '');
    const numId = parseInt(rawId, 10);
    if (_constructsTable && !isNaN(numId)) {
      showLibraryTab('table');
      const r = _constructsTable.getRow(numId);
      if (r) { r.select(); r.scrollTo(); }
    }
  });

  _cy.on('mouseover', 'node', function(evt) { _showCyTooltip(evt); });
  _cy.on('mousemove', 'node', function(evt) { _positionCyTooltip(evt); });
  _cy.on('mouseout', 'node', function() { _hideCyTooltip(); });
  _cy.on('tap', 'node[nodeType="backbone"], node[nodeType="insert"]', function() {
    _hideCyTooltip();
  });
  _cy.on('tap', function(evt) {
    if (evt.target === _cy) _hideCyTooltip();
  });
}

async function _loadGraphData() {
  if (!_cy) return;
  try {
    const r = await fetch('/api/db/graph');
    const data = await r.json();
    _cy.elements().remove();
    _cy.add(data.nodes || []);
    _cy.add(data.edges || []);
    if (data.nodes && data.nodes.length) {
      _cy.layout({name: 'klay', klay: {spacing: 50, direction: 'RIGHT'}}).run();
    }
  } catch(e) {
    console.warn('Graph load failed', e);
  }
}

function downloadBatchFile(jobId, rowIdx, expIdx, filename) {
  fetch('/api/batch/' + jobId + '/download/' + rowIdx + '/' + expIdx)
  .then(function(r) { return r.blob(); })
  .then(function(blob) {
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url; a.download = filename;
    document.body.appendChild(a); a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  })
  .catch(function(e) { alert('Download failed: ' + e); });
}

function saveBatchConstruct(jobId, rowIdx, expIdx, btn) {
  if (!btn || btn.disabled) return;
  btn.disabled = true;
  btn.innerHTML = _SAVE_SVG + ' Saving…';
  fetch('/api/batch/' + jobId + '/rows/' + rowIdx + '/save-construct/' + expIdx, {method: 'POST'})
  .then(function(r) { return r.json(); })
  .then(function(data) {
    if (data.id) {
      btn.innerHTML = '<svg width="12" height="12" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><polyline points="20 6 9 17 4 12"/></svg> Saved';
      btn.style.opacity = '0.7';
    } else {
      btn.innerHTML = _SAVE_SVG + ' Save failed';
      btn.disabled = false;
    }
  })
  .catch(function() {
    btn.innerHTML = _SAVE_SVG + ' Save failed';
    btn.disabled = false;
  });
}

function saveBatchToLocal(jobId, rowIdx, expIdx, btn) {
  if (!btn || btn.disabled) return;
  btn.disabled = true;
  var origHtml = btn.innerHTML;
  btn.textContent = 'Saving…';
  fetch('/api/batch/' + jobId + '/rows/' + rowIdx + '/save-local/' + expIdx, {method: 'POST'})
  .then(function(r) { return r.json(); })
  .then(function(data) {
    if (data.saved_to) {
      btn.textContent = '✓ Saved to library';
    } else {
      btn.innerHTML = origHtml;
      btn.disabled = false;
      alert('Save failed: ' + (data.error || 'unknown error'));
    }
  })
  .catch(function() {
    btn.innerHTML = origHtml;
    btn.disabled = false;
    alert('Save failed');
  });
}

function saveAllBatchConstructs(jobId, btn) {
  if (!btn || btn.disabled) return;
  btn.disabled = true;
  btn.innerHTML = _SAVE_SVG + ' Saving…';
  fetch('/api/batch/' + jobId + '/save-all-constructs', {method: 'POST'})
  .then(function(r) { return r.json(); })
  .then(function(data) {
    if (data.saved !== undefined) {
      btn.innerHTML = '<svg width="12" height="12" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><polyline points="20 6 9 17 4 12"/></svg> Saved ' + data.saved + ' construct' + (data.saved === 1 ? '' : 's');
      btn.style.opacity = '0.7';
    } else {
      btn.innerHTML = _SAVE_SVG + ' Save failed';
      btn.disabled = false;
    }
  })
  .catch(function() {
    btn.innerHTML = _SAVE_SVG + ' Save failed';
    btn.disabled = false;
  });
}

function saveAllBatchToLocal(jobId, btn) {
  if (!btn || btn.disabled) return;
  btn.disabled = true;
  btn.textContent = 'Saving…';
  fetch('/api/batch/' + jobId + '/save-all-local', {method: 'POST'})
  .then(function(r) { return r.json(); })
  .then(function(data) {
    if (data.saved !== undefined) {
      btn.textContent = '✓ Saved ' + data.saved + ' to library';
    } else {
      btn.textContent = 'Save failed';
      btn.disabled = false;
    }
  })
  .catch(function() {
    btn.textContent = 'Save failed';
    btn.disabled = false;
  });
}

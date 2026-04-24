
const API = '';  // Same origin
let _pollTimer = null;

// v14 Phase 8: safeFetch global error boundary
async function safeFetch(url, options = {}) {
    try {
        const resp = await fetch(url, options);
        if (!resp.ok) {
            const msg = resp.status === 404 ? '资源未找到' :
                        resp.status === 500 ? '服务器内部错误' : `请求失败 (${resp.status})`;
            showToast(msg, 'error');
            return null;
        }
        return resp;
    } catch (e) {
        showToast('网络连接失败，请检查服务是否运行', 'error');
        return null;
    }
}

// v14 Phase 8: leaderboard pagination state
let _lbOffset = 0;
const _LB_PAGE_SIZE = 20;
let _currentRunId = null;
let _lastProgressSnapshot = { completed: 0, total: 0, phase: 'queued', lastLayer: null, lastProbe: null, lastEvidenceCount: 0 };
let _consoleLines = [];
let _typingTimer = null;
let _logPinnedToBottom = true;
let _logFilter = 'all';

let _allRuns = [];
let _filteredRuns = [];
let _runPage = 1;
const _runPageSize = 10;
let _runSelected = new Set();

// v6 fix: Event delegation for dynamic buttons (replaces vulnerable inline onclick)
document.addEventListener('click', (e) => {
  const btn = e.target.closest('[data-action]');
  if (!btn) return;
  const action = btn.dataset.action;
  const runId = btn.dataset.runId;
  const baselineId = btn.dataset.baselineId;

  if (action === 'cancel') cancelRun(runId);
  else if (action === 'retry') retryRun(runId);
  else if (action === 'markBaseline') markAsBaseline(runId);
  else if (action === 'unmarkBaseline') unmarkAsBaseline(runId, baselineId);
  else if (action === 'compareBaseline') compareWithBaseline(runId);
  else if (action === 'exportPdf') exportReportPdf(runId);
  else if (action === 'downloadRadar') downloadRadarSvg(runId);
  else if (action === 'deleteRun') deleteRunFromList(e, runId);
  else if (action === 'openTask') openTask(runId);
  else if (action === 'toggleCase') toggleCase(btn.dataset.caseId);
  else if (action === 'changePage') changeRunsPage(parseInt(btn.dataset.dir));
  else if (action === 'viewBaseline') viewBaselineReport(baselineId);
  else if (action === 'deleteBaseline') deleteBaseline(baselineId);
  else if (action === 'continueFullTest') continueFullTest(runId);
  else if (action === 'skipTesting') skipTesting(runId);

  else if (action === 'copyDiagnostic') {
    const msgEl = document.getElementById('error-msg-' + runId);
    if (msgEl) {
      copyText(msgEl.textContent);
    }
  }
});

// v6 fix: Attribute escape helper (prevents XSS in data-* attributes)
function escAttr(s) {
  return String(s||'')
    .replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/'/g,'&#39;')
    .replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/\n/g,'\\n').replace(/\r/g,'\\r');
}

// Copy text to clipboard
async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    showToast('已复制到剪贴板');
  } catch (err) {
    // Fallback for older browsers or non-secure contexts
    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.style.position = 'fixed';
    textarea.style.opacity = '0';
    document.body.appendChild(textarea);
    textarea.select();
    try {
      document.execCommand('copy');
      showToast('已复制到剪贴板');
    } catch (e) {
      showToast('复制失败，请手动复制');
    }
    document.body.removeChild(textarea);
  }
}

// ── Markdown stripping (for plain-text log display) ───────────────────────
function stripMd(s) {
  if (!s) return s;
  return s
    .replace(/```[\s\S]*?```/g, t => t.replace(/```\w*\n?/g, '').replace(/```/g, ''))  // code fences
    .replace(/\*\*(.+?)\*\*/g, '$1')    // **bold**
    .replace(/\*(.+?)\*/g, '$1')        // *italic*
    .replace(/__(.+?)__/g, '$1')        // __bold__
    .replace(/_(.+?)_/g, '$1')          // _italic_
    .replace(/^#{1,6}\s+/gm, '')        // # headings
    .replace(/^>\s?/gm, '')             // > blockquotes
    .replace(/^[-*+]\s+/gm, '• ')       // - / * / + list items → bullet
    .replace(/^\d+\.\s+/gm, t => t)     // keep numbered lists
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1');  // [text](url) → text
}

// ── Page routing ───────────────────────────────────────────────────────────

function showPage(name) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));

  const pageMap = {home:'page-home', runs:'page-runs', benchmarks:'page-benchmarks', task:'page-task', leaderboard:'page-leaderboard'};
  const navMap  = {home:'nav-home', runs:'nav-runs', benchmarks:'nav-benchmarks', leaderboard:'nav-leaderboard'};

  const pageEl = document.getElementById(pageMap[name]);
  if (pageEl) pageEl.classList.add('active');
  const navEl = document.getElementById(navMap[name]);
  if (navEl) navEl.classList.add('active');

  if (name === 'runs')       loadRuns();
  if (name === 'benchmarks') loadBaselines();
  if (name === 'leaderboard') loadLeaderboard();
  if (_pollTimer && name !== 'task') { clearInterval(_pollTimer); _pollTimer = null; }
  // Close the SSE connection when leaving the task page so it doesn't hold
  // the server's thread and block subsequent API requests.
  if (name !== 'task' && _eventSource) { _eventSource.close(); _eventSource = null; }
}

async function loadLeaderboard(offset = 0) {
  _lbOffset = offset;
  const c = document.getElementById('leaderboard-content');
  c.innerHTML = '<div class="loading"><div class="spinner"></div><div>加载中...</div></div>';
  const resp = await safeFetch(API + `/api/v1/elo-leaderboard?limit=${_LB_PAGE_SIZE}&offset=${offset}`);
  if (!resp) { c.innerHTML = '<div class="fail">加载失败</div>'; return; }
  const data = await resp.json().catch(() => []);

  // Search input
  const searchId = 'lb-search-input';
  let searchVal = '';
  const existingSearch = document.getElementById(searchId);
  if (existingSearch) searchVal = existingSearch.value;

  function renderTable(rows) {
    if (!rows || rows.length === 0) {
      return '<div style="color:var(--ink4);padding:40px;text-align:center">暂无排行榜数据，请先完成基准测试对比</div>';
    }
    let html = `
      <table class="data-table">
        <thead>
          <tr>
            <th>排名</th>
            <th>模型名称</th>
            <th>ELO 积分 (对战总局数)</th>
            <th>胜 / 平 / 负</th>
            <th>最高分</th>
            <th>最后测试时间</th>
          </tr>
        </thead>
        <tbody>
    `;
    rows.forEach((r, idx) => {
      const globalIdx = offset + idx;
      const medals = ['🥇', '🥈', '🥉'];
      const rankLabel = globalIdx < 3 ? medals[globalIdx] : `#${globalIdx+1}`;
      const dt = new Date(r.updated_at).toLocaleString();
      html += `
        <tr>
          <td style="font-weight:bold;font-size:16px">${rankLabel}</td>
          <td style="font-weight:600">${escHtml(r.display_name)} <br><span style="font-size:11px;font-weight:normal;color:var(--ink4)">${escHtml(r.model_name)}</span></td>
          <td><b>${r.elo_rating.toFixed(0)}</b> <span style="color:var(--ink4);font-weight:normal;font-size:12px">(${r.games_played} 局)</span></td>
          <td style="color:var(--ink3);font-size:13px"><span style="color:var(--green)">${r.wins}</span> / <span style="color:var(--amber)">${r.draws}</span> / <span style="color:var(--red)">${r.losses}</span></td>
          <td style="color:var(--ink3)">${r.peak_elo.toFixed(0)}</td>
          <td style="color:var(--ink4);font-size:12px">${dt}</td>
        </tr>
      `;
    });
    html += '</tbody></table>';
    return html;
  }

  const filtered = searchVal
    ? (data || []).filter(r => (r.display_name || r.model_name || '').toLowerCase().includes(searchVal.toLowerCase()))
    : (data || []);

  const hasNext = (data || []).length >= _LB_PAGE_SIZE;
  const hasPrev = offset > 0;

  let out = `
    <div style="display:flex;gap:8px;align-items:center;margin-bottom:12px">
      <input id="${searchId}" type="text" placeholder="搜索模型名称…" value="${escAttr(searchVal)}"
             style="flex:1;padding:7px 10px;border:1px solid var(--rule2);border-radius:var(--radius);font-size:13px"
             oninput="document.getElementById('lb-table-body').innerHTML=renderLbTable(this.value)">
      <span style="font-size:12px;color:var(--ink4)">第 ${offset+1}–${offset+(data||[]).length} 条</span>
      <button class="btn" style="padding:6px 12px;font-size:12px" onclick="loadLeaderboard(${offset - _LB_PAGE_SIZE})" ${!hasPrev ? 'disabled' : ''}>← 上页</button>
      <button class="btn" style="padding:6px 12px;font-size:12px" onclick="loadLeaderboard(${offset + _LB_PAGE_SIZE})" ${!hasNext ? 'disabled' : ''}>下页 →</button>
    </div>
    <div id="lb-table-body">${renderTable(filtered)}</div>`;

  c.innerHTML = out;

  // attach live search after render
  const si = document.getElementById(searchId);
  if (si) {
    // store data on container for client-side filter
    c.dataset.rows = JSON.stringify(data || []);
    c.dataset.offset = String(offset);
    si.oninput = function() {
      const rows = JSON.parse(c.dataset.rows || '[]');
      const q = this.value.toLowerCase();
      const f = q ? rows.filter(r => (r.display_name || r.model_name || '').toLowerCase().includes(q)) : rows;
      document.getElementById('lb-table-body').innerHTML = renderTable(f);
    };
  }
}


// ── API helpers ────────────────────────────────────────────────────────────

async function api(method, path, body, timeoutMs = 15000) {
  const opts = { method, headers: {'Content-Type':'application/json'} };
  if (body) opts.body = JSON.stringify(body);
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  opts.signal = controller.signal;
  try {
    const r = await fetch(API + path, opts);
    clearTimeout(timer);
    const text = await r.text();
    let data;
    try { data = JSON.parse(text); } catch { data = {error: text}; }
    return { ok: r.ok, status: r.status, data };
  } catch (e) {
    clearTimeout(timer);
    const msg = e.name === 'AbortError' ? '请求超时，请检查服务器状态' : `网络错误: ${e.message}`;
    return { ok: false, status: 0, data: { error: msg } };
  }
}

// ── Form validation ────────────────────────────────────────────────────────

function validateUrl(input) {
  const v = input.value.trim();
  const hint = document.getElementById('url-hint');
  const blocked = /localhost|127\.0\.0\.1|0\.0\.0\.0|192\.168\.|10\.|172\.(1[6-9]|2\d|3[01])\./i;
  if (v && blocked.test(v)) {
    hint.style.color = 'var(--red)';
    hint.textContent = '不允许内网/本地地址（SSRF防护）';
    input.style.borderColor = 'var(--red)';
  } else if (v && !v.startsWith('http')) {
    hint.style.color = 'var(--amber)';
    hint.textContent = '必须以 http:// 或 https:// 开头';
    input.style.borderColor = 'var(--amber)';
  } else {
    hint.style.color = 'var(--ink4)';
    hint.textContent = '仅支持 http/https，不允许内网地址';
    input.style.borderColor = '';
  }
}

// ── Submit run ─────────────────────────────────────────────────────────────

// ── Form submit handler ──────────────────────────────────────────────────────

function submitForm(event) {
  event.preventDefault();
  submitRun();
  return false;
}

// ── LocalStorage auto-fill ───────────────────────────────────────────────────

const STORAGE_KEY = 'llm_inspector_form';

function saveForm() {
  const data = {
    url:   document.getElementById('f-url').value.trim(),
    key:   document.getElementById('f-key').value.trim(),
    model: document.getElementById('f-model').value.trim(),
  };
  if (data.url && data.key && data.model) {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(data)); } catch {}
  }
}

function restoreForm() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    const data = JSON.parse(raw);
    if (data.url)   document.getElementById('f-url').value = data.url;
    if (data.key)   document.getElementById('f-key').value = data.key;
    if (data.model) document.getElementById('f-model').value = data.model;
  } catch {}
}

function clearSavedForm() {
  try { localStorage.removeItem(STORAGE_KEY); } catch {}
  document.getElementById('f-url').value = '';
  document.getElementById('f-key').value = '';
  document.getElementById('f-model').value = '';
  document.getElementById('submit-hint').textContent = '已清除保存的信息';
  document.getElementById('submit-hint').style.color = 'var(--ink4)';
}

async function submitRun() {
  const url   = document.getElementById('f-url').value.trim();
  const key   = document.getElementById('f-key').value.trim();
  const model = document.getElementById('f-model').value.trim();
  const mode  = document.getElementById('f-mode').value;
  const hint  = document.getElementById('submit-hint');
  const btn   = document.getElementById('submit-btn');

  if (!url || !key || !model) { hint.textContent = '请填写所有必填项'; return; }
  if (!url.startsWith('http')) { hint.textContent = 'URL 必须以 http:// 或 https:// 开头'; return; }

  btn.disabled = true;
  btn.textContent = '提交中...';
  hint.textContent = '';

  const {ok, data} = await api('POST', '/api/v1/runs', {
    base_url: url, api_key: key, model, test_mode: mode
  });

  btn.disabled = false;
  btn.textContent = '开始检测';

  if (!ok) {
    hint.textContent = '错误: ' + (data.error || '提交失败');
    hint.style.color = 'var(--red)';
    return;
  }

  // Save for next time
  saveForm();

  // Navigate to task page
  openTask(data.run_id);
}

// ── Task detail page ───────────────────────────────────────────────────────

let _eventSource = null;

function openTask(runId) {
  _currentRunId = runId;
  _lastProgressSnapshot = { completed: 0, total: 0, phase: 'queued', lastLayer: null, lastProbe: null, lastEvidenceCount: 0 };
  _consoleLines = [{ text: '初始化任务上下文...', type: 'normal' }];
  showPage('task');
  document.getElementById('task-title').textContent = '检测: ' + runId.slice(0,8) + '...';
  document.getElementById('task-status-badge').innerHTML = '';
  document.getElementById('task-content').innerHTML =
    '<div class="loading"><div class="spinner"></div><div>检测进行中，请稍候...</div></div>';

  if (_pollTimer) clearInterval(_pollTimer);
  if (_eventSource) { _eventSource.close(); _eventSource = null; }

  // Fallback polling for overall progress
  _pollTimer = setInterval(() => pollTask(runId), 2000);
  pollTask(runId);

  // Setup SSE for real-time logs
  setupSSE(runId);
}

function setupSSE(runId) {
  _eventSource = new EventSource(API + '/api/v10/runs/' + runId + '/logs/stream');
  _eventSource.onmessage = (e) => {
    try {
      const log = JSON.parse(e.data);
      if (log.event_type) {
        // Classify by event_type from Phase 4 EventKind enum
        const kind = log.event_type.split('.')[0]; // "probe", "case", "judge", "cb", "retry"
        const isBad = log.event_type.includes('failed') || log.event_type.includes('open');
        pushConsole(
          `[${log.event_type}] ${log.message || JSON.stringify(log.payload || {})}`,
          isBad ? 'del' : 'meta',
          kind
        );
      } else if (log.message) {
        pushConsole(`[SSE] ${log.message}`, log.level === 'error' ? 'del' : 'normal');
      } else if (log.event === 'judgment') {
        pushConsole(`[SSE 判题] ${log.method}: ${log.result?.passed ? '通过' : '未通过'}`, log.result?.passed ? 'normal' : 'del');
      }
    } catch (err) {}
  };
  _eventSource.onerror = () => {
    _eventSource.close();
  };
}

async function pollTask(runId) {
  const [runRes, respRes] = await Promise.all([
    api('GET', '/api/v1/runs/' + runId),
    api('GET', '/api/v1/runs/' + runId + '/responses'),
  ]);
  if (!runRes.ok) return;

  const data = runRes.data;
  const responses = respRes.ok ? respRes.data : [];
  const status = data.status;

  document.getElementById('task-status-badge').innerHTML = renderStatusBadge(status);
  renderTaskActions(runId, status, data.baseline_id, data);
  setNavStatus(status);

  renderTaskProgress(data, responses);

  if (['completed','partial_failed','failed'].includes(status)) {
    clearInterval(_pollTimer); _pollTimer = null;
    if (status !== 'failed') loadReport(runId);
  }
  // pre_detected: stop polling, wait for user action (continue/skip buttons)
  if (status === 'pre_detected') {
    clearInterval(_pollTimer); _pollTimer = null;
  }
  // preflight_failed / suspended / cancelled: terminal, stop polling
  if (['preflight_failed','suspended','cancelled'].includes(status)) {
    clearInterval(_pollTimer); _pollTimer = null;
  }
}

function renderStatusBadge(status) {
  const labels = {
    queued:'排队中', pre_detecting:'预检测中', pre_detected:'预检测完成',
    running:'测试执行中', completed:'已完成', partial_failed:'部分失败', failed:'失败',
    preflight_running:'连接预检中', preflight_failed:'连接预检失败'
  };
  const cls = {completed:'badge-green', failed:'badge-red', partial_failed:'badge-amber',
               preflight_failed:'badge-red'};
  const c = cls[status] || 'badge-blue';
  return `<span class="badge ${c}">${labels[status]||status}</span>`;
}

function renderTaskActions(runId, status, baselineId, data) {
  const el = document.getElementById('task-actions');
  if (!el) return;

  const canCancel = ['queued','preflight_running','pre_detecting','running'].includes(status);
  const canContinue = status === 'pre_detected';
  const canRetry = ['failed','partial_failed'].includes(status);
  const canExport = ['completed','partial_failed'].includes(status);

  let html = '';
  if (canCancel) {
    html += `<button class="btn danger" style="padding:6px 12px;font-size:12px" data-action="cancel" data-run-id="${escAttr(runId)}">停止任务</button>`;
  }
  if (canContinue) {
    html += `<button class="btn primary" style="padding:6px 12px;font-size:12px" data-action="continueFullTest" data-run-id="${escAttr(runId)}">继续完整测试</button>`;
    html += `<button class="btn" style="padding:6px 12px;font-size:12px" data-action="skipTesting" data-run-id="${escAttr(runId)}">直接生成报告</button>`;
  }
  if (canRetry) {
    html += `<button class="btn" style="padding:6px 12px;font-size:12px" data-action="retry" data-run-id="${escAttr(runId)}">重新重试</button>`;
  }
  if (canExport) {
    if (!baselineId) {
      html += `<button class="btn primary" style="padding:6px 12px;font-size:12px" data-action="markBaseline" data-run-id="${escAttr(runId)}">标记为基准</button>`;
    }
    html += `<button class="btn" style="padding:6px 12px;font-size:12px" data-action="compareBaseline" data-run-id="${escAttr(runId)}">与基准对比</button>`;
    html += `<button class="btn" style="padding:6px 12px;font-size:12px" data-action="exportPdf" data-run-id="${escAttr(runId)}">导出报告</button>`;
  }
  el.innerHTML = html;
}

async function cancelRun(runId) {
  showConfirmModal('确认停止当前任务？已发出的请求可能仍会完成，但会尽快停止后续请求。', async () => {
    const {ok, data} = await api('POST', `/api/v1/runs/${runId}/cancel`);
    if (!ok) {
      showToast('停止失败: ' + (data.error || 'unknown error'), 'error');
      return;
    }
    setNavStatus('cancelling');
    pollTask(runId);
  });
}

async function retryRun(runId) {
  const {ok, data} = await api('POST', `/api/v1/runs/${runId}/retry`);
  if (!ok) {
    showToast('重试失败: ' + (data.error || 'unknown error'), 'error');
    return;
  }
  openTask(data.run_id || runId);
}

async function continueFullTest(runId) {
  const {ok, data} = await api('POST', `/api/v1/runs/${runId}/continue`);
  if (!ok) {
    showToast('继续测试失败: ' + (data.error || 'unknown error'), 'error');
    return;
  }
  // Restart polling
  if (_pollTimer) clearInterval(_pollTimer);
  _pollTimer = setInterval(() => pollTask(runId), 2000);
  pollTask(runId);
}

async function unmarkAsBaseline(runId, baselineId) {
  if (!await showConfirmModal('确认该模型不再作为对比基准？这不会删除原始检测记录。')) return;
  const {ok, data} = await api('DELETE', '/api/v1/baselines/' + baselineId);
  if (!ok) {
    showToast('移除失败: ' + (data.error || 'unknown error'), 'error');
    return;
  }
  // Refresh current page actions
  const {ok: ok2, data: data2} = await api('GET', '/api/v1/runs/' + runId);
  if (ok2) {
    renderTaskActions(runId, data2.status, data2.baseline_id);
  }
}

async function skipTesting(runId) {
  const {ok, data} = await api('POST', `/api/v1/runs/${runId}/skip-testing`);
  if (!ok) {
    showToast('跳过失败: ' + (data.error || 'unknown error'), 'error');
    return;
  }
  // Restart polling to catch completion
  if (_pollTimer) clearInterval(_pollTimer);
  _pollTimer = setInterval(() => pollTask(runId), 2000);
  pollTask(runId);
}

function exportReportPdf(runId) {
  // Use browser's native high-quality PDF generation
  // We've configured CSS @media print to layout left first, then right (full logs)
  const oldTitle = document.title;
  document.title = 'LLM_Inspector_Report_' + runId.slice(0, 8);
  window.print();
  document.title = oldTitle;
}

function downloadRadarSvg(runId) {
  window.open(`/api/v1/runs/${encodeURIComponent(runId)}/radar.svg`, '_blank');
}

// -- ECharts Radar Chart

// Radar dimensions: [label, path-in-scorecard]
// API returns scores already on 0-100 integer scale under scorecard.breakdown.*
// Top-level keys: capability_score, authenticity_score, performance_score (all 0-100)
const RADAR_DIMS = [
  { key: 'reasoning',           label: '推理',   path: ['breakdown', 'reasoning'] },
  { key: 'adversarial',        label: '对抗',   path: ['breakdown', 'adversarial_reasoning'] },
  { key: 'instruction',        label: '指令',   path: ['breakdown', 'instruction'] },
  { key: 'coding',             label: '编码',   path: ['breakdown', 'coding'] },
  { key: 'safety',             label: '安全',   path: ['breakdown', 'safety'] },
  { key: 'consistency',        label: '一致性', path: ['breakdown', 'consistency'] },
  { key: 'authenticity_score', label: '真实性', path: ['authenticity_score'] },
  { key: 'performance_score',  label: '性能',   path: ['performance_score'] },
];

// Read a potentially-nested value from scorecard object
function _scGet(sc, path) {
  let v = sc;
  for (const k of path) { v = (v != null) ? v[k] : undefined; }
  return (v != null && !isNaN(v)) ? Number(v) : null;
}

let _radarCurrentData = null;

// v14 Phase 8: bar chart fallback when < 5 active dims
function renderBarChartFallback(dims, containerId) {
  const container = document.getElementById(containerId);
  if (!container) return;
  const rows = dims.map(d => {
    const pct = Math.min(100, Math.max(0, d.value || 0));
    const color = pct >= 80 ? '#16a34a' : pct >= 60 ? '#2563eb' : pct >= 40 ? '#d97706' : '#dc2626';
    return `<div style="margin-bottom:8px">
      <div style="display:flex;justify-content:space-between;font-size:12px;color:var(--ink3);margin-bottom:3px">
        <span>${escHtml(d.label)}</span><span style="color:${color};font-weight:600">${pct.toFixed(1)}</span>
      </div>
      <div style="height:8px;background:var(--bg2);border-radius:4px;overflow:hidden">
        <div style="width:${pct}%;height:100%;background:${color};border-radius:4px;transition:width .4s"></div>
      </div>
    </div>`;
  }).join('');
  container.innerHTML = `<div style="padding:16px">${rows || '<p style="color:var(--ink4)">暂无雷达图数据</p>'}</div>`;
}

function renderRadarChart(containerId, data, mode) {
  _radarCurrentData = data;
  const container = document.getElementById(containerId);
  if (!container) return;

  if (!window.echarts) {
    // Fallback to SVG iframe when ECharts CDN is unavailable
    container.innerHTML =
      `<iframe src="/api/v1/runs/${encodeURIComponent(data.run_id || '')}/radar.svg" style="width:100%;height:400px;border:0"></iframe>`;
    return;
  }

  // Dispose existing chart instance if any
  const existing = echarts.getInstanceByDom(container);
  if (existing) existing.dispose();

  const scorecard = data.scorecard || {};

  // API stores scores in 0-10000 scale:
  //   ScoreCard internally uses 0-100, to_dict() multiplies by 100 → 0-10000
  //   (SVG radar confirms: rings at 2000/4000/6000/8000/10000, max 10000)
  // Normalise to 0-100 for human-readable display.
  // v14 Phase 8: filter out null/undefined dimensions
  const allDims = RADAR_DIMS.map(d => {
    const raw = _scGet(scorecard, d.path);
    return { ...d, value: raw !== null ? raw / 100 : null };
  });
  const activeDims = allDims.filter(d => d.value !== null && d.value !== undefined);

  if (activeDims.length < 5) {
    renderBarChartFallback(activeDims.map(d => ({ label: d.label, value: d.value })), containerId);
    return;
  }

  const chart = echarts.init(container);

  let values, maxVal;
  if (mode === 'stanine') {
    // 0-100 pct → Stanine 1-9
    maxVal = 9;
    values = activeDims.map(d => Math.max(1, Math.min(9, Math.round(d.value / 100 * 8 + 1))));
  } else if (mode === 'theta') {
    // 0-100 pct → θ ∈ [-3, 3]
    maxVal = 3;
    values = activeDims.map(d => {
      const t = (d.value / 100 - 0.5) * 4;
      return Math.max(-3, Math.min(3, Math.round(t * 10) / 10));
    });
  } else {
    // percent: 0-100
    maxVal = 100;
    values = activeDims.map(d => Math.round(d.value * 10) / 10);   // 1 decimal place
  }

  const indicators = activeDims.map(d => ({
    name: d.label,
    max: maxVal,
    min: mode === 'theta' ? -3 : 0,
  }));

  chart.setOption({
    tooltip: {
      trigger: 'item',
      formatter: params => {
        const vals = params.value || [];
        const unit = mode === 'stanine' ? '' : mode === 'theta' ? '' : '%';
        return activeDims.map((d, i) => {
          const v = vals[i] ?? 0;
          return `${d.label}: <b>${v}${unit}</b>`;
        }).join('<br/>');
      }
    },
    radar: {
      indicator: indicators,
      center: ['50%', '50%'],
      radius: '65%',
      splitNumber: 4,
      axisName: { color: '#555', fontSize: 12, fontWeight: 'bold' },
      splitArea: { areaStyle: { color: ['rgba(75,108,247,0.03)', 'rgba(75,108,247,0.06)'] } },
      axisLine: { lineStyle: { color: 'rgba(0,0,0,0.15)' } },
      splitLine: { lineStyle: { color: 'rgba(0,0,0,0.1)' } },
    },
    series: [{
      type: 'radar',
      data: [{
        value: values,
        name: scorecard.claimed_model || '待测模型',
        areaStyle: { color: 'rgba(75,108,247,0.15)' },
        lineStyle: { width: 2, color: '#4b6cf7' },
        itemStyle: { color: '#4b6cf7' },
        symbol: 'circle',
        symbolSize: 5,
      }]
    }]
  });
}

function switchRadarScale(mode, btn) {
  // Toggle active class on sibling chips
  if (btn && btn.parentElement) {
    btn.parentElement.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
    btn.classList.add('active');
  }
  if (_radarCurrentData) {
    renderRadarChart('radar-chart-container', _radarCurrentData, mode);
  }
}

function exportReportPdfFromList(evt, runId) {
  if (evt) evt.stopPropagation();
  openTask(runId);
  // Wait short time to ensure it's loaded, then print
  setTimeout(() => {
    exportReportPdf(runId);
  }, 1000);
}

function quickExportRadar(evt, runId) {
  if (evt) evt.stopPropagation();
  downloadRadarSvg(runId);
}

function renderTaskProgress(data, responses = []) {
  const prog = data.progress || {};
  const pct = prog.total ? Math.round(prog.completed / prog.total * 100) : 0;
  const scoringProfileVersion = data.scoring_profile_version || 'v1';

  animateProcessConsole(prog, data.status, data.predetect_result, responses);

  const pre = data.predetect_result;
  const pf = data.preflight_report || null;

  const preflightHtml = pf ? renderPreflightCard(pf) : '';
  const preHtml = pre ? renderPredetectCard(pre, data.run_id, data.status) : '';
  const progressHtml = prog.total > 0 ? `
    <div class="card">
      <div class="card-label">测试进度</div>
      <div style="display:flex;justify-content:space-between;font-size:12px;color:var(--ink3);margin-bottom:4px">
        <span>已完成 ${prog.completed} / ${prog.total} 题</span>
        <span>${pct}%</span>
      </div>
      <div class="progress-bar"><div class="progress-fill" style="width:${pct}%"></div></div>
    </div>` : '';

  const errorHtml = data.error_message ? `<div class="card" style="border-color:#f0b8b8;background:var(--red-bg)">
      <div style="color:var(--red);font-size:13px;font-weight:600">诊断错误 (Code: ${data.error_code || 'E_RUN_FAIL'})</div>
      <div style="color:var(--red);font-size:12px;margin-top:4px" id="error-msg-${data.run_id || 'current'}">${escHtml(data.error_message)}</div>
      <div style="margin-top:8px"><button class="btn" style="padding:2px 8px;font-size:11px" data-action="copyDiagnostic" data-run-id="${escAttr(data.run_id || 'current')}">复制诊断信息</button></div>
      </div>` : '';

  let html = `
    <div class="task-split">
      <div id="task-main-col">
        <div class="card">
          <div class="card-label">检测目标</div>
          <div style="font-size:13px;color:var(--ink2);margin-bottom:4px">
            <strong>${escHtml(data.model)}</strong>
            <span style="color:var(--ink4);margin-left:8px;font-family:var(--mono);font-size:11px">${escHtml(data.base_url)}</span>
          </div>
          <div style="font-size:11px;color:var(--ink4)">模式: ${data.test_mode} · 创建: ${fmtTime(data.created_at)}</div>
        <div style="font-size:11px;color:var(--ink4);margin-top:2px">评分配置版本: ${escHtml(scoringProfileVersion)}</div>
        </div>

        ${renderStageExperience(data, prog, pct, {includeConsole:false})}
        ${preflightHtml}
        ${preHtml}
        ${progressHtml}
        ${errorHtml}
        <div id="report-section"></div>
      </div>
      <aside class="task-right" id="task-log-col">
        ${renderLogPanel(data)}
      </aside>
    </div>`;

  const oldBox = document.getElementById('live-console-body');
  let oldTop = 0;
  if (oldBox) {
    const distanceToBottom = oldBox.scrollHeight - oldBox.scrollTop - oldBox.clientHeight;
    _logPinnedToBottom = distanceToBottom <= 24;
    oldTop = oldBox.scrollTop;
  }

  document.getElementById('task-content').innerHTML = html;
  bindLogScrollBehavior();

  const newBox = document.getElementById('live-console-body');
  if (newBox) {
    if (_logPinnedToBottom) {
      newBox.scrollTop = newBox.scrollHeight;
    } else {
      newBox.scrollTop = oldTop;
    }
  }
}

function renderLogPanel(data) {
  return `
    <div class="card" style="padding:0;overflow:hidden">
      <div class="console-head" style="position:sticky;top:0;z-index:1"><span>执行日志</span><span>${escHtml(data.status || 'running')}</span></div>
      <div class="log-filter" id="log-filter-bar">
        <button class="chip ${_logFilter === 'all' ? 'active' : ''}" onclick="setLogFilter('all')">全部</button>
        <button class="chip ${_logFilter === 'prompt' ? 'active' : ''}" onclick="setLogFilter('prompt')">仅看提示词</button>
        <button class="chip ${_logFilter === 'error' ? 'active' : ''}" onclick="setLogFilter('error')">仅看 error</button>
        <button class="chip ${_logFilter === 'probe' ? 'active' : ''}" onclick="setLogFilter('probe')">PROBE</button>
        <button class="chip ${_logFilter === 'case' ? 'active' : ''}" onclick="setLogFilter('case')">CASE</button>
        <button class="chip ${_logFilter === 'judge' ? 'active' : ''}" onclick="setLogFilter('judge')">JUDGE</button>
        <button class="chip ${_logFilter === 'cb' ? 'active' : ''}" onclick="setLogFilter('cb')">断路器</button>
        <button class="chip ${_logFilter === 'retry' ? 'active' : ''}" onclick="setLogFilter('retry')">重试</button>
      </div>
      <div class="console-body" id="live-console-body" style="min-height:420px;max-height:80vh">${renderConsoleLines()}</div>
      <div style="padding:8px 10px;border-top:1px solid var(--rule);font-size:11px;color:var(--ink4)">
        提示：滚动查看历史日志时，不会自动打断当前位置。
      </div>
    </div>`;
}

function bindLogScrollBehavior() {
  const box = document.getElementById('live-console-body');
  if (!box || box.dataset.bound === '1') return;
  box.dataset.bound = '1';
  box.addEventListener('scroll', () => {
    const threshold = 24;
    const distanceToBottom = box.scrollHeight - box.scrollTop - box.clientHeight;
    _logPinnedToBottom = distanceToBottom <= threshold;
  });
}

function updateLogPanel() {
  const box = document.getElementById('live-console-body');
  if (!box) return;
  const shouldStick = _logPinnedToBottom;
  const prevTop = box.scrollTop;
  box.innerHTML = renderConsoleLines();
  if (shouldStick) {
    box.scrollTop = box.scrollHeight;
  } else {
    box.scrollTop = prevTop;
  }
}

function setLogFilter(mode) {
  _logFilter = mode;
  const bar = document.getElementById('log-filter-bar');
  if (bar) {
    bar.querySelectorAll('.chip').forEach(chip => chip.classList.remove('active'));
    const map = { all: 0, prompt: 1, error: 2 };
    const idx = map[mode] ?? 0;
    const target = bar.querySelectorAll('.chip')[idx];
    if (target) target.classList.add('active');
  }
  updateLogPanel();
}

function matchLogFilter(line) {
  if (_logFilter === 'all') return true;
  const text = String(line?.text || '').toLowerCase();
  const tag = String(line?.tag || '');
  if (_logFilter === 'prompt') return text.includes('[提示词]');
  if (_logFilter === 'error') {
    return text.includes('[结果] 错误') || text.includes(' error') || text.includes('failed') || line?.type === 'del';
  }
  if (_logFilter === 'probe') return tag === 'probe' || text.includes('[预检测]') || text.includes('probe.');
  if (_logFilter === 'case') return tag === 'case' || text.includes('[用例]') || text.includes('case.');
  if (_logFilter === 'judge') return tag === 'judge' || text.includes('[判题]') || text.includes('judge.');
  if (_logFilter === 'cb') return tag === 'cb' || text.includes('cb.') || text.includes('断路器');
  if (_logFilter === 'retry') return tag === 'retry' || text.includes('retry.') || text.includes('重试');
  return true;
}

function renderStageExperience(data, prog, pct, opts = {includeConsole:true}) {
  const stage = stageFromStatus(data.status, prog, data.predetect_result);
  const stageNames = ['排队', '预检测', '能力测试', '分析报告'];

  const stageItems = stageNames.map((name, idx) => {
    const state = idx < stage ? 'done' : idx === stage ? 'active' : '';
    const step1Text = data.status === 'preflight_running' ? '连接预检中...'
      : data.status === 'preflight_failed' ? `预检失败: ${data.error_message || '连接异常'}`
      : `任务已提交${data.created_at ? ' · ' + fmtTime(data.created_at) : ''}`;
    const texts = [
      step1Text,
      (() => {
        const cl = data.predetect_result?.current_layer || '';
        const pm = cl.match(/^parallel_([^(]+)\((\d+)\/(\d+)\)$/);
        if (pm) return `⚡ 并行探测 ${pm[2]}/${pm[3]} 层`;
        if (data.predetect_result?.identified_as) return `识别: ${data.predetect_result.identified_as}`;
        return cl ? `${cl.split('/')[1] || cl}` : '执行指纹探测';
      })(),
      prog.total ? `${prog.completed}/${prog.total} 用例` : '等待执行',
      ['completed','partial_failed'].includes(data.status) ? '报告已生成' : '等待汇总',
    ];
    return `<div class="stage-item ${state}">
      <div class="k">STEP ${idx + 1}</div>
      <div class="v">${name}</div>
      <div class="stage-sub" style="margin-top:6px">${escHtml(texts[idx])}</div>
    </div>`;
  }).join('');

  const consoleBlock = opts.includeConsole ? `
      <div class="live-console">
        <div class="console-head"><span>执行日志</span><span>${escHtml(data.status || 'running')}</span></div>
        <div class="console-body" id="live-console-body">${renderConsoleLines()}</div>
      </div>` : '';

  return `
    <div class="stage-card">
      <div class="stage-head">
        <div>
          <div class="stage-title">检测过程可视化</div>
          <div class="stage-sub">当前阶段：${escHtml(
            data.status === 'preflight_running' ? '连接预检' :
            data.status === 'preflight_failed'  ? '预检失败' :
            stageNames[stage] || '分析报告'
          )} · 进度 ${pct}%</div>
        </div>
        <div>${renderStatusBadge(data.status)}</div>
      </div>
      <div class="stage-track">${stageItems}</div>
      ${consoleBlock}
    </div>`;
}

function stageFromStatus(status, prog, pre) {
  if (status === 'queued' || status === 'preflight_running' || status === 'preflight_failed') return 0;
  if (status === 'pre_detecting' || status === 'pre_detected') return 1;
  if (status === 'running' || ((prog?.completed || 0) > 0 && !['completed','partial_failed','failed'].includes(status))) return 2;
  if (['completed','partial_failed','failed'].includes(status)) return 3;
  if (pre?.identified_as) return 1;
  return 0;
}

function animateProcessConsole(prog, status, pre, responses = []) {
  const prev = _lastProgressSnapshot;
  const currCompleted = prog?.completed || 0;
  const currTotal = prog?.total || 0;

  if (status !== prev.phase) {
    const phaseLabel = {
      queued: '排队中',
      preflight_running: '连接预检中',
      preflight_failed: '连接预检失败',
      pre_detecting: '预检测中',
      pre_detected: '预检测完成',
      running: '能力测试中',
      completed: '已完成',
      partial_failed: '部分失败',
      failed: '失败',
      suspended: '已暂停',
      cancelled: '已取消',
    };
    pushConsoleSeparator('阶段切换');
    pushConsole(`[阶段] ${phaseLabel[status] || status}`, 'meta');
  }

  if (status === 'pre_detecting') {
    const currentLayer = pre?.current_layer || '';
    const currentProbe = pre?.current_probe || '';
    const probeDetail = pre?.probe_detail || {};
    const evidence = pre?.evidence || [];

    const layerNames = {
      'Layer0/HTTP': 'HTTP指纹分析',
      'Layer1/SelfReport': '模型自报身份',
      'Layer2/Identity': '身份探针矩阵',
      'Layer3/Knowledge': '知识截止日期',
      'Layer4/Bias': '行为偏好指纹',
      'Layer5/Tokenizer': 'Tokenizer指纹',
      'Layer6/Extraction': '主动提取探测',
      'Layer6b/MultiTurn': '多轮上下文',
      'Layer7/Logprobs': 'Logprobs指纹',
      'Layer8/SemanticFP': '语义指纹',
      'Layer9/AdvExtractionV2': '高级提取v2',
      'Layer10/Differential': '差分一致性',
      'Layer11/ToolCapability': '工具能力',
      'Layer12/MultiTurn': '多轮上下文',
      'Layer13/Adversarial': '对抗分析',
      'Layer14/Multilingual': '多语言攻击',
      'Layer15/ASCIIArt': 'ASCII视觉注入',
      'Layer16/IndirectInject': '间接提示注入',
      'Layer17/IdentityExposure': '身份暴露分析',
      'Layer18/TimingSideChannel': '时序侧信道',
      'Layer19/TokenDistribution': 'Token分布分析',
    };

    // Detect parallel group progress markers: "parallel_L3_L4_L5(2/3)"
    let layerDisplay;
    const parallelMatch = currentLayer.match(/^parallel_([^(]+)\((\d+)\/(\d+)\)$/);
    if (parallelMatch) {
      const done = parseInt(parallelMatch[2]);
      const total = parseInt(parallelMatch[3]);
      const groupKey = parallelMatch[1]; // e.g. "L3_L4_L5" or "L6b_L16"
      const groupLabel = groupKey === 'L3_L4_L5'
        ? '知识+偏好+Tokenizer 并行检测'
        : groupKey === 'L6b_L16'
        ? '深度多维探测 并行执行'
        : `并行组 ${groupKey}`;
      layerDisplay = `⚡ ${groupLabel} (${done}/${total} 已完成)`;
    } else {
      layerDisplay = layerNames[currentLayer] || currentLayer || '初始化';
    }

    // Show detailed probe info
    if (currentProbe && prev.lastProbe !== currentProbe) {
      const probeNames = {
        'identity_probe_0': '身份探针 #1',
        'identity_probe_1': '身份探针 #2',
        'identity_probe_2': '身份探针 #3',
        'identity_probe_3': '身份探针 #4',
        'identity_probe_4': '身份探针 #5',
        'tokenizer_count_probe': 'Tokenizer计数探针',
        'tokenizer_matched': 'Tokenizer匹配',
        'training_cutoff_probe': '训练截止日期探针',
        'training_cutoff_response': '截止日期响应',
        'format_probe': '格式指纹探针',
        'format_response': '格式响应分析',
      };
      const probeDisplay = probeNames[currentProbe] || currentProbe;

      if (probeDetail.prompt) {
        pushConsole(`[预检测][${layerDisplay}] ${probeDisplay}`, 'meta');
        pushConsole(`  提示: "${probeDetail.prompt}..."`, 'normal');
      } else if (probeDetail.response) {
        pushConsole(`  响应: "${probeDetail.response.substring(0, 60)}${probeDetail.response.length > 60 ? '...' : ''}"`, 'normal');
      } else if (probeDetail.count && probeDetail.tokenizer) {
        pushConsole(`  匹配: ${probeDetail.tokenizer} (count=${probeDetail.count})`, 'normal');
      } else {
        pushConsole(`[预检测][${layerDisplay}] ${probeDisplay} ...`, 'meta');
      }

      // Show collected evidence
      if (evidence && evidence.length > 0) {
        const newEvidence = evidence.slice(prev.lastEvidenceCount || 0);
        newEvidence.forEach(ev => {
          pushConsole(`  证据: ${ev.substring(0, 80)}${ev.length > 80 ? '...' : ''}`, 'normal');
        });
        prev.lastEvidenceCount = evidence.length;
      }

      prev.lastProbe = currentProbe;
    } else if (prev.lastLayer !== currentLayer) {
      if (parallelMatch) {
        const done2 = parseInt(parallelMatch[2]);
        const total2 = parseInt(parallelMatch[3]);
        if (done2 === 0) {
          pushConsole(`[预检测] ⚡ 启动并行探测: ${layerDisplay}`, 'meta');
        } else {
          pushConsole(`[预检测] ⚡ 并行进度: ${done2}/${total2} 层已完成`, 'meta');
          // Show new evidence from parallel layers
          if (evidence && evidence.length > (prev.lastEvidenceCount || 0)) {
            const newEvs = evidence.slice(prev.lastEvidenceCount || 0);
            newEvs.forEach(ev => {
              pushConsole(`  证据: ${ev.substring(0, 80)}${ev.length > 80 ? '...' : ''}`, 'normal');
            });
            prev.lastEvidenceCount = evidence.length;
          }
        }
      } else {
        pushConsole(`[预检测] 进入层: ${layerDisplay}`, 'meta');
      }
      prev.lastLayer = currentLayer;
      prev.lastEvidenceCount = (evidence || []).length;
    }
  }

  if (pre?.identified_as && !prev.predetectLogged) {
    pushConsoleSeparator('预检测结果');
    
    const layers = pre.layer_results || [];
    for (const l of layers) {
      pushConsole(`[预检测] 执行层级 Layer ${l.layer_id || '?'}`, 'meta');
      for (const ev of (l.evidence || [])) {
        pushConsole(`[证据] ${ev}`);
      }
    }
    
    if (pre.routing_info && pre.routing_info.returned_model) {
      pushConsole(`[预检测] 路由探测实际模型: ${pre.routing_info.returned_model}`, pre.routing_info.is_routed ? 'del' : 'meta');
    }

    pushConsole(`[预检测] 最终识别: ${pre.identified_as}`);
    pushConsole(`[预检测] 置信度: ${(pre.confidence * 100).toFixed(0)}%`, 'meta');
    _lastProgressSnapshot.predetectLogged = true;
  }

  const seen = prev.responseSeen || {};
  const ordered = [...responses].sort((a, b) => {
    if (a.case_id === b.case_id) return (a.sample_index || 0) - (b.sample_index || 0);
    return String(a.case_id || '').localeCompare(String(b.case_id || ''));
  });

  for (const r of ordered) {
    const key = `${r.case_id}#${r.sample_index}`;
    if (seen[key]) continue;

    const prompt = r.request_preview?.user_prompt || '';
    const temp = r.request_preview?.temperature;
    const maxT = r.request_preview?.max_tokens;
    const ans = r.response_text || '';

    pushConsoleSeparator(`CASE ${r.case_id || '-'} · sample=${r.sample_index ?? '-'}`);

    if (prompt) {
      pushConsole(`[提示词] ${prompt}`);
    } else {
      pushConsole(`[提示词] (无预览内容)`, 'meta');
    }

    if (temp != null || maxT != null) {
      pushConsole(`[参数] temperature=${temp ?? '-'} · max_tokens=${maxT ?? '-'}`, 'meta');
    }

    if (ans) {
      pushConsole(`[回答] ${stripMd(ans)}`);
    }

    if (r.error_type) {
      pushConsole(`[结果] 错误: ${r.error_type}`, 'del');
    } else if (r.judge_passed === true) {
      pushConsole('[结果] 判定: 通过');
    } else if (r.judge_passed === false) {
      pushConsole('[结果] 判定: 未通过');
    }

    seen[key] = true;
  }

  if (currCompleted < (prev.completed || 0)) {
    pushConsoleDelete(`[回滚] 进度 ${(prev.completed || 0)} -> ${currCompleted}`);
  }

  if (['completed','partial_failed','failed'].includes(status) && status !== prev.phase) {
    pushConsoleSeparator('结束');
    pushConsole('[总结] 流程结束，正在汇总分析报告。');
  }

  _lastProgressSnapshot = {
    completed: currCompleted,
    total: currTotal,
    phase: status,
    predetectLogged: _lastProgressSnapshot.predetectLogged,
    responseSeen: seen,
    lastLayer: _lastProgressSnapshot.lastLayer,
    lastProbe: _lastProgressSnapshot.lastProbe,
    lastEvidenceCount: _lastProgressSnapshot.lastEvidenceCount,
  };
}

function pushConsole(line, type = 'normal', tag = '') {
  _consoleLines.push({ text: line, type, tag });
  if (_consoleLines.length > 3000) _consoleLines.shift();
  updateLogPanel();
}

function pushConsoleSeparator(title = '') {
  const bar = '─'.repeat(20);
  const text = title ? `${bar} ${title} ${bar}` : `${bar}${bar}`;
  pushConsole(text, 'sep');
}

function pushConsoleDelete(line) {
  pushConsole(line, 'del');
}

function pushConsoleTyping(line) {
  if (_typingTimer) return;
  _typingTimer = setTimeout(() => {
    pushConsole(line, 'typing');
    updateLogPanel();
    _typingTimer = null;
  }, 260);
}

function renderConsoleLines() {
  if (!_consoleLines.length) {
    return '<div class="console-line">等待任务开始...</div>';
  }
  const visible = _consoleLines.filter(matchLogFilter);
  if (!visible.length) {
    return '<div class="console-line meta">当前过滤条件下暂无日志</div>';
  }
  return visible.map(l => `<div class="console-line ${l.type === 'typing' ? 'typing' : ''} ${l.type === 'del' ? 'del' : ''} ${l.type === 'sep' ? 'sep' : ''} ${l.type === 'meta' ? 'meta' : ''}">${escHtml(l.text)}</div>`).join('');
}

// Human-readable layer name lookup
const _LAYER_NAMES = {
  'Layer0/HTTP':              'L0 · HTTP 头指纹',
  'Layer1/SelfReport':        'L1 · 自我报告身份',
  'Layer2/Identity':          'L2 · 身份探针矩阵',
  'Layer3/Knowledge':         'L3 · 知识截止验证',
  'Layer4/Bias':              'L4 · 行为偏好分析',
  'Layer5/Tokenizer':         'L5 · Tokenizer 指纹',
  'Layer6/Extraction':        'L6 · 主动提取攻击',
  'Layer6/Extraction(early-skip)': 'L6 · 主动提取（快速路径）',
  'Layer6/Extraction(fast-path)':  'L6 · 主动提取（快速路径）',
  'Layer6b/MultiTurn':        'L6b · 多轮对话探测',
  'Layer7/Logprobs':          'L7 · Logprobs 指纹',
  'Layer8/SemanticFP':        'L8 · 语义指纹匹配',
  'Layer9/AdvExtractionV2':   'L9 · 高级提取攻击 v2',
  'Layer10/Differential':     'L10 · 差分一致性测试',
  'Layer11/ToolCapability':   'L11 · 工具能力探测',
  'Layer12/MultiTurn':        'L12 · 多轮上下文溢出',
  'Layer13/Adversarial':      'L13 · 对抗性响应分析',
  'Layer14/Multilingual':     'L14 · 多语言翻译攻击',
  'Layer15/ASCIIArt':         'L15 · ASCII 艺术视觉注入',
  'Layer16/IndirectInject':   'L16 · 间接提示注入',
  'Layer17/IdentityExposure': 'L17 · 真实模型身份暴露',
  'Layer18/TimingSideChannel':'L18 · 响应时序侧信道',
  'Layer19/TokenDistribution':'L19 · Token 分布侧信道',
};

function renderPredetectCard(pre, runId, status) {
  // Always show if there's any data at all (even failed/unidentified runs)
  if (!pre || (!pre.layer_results?.length && !pre.identified_as && !pre.current_layer)) return '';

  const conf = Math.round((pre.confidence || 0) * 100);
  const layers = pre.layer_results || [];
  const stopLayer = pre.layer_stopped || null;
  const routing = pre.routing_info || {};

  // ── Summary header ──────────────────────────────────────────────────────────
  const confColor = conf >= 85 ? '#16a34a' : conf >= 60 ? '#d97706' : '#dc2626';
  const identText = pre.identified_as
    ? `<span style="font-size:16px;font-weight:700;color:var(--ink1)">${escHtml(pre.identified_as)}</span>`
    : `<span style="font-size:14px;color:var(--ink3);font-style:italic">未识别到模型家族</span>`;

  const confBar = `
    <div style="margin-top:6px">
      <div style="display:flex;justify-content:space-between;font-size:11px;color:var(--ink4);margin-bottom:2px">
        <span>综合置信度</span><span style="font-weight:600;color:${confColor}">${conf}%</span>
      </div>
      <div class="progress-bar" style="height:5px">
        <div class="progress-fill" style="width:${conf}%;background:${confColor}"></div>
      </div>
    </div>`;

  // ── Routing info ─────────────────────────────────────────────────────────────
  let routingHtml = '';
  if (routing.is_routed) {
    routingHtml = `
      <div style="margin-top:10px;padding:8px 12px;background:rgba(220,38,38,.07);border:1px solid rgba(220,38,38,.2);border-radius:6px;font-size:12px">
        <div style="font-weight:600;color:var(--red);margin-bottom:4px">⚠ 路由检测 — 实际模型与声称不符</div>
        <div style="display:flex;gap:20px;flex-wrap:wrap;color:var(--ink3)">
          <div>声称：<code style="color:var(--ink2)">${escHtml(routing.claimed_model || '?')}</code></div>
          <div>实际：<code style="color:var(--red);font-weight:600">${escHtml(routing.returned_model || '?')}</code></div>
          ${routing.probe_latency_ms ? `<div>延迟：${routing.probe_latency_ms}ms</div>` : ''}
        </div>
      </div>`;
  } else if (routing.returned_model) {
    routingHtml = `
      <div style="margin-top:8px;font-size:12px;color:var(--ink3)">
        实际返回模型：<code style="color:var(--ink2)">${escHtml(routing.returned_model)}</code>
        ${routing.probe_latency_ms ? ` · 延迟 ${routing.probe_latency_ms}ms` : ''}
      </div>`;
  }

  // ── Per-layer breakdown ──────────────────────────────────────────────────────
  let layersHtml = '';
  if (layers.length > 0) {
    const layerRows = layers.map(l => {
      const lConf = Math.round((l.confidence || 0) * 100);
      const lColor = lConf >= 85 ? '#16a34a' : lConf >= 50 ? '#d97706' : '#6b7280';
      const lName = _LAYER_NAMES[l.layer] || escHtml(l.layer || '');
      const evidence = l.evidence || [];
      const evHtml = evidence.length
        ? `<ul style="margin:3px 0 0 12px;padding:0;list-style:disc;font-size:11px;color:var(--ink3)">
            ${evidence.map(e => `<li style="margin-bottom:1px">${escHtml(String(e).length > 100 ? String(e).slice(0,100)+'…' : e)}</li>`).join('')}
           </ul>`
        : '';
      const tokText = l.tokens_used > 0 ? `<span style="color:var(--ink4);font-size:10px">${l.tokens_used}tok</span>` : '';
      const confBadge = lConf > 0
        ? `<span style="font-size:10px;font-weight:600;color:${lColor};min-width:32px;text-align:right">${lConf}%</span>`
        : `<span style="font-size:10px;color:var(--ink4);min-width:32px;text-align:right">-</span>`;
      const idText = l.identified_as
        ? `<span style="font-size:10px;background:rgba(59,130,246,.1);color:#2563eb;padding:1px 5px;border-radius:3px;margin-left:4px">${escHtml(l.identified_as)}</span>`
        : '';
      return `
        <div style="padding:5px 0;border-bottom:1px solid var(--rule)">
          <div style="display:flex;align-items:center;gap:6px">
            <span style="font-size:12px;font-weight:600;color:var(--ink2);flex:1">${lName}${idText}</span>
            ${tokText}
            ${confBadge}
          </div>
          ${evHtml}
        </div>`;
    }).join('');

    // Note about skipped layers
    const modeNote = (status !== 'pre_detecting') ? (() => {
      const deepOnlyCount = 20 - 7; // L7-L19 = 13 layers only in deep mode
      const ranCount = layers.length;
      if (ranCount < 7) {
        const stopNote = stopLayer && stopLayer !== 'null' && stopLayer !== 'None'
          ? `（置信度阈值已在 ${escHtml(String(stopLayer))} 达到，提前停止）`
          : '（置信度阈值提前达到）';
        return `<div style="font-size:11px;color:var(--ink4);margin-top:6px;padding:4px 8px;background:var(--bg2);border-radius:4px">
          ⚡ 共运行 ${ranCount} 层 ${stopNote}。L7-L19 仅在 <strong>Deep 模式</strong>下运行。
        </div>`;
      }
      return `<div style="font-size:11px;color:var(--ink4);margin-top:6px;padding:4px 8px;background:var(--bg2);border-radius:4px">
        ⚡ 共运行 ${ranCount} 层。L7-L19（深度对抗层）仅在 <strong>Deep 模式</strong>下启用。
      </div>`;
    })() : '';

    layersHtml = `
      <div style="margin-top:10px">
        <div style="font-size:11px;font-weight:600;color:var(--ink4);text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px">层级明细</div>
        ${layerRows}
        ${modeNote}
      </div>`;
  }

  // ── Action buttons ────────────────────────────────────────────────────────────
  let actionHtml = '';
  if (status === 'pre_detected' && pre.should_proceed_to_testing === false) {
    actionHtml = `
      <div style="margin-top:12px;display:flex;gap:10px;align-items:center;flex-wrap:wrap">
        <button class="btn primary" style="padding:8px 16px;font-size:13px" data-action="continueFullTest" data-run-id="${escAttr(runId)}">继续完整测试</button>
        <button class="btn" style="padding:8px 16px;font-size:13px" data-action="skipTesting" data-run-id="${escAttr(runId)}">跳过测试，直接出报告</button>
        <span style="font-size:11px;color:var(--ink4)">完整测试可获得详细评分和相似度对比</span>
      </div>`;
  } else if (pre.should_proceed_to_testing === false && status !== 'pre_detected') {
    actionHtml = '<div style="margin-top:8px;font-size:11px;color:var(--green,#16a34a);font-weight:600">✓ 置信度充足，已跳过完整能力测试</div>';
  }

  return `
    <div class="predetect-card">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:6px">
        <div>
          <div class="card-label" style="color:var(--blue);opacity:.7;margin-bottom:4px">预检测指纹分析 · ${layers.length} 层运行</div>
          ${identText}
        </div>
        <div style="font-size:11px;color:var(--ink4);text-align:right">
          消耗 ${pre.total_tokens_used || 0} tokens
        </div>
      </div>
      ${confBar}
      ${routingHtml}
      ${layersHtml}
      ${actionHtml}
    </div>`;
}

function renderPreflightCard(pf) {
  if (!pf) return '';
  const steps = pf.steps || [];
  const passed = pf.passed;
  const totalMs = pf.total_duration_ms != null ? Math.round(pf.total_duration_ms) : null;

  const stepNames = { A1:'输入校验', A2:'网络连通', A3:'鉴权探测', A4:'响应格式', A5:'能力探测' };
  const stepsHtml = steps.map(s => {
    const icon = s.passed ? '✅' : (s.notes && s.notes.startsWith('skipped') ? '⏭' : '❌');
    const dur = s.duration_ms != null && s.duration_ms > 0 ? `<span style="color:var(--ink4);font-size:11px">${Math.round(s.duration_ms)}ms</span>` : '';
    const label = stepNames[s.step] || s.name || s.step;
    let errHtml = '';
    if (s.error) {
      const e = s.error;
      errHtml = `<div style="margin-top:3px;font-size:11px;color:var(--red)">${escHtml(e.user_message_zh || e.message || e.code || '')}</div>`;
    } else if (s.notes && !s.notes.startsWith('skipped')) {
      errHtml = `<div style="margin-top:2px;font-size:11px;color:var(--ink4)">${escHtml(s.notes)}</div>`;
    }
    return `
      <div style="display:flex;align-items:flex-start;gap:8px;padding:6px 0;border-bottom:1px solid var(--rule)">
        <span style="font-size:14px;line-height:1.4">${icon}</span>
        <div style="flex:1;min-width:0">
          <div style="display:flex;justify-content:space-between;align-items:center">
            <span style="font-size:12px;font-weight:600;color:var(--ink2)">${escHtml(s.step)} · ${escHtml(label)}</span>
            ${dur}
          </div>
          ${errHtml}
        </div>
      </div>`;
  }).join('');

  const headerColor = passed ? 'var(--green,#16a34a)' : 'var(--red,#dc2626)';
  const headerText = passed ? '✅ 连接预检通过' : '❌ 连接预检失败';
  const totalText = totalMs != null ? ` · ${totalMs}ms` : '';

  return `
    <div class="card" id="preflight-result-card" style="border-left:3px solid ${headerColor}">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
        <div style="font-size:13px;font-weight:700;color:${headerColor}">${headerText}${escHtml(totalText)}</div>
        <div style="font-size:11px;color:var(--ink4)">${pf.checked_at ? fmtTime(pf.checked_at) : ''}</div>
      </div>
      ${stepsHtml}
    </div>`;
}

async function loadReport(runId) {
  const resp = await safeFetch(API + '/api/v1/runs/' + runId + '/report');
  if (!resp) return;
  const data = await resp.json().catch(() => null);
  if (!data) return;
  const el = document.getElementById('report-section');
  if (el) {
    el.innerHTML = renderReport(data);
    // Initialize ECharts radar chart after DOM insertion
    const container = document.getElementById('radar-chart-container');
    if (container) {
      const sc = data.scorecard || {};
      renderRadarChart('radar-chart-container', { run_id: runId, scorecard: sc }, 'percent');
    }
    // v14 Phase 8: render new v14 cards
    renderV14Cards(data, runId);
  }
  // v14 Phase 3: fetch and render identity exposure
  fetch('/api/v14/runs/' + runId + '/identity-exposure')
    .then(r => r.json())
    .then(ieData => renderIdentityExposure(ieData))
    .catch(() => {});
}

function renderNarrative(narrative) {
  if (!narrative) return '';
  const exec = narrative.executive_summary || '';
  const proc = narrative.detection_process || '';
  const dim = narrative.dimension_analysis || '';
  const sim = narrative.similarity_narrative || '';
  const risk = narrative.risk_narrative || '';
  const recs = narrative.recommendations || [];
  const conf = narrative.confidence_statement || '';

  // Simple markdown-like rendering: **bold** and - lists
  const md = s => escHtml(s)
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n- /g, '\n&bull; ')
    .replace(/\n/g, '<br>');

  return `
  <div class="card" style="border-left:4px solid var(--primary)">
    <h2>📋 检测总结</h2>
    <div style="font-size:14px;line-height:1.8;color:var(--ink2)">
      ${exec ? `<div style="margin-bottom:12px">${md(exec)}</div>` : ''}
      ${proc ? `<div style="margin-bottom:8px"><strong>检测过程：</strong>${md(proc)}</div>` : ''}
      ${dim ? `<div style="margin-bottom:8px"><strong>维度分析：</strong>${md(dim)}</div>` : ''}
      ${sim ? `<div style="margin-bottom:8px;background:var(--bg2);padding:8px;border-radius:6px">${md(sim)}</div>` : ''}
      ${risk ? `<div style="margin-bottom:8px;background:var(--bg2);padding:8px;border-radius:6px">${md(risk)}</div>` : ''}
      ${recs.length ? `
        <div style="margin-bottom:8px"><strong>建议：</strong></div>
        <ul style="margin:4px 0;padding-left:20px">${recs.map(r => `<li>${md(r)}</li>`).join('')}</ul>
      ` : ''}
      ${conf ? `<div style="font-size:12px;color:var(--ink4);font-style:italic;margin-top:6px">${md(conf)}</div>` : ''}
    </div>
  </div>`;
}

function renderReport(r) {
  if (!r) return '';
  const narrative = r.narrative || null;
  const sim    = r.similarity || [];
  const risk   = r.risk || {};
  const scorecard = r.scorecard || null;
  const verdict = r.verdict || null;
  const theta = r.theta || null;
  const pairwise = r.pairwise_rank || null;
  const cases  = r.case_results || [];
  const dimensions = r.dimensions || {};
  const tagBreakdown = r.tag_breakdown || {};
  const failureAttribution = r.failure_attribution || {};
  const abSignificance = r.ab_significance || (r.details && r.details.ab_significance) || [];

  let html = '';

  if (narrative) {
    html += renderNarrative(narrative);
  }

  // Incomplete data warnings
  const warnings = r.warnings || [];
  if (warnings.length > 0) {
    html += `
      <div class="card" style="border-left:4px solid #d97706;background:var(--surface2)">
        <div style="font-weight:600;color:#d97706;margin-bottom:6px">&#9888; 数据完整性提示</div>
        <ul style="margin:0;padding-left:18px;font-size:12px;color:var(--ink3)">
          ${warnings.map(w => `<li>${escHtml(w.warning)}</li>`).join('')}
        </ul>
      </div>`;
  }

  if (theta) {
    const dims = theta.dimensions || [];
    const dimRows = dims.map(d => {
      const t = Number(d.theta || 0).toFixed(2);
      const lo = Number(d.ci_low || 0).toFixed(2);
      const hi = Number(d.ci_high || 0).toFixed(2);
      const pctVal = d.percentile != null ? Math.min(99.9, Number(d.percentile)) : null;
      const pctText = pctVal != null ? `${pctVal.toFixed(1)}%` : '-';
      // Bar width matches the displayed percentile; falls back to theta mapping if no percentile
      const width = pctVal != null
        ? Math.max(2, Math.min(100, pctVal))
        : Math.max(2, Math.min(100, ((Number(d.theta || 0) + 4) / 8) * 100));
      return `<div style="margin-bottom:8px">
        <div style="display:flex;justify-content:space-between;font-size:12px;color:var(--ink3)">
          <span>${escHtml(zhLabel(d.dimension))}</span>
          <span>${pctText !== '-' ? `超越 ${pctText}` : `得分为 ${t}`}</span>
        </div>
        <div class="progress-bar" style="height:7px"><div class="progress-fill" style="width:${width}%"></div></div>
      </div>`;
    }).join('');

    html += `
      <div class="card">
        <h2>MDIRT 标准分 (均值500, 标准差100)</h2>
        <div class="score-grid" style="grid-template-columns:repeat(2,1fr)">
          ${scoreCard(theta.global_theta, '综合评估水平 (MDIRT)')}
          ${scoreCard(theta.global_percentile == null ? '-' : `${Number(theta.global_percentile).toFixed(1)}%`, '击败同期主流模型')}
        </div>
        <div class="card" style="margin-top:8px;padding:10px">${dimRows || '<div style="font-size:12px;color:var(--ink4)">暂无维度数据</div>'}</div>
      </div>`;
  }

  if (pairwise) {
    html += `
      <div class="card">
        <h2>Pairwise 对比</h2>
        <div style="font-size:13px;color:var(--ink3);line-height:1.7">
          <div>Δθ: <strong>${Number(pairwise.delta_theta||0).toFixed(3)}</strong></div>
          <div>胜率: <strong>${Number((pairwise.win_prob||0)*100).toFixed(1)}%</strong></div>
          <div>基准 θ: <strong>${Number(pairwise.baseline_theta||0).toFixed(3)}</strong></div>
        </div>
      </div>`;
  }

  if (scorecard) {
    const b = scorecard.breakdown || {};
    html += `
      <div class="card">
        <h2>核心能力评分</h2>
        <div class="score-grid">
          ${scoreCard(scorecard.total_score, '总分')}
          ${scoreCard(scorecard.capability_score, '能力分')}
          ${scoreCard(scorecard.authenticity_score, '真实性')}
          ${scoreCard(scorecard.performance_score, '性能分')}
        </div>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:8px">
          <div class="score-card" style="text-align:left">
            <div class="card-label">能力细分</div>
            <div style="font-size:12px;color:var(--ink3)">推理能力: ${fmtScore(b.reasoning)}</div>
            <div style="font-size:12px;color:var(--ink3)">指令遵循: ${fmtScore(b.instruction)}</div>
            <div style="font-size:12px;color:var(--ink3)">编程能力: ${fmtScore(b.coding)}</div>
            <div style="font-size:12px;color:var(--ink3)">安全性: ${fmtScore(b.safety)}</div>
            <div style="font-size:12px;color:var(--ink3)">协议兼容: ${fmtScore(b.protocol)}</div>
            <div style="font-size:12px;color:var(--ink3)">知识: ${fmtScore(b.knowledge_score)}</div>
            <div style="font-size:12px;color:var(--ink3)">工具使用: ${fmtScore(b.tool_use_score)}</div>
          </div>
          <div class="score-card" style="text-align:left">
            <div class="card-label">真实性信号</div>
            <div style="font-size:12px;color:var(--ink3)">一致性: ${fmtScore(b.consistency)}</div>
            <div style="font-size:12px;color:var(--ink3)">行为不变量: ${fmtScore(b.behavioral_invariant)}</div>
            <div style="font-size:12px;color:var(--ink3)">提取抵抗: ${fmtScore(b.extraction_resistance)}</div>
            <div style="font-size:12px;color:var(--ink3)">指纹匹配: ${fmtScore(b.fingerprint_match)}</div>
          </div>
          <div class="score-card" style="text-align:left">
            <div class="card-label">性能信号</div>
            <div style="font-size:12px;color:var(--ink3)">响应速度: ${fmtScore(b.speed)}</div>
            <div style="font-size:12px;color:var(--ink3)">稳定性: ${fmtScore(b.stability)}</div>
            <div style="font-size:12px;color:var(--ink3)">成本效率: ${fmtScore(b.cost_efficiency)}</div>
            <div style="font-size:12px;color:var(--ink3)">TTFT合理性: ${fmtScore(b.ttft_plausibility)}</div>
          </div>
        </div>
      </div>`;
  }

  if (verdict && verdict.level) {
    const cls = verdict.level === 'trusted' ? 'low' : verdict.level === 'suspicious' ? 'medium' : 'high';
    html += `
      <div class="risk-block ${cls}">
        <div class="card-label">真实性结论</div>
        <div class="risk-level ${cls}">${escHtml(verdict.label || verdict.level)}</div>
        <ul class="evidence-list">
          ${(verdict.reasons||[]).map(r2 => `<li>${escHtml(r2)}</li>`).join('')}
        </ul>`;
    if (verdict.confidence_real !== undefined) {
      const cr = Number(verdict.confidence_real || 0);
      const crColor = cr >= 80 ? '#16a34a' : cr >= 60 ? '#d97706' : '#dc2626';
      const sd = verdict.signal_details || {};
      const SIGNAL_LABELS = {
        behavioral_similarity: '行为相似度',
        capability_score:      '能力分',
        timing_fingerprint:    '时序指纹',
        consistency_score:     '一致性',
        protocol_compliance:   '协议合规',
        predetect_identity:    '身份自报',
      };
      const SIGNAL_WEIGHTS = {
        behavioral_similarity: 30,
        capability_score:      20,
        timing_fingerprint:    20,
        consistency_score:     15,
        protocol_compliance:   10,
        predetect_identity:     5,
      };
      html += `
        <div style="margin-top:12px;padding:12px;background:var(--bg2);border-radius:6px">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">
            <span style="font-size:13px;color:var(--ink3)">综合置信度</span>
            <span style="font-size:22px;font-weight:700;color:${crColor}">${cr.toFixed(1)}</span>
            <span style="font-size:12px;color:var(--ink4)">/ 100</span>
            <div style="flex:1;height:6px;background:var(--rule);border-radius:3px;overflow:hidden">
              <div style="width:${cr}%;height:100%;background:${crColor};border-radius:3px"></div>
            </div>
          </div>
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:6px">
            ${Object.entries(SIGNAL_LABELS).map(([k,label]) => {
              const val = Number(sd[k] || 0);
              const w = SIGNAL_WEIGHTS[k];
              const barW = Math.min(100, val);
              const c = val >= 70 ? '#16a34a' : val >= 45 ? '#d97706' : '#dc2626';
              return `<div style="background:var(--bg);padding:6px 8px;border-radius:4px;font-size:11px">
                <div style="color:var(--ink3);margin-bottom:3px">${label} <span style="color:var(--ink4)">(×${w}%)</span></div>
                <div style="display:flex;align-items:center;gap:6px">
                  <div style="flex:1;height:4px;background:var(--rule);border-radius:2px">
                    <div style="width:${barW}%;height:100%;background:${c};border-radius:2px"></div>
                  </div>
                  <span style="color:${c};font-weight:600;min-width:28px;text-align:right">${val.toFixed(0)}</span>
                </div>
              </div>`;
            }).join('')}
          </div>
        </div>`;
    }
    html += `<div class="risk-disclaimer">${escHtml(verdict.disclaimer||'')}</div>
      </div>`;
  }


  if (Object.keys(dimensions).length || Object.keys(tagBreakdown).length || Object.keys(failureAttribution).length) {
    html += `
      <div class="card">
        <h2>多维诊断看板</h2>
        <div class="metric-grid">
          ${renderMetricCard('维度分布', dimensions, true)}
          ${renderMetricCard('标签分布', tagBreakdown, true)}
          ${renderMetricCard('失败归因', failureAttribution, false)}
        </div>
      </div>`;
  }

  // Similarity (v6 enhanced with data source and confidence visualization)
  if (sim.length) {
    // v6: Add confidence level visualization
    const getConfidenceLevel = (score) => {
      if (score >= 0.75) return { level: "高置信", color: "#16a34a", desc: "与基准模型高度相似" };
      if (score >= 0.55) return { level: "中置信", color: "#f59e0b", desc: "与基准模型中度相似" };
      return { level: "低置信", color: "#dc2626", desc: "与基准模型差异较大" };
    };

    html += `
      <div class="card">
        <h2>相似度排名</h2>
        <div style="font-size:12px;color:var(--ink4);margin-bottom:12px">
          📊 数据来源：基于特征向量余弦相似度计算，95%置信区间通过bootstrap采样获得
        </div>
        ${sim.map((s, idx) => {
          const pct = Math.min(100, Math.round(s.score * 100));
          const hasCi = s.ci_95_low != null && s.ci_95_high != null;
          const ciLo = hasCi ? Math.min(100, Math.round(s.ci_95_low * 100)) : null;
          const ciHi = hasCi ? Math.min(100, Math.round(s.ci_95_high * 100)) : null;
          const ciText = hasCi
            ? `[${fmtScore(ciLo)}–${fmtScore(ciHi)}%]`
            : `[样本不足，CI 不可用]`;
          const barColor = s.score > 0.75 ? '#E24B4A' : s.score > 0.55 ? '#EF9F27' : '#1D9E75';
          const ciBarLeft = hasCi ? ciLo : 0;
          const ciBarWidth = hasCi ? Math.max(1, ciHi - ciLo) : pct;
          
          // v6: Confidence level visualization
          const confidence = getConfidenceLevel(s.score);
          
          return `
            <div class="sim-row" style="position:relative">
              <div style="display:flex;align-items:center;gap:8px">
                <div style="font-size:11px;color:var(--ink4);min-width:20px">#${idx + 1}</div>
                <div class="sim-name">${escHtml(s.benchmark)}</div>
                <div style="font-size:10px;padding:2px 6px;background:${confidence.color}20;color:${confidence.color};border-radius:3px">
                  ${confidence.level}
                </div>
              </div>
              <div class="sim-bar-wrap">
                <div class="sim-bar" style="width:${pct}%;background:${barColor}"></div>
                <div class="sim-ci" style="left:${hasCi ? ciBarLeft : 0}%;width:${ciBarWidth}%;background:${barColor}"></div>
              </div>
              <div class="sim-score">${fmtScore(pct)}%</div>
              <div style="font-size:10px;color:var(--ink4);min-width:80px">${ciText}</div>
            </div>`;
        }).join('')}
        <div style="font-size:11px;color:var(--ink4);margin-top:12px;padding-top:8px;border-top:1px solid var(--border)">
          <div>📈 [ ] 为 95% 置信区间</div>
          <div>🎯 相似度解读：≥75%高度相似，55-74%中度相似，&lt;55%差异较大</div>
          <div>⚠️  置信区间宽度反映测量精度，区间越窄结果越可靠</div>
        </div>
      </div>`;
  } else {
    // v6: Show message when no similarity data
    html += `
      <div class="card">
        <h2>相似度排名</h2>
        <div style="padding:20px;text-align:center;color:var(--ink4)">
          <div style="font-size:24px;margin-bottom:8px">📊</div>
          <div>暂无相似度数据</div>
          <div style="font-size:12px;margin-top:4px">需要基准模型进行对比分析</div>
        </div>
      </div>`;
  }

  const evidenceChain = r.evidence_chain || [];
  if (evidenceChain.length) {
    const SEV_COLOR = { info: '#16a34a', warn: '#d97706', critical: '#dc2626' };
    const PHASE_LABEL = { predetect: '预检测', testing: '行为测试', timing: '时序分析', verdict: '最终判定' };
    html += `
      <div class="card">
        <h2>检测证据链</h2>
        <div style="display:flex;flex-direction:column;gap:4px">
          ${evidenceChain.map(item => {
            const color = SEV_COLOR[item.severity] || '#888';
            const phase = PHASE_LABEL[item.phase] || item.phase;
            const extra = item.pass_rate !== undefined
              ? `<span style="color:var(--ink4);font-size:11px">通过率 ${item.pass_rate}%</span>`
              : item.confidence !== undefined
              ? `<span style="color:var(--ink4);font-size:11px">${item.confidence}%</span>`
              : item.value !== undefined
              ? `<span style="color:var(--ink4);font-size:11px">${item.value}${item.unit==='ms_gap'?' ms间距':''}</span>`
              : '';
            return `
              <div style="display:flex;align-items:center;gap:8px;padding:6px 10px;border-left:3px solid ${color};background:var(--bg2);border-radius:0 4px 4px 0">
                <span style="font-size:10px;color:var(--ink4);min-width:52px;background:var(--bg);padding:1px 5px;border-radius:3px">${escHtml(phase)}</span>
                <span style="font-size:12px;color:var(--ink2);flex:1">${escHtml(item.signal || '')}</span>
                ${extra}
              </div>`;
          }).join('')}
        </div>
      </div>`;
  }

  if (abSignificance.length) {
    html += `
      <div class="card">
        <h2>A/B 统计显著性</h2>
        ${abSignificance.map(renderAbRow).join('')}
      </div>`;
  }

  html += `
    <div class="card">
      <h2>雷达图预览</h2>
      <div style="font-size:12px;color:var(--ink4);margin-bottom:10px">可右上角按钮下载 SVG 原图</div>
      <div style="display:flex;gap:8px;margin-bottom:8px">
        <button class="chip active" onclick="switchRadarScale('percent', this)">百分制</button>
        <button class="chip" onclick="switchRadarScale('stanine', this)">Stanine-9</button>
        <button class="chip" onclick="switchRadarScale('theta', this)">θ 逻辑分</button>
      </div>
      <div id="radar-chart-container" style="width:100%;height:400px"
           data-run-id="${escHtml(r.run_id)}"
           data-scorecard="${escHtml(JSON.stringify(r.scorecard || {}))}"></div>
    </div>`;

  // Case results
  if (cases.length) {
    html += '<div class="card"><h2>题目详情</h2>';
    html += cases.map(c => renderCaseItem(c)).join('');
    html += '</div>';
  }

  // Extraction Audit
  const extAudit = r.extraction_audit || null;
  if (extAudit) {
    html += renderExtractionAudit(extAudit, r.proxy_latency_analysis);
  }

  return html;
}

function scoreCard(val, label) {
  const v = val != null
    ? (typeof val === 'number' ? fmtScore(val) : val)
    : '–';
  let colorClass = 'score-neutral';
  if (typeof val === 'number') {
    // scores are on a 0–10000 scale (backend multiplies by 100)
    if (val >= 8000) colorClass = 'score-excellent';
    else if (val >= 6000) colorClass = 'score-good';
    else if (val >= 4000) colorClass = 'score-warning';
    else colorClass = 'score-danger';
  }
  return `<div class="score-card ${colorClass}"><div class="score">${v}</div><div class="score-label">${label}</div></div>`;
}

const _zhLabels = {
  // 维度名称
  instruction: '指令理解', protocol: '协议兼容', performance: '性能表现',
  reasoning: '推理能力', coding: '编程能力', consistency: '一致性',
  system: '系统提示', param: '参数控制', style: '风格特征', refusal: '安全控制',
  safety: '安全控制', speed: '响应速度', stability: '稳定性', cost_efficiency: '成本效率',
  // 补充缺失的维度名称
  authenticity: '真实性', algorithm: '算法', adversarial: '对抗性',
  adversarial_re: '对抗正则', adversarial_reasoning: '对抗推理',
  antispoof: '反欺骗', complexity: '复杂度',
  extraction: '提取能力', fallacy: '谬误检测', fingerprint: '指纹特征',
  hallucination: '幻觉检测', instruction_hierarchy: '指令层级', knowledge: '知识覆盖',
  logic: '逻辑推理', precise: '精确性', precision: '精确度',
  privacy: '隐私保护', robustness: '鲁棒性', safety_multilingual: '多语言安全',
  tool_use: '工具使用', unicode: 'Unicode处理', value: '价值评估', verbose: '冗长检测',
  // 失败归因
  format_violation: '格式违规', reasoning_failure: '推理失败',
  error_response: '错误响应', safety_violation: '安全违规', unknown: '未知原因',
  timeout: '连接超时', connection_error: '连接错误',
  // 标签（常见）
  exact_match: '精确匹配', json_schema: 'JSON格式', regex_match: '正则匹配',
  line_count: '行数控制', temperature: '温度参数', markdown: 'Markdown',
  format_strict: '严格格式', instruction_following: '指令遵循', deterministic: '确定性',
  json_output: 'JSON 输出', schema_compliance: 'Schema 符合度', math_reasoning: '数学推理',
  constraint_reasoning: '约束推理', token_control: 'Token 控制', char_count: '字符统计',
  forbidden_chars: '禁用词检查', python: 'Python', code_execution: '代码执行',
  function_correctness: '函数正确性',
  // extraction dimensions
  identity_stress: '身份压力', prompt_extraction: '提示词提取',
  behavioral_fingerprint: '行为指纹', extraction: '提取审计',
};

function renderExtractionAudit(audit, proxyAnalysis) {
  const severityColor = {
    CRITICAL: 'var(--red)', HIGH: 'var(--amber)', MEDIUM: '#996600', NONE: 'var(--green)'
  };
  const sev = audit.overall_severity || 'NONE';
  const sevClr = severityColor[sev] || 'var(--ink3)';
  const sevCls = sev === 'CRITICAL' || sev === 'HIGH' ? 'high' : sev === 'MEDIUM' ? 'medium' : 'low';

  let html = `
    <div class="card" style="border-left:4px solid ${sevClr}">
      <h2 style="color:${sevClr}">🔍 提取审计报告 Extraction Audit</h2>
      <div class="risk-block ${sevCls}" style="margin-bottom:16px">
        <div class="risk-level ${sevCls}" style="font-size:18px">
          综合严重等级: ${sev}
        </div>
      </div>
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:16px">
        <div class="score-card">
          <div class="score" style="color:${audit.real_model_exposed?'var(--red)':'var(--green)'}">
            ${audit.real_model_exposed ? '⚠️ 是' : '✓ 否'}
          </div>
          <div class="score-label">真实模型暴露</div>
        </div>
        <div class="score-card">
          <div class="score" style="color:${audit.prompt_leaked?'var(--red)':'var(--green)'}">
            ${audit.prompt_leaked ? '⚠️ 是' : '✓ 否'}
          </div>
          <div class="score-label">提示词泄露</div>
        </div>
        <div class="score-card">
          <div class="score" style="color:${audit.language_bias_detected?'var(--amber)':'var(--green)'}">
            ${audit.language_bias_detected ? '⚠️ 是' : '✓ 否'}
          </div>
          <div class="score-label">语言偏向异常</div>
        </div>
      </div>`;

  if (audit.real_model_names && audit.real_model_names.length) {
    html += `<div style="background:var(--red-bg);border:1px solid var(--red);border-radius:var(--radius);padding:12px;margin-bottom:12px">
      <strong style="color:var(--red)">🚨 检测到真实底层模型:</strong>
      <span style="font-size:16px;font-weight:700;color:var(--red);margin-left:8px">${audit.real_model_names.map(escHtml).join(', ')}</span>
    </div>`;
  }

  if (audit.forbidden_words_leaked && audit.forbidden_words_leaked.length) {
    html += `<div style="background:var(--red-bg);border:1px solid var(--red);border-radius:var(--radius);padding:12px;margin-bottom:12px">
      <strong style="color:var(--red)">禁词列表泄露:</strong>
      <span style="color:var(--red)">${audit.forbidden_words_leaked.map(escHtml).join(', ')}</span>
    </div>`;
  }

  if (audit.file_paths_leaked && audit.file_paths_leaked.length) {
    html += `<div style="background:var(--amber-bg);border:1px solid var(--amber);border-radius:var(--radius);padding:12px;margin-bottom:12px">
      <strong style="color:var(--amber)">文件路径泄露:</strong>
      <code style="font-size:11px">${audit.file_paths_leaked.map(escHtml).join(', ')}</code>
    </div>`;
  }

  if (audit.spec_contradictions && audit.spec_contradictions.length) {
    html += `<div style="background:var(--amber-bg);border:1px solid var(--amber);border-radius:var(--radius);padding:12px;margin-bottom:12px">
      <strong style="color:var(--amber)">规格矛盾:</strong>
      ${audit.spec_contradictions.map(sc =>
        `<div style="font-size:12px;margin-top:4px">
          用例 ${escHtml(sc.case)}: 报告值 ${sc.reported}，期望值 ${sc.expected}，
          实际匹配 <strong>${escHtml(sc.actual_match || '?')}</strong>
        </div>`
      ).join('')}
    </div>`;
  }

  if (audit.evidence_chain && audit.evidence_chain.length) {
    html += `<div style="margin-top:12px">
      <h3>证据链 Evidence Chain</h3>
      <div style="background:var(--bg2);border-radius:var(--radius);padding:12px;max-height:300px;overflow-y:auto">
        ${audit.evidence_chain.map(ev =>
          `<div style="font-family:var(--mono);font-size:11px;padding:3px 0;color:${ev.includes('[CRITICAL]')?'var(--red)':ev.includes('[HIGH]')?'var(--amber)':'var(--ink3)'}">
            ${escHtml(ev)}
          </div>`
        ).join('')}
      </div>
    </div>`;
  }

  // Proxy Latency Analysis
  if (proxyAnalysis && proxyAnalysis.status !== 'insufficient_samples') {
    const pc = proxyAnalysis.proxy_confidence || 0;
    const pcClr = pc > 0.7 ? 'var(--red)' : pc > 0.5 ? 'var(--amber)' : 'var(--green)';
    html += `
      <div style="margin-top:16px;border-top:1px solid var(--rule);padding-top:12px">
        <h3>延迟代理分析 TTFT Analysis</h3>
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:8px">
          ${scoreCard(Math.round(proxyAnalysis.ttft_mean_ms), 'TTFT 均值(ms)')}
          ${scoreCard(proxyAnalysis.ttft_p50_ms, 'P50(ms)')}
          ${scoreCard(proxyAnalysis.ttft_p95_ms, 'P95(ms)')}
          <div class="score-card">
            <div class="score" style="color:${pcClr}">${Math.round(pc * 100)}%</div>
            <div class="score-label">代理置信度</div>
          </div>
        </div>
        ${proxyAnalysis.proxy_signals && proxyAnalysis.proxy_signals.length ?
          `<div style="font-size:12px;color:var(--ink3)">
            ${proxyAnalysis.proxy_signals.map(s => `<div>⚡ ${escHtml(s)}</div>`).join('')}
          </div>` : ''}
      </div>`;
  }

  html += '</div>';
  return html;
}
function zhLabel(key) { return _zhLabels[key] || key; }

function renderMetricCard(title, obj, asPercent) {
  const keys = Object.keys(obj || {});
  if (!keys.length) {
    return `<div class="metric-card"><div class="metric-title">${escHtml(title)}</div><div style="font-size:11px;color:var(--ink4)">暂无数据</div></div>`;
  }
  const rows = keys
    .map(k => ({ k, v: Number(obj[k] || 0) }))
    .sort((a, b) => b.v - a.v)
    .slice(0, 8)
    .map(({k, v}) => {
      const pct = Math.max(0, Math.min(100, Math.round(v * 100)));
      const label = asPercent ? `${pct}%` : `${pct}%`;
      const displayName = zhLabel(k);
      return `<div class="metric-row">
        <div class="metric-name" title="${escHtml(k)}">${escHtml(displayName)}</div>
        <div class="metric-track"><div class="metric-fill" style="width:${pct}%"></div></div>
        <div class="metric-val">${label}</div>
      </div>`;
    }).join('');

  return `<div class="metric-card"><div class="metric-title">${escHtml(title)}</div>${rows}</div>`;
}

function renderAbRow(item) {
  const delta = Number(item.delta || 0);
  const sig = !!item.significant;
  const cls = delta > 0 ? 'ab-up' : delta < 0 ? 'ab-down' : 'ab-neutral';
  const sign = delta > 0 ? '+' : '';
  const sigBadge = sig
    ? (delta >= 0 ? '<span class="badge badge-green">显著提升</span>' : '<span class="badge badge-red">显著退化</span>')
    : '<span class="badge badge-gray">不显著</span>';

  return `<div class="ab-row">
    <div class="ab-left">
      <div class="ab-metric">${escHtml(zhLabel(item.metric) || item.metric)}</div>
      <div class="ab-meta">波动区间 [${Number(item.ci_95_low||0).toFixed(4)}, ${Number(item.ci_95_high||0).toFixed(4)}]</div>
    </div>
    <div style="display:flex;align-items:center;gap:10px">
      <div class="ab-delta ${cls}">${sign}${delta.toFixed(4)}</div>
      ${sigBadge}
    </div>
  </div>`;
}

function renderCaseItem(c) {
  const passRate = Math.round((c.pass_rate||0)*100);
  const badgeCls = c.pass_rate >= 0.8 ? 'badge-green' : c.pass_rate >= 0.5 ? 'badge-amber' : 'badge-red';
  const samples = c.samples || [];
  const tid = 'case-' + c.case_id;

  const samplesHtml = samples.map(s => `
    <div style="margin-bottom:8px">
      <div style="display:flex;gap:8px;align-items:center;font-size:11px;color:var(--ink4);margin-bottom:4px">
        <span>Sample ${s.sample_index}</span>
        <span>${s.latency_ms ? s.latency_ms + 'ms' : ''}</span>
        ${s.passed === true ? '<span style="color:var(--green)">✓ 通过</span>'
          : s.passed === false ? '<span style="color:var(--red)">✗ 未通过</span>' : ''}
        ${s.error_type ? `<span style="color:var(--red)">${escHtml(s.error_type)}</span>` : ''}
      </div>
      ${s.output ? `<div class="sample-output">${escHtml(s.output)}</div>` : ''}
    </div>`).join('');

  // v6 fix: Use data-* attributes instead of inline onclick
  return `
    <div class="case-item">
      <div class="case-header" data-action="toggleCase" data-case-id="${escAttr(tid)}">
        <span class="badge ${badgeCls}">${passRate}%</span>
        <span style="font-family:var(--mono);font-size:11px;color:var(--ink4)">${escHtml(c.case_id)}</span>
        <span style="font-size:12px;color:var(--ink2)">${escHtml(c.name)}</span>
        <span style="font-size:11px;color:var(--ink4);margin-left:auto">${c.mean_latency_ms ? Math.round(c.mean_latency_ms)+'ms avg' : ''}</span>
      </div>
      <div class="case-body" id="${tid}">${samplesHtml}</div>
    </div>`;
}

function toggleCase(id) {
  const el = document.getElementById(id);
  if (el) el.classList.toggle('open');
}

// ── Runs list ──────────────────────────────────────────────────────────────

async function loadRuns() {
  document.getElementById('runs-list').innerHTML =
    '<div class="loading"><div class="spinner"></div><div>加载中...</div></div>';
  const {ok, data} = await api('GET', '/api/v1/runs?limit=200');
  if (!ok) { document.getElementById('runs-list').innerHTML = '<div class="empty">加载失败</div>'; return; }

  _allRuns = (data && data.runs) ? data.runs : (Array.isArray(data) ? data : []);
  if (!_allRuns.length) {
    document.getElementById('runs-list').innerHTML =
      '<div class="empty"><div class="empty-icon">○</div><div>暂无检测记录</div></div>';
    document.getElementById('runs-pager').innerHTML = '';
    return;
  }
  applyRunFilters();
}

function applyRunFilters() {
  const status = (document.getElementById('runs-filter-status')?.value || 'all').toLowerCase();
  const keyword = (document.getElementById('runs-filter-model')?.value || '').trim().toLowerCase();

  _filteredRuns = _allRuns.filter(r => {
    const okStatus = status === 'all' ? true : String(r.status || '').toLowerCase() === status;
    const m = String(r.model || '').toLowerCase();
    const okKeyword = keyword ? m.includes(keyword) : true;
    return okStatus && okKeyword;
  });

  _runPage = 1;
  renderRunsPage();
}

function renderRunsPage() {
  const total = _filteredRuns.length;
  const pages = Math.max(1, Math.ceil(total / _runPageSize));
  if (_runPage > pages) _runPage = pages;

  const start = (_runPage - 1) * _runPageSize;
  const pageRows = _filteredRuns.slice(start, start + _runPageSize);

  const html = pageRows.map(r => {
    const statusDot = `<span class="status-dot dot-${r.status}"></span>`;
    const pre = r.predetect_identified ? '<span class="badge badge-blue" style="margin-left:6px">已识别</span>' : '';
    const canExport = ['completed','partial_failed'].includes(r.status);
    const checked = _runSelected.has(r.run_id) ? 'checked' : '';

    // v6 fix: Use data-* attributes instead of inline onclick
    return `
      <div class="run-row" data-action="openTask" data-run-id="${escAttr(r.run_id)}">
        <input type="checkbox" ${checked} onclick="toggleRunSelect(event, '${escAttr(r.run_id)}')" />
        ${statusDot}
        <div class="run-model">${escHtml(r.model)}${pre}</div>
        <div class="run-url">${escHtml(r.base_url)}</div>
        <div style="width:80px">${renderStatusBadge(r.status)}</div>
        <div style="display:flex;gap:6px;align-items:center;width:60px;justify-content:flex-end">
          <button class="btn danger" style="padding:4px 8px;font-size:11px" data-action="deleteRun" data-run-id="${escAttr(r.run_id)}">删除</button>
        </div>
        <div class="run-time" style="width:80px;text-align:right">${fmtTime(r.created_at)}</div>
      </div>`;
  }).join('');

  document.getElementById('runs-list').innerHTML = html || '<div class="empty">筛选后暂无记录</div>';

  // v6 fix: Use data-* attributes instead of inline onclick
  document.getElementById('runs-pager').innerHTML = `
    <button class="btn" style="padding:4px 10px;font-size:12px" ${_runPage <= 1 ? 'disabled' : ''} data-action="changePage" data-dir="-1">上一页</button>
    <span style="font-size:12px;color:var(--ink4);display:inline-flex;align-items:center">第 ${_runPage} / ${pages} 页 · 共 ${total} 条</span>
    <button class="btn" style="padding:4px 10px;font-size:12px" ${_runPage >= pages ? 'disabled' : ''} data-action="changePage" data-dir="1">下一页</button>
  `;

  const hint = document.getElementById('runs-filter-hint');
  if (hint) hint.textContent = `已选择 ${_runSelected.size} 条`;
}

function changeRunsPage(delta) {
  _runPage = Math.max(1, _runPage + delta);
  renderRunsPage();
}

function toggleRunSelect(evt, runId) {
  evt.stopPropagation();
  if (_runSelected.has(runId)) _runSelected.delete(runId);
  else _runSelected.add(runId);
  const hint = document.getElementById('runs-filter-hint');
  if (hint) hint.textContent = `已选择 ${_runSelected.size} 条`;
}

function batchExportSelected(type) {
  const ids = [..._runSelected];
  if (!ids.length) {
    showToast('请先勾选至少一条记录', 'warn');
    return;
  }

  const t = type === 'csv' ? 'csv' : type === 'svg' ? 'svg' : 'csv,svg';
  const url = `/api/v1/exports/runs.zip?run_ids=${encodeURIComponent(ids.join(','))}&types=${encodeURIComponent(t)}`;
  window.open(url, '_blank');
}

function batchExportZipAll() {
  const ids = [..._runSelected];
  if (!ids.length) {
    showToast('请先勾选至少一条记录', 'warn');
    return;
  }
  const url = `/api/v1/exports/runs.zip?run_ids=${encodeURIComponent(ids.join(','))}&types=csv,svg`;
  window.open(url, '_blank');
}

async function batchDeleteSelected() {
  const ids = [..._runSelected];
  if (!ids.length) {
    showToast('请先勾选至少一条记录', 'warn');
    return;
  }

  if (ids.length > 0 && !await showConfirmModal(`确定要删除选中的 ${ids.length} 条记录吗？此操作不可恢复。`)) return;

  const {ok, data} = await api('POST', '/api/v1/runs/batch-delete', { run_ids: ids });
  if (!ok) {
    showToast('批量删除失败: ' + (data.error || 'unknown error'), 'error');
    return;
  }

  showToast(`成功删除 ${data.deleted_count} 条记录${data.errors && data.errors.length ? `，${data.errors.length} 条失败` : ''}`, 'info');
  _runSelected.clear();
  loadRuns();
}

async function deleteRunFromList(evt, runId) {
  if (evt) evt.stopPropagation();
  if (!await showConfirmModal('确认删除该检测记录？此操作不可恢复。')) return;

  const {ok, data} = await api('DELETE', '/api/v1/runs/' + runId);
  if (!ok) {
    showToast('删除失败: ' + (data.error || 'unknown error'), 'error');
    return;
  }
  loadRuns();
}

// ── Benchmarks ─────────────────────────────────────────────────────────────

async function markAsBaseline(runId) {
  const {ok, data} = await api('GET', '/api/v1/runs/' + runId);
  if (!ok) {
    showToast('无法获取任务信息', 'error');
    return;
  }
  const defaultName = data.model || '';

  // Let user confirm or change the model name (unique key for baselines)
  const inputName = prompt(
    '请输入基准模型名称（同名基准将被覆盖）：',
    defaultName
  );
  if (!inputName || !inputName.trim()) return;

  const model_name = inputName.trim();
  const display_name = model_name;

  const {ok: ok2, data: data2} = await api('POST', '/api/v1/baselines', {
    run_id: runId,
    model_name,
    display_name,
    notes: '',
  });
  if (!ok2) {
    showToast(data2.error || '创建基准失败', 'error');
    return;
  }
  showToast('已创建基准: ' + display_name, 'info');
  const {ok: ok3, data: data3} = await api('GET', '/api/v1/runs/' + runId);
  if (ok3) {
    renderTaskActions(runId, data3.status, data3.baseline_id);
  }
}

async function compareWithBaseline(runId) {
  // First, fetch available baselines
  const {ok: baselinesOk, data: baselinesData} = await api('GET', '/api/v1/baselines');
  if (!baselinesOk) {
    showToast('无法获取基准模型列表: ' + (baselinesData.error || 'unknown error'), 'error');
    return;
  }

  const baselines = baselinesData.baselines || [];
  if (!baselines.length) {
    showToast('暂无可用基准模型，请先标记某个模型为基准', 'warn');
    return;
  }

  // Create baseline selection modal
  const modal = document.createElement('div');
  modal.style.cssText = `
    position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.5); z-index: 10000;
    display: flex; align-items: center; justify-content: center;
  `;

  const options = baselines.map(b => 
    `<option value="${b.id}">${escHtml(b.display_name || b.model_name)} (${b.model_name})</option>`
  ).join('');

  modal.innerHTML = `
    <div class="card" style="max-width: 500px; width: 90%;">
      <h3 style="margin-top: 0;">选择基准模型进行对比</h3>
      <p style="color: #666; margin-bottom: 16px;">请选择要与当前检测结果对比的基准模型：</p>
      <select id="baseline-select" style="width: 100%; padding: 8px; margin-bottom: 16px; border: 1px solid #ddd; border-radius: 4px;">
        ${options}
      </select>
      <div style="display: flex; gap: 8px; justify-content: flex-end;">
        <button class="btn" onclick="this.closest('[style*=fixed]').remove()">取消</button>
        <button class="btn primary" onclick="confirmBaselineComparison('${runId}', this.closest('[style*=fixed]'))">开始对比</button>
      </div>
    </div>
  `;

  document.body.appendChild(modal);
  document.getElementById('baseline-select').focus();
}

async function confirmBaselineComparison(runId, modal) {
  const select = document.getElementById('baseline-select');
  const baseline_id = select.value;

  if (!baseline_id) {
    showToast('请选择一个基准模型', 'warn');
    return;
  }

  modal.remove();

  // Now perform the comparison with both baseline_id and run_id
  const {ok, data} = await api('POST', '/api/v1/baselines/compare', {
    baseline_id: baseline_id,
    run_id: runId
  });
  if (!ok) {
    showToast('基准对比失败: ' + (data.error || 'unknown error'), 'error');
    return;
  }

  const verdict = String(data.verdict || '').toLowerCase();
  const verdictClass = verdict === 'match'
    ? 'verdict-match'
    : verdict === 'suspicious'
      ? 'verdict-suspicious'
      : 'verdict-mismatch';

  const d = data.score_delta || {};
  const top5 = data.feature_drift_top5 || {};
  const rows = Object.entries(top5).map(([k, v]) => {
    const pct = Number((v && v.delta_pct) || 0);
    const cls = Math.abs(pct) > 20 ? 'drift-high' : 'drift-low';
    return `
      <tr class="${cls}">
        <td>${escHtml(k)}</td>
        <td>${Number((v && v.baseline) || 0).toFixed(4)}</td>
        <td>${Number((v && v.current) || 0).toFixed(4)}</td>
        <td>${pct.toFixed(2)}%</td>
      </tr>`;
  }).join('');

  const reportHtml = `
    <div class="card" style="padding:0;overflow:hidden">
      <div class="console-head" style="position:sticky;top:0;z-index:1;background:#1a4a8a">
        <span>基准对比报告</span>
        <button class="chip" onclick="restoreLogPanel()" style="background:#2a5a9a;border:1px solid #a0c0e8;color:#d6e4f0;cursor:pointer">返回日志</button>
      </div>
      <div style="padding:16px;background:var(--bg);min-height:420px;max-height:80vh;overflow:auto">
        <div class="${verdictClass}" style="font-size:18px;font-weight:700;margin-bottom:12px">判定：${verdict}</div>
        <div style="font-size:12px;color:var(--ink3);margin-bottom:12px">
          余弦相似度: <b>${Number(data.cosine_similarity || 0).toFixed(4)}</b><br>
          分数差异: 总分 ${fmtScore(d.total || 0)}, 能力 ${fmtScore(d.capability || 0)},
          真实性 ${fmtScore(d.authenticity || 0)}, 性能 ${fmtScore(d.performance || 0)}<br>
          ${escHtml(data.verdict_reason || '')}
        </div>
        <table style="width:100%;border-collapse:collapse;font-size:12px">
          <thead>
            <tr style="background:var(--bg2)">
              <th style="padding:8px 12px;text-align:left;font-size:10px;color:var(--ink4)">特征</th>
              <th style="padding:8px 12px;text-align:right;font-size:10px;color:var(--ink4)">基准值</th>
              <th style="padding:8px 12px;text-align:right;font-size:10px;color:var(--ink4)">当前值</th>
              <th style="padding:8px 12px;text-align:right;font-size:10px;color:var(--ink4)">差异%</th>
            </tr>
          </thead>
          <tbody>${rows || '<tr><td colspan="4" style="padding:12px;text-align:center;color:var(--ink4)">无漂移数据</td></tr>'}</tbody>
        </table>
      </div>
    </div>`;

  const logCol = document.getElementById('task-log-col');
  if (logCol) {
    _savedLogPanelHtml = logCol.innerHTML;
    logCol.innerHTML = reportHtml;
  }
}

let _savedLogPanelHtml = '';

function saveLogPanel() {
  const logCol = document.getElementById('task-log-col');
  if (logCol) {
    _savedLogPanelHtml = logCol.innerHTML;
  }
}

function restoreLogPanel() {
  const logCol = document.getElementById('task-log-col');
  if (logCol && _savedLogPanelHtml) {
    logCol.innerHTML = _savedLogPanelHtml;
  }
}

async function loadBaselines() {
  const {ok, data} = await api('GET', '/api/v1/baselines');
  if (!ok) { document.getElementById('benchmarks-content').innerHTML = '<div class="empty">加载失败</div>'; return; }

  const baselines = data.baselines || [];

  if (!baselines.length) {
    document.getElementById('benchmarks-content').innerHTML =
      '<div class="empty"><div class="empty-icon">○</div><div>暂无基准模型，请先在任务中标记</div></div>';
    return;
  }

  const html = `
    <div class="card">
      <table>
        <thead><tr>
          <th>模型名称</th>
          <th>总分</th>
          <th>能力分</th>
          <th>真实分</th>
          <th>性能分</th>
          <th>创建时间</th>
          <th>操作</th>
        </tr></thead>
        <tbody>
          ${baselines.map(b => `<tr>
            <td><strong>${escHtml(b.display_name || b.model_name)}</strong></td>
            <td>${fmtScore(b.total_score)}</td>
            <td>${fmtScore(b.capability_score)}</td>
            <td>${fmtScore(b.authenticity_score)}</td>
            <td>${fmtScore(b.performance_score)}</td>
            <td>${fmtTime(b.created_at)}</td>
            <td>
              <button class="btn" style="padding:4px 10px;font-size:11px" data-action="viewBaseline" data-baseline-id="${escAttr(b.id)}">查看报告</button>
              <button class="btn danger" style="padding:4px 10px;font-size:11px" data-action="deleteBaseline" data-baseline-id="${escAttr(b.id)}">删除</button>
            </td>
          </tr>`).join('')}
        </tbody>
      </table>
    </div>`;

  document.getElementById('benchmarks-content').innerHTML = html;
}

async function viewBaselineReport(baselineId) {
  const {ok, data} = await api('GET', '/api/v1/baselines/' + baselineId);
  if (!ok) { showToast('加载失败', 'error'); return; }

  if (data.source_run_id) {
    openTask(data.source_run_id);
    return;
  }

  const b = data;
  const html = `
    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px">
        <h2 style="margin:0">${escHtml(b.display_name || b.model_name)}</h2>
        <button class="btn" onclick="loadBaselines()">返回列表</button>
      </div>
      <div class="score-grid">
        <div class="score-card"><div class="score">${fmtScore(b.total_score)}</div><div class="score-label">总分</div></div>
        <div class="score-card"><div class="score">${fmtScore(b.capability_score)}</div><div class="score-label">能力分</div></div>
        <div class="score-card"><div class="score">${fmtScore(b.authenticity_score)}</div><div class="score-label">真实分</div></div>
        <div class="score-card"><div class="score">${fmtScore(b.performance_score)}</div><div class="score-label">性能分</div></div>
      </div>
      <div style="margin-top:16px;font-size:12px;color:var(--ink4)">
        创建时间: ${fmtTime(b.created_at)} | 采样次数: ${b.sample_count || 1}
      </div>
    </div>`;

  document.getElementById('benchmarks-content').innerHTML = html;
}

async function deleteBaseline(baselineId) {
  if (!await showConfirmModal('确认删除该基准模型？')) return;
  const {ok, data} = await api('DELETE', '/api/v1/baselines/' + baselineId);
  if (!ok) { showToast('删除失败: ' + (data.error || 'unknown error'), 'error'); return; }
  loadBaselines();
}

// ── Utilities ──────────────────────────────────────────────────────────────

// Show toast notification (v14: supports type='info'|'error'|'warn')
function showToast(message, type = 'info', duration = 3500) {
  // Legacy call signature: showToast(msg, number) — treat as duration
  if (typeof type === 'number') { duration = type; type = 'info'; }
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.textContent = message;
  document.body.appendChild(toast);
  setTimeout(() => { if (toast.parentNode) toast.remove(); }, duration);
}

// v15 Phase 12: Modal confirmation (replaces browser confirm())
function showConfirmModal(message) {
  return new Promise((resolve) => {
    const existing = document.getElementById('v15-confirm-modal');
    if (existing) existing.remove();

    const modal = document.createElement('div');
    modal.id = 'v15-confirm-modal';
    modal.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,.5);z-index:9999;display:flex;align-items:center;justify-content:center';
    modal.innerHTML = `
      <div style="background:#fff;border-radius:8px;padding:24px;max-width:400px;width:90%;box-shadow:0 4px 24px rgba(0,0,0,.2)">
        <p style="margin:0 0 16px;font-size:15px;color:#1e293b">${escHtml(message)}</p>
        <div style="display:flex;gap:8px;justify-content:flex-end">
          <button id="v15-confirm-cancel" style="padding:8px 16px;border:1px solid #e2e8f0;border-radius:6px;background:#fff;cursor:pointer;font-size:13px">取消</button>
          <button id="v15-confirm-ok" style="padding:8px 16px;border:none;border-radius:6px;background:#ef4444;color:#fff;cursor:pointer;font-size:13px">确认</button>
        </div>
      </div>
    `;
    document.body.appendChild(modal);
    document.getElementById('v15-confirm-ok').onclick = () => { modal.remove(); resolve(true); };
    document.getElementById('v15-confirm-cancel').onclick = () => { modal.remove(); resolve(false); };
    modal.onclick = (e) => { if (e.target === modal) { modal.remove(); resolve(false); } };
  });
}

function escHtml(s) {
  return String(s||'')
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

// v6 fix: Score color grading - different colors for different score ranges
function fmtScore(v) {
  if (v === null || v === undefined || v === '-') return '<span style="color:var(--ink4)">—</span>';
  const n = Number(v);
  if (isNaN(n)) return '<span style="color:var(--ink4)">—</span>';
  let color;
  if (n >= 80) color = '#16a34a';      // 绿色: 优秀
  else if (n >= 60) color = '#2563eb';  // 蓝色: 良好
  else if (n >= 40) color = '#d97706';  // 橙色: 一般
  else color = '#dc2626';               // 红色: 差
  return `<span style="color:${color};font-weight:600">${Math.round(n)}</span>`;
}

// v15: Data quality label for score fields
function dataLabel(value, reason) {
  if (value !== null && value !== undefined) return 'measured';
  if (reason) return reason;
  return 'unavailable';
}

// v15: Render a score with its data label
function renderScoreWithLabel(value, label) {
  const formattedScore = fmtScore(value);
  if (value === null || value === undefined) {
    return `<span class="score-na" title="${label || 'Not measured'}">${formattedScore}</span>`;
  }
  return `<span class="score-measured">${formattedScore}</span>`;
}

// v6 fix: Render dimension score with null handling
function renderDimensionScore(label, value) {
  if (value === null || value === undefined) {
    return `<div style="font-size:12px;color:var(--ink4)">
      ${escHtml(label)}: <span style="font-style:italic">数据不足</span>
    </div>`;
  }
  return `<div style="font-size:12px;color:var(--ink3)">${escHtml(label)}: ${fmtScore(value)}</div>`;
}

// v14 Phase 3: Identity Exposure Card
function toggleIdentityCard() {
  const body = document.getElementById('identity-exposure-body');
  if (body) {
    body.style.display = body.style.display === 'none' ? '' : 'none';
  }
}

function renderIdentityExposure(data) {
  // Insert card from template if not already in DOM
  if (!document.getElementById('identity-exposure-card')) {
    const tpl = document.getElementById('identity-exposure-tpl');
    if (tpl) {
      const clone = tpl.content.cloneNode(true);
      const reportSection = document.getElementById('report-section');
      if (reportSection) reportSection.appendChild(clone);
    }
  }

  const card = document.getElementById('identity-exposure-card');
  const content = document.getElementById('identity-exposure-content');
  const badge = document.getElementById('identity-collision-badge');
  if (!card || !content) return;

  if (!data || (!data.identity_collision && !data.extracted_system_prompt)) {
    card.style.display = 'none';
    return;
  }

  card.style.display = '';

  // Badge
  if (badge) {
    badge.textContent = data.identity_collision
      ? '置信度 ' + (data.collision_confidence * 100).toFixed(0) + '%'
      : '系统提示词已抽取';
    badge.style.color = data.identity_collision ? '#e74c3c' : '#f39c12';
  }

  // Collapse body by default if not collision
  const body = document.getElementById('identity-exposure-body');
  if (body && !data.identity_collision) {
    body.style.display = 'none';
  }

  let html = '';

  if (data.identity_collision && data.top_families && data.top_families.length > 0) {
    html += '<p><strong>声称模型:</strong> ' + escapeHtml(data.claimed_model || '未知') + '</p>';
    html += '<table style="width:100%; border-collapse: collapse; margin: 0.5rem 0;">';
    html += '<tr style="background:#f8f8f8;"><th style="padding:4px 8px; text-align:left;">排名</th><th style="text-align:left; padding:4px 8px;">疑似模型家族</th><th style="text-align:left; padding:4px 8px;">置信度</th><th style="text-align:left; padding:4px 8px;">命中信号</th></tr>';
    data.top_families.forEach(function(f, i) {
      if (f.raw_score > 0) {
        const pct = (f.posterior * 100).toFixed(1);
        const ev = f.evidence ? f.evidence.slice(0, 2).map(function(e) { return '<em>' + escapeHtml(e.matched_text) + '</em>'; }).join(', ') : '';
        html += '<tr style="border-top:1px solid #eee;"><td style="padding:4px 8px;">' + (i+1) + '</td><td style="padding:4px 8px; font-weight:bold;">' + escapeHtml(f.family) + '</td><td style="padding:4px 8px;">' + pct + '%</td><td style="padding:4px 8px; font-size:0.85em; color:#666;">' + ev + '</td></tr>';
      }
    });
    html += '</table>';

    // Show evidence snippets for top family
    const top = data.top_families[0];
    if (top && top.evidence && top.evidence.length > 0) {
      html += '<details style="margin-top:0.5rem;"><summary style="cursor:pointer; color:#666;">查看响应证据</summary><ul style="margin:0.5rem 0; padding-left:1.5rem;">';
      top.evidence.slice(0, 3).forEach(function(e) {
        html += '<li style="margin:0.25rem 0; font-size:0.85em;"><code>' + escapeHtml(e.matched_text) + '</code> — <span style="color:#666;">' + escapeHtml(e.snippet || '') + '</span> <small style="color:#999;">[' + escapeHtml(e.case_id || '') + ']</small></li>';
      });
      html += '</ul></details>';
    }
  }

  if (data.extracted_system_prompt) {
    html += '<hr style="margin:0.75rem 0; border:none; border-top:1px solid #eee;">';
    html += '<p><strong>抽取到的系统提示词</strong> <button onclick="navigator.clipboard.writeText(document.getElementById(\'sp-text\').textContent)" style="font-size:0.75em; padding:2px 6px; margin-left:8px;">复制</button></p>';
    html += '<pre id="sp-text" style="background:#f8f8f8; padding:0.75rem; border-radius:4px; font-size:0.8em; max-height:200px; overflow-y:auto; white-space:pre-wrap;">' + escapeHtml(data.extracted_system_prompt) + '</pre>';
  }

  content.innerHTML = html || '<p style="color:#666; font-style:italic;">无法提取足够证据</p>';
}

// v14 Phase 8: Render new v14 report section cards
function renderV14Cards(report, runId) {
  const el = document.getElementById('report-section');
  if (!el) return;
  const sc = (report && report.scorecard) || {};
  const ta = sc.token_analysis || null;
  const completeness = (sc.v13 && sc.v13.completeness != null) ? sc.v13.completeness : sc.completeness;
  const pre = (report && report.predetect_result) || null;

  let html = '';

  // Token Analysis card
  if (ta) {
    html += `
      <div class="card" id="token-analysis-card">
        <h3>Token 分析</h3>
        <p style="font-size:12px;color:var(--ink3)">计数方法: ${escHtml(ta.counting_method || ta.token_counting_method || 'N/A')}</p>
        <p style="font-size:12px;color:var(--ink3)">优化器已启用: <strong>${ta.prompt_optimizer_used ? '是' : '否'}</strong></p>
        <p style="font-size:12px;color:var(--ink3)">节省估算: <strong>${ta.tokens_saved_estimate != null ? ta.tokens_saved_estimate : 'N/A'}</strong> tokens</p>
      </div>`;
  }

  // Data completeness card
  if (completeness != null) {
    const pct = Math.round(completeness * 100);
    html += `
      <div class="card" id="completeness-card">
        <h3>数据完整性</h3>
        <div class="v14-progress-bar"><div style="width:${pct}%"></div></div>
        <p style="font-size:12px;color:var(--ink3);margin-top:6px">${pct}% 维度有效数据</p>
      </div>`;
  }

  // PreDetect layer status dots
  if (pre && pre.layers && Array.isArray(pre.layers)) {
    const TOTAL_LAYERS = 20;
    const dots = [];
    for (let i = 0; i < TOTAL_LAYERS; i++) {
      const layer = pre.layers.find(l => l.layer === i || l.layer_id === i);
      let color, title;
      if (!layer) { color = '#d1d5db'; title = `L${i}: 未运行`; }
      else if (layer.skipped) { color = '#fbbf24'; title = `L${i}: 已跳过`; }
      else { color = '#22c55e'; title = `L${i}: ${layer.result || '通过'}`; }
      dots.push(`<span title="${escAttr(title)}" style="display:inline-block;width:14px;height:14px;border-radius:50%;background:${color};margin:2px;cursor:help"></span>`);
    }
    html += `
      <div class="card" id="predetect-layers-card">
        <h3>预检测层状态 (${pre.layers.length}/${TOTAL_LAYERS} 层运行)</h3>
        <div style="display:flex;flex-wrap:wrap;gap:2px;margin-top:6px">${dots.join('')}</div>
        <div style="font-size:11px;color:var(--ink4);margin-top:6px">
          <span style="display:inline-block;width:12px;height:12px;background:#22c55e;border-radius:50%;vertical-align:middle"></span> 通过 &nbsp;
          <span style="display:inline-block;width:12px;height:12px;background:#fbbf24;border-radius:50%;vertical-align:middle"></span> 跳过 &nbsp;
          <span style="display:inline-block;width:12px;height:12px;background:#d1d5db;border-radius:50%;vertical-align:middle"></span> 未运行
        </div>
      </div>`;
  }

  if (html) {
    const v14Container = document.createElement('div');
    v14Container.id = 'v14-cards-container';
    v14Container.innerHTML = html;
    el.appendChild(v14Container);
  }
}

function escapeHtml(text) {
  if (!text) return '';
  return String(text).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function fmtTime(iso) {
  if (!iso) return '–';
  try {
    const d = new Date(iso);
    return d.toLocaleString('zh-CN', {month:'2-digit',day:'2-digit',
      hour:'2-digit',minute:'2-digit'});
  } catch { return iso.slice(0,16); }
}

function setNavStatus(status) {
  const el = document.getElementById('nav-status');
  if (el) el.textContent = status;
}

// Init
restoreForm();
showPage('home');

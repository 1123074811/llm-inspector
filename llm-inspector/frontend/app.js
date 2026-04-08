
const API = '';  // Same origin
let _pollTimer = null;
let _currentRunId = null;
let _lastProgressSnapshot = { completed: 0, total: 0, phase: 'queued' };
let _consoleLines = [];
let _typingTimer = null;
let _logPinnedToBottom = true;
let _logFilter = 'all';

let _allRuns = [];
let _filteredRuns = [];
let _runPage = 1;
const _runPageSize = 10;
let _runSelected = new Set();

// ── Page routing ───────────────────────────────────────────────────────────

function showPage(name) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));

  const pageMap = {home:'page-home', runs:'page-runs', benchmarks:'page-benchmarks', task:'page-task'};
  const navMap  = {home:'nav-home', runs:'nav-runs', benchmarks:'nav-benchmarks'};

  const pageEl = document.getElementById(pageMap[name]);
  if (pageEl) pageEl.classList.add('active');
  const navEl = document.getElementById(navMap[name]);
  if (navEl) navEl.classList.add('active');

  if (name === 'runs')       loadRuns();
  if (name === 'benchmarks') loadBaselines();
  if (_pollTimer && name !== 'task') { clearInterval(_pollTimer); _pollTimer = null; }
}

// ── API helpers ────────────────────────────────────────────────────────────

async function api(method, path, body) {
  const opts = { method, headers: {'Content-Type':'application/json'} };
  if (body) opts.body = JSON.stringify(body);
  const r = await fetch(API + path, opts);
  const text = await r.text();
  let data;
  try { data = JSON.parse(text); } catch { data = {error: text}; }
  return { ok: r.ok, status: r.status, data };
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

  // Clear form
  document.getElementById('f-key').value = '';

  // Navigate to task page
  openTask(data.run_id);
}

// ── Task detail page ───────────────────────────────────────────────────────

function openTask(runId) {
  _currentRunId = runId;
  _lastProgressSnapshot = { completed: 0, total: 0, phase: 'queued' };
  _consoleLines = [{ text: '初始化任务上下文...', type: 'normal' }];
  showPage('task');
  document.getElementById('task-title').textContent = '检测: ' + runId.slice(0,8) + '...';
  document.getElementById('task-status-badge').innerHTML = '';
  document.getElementById('task-content').innerHTML =
    '<div class="loading"><div class="spinner"></div><div>检测进行中，请稍候...</div></div>';

  if (_pollTimer) clearInterval(_pollTimer);
  _pollTimer = setInterval(() => pollTask(runId), 2000);
  pollTask(runId);
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
  renderTaskActions(runId, status, data.baseline_id);
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
}

function renderStatusBadge(status) {
  const labels = {
    queued:'排队中', pre_detecting:'预检测中', pre_detected:'预检测完成',
    running:'测试执行中', completed:'已完成', partial_failed:'部分失败', failed:'失败'
  };
  const cls = {completed:'badge-green', failed:'badge-red', partial_failed:'badge-amber'};
  const c = cls[status] || 'badge-blue';
  return `<span class="badge ${c}">${labels[status]||status}</span>`;
}

function renderTaskActions(runId, status, baselineId) {
  const el = document.getElementById('task-actions');
  if (!el) return;

  const canCancel = ['queued','pre_detecting','running'].includes(status);
  const canRetry = ['failed','partial_failed'].includes(status);
  const canExport = ['completed','partial_failed'].includes(status);

  let html = '';
  if (canCancel) {
    html += `<button class="btn danger" style="padding:6px 12px;font-size:12px" onclick="cancelRun('${runId}')">停止任务</button>`;
  }
  if (canRetry) {
    html += `<button class="btn" style="padding:6px 12px;font-size:12px" onclick="retryRun('${runId}')">从失败处重试</button>`;
  }
  if (canExport) {
    if (!baselineId) {
      html += `<button class="btn primary" style="padding:6px 12px;font-size:12px" onclick="markAsBaseline('${runId}')">标记为基准模型</button>`;
    } else {
      html += `<button class="btn danger" style="padding:6px 12px;font-size:12px" onclick="unmarkAsBaseline('${runId}', '${baselineId}')">取消基准标记</button>`;
    }
    html += `<button class="btn" style="padding:6px 12px;font-size:12px" onclick="compareWithBaseline('${runId}')">与基准对比</button>`;
    html += `<button class="btn" style="padding:6px 12px;font-size:12px" onclick="exportReportPdf('${runId}')">生成 PDF 报告</button>`;
    html += `<button class="btn" style="padding:6px 12px;font-size:12px" onclick="downloadRadarSvg('${runId}')">导出雷达图</button>`;
  }
  el.innerHTML = html;
}

async function cancelRun(runId) {
  if (!confirm('确认停止当前任务？已发出的请求可能仍会完成，但会尽快停止后续请求。')) return;
  const {ok, data} = await api('POST', `/api/v1/runs/${runId}/cancel`);
  if (!ok) {
    alert('停止失败: ' + (data.error || 'unknown error'));
    return;
  }
  setNavStatus('cancelling');
  pollTask(runId);
}

async function retryRun(runId) {
  const {ok, data} = await api('POST', `/api/v1/runs/${runId}/retry`);
  if (!ok) {
    alert('重试失败: ' + (data.error || 'unknown error'));
    return;
  }
  openTask(data.run_id || runId);
}

async function continueFullTest(runId) {
  const {ok, data} = await api('POST', `/api/v1/runs/${runId}/continue`);
  if (!ok) {
    alert('继续测试失败: ' + (data.error || 'unknown error'));
    return;
  }
  // Restart polling
  if (_pollTimer) clearInterval(_pollTimer);
  _pollTimer = setInterval(() => pollTask(runId), 2000);
  pollTask(runId);
}

async function unmarkAsBaseline(runId, baselineId) {
  if (!confirm('确认该模型不再作为对比基准？这不会删除原始检测记录。')) return;
  const {ok, data} = await api('DELETE', '/api/v1/baselines/' + baselineId);
  if (!ok) {
    alert('移除失败: ' + (data.error || 'unknown error'));
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
    alert('跳过失败: ' + (data.error || 'unknown error'));
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

async function previewIsomorphicCases() {
  const hint = document.getElementById('iso-hint');
  if (hint) {
    hint.style.color = 'var(--ink4)';
    hint.textContent = '预览中...';
  }
  const {ok, data} = await api('POST', '/api/v1/tools/generate-isomorphic?apply=false');
  if (!ok) {
    if (hint) {
      hint.style.color = 'var(--red)';
      hint.textContent = '预览失败: ' + (data.error || 'unknown error');
    }
    return;
  }
  const ids = (data.to_add_ids || []).slice(0, 5).join(', ');
  if (hint) {
    hint.style.color = 'var(--blue)';
    hint.textContent = `可新增 ${data.preview_count || 0} 题${ids ? `（${ids}${(data.to_add_ids || []).length > 5 ? ' ...' : ''}）` : ''}`;
  }
}

async function applyIsomorphicCases() {
  const hint = document.getElementById('iso-hint');
  if (hint) {
    hint.style.color = 'var(--ink4)';
    hint.textContent = '写入中...';
  }
  const {ok, data} = await api('POST', '/api/v1/tools/generate-isomorphic?apply=true');
  if (!ok) {
    if (hint) {
      hint.style.color = 'var(--red)';
      hint.textContent = '写入失败: ' + (data.error || 'unknown error');
    }
    return;
  }
  if (hint) {
    hint.style.color = 'var(--green)';
    hint.textContent = `已写入 ${data.added_count || 0} 题`;
  }
}

function renderTaskProgress(data, responses = []) {
  const prog = data.progress || {};
  const pct = prog.total ? Math.round(prog.completed / prog.total * 100) : 0;
  const scoringProfileVersion = data.scoring_profile_version || 'v1';

  animateProcessConsole(prog, data.status, data.predetect_result, responses);

  const pre = data.predetect_result;

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
      <div style="color:var(--red);font-size:13px">错误: ${escHtml(data.error_message)}</div></div>` : '';

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
  if (_logFilter === 'prompt') return text.includes('[提示词]');
  if (_logFilter === 'error') {
    return text.includes('[结果] 错误') || text.includes(' error') || text.includes('failed') || line?.type === 'del';
  }
  return true;
}

function renderStageExperience(data, prog, pct, opts = {includeConsole:true}) {
  const stage = stageFromStatus(data.status, prog, data.predetect_result);
  const stageNames = ['排队', '预检测', '能力测试', '分析报告'];

  const stageItems = stageNames.map((name, idx) => {
    const state = idx < stage ? 'done' : idx === stage ? 'active' : '';
    const texts = [
      `任务已提交${data.created_at ? ' · ' + fmtTime(data.created_at) : ''}`,
      data.predetect_result?.identified_as ? `识别: ${data.predetect_result.identified_as}` : '执行指纹探测',
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
          <div class="stage-sub">当前阶段：${escHtml(stageNames[stage] || '分析报告')} · 进度 ${pct}%</div>
        </div>
        <div>${renderStatusBadge(data.status)}</div>
      </div>
      <div class="stage-track">${stageItems}</div>
      ${consoleBlock}
    </div>`;
}

function stageFromStatus(status, prog, pre) {
  if (status === 'queued') return 0;
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
      pre_detecting: '预检测中',
      pre_detected: '预检测完成',
      running: '能力测试中',
      completed: '已完成',
      partial_failed: '部分失败',
      failed: '失败',
    };
    pushConsoleSeparator('阶段切换');
    pushConsole(`[阶段] ${phaseLabel[status] || status}`, 'meta');
  }

  if (status === 'pre_detecting') {
    pushConsoleTyping('[预检测] 正在进行预检测：header / self-report / identity probe ...');
  }

  if (pre?.identified_as && !prev.predetectLogged) {
    pushConsoleSeparator('预检测结果');
    pushConsole(`[预检测] 识别模型: ${pre.identified_as}`);
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
      pushConsole(`[回答] ${ans}`);
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
  };
}

function pushConsole(line, type = 'normal') {
  _consoleLines.push({ text: line, type });
  if (_consoleLines.length > 160) _consoleLines.shift();
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

function renderPredetectCard(pre, runId, status) {
  if (!pre.success && !pre.identified_as) return '';
  const conf = Math.round((pre.confidence || 0) * 100);
  const layers = pre.layer_results || [];
  const stopLayer = pre.layer_stopped || '全部';
  const routing = pre.routing_info || {};

  // Routing detail section
  let routingHtml = '';
  if (routing.is_routed) {
    routingHtml = `
      <div style="margin-top:10px;padding:10px 12px;background:rgba(255,255,255,.6);border-radius:8px;font-size:12px">
        <div style="font-weight:600;color:var(--ink2);margin-bottom:4px">路由检测</div>
        <div style="display:flex;gap:20px;flex-wrap:wrap">
          <div><span style="color:var(--ink4)">声称模型：</span><code>${escHtml(routing.claimed_model || '?')}</code></div>
          <div><span style="color:var(--ink4)">实际模型：</span><code style="color:var(--red);font-weight:600">${escHtml(routing.returned_model || '?')}</code></div>
        </div>
        ${routing.probe_latency_ms ? `<div style="margin-top:4px"><span style="color:var(--ink4)">探针延迟：</span>${routing.probe_latency_ms}ms</div>` : ''}
      </div>`;
  } else if (routing.returned_model) {
    routingHtml = `
      <div style="margin-top:8px;font-size:12px;color:var(--ink3)">
        实际返回模型：<code>${escHtml(routing.returned_model)}</code>
        ${routing.probe_latency_ms ? ` · 延迟 ${routing.probe_latency_ms}ms` : ''}
      </div>`;
  }

  let evidenceHtml = '';
  if (layers.length) {
    const allEvidence = layers.flatMap(l => l.evidence || []);
    evidenceHtml = `<ul class="evidence-list" style="margin-top:8px">
      ${allEvidence.slice(0, 6).map(e => `<li>${escHtml(e)}</li>`).join('')}
    </ul>`;
  }

  // Action buttons: show when pre_detected and testing was skipped
  let actionHtml = '';
  if (status === 'pre_detected' && pre.should_proceed_to_testing === false) {
    actionHtml = `
      <div style="margin-top:12px;display:flex;gap:10px;align-items:center">
        <button class="btn primary" style="padding:8px 16px;font-size:13px" onclick="continueFullTest('${runId}')">
          继续完整测试
        </button>
        <button class="btn" style="padding:8px 16px;font-size:13px" onclick="skipTesting('${runId}')">
          跳过测试，直接出报告
        </button>
        <span style="font-size:11px;color:var(--ink4)">完整测试将获得详细评分、相似度对比等数据</span>
      </div>`;
  } else if (pre.should_proceed_to_testing === false && !['pre_detected'].includes(status)) {
    actionHtml = '<div style="margin-top:8px;font-size:11px;color:var(--blue);font-weight:600">✓ 置信度充足，已跳过完整能力测试</div>';
  }

  return `
    <div class="predetect-card">
      <div class="card-label" style="color:var(--blue);opacity:.7">预检测结果 · Layer ${escHtml(String(stopLayer))}</div>
      <div class="predetect-id">${pre.identified_as ? escHtml(pre.identified_as) : '未识别'}</div>
      <div class="predetect-conf">置信度 ${conf}% · 消耗约 ${pre.total_tokens_used || 0} tokens</div>
      ${routingHtml}
      ${evidenceHtml}
      ${actionHtml}
    </div>`;
}

async function loadReport(runId) {
  const {ok, data} = await api('GET', '/api/v1/runs/' + runId + '/report');
  if (!ok) return;
  const el = document.getElementById('report-section');
  if (el) el.innerHTML = renderReport(data);
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
      const pct = d.percentile == null ? '-' : `${Number(d.percentile).toFixed(1)}%`;
      const width = Math.max(2, Math.min(100, ((Number(d.theta || 0) + 4) / 8) * 100));
      return `<div style="margin-bottom:8px">
        <div style="display:flex;justify-content:space-between;font-size:12px;color:var(--ink3)">
          <span>${escHtml(zhLabel(d.dimension))}</span>
          <span>${pct !== '-' ? `超越 ${pct}` : `得分为 ${t}`}</span>
        </div>
        <div class="progress-bar" style="height:7px"><div class="progress-fill" style="width:${width}%"></div></div>
      </div>`;
    }).join('');

    html += `
      <div class="card">
        <h2>相对能力标尺（Theta）</h2>
        <div class="score-grid" style="grid-template-columns:repeat(2,1fr)">
          ${scoreCard(theta.global_theta, '综合评估水平 (θ)')}
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

  // Similarity
  if (sim.length) {
    html += `
      <div class="card">
        <h2>相似度排名</h2>
        ${sim.map(s => {
          const pct = Math.round(s.score * 100);
          const hasCi = s.ci_95_low != null && s.ci_95_high != null;
          const ciLo = hasCi ? Math.round(s.ci_95_low * 100) : null;
          const ciHi = hasCi ? Math.round(s.ci_95_high * 100) : null;
          const ciText = hasCi
            ? `[${fmtScore(ciLo)}–${fmtScore(ciHi)}%]`
            : `[样本不足，CI 不可用]`;
          const barColor = s.score > 0.75 ? '#E24B4A' : s.score > 0.55 ? '#EF9F27' : '#1D9E75';
          const ciBarLeft = hasCi ? ciLo : 0;
          const ciBarWidth = hasCi ? Math.max(1, ciHi - ciLo) : pct;
          return `
            <div class="sim-row">
              <div class="sim-name">${escHtml(s.benchmark)}</div>
              <div class="sim-bar-wrap">
                <div class="sim-bar" style="width:${pct}%;background:${barColor}"></div>
                <div class="sim-ci" style="left:${hasCi ? ciBarLeft : 0}%;width:${ciBarWidth}%;background:${barColor}"></div>
              </div>
              <div class="sim-score">${fmtScore(pct)}%</div>
              <div style="font-size:10px;color:var(--ink4);min-width:80px">${ciText}</div>
            </div>`;
        }).join('')}
        <div style="font-size:11px;color:var(--ink4);margin-top:8px">[ ] 为 95% 置信区间</div>
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
      <div style="border:1px solid var(--rule);border-radius:10px;overflow:hidden;background:#fff">
        <iframe src="/api/v1/runs/${encodeURIComponent(r.run_id)}/radar.svg" style="width:100%;aspect-ratio:760/560;border:0" scrolling="no" loading="lazy"></iframe>
      </div>
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
  return `<div class="score-card"><div class="score">${v}</div><div class="score-label">${label}</div></div>`;
}

const _zhLabels = {
  // 维度名称
  instruction: '指令理解', protocol: '协议兼容', performance: '性能表现',
  reasoning: '推理能力', coding: '编程能力', consistency: '一致性',
  system: '系统提示', param: '参数控制', style: '风格特征', refusal: '安全控制',
  safety: '安全控制', speed: '响应速度', stability: '稳定性', cost_efficiency: '成本效率',
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

  return `
    <div class="case-item">
      <div class="case-header" onclick="toggleCase('${tid}')">
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

    return `
      <div class="run-row" onclick="openTask('${r.run_id}')">
        <input type="checkbox" ${checked} onclick="toggleRunSelect(event, '${r.run_id}')" />
        ${statusDot}
        <div class="run-model">${escHtml(r.model)}${pre}</div>
        <div class="run-url">${escHtml(r.base_url)}</div>
        <div style="width:80px">${renderStatusBadge(r.status)}</div>
        <div style="display:flex;gap:6px;align-items:center;width:60px;justify-content:flex-end">
          <button class="btn danger" style="padding:4px 8px;font-size:11px" onclick="deleteRunFromList(event, '${r.run_id}')">删除</button>
        </div>
        <div class="run-time" style="width:80px;text-align:right">${fmtTime(r.created_at)}</div>
      </div>`;
  }).join('');

  document.getElementById('runs-list').innerHTML = html || '<div class="empty">筛选后暂无记录</div>';

  document.getElementById('runs-pager').innerHTML = `
    <button class="btn" style="padding:4px 10px;font-size:12px" ${_runPage <= 1 ? 'disabled' : ''} onclick="changeRunsPage(-1)">上一页</button>
    <span style="font-size:12px;color:var(--ink4);display:inline-flex;align-items:center">第 ${_runPage} / ${pages} 页 · 共 ${total} 条</span>
    <button class="btn" style="padding:4px 10px;font-size:12px" ${_runPage >= pages ? 'disabled' : ''} onclick="changeRunsPage(1)">下一页</button>
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
    alert('请先勾选至少一条记录');
    return;
  }

  const t = type === 'csv' ? 'csv' : type === 'svg' ? 'svg' : 'csv,svg';
  const url = `/api/v1/exports/runs.zip?run_ids=${encodeURIComponent(ids.join(','))}&types=${encodeURIComponent(t)}`;
  window.open(url, '_blank');
}

function batchExportZipAll() {
  const ids = [..._runSelected];
  if (!ids.length) {
    alert('请先勾选至少一条记录');
    return;
  }
  const url = `/api/v1/exports/runs.zip?run_ids=${encodeURIComponent(ids.join(','))}&types=csv,svg`;
  window.open(url, '_blank');
}

async function batchDeleteSelected() {
  const ids = [..._runSelected];
  if (!ids.length) {
    alert('请先勾选至少一条记录');
    return;
  }

  if (!confirm(`确定要删除选中的 ${ids.length} 条记录吗？此操作不可恢复。`)) return;

  const {ok, data} = await api('POST', '/api/v1/runs/batch-delete', { run_ids: ids });
  if (!ok) {
    alert('批量删除失败: ' + (data.error || 'unknown error'));
    return;
  }

  alert(`成功删除 ${data.deleted_count} 条记录${data.errors && data.errors.length ? `，${data.errors.length} 条失败` : ''}`);
  _runSelected.clear();
  loadRuns();
}

async function deleteRunFromList(evt, runId) {
  if (evt) evt.stopPropagation();
  if (!confirm('确认删除该检测记录？此操作不可恢复。')) return;

  const {ok, data} = await api('DELETE', '/api/v1/runs/' + runId);
  if (!ok) {
    alert('删除失败: ' + (data.error || 'unknown error'));
    return;
  }
  loadRuns();
}

// ── Benchmarks ─────────────────────────────────────────────────────────────

async function markAsBaseline(runId) {
  const {ok, data} = await api('GET', '/api/v1/runs/' + runId);
  if (!ok) {
    alert('无法获取任务信息');
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
    alert(data2.error || '创建基准失败');
    return;
  }
  alert('已创建基准: ' + display_name);
  const {ok: ok3, data: data3} = await api('GET', '/api/v1/runs/' + runId);
  if (ok3) {
    renderTaskActions(runId, data3.status, data3.baseline_id);
  }
}

async function compareWithBaseline(runId) {
  const {ok, data} = await api('POST', '/api/v1/baselines/compare', { run_id: runId });
  if (!ok) {
    alert('基准对比失败: ' + (data.error || 'unknown error'));
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
          分数差异: Total ${fmtScore(d.total || 0)}, Capability ${fmtScore(d.capability || 0)},
          Authenticity ${fmtScore(d.authenticity || 0)}, Performance ${fmtScore(d.performance || 0)}<br>
          ${escHtml(data.verdict_reason || '')}
        </div>
        <table style="width:100%;border-collapse:collapse;font-size:12px">
          <thead>
            <tr style="background:var(--bg2)">
              <th style="padding:8px 12px;text-align:left;font-size:10px;color:var(--ink4)">Feature</th>
              <th style="padding:8px 12px;text-align:right;font-size:10px;color:var(--ink4)">Baseline</th>
              <th style="padding:8px 12px;text-align:right;font-size:10px;color:var(--ink4)">Current</th>
              <th style="padding:8px 12px;text-align:right;font-size:10px;color:var(--ink4)">Δ%</th>
            </tr>
          </thead>
          <tbody>${rows || '<tr><td colspan="4" style="padding:12px;text-align:center;color:var(--ink4)">No drift data</td></tr>'}</tbody>
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
              <button class="btn" style="padding:4px 10px;font-size:11px" onclick="viewBaselineReport('${b.id}')">查看报告</button>
              <button class="btn danger" style="padding:4px 10px;font-size:11px" onclick="deleteBaseline('${b.id}')">删除</button>
            </td>
          </tr>`).join('')}
        </tbody>
      </table>
    </div>`;

  document.getElementById('benchmarks-content').innerHTML = html;
}

async function viewBaselineReport(baselineId) {
  const {ok, data} = await api('GET', '/api/v1/baselines/' + baselineId);
  if (!ok) { alert('加载失败'); return; }

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
  if (!confirm('确认删除该基准模型？')) return;
  const {ok, data} = await api('DELETE', '/api/v1/baselines/' + baselineId);
  if (!ok) { alert('删除失败: ' + (data.error || 'unknown error')); return; }
  loadBaselines();
}

// ── Utilities ──────────────────────────────────────────────────────────────

function escHtml(s) {
  return String(s||'')
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

function fmtScore(v) {
  if (v == null || v === '' || v === undefined) return '–';
  const n = Number(v);
  if (isNaN(n)) return '–';
  return n.toFixed(1);
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
showPage('home');

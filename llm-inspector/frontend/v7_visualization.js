/**
 * V7 Frontend Visualization Module
 * 
 * Phase 4: UI/UX Enhancement - Advanced visualization features
 * - IRT Parameter Visualization
 * - Confidence Interval Visualization
 * - Score Traceability Display
 * - Real-time Test Progress Monitoring
 */

// ═════════════════════════════════════════════════════════════════════════════
// 9.1 IRT Parameter Visualization
// ═════════════════════════════════════════════════════════════════════════════

/**
 * Calculate Item Characteristic Curve (ICC) for 2PL IRT model
 * P(theta) = 1 / (1 + exp(-a * (theta - b)))
 */
function calculateICC(a, b, thetaRange = null) {
  if (!thetaRange) {
    thetaRange = [];
    for (let t = -3; t <= 3; t += 0.1) {
      thetaRange.push(t);
    }
  }
  
  return thetaRange.map(theta => ({
    theta: theta,
    probability: 1 / (1 + Math.exp(-a * (theta - b)))
  }));
}

/**
 * Calculate Information Function for 2PL IRT model
 * I(theta) = a^2 * P(theta) * (1 - P(theta))
 */
function calculateInformation(a, b, thetaRange = null) {
  if (!thetaRange) {
    thetaRange = [];
    for (let t = -3; t <= 3; t += 0.1) {
      thetaRange.push(t);
    }
  }
  
  return thetaRange.map(theta => {
    const p = 1 / (1 + Math.exp(-a * (theta - b)));
    const info = (a * a) * p * (1 - p);
    return { theta: theta, information: info };
  });
}

/**
 * Render IRT parameters with ICC and information function charts
 */
function renderIRTParameters(caseId, irtParams) {
  const { a, b, fit_rmse, info_max } = irtParams;
  
  // Calculate ICC data
  const iccData = calculateICC(a, b);
  const maxProb = Math.max(...iccData.map(d => d.probability));
  const minProb = Math.min(...iccData.map(d => d.probability));
  
  // Calculate Information function data
  const infoData = calculateInformation(a, b);
  const maxInfo = Math.max(...infoData.map(d => d.information));
  
  // Generate SVG for ICC
  const iccSvg = renderICCSVG(iccData, a, b);
  const infoSvg = renderInformationSVG(infoData, a, b);
  
  const aStatus = a > 1.0 ? 'good' : a > 0.5 ? 'warning' : 'poor';
  const aStatusText = a > 1.0 ? '✓ 优良' : a > 0.5 ? '⚠ 中等' : '✗ 偏低';
  
  const rmseStatus = fit_rmse < 0.05 ? 'good' : fit_rmse < 0.1 ? 'warning' : 'poor';
  const rmseStatusText = fit_rmse < 0.05 ? '✓ 优良' : fit_rmse < 0.1 ? '⚠ 可接受' : '✗ 需改进';
  
  return `
    <div class="irt-visualization">
      <h4>IRT 2PL 题目参数分析</h4>
      <div class="param-grid" style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:16px 0">
        <div class="param-card param-card-${aStatus}">
          <div class="param-label">区分度 (a)</div>
          <div class="param-value">${a.toFixed(2)}</div>
          <div class="param-status">${aStatusText}</div>
          <div class="param-hint">理想范围: 0.5-2.0</div>
        </div>
        <div class="param-card">
          <div class="param-label">难度 (b)</div>
          <div class="param-value">${b.toFixed(2)}</div>
          <div class="param-status">${b > 2 ? '困难' : b > 0 ? '中等' : '简单'}</div>
          <div class="param-hint">理想范围: -3 至 +3</div>
        </div>
        <div class="param-card param-card-${rmseStatus}">
          <div class="param-label">拟合度 (RMSE)</div>
          <div class="param-value">${fit_rmse.toFixed(3)}</div>
          <div class="param-status">${rmseStatusText}</div>
          <div class="param-hint">目标: <0.05, 可接受: <0.1</div>
        </div>
      </div>
      
      <div class="chart-container" style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
        <div class="chart-card">
          <div class="chart-title">题目特征曲线 (ICC)</div>
          <div class="chart-svg">${iccSvg}</div>
          <div class="chart-legend">
            <span style="color:#3b82f6">●</span> P(θ) - 答对概率
          </div>
        </div>
        <div class="chart-card">
          <div class="chart-title">信息函数</div>
          <div class="chart-svg">${infoSvg}</div>
          <div class="chart-legend">
            <span style="color:#10b981">●</span> I(θ) - 测量精度
          </div>
        </div>
      </div>
    </div>
  `;
}

/**
 * Render SVG for Item Characteristic Curve
 */
function renderICCSVG(data, a, b) {
  const width = 280;
  const height = 180;
  const padding = 30;
  
  const xScale = (theta) => padding + (theta + 3) / 6 * (width - 2 * padding);
  const yScale = (prob) => height - padding - prob * (height - 2 * padding);
  
  // Generate path
  let path = '';
  data.forEach((d, i) => {
    const x = xScale(d.theta);
    const y = yScale(d.probability);
    path += (i === 0 ? 'M' : 'L') + `${x},${y}`;
  });
  
  return `
    <svg viewBox="0 0 ${width} ${height}" style="width:100%;height:auto">
      <!-- Grid lines -->
      <g class="grid">
        ${[-3, -2, -1, 0, 1, 2, 3].map(t => `
          <line x1="${xScale(t)}" y1="${padding}" x2="${xScale(t)}" y2="${height-padding}" stroke="#e5e7eb" stroke-dasharray="2"/>
        `).join('')}
        ${[0, 0.25, 0.5, 0.75, 1].map(p => `
          <line x1="${padding}" y1="${yScale(p)}" x2="${width-padding}" y2="${yScale(p)}" stroke="#e5e7eb" stroke-dasharray="2"/>
        `).join('')}
      </g>
      
      <!-- Axes -->
      <line x1="${padding}" y1="${height-padding}" x2="${width-padding}" y2="${height-padding}" stroke="#6b7280" stroke-width="1.5"/>
      <line x1="${padding}" y1="${padding}" x2="${padding}" y2="${height-padding}" stroke="#6b7280" stroke-width="1.5"/>
      
      <!-- X-axis labels -->
      ${[-3, -2, -1, 0, 1, 2, 3].map(t => `
        <text x="${xScale(t)}" y="${height-padding+15}" text-anchor="middle" font-size="10" fill="#6b7280">${t}</text>
      `).join('')}
      
      <!-- Y-axis labels -->
      ${[0, 0.5, 1].map(p => `
        <text x="${padding-8}" y="${yScale(p)+4}" text-anchor="end" font-size="10" fill="#6b7280">${p}</text>
      `).join('')}
      
      <!-- ICC Curve -->
      <path d="${path}" fill="none" stroke="#3b82f6" stroke-width="2"/>
      
      <!-- Difficulty marker (b) -->
      <circle cx="${xScale(b)}" cy="${yScale(0.5)}" r="4" fill="#ef4444"/>
      <text x="${xScale(b)}" y="${yScale(0.5)-10}" text-anchor="middle" font-size="9" fill="#ef4444">b=${b.toFixed(2)}</text>
    </svg>
  `;
}

/**
 * Render SVG for Information Function
 */
function renderInformationSVG(data, a, b) {
  const width = 280;
  const height = 180;
  const padding = 30;
  
  const maxInfo = Math.max(...data.map(d => d.information));
  
  const xScale = (theta) => padding + (theta + 3) / 6 * (width - 2 * padding);
  const yScale = (info) => height - padding - (info / maxInfo) * (height - 2 * padding);
  
  // Generate path
  let path = '';
  data.forEach((d, i) => {
    const x = xScale(d.theta);
    const y = yScale(d.information);
    path += (i === 0 ? 'M' : 'L') + `${x},${y}`;
  });
  
  // Find theta with max information
  const maxInfoPoint = data.reduce((max, d) => d.information > max.information ? d : max, data[0]);
  
  return `
    <svg viewBox="0 0 ${width} ${height}" style="width:100%;height:auto">
      <!-- Grid lines -->
      <g class="grid">
        ${[-3, -2, -1, 0, 1, 2, 3].map(t => `
          <line x1="${xScale(t)}" y1="${padding}" x2="${xScale(t)}" y2="${height-padding}" stroke="#e5e7eb" stroke-dasharray="2"/>
        `).join('')}
      </g>
      
      <!-- Axes -->
      <line x1="${padding}" y1="${height-padding}" x2="${width-padding}" y2="${height-padding}" stroke="#6b7280" stroke-width="1.5"/>
      <line x1="${padding}" y1="${padding}" x2="${padding}" y2="${height-padding}" stroke="#6b7280" stroke-width="1.5"/>
      
      <!-- X-axis labels -->
      ${[-3, -2, -1, 0, 1, 2, 3].map(t => `
        <text x="${xScale(t)}" y="${height-padding+15}" text-anchor="middle" font-size="10" fill="#6b7280">${t}</text>
      `).join('')}
      
      <!-- Y-axis label -->
      <text x="${padding-8}" y="${padding+4}" text-anchor="end" font-size="10" fill="#6b7280">${maxInfo.toFixed(2)}</text>
      <text x="${padding-8}" y="${height-padding+4}" text-anchor="end" font-size="10" fill="#6b7280">0</text>
      
      <!-- Information Curve -->
      <path d="${path}" fill="none" stroke="#10b981" stroke-width="2"/>
      
      <!-- Max info marker -->
      <circle cx="${xScale(maxInfoPoint.theta)}" cy="${yScale(maxInfoPoint.information)}" r="4" fill="#f59e0b"/>
      <text x="${xScale(maxInfoPoint.theta)}" y="${yScale(maxInfoPoint.information)-10}" text-anchor="middle" font-size="9" fill="#f59e0b">
        Max @ θ=${maxInfoPoint.theta.toFixed(1)}
      </text>
    </svg>
  `;
}

// ═════════════════════════════════════════════════════════════════════════════
// 9.2 Confidence Interval Visualization
// ═════════════════════════════════════════════════════════════════════════════

/**
 * Render similarity score with confidence interval visualization
 */
function renderSimilarityWithCI(similarity) {
  const { benchmark, score, ci_low, ci_high, confidence_level } = similarity;
  
  const ciWidth = ci_high - ci_low;
  const ciColor = confidence_level === 'high' ? '#16a34a' : 
                  confidence_level === 'medium' ? '#f59e0b' : '#dc2626';
  
  const scorePercent = (score * 100).toFixed(1);
  const ciLowPercent = (ci_low * 100).toFixed(1);
  const ciHighPercent = (ci_high * 100).toFixed(1);
  const ciWidthPercent = (ciWidth * 100).toFixed(1);
  
  const badgeClass = confidence_level === 'high' ? 'badge-high' : 
                     confidence_level === 'medium' ? 'badge-medium' : 'badge-low';
  
  return `
    <div class="similarity-ci-item">
      <div class="similarity-header">
        <div class="benchmark-name">${escHtml(benchmark)}</div>
        <div class="confidence-badge ${badgeClass}">${confidence_level}</div>
      </div>
      <div class="similarity-bar-container">
        <div class="score-bar-bg">
          <div class="score-bar-fill" style="width: ${scorePercent}%; background: ${ciColor}"></div>
          <div class="ci-band" style="left: ${ciLowPercent}%; width: ${ciWidthPercent}%; background: ${ciColor}33"></div>
        </div>
      </div>
      <div class="similarity-details">
        <span class="score-value">${scorePercent}%</span>
        <span class="ci-range">[${ciLowPercent}%, ${ciHighPercent}%]</span>
        <span class="ci-width">区间宽度: ${ciWidthPercent}%</span>
      </div>
    </div>
  `;
}

/**
 * Render a list of similarity items with confidence intervals
 */
function renderSimilarityListWithCI(similarities) {
  if (!similarities || similarities.length === 0) {
    return '<div class="empty-state">暂无相似度数据</div>';
  }
  
  const items = similarities.map(sim => renderSimilarityWithCI(sim)).join('');
  
  return `
    <div class="similarity-ci-list">
      <div class="list-header">
        <span>基准模型</span>
        <span>置信度</span>
      </div>
      ${items}
      <div class="list-footer">
        <div class="legend">
          <span class="legend-item"><span class="dot" style="background:#16a34a"></span> 高置信 (≥75%)</span>
          <span class="legend-item"><span class="dot" style="background:#f59e0b"></span> 中置信 (55-75%)</span>
          <span class="legend-item"><span class="dot" style="background:#dc2626"></span> 低置信 (<55%)</span>
        </div>
      </div>
    </div>
  `;
}

// ═════════════════════════════════════════════════════════════════════════════
// 9.3 Score Traceability Display
// ═════════════════════════════════════════════════════════════════════════════

/**
 * Render score calculation trace with full transparency
 */
function renderScoreTrace(scorecard) {
  if (!scorecard) return '';
  
  const rawScores = scorecard.raw_scores || {};
  const weights = scorecard.weights || {};
  const irtA = scorecard.irt_a || {};
  const overallScore = scorecard.overall_score || 0;
  const weightSource = scorecard.weight_source || '人工设定';
  
  // Calculate weighted contribution for each dimension
  const contributions = {};
  Object.keys(rawScores).forEach(dim => {
    const score = rawScores[dim] || 0;
    const weight = weights[dim] || 0;
    contributions[dim] = score * weight;
  });
  
  const dimensionRows = Object.keys(rawScores).map(dim => {
    const score = rawScores[dim];
    const weight = weights[dim] || 0;
    const a = irtA[dim] || '-';
    const contribution = contributions[dim] || 0;
    
    return `
      <tr>
        <td>${escHtml(dim)}</td>
        <td>${score ? score.toFixed(1) : 'N/A'}</td>
        <td>${(weight * 100).toFixed(1)}%</td>
        <td>${typeof a === 'number' ? a.toFixed(2) : a}</td>
        <td>${contribution.toFixed(2)}</td>
      </tr>
    `;
  }).join('');
  
  // Build calculation formula
  const calcTerms = Object.keys(contributions).map(dim => {
    return `(${rawScores[dim].toFixed(1)} × ${(weights[dim] * 100).toFixed(1)}%)`;
  }).join(' + ');
  
  return `
    <div class="score-trace-container">
      <h4>评分计算溯源</h4>
      
      <div class="trace-step">
        <div class="step-header">
          <span class="step-number">1</span>
          <span class="step-title">原始维度分数</span>
        </div>
        <table class="trace-table">
          <thead>
            <tr>
              <th>维度</th>
              <th>原始分数</th>
              <th>权重</th>
              <th>IRT区分度(a)</th>
              <th>加权贡献</th>
            </tr>
          </thead>
          <tbody>
            ${dimensionRows}
          </tbody>
        </table>
      </div>
      
      <div class="trace-step">
        <div class="step-header">
          <span class="step-number">2</span>
          <span class="step-title">权重来源</span>
        </div>
        <div class="weight-source">
          <span class="source-label">计算方法:</span>
          <span class="source-value">${escHtml(weightSource)}</span>
        </div>
        <div class="weight-explanation">
          权重基于IRT信息函数积分计算，确保测量精度最大化。
          区分度(a)越高，该维度权重越大。
        </div>
      </div>
      
      <div class="trace-step">
        <div class="step-header">
          <span class="step-number">3</span>
          <span class="step-title">加权聚合计算</span>
        </div>
        <div class="calculation-formula">
          <div class="formula">总分 = Σ(维度分数 × 权重)</div>
          <div class="formula-detail">= ${calcTerms}</div>
          <div class="formula-result">= ${overallScore.toFixed(2)}</div>
        </div>
      </div>
      
      <div class="trace-step">
        <div class="step-header">
          <span class="step-number">4</span>
          <span class="step-title">最终评分</span>
        </div>
        <div class="final-score-display">
          <div class="score-value-large">${overallScore.toFixed(1)}</div>
          <div class="score-scale">/ 100</div>
        </div>
      </div>
    </div>
  `;
}

// ═════════════════════════════════════════════════════════════════════════════
// 9.3 Real-time Test Progress Monitoring
// ═════════════════════════════════════════════════════════════════════════════

/**
 * WebSocket-based real-time progress monitor
 */
class TestProgressMonitor {
  constructor(runId, options = {}) {
    this.runId = runId;
    this.options = {
      onLayerComplete: options.onLayerComplete || (() => {}),
      onCaseComplete: options.onCaseComplete || (() => {}),
      onAbilityUpdate: options.onAbilityUpdate || (() => {}),
      onEarlyStop: options.onEarlyStop || (() => {}),
      onComplete: options.onComplete || (() => {}),
      onError: options.onError || (() => {}),
    };
    this.ws = null;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.reconnectDelay = 1000;
  }
  
  connect() {
    const wsUrl = `${API.replace(/^http/, 'ws')}/api/v1/runs/${this.runId}/stream`;
    
    try {
      this.ws = new WebSocket(wsUrl);
      this.setupHandlers();
    } catch (error) {
      this.options.onError(error);
    }
  }
  
  setupHandlers() {
    this.ws.onopen = () => {
      console.log(`[ProgressMonitor] Connected to run ${this.runId}`);
      this.reconnectAttempts = 0;
    };
    
    this.ws.onmessage = (event) => {
      try {
        const update = JSON.parse(event.data);
        this.handleUpdate(update);
      } catch (error) {
        console.error('[ProgressMonitor] Failed to parse message:', error);
      }
    };
    
    this.ws.onclose = (event) => {
      if (!event.wasClean && this.reconnectAttempts < this.maxReconnectAttempts) {
        this.reconnectAttempts++;
        setTimeout(() => this.connect(), this.reconnectDelay * this.reconnectAttempts);
      }
    };
    
    this.ws.onerror = (error) => {
      this.options.onError(error);
    };
  }
  
  handleUpdate(update) {
    switch(update.type) {
      case 'layer_complete':
        this.options.onLayerComplete(update.layer, update.confidence, update.evidence);
        break;
        
      case 'case_complete':
        this.options.onCaseComplete(update.case_id, update.passed, update.dimension);
        break;
        
      case 'ability_update':
        this.options.onAbilityUpdate(update.theta, update.se, update.n_items);
        break;
        
      case 'early_stop':
        this.options.onEarlyStop(update.reason, update.final_theta);
        break;
        
      case 'complete':
        this.options.onComplete(update.result);
        this.disconnect();
        break;
        
      case 'error':
        this.options.onError(new Error(update.message));
        break;
    }
  }
  
  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}

/**
 * Render real-time progress UI component
 */
function renderRealtimeProgress(runId, options = {}) {
  const containerId = options.containerId || `progress-${runId}`;
  
  // Create container if it doesn't exist
  let container = document.getElementById(containerId);
  if (!container) {
    container = document.createElement('div');
    container.id = containerId;
    container.className = 'realtime-progress-container';
  }
  
  container.innerHTML = `
    <div class="progress-header">
      <span class="status-indicator">
        <span class="pulse-dot"></span>
        实时检测中
      </span>
      <span class="progress-stats" id="${containerId}-stats">准备中...</span>
    </div>
    
    <div class="ability-tracker" id="${containerId}-ability" style="display:none">
      <div class="ability-label">当前能力估计 (θ)</div>
      <div class="ability-value" id="${containerId}-theta">-</div>
      <div class="ability-precision">
        标准误: <span id="${containerId}-se">-</span> | 
        已测题目: <span id="${containerId}-items">0</span>
      </div>
      <div class="precision-progress">
        <div class="precision-bar" id="${containerId}-precision-bar"></div>
      </div>
    </div>
    
    <div class="layer-progress" id="${containerId}-layers">
      <div class="layer-list"></div>
    </div>
    
    <div class="case-progress" id="${containerId}-cases" style="display:none">
      <div class="case-grid"></div>
    </div>
    
    <div class="progress-log" id="${containerId}-log">
      <div class="log-entries"></div>
    </div>
  `;
  
  // Initialize monitor
  const monitor = new TestProgressMonitor(runId, {
    onLayerComplete: (layer, confidence, evidence) => {
      updateLayerProgress(containerId, layer, confidence, evidence);
    },
    onCaseComplete: (caseId, passed, dimension) => {
      updateCaseProgress(containerId, caseId, passed, dimension);
    },
    onAbilityUpdate: (theta, se, nItems) => {
      updateAbilityDisplay(containerId, theta, se, nItems);
    },
    onEarlyStop: (reason, finalTheta) => {
      addLogEntry(containerId, `提前停止: ${reason} (θ=${finalTheta.toFixed(2)})`);
    },
    onComplete: (result) => {
      markComplete(containerId, result);
    },
    onError: (error) => {
      markError(containerId, error);
    }
  });
  
  monitor.connect();
  
  // Store monitor reference for cleanup
  container._monitor = monitor;
  
  return container;
}

/**
 * Update layer progress display
 */
function updateLayerProgress(containerId, layer, confidence, evidence) {
  const layersContainer = document.getElementById(`${containerId}-layers`);
  const layerList = layersContainer.querySelector('.layer-list');
  
  const layerItem = document.createElement('div');
  layerItem.className = 'layer-item';
  layerItem.innerHTML = `
    <div class="layer-status complete">✓</div>
    <div class="layer-info">
      <div class="layer-name">${escHtml(layer)}</div>
      <div class="layer-confidence">置信度: ${(confidence * 100).toFixed(1)}%</div>
    </div>
  `;
  
  layerList.appendChild(layerItem);
  
  // Update stats
  updateProgressStats(containerId);
}

/**
 * Update case progress display
 */
function updateCaseProgress(containerId, caseId, passed, dimension) {
  const casesContainer = document.getElementById(`${containerId}-cases`);
  const caseGrid = casesContainer.querySelector('.case-grid');
  
  casesContainer.style.display = 'block';
  
  const caseItem = document.createElement('div');
  caseItem.className = `case-item ${passed ? 'passed' : 'failed'}`;
  caseItem.title = `${caseId} (${dimension})`;
  caseItem.innerHTML = passed ? '✓' : '✗';
  
  caseGrid.appendChild(caseItem);
}

/**
 * Update ability display (for CAT - Computerized Adaptive Testing)
 */
function updateAbilityDisplay(containerId, theta, se, nItems) {
  const abilityContainer = document.getElementById(`${containerId}-ability`);
  abilityContainer.style.display = 'block';
  
  document.getElementById(`${containerId}-theta`).textContent = theta.toFixed(2);
  document.getElementById(`${containerId}-se`).textContent = se.toFixed(3);
  document.getElementById(`${containerId}-items`).textContent = nItems;
  
  // Update precision progress bar
  // Target SE is 0.3, progress = 1 - (se / 0.3)
  const progress = Math.max(0, Math.min(1, 1 - se / 0.3));
  const bar = document.getElementById(`${containerId}-precision-bar`);
  bar.style.width = `${progress * 100}%`;
  bar.style.background = progress > 0.8 ? '#16a34a' : progress > 0.5 ? '#f59e0b' : '#dc2626';
}

/**
 * Add log entry
 */
function addLogEntry(containerId, message) {
  const logContainer = document.getElementById(`${containerId}-log`);
  const logEntries = logContainer.querySelector('.log-entries');
  
  const entry = document.createElement('div');
  entry.className = 'log-entry';
  const time = new Date().toLocaleTimeString();
  entry.innerHTML = `<span class="log-time">${time}</span> ${escHtml(message)}`;
  
  logEntries.appendChild(entry);
  logEntries.scrollTop = logEntries.scrollHeight;
}

/**
 * Update progress stats
 */
function updateProgressStats(containerId) {
  const stats = document.getElementById(`${containerId}-stats`);
  const layersContainer = document.getElementById(`${containerId}-layers`);
  const completedLayers = layersContainer.querySelectorAll('.layer-item').length;
  
  stats.textContent = `已完成 ${completedLayers} 个检测层`;
}

/**
 * Mark progress as complete
 */
function markComplete(containerId, result) {
  const container = document.getElementById(containerId);
  const indicator = container.querySelector('.status-indicator');
  indicator.innerHTML = '<span class="complete-dot"></span> 检测完成';
  indicator.classList.add('complete');
}

/**
 * Mark progress as error
 */
function markError(containerId, error) {
  const container = document.getElementById(containerId);
  const indicator = container.querySelector('.status-indicator');
  indicator.innerHTML = '<span class="error-dot"></span> 检测出错';
  indicator.classList.add('error');
  
  addLogEntry(containerId, `错误: ${error.message}`);
}

// ═════════════════════════════════════════════════════════════════════════════
// Helper Functions
// ═════════════════════════════════════════════════════════════════════════════

function escHtml(s) {
  return String(s || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ═════════════════════════════════════════════════════════════════════════════
// CSS Styles (to be added to styles.css)
// ═════════════════════════════════════════════════════════════════════════════

const V7_STYLES = `
/* IRT Parameter Visualization */
.irt-visualization {
  background: var(--bg1);
  border-radius: 8px;
  padding: 16px;
  margin: 16px 0;
}

.irt-visualization h4 {
  margin: 0 0 16px 0;
  color: var(--ink1);
  font-size: 16px;
}

.param-card {
  background: var(--bg);
  border-radius: 6px;
  padding: 12px;
  text-align: center;
  border: 1px solid var(--rule);
}

.param-card-good {
  border-color: #16a34a;
  background: #f0fdf4;
}

.param-card-warning {
  border-color: #f59e0b;
  background: #fffbeb;
}

.param-card-poor {
  border-color: #dc2626;
  background: #fef2f2;
}

.param-label {
  font-size: 11px;
  color: var(--ink3);
  margin-bottom: 4px;
}

.param-value {
  font-size: 20px;
  font-weight: 600;
  color: var(--ink1);
}

.param-status {
  font-size: 11px;
  margin-top: 4px;
}

.param-hint {
  font-size: 10px;
  color: var(--ink4);
  margin-top: 4px;
}

.chart-container {
  margin-top: 16px;
}

.chart-card {
  background: var(--bg);
  border-radius: 6px;
  padding: 12px;
  border: 1px solid var(--rule);
}

.chart-title {
  font-size: 12px;
  font-weight: 600;
  color: var(--ink2);
  margin-bottom: 8px;
}

/* Similarity with Confidence Intervals */
.similarity-ci-list {
  background: var(--bg1);
  border-radius: 8px;
  padding: 16px;
}

.similarity-ci-item {
  padding: 12px;
  border-bottom: 1px solid var(--rule);
}

.similarity-ci-item:last-child {
  border-bottom: none;
}

.similarity-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.benchmark-name {
  font-weight: 500;
  color: var(--ink1);
}

.confidence-badge {
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 500;
}

.badge-high {
  background: #dcfce7;
  color: #166534;
}

.badge-medium {
  background: #fef3c7;
  color: #92400e;
}

.badge-low {
  background: #fee2e2;
  color: #991b1b;
}

.similarity-bar-container {
  margin: 8px 0;
}

.score-bar-bg {
  position: relative;
  height: 12px;
  background: var(--rule);
  border-radius: 6px;
  overflow: hidden;
}

.score-bar-fill {
  position: absolute;
  height: 100%;
  border-radius: 6px;
  transition: width 0.3s ease;
}

.ci-band {
  position: absolute;
  height: 100%;
  opacity: 0.3;
  border-radius: 6px;
}

.similarity-details {
  display: flex;
  gap: 12px;
  font-size: 11px;
  color: var(--ink3);
  margin-top: 4px;
}

.score-value {
  font-weight: 600;
  color: var(--ink1);
}

/* Score Trace */
.score-trace-container {
  background: var(--bg1);
  border-radius: 8px;
  padding: 20px;
  margin: 16px 0;
}

.score-trace-container h4 {
  margin: 0 0 20px 0;
  color: var(--ink1);
  font-size: 16px;
}

.trace-step {
  margin-bottom: 20px;
  padding-bottom: 20px;
  border-bottom: 1px solid var(--rule);
}

.trace-step:last-child {
  border-bottom: none;
  margin-bottom: 0;
  padding-bottom: 0;
}

.step-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 12px;
}

.step-number {
  width: 24px;
  height: 24px;
  background: var(--accent);
  color: white;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-weight: 600;
}

.step-title {
  font-weight: 600;
  color: var(--ink1);
}

.trace-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
}

.trace-table th,
.trace-table td {
  padding: 8px;
  text-align: left;
  border-bottom: 1px solid var(--rule);
}

.trace-table th {
  font-weight: 600;
  color: var(--ink2);
  background: var(--bg);
}

.weight-source {
  display: flex;
  gap: 8px;
  font-size: 12px;
  margin-bottom: 8px;
}

.source-label {
  color: var(--ink3);
}

.source-value {
  color: var(--ink1);
  font-weight: 500;
}

.weight-explanation {
  font-size: 11px;
  color: var(--ink4);
  line-height: 1.5;
}

.calculation-formula {
  background: var(--bg);
  border-radius: 6px;
  padding: 12px;
  font-family: monospace;
  font-size: 12px;
}

.formula {
  color: var(--ink2);
  margin-bottom: 8px;
}

.formula-detail {
  color: var(--ink3);
  margin-bottom: 8px;
  word-break: break-all;
}

.formula-result {
  color: var(--accent);
  font-weight: 600;
  font-size: 14px;
}

.final-score-display {
  display: flex;
  align-items: baseline;
  gap: 8px;
}

.score-value-large {
  font-size: 36px;
  font-weight: 700;
  color: var(--accent);
}

.score-scale {
  font-size: 16px;
  color: var(--ink3);
}

/* Real-time Progress */
.realtime-progress-container {
  background: var(--bg1);
  border-radius: 8px;
  padding: 16px;
  margin: 16px 0;
}

.progress-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.status-indicator {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
  font-weight: 500;
  color: var(--ink2);
}

.pulse-dot {
  width: 8px;
  height: 8px;
  background: #22c55e;
  border-radius: 50%;
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.complete-dot {
  width: 8px;
  height: 8px;
  background: #16a34a;
  border-radius: 50%;
}

.error-dot {
  width: 8px;
  height: 8px;
  background: #dc2626;
  border-radius: 50%;
}

.progress-stats {
  font-size: 12px;
  color: var(--ink3);
}

.ability-tracker {
  background: var(--bg);
  border-radius: 6px;
  padding: 12px;
  margin-bottom: 16px;
}

.ability-label {
  font-size: 11px;
  color: var(--ink3);
  margin-bottom: 4px;
}

.ability-value {
  font-size: 24px;
  font-weight: 700;
  color: var(--accent);
}

.ability-precision {
  font-size: 11px;
  color: var(--ink4);
  margin-top: 4px;
}

.precision-progress {
  height: 6px;
  background: var(--rule);
  border-radius: 3px;
  margin-top: 8px;
  overflow: hidden;
}

.precision-bar {
  height: 100%;
  background: #22c55e;
  border-radius: 3px;
  transition: width 0.3s ease;
}

.layer-progress {
  margin-bottom: 16px;
}

.layer-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.layer-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px;
  background: var(--bg);
  border-radius: 4px;
  font-size: 12px;
}

.layer-status {
  width: 20px;
  height: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  font-size: 11px;
}

.layer-status.complete {
  background: #dcfce7;
  color: #166534;
}

.layer-name {
  font-weight: 500;
  color: var(--ink1);
}

.layer-confidence {
  font-size: 11px;
  color: var(--ink4);
  margin-left: auto;
}

.case-progress {
  margin-bottom: 16px;
}

.case-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(24px, 1fr));
  gap: 4px;
}

.case-item {
  width: 24px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 4px;
  font-size: 11px;
}

.case-item.passed {
  background: #dcfce7;
  color: #166534;
}

.case-item.failed {
  background: #fee2e2;
  color: #991b1b;
}

.progress-log {
  max-height: 200px;
  overflow-y: auto;
  background: var(--bg);
  border-radius: 6px;
  padding: 12px;
}

.log-entries {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.log-entry {
  font-size: 11px;
  color: var(--ink3);
}

.log-time {
  color: var(--ink4);
  margin-right: 8px;
}
`;

// Export functions for use in app.js
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    renderIRTParameters,
    renderSimilarityWithCI,
    renderSimilarityListWithCI,
    renderScoreTrace,
    TestProgressMonitor,
    renderRealtimeProgress,
    calculateICC,
    calculateInformation,
    V7_STYLES
  };
}

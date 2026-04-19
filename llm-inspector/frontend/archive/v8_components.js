/**
 * v8.0 Phase 5 Frontend Components
 * 
 * Features:
 * - Data Provenance Visualization (数据来源可视化)
 * - Confidence Interval Display (置信度区间显示)
 * - Real-time Judgment Process Display (实时判题过程展示)
 * 
 * Reference: V8_UPGRADE_PLAN.md Phase 5
 */

// ═══════════════════════════════════════════════════════════════
// 1. Data Provenance Visualization (数据来源可视化)
// ═══════════════════════════════════════════════════════════════

/**
 * Render data provenance badge showing source of data
 * @param {Object} provenance - Provenance data with source, confidence, etc.
 * @returns {string} HTML string
 */
function renderProvenanceBadge(provenance) {
  if (!provenance) return '';
  
  const source = provenance.source || 'unknown';
  const confidence = provenance.confidence || 0;
  const timestamp = provenance.timestamp || '';
  
  // Source type icons and colors
  const sourceStyles = {
    'irt_calibration': { icon: '📊', color: '#1a4a8a', label: 'IRT校准' },
    'expert_annotation': { icon: '👤', color: '#1a6040', label: '专家标注' },
    'literature': { icon: '📚', color: '#7a4a00', label: '文献参考' },
    'default': { icon: '⚙️', color: '#6b6b66', label: '默认值' },
    'fallback': { icon: '⚠️', color: '#8a1a1a', label: '降级值' }
  };
  
  const style = sourceStyles[source] || sourceStyles['default'];
  const confPercent = Math.round(confidence * 100);
  
  return `
    <div class="provenance-badge" style="
      display: inline-flex;
      align-items: center;
      gap: 4px;
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 11px;
      background: ${style.color}15;
      border: 1px solid ${style.color}40;
      color: ${style.color};
    " title="来源: ${source} | 置信度: ${confPercent}% | 时间: ${timestamp.slice(0, 19) || 'N/A'}">
      <span>${style.icon}</span>
      <span>${style.label}</span>
      ${confidence > 0 ? `<span style="opacity: 0.7">(${confPercent}%)</span>` : ''}
    </div>
  `;
}

/**
 * Render threshold source info with DOI/URL reference
 * @param {Object} thresholdInfo - Threshold information
 * @returns {string} HTML string
 */
function renderThresholdSource(thresholdInfo) {
  if (!thresholdInfo) return '';
  
  const { value, source, doi, url } = thresholdInfo;
  
  let reference = '';
  if (doi) {
    reference = `<a href="https://doi.org/${doi}" target="_blank" class="ref-link">DOI: ${doi}</a>`;
  } else if (url) {
    reference = `<a href="url" target="_blank" class="ref-link">参考链接</a>`;
  }
  
  return `
    <div class="threshold-source" style="
      font-size: 11px;
      color: var(--ink4);
      margin-top: 4px;
      display: flex;
      align-items: center;
      gap: 8px;
    ">
      <span>阈值: ${value}</span>
      <span style="opacity: 0.6">•</span>
      <span>来源: ${source}</span>
      ${reference ? `<span style="opacity: 0.6">•</span>${reference}` : ''}
    </div>
  `;
}

/**
 * Render data lineage tree showing the full chain
 * @param {Object} lineage - Data lineage object
 * @returns {string} HTML string
 */
function renderDataLineage(lineage) {
  if (!lineage || !lineage.steps) return '';
  
  const steps = lineage.steps.map((step, idx) => `
    <div class="lineage-step" style="
      display: flex;
      align-items: flex-start;
      gap: 8px;
      padding: 8px 0;
      border-left: 2px solid ${idx === lineage.steps.length - 1 ? 'var(--green)' : 'var(--rule)'};
      margin-left: 8px;
      padding-left: 12px;
    ">
      <div class="step-marker" style="
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: ${idx === lineage.steps.length - 1 ? 'var(--green)' : 'var(--rule)'};
        margin-left: -17px;
        margin-top: 4px;
        flex-shrink: 0;
      "></div>
      <div style="flex: 1">
        <div style="font-size: 12px; font-weight: 500; color: var(--ink2)">${step.operation}</div>
        <div style="font-size: 11px; color: var(--ink4); margin-top: 2px">${step.description || ''}</div>
        ${step.timestamp ? `<div style="font-size: 10px; color: var(--ink4); opacity: 0.7; margin-top: 2px">${step.timestamp.slice(0, 19)}</div>` : ''}
      </div>
      ${step.provenance ? renderProvenanceBadge(step.provenance) : ''}
    </div>
  `).join('');
  
  return `
    <div class="data-lineage" style="margin-top: 12px;">
      <div style="font-size: 11px; font-weight: 600; color: var(--ink3); margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.05em">数据血缘追踪</div>
      <div class="lineage-tree">${steps}</div>
    </div>
  `;
}

// ═══════════════════════════════════════════════════════════════
// 2. Confidence Interval Display (置信度区间显示)
// ═══════════════════════════════════════════════════════════════

/**
 * Render confidence interval bar with visual representation
 * @param {Object} ci - Confidence interval data {value, lower, upper, confidence}
 * @param {Object} options - Display options
 * @returns {string} HTML string
 */
function renderConfidenceInterval(ci, options = {}) {
  if (!ci || ci.value === undefined) return '';
  
  const { value, lower, upper, confidence = 0.95 } = ci;
  const { showLabel = true, colorScale = true } = options;
  
  // Calculate display range (0-100%)
  const range = 100;
  const center = Math.max(0, Math.min(100, value * 100));
  const ciLower = lower !== undefined ? Math.max(0, Math.min(100, lower * 100)) : center - 10;
  const ciUpper = upper !== undefined ? Math.max(0, Math.min(100, upper * 100)) : center + 10;
  
  // Color based on confidence level
  let ciColor = '#1a4a8a';
  if (colorScale) {
    if (confidence >= 0.95) ciColor = '#1a6040';  // High confidence - green
    else if (confidence >= 0.90) ciColor = '#1a4a8a';  // Medium - blue
    else if (confidence >= 0.80) ciColor = '#7a4a00';  // Low-medium - amber
    else ciColor = '#8a1a1a';  // Low - red
  }
  
  const confidencePercent = Math.round(confidence * 100);
  
  return `
    <div class="confidence-interval-display" style="margin: 8px 0;">
      ${showLabel ? `
        <div style="display: flex; justify-content: space-between; font-size: 11px; color: var(--ink4); margin-bottom: 4px;">
          <span>置信区间 (${confidencePercent}% CI)</span>
          <span>${ciLower.toFixed(1)}% - ${ciUpper.toFixed(1)}%</span>
        </div>
      ` : ''}
      <div style="
        position: relative;
        height: 24px;
        background: var(--bg2);
        border-radius: 4px;
        overflow: hidden;
      ">
        <!-- Confidence interval range -->
        <div style="
          position: absolute;
          left: ${ciLower}%;
          right: ${100 - ciUpper}%;
          top: 4px;
          bottom: 4px;
          background: ${ciColor}20;
          border-radius: 2px;
          border: 1px solid ${ciColor}40;
        "></div>
        <!-- Center point -->
        <div style="
          position: absolute;
          left: ${center}%;
          top: 50%;
          transform: translate(-50%, -50%);
          width: 8px;
          height: 8px;
          background: ${ciColor};
          border-radius: 50%;
          border: 2px solid #fff;
          box-shadow: 0 1px 2px rgba(0,0,0,0.2);
        "></div>
        <!-- Value label -->
        <div style="
          position: absolute;
          left: ${center}%;
          top: -16px;
          transform: translateX(-50%);
          font-size: 11px;
          font-weight: 600;
          color: ${ciColor};
          font-family: var(--mono);
        ">${center.toFixed(1)}%</div>
      </div>
    </div>
  `;
}

/**
 * Render score card with confidence interval
 * @param {string} label - Score label
 * @param {Object} scoreData - Score data with value and confidence interval
 * @param {string} grade - Quality grade (A, B, C, etc.)
 * @returns {string} HTML string
 */
function renderScoreWithCI(label, scoreData, grade) {
  const value = typeof scoreData === 'number' ? scoreData : (scoreData?.value || 0);
  const ci = typeof scoreData === 'object' ? scoreData : null;
  
  const gradeColors = {
    'A': '#1a6040', 'B': '#1a4a8a', 'C': '#7a4a00', 'D': '#8a1a1a', 'F': '#8a1a1a'
  };
  const gradeColor = gradeColors[grade] || 'var(--ink)';
  
  return `
    <div class="score-card-ci" style="
      background: var(--bg2);
      border-radius: var(--radius);
      padding: 16px;
      text-align: center;
    ">
      <div style="font-size: 11px; color: var(--ink4); margin-bottom: 4px;">${label}</div>
      <div style="
        font-family: var(--mono);
        font-size: 28px;
        font-weight: 300;
        color: ${gradeColor};
        white-space: nowrap;
      ">${(value * 100).toFixed(1)}%</div>
      ${grade ? `
        <div style="
          display: inline-block;
          font-size: 11px;
          font-weight: 600;
          color: ${gradeColor};
          background: ${gradeColor}15;
          padding: 2px 8px;
          border-radius: 4px;
          margin-top: 4px;
        ">等级 ${grade}</div>
      ` : ''}
      ${ci ? renderConfidenceInterval(ci, { showLabel: false }) : ''}
    </div>
  `;
}

// ═══════════════════════════════════════════════════════════════
// 3. Real-time Judgment Process Display (实时判题过程展示)
// ═══════════════════════════════════════════════════════════════

/**
 * Judgment process visualizer component
 */
class JudgmentProcessVisualizer {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.steps = [];
    this.currentStep = -1;
    this.isComplete = false;
  }
  
  /**
   * Initialize with judgment steps
   * @param {Array} steps - Array of step definitions
   */
  init(steps) {
    this.steps = steps.map((step, idx) => ({
      id: step.id || `step_${idx}`,
      name: step.name,
      description: step.description,
      status: 'pending', // pending, running, complete, error
      result: null,
      startTime: null,
      endTime: null
    }));
    this.render();
  }
  
  /**
   * Start a step
   * @param {string} stepId - Step identifier
   */
  startStep(stepId) {
    const step = this.steps.find(s => s.id === stepId);
    if (!step) return;
    
    step.status = 'running';
    step.startTime = Date.now();
    this.currentStep = this.steps.indexOf(step);
    this.render();
  }
  
  /**
   * Complete a step with result
   * @param {string} stepId - Step identifier
   * @param {Object} result - Step result
   */
  completeStep(stepId, result) {
    const step = this.steps.find(s => s.id === stepId);
    if (!step) return;
    
    step.status = 'complete';
    step.result = result;
    step.endTime = Date.now();
    this.render();
  }
  
  /**
   * Mark step as error
   * @param {string} stepId - Step identifier
   * @param {string} error - Error message
   */
  errorStep(stepId, error) {
    const step = this.steps.find(s => s.id === stepId);
    if (!step) return;
    
    step.status = 'error';
    step.error = error;
    step.endTime = Date.now();
    this.render();
  }
  
  /**
   * Update step with progress data
   * @param {string} stepId - Step identifier
   * @param {Object} data - Progress data
   */
  updateStep(stepId, data) {
    const step = this.steps.find(s => s.id === stepId);
    if (!step) return;
    
    step.progress = data;
    this.render();
  }
  
  /**
   * Mark entire process as complete
   */
  complete() {
    this.isComplete = true;
    this.render();
  }
  
  /**
   * Render the visualizer
   */
  render() {
    if (!this.container) return;
    
    const stepElements = this.steps.map((step, idx) => {
      const statusConfig = {
        'pending': { icon: '○', color: 'var(--ink4)', bg: 'transparent' },
        'running': { icon: '◐', color: '#e8a068', bg: '#fdf5e0' },
        'complete': { icon: '●', color: '#1a6040', bg: '#eaf5ee' },
        'error': { icon: '✕', color: '#8a1a1a', bg: '#faeaea' }
      };
      
      const config = statusConfig[step.status];
      const isLast = idx === this.steps.length - 1;
      const showConnector = !isLast;
      
      let resultHtml = '';
      if (step.result) {
        const passed = step.result.passed;
        const confidence = step.result.confidence || 1;
        resultHtml = `
          <div style="
            margin-top: 6px;
            font-size: 11px;
            display: flex;
            align-items: center;
            gap: 8px;
          ">
            <span style="color: ${passed ? '#1a6040' : '#8a1a1a'}">${passed ? '✓ 通过' : '✗ 未通过'}</span>
            ${confidence < 1 ? `<span style="color: var(--ink4)">置信度: ${(confidence * 100).toFixed(0)}%</span>` : ''}
          </div>
        `;
      }
      
      let progressHtml = '';
      if (step.status === 'running' && step.progress) {
        progressHtml = `
          <div style="margin-top: 6px;">
            <div style="
              height: 3px;
              background: var(--rule);
              border-radius: 2px;
              overflow: hidden;
            ">
              <div style="
                height: 100%;
                width: ${step.progress.percent || 0}%;
                background: #e8a068;
                transition: width 0.3s;
              "></div>
            </div>
            <div style="font-size: 10px; color: var(--ink4); margin-top: 2px;">${step.progress.message || '处理中...'}</div>
          </div>
        `;
      }
      
      return `
        <div class="judgment-step" style="
          display: flex;
          align-items: flex-start;
          gap: 12px;
          padding: 10px 0;
        ">
          <div style="
            width: 24px;
            height: 24px;
            border-radius: 50%;
            background: ${config.bg};
            border: 2px solid ${config.color};
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            color: ${config.color};
            flex-shrink: 0;
            ${step.status === 'running' ? 'animation: pulse 1.5s infinite' : ''}
          ">${config.icon}</div>
          <div style="flex: 1; min-width: 0;">
            <div style="font-size: 13px; font-weight: 500; color: var(--ink2)">${step.name}</div>
            <div style="font-size: 11px; color: var(--ink4); margin-top: 2px">${step.description}</div>
            ${resultHtml}
            ${progressHtml}
          </div>
          ${showConnector ? `
            <div style="
              position: absolute;
              left: 11px;
              top: 34px;
              bottom: -10px;
              width: 2px;
              background: ${step.status === 'complete' ? '#1a6040' : 'var(--rule)'};
            "></div>
          ` : ''}
        </div>
      `;
    }).join('');
    
    this.container.innerHTML = `
      <div class="judgment-process-viz" style="
        background: var(--card);
        border: 1px solid var(--rule);
        border-radius: var(--radius-lg);
        padding: 16px;
      ">
        <div style="
          font-size: 12px;
          font-weight: 600;
          color: var(--ink3);
          margin-bottom: 12px;
          text-transform: uppercase;
          letter-spacing: 0.05em;
        ">判题过程</div>
        <div style="position: relative;">${stepElements}</div>
        ${this.isComplete ? `
          <div style="
            margin-top: 12px;
            padding-top: 12px;
            border-top: 1px solid var(--rule);
            font-size: 12px;
            color: #1a6040;
            text-align: center;
          ">✓ 判题完成</div>
        ` : ''}
      </div>
    `;
  }
}

/**
 * Real-time judgment log stream component
 */
class JudgmentLogStream {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.logs = [];
    this.maxLogs = 100;
  }
  
  /**
   * Add a log entry
   * @param {Object} entry - Log entry
   */
  addLog(entry) {
    this.logs.push({
      timestamp: new Date().toISOString(),
      ...entry
    });
    
    if (this.logs.length > this.maxLogs) {
      this.logs.shift();
    }
    
    this.render();
  }
  
  /**
   * Add judgment step log
   * @param {string} step - Step name
   * @param {Object} data - Step data
   */
  addStepLog(step, data) {
    this.addLog({
      type: 'step',
      step,
      ...data
    });
  }
  
  /**
   * Add threshold application log
   * @param {string} threshold - Threshold name
   * @param {number} value - Threshold value
   * @param {string} source - Threshold source
   */
  addThresholdLog(threshold, value, source) {
    this.addLog({
      type: 'threshold',
      threshold,
      value,
      source
    });
  }
  
  /**
   * Render the log stream
   */
  render() {
    if (!this.container) return;
    
    const logElements = this.logs.slice(-20).map(log => {
      const time = log.timestamp.slice(11, 19);
      
      if (log.type === 'step') {
        return `
          <div style="
            padding: 6px 0;
            border-bottom: 1px solid var(--rule);
            font-size: 11px;
          ">
            <span style="color: var(--ink4); font-family: var(--mono)">[${time}]</span>
            <span style="color: var(--ink3); margin-left: 8px">${log.step}</span>
            ${log.input ? `<div style="color: var(--ink4); margin-left: 52px; margin-top: 2px">输入: ${JSON.stringify(log.input).slice(0, 80)}</div>` : ''}
            ${log.output ? `<div style="color: #1a4a8a; margin-left: 52px; margin-top: 2px">输出: ${JSON.stringify(log.output).slice(0, 80)}</div>` : ''}
          </div>
        `;
      }
      
      if (log.type === 'threshold') {
        return `
          <div style="
            padding: 6px 0;
            border-bottom: 1px solid var(--rule);
            font-size: 11px;
          ">
            <span style="color: var(--ink4); font-family: var(--mono)">[${time}]</span>
            <span style="color: #7a4a00; margin-left: 8px">⚙️ 阈值应用</span>
            <span style="color: var(--ink3); margin-left: 4px">${log.threshold} = ${log.value}</span>
            <span style="color: var(--ink4); margin-left: 4px">(${log.source})</span>
          </div>
        `;
      }
      
      return `
        <div style="
          padding: 6px 0;
          border-bottom: 1px solid var(--rule);
          font-size: 11px;
          color: var(--ink3);
        ">
          <span style="color: var(--ink4); font-family: var(--mono)">[${time}]</span>
          <span style="margin-left: 8px">${log.message || ''}</span>
        </div>
      `;
    }).join('');
    
    this.container.innerHTML = `
      <div class="judgment-log-stream" style="
        background: var(--card);
        border: 1px solid var(--rule);
        border-radius: var(--radius-lg);
        padding: 12px 16px;
        max-height: 300px;
        overflow-y: auto;
      ">
        <div style="
          font-size: 11px;
          font-weight: 600;
          color: var(--ink3);
          margin-bottom: 8px;
          text-transform: uppercase;
          letter-spacing: 0.05em;
        ">判题日志</div>
        ${logElements || '<div style="color: var(--ink4); font-size: 11px; padding: 12px 0;">等待判题开始...</div>'}
      </div>
    `;
    
    // Auto-scroll to bottom
    this.container.scrollTop = this.container.scrollHeight;
  }
}

// ═══════════════════════════════════════════════════════════════
// 4. Enhanced Report Components (增强的报告组件)
// ═══════════════════════════════════════════════════════════════

/**
 * Render case result with full v8 provenance information
 * @param {Object} caseResult - Case result data
 * @param {Object} options - Display options
 * @returns {string} HTML string
 */
function renderV8CaseResult(caseResult, options = {}) {
  const { expanded = false } = options;
  const hasProvenance = caseResult.weight_provenance || caseResult.threshold_source;
  
  const baseHtml = `
    <div class="case-result-v8" style="
      background: var(--card);
      border: 1px solid var(--rule);
      border-radius: var(--radius);
      margin-bottom: 8px;
      overflow: hidden;
    ">
      <div class="case-header-v8" style="
        padding: 12px 16px;
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 10px;
        background: ${caseResult.passed === false ? '#faeaea' : (caseResult.passed === true ? '#eaf5ee' : 'var(--card)')};
      ">
        <span style="
          width: 20px;
          height: 20px;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 12px;
          background: ${caseResult.passed === true ? '#1a6040' : (caseResult.passed === false ? '#8a1a1a' : 'var(--ink4)')};
          color: #fff;
        ">${caseResult.passed === true ? '✓' : (caseResult.passed === false ? '✗' : '?')}</span>
        <div style="flex: 1; min-width: 0;">
          <div style="font-size: 13px; font-weight: 500; color: var(--ink2)">${caseResult.case_name || caseResult.case_id || 'Unknown Case'}</div>
          <div style="font-size: 11px; color: var(--ink4); margin-top: 2px">方法: ${caseResult.judge_method || 'unknown'}</div>
        </div>
        ${caseResult.confidence !== undefined ? `
          <div style="text-align: right;">
            <div style="font-size: 12px; font-family: var(--mono); color: var(--ink2)">${(caseResult.confidence * 100).toFixed(0)}%</div>
            <div style="font-size: 10px; color: var(--ink4)">置信度</div>
          </div>
        ` : ''}
      </div>
      
      ${expanded ? `
        <div class="case-body-v8" style="
          padding: 12px 16px;
          border-top: 1px solid var(--rule);
          background: var(--bg);
        ">
          ${caseResult.threshold_source ? renderThresholdSource({
            value: caseResult.threshold_value,
            source: caseResult.threshold_source,
            doi: caseResult.threshold_doi,
            url: caseResult.threshold_url
          }) : ''}
          
          ${caseResult.detail ? `
            <div style="margin-top: 12px;">
              <div style="font-size: 11px; font-weight: 600; color: var(--ink3); margin-bottom: 6px;">判题详情</div>
              <pre style="
                font-family: var(--mono);
                font-size: 11px;
                background: #1a1a1a;
                color: #d4d0c8;
                padding: 10px 12px;
                border-radius: 4px;
                overflow-x: auto;
                max-height: 120px;
                overflow-y: auto;
              ">${JSON.stringify(caseResult.detail, null, 2)}</pre>
            </div>
          ` : ''}
          
          ${caseResult.data_lineage ? renderDataLineage(caseResult.data_lineage) : ''}
        </div>
      ` : ''}
    </div>
  `;
  
  return baseHtml;
}

// ═══════════════════════════════════════════════════════════════
// 5. Export for global access
// ═══════════════════════════════════════════════════════════════

// Expose to global scope for use in app.js
window.v8Components = {
  // Provenance
  renderProvenanceBadge,
  renderThresholdSource,
  renderDataLineage,
  
  // Confidence Intervals
  renderConfidenceInterval,
  renderScoreWithCI,
  
  // Real-time judgment
  JudgmentProcessVisualizer,
  JudgmentLogStream,
  
  // Enhanced reports
  renderV8CaseResult
};

// Also export as ES modules for modern usage
if (typeof module !== 'undefined' && module.exports) {
  module.exports = window.v8Components;
}

/**
 * LLM Inspector v7.0 - Frontend Visualization Components
 * 
 * Provides comprehensive visualization capabilities for:
 * - IRT parameter analysis and visualization
 * - Similarity analysis with confidence intervals
 * - Score trace and progress monitoring
 * - Real-time test progress tracking
 * - Statistical calculations (ICC, Information Theory)
 * 
 * @version 7.0
 * @author LLM Inspector Team
 */

// ============================================================================
// CSS Styles for v7 Visualization Components
// ============================================================================

export const V7_STYLES = `
/* v7 Visualization Styles */
.v7-container {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

.v7-chart {
    background: #ffffff;
    border: 1px solid #e1e5e9;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 16px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.v7-progress-bar {
    width: 100%;
    height: 8px;
    background: #f1f3f4;
    border-radius: 4px;
    overflow: hidden;
    margin: 8px 0;
}

.v7-progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #4285f4, #34a853);
    transition: width 0.3s ease;
}

.v7-metric-card {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 6px;
    padding: 12px;
    margin: 8px 0;
}

.v7-metric-value {
    font-size: 24px;
    font-weight: bold;
    color: #202124;
}

.v7-metric-label {
    font-size: 12px;
    color: #5f6368;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.v7-status-indicator {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    margin-right: 8px;
}

.v7-status-success { background-color: #34a853; }
.v7-status-warning { background-color: #fbbc04; }
.v7-status-error { background-color: #ea4335; }
.v7-status-running { background-color: #4285f4; animation: pulse 2s infinite; }

@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}

.v7-tooltip {
    position: absolute;
    background: rgba(0, 0, 0, 0.8);
    color: white;
    padding: 8px 12px;
    border-radius: 4px;
    font-size: 12px;
    pointer-events: none;
    z-index: 1000;
    max-width: 200px;
}
`;

// ============================================================================
// IRT Parameters Visualization
// ============================================================================

/**
 * Renders IRT (Item Response Theory) parameters visualization
 * @param {Object} params - IRT parameters data
 * @param {string} containerId - Container element ID
 * @param {Object} options - Rendering options
 */
export function renderIRTParameters(params, containerId, options = {}) {
    const container = document.getElementById(containerId);
    if (!container) {
        console.error(`Container ${containerId} not found`);
        return;
    }

    const {
        discrimination = [],
        difficulty = [],
        guessing = [],
        itemIds = []
    } = params;

    const {
        width = 800,
        height = 400,
        showGrid = true,
        interactive = true
    } = options;

    // Clear container
    container.innerHTML = '';
    container.className = 'v7-chart';

    // Create title
    const title = document.createElement('h3');
    title.textContent = 'IRT Parameters Analysis';
    title.style.marginBottom = '16px';
    container.appendChild(title);

    // Create SVG canvas
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('width', width);
    svg.setAttribute('height', height);
    svg.style.border = '1px solid #e1e5e9';
    container.appendChild(svg);

    // Calculate scales
    const margin = { top: 20, right: 30, bottom: 40, left: 50 };
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;

    // Difficulty vs Discrimination scatter plot
    const difficultyScale = createLinearScale(
        Math.min(...difficulty), 
        Math.max(...difficulty), 
        0, 
        chartWidth
    );
    
    const discriminationScale = createLinearScale(
        Math.min(...discrimination), 
        Math.max(...discrimination), 
        chartHeight, 
        0
    );

    // Draw grid
    if (showGrid) {
        drawGrid(svg, margin, chartWidth, chartHeight);
    }

    // Draw axes
    drawAxes(svg, margin, chartWidth, chartHeight, 'Difficulty', 'Discrimination');

    // Draw data points
    const points = [];
    for (let i = 0; i < difficulty.length; i++) {
        const x = margin.left + difficultyScale(difficulty[i]);
        const y = margin.top + discriminationScale(discrimination[i]);
        
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', x);
        circle.setAttribute('cy', y);
        circle.setAttribute('r', 6);
        circle.setAttribute('fill', '#4285f4');
        circle.setAttribute('stroke', '#ffffff');
        circle.setAttribute('stroke-width', 2);
        circle.style.cursor = 'pointer';
        
        // Add tooltip
        if (interactive) {
            circle.addEventListener('mouseenter', (e) => {
                showTooltip(e, `Item: ${itemIds[i] || i}<br>Difficulty: ${difficulty[i].toFixed(3)}<br>Discrimination: ${discrimination[i].toFixed(3)}`);
            });
            
            circle.addEventListener('mouseleave', hideTooltip);
        }
        
        svg.appendChild(circle);
        points.push({ x, y, item: itemIds[i] || i, difficulty: difficulty[i], discrimination: discrimination[i] });
    }

    // Add legend
    const legend = document.createElement('div');
    legend.innerHTML = `
        <div style="display: flex; align-items: center; margin-top: 12px;">
            <div style="width: 12px; height: 12px; background: #4285f4; border-radius: 50%; margin-right: 8px;"></div>
            <span style="font-size: 12px; color: #5f6368;">Test Items</span>
        </div>
    `;
    container.appendChild(legend);

    return { svg, points };
}

// ============================================================================
// Similarity Visualization with Confidence Intervals
// ============================================================================

/**
 * Renders similarity analysis with confidence intervals
 * @param {Object} data - Similarity data with confidence intervals
 * @param {string} containerId - Container element ID
 * @param {Object} options - Rendering options
 */
export function renderSimilarityWithCI(data, containerId, options = {}) {
    const container = document.getElementById(containerId);
    if (!container) {
        console.error(`Container ${containerId} not found`);
        return;
    }

    const {
        similarities = [],
        confidenceIntervals = [],
        labels = [],
        baseline = null
    } = data;

    const {
        width = 800,
        height = 300,
        showCI = true,
        colors = ['#4285f4', '#ea4335', '#fbbc04', '#34a853']
    } = options;

    container.innerHTML = '';
    container.className = 'v7-chart';

    // Create title
    const title = document.createElement('h3');
    title.textContent = 'Similarity Analysis with Confidence Intervals';
    title.style.marginBottom = '16px';
    container.appendChild(title);

    // Create canvas
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    canvas.style.border = '1px solid #e1e5e9';
    container.appendChild(canvas);

    const ctx = canvas.getContext('2d');
    const margin = { top: 20, right: 30, bottom: 60, left: 60 };
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Calculate scales
    const maxSim = Math.max(...similarities, ...confidenceIntervals.map(ci => ci[1]));
    const minSim = Math.min(...similarities, ...confidenceIntervals.map(ci => ci[0]));
    const simScale = createLinearScale(minSim, maxSim, chartHeight, 0);
    const labelScale = createLinearScale(0, labels.length - 1, 0, chartWidth);

    // Draw axes
    ctx.strokeStyle = '#e1e5e9';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(margin.left, margin.top);
    ctx.lineTo(margin.left, margin.top + chartHeight);
    ctx.lineTo(margin.left + chartWidth, margin.top + chartHeight);
    ctx.stroke();

    // Draw baseline if provided
    if (baseline !== null) {
        const baselineY = margin.top + simScale(baseline);
        ctx.strokeStyle = '#ea4335';
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(margin.left, baselineY);
        ctx.lineTo(margin.left + chartWidth, baselineY);
        ctx.stroke();
        ctx.setLineDash([]);
    }

    // Draw confidence intervals
    if (showCI) {
        confidenceIntervals.forEach((ci, i) => {
            const x = margin.left + labelScale(i);
            const yLow = margin.top + simScale(ci[0]);
            const yHigh = margin.top + simScale(ci[1]);
            
            ctx.fillStyle = 'rgba(66, 133, 244, 0.2)';
            ctx.fillRect(x - 15, yLow, 30, yHigh - yLow);
        });
    }

    // Draw similarity points and lines
    ctx.strokeStyle = '#4285f4';
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    similarities.forEach((sim, i) => {
        const x = margin.left + labelScale(i);
        const y = margin.top + simScale(sim);
        
        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    });
    ctx.stroke();

    // Draw points
    similarities.forEach((sim, i) => {
        const x = margin.left + labelScale(i);
        const y = margin.top + simScale(sim);
        
        ctx.fillStyle = '#4285f4';
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, 2 * Math.PI);
        ctx.fill();
        
        // Draw labels
        ctx.fillStyle = '#5f6368';
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(labels[i], x, margin.top + chartHeight + 20);
    });

    // Draw y-axis labels
    ctx.fillStyle = '#5f6368';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'right';
    for (let i = 0; i <= 5; i++) {
        const value = minSim + (maxSim - minSim) * (i / 5);
        const y = margin.top + simScale(value);
        ctx.fillText(value.toFixed(2), margin.left - 10, y + 3);
    }

    return { canvas, similarities, confidenceIntervals };
}

// ============================================================================
// Score Trace Visualization
// ============================================================================

/**
 * Renders score trace visualization over time
 * @param {Object} data - Score trace data
 * @param {string} containerId - Container element ID
 * @param {Object} options - Rendering options
 */
export function renderScoreTrace(data, containerId, options = {}) {
    const container = document.getElementById(containerId);
    if (!container) {
        console.error(`Container ${containerId} not found`);
        return;
    }

    const {
        timestamps = [],
        scores = [],
        dimensions = {},
        events = []
    } = data;

    const {
        width = 800,
        height = 400,
        showDimensions = true,
        showEvents = true
    } = options;

    container.innerHTML = '';
    container.className = 'v7-chart';

    // Create title
    const title = document.createElement('h3');
    title.textContent = 'Score Trace Analysis';
    title.style.marginBottom = '16px';
    container.appendChild(title);

    // Create SVG
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('width', width);
    svg.setAttribute('height', height);
    container.appendChild(svg);

    const margin = { top: 20, right: 30, bottom: 40, left: 50 };
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;

    // Calculate scales
    const maxScore = Math.max(...scores);
    const minScore = Math.min(...scores);
    const timeScale = createLinearScale(0, timestamps.length - 1, 0, chartWidth);
    const scoreScale = createLinearScale(minScore, maxScore, chartHeight, 0);

    // Draw axes
    drawAxes(svg, margin, chartWidth, chartHeight, 'Time', 'Score');

    // Draw grid
    drawGrid(svg, margin, chartWidth, chartHeight);

    // Draw main score line
    const mainLine = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
    const points = scores.map((score, i) => {
        const x = margin.left + timeScale(i);
        const y = margin.top + scoreScale(score);
        return `${x},${y}`;
    }).join(' ');
    
    mainLine.setAttribute('points', points);
    mainLine.setAttribute('fill', 'none');
    mainLine.setAttribute('stroke', '#4285f4');
    mainLine.setAttribute('stroke-width', 2);
    svg.appendChild(mainLine);

    // Draw data points
    scores.forEach((score, i) => {
        const x = margin.left + timeScale(i);
        const y = margin.top + scoreScale(score);
        
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', x);
        circle.setAttribute('cy', y);
        circle.setAttribute('r', 3);
        circle.setAttribute('fill', '#4285f4');
        svg.appendChild(circle);
    });

    // Draw dimension scores if available
    if (showDimensions && dimensions) {
        const colors = ['#ea4335', '#fbbc04', '#34a853', '#ff6d01'];
        let colorIndex = 0;
        
        for (const [dimName, dimScores] of Object.entries(dimensions)) {
            const dimLine = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
            const dimPoints = dimScores.map((score, i) => {
                const x = margin.left + timeScale(i);
                const y = margin.top + scoreScale(score);
                return `${x},${y}`;
            }).join(' ');
            
            dimLine.setAttribute('points', dimPoints);
            dimLine.setAttribute('fill', 'none');
            dimLine.setAttribute('stroke', colors[colorIndex % colors.length]);
            dimLine.setAttribute('stroke-width', 1);
            dimLine.setAttribute('opacity', '0.7');
            svg.appendChild(dimLine);
            colorIndex++;
        }
    }

    // Draw events if available
    if (showEvents && events) {
        events.forEach(event => {
            const x = margin.left + timeScale(event.timestamp);
            
            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('x1', x);
            line.setAttribute('y1', margin.top);
            line.setAttribute('x2', x);
            line.setAttribute('y2', margin.top + chartHeight);
            line.setAttribute('stroke', '#ea4335');
            line.setAttribute('stroke-width', 1);
            line.setAttribute('stroke-dasharray', '5,5');
            line.setAttribute('opacity', '0.5');
            svg.appendChild(line);
            
            // Add event label
            const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            text.setAttribute('x', x);
            text.setAttribute('y', margin.top - 5);
            text.setAttribute('text-anchor', 'middle');
            text.setAttribute('font-size', '10');
            text.setAttribute('fill', '#ea4335');
            text.textContent = event.label;
            svg.appendChild(text);
        });
    }

    return { svg, scores, timestamps };
}

// ============================================================================
// Test Progress Monitor
// ============================================================================

/**
 * Test progress monitoring class for real-time updates
 */
export class TestProgressMonitor {
    constructor(containerId, options = {}) {
        this.containerId = containerId;
        this.container = document.getElementById(containerId);
        this.options = {
            showETA: true,
            showMetrics: true,
            updateInterval: 1000,
            ...options
        };
        
        this.startTime = null;
        this.currentProgress = 0;
        this.totalTests = 0;
        this.completedTests = 0;
        this.failedTests = 0;
        this.metrics = {};
        
        this.init();
    }
    
    init() {
        if (!this.container) {
            console.error(`Container ${this.containerId} not found`);
            return;
        }
        
        this.container.innerHTML = '';
        this.container.className = 'v7-chart';
        
        // Create header
        const header = document.createElement('div');
        header.innerHTML = `
            <h3>Test Progress Monitor</h3>
            <div class="v7-progress-bar">
                <div class="v7-progress-fill" id="${this.containerId}-progress" style="width: 0%"></div>
            </div>
            <div id="${this.containerId}-status">Ready to start...</div>
        `;
        this.container.appendChild(header);
        
        // Create metrics section
        if (this.options.showMetrics) {
            const metricsSection = document.createElement('div');
            metricsSection.id = `${this.containerId}-metrics`;
            metricsSection.style.display = 'grid';
            metricsSection.style.gridTemplateColumns = 'repeat(auto-fit, minmax(150px, 1fr))';
            metricsSection.style.gap = '12px';
            metricsSection.style.marginTop = '16px';
            this.container.appendChild(metricsSection);
        }
    }
    
    start(totalTests) {
        this.totalTests = totalTests;
        this.startTime = Date.now();
        this.completedTests = 0;
        this.failedTests = 0;
        this.currentProgress = 0;
        
        this.updateDisplay();
        
        if (this.options.updateInterval > 0) {
            this.updateTimer = setInterval(() => this.updateDisplay(), this.options.updateInterval);
        }
    }
    
    updateTestResult(testId, result) {
        this.completedTests++;
        if (result.status === 'failed') {
            this.failedTests++;
        }
        
        this.currentProgress = (this.completedTests / this.totalTests) * 100;
        this.updateDisplay();
    }
    
    updateMetric(name, value) {
        this.metrics[name] = value;
        this.updateDisplay();
    }
    
    updateDisplay() {
        const progressBar = document.getElementById(`${this.containerId}-progress`);
        const status = document.getElementById(`${this.containerId}-status`);
        
        if (progressBar) {
            progressBar.style.width = `${this.currentProgress}%`;
        }
        
        if (status) {
            const elapsed = this.startTime ? (Date.now() - this.startTime) / 1000 : 0;
            const eta = this.currentProgress > 0 ? (elapsed / this.currentProgress * 100) - elapsed : 0;
            
            let statusText = `Progress: ${this.completedTests}/${this.totalTests} (${this.currentProgress.toFixed(1)}%)`;
            
            if (this.options.showETA && eta > 0) {
                statusText += ` - ETA: ${formatDuration(eta)}`;
            }
            
            if (this.failedTests > 0) {
                statusText += ` - Failed: ${this.failedTests}`;
            }
            
            status.textContent = statusText;
        }
        
        // Update metrics
        if (this.options.showMetrics) {
            const metricsContainer = document.getElementById(`${this.containerId}-metrics`);
            if (metricsContainer) {
                metricsContainer.innerHTML = '';
                
                // Add default metrics
                const defaultMetrics = [
                    { name: 'Completed', value: `${this.completedTests}/${this.totalTests}` },
                    { name: 'Success Rate', value: this.totalTests > 0 ? `${((this.totalTests - this.failedTests) / this.totalTests * 100).toFixed(1)}%` : 'N/A' },
                    { name: 'Elapsed', value: this.startTime ? formatDuration((Date.now() - this.startTime) / 1000) : 'N/A' }
                ];
                
                [...defaultMetrics, ...Object.entries(this.metrics).map(([name, value]) => ({ name, value }))]
                    .forEach(metric => {
                        const card = document.createElement('div');
                        card.className = 'v7-metric-card';
                        card.innerHTML = `
                            <div class="v7-metric-value">${metric.value}</div>
                            <div class="v7-metric-label">${metric.name}</div>
                        `;
                        metricsContainer.appendChild(card);
                    });
            }
        }
    }
    
    finish() {
        if (this.updateTimer) {
            clearInterval(this.updateTimer);
        }
        
        const status = document.getElementById(`${this.containerId}-status`);
        if (status) {
            const totalDuration = (Date.now() - this.startTime) / 1000;
            status.innerHTML = `
                <span class="v7-status-indicator v7-status-success"></span>
                Completed: ${this.completedTests}/${this.totalTests} tests in ${formatDuration(totalDuration)}
            `;
        }
    }
}

// ============================================================================
// Real-time Progress Rendering
// ============================================================================

/**
 * Renders real-time progress with animations
 * @param {Object} data - Progress data
 * @param {string} containerId - Container element ID
 * @param {Object} options - Rendering options
 */
export function renderRealtimeProgress(data, containerId, options = {}) {
    const container = document.getElementById(containerId);
    if (!container) {
        console.error(`Container ${containerId} not found`);
        return;
    }
    
    const {
        current = 0,
        total = 100,
        status = 'running',
        message = '',
        stages = []
    } = data;
    
    const {
        animated = true,
        showStages = true,
        colorScheme = 'blue'
    } = options;
    
    container.innerHTML = '';
    container.className = 'v7-chart';
    
    // Create progress section
    const progressSection = document.createElement('div');
    progressSection.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-weight: bold;">Progress</span>
            <span id="${containerId}-percentage">${current.toFixed(1)}%</span>
        </div>
        <div class="v7-progress-bar">
            <div class="v7-progress-fill" id="${containerId}-bar" style="width: ${current}%; background: ${getProgressColor(colorScheme)};"></div>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 8px; font-size: 12px; color: #5f6368;">
            <span id="${containerId}-message">${message}</span>
            <span class="v7-status-indicator v7-status-${status}"></span>
        </div>
    `;
    container.appendChild(progressSection);
    
    // Create stages section
    if (showStages && stages.length > 0) {
        const stagesSection = document.createElement('div');
        stagesSection.style.marginTop = '16px';
        
        const stagesTitle = document.createElement('h4');
        stagesTitle.textContent = 'Stages';
        stagesTitle.style.marginBottom = '8px';
        stagesSection.appendChild(stagesTitle);
        
        const stagesList = document.createElement('div');
        stagesList.style.display = 'grid';
        stagesList.style.gridTemplateColumns = 'repeat(auto-fit, minmax(200px, 1fr))';
        stagesList.style.gap = '8px';
        
        stages.forEach((stage, index) => {
            const stageCard = document.createElement('div');
            stageCard.className = 'v7-metric-card';
            stageCard.innerHTML = `
                <div style="display: flex; align-items: center;">
                    <span class="v7-status-indicator v7-status-${stage.status}"></span>
                    <span style="font-weight: bold; margin-left: 8px;">${stage.name}</span>
                </div>
                <div style="font-size: 12px; color: #5f6368; margin-top: 4px;">${stage.progress || 0}% complete</div>
            `;
            stagesList.appendChild(stageCard);
        });
        
        stagesSection.appendChild(stagesList);
        container.appendChild(stagesSection);
    }
    
    // Animate if requested
    if (animated) {
        animateProgress(containerId, current, total);
    }
    
    return { container, current, total };
}

// ============================================================================
// Statistical Calculations
// ============================================================================

/**
 * Calculates Intraclass Correlation Coefficient (ICC)
 * @param {Array} data - Data array for ICC calculation
 * @param {string} model - ICC model type ('one-way', 'two-way', 'consistency')
 * @param {string} type - ICC type ('single', 'average')
 * @returns {number} ICC value
 */
export function calculateICC(data, model = 'two-way', type = 'average') {
    if (!data || data.length === 0) {
        return 0;
    }
    
    // Simplified ICC calculation (two-way random effects, absolute agreement)
    const n = data.length;
    const k = data[0] ? data[0].length : 1; // Number of raters/measurements
    
    // Calculate means
    const grandMean = data.flat().reduce((sum, val) => sum + val, 0) / (n * k);
    const subjectMeans = data.map(subject => subject.reduce((sum, val) => sum + val, 0) / k);
    const raterMeans = [];
    
    for (let j = 0; j < k; j++) {
        const raterSum = data.reduce((sum, subject) => sum + subject[j], 0);
        raterMeans.push(raterSum / n);
    }
    
    // Calculate sum of squares
    let sst = 0; // Total sum of squares
    let sss = 0; // Subject sum of squares
    let ssr = 0; // Rater sum of squares
    let sse = 0; // Error sum of squares
    
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < k; j++) {
            const value = data[i][j];
            sst += Math.pow(value - grandMean, 2);
            sss += Math.pow(subjectMeans[i] - grandMean, 2);
            ssr += Math.pow(raterMeans[j] - grandMean, 2);
            sse += Math.pow(value - subjectMeans[i] - raterMeans[j] + grandMean, 2);
        }
    }
    
    // Calculate ICC based on model and type
    const msBetweenSubjects = sss / (n - 1);
    const msError = sse / ((n - 1) * (k - 1));
    
    if (type === 'single') {
        return (msBetweenSubjects - msError) / (msBetweenSubjects + (k - 1) * msError);
    } else {
        return (msBetweenSubjects - msError) / msBetweenSubjects;
    }
}

/**
 * Calculates information theory metrics
 * @param {Array} probabilities - Probability distribution
 * @param {string} metric - Metric type ('entropy', 'mutual_info', 'kl_divergence')
 * @param {Array} reference - Reference distribution (for KL divergence)
 * @returns {number} Calculated metric value
 */
export function calculateInformation(probabilities, metric = 'entropy', reference = null) {
    if (!probabilities || probabilities.length === 0) {
        return 0;
    }
    
    // Normalize probabilities
    const sum = probabilities.reduce((a, b) => a + b, 0);
    const probs = probabilities.map(p => p / sum);
    
    switch (metric) {
        case 'entropy':
            return calculateEntropy(probs);
            
        case 'mutual_info':
            if (!reference) return 0;
            return calculateMutualInformation(probs, reference);
            
        case 'kl_divergence':
            if (!reference) return 0;
            return calculateKLDivergence(probs, reference);
            
        default:
            return calculateEntropy(probs);
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Creates a linear scale function
 * @param {number} domainMin - Minimum domain value
 * @param {number} domainMax - Maximum domain value
 * @param {number} rangeMin - Minimum range value
 * @param {number} rangeMax - Maximum range value
 * @returns {Function} Scale function
 */
function createLinearScale(domainMin, domainMax, rangeMin, rangeMax) {
    const domainRange = domainMax - domainMin;
    const rangeRange = rangeMax - rangeMin;
    
    return (value) => {
        if (domainRange === 0) return rangeMin;
        return rangeMin + ((value - domainMin) / domainRange) * rangeRange;
    };
}

/**
 * Draws grid on SVG canvas
 * @param {SVGElement} svg - SVG element
 * @param {Object} margin - Margin object
 * @param {number} width - Chart width
 * @param {number} height - Chart height
 */
function drawGrid(svg, margin, width, height) {
    const gridGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    gridGroup.setAttribute('class', 'grid');
    
    // Vertical grid lines
    for (let i = 0; i <= 10; i++) {
        const x = margin.left + (width / 10) * i;
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', x);
        line.setAttribute('y1', margin.top);
        line.setAttribute('x2', x);
        line.setAttribute('y2', margin.top + height);
        line.setAttribute('stroke', '#f1f3f4');
        line.setAttribute('stroke-width', '1');
        gridGroup.appendChild(line);
    }
    
    // Horizontal grid lines
    for (let i = 0; i <= 10; i++) {
        const y = margin.top + (height / 10) * i;
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', margin.left);
        line.setAttribute('y1', y);
        line.setAttribute('x2', margin.left + width);
        line.setAttribute('y2', y);
        line.setAttribute('stroke', '#f1f3f4');
        line.setAttribute('stroke-width', '1');
        gridGroup.appendChild(line);
    }
    
    svg.appendChild(gridGroup);
}

/**
 * Draws axes on SVG canvas
 * @param {SVGElement} svg - SVG element
 * @param {Object} margin - Margin object
 * @param {number} width - Chart width
 * @param {number} height - Chart height
 * @param {string} xLabel - X-axis label
 * @param {string} yLabel - Y-axis label
 */
function drawAxes(svg, margin, width, height, xLabel, yLabel) {
    // X-axis
    const xAxis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    xAxis.setAttribute('x1', margin.left);
    xAxis.setAttribute('y1', margin.top + height);
    xAxis.setAttribute('x2', margin.left + width);
    xAxis.setAttribute('y2', margin.top + height);
    xAxis.setAttribute('stroke', '#202124');
    xAxis.setAttribute('stroke-width', '2');
    svg.appendChild(xAxis);
    
    // Y-axis
    const yAxis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    yAxis.setAttribute('x1', margin.left);
    yAxis.setAttribute('y1', margin.top);
    yAxis.setAttribute('x2', margin.left);
    yAxis.setAttribute('y2', margin.top + height);
    yAxis.setAttribute('stroke', '#202124');
    yAxis.setAttribute('stroke-width', '2');
    svg.appendChild(yAxis);
    
    // X-axis label
    const xText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    xText.setAttribute('x', margin.left + width / 2);
    xText.setAttribute('y', margin.top + height + 35);
    xText.setAttribute('text-anchor', 'middle');
    xText.setAttribute('font-size', '12');
    xText.setAttribute('fill', '#5f6368');
    xText.textContent = xLabel;
    svg.appendChild(xText);
    
    // Y-axis label
    const yText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    yText.setAttribute('x', margin.top + height / 2);
    yText.setAttribute('y', margin.left - 35);
    yText.setAttribute('text-anchor', 'middle');
    yText.setAttribute('font-size', '12');
    yText.setAttribute('fill', '#5f6368');
    yText.setAttribute('transform', `rotate(-90, ${margin.left - 35}, ${margin.top + height / 2})`);
    yText.textContent = yLabel;
    svg.appendChild(yText);
}

/**
 * Shows tooltip at mouse position
 * @param {Event} event - Mouse event
 * @param {string} content - Tooltip content
 */
function showTooltip(event, content) {
    hideTooltip(); // Remove existing tooltip
    
    const tooltip = document.createElement('div');
    tooltip.className = 'v7-tooltip';
    tooltip.innerHTML = content;
    tooltip.style.left = event.pageX + 10 + 'px';
    tooltip.style.top = event.pageY - 30 + 'px';
    document.body.appendChild(tooltip);
}

/**
 * Hides tooltip
 */
function hideTooltip() {
    const existing = document.querySelector('.v7-tooltip');
    if (existing) {
        existing.remove();
    }
}

/**
 * Formats duration in seconds to human-readable format
 * @param {number} seconds - Duration in seconds
 * @returns {string} Formatted duration
 */
function formatDuration(seconds) {
    if (seconds < 60) {
        return `${seconds.toFixed(1)}s`;
    } else if (seconds < 3600) {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = seconds % 60;
        return `${minutes}m ${remainingSeconds.toFixed(0)}s`;
    } else {
        const hours = Math.floor(seconds / 3600);
        const remainingMinutes = Math.floor((seconds % 3600) / 60);
        return `${hours}h ${remainingMinutes}m`;
    }
}

/**
 * Gets progress color based on color scheme
 * @param {string} scheme - Color scheme name
 * @returns {string} CSS gradient string
 */
function getProgressColor(scheme) {
    const schemes = {
        blue: 'linear-gradient(90deg, #4285f4, #34a853)',
        red: 'linear-gradient(90deg, #ea4335, #fbbc04)',
        green: 'linear-gradient(90deg, #34a853, #4285f4)',
        purple: 'linear-gradient(90deg, #9c27b0, #673ab7)'
    };
    return schemes[scheme] || schemes.blue;
}

/**
 * Animates progress bar
 * @param {string} containerId - Container ID
 * @param {number} target - Target progress
 * @param {number} total - Total value
 */
function animateProgress(containerId, target, total) {
    const bar = document.getElementById(`${containerId}-bar`);
    const percentage = document.getElementById(`${containerId}-percentage`);
    
    if (!bar) return;
    
    const targetPercent = (target / total) * 100;
    const duration = 1000; // 1 second animation
    const startTime = Date.now();
    const startPercent = parseFloat(bar.style.width) || 0;
    
    function update() {
        const elapsed = Date.now() - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        const currentPercent = startPercent + (targetPercent - startPercent) * easeInOutQuad(progress);
        bar.style.width = currentPercent + '%';
        
        if (percentage) {
            percentage.textContent = currentPercent.toFixed(1) + '%';
        }
        
        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }
    
    update();
}

/**
 * Easing function for smooth animations
 * @param {number} t - Progress value (0-1)
 * @returns {number} Eased value
 */
function easeInOutQuad(t) {
    return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
}

/**
 * Calculates Shannon entropy
 * @param {Array} probabilities - Probability distribution
 * @returns {number} Entropy value
 */
function calculateEntropy(probabilities) {
    return probabilities.reduce((entropy, p) => {
        if (p > 0) {
            return entropy - p * Math.log2(p);
        }
        return entropy;
    }, 0);
}

/**
 * Calculates mutual information
 * @param {Array} p - First probability distribution
 * @param {Array} q - Second probability distribution
 * @returns {number} Mutual information value
 */
function calculateMutualInformation(p, q) {
    // Simplified mutual information calculation
    const entropyP = calculateEntropy(p);
    const entropyQ = calculateEntropy(q);
    
    // This is a simplified version - real MI would need joint distribution
    return Math.abs(entropyP - entropyQ) * 0.5;
}

/**
 * Calculates Kullback-Leibler divergence
 * @param {Array} p - First probability distribution
 * @param {Array} q - Second probability distribution
 * @returns {number} KL divergence value
 */
function calculateKLDivergence(p, q) {
    return p.reduce((divergence, pi, i) => {
        if (pi > 0 && q[i] > 0) {
            return divergence + pi * Math.log2(pi / q[i]);
        }
        return divergence;
    }, 0);
}

// ============================================================================
// Export all functions for external use
// ============================================================================

export default {
    // Main visualization functions
    renderIRTParameters,
    renderSimilarityWithCI,
    renderScoreTrace,
    renderRealtimeProgress,
    
    // Classes
    TestProgressMonitor,
    
    // Statistical functions
    calculateICC,
    calculateInformation,
    
    // Utilities
    V7_STYLES,
    createLinearScale,
    formatDuration
};

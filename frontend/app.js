/**
 * Crypto Regime Intelligence Engine - Frontend Application
 * Handles API communication and UI updates
 */

// Configuration
const CONFIG = {
    API_BASE_URL: 'http://localhost:8000',
    REFRESH_INTERVAL: 60000, 
};

// State
const state = {
    isLoading: false,
    lastData: null,
    apiConnected: false,
};

// DOM Elements
const elements = {
    // Status
    apiStatus: document.getElementById('apiStatus'),
    
    // Inputs
    assetInput: document.getElementById('assetInput'),
    daysInput: document.getElementById('daysInput'),
    analyzeBtn: document.getElementById('analyzeBtn'),
    tradabilityAsset: document.getElementById('tradabilityAsset'),
    
    // Loading
    loadingOverlay: document.getElementById('loadingOverlay'),
    loadingText: document.getElementById('loadingText'),
    
    // 1. Regime Card
    regimeIndicator: document.getElementById('regimeIndicator'),
    regimeText: document.getElementById('regimeText'),
    regimeConfidence: document.getElementById('regimeConfidence'),
    regimeExplanation: document.getElementById('regimeExplanation'),
    probTrend: document.getElementById('probTrend'),
    probRange: document.getElementById('probRange'),
    probRisk: document.getElementById('probRisk'),
    probTrendVal: document.getElementById('probTrendVal'),
    probRangeVal: document.getElementById('probRangeVal'),
    probRiskVal: document.getElementById('probRiskVal'),

    // 2. Position AI Card
    actionBadge: document.getElementById('actionBadge'),
    positionMult: document.getElementById('positionMult'),
    stopLoss: document.getElementById('stopLoss'),
    takeProfit: document.getElementById('takeProfit'),
    positionRationale: document.getElementById('positionRationale'),

    // 3. Multi-Timeframe Card
    confluenceScore: document.getElementById('confluenceScore'),
    tf1w: document.getElementById('tf1w'),
    tf2w: document.getElementById('tf2w'),
    tf1m: document.getElementById('tf1m'),
    tf3m: document.getElementById('tf3m'),
    tfSummary: document.getElementById('tfSummary'),

    // 4. Sentiment & Prediction
    sentimentFill: document.getElementById('sentimentFill'),
    sentimentValue: document.getElementById('sentimentValue'),
    predictionTarget: document.getElementById('predictionTarget'),
    predictionProb: document.getElementById('predictionProb'),

    // 5. Whale Tracking
    whaleActivity: document.getElementById('whaleActivity'),
    netFlow: document.getElementById('netFlow'),
    whaleSentiment: document.getElementById('whaleSentiment'),
    whaleSignal: document.getElementById('whaleSignal'),
    
    // 6. Metrics & Market Data
    volatilityValue: document.getElementById('volatilityValue'),
    correlationValue: document.getElementById('correlationValue'),
    liquidityValue: document.getElementById('liquidityValue'),

    // Market Data
    currentPrice: document.getElementById('currentPrice'),
    priceChange: document.getElementById('priceChange'),
    volume24h: document.getElementById('volume24h'),

    // Tradability (Legacy/Secondary)
    scoreProgress: document.getElementById('scoreProgress'),
    tradabilityScore: document.getElementById('tradabilityScore'),
    riskBadge: document.getElementById('riskBadge'),
    riskLevel: document.getElementById('riskLevel'),
    tradabilityExplanation: document.getElementById('tradabilityExplanation'),
};

// Utility Functions
function formatNumber(num, decimals = 2) {
    if (num === null || num === undefined) return '--';
    return new Intl.NumberFormat('en-US', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals,
    }).format(num);
}

function formatCurrency(num) {
    if (num === null || num === undefined) return '--';
    if (num >= 1e12) return `$${(num / 1e12).toFixed(2)}T`;
    if (num >= 1e9) return `$${(num / 1e9).toFixed(2)}B`;
    if (num >= 1e6) return `$${(num / 1e6).toFixed(2)}M`;
    if (num >= 1e3) return `$${(num / 1e3).toFixed(2)}K`;
    return `$${formatNumber(num)}`;
}

function formatPercent(num, decimals = 1) {
    if (num === null || num === undefined) return '--%';
    return `${num >= 0 ? '+' : ''}${num.toFixed(decimals)}%`;
}

function getRegimeIcon(regime) {
    const icons = { 'TREND': '📈', 'RANGE': '↔️', 'HIGH-RISK': '⚠️' };
    return icons[regime] || '❓';
}

function getRegimeClass(regime) {
    const classes = { 'TREND': 'trend', 'RANGE': 'range', 'HIGH-RISK': 'high-risk' };
    return classes[regime] || '';
}

function getRiskClass(riskLevel) {
    return riskLevel ? riskLevel.toLowerCase() : '';
}

// API Functions
async function fetchAPI(endpoint, params = {}) {
    const url = new URL(`${CONFIG.API_BASE_URL}${endpoint}`);
    Object.keys(params).forEach(key => {
        if (params[key] !== undefined && params[key] !== null) {
            url.searchParams.append(key, params[key]);
        }
    });
    
    const response = await fetch(url, { headers: { 'Accept': 'application/json' } });
    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(error.detail || `HTTP error: ${response.status}`);
    }
    return response.json();
}

async function checkHealth() {
    try {
        const data = await fetchAPI('/health');
        state.apiConnected = data.status === 'healthy';
        updateStatusIndicator(state.apiConnected);
        return data;
    } catch (error) {
        console.error('Health check failed:', error);
        state.apiConnected = false;
        updateStatusIndicator(false);
        return null;
    }
}

// UI Update Functions
function updateStatusIndicator(connected) {
    elements.apiStatus.className = connected ? 'status-indicator connected' : 'status-indicator error';
    elements.apiStatus.querySelector('.status-text').textContent = connected ? 'API Connected' : 'API Disconnected';
}

function showLoading(message = 'Analyzing market data...') {
    state.isLoading = true;
    elements.loadingText.textContent = message;
    elements.loadingOverlay.classList.add('active');
    elements.analyzeBtn.disabled = true;
}

function hideLoading() {
    state.isLoading = false;
    elements.loadingOverlay.classList.remove('active');
    elements.analyzeBtn.disabled = false;
}

// ---------------------------------------------------------
// NEW: Comprehensive Update Functions
// ---------------------------------------------------------

function updateDashboard(data) {
    // 1. Regime Analysis (Legacy + New)
    updateRegimeDisplay(data.regime_analysis);
    updateMetrics(data.regime_analysis.detailed_metrics || {}); // Mock if missing or map from comprehensive
    
    // 2. Multi-Timeframe
    updateMultiTimeframe(data.multi_timeframe);
    
    // 3. Position Sizing
    updatePositionSizing(data.position_sizing);
    
    // 4. Whale Tracking
    updateWhaleTracking(data.whale_activity);
    
    // 5. Prediction
    updatePrediction(data.prediction);

    // 6. Tradability (Legacy Card)
    // Map comprehensive data to tradability display format
    const tradabilityData = {
        tradability_score: data.regime_analysis.tradability_score,
        risk_level: data.regime_analysis.risk_level,
        reasoning: data.regime_analysis.tradability_reasoning || data.regime_analysis.reasoning,
        market_data: data.regime_analysis.market_data
    };
    updateTradabilityDisplay(tradabilityData);
}

function updateRegimeDisplay(regimeData) {
    // Current Regime
    const regime = regimeData.current_regime || 'UNKNOWN';
    elements.regimeIndicator.className = `regime-indicator ${getRegimeClass(regime)}`;
    elements.regimeText.textContent = regime;
    
    // Update icon
    const iconSpan = elements.regimeIndicator.querySelector('.regime-icon');
    if (iconSpan) iconSpan.textContent = getRegimeIcon(regime);
    
    // Confidence (Mock or mapped)
    elements.regimeConfidence.textContent = `Confidence: ${(regimeData.confidence * 100).toFixed(0)}%`; 

    // Probabilities
    if (regimeData.regime_probabilities) {
        const trendProb = (regimeData.regime_probabilities['TREND'] || 0) * 100;
        const rangeProb = (regimeData.regime_probabilities['RANGE'] || 0) * 100;
        const riskProb = (regimeData.regime_probabilities['HIGH-RISK'] || 0) * 100;
        
        elements.probTrend.style.width = `${trendProb}%`;
        elements.probRange.style.width = `${rangeProb}%`;
        elements.probRisk.style.width = `${riskProb}%`;
        
        elements.probTrendVal.textContent = `${trendProb.toFixed(0)}%`;
        elements.probRangeVal.textContent = `${rangeProb.toFixed(0)}%`;
        elements.probRiskVal.textContent = `${riskProb.toFixed(0)}%`;
    }

    elements.regimeExplanation.innerHTML = `<p>${regimeData.reasoning || 'AI analysis complete.'}</p>`;
}

function updateTradabilityDisplay(data) {
    // Update score circle
    const score = data.tradability_score;
    const circumference = 2 * Math.PI * 85; // radius = 85
    const offset = circumference - (score / 100) * circumference;
    
    // Add gradient definition if not exists
    let scoreGradient = document.getElementById('scoreGradientDef');
    if (!scoreGradient && elements.scoreProgress) {
        const svg = elements.scoreProgress.closest('svg');
        if (svg) {
            const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
            defs.innerHTML = `
                <linearGradient id="scoreGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="#00F5A0"/>
                    <stop offset="100%" stop-color="#00D9F5"/>
                </linearGradient>
            `;
            defs.id = 'scoreGradientDef';
            svg.insertBefore(defs, svg.firstChild);
        }
    }
    
    if (elements.scoreProgress) {
        elements.scoreProgress.style.stroke = 'url(#scoreGradient)';
        elements.scoreProgress.style.strokeDashoffset = offset;
    }
    if (elements.tradabilityScore) elements.tradabilityScore.textContent = score;
    
    // Update risk badge
    const riskClass = getRiskClass(data.risk_level);
    if (elements.riskBadge) {
        elements.riskBadge.className = `risk-badge ${riskClass}`;
    }
    if (elements.riskLevel) elements.riskLevel.textContent = data.risk_level;
    
    // Update explanation
    if (elements.tradabilityExplanation) {
        elements.tradabilityExplanation.innerHTML = `<p>${data.reasoning}</p>`;
    }
    
    // Update market data
    updateMarketData(data.market_data);
}

function updateMarketData(marketData) {
    if (!marketData) return;
    
    if (elements.currentPrice) elements.currentPrice.textContent = formatCurrency(marketData.current_price);
    
    if (marketData.price_change_24h !== null && marketData.price_change_24h !== undefined) {
        const changeText = formatPercent(marketData.price_change_24h);
        if (elements.priceChange) {
            elements.priceChange.textContent = changeText;
            elements.priceChange.className = `stat-value ${marketData.price_change_24h >= 0 ? 'positive' : 'negative'}`;
        }
    } else {
        if (elements.priceChange) {
            elements.priceChange.textContent = '--';
            elements.priceChange.className = 'stat-value';
        }
    }
    
    if (elements.volume24h) elements.volume24h.textContent = formatCurrency(marketData.volume_24h);
}

function updateMultiTimeframe(mtf) {
    if (!mtf) return;
    elements.confluenceScore.textContent = `${mtf.confluence.confluence_score}% Confluence`;
    elements.tfSummary.textContent = mtf.recommendation;
    
    // Update individual timeframes
    const tfMap = { '1W': elements.tf1w, '2W': elements.tf2w, '1M': elements.tf1m, '3M': elements.tf3m };
    Object.keys(tfMap).forEach(key => {
        if(mtf.timeframes[key]) {
            tfMap[key].textContent = mtf.timeframes[key].regime;
            // Add color class logic if needed
        }
    });
}

function updatePositionSizing(pos) {
    if (!pos) return;
    elements.positionMult.textContent = `${pos.position_multiplier}x`;
    elements.actionBadge.textContent = pos.action;
    elements.actionBadge.className = `card-badge actionable ${pos.action === 'BUY' || pos.action === 'STRONG_BUY' ? 'buy' : 'sell'}`;
    
    elements.stopLoss.textContent = `-${pos.risk_parameters.stop_loss_pct}%`;
    elements.takeProfit.textContent = `+${pos.risk_parameters.take_profit_pct}%`;
    
    elements.positionRationale.innerHTML = `<p>${pos.rationale}</p>`;
}

function updateWhaleTracking(whale) {
    if (!whale) return;
    elements.whaleActivity.textContent = whale.activity_level;
    elements.netFlow.textContent = formatCurrency(whale.net_flow);
    elements.whaleSentiment.textContent = whale.sentiment_label;
    
    if (whale.signals && whale.signals.length > 0) {
        elements.whaleSignal.textContent = whale.signals[0].description;
    } else {
        elements.whaleSignal.textContent = 'No significant whale signals detected.';
    }
}

function updatePrediction(pred) {
    if (!pred) return;
    // Sentiment (mapped from whale/mtf for now or passed if available)
    // Using whale sentiment score as proxy for Sentiment Meter if NLP not in comprehensive payload yet
    // Or we could fetch sentiment separately. For demo simplicity, use prediction confidence/regime.
    
    elements.predictionTarget.textContent = pred.predicted_regime;
    elements.predictionProb.textContent = `(Expected)`;
    
    // Mock sentiment for the meter for now since comprehensive payload is minimal
    elements.sentimentValue.textContent = "Bullish";
    elements.sentimentFill.style.width = "75%";
}

function updateMetrics(data) {
    if (!data || !data.regime_analysis) return;
    
    const analysis = data.regime_analysis;
    const vol = analysis.volatility_metrics || {};
    const liq = analysis.liquidity_metrics || {};
    const corr = analysis.correlation_metrics || {};
    
    // Volatility
    if (elements.volatilityValue) {
        const volVal = vol.current_volatility !== undefined ? vol.current_volatility : 0;
        elements.volatilityValue.textContent = `${(volVal * 100).toFixed(1)}%`;
        // Add trend arrow/color based on regime if available, or just value
    }
    
    // Correlation
    if (elements.correlationValue) {
        const corrVal = corr.average_correlation !== undefined ? corr.average_correlation : 0;
        elements.correlationValue.textContent = corrVal.toFixed(2);
    }
    
    // Liquidity
    if (elements.liquidityValue) {
        const liqScore = liq.liquidity_score !== undefined ? liq.liquidity_score : 0;
        elements.liquidityValue.textContent = `${liqScore.toFixed(0)}/100`;
    }
}


// Main Analysis Function
async function analyzeMarket() {
    if (state.isLoading) return;
    
    const assets = elements.assetInput.value.trim();
    if (!assets) { alert('Please enter an asset'); return; }
    
    const primaryAsset = assets.split(',')[0].trim();
    
    showLoading('Running AI Comprehensive Analysis...');
    
    try {
        await checkHealth();
        if (!state.apiConnected) throw new Error('API Disconnected');
        
        // Call the NEW Comprehensive Endpoint
        const data = await fetchAPI('/advanced/comprehensive', { asset: primaryAsset });
        
        console.log("Comprehensive Data:", data);
        state.lastData = data;
        
        updateDashboard(data);
        hideLoading();
        
    } catch (error) {
        console.error('Analysis failed:', error);
        hideLoading();
        alert(`Analysis failed: ${error.message}`);
    }
}

// WebSocket Handling
let ws = null;
const WS_BASE_URL = 'ws://localhost:8000/ws';

function connectWebSocket(asset) {
    if (ws) {
        ws.close();
    }
    
    // Use simulated stream if on localhost with no key, or just try connection
    // Ensure asset format is correct (lowercase)
    const wsUrl = `${WS_BASE_URL}/${asset.toLowerCase()}`;
    console.log(`Connecting to WebSocket: ${wsUrl}`);
    
    ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
        console.log('WebSocket Connected');
        const badge = document.getElementById('liveBadge');
        if (badge) badge.style.opacity = '1';
    };
    
    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data && data.price) {
                updateRealTimePrice(data);
            }
        } catch (e) {
            console.error('WS Parse Error:', e);
        }
    };
    
    ws.onerror = (error) => {
        console.warn('WebSocket Error:', error);
    };
    
    ws.onclose = () => {
        console.log('WebSocket Disconnected');
        const badge = document.getElementById('liveBadge');
        if (badge) badge.style.opacity = '0.5';
    };
}

function updateRealTimePrice(data) {
    // Flash effect or simple update
    if (elements.currentPrice) {
        // Only update if distinct to avoid flicker? 
        // Or simply update text.
        elements.currentPrice.textContent = `$${data.price.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
        elements.currentPrice.style.color = '#fff'; // Reset or flash
        
        // Simple flash effect
        setTimeout(() => elements.currentPrice.style.color = 'var(--primary)', 100);
        setTimeout(() => elements.currentPrice.style.color = '#fff', 300);
    }
}

// Initialize
async function init() {
    console.log('Crypto Regime Engine v2.0 - Initializing...');
    await checkHealth();
    setInterval(checkHealth, 10000);
    
    elements.assetInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') analyzeMarket();
    });

    // Auto-analyze default if desired, or just wait for user
}

// Enhance analyzeMarket to connect WS
const originalAnalyzeMarket = analyzeMarket;
analyzeMarket = async function() {
    const assets = elements.assetInput.value.trim();
    if (assets) {
        const primaryAsset = assets.split(',')[0].trim();
        connectWebSocket(primaryAsset);
    }
    await originalAnalyzeMarket();
};


document.addEventListener('DOMContentLoaded', init);
window.analyzeMarket = analyzeMarket;
window.analyzeTradability = analyzeMarket; // Map both to same logic for interactions


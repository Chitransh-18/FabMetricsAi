/**
 * WealthPortal Wafer Defect Detection Dashboard
 */

const API_BASE = window.location.origin;

const CLASS_COLORS = {
  'Center': '#f59e0b',
  'Donut': '#8b5cf6',
  'Edge-Loc': '#ef4444',
  'Edge-Ring': '#ec4899',
  'Loc': '#06b6d4',
  'Near-full': '#10b981',
  'Random': '#6366f1',
  'Scratch': '#f97316',
  'none': '#64748b',
};

let baselineChart = null;
let selectedFile = null;
let activeTab = 'file';

// ── DOM refs ──────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);

const statusDot = $('#status-dot');
const statusText = $('#status-text');
const dropZone = $('#drop-zone');
const fileInput = $('#file-input');
const fileName = $('#file-name');
const predictBtn = $('#predict-btn');
const errorBox = $('#error-box');
const resultCard = $('#result-card');
const placeholderCard = $('#placeholder-card');
const predictedClass = $('#predicted-class');
const confidenceBadge = $('#confidence-badge');
const deviceInfo = $('#device-info');
const shapeInfo = $('#shape-info');
const waferPreview = $('#wafer-preview');
const probList = $('#prob-list');
const advantageText = $('#advantage-text');

// ── Tabs ──────────────────────────────────────────────────
$('#tab-file').addEventListener('click', () => setTab('file'));
$('#tab-json').addEventListener('click', () => setTab('json'));

function setTab(tab) {
  activeTab = tab;
  $('#panel-file').classList.toggle('hidden', tab !== 'file');
  $('#panel-json').classList.toggle('hidden', tab !== 'json');
  $('#tab-file').classList.toggle('bg-portal-accent/20', tab === 'file');
  $('#tab-file').classList.toggle('text-portal-glow', tab === 'file');
  $('#tab-file').classList.toggle('text-slate-400', tab !== 'file');
  $('#tab-json').classList.toggle('bg-portal-accent/20', tab === 'json');
  $('#tab-json').classList.toggle('text-portal-glow', tab === 'json');
  $('#tab-json').classList.toggle('text-slate-400', tab !== 'json');
}

// ── File upload ───────────────────────────────────────────
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files.length) handleFile(fileInput.files[0]);
});

function handleFile(file) {
  selectedFile = file;
  fileName.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
}

// ── Health check ──────────────────────────────────────────
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    const data = await res.json();
    if (data.model_loaded) {
      statusDot.className = 'w-2 h-2 rounded-full bg-emerald-400';
      statusText.textContent = `Model ready · ${data.device}`;
      predictBtn.disabled = false;
    } else {
      statusDot.className = 'w-2 h-2 rounded-full bg-red-400';
      statusText.textContent = 'Model not loaded';
      predictBtn.disabled = true;
    }
  } catch {
    statusDot.className = 'w-2 h-2 rounded-full bg-red-400';
    statusText.textContent = 'API unreachable';
    predictBtn.disabled = true;
  }
}

// ── Baseline chart ────────────────────────────────────────
async function loadBaselines() {
  try {
    const res = await fetch(`${API_BASE}/api/baselines`);
    const data = await res.json();
    renderBaselineChart(data);
    advantageText.innerHTML =
      `ResNet-34 outperforms HOG+RF by <strong class="text-emerald-400">+${data.resnet34_advantage_over_hog_rf} pp</strong> ` +
      `and Shallow CNN by <strong class="text-emerald-400">+${data.resnet34_advantage_over_shallow_cnn} pp</strong> ` +
      `in validation Macro F1.`;
  } catch {
    advantageText.textContent = 'Could not load baseline metrics.';
  }
}

function renderBaselineChart(data) {
  const ctx = document.getElementById('baseline-chart').getContext('2d');
  const models = data.models;
  const labels = models.map((m) => m.name);
  const values = models.map((m) => m.macro_f1_percent);
  const colors = models.map((m) =>
    m.id === 'resnet34_transfer' ? 'rgba(59, 130, 246, 0.85)' :
    m.id === 'hog_random_forest' ? 'rgba(239, 68, 68, 0.65)' :
    'rgba(245, 158, 11, 0.65)'
  );

  if (baselineChart) baselineChart.destroy();

  baselineChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Validation Macro F1 (%)',
        data: values,
        backgroundColor: colors,
        borderColor: colors.map((c) => c.replace('0.85', '1').replace('0.65', '1')),
        borderWidth: 1,
        borderRadius: 6,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: 'y',
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            afterLabel: (ctx) => models[ctx.dataIndex]?.description || '',
          },
        },
      },
      scales: {
        x: {
          min: 0,
          max: 100,
          grid: { color: 'rgba(100,116,139,0.15)' },
          ticks: { color: '#94a3b8', callback: (v) => v + '%' },
          title: { display: true, text: 'Macro F1-Score (%)', color: '#94a3b8' },
        },
        y: {
          grid: { display: false },
          ticks: { color: '#cbd5e1', font: { size: 12 } },
        },
      },
    },
  });
}

// ── Inference ─────────────────────────────────────────────
predictBtn.addEventListener('click', runInference);

async function runInference() {
  errorBox.classList.add('hidden');
  predictBtn.disabled = true;
  predictBtn.textContent = 'Running…';

  try {
    let result;
    if (activeTab === 'file') {
      if (!selectedFile) throw new Error('Please select a wafer map image first.');
      const form = new FormData();
      form.append('file', selectedFile);
      const res = await fetch(`${API_BASE}/api/predict/upload`, { method: 'POST', body: form });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Request failed (${res.status})`);
      }
      result = await res.json();
    } else {
      const raw = $('#json-input').value.trim();
      if (!raw) throw new Error('Please paste a wafer_map JSON payload.');
      let body;
      try { body = JSON.parse(raw); } catch { throw new Error('Invalid JSON format.'); }
      const res = await fetch(`${API_BASE}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        const detail = typeof err.detail === 'string' ? err.detail : JSON.stringify(err.detail);
        throw new Error(detail || `Request failed (${res.status})`);
      }
      result = await res.json();
    }
    renderResult(result);
  } catch (err) {
    errorBox.textContent = err.message;
    errorBox.classList.remove('hidden');
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = 'Run Inference';
  }
}

// ── Render results ────────────────────────────────────────
function renderResult(data) {
  placeholderCard.classList.add('hidden');
  resultCard.classList.remove('hidden');

  const color = CLASS_COLORS[data.predicted_class] || '#3b82f6';
  predictedClass.textContent = data.predicted_class;
  predictedClass.style.color = color;
  confidenceBadge.textContent = `${data.confidence_percent}% confidence`;
  deviceInfo.textContent = data.device;
  shapeInfo.textContent = data.wafer_map_shape.join(' × ');

  renderWaferPreview(data.wafer_map_preview);
  renderProbabilities(data.probabilities, data.predicted_class);
}

function renderWaferPreview(grid) {
  waferPreview.innerHTML = '';
  const table = document.createElement('div');
  table.style.display = 'grid';
  table.style.gridTemplateColumns = `repeat(${grid[0].length}, min-content)`;
  table.style.gap = '0px';

  const colorMap = { 0: '#0f1729', 1: '#1e3a5f', 2: '#ef4444' };

  for (const row of grid) {
    for (const cell of row) {
      const div = document.createElement('div');
      div.className = 'wafer-cell';
      div.style.backgroundColor = colorMap[cell] ?? '#334155';
      table.appendChild(div);
    }
  }
  waferPreview.appendChild(table);
}

function renderProbabilities(probs, predicted) {
  probList.innerHTML = '';
  const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1]);

  for (const [name, prob] of sorted) {
    const pct = (prob * 100).toFixed(1);
    const isTop = name === predicted;
    const color = CLASS_COLORS[name] || '#3b82f6';

    const row = document.createElement('div');
    row.className = 'flex items-center gap-3';
    row.innerHTML = `
      <span class="text-xs font-mono w-24 truncate ${isTop ? 'text-white font-semibold' : 'text-slate-400'}">${name}</span>
      <div class="flex-1 h-5 bg-portal-900 rounded-full overflow-hidden border border-portal-700/40">
        <div class="prob-bar h-full rounded-full" style="width:${pct}%; background:${color}${isTop ? '' : '99'}"></div>
      </div>
      <span class="text-xs font-mono w-12 text-right ${isTop ? 'text-white' : 'text-slate-500'}">${pct}%</span>
    `;
    probList.appendChild(row);
  }
}

// ── Init ──────────────────────────────────────────────────
checkHealth();
loadBaselines();

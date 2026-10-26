/**
 * ColonVision AI — Clinical Research Workstation Controller
 * ==========================================================
 * Orchestrates navigation, interactive diagnostic canvases, real-time
 * Grad-CAM saliency blending, multi-model comparisons, and automated
 * ReportLab PDF generation via the FastAPI medical backend.
 */

// Application State
const state = {
  currentView: 'overview',
  selectedModel: 'densenet121',
  selectedSampleId: null,
  selectedFile: null,
  selectedColormap: 'jet',
  blendAlpha: 0.45,
  samples: {},
  lastAnalysis: null,
  lastComparison: null,
  zoomLevel: 1.0,
  viewportMode: 'overlay',
};

// Model Config Metadata
const MODEL_CONFIGS = {
  densenet121: {
    name: 'DenseNet-121',
    params: '8.0M',
    f1: '0.9328',
    targetLayer: 'features.denseblock4',
    badge: 'Optimal',
  },
  resnet50: {
    name: 'ResNet-50',
    params: '25.6M',
    f1: '1.0000',
    targetLayer: 'layer4[-1]',
    badge: 'Reference',
  },
  efficientnet_b0: {
    name: 'EfficientNet-B0',
    params: '5.3M',
    f1: '0.9328',
    targetLayer: 'features[-1]',
    badge: 'Lightweight',
  },
};

// ── DOM Elements ─────────────────────────────────────────────────────────────
const elements = {
  viewTitleText: document.getElementById('view-title-text'),
  workspacePreviewPanel: document.getElementById('workspace-preview-panel'),
  workspacePreviewImg: document.getElementById('workspace-preview-img'),
  workspaceFilename: document.getElementById('workspace-filename'),
  workspaceDim: document.getElementById('workspace-dim'),
  workspaceFilesize: document.getElementById('workspace-filesize'),
  workspaceStatusBadge: document.getElementById('workspace-status-badge'),
  workspaceDropzone: document.getElementById('workspace-dropzone'),
  workspaceFileInput: document.getElementById('workspace-file-input'),
  analyzeAlertBanner: document.getElementById('analyze-alert-banner'),
  analyzeAlertTitle: document.getElementById('analyze-alert-title'),
  analyzeAlertMsg: document.getElementById('analyze-alert-msg'),
  sliderOpacity: document.getElementById('slider-opacity'),
  sliderOpacityVal: document.getElementById('slider-opacity-val'),
  btnRunAnalysis: document.getElementById('btn-run-analysis'),
  runBtnIcon: document.getElementById('run-btn-icon'),
  runBtnText: document.getElementById('run-btn-text'),
  btnDownloadPdf: document.getElementById('btn-download-pdf'),
  overviewHistoryContainer: document.getElementById('overview-history-container'),
  
  // Results Elements
  resDomainScore: document.getElementById('res-domain-score'),
  resModelBadge: document.getElementById('res-model-badge'),
  resTimestamp: document.getElementById('res-timestamp'),
  resRunId: document.getElementById('res-run-id'),
  resTargetLayerChip: document.getElementById('res-target-layer-chip'),
  resPredTitle: document.getElementById('res-pred-title'),
  resPredDesc: document.getElementById('res-pred-desc'),
  resConfidencePill: document.getElementById('res-confidence-pill'),
  probValCancer: document.getElementById('prob-val-cancer'),
  probBarCancer: document.getElementById('prob-bar-cancer'),
  probValDiverticula: document.getElementById('prob-val-diverticula'),
  probBarDiverticula: document.getElementById('prob-bar-diverticula'),
  resBaseImg: document.getElementById('res-base-img'),
  resHeatmapImg: document.getElementById('res-heatmap-img'),
  resOverlayImg: document.getElementById('res-overlay-img'),
  saliencyReticleBox: document.getElementById('saliency-reticle-box'),
  reticleCoordsLabel: document.getElementById('reticle-coords-label'),
  spectrumLayerTag: document.getElementById('spectrum-layer-tag'),
  resAlphaSlider: document.getElementById('res-alpha-slider'),
  resAlphaDisplay: document.getElementById('res-alpha-display'),
  
  // Comparison Elements
  compareConsensusTitle: document.getElementById('compare-consensus-title'),
  compareConsensusDesc: document.getElementById('compare-consensus-desc'),
  compareCurrentFrameName: document.getElementById('compare-current-frame-name'),
  compareCurrentFrameDesc: document.getElementById('compare-current-frame-desc'),
  footprintDensenet: document.getElementById('footprint-densenet'),
  footprintResnet: document.getElementById('footprint-resnet'),
  footprintEfficientnet: document.getElementById('footprint-efficientnet'),
  footprintLoadingDensenet: document.getElementById('footprint-loading-densenet'),
  footprintLoadingResnet: document.getElementById('footprint-loading-resnet'),
  footprintLoadingEfficientnet: document.getElementById('footprint-loading-efficientnet'),
  footprintLayerDensenet: document.getElementById('footprint-layer-densenet'),
  footprintLayerResnet: document.getElementById('footprint-layer-resnet'),
  footprintLayerEfficientnet: document.getElementById('footprint-layer-efficientnet'),
  compareConfDensenet: document.getElementById('compare-conf-densenet'),
  comparePredDensenet: document.getElementById('compare-pred-densenet'),
  compareBarDensenet: document.getElementById('compare-bar-densenet'),
  compareLatDensenet: document.getElementById('compare-lat-densenet'),
  compareLayerDensenet: document.getElementById('compare-layer-densenet'),
  compareStatusBadgeDensenet: document.getElementById('compare-status-badge-densenet'),
  compareConfResnet: document.getElementById('compare-conf-resnet'),
  comparePredResnet: document.getElementById('compare-pred-resnet'),
  compareBarResnet: document.getElementById('compare-bar-resnet'),
  compareLatResnet: document.getElementById('compare-lat-resnet'),
  compareLayerResnet: document.getElementById('compare-layer-resnet'),
  compareStatusBadgeResnet: document.getElementById('compare-status-badge-resnet'),
  compareConfEfficientnet: document.getElementById('compare-conf-efficientnet'),
  comparePredEfficientnet: document.getElementById('compare-pred-efficientnet'),
  compareBarEfficientnet: document.getElementById('compare-bar-efficientnet'),
  compareLatEfficientnet: document.getElementById('compare-lat-efficientnet'),
  compareLayerEfficientnet: document.getElementById('compare-layer-efficientnet'),
  compareStatusBadgeEfficientnet: document.getElementById('compare-status-badge-efficientnet'),
};

// ── Centralized API Base URL Configuration ─────────────────────────────────────
// In local development, empty VITE_API_BASE_URL delegates to Vite dev server proxy (/api -> :8000).
// In Vercel deployment, VITE_API_BASE_URL points directly to Render (https://colonvision-ai.onrender.com).
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || '').trim().replace(/\/+$/, '');

/**
 * Resolves an API path or image URL against the configured backend base URL.
 * Handles relative paths ('/api/...'), absolute URLs ('https://...'), and data URIs safely.
 */
function apiUrl(path) {
  if (!path) return '';
  if (/^https?:\/\//i.test(path) || path.startsWith('data:')) {
    return path;
  }
  const cleanPath = path.startsWith('/') ? path : `/${path}`;
  return API_BASE_URL ? `${API_BASE_URL}${cleanPath}` : cleanPath;
}

/**
 * Safely executes an API request with strict HTTP status and content-type validation.
 * Resolves endpoints against the centralized API base URL (VITE_API_BASE_URL).
 * Prevents "Unexpected token '<', '<!DOCTYPE '... is not valid JSON" parsing syntax errors
 * by validating that the response is genuine JSON before calling res.json().
 *
 * If the response fails or returns non-JSON (e.g. 502/504 Bad Gateway, 404, or SPA redirect),
 * this helper extracts meaningful clinical/server error information without masking genuine PyTorch errors.
 */
async function safeApiRequest(endpoint, options = {}) {
  const url = apiUrl(endpoint);
  const timeoutMs = options.timeoutMs || 60000;
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  const signal = options.signal || controller.signal;

  let res;
  try {
    res = await fetch(url, { ...options, signal });
  } catch (networkErr) {
    clearTimeout(timeoutId);
    if (networkErr.name === 'AbortError') {
      const err = new Error(
        `Request to ${url} timed out after ${timeoutMs / 1000}s. The Render free-tier backend may be warming up from cold-start; please allow 15 seconds and retry.`
      );
      err.errorType = 'Gateway Timeout';
      throw err;
    }
    const err = new Error(`Network connection error: Unable to reach PyTorch backend at ${url}. ${networkErr.message}`);
    err.errorType = 'Connection Error';
    throw err;
  } finally {
    clearTimeout(timeoutId);
  }

  const contentType = res.headers.get('content-type') || '';
  const isJson = contentType.includes('application/json');

  if (!res.ok) {
    if (isJson) {
      let errData;
      try {
        errData = await res.json();
      } catch (_) {
        throw new Error(`Server returned HTTP ${res.status} ${res.statusText}`);
      }
      const message = errData.message || errData.detail || errData.error || `Server returned HTTP ${res.status} ${res.statusText}`;
      const err = new Error(typeof message === 'object' ? JSON.stringify(message) : message);
      err.errorType = errData.error || `HTTP ${res.status} Error`;
      err.data = errData;
      err.status = res.status;
      throw err;
    } else {
      // Non-JSON response (e.g. Render HTML 502 Bad Gateway / 504 Timeout / 404 Not Found)
      const rawText = await res.text().catch(() => '');
      let helpfulMsg = `Backend server returned HTTP ${res.status} ${res.statusText}`;
      if (res.status === 502 || res.status === 504) {
        helpfulMsg = `PyTorch backend service is currently warming up or temporarily unavailable (HTTP ${res.status}). Please allow 10-15 seconds and try again.`;
      } else if (res.status === 404) {
        helpfulMsg = `The requested endpoint '${url}' was not found (HTTP 404). Ensure backend base URL points to the active FastAPI server.`;
      } else if (rawText && !rawText.includes('<!DOCTYPE') && rawText.length < 200) {
        helpfulMsg += `: ${rawText.trim()}`;
      }
      const err = new Error(helpfulMsg);
      err.errorType = `Server Error (HTTP ${res.status})`;
      err.status = res.status;
      throw err;
    }
  }

  // Response status is 2xx OK
  if (!isJson) {
    // If an HTML document was returned with 200 OK (e.g. SPA index.html intercept)
    const err = new Error(
      `Received non-JSON response from ${url} (Content-Type: ${contentType}). The endpoint may be intercepted by a static web server.`
    );
    err.errorType = 'Invalid Response Format';
    throw err;
  }

  try {
    return await res.json();
  } catch (jsonErr) {
    const err = new Error(`Failed to parse backend JSON response: ${jsonErr.message}`);
    err.errorType = 'JSON Parse Error';
    throw err;
  }
}

// ── Navigation Router ────────────────────────────────────────────────────────
function switchView(viewName) {
  state.currentView = viewName;

  // Hide all view panels
  document.querySelectorAll('.view-panel').forEach(panel => {
    panel.classList.remove('active');
  });

  // Activate target panel
  const targetPanel = document.getElementById(`view-${viewName}`);
  if (targetPanel) {
    targetPanel.classList.add('active');
  }

  // Update Breadcrumb & Header Title
  const titles = {
    overview: 'Overview',
    analyze: 'Analyze Workspace',
    results: 'Results & Explainability',
    models: 'Comparative Models',
    about: 'Academic Rigor',
  };
  if (elements.viewTitleText) {
    elements.viewTitleText.textContent = titles[viewName] || 'Clinical Platform';
  }

  // Update Clinical Drawer Navigation Items
  document.querySelectorAll('.drawer-nav-item').forEach(btn => {
    if (btn.dataset.drawerTarget === viewName) {
      btn.classList.add('active');
    } else {
      btn.classList.remove('active');
    }
  });

  // Update Desktop Navigation Bar Active State
  document.querySelectorAll('.desktop-nav-btn').forEach(btn => {
    if (btn.dataset.target === viewName) {
      btn.classList.add('text-primary-container', 'bg-surface-container-lowest', 'shadow-xs');
      btn.classList.remove('text-secondary');
    } else {
      btn.classList.remove('text-primary-container', 'bg-surface-container-lowest', 'shadow-xs');
      btn.classList.add('text-secondary');
    }
  });

  // Update Mobile Navigation Bar Active State
  document.querySelectorAll('.mobile-nav-btn').forEach(btn => {
    if (btn.dataset.target === viewName) {
      btn.classList.add('active', 'text-primary-container');
      btn.classList.remove('text-secondary');
    } else {
      btn.classList.remove('active', 'text-primary-container');
      btn.classList.add('text-secondary');
    }
  });

  // Handle Results View Empty State
  if (viewName === 'results') {
    const emptyState = document.getElementById('results-empty-state');
    const contentWrapper = document.getElementById('results-content-wrapper');
    if (!state.lastAnalysis) {
      emptyState?.classList.remove('hidden');
      contentWrapper?.classList.add('hidden');
      contentWrapper?.classList.remove('flex');
    } else {
      emptyState?.classList.add('hidden');
      contentWrapper?.classList.remove('hidden');
      contentWrapper?.classList.add('flex');
    }
  }

  // If entering comparative models and an image is ready, run comparison automatically
  if (viewName === 'models') {
    triggerMultiModelComparison();
  }

  // Smooth scroll to top
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ── Preloaded Research Samples ───────────────────────────────────────────────
async function loadSamples() {
  try {
    const data = await safeApiRequest('/api/samples');
    if (data && data.samples) {
      data.samples.forEach(s => {
        state.samples[s.id] = s;
      });
    }
    if (data && data.missing_count > 0) {
      console.warn('Some benchmark samples are missing on server:', data.missing_files);
    }
  } catch (err) {
    console.error('Samples loading error:', err);
  }
}

async function applySample(sampleId) {
  let sample = state.samples[sampleId];
  if (!sample) {
    await loadSamples();
    sample = state.samples[sampleId];
  }

  if (!sample) {
    showAlert('Sample Unavailable', `Benchmark sample '${sampleId}' could not be loaded from backend.`);
    return;
  }

  if (sample.available === false || sample.error) {
    showAlert('Sample File Missing', sample.error || `Benchmark sample file for '${sampleId}' not found on server.`);
    return;
  }

  state.selectedSampleId = sampleId;
  state.selectedFile = null;
  if (elements.workspaceFileInput) {
    elements.workspaceFileInput.value = '';
  }

  // Highlight active sample button
  const cancerBtn = document.getElementById('sample-cancer-btn');
  const divertBtn = document.getElementById('sample-diverticula-btn');
  if (cancerBtn && divertBtn) {
    if (sampleId === 'cancer') {
      cancerBtn.classList.add('sample-active');
      divertBtn.classList.remove('sample-active');
    } else {
      divertBtn.classList.add('sample-active');
      cancerBtn.classList.remove('sample-active');
    }
  }

  // Reveal inspection preview panel and hide empty dropzone
  if (elements.workspacePreviewPanel) {
    elements.workspacePreviewPanel.classList.remove('hidden');
  }
  if (elements.workspaceDropzone) {
    elements.workspaceDropzone.classList.add('hidden');
  }

  // Prefer direct HTTP image URL on deployed domain, fallback to base64
  const rawImageSrc = sample.image_url || sample.static_url || sample.data_uri || `/api/samples/image/${sampleId}`;
  const imageSrc = apiUrl(rawImageSrc);
  elements.workspacePreviewImg.src = imageSrc;
  elements.workspaceFilename.textContent = sample.filename || `${sampleId}.jpg`;
  elements.workspaceDim.textContent = sample.dimensions || '—';
  elements.workspaceFilesize.textContent = sample.filesize || '—';
  elements.workspaceStatusBadge.textContent = 'Validated';
  elements.workspaceStatusBadge.className = 'px-2.5 py-0.5 rounded bg-primary-container text-on-primary-container font-label-sm text-label-sm uppercase flex-shrink-0 font-medium font-mono';

  hideAlert();
}

// ── Reset & Clean Upload State ───────────────────────────────────────────────
function resetWorkspaceState(resetAnalysis = false) {
  if (elements.workspaceFileInput) {
    elements.workspaceFileInput.value = '';
  }
  state.selectedFile = null;
  state.selectedSampleId = null;

  // Clear sample button active highlights
  document.querySelectorAll('.sample-select-btn').forEach(b => b.classList.remove('sample-active'));

  // Reset preview panel DOM
  if (elements.workspacePreviewImg) {
    elements.workspacePreviewImg.src = '';
  }
  if (elements.workspaceFilename) {
    elements.workspaceFilename.textContent = 'No frame loaded';
  }
  if (elements.workspaceDim) {
    elements.workspaceDim.textContent = '—';
  }
  if (elements.workspaceFilesize) {
    elements.workspaceFilesize.textContent = '—';
  }
  if (elements.workspaceStatusBadge) {
    elements.workspaceStatusBadge.textContent = 'Pending';
    elements.workspaceStatusBadge.className = 'px-2.5 py-0.5 rounded bg-surface-container text-secondary font-label-sm text-label-sm uppercase flex-shrink-0 font-medium font-mono';
  }

  // Return to clean upload state: hide preview panel and show dropzone with Browse Image button
  if (elements.workspacePreviewPanel) {
    elements.workspacePreviewPanel.classList.add('hidden');
  }
  if (elements.workspaceDropzone) {
    elements.workspaceDropzone.classList.remove('hidden');
  }

  hideAlert();

  if (resetAnalysis) {
    state.lastAnalysis = null;
    const emptyState = document.getElementById('results-empty-state');
    const contentWrapper = document.getElementById('results-content-wrapper');
    emptyState?.classList.remove('hidden');
    contentWrapper?.classList.add('hidden');
    contentWrapper?.classList.remove('flex');
  }
}

// ── Dropzone & File Upload ───────────────────────────────────────────────────
function initDropzone() {
  const dropzone = elements.workspaceDropzone;
  const fileInput = elements.workspaceFileInput;
  const browseTrigger = document.getElementById('btn-browse-trigger');

  if (browseTrigger) {
    browseTrigger.addEventListener('click', (e) => {
      e.stopPropagation();
      fileInput.click();
    });
  }

  if (dropzone) {
    dropzone.addEventListener('click', () => fileInput.click());

    dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropzone.classList.add('border-primary', 'bg-surface-container');
    });

    dropzone.addEventListener('dragleave', () => {
      dropzone.classList.remove('border-primary', 'bg-surface-container');
    });

    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('border-primary', 'bg-surface-container');
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleUserFile(e.dataTransfer.files[0]);
      }
    });
  }

  if (fileInput) {
    fileInput.addEventListener('change', () => {
      if (fileInput.files && fileInput.files[0]) {
        handleUserFile(fileInput.files[0]);
      }
    });
  }

  // Replace / Remove / Reset buttons (Only shown inside active preview panel)
  document.getElementById('btn-workspace-replace')?.addEventListener('click', () => {
    fileInput.click();
  });

  document.getElementById('btn-workspace-remove')?.addEventListener('click', () => {
    resetWorkspaceState(false);
  });

  document.getElementById('btn-workspace-reset')?.addEventListener('click', () => {
    resetWorkspaceState(true);
  });
}

async function handleUserFile(file) {
  // Validate file type
  if (!['image/jpeg', 'image/png', 'image/jpg'].includes(file.type)) {
    showAlert('Unsupported File Type', 'Please upload a colonoscopy frame in JPEG or PNG format.');
    return;
  }
  // Validate file size (< 15MB)
  if (file.size > 15 * 1024 * 1024) {
    showAlert('File Size Limit Exceeded', 'Frame exceeds the 15MB clinical processing limit.');
    return;
  }

  // Clear sample button highlights
  document.querySelectorAll('.sample-select-btn').forEach(b => b.classList.remove('sample-active'));

  state.selectedFile = file;
  state.selectedSampleId = null;

  // Read file as Data URL
  const reader = new FileReader();
  reader.onload = async (e) => {
    const dataUri = e.target.result;

    // Show preview panel and hide dropzone with Browse Image button
    if (elements.workspacePreviewPanel) {
      elements.workspacePreviewPanel.classList.remove('hidden');
    }
    if (elements.workspaceDropzone) {
      elements.workspaceDropzone.classList.add('hidden');
    }

    elements.workspacePreviewImg.src = dataUri;
    elements.workspaceFilename.textContent = file.name;
    const sizeMb = (file.size / (1024 * 1024)).toFixed(2);
    elements.workspaceFilesize.textContent = `${sizeMb} MB`;

    // Measure dimensions
    const img = new Image();
    img.onload = () => {
      elements.workspaceDim.textContent = `${img.naturalWidth} × ${img.naturalHeight} px`;
    };
    img.src = dataUri;

    // Run domain validation check
    await validateFile(file);
  };
  reader.readAsDataURL(file);
}

async function validateFile(file) {
  try {
    elements.workspaceStatusBadge.textContent = 'Verifying...';
    elements.workspaceStatusBadge.className = 'px-2.5 py-0.5 rounded bg-surface-container text-primary font-label-sm text-label-sm uppercase flex-shrink-0 font-medium font-mono animate-pulse';

    const formData = new FormData();
    formData.append('file', file);

    const data = await safeApiRequest('/api/validate', {
      method: 'POST',
      body: formData,
    });

    if (data.is_endoscopy) {
      elements.workspaceStatusBadge.textContent = 'Validated';
      elements.workspaceStatusBadge.className = 'px-2.5 py-0.5 rounded bg-primary-container text-on-primary-container font-label-sm text-label-sm uppercase flex-shrink-0 font-medium font-mono';
      hideAlert();
    } else {
      elements.workspaceStatusBadge.textContent = 'OOD Rejected';
      elements.workspaceStatusBadge.className = 'px-2.5 py-0.5 rounded bg-error text-white font-label-sm text-label-sm uppercase flex-shrink-0 font-medium font-mono';
      showAlert(
        'Out-of-Domain Image Detected',
        `The submitted image failed endoscopy domain verification (Score: ${data.domain_score.toFixed(4)} < Threshold: ${data.threshold.toFixed(4)}). Please upload an authentic endoscopic colonoscopy frame.`
      );
    }
  } catch (err) {
    console.error('Validation error:', err);
    elements.workspaceStatusBadge.textContent = 'Check Failed';
    elements.workspaceStatusBadge.className = 'px-2.5 py-0.5 rounded bg-amber-500 text-white font-label-sm text-label-sm uppercase flex-shrink-0 font-medium font-mono';
    showAlert(err.errorType || 'Validation Warning', err.message);
  }
}

function showAlert(title, msg) {
  elements.analyzeAlertTitle.textContent = title;
  elements.analyzeAlertMsg.textContent = msg;
  elements.analyzeAlertBanner.classList.remove('hidden');
}

function hideAlert() {
  elements.analyzeAlertBanner.classList.add('hidden');
}

// ── Model Selection Controls ─────────────────────────────────────────────────
function initModelSelector() {
  const radios = document.querySelectorAll('input[name="model_choice"]');
  radios.forEach(radio => {
    radio.addEventListener('change', () => {
      state.selectedModel = radio.value;

      // Update card visual borders
      document.querySelectorAll('.model-radio-card').forEach(card => {
        card.classList.remove('bg-secondary-container/60', 'border-2', 'border-primary-container');
        card.classList.add('bg-surface-container-low', 'border', 'border-outline-variant/40');
      });

      const parentCard = radio.closest('.model-radio-card');
      if (parentCard) {
        parentCard.classList.remove('bg-surface-container-low', 'border-outline-variant/40');
        parentCard.classList.add('bg-secondary-container/60', 'border-2', 'border-primary-container');
      }
    });
  });
}

// ── Saliency Opacity & Colormaps ──────────────────────────────────────────────
function initSaliencyControls() {
  elements.sliderOpacity.addEventListener('input', (e) => {
    state.blendAlpha = parseFloat(e.target.value) / 100.0;
    elements.sliderOpacityVal.textContent = `${e.target.value}%`;
  });

  // Colormap toggle buttons
  document.querySelectorAll('.cmap-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.cmap-btn').forEach(b => {
        b.className = 'cmap-btn px-2.5 py-1 rounded text-[11px] font-mono text-secondary hover:text-on-surface';
      });
      btn.className = 'cmap-btn px-2.5 py-1 rounded text-[11px] font-mono font-semibold bg-primary-container text-on-primary shadow-xs';
      state.selectedColormap = btn.dataset.cmap;
    });
  });
}

// ── Run Saliency Analysis ────────────────────────────────────────────────────
function initAnalysisExecution() {
  elements.btnRunAnalysis.addEventListener('click', async () => {
    if (!state.selectedFile && !state.selectedSampleId) {
      showAlert('No Image Selected', 'Please select a preloaded benchmark sample or upload a colonoscopy frame.');
      return;
    }

    // Set Loading State
    elements.btnRunAnalysis.disabled = true;
    elements.runBtnIcon.textContent = 'progress_activity';
    elements.runBtnIcon.classList.add('animate-spin');
    elements.runBtnText.textContent = 'Analyzing Neural Activations...';
    hideAlert();

    const formData = new FormData();
    formData.append('model_choice', state.selectedModel);
    formData.append('alpha', state.blendAlpha);
    formData.append('colormap', state.selectedColormap);

    if (state.selectedFile) {
      formData.append('file', state.selectedFile);
    } else if (state.selectedSampleId) {
      formData.append('sample_id', state.selectedSampleId);
    }

    try {
      const result = await safeApiRequest('/api/analyze', {
        method: 'POST',
        body: formData,
      });

      state.lastAnalysis = result;

      // Populate Results View
      renderAnalysisResults(result);

      // Switch to Results
      switchView('results');

      // Refresh overview history list
      loadHistory();
    } catch (err) {
      console.error('Inference execution error:', err);
      showAlert(err.errorType || 'Inference Error', err.message);
    } finally {
      elements.btnRunAnalysis.disabled = false;
      elements.runBtnIcon.textContent = 'biotech';
      elements.runBtnIcon.classList.remove('animate-spin');
      elements.runBtnText.textContent = 'Run Saliency Analysis';
    }
  });
}

function renderAnalysisResults(res) {
  // Reveal content wrapper and hide empty state
  const emptyState = document.getElementById('results-empty-state');
  const contentWrapper = document.getElementById('results-content-wrapper');
  emptyState?.classList.add('hidden');
  contentWrapper?.classList.remove('hidden');
  contentWrapper?.classList.add('flex');

  // Domain score & metadata
  elements.resDomainScore.textContent = `Endoscopy (${res.domain_score.toFixed(4)} >= ${res.threshold.toFixed(4)})`;
  elements.resModelBadge.textContent = `${res.model_name}`;
  elements.resTimestamp.textContent = `Analyzed ${res.timestamp}`;
  elements.resRunId.textContent = `Run ID: #${res.run_id}`;
  elements.resTargetLayerChip.textContent = res.target_layer;
  elements.spectrumLayerTag.textContent = `${res.model_name} ${res.target_layer}`;

  // Prediction Titles
  const isCancer = res.prediction_raw === 'colorectal_cancer';
  elements.resPredTitle.textContent = isCancer
    ? 'Colorectal Carcinoma Features Detected'
    : 'Colon Diverticula Mucosal Alteration Detected';

  elements.resPredDesc.textContent = isCancer
    ? 'Inference score, not a definitive clinical diagnosis. Neural activations match polypoid and mucosal architectural disruption.'
    : 'Inference score, not a definitive clinical diagnosis. Neural activations identify colon diverticular pouching orifices and mucosal folding.';

  elements.resConfidencePill.textContent = `${res.confidence_pct} Confidence`;

  // Probabilities
  const pCancer = (res.probabilities.colorectal_cancer * 100).toFixed(1);
  const pDivert = (res.probabilities.colon_diverticula * 100).toFixed(1);

  elements.probValCancer.textContent = `${pCancer}%`;
  elements.probBarCancer.style.width = `${pCancer}%`;

  elements.probValDiverticula.textContent = `${pDivert}%`;
  elements.probBarDiverticula.style.width = `${pDivert}%`;

  // Images in Saliency Viewport
  elements.resBaseImg.src = res.original_uri;
  elements.resHeatmapImg.src = res.heatmap_uri;
  elements.resOverlayImg.src = res.overlay_uri;

  // Reticle positioning
  const reticleX = Math.round(res.peak_x * 100);
  const reticleY = Math.round(res.peak_y * 100);
  elements.saliencyReticleBox.style.left = `${reticleX}%`;
  elements.saliencyReticleBox.style.top = `${reticleY}%`;
  elements.reticleCoordsLabel.textContent = `Peak Attention (x:${res.peak_x}, y:${res.peak_y})`;

  // Reset viewport mode to overlay
  setViewportMode('overlay');
}

// ── Results Viewport Modes & Zoom ────────────────────────────────────────────
function initResultsViewport() {
  // Mode tabs
  document.querySelectorAll('.mode-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      setViewportMode(tab.dataset.mode);
    });
  });

  // Dynamic Alpha slider in results
  elements.resAlphaSlider.addEventListener('input', (e) => {
    const alpha = parseFloat(e.target.value) / 100.0;
    elements.resAlphaDisplay.textContent = `${e.target.value}%`;
    elements.resOverlayImg.style.opacity = alpha;
    elements.resHeatmapImg.style.opacity = alpha;
  });

  // Zoom controls
  document.getElementById('btn-zoom-in').addEventListener('click', () => {
    state.zoomLevel = Math.min(2.5, state.zoomLevel + 0.25);
    applyZoom();
  });

  document.getElementById('btn-zoom-out').addEventListener('click', () => {
    state.zoomLevel = Math.max(1.0, state.zoomLevel - 0.25);
    applyZoom();
  });

  document.getElementById('btn-zoom-reset').addEventListener('click', () => {
    state.zoomLevel = 1.0;
    applyZoom();
  });

  // Navigate to compare from results
  document.getElementById('btn-compare-from-results').addEventListener('click', () => {
    switchView('models');
  });

  // Start new analysis button
  const btnNewAnalysis = document.getElementById('btn-new-analysis');
  if (btnNewAnalysis) {
    btnNewAnalysis.addEventListener('click', () => {
      switchView('analyze');
    });
  }

  // PDF Download Action
  elements.btnDownloadPdf.addEventListener('click', downloadPdfReport);
}

function setViewportMode(mode) {
  state.viewportMode = mode;

  document.querySelectorAll('.mode-tab').forEach(t => {
    if (t.dataset.mode === mode) {
      t.className = 'mode-tab flex-1 py-1.5 rounded text-center font-label-sm text-label-sm font-semibold transition-all bg-primary-container text-on-primary shadow-xs';
    } else {
      t.className = 'mode-tab flex-1 py-1.5 rounded text-center font-label-sm text-label-sm text-secondary hover:text-on-surface transition-all';
    }
  });

  if (mode === 'overlay') {
    elements.resBaseImg.style.opacity = '1';
    elements.resHeatmapImg.style.display = 'none';
    elements.resOverlayImg.style.display = 'block';
    elements.resOverlayImg.style.opacity = '1';
    elements.saliencyReticleBox.style.display = 'flex';
  } else if (mode === 'original') {
    elements.resBaseImg.style.opacity = '1';
    elements.resHeatmapImg.style.display = 'none';
    elements.resOverlayImg.style.display = 'none';
    elements.saliencyReticleBox.style.display = 'none';
  } else if (mode === 'heatmap') {
    elements.resBaseImg.style.opacity = '0';
    elements.resHeatmapImg.style.display = 'block';
    elements.resHeatmapImg.style.opacity = '1';
    elements.resOverlayImg.style.display = 'none';
    elements.saliencyReticleBox.style.display = 'flex';
  }
}

function applyZoom() {
  const transform = `scale(${state.zoomLevel})`;
  elements.resBaseImg.style.transform = transform;
  elements.resHeatmapImg.style.transform = transform;
  elements.resOverlayImg.style.transform = transform;
}

// ── PDF Report Generation ────────────────────────────────────────────────────
async function downloadPdfReport() {
  if (!state.lastAnalysis) {
    alert('Please execute an image analysis first before generating a report.');
    return;
  }

  const btn = elements.btnDownloadPdf;
  const originalHtml = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = `
    <span class="material-symbols-outlined text-[18px] animate-spin">progress_activity</span>
    <span>Generating Clinical PDF...</span>
  `;

  try {
    const payload = {
      model_name: state.lastAnalysis.model_name,
      prediction: state.lastAnalysis.prediction,
      confidence: state.lastAnalysis.confidence,
      prob_cancer: state.lastAnalysis.probabilities.colorectal_cancer,
      prob_diverticula: state.lastAnalysis.probabilities.colon_diverticula,
      validation_score: state.lastAnalysis.domain_score,
      threshold: state.lastAnalysis.threshold,
      original_uri: state.lastAnalysis.original_uri,
      heatmap_uri: state.lastAnalysis.heatmap_uri,
      overlay_uri: state.lastAnalysis.overlay_uri,
    };

    const res = await fetch(apiUrl('/api/report'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const contentType = res.headers.get('content-type') || '';
      let errMsg = `Server returned HTTP ${res.status} ${res.statusText}`;
      if (contentType.includes('application/json')) {
        const errJson = await res.json().catch(() => ({}));
        errMsg = errJson.message || errJson.detail || errMsg;
      }
      throw new Error(errMsg);
    }

    const blob = await res.blob();
    const downloadUrl = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = downloadUrl;
    a.download = `ColonVision_AI_Report_${state.lastAnalysis.run_id}.pdf`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(downloadUrl);
  } catch (err) {
    console.error('PDF download error:', err);
    alert('Failed to generate PDF report: ' + err.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = originalHtml;
  }
}

// ── Multi-Model Comparison Execution ─────────────────────────────────────────
async function triggerMultiModelComparison() {
  const models = [
    {
      key: 'densenet121',
      name: 'DenseNet-121',
      img: elements.footprintDensenet,
      loading: elements.footprintLoadingDensenet,
      layerFootprint: elements.footprintLayerDensenet,
      conf: elements.compareConfDensenet,
      pred: elements.comparePredDensenet,
      bar: elements.compareBarDensenet,
      lat: elements.compareLatDensenet,
      layerCard: elements.compareLayerDensenet,
      badge: elements.compareStatusBadgeDensenet,
    },
    {
      key: 'resnet50',
      name: 'ResNet-50',
      img: elements.footprintResnet,
      loading: elements.footprintLoadingResnet,
      layerFootprint: elements.footprintLayerResnet,
      conf: elements.compareConfResnet,
      pred: elements.comparePredResnet,
      bar: elements.compareBarResnet,
      lat: elements.compareLatResnet,
      layerCard: elements.compareLayerResnet,
      badge: elements.compareStatusBadgeResnet,
    },
    {
      key: 'efficientnet_b0',
      name: 'EfficientNet-B0',
      img: elements.footprintEfficientnet,
      loading: elements.footprintLoadingEfficientnet,
      layerFootprint: elements.footprintLayerEfficientnet,
      conf: elements.compareConfEfficientnet,
      pred: elements.comparePredEfficientnet,
      bar: elements.compareBarEfficientnet,
      lat: elements.compareLatEfficientnet,
      layerCard: elements.compareLayerEfficientnet,
      badge: elements.compareStatusBadgeEfficientnet,
    },
  ];

  // 1. Verify frame selection
  if (!state.selectedFile && !state.selectedSampleId) {
    if (elements.compareCurrentFrameName) {
      elements.compareCurrentFrameName.textContent = 'No frame selected';
    }
    if (elements.compareCurrentFrameDesc) {
      elements.compareCurrentFrameDesc.textContent = 'Please choose a benchmark sample below or upload an image to run multi-model comparative inference.';
    }
    if (elements.compareConsensusTitle) {
      elements.compareConsensusTitle.textContent = 'Comparative Consensus: Awaiting Frame Selection';
    }
    if (elements.compareConsensusDesc) {
      elements.compareConsensusDesc.textContent = 'Select a benchmark sample or upload a colonoscopy frame on the Analyze page to compare attribution footprints across architectures.';
    }
    models.forEach(m => {
      if (m.img) m.img.classList.add('hidden');
      if (m.loading) {
        m.loading.classList.remove('hidden');
        m.loading.innerHTML = `
          <span class="material-symbols-outlined text-[24px] text-secondary">center_focus_strong</span>
          <span class="font-label-sm text-[11px] text-secondary">Awaiting frame selection</span>
        `;
      }
      if (m.badge) {
        m.badge.textContent = 'Standby';
        m.badge.className = 'px-1.5 py-0.5 rounded bg-surface-container text-secondary font-label-sm text-[10px] font-mono';
      }
      if (m.pred) {
        m.pred.textContent = 'Standby';
        m.pred.className = 'font-label-sm text-label-sm text-secondary';
      }
      if (m.conf) m.conf.textContent = '—';
      if (m.lat) m.lat.textContent = '—';
      if (m.bar) m.bar.style.width = '0%';
    });
    return;
  }

  // 2. Update frame banner
  if (state.selectedFile) {
    if (elements.compareCurrentFrameName) elements.compareCurrentFrameName.textContent = state.selectedFile.name;
    if (elements.compareCurrentFrameDesc) elements.compareCurrentFrameDesc.textContent = 'Custom upload frame • Ready for multi-model cross-architecture evaluation.';
  } else if (state.selectedSampleId) {
    const sMeta = state.samples[state.selectedSampleId];
    if (elements.compareCurrentFrameName) elements.compareCurrentFrameName.textContent = sMeta ? sMeta.filename : `${state.selectedSampleId}.jpg`;
    if (elements.compareCurrentFrameDesc) elements.compareCurrentFrameDesc.textContent = sMeta ? sMeta.morphology : 'Clinical research benchmark sample.';
  }

  // 3. Set Loading State on All Models
  const btnRerun = document.getElementById('btn-rerun-compare');
  const rerunIcon = document.getElementById('rerun-compare-icon');
  const rerunText = document.getElementById('rerun-compare-text');
  if (btnRerun) btnRerun.disabled = true;
  rerunIcon?.classList.add('animate-spin');
  if (rerunText) rerunText.textContent = 'Comparing Models...';

  if (elements.compareConsensusTitle) {
    elements.compareConsensusTitle.textContent = 'Multi-Model Inference in Progress';
  }
  if (elements.compareConsensusDesc) {
    elements.compareConsensusDesc.textContent = 'Evaluating DenseNet-121, ResNet-50, and EfficientNet-B0 activations on the current frame...';
  }

  models.forEach(m => {
    if (m.img) m.img.classList.add('hidden');
    if (m.loading) {
      m.loading.classList.remove('hidden');
      m.loading.innerHTML = `
        <span class="material-symbols-outlined text-[24px] text-primary-container animate-spin">progress_activity</span>
        <span class="font-label-sm text-[11px] text-primary-container font-mono">Computing activations...</span>
      `;
    }
    if (m.badge) {
      m.badge.textContent = 'Evaluating';
      m.badge.className = 'px-1.5 py-0.5 rounded bg-primary-container/20 text-primary-container font-label-sm text-[10px] font-mono animate-pulse';
    }
    if (m.pred) {
      m.pred.textContent = 'Computing...';
      m.pred.className = 'font-label-sm text-label-sm text-primary font-mono';
    }
    if (m.conf) m.conf.textContent = '...';
    if (m.lat) m.lat.textContent = '...';
    if (m.bar) m.bar.style.width = '0%';
  });

  // 4. Execute Backend Comparative Request
  const formData = new FormData();
  if (state.selectedFile) {
    formData.append('file', state.selectedFile);
  } else if (state.selectedSampleId) {
    formData.append('sample_id', state.selectedSampleId);
  }

  try {
    const data = await safeApiRequest('/api/compare', {
      method: 'POST',
      body: formData,
    });
    state.lastComparison = data;

    // Consensus Banner
    if (data.consensus) {
      if (elements.compareConsensusTitle) {
        elements.compareConsensusTitle.textContent = data.consensus.agreement_rate === 1.0
          ? 'Model Consensus: High Agreement'
          : (data.consensus.agreement_rate > 0.5 ? 'Model Consensus: Majority Agreement' : 'Model Consensus: Inconclusive');
      }
      if (elements.compareConsensusDesc) {
        elements.compareConsensusDesc.innerHTML = `${data.consensus.summary} (Concordance κ = ${data.consensus.kappa}).`;
      }
    }

    // Process each model independently (fault tolerance)
    models.forEach(m => {
      const res = data.models ? data.models[m.key] : null;
      if (res && res.status !== 'error' && (res.overlay_uri || res.heatmap_uri)) {
        const imgSrc = res.overlay_uri || res.heatmap_uri;
        if (m.img) {
          m.img.onload = () => {
            m.img.classList.remove('hidden');
            if (m.loading) m.loading.classList.add('hidden');
          };
          m.img.onerror = () => {
            m.img.classList.add('hidden');
            if (m.loading) {
              m.loading.classList.remove('hidden');
              m.loading.innerHTML = `
                <span class="material-symbols-outlined text-error text-[24px]">broken_image</span>
                <span class="text-error font-label-sm text-[11px] font-mono">Image render error</span>
              `;
            }
          };
          m.img.src = imgSrc;
        }
        if (m.conf) m.conf.textContent = res.confidence_pct || `${Math.round(res.confidence * 100)}%`;
        const isCancer = (res.prediction || '').toLowerCase().includes('cancer') || (res.prediction || '').toLowerCase().includes('carcinoma');
        if (m.pred) {
          m.pred.textContent = isCancer ? 'Carcinoma' : 'Diverticula';
          m.pred.className = 'font-label-sm text-label-sm font-semibold ' + (isCancer ? 'text-error' : 'text-primary');
        }
        if (m.bar) m.bar.style.width = res.confidence_pct || `${Math.round(res.confidence * 100)}%`;
        if (m.lat) m.lat.textContent = `${res.latency_ms} ms`;
        if (m.layerCard && res.attribution_layer) m.layerCard.textContent = res.attribution_layer;
        if (m.layerFootprint && res.attribution_layer) m.layerFootprint.textContent = res.attribution_layer;
        if (m.badge) {
          m.badge.textContent = 'Evaluated';
          m.badge.className = 'px-1.5 py-0.5 rounded bg-primary-container text-surface-container-lowest font-label-sm text-[10px] font-mono';
        }
      } else {
        // Individual model failure
        if (m.img) m.img.classList.add('hidden');
        if (m.loading) {
          m.loading.classList.remove('hidden');
          m.loading.innerHTML = `
            <span class="material-symbols-outlined text-error text-[24px]">error</span>
            <span class="text-error font-label-sm text-[11px] font-mono">${res?.error || 'Inference failed'}</span>
          `;
        }
        if (m.badge) {
          m.badge.textContent = 'Failed';
          m.badge.className = 'px-1.5 py-0.5 rounded bg-error text-white font-label-sm text-[10px] font-mono';
        }
        if (m.pred) {
          m.pred.textContent = 'Failed';
          m.pred.className = 'font-label-sm text-label-sm font-semibold text-error';
        }
        if (m.conf) m.conf.textContent = '—';
        if (m.lat) m.lat.textContent = res && res.latency_ms ? `${res.latency_ms} ms` : '—';
        if (m.bar) m.bar.style.width = '0%';
      }
    });
  } catch (err) {
    console.error('Comparative models request error:', err);
    if (elements.compareConsensusTitle) {
      elements.compareConsensusTitle.textContent = 'Comparison Failed';
    }
    if (elements.compareConsensusDesc) {
      elements.compareConsensusDesc.textContent = `Inference engine error: ${err.message}`;
    }
    models.forEach(m => {
      if (m.img) m.img.classList.add('hidden');
      if (m.loading) {
        m.loading.classList.remove('hidden');
        m.loading.innerHTML = `
          <span class="material-symbols-outlined text-error text-[24px]">cloud_off</span>
          <span class="text-error font-label-sm text-[11px] font-mono">Service unavailable</span>
        `;
      }
      if (m.badge) {
        m.badge.textContent = 'Error';
        m.badge.className = 'px-1.5 py-0.5 rounded bg-error text-white font-label-sm text-[10px] font-mono';
      }
      if (m.pred) {
        m.pred.textContent = 'Unavailable';
        m.pred.className = 'font-label-sm text-label-sm font-semibold text-error';
      }
      if (m.conf) m.conf.textContent = '—';
      if (m.lat) m.lat.textContent = '—';
      if (m.bar) m.bar.style.width = '0%';
    });
  } finally {
    if (btnRerun) btnRerun.disabled = false;
    rerunIcon?.classList.remove('animate-spin');
    if (rerunText) rerunText.textContent = 'Re-run Comparison';
  }
}

// ── Overview Retrospective History ───────────────────────────────────────────
async function clearHistory() {
  try {
    await safeApiRequest('/api/history', { method: 'DELETE' });
    await loadHistory();
  } catch (err) {
    console.error('History clear error:', err);
  }
}

async function loadHistory() {
  try {
    const data = await safeApiRequest('/api/history');
    const history = data.history || [];

    const container = elements.overviewHistoryContainer;
    container.innerHTML = '';

    if (history.length === 0) {
      container.innerHTML = `
        <div class="col-span-full py-8 text-center bg-surface-container-lowest rounded-xl border border-dashed border-outline-variant/40 flex flex-col items-center justify-center gap-2 p-4">
          <span class="material-symbols-outlined text-[32px] text-secondary">history</span>
          <p class="font-headline-sm text-body-sm font-semibold text-on-surface">No analyses logged in this session</p>
          <p class="font-body-sm text-body-sm text-secondary max-w-sm">Upload a colonoscopy frame or run inference on a preloaded benchmark sample to log session records here.</p>
          <button id="empty-state-analyze-btn" type="button" class="mt-2 px-3.5 py-1.5 rounded-lg bg-primary-container text-on-primary font-headline-sm text-body-sm shadow-xs hover:bg-primary active:scale-95 transition-all">
            Analyze First Image
          </button>
        </div>
      `;
      document.getElementById('empty-state-analyze-btn')?.addEventListener('click', () => switchView('analyze'));
      return;
    }

    history.slice(0, 6).forEach(item => {
      const card = document.createElement('div');
      card.className = 'bg-surface-container-lowest rounded-xl p-3.5 shadow-xs border border-outline-variant/30 flex gap-3 items-center hover:border-primary/50 hover:shadow-sm transition-all cursor-pointer group';
      card.title = 'Click to inspect this analysis in Results & Explainability';
      
      const isCancer = item.prediction.toLowerCase().includes('cancer') || item.prediction.toLowerCase().includes('carcinoma');

      card.innerHTML = `
        <div class="w-12 h-12 rounded-lg bg-surface-container flex items-center justify-center flex-shrink-0 text-primary group-hover:bg-primary-container group-hover:text-on-primary transition-colors">
          <span class="material-symbols-outlined text-[24px]">biotech</span>
        </div>
        <div class="flex flex-col min-w-0 flex-1 gap-1">
          <div class="flex items-center justify-between gap-1 font-mono text-[11px]">
            <span class="text-secondary">${item.timestamp}</span>
            <span class="px-1.5 py-0.2 rounded bg-surface-container text-on-surface">${item.model_name}</span>
          </div>
          <h3 class="font-headline-sm text-body-sm text-on-surface font-semibold truncate group-hover:text-primary transition-colors">${item.prediction}</h3>
          <div class="flex items-center gap-2 pt-0.5">
            <div class="flex-1 bg-surface-container-high h-1.5 rounded-full overflow-hidden">
              <div class="bg-primary-container h-full rounded-full" style="width: ${item.confidence_pct || '90%'}"></div>
            </div>
            <span class="font-label-sm text-[11px] font-bold text-primary-container font-mono">${item.confidence_pct || '90%'}</span>
          </div>
        </div>
        <span class="material-symbols-outlined text-secondary group-hover:text-primary text-[20px] flex-shrink-0 transition-colors">chevron_right</span>
      `;

      card.addEventListener('click', () => {
        if (item.original_uri && item.overlay_uri) {
          state.lastAnalysis = item;
          renderAnalysisResults(item);
          switchView('results');
        } else {
          switchView('analyze');
        }
      });

      container.appendChild(card);
    });
  } catch (err) {
    console.error('History load error:', err);
  }
}

// ── Module Drawer Toggle ─────────────────────────────────────────────────────
function initMenuDrawer() {
  const menuBtn = document.getElementById('btn-toggle-menu');
  const closeBtn = document.getElementById('btn-close-drawer');
  const backdrop = document.getElementById('drawer-backdrop');
  const drawer = document.getElementById('clinical-drawer');
  const drawerClearBtn = document.getElementById('btn-drawer-clear-history');

  function openDrawer() {
    drawer?.classList.add('drawer-open');
    backdrop?.classList.add('backdrop-open');
  }

  function closeDrawer() {
    drawer?.classList.remove('drawer-open');
    backdrop?.classList.remove('backdrop-open');
  }

  const desktopTelemetryBtn = document.getElementById('btn-desktop-telemetry');
  const statusIndicator = document.getElementById('system-status-indicator');

  menuBtn?.addEventListener('click', openDrawer);
  closeBtn?.addEventListener('click', closeDrawer);
  backdrop?.addEventListener('click', closeDrawer);
  desktopTelemetryBtn?.addEventListener('click', openDrawer);
  statusIndicator?.addEventListener('click', openDrawer);

  document.querySelectorAll('[data-drawer-target]').forEach(btn => {
    btn.addEventListener('click', () => {
      const target = btn.dataset.drawerTarget;
      closeDrawer();
      if (target) switchView(target);
    });
  });

  drawerClearBtn?.addEventListener('click', async () => {
    closeDrawer();
    await clearHistory();
  });
}

// ── System Health & Hardware Telemetry ───────────────────────────────────────
async function fetchSystemHealth() {
  try {
    const health = await safeApiRequest('/api/health');
    const drawerDevice = document.getElementById('drawer-device');
    if (drawerDevice && health.device) {
      drawerDevice.textContent = `${health.device.toUpperCase()} (Torch ${health.torch_version})`;
    }
  } catch (err) {
    console.warn('Health check warning:', err);
  }
}

// ── CTA Buttons on Overview & Header ─────────────────────────────────────────
function initOverviewCtas() {
  // Brand logo home link
  document.getElementById('btn-nav-home')?.addEventListener('click', () => switchView('overview'));

  // Clear history from overview
  document.getElementById('btn-overview-clear-history')?.addEventListener('click', clearHistory);

  // Overview & Results CTA buttons
  document.getElementById('cta-analyze-new')?.addEventListener('click', () => switchView('analyze'));
  document.getElementById('cta-explore-models')?.addEventListener('click', () => switchView('models'));
  document.getElementById('btn-results-to-analyze')?.addEventListener('click', () => switchView('analyze'));

  // Preset sample buttons on Analyze page
  document.getElementById('sample-cancer-btn')?.addEventListener('click', () => applySample('cancer'));
  document.getElementById('sample-diverticula-btn')?.addEventListener('click', () => applySample('diverticula'));

  // Quick benchmark selectors on Comparative Models page
  document.getElementById('compare-select-cancer-btn')?.addEventListener('click', async () => {
    await applySample('cancer');
    await triggerMultiModelComparison();
  });
  document.getElementById('compare-select-divert-btn')?.addEventListener('click', async () => {
    await applySample('diverticula');
    await triggerMultiModelComparison();
  });
  document.getElementById('compare-goto-analyze-btn')?.addEventListener('click', () => {
    switchView('analyze');
  });

  // Re-run comparison button
  const btnRerun = document.getElementById('btn-rerun-compare');
  if (btnRerun) {
    btnRerun.addEventListener('click', async () => {
      if (!state.selectedFile && !state.selectedSampleId) {
        alert('Please select a benchmark sample or upload a colonoscopy frame first.');
        return;
      }
      await triggerMultiModelComparison();
    });
  }

  // Nav buttons registration (Mobile & Desktop)
  document.querySelectorAll('.mobile-nav-btn, .desktop-nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const target = btn.dataset.target;
      if (target) switchView(target);
    });
  });
}

// ── Initialization ───────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  initDropzone();
  initModelSelector();
  initSaliencyControls();
  initAnalysisExecution();
  initResultsViewport();
  initMenuDrawer();
  initOverviewCtas();

  // Load initial data
  await fetchSystemHealth();
  await loadSamples();
  await loadHistory();

  // Initial view is Overview
  switchView('overview');
});

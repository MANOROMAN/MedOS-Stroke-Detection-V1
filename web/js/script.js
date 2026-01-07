// =============================================================================
// [CORE] THEME MANAGEMENT SYSTEM
// NOTE(UX): Persistent theme state via localStorage.
// =============================================================================
const html = document.documentElement;
const themeToggle = document.getElementById('theme-toggle');
const icon = themeToggle ? themeToggle.querySelector('i') : null;

function loadTheme() {
    const saved = localStorage.getItem('theme') || 'light';
    html.setAttribute('data-theme', saved);
    if (icon) icon.className = saved === 'dark' ? 'fa-solid fa-sun' : 'fa-solid fa-moon';
}

function toggleTheme() {
    const current = html.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    html.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
    if (icon) icon.className = next === 'dark' ? 'fa-solid fa-sun' : 'fa-solid fa-moon';
}
if (themeToggle) themeToggle.addEventListener('click', toggleTheme);
loadTheme();

// =============================================================================
// [STATE] VIRTUAL STACK & HISTORY
// TODO(Architecture): Move state management to a dedicated store (e.g., Redux/Zustand) if complexity grows.
// =============================================================================
let analysisStack = []; // Structure: { type: 'main'|'roi', img: 'data:..', risk: 90.0, area: 500, label: '...' }
let currentIndex = 0;

const navPrev = document.getElementById('nav-prev');
const navNext = document.getElementById('nav-next');
const viewIndicator = document.getElementById('view-indicator');

// --- HISTORY & NAVIGATION ---
const historyContainer = document.getElementById('analysis-history');
const viewLabelSmall = document.getElementById('view-label-small');

// Render Traceability Matrix (History List)
function renderHistory() {
    if (!historyContainer) return;
    historyContainer.innerHTML = '';

    analysisStack.forEach((item, index) => {
        const row = document.createElement('div');
        row.style.display = 'flex';
        row.style.alignItems = 'center';
        row.style.gap = '8px';
        row.style.padding = '6px';
        row.style.borderRadius = '6px';
        row.style.cursor = 'pointer';
        row.style.transition = 'background 0.2s';

        // Active State Indicator
        if (index === currentIndex) {
            row.style.background = 'var(--primary)';
            row.style.color = '#fff';
        } else {
            row.style.background = 'var(--bg-main)';
            row.className = 'history-item';
        }

        // Thumbnail Cache
        const icon = document.createElement('div');
        Object.assign(icon.style, {
            width: '32px', height: '32px', borderRadius: '4px',
            background: '#000', border: '1px solid #333',
            overflow: 'hidden', display: 'flex',
            alignItems: 'center', justifyContent: 'center'
        });

        const imgIcon = document.createElement('img');
        imgIcon.src = item.img;
        Object.assign(imgIcon.style, { width: '100%', height: '100%', objectFit: 'cover' });
        icon.appendChild(imgIcon);

        // Metadata Display
        const info = document.createElement('div');
        info.style.flex = '1';

        const rLabel = document.createElement('div');
        rLabel.style.fontSize = '0.75rem';
        rLabel.style.fontWeight = '600';
        rLabel.innerText = item.label || (item.type === 'main' ? 'Main View' : `ROI #${index}`);

        const rRisk = document.createElement('div');
        rRisk.style.fontSize = '0.7rem';
        if (index === currentIndex) rRisk.style.color = '#eee';
        else rRisk.style.color = item.has_stroke ? '#ef4444' : '#22c55e';
        rRisk.innerText = `Risk: %${item.risk.toFixed(1)}`;

        info.appendChild(rLabel);
        info.appendChild(rRisk);

        row.appendChild(icon);
        row.appendChild(info);

        // Context Switch Handler
        row.onclick = () => {
            currentIndex = index;
            renderCurrentView();
        };

        historyContainer.appendChild(row);
    });
}

// Stack Update Logic
// NOTE(Feature): preserveHistory flag ensures ROI context isn't lost during parameter tuning.
function updateStack(newItem, preserveHistory = false) {
    if (newItem.type === 'main') {
        if (preserveHistory && analysisStack.length > 0) {
            // Update Main View in-place (Index 0), preserving child ROIs
            analysisStack[0] = newItem;
            currentIndex = 0;
        } else {
            // Hard Reset: New Source Image
            analysisStack = [newItem];
            currentIndex = 0;
        }
    } else {
        // Append ROI
        analysisStack.push(newItem);
        currentIndex = analysisStack.length - 1;
    }
    renderCurrentView();
}


function renderCurrentView() {
    if (analysisStack.length === 0) return;
    const item = analysisStack[currentIndex];

    // 1. Update Viewport
    if (mainImg) {
        mainImg.onload = null; // Prevent recursive loop
        mainImg.src = item.img;
    }

    // 2. Update Metrics Panel
    const riskEl = document.getElementById('risk-res');
    const areaEl = document.getElementById('area-res');
    const pathEl = document.getElementById('pathology-res');

    // Update Dynamic Breadcrumb
    if (viewLabelSmall) viewLabelSmall.innerText = item.label || (item.type === 'main' ? 'Main View' : `ROI #${currentIndex}`);

    if (item.has_stroke) {
        if (riskEl) { riskEl.innerText = "%" + item.risk.toFixed(1); riskEl.style.color = "#ef4444"; }
        if (areaEl) { areaEl.innerText = item.area + " px²"; areaEl.style.color = "#ef4444"; }
        if (pathEl) { pathEl.innerText = "Detected"; pathEl.style.color = "#ef4444"; }
    } else {
        if (riskEl) { riskEl.innerText = "%" + item.risk.toFixed(1); riskEl.style.color = "#22c55e"; }
        if (areaEl) { areaEl.innerText = "0 px²"; areaEl.style.color = "#ccc"; }
        if (pathEl) { pathEl.innerText = "Normal"; pathEl.style.color = "#22c55e"; }
    }

    // 3. Sync List State
    renderHistory();

    // Reset Selection Overlay
    if (roiBox) roiBox.style.display = 'none';
}

// =============================================================================
// [INTERACTION] TOOLBAR & CANVAS CONTROLS
// =============================================================================
let currentTool = 'pan';
let scale = 1, translateX = 0, translateY = 0;
let isDragging = false, startX, startY;

// DOM References
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const mainImg = document.getElementById('main-image');
const uploadVisual = document.getElementById('upload-instruction');
const transformLayer = document.getElementById('transform-layer');
const btnPan = document.getElementById('btn-pan');
const btnReset = document.getElementById('btn-reset');
const btnZoomIn = document.getElementById('btn-zoom-in');
const btnZoomOut = document.getElementById('btn-zoom-out');

// Drawing Tools
const btnDraw = document.getElementById('btn-draw');
const drawGroup = document.getElementById('draw-group');
const drawCanvas = document.getElementById('draw-canvas');
let isDrawing = false;
let ctx = null;

if (btnPan) btnPan.addEventListener('click', () => setTool('pan'));
if (btnDraw) btnDraw.addEventListener('click', () => setTool('draw'));


// ROI Tool Configuration
const btnRoi = document.getElementById('btn-roi');
const roiBox = document.getElementById('roi-selection-box');
const roiGallery = document.getElementById('roi-gallery');
const galleryContent = document.getElementById('gallery-content');

let isRoiSelecting = false;
let roiStart = { x: 0, y: 0 };

if (btnRoi) btnRoi.addEventListener('click', () => setTool('roi'));

// Tool State Manager
function setTool(tool) {
    currentTool = tool;
    document.querySelectorAll('.tool-btn').forEach(b => b.classList.remove('active'));

    if (tool === 'pan' && btnPan) btnPan.classList.add('active');
    if (tool === 'draw' && drawGroup) drawGroup.classList.add('active');
    if (tool === 'roi' && btnRoi) btnRoi.classList.add('active');

    if (transformLayer) {
        if (tool === 'pan') transformLayer.style.cursor = 'grab';
        else if (tool === 'draw') transformLayer.style.cursor = 'crosshair';
        else if (tool === 'roi') transformLayer.style.cursor = 'cell';
        else transformLayer.style.cursor = 'default';
    }

    // Pointer Event Routing
    if (drawCanvas) {
        drawCanvas.style.pointerEvents = tool === 'draw' ? 'auto' : 'none';
        if (tool === 'draw') {
            ctx = drawCanvas.getContext('2d');
            ctx.strokeStyle = penColor;
        }
    }

    if (tool !== 'roi' && roiBox) roiBox.style.display = 'none';
}

// ROI Selection Events
if (transformLayer) {
    transformLayer.addEventListener('mousedown', (e) => {
        if (currentTool !== 'roi') return;
        e.preventDefault();
        e.stopPropagation();

        isRoiSelecting = true;

        roiStart.x = e.offsetX;
        roiStart.y = e.offsetY;

        roiBox.style.display = 'block';
        roiBox.style.left = roiStart.x + 'px';
        roiBox.style.top = roiStart.y + 'px';
        roiBox.style.width = '0px';
        roiBox.style.height = '0px';
    });

    transformLayer.addEventListener('mousemove', (e) => {
        if (!isRoiSelecting || currentTool !== 'roi') return;
        e.preventDefault();

        const currentX = e.offsetX;
        const currentY = e.offsetY;

        const width = currentX - roiStart.x;
        const height = currentY - roiStart.y;

        roiBox.style.width = Math.abs(width) + 'px';
        roiBox.style.height = Math.abs(height) + 'px';
        roiBox.style.left = (width < 0 ? currentX : roiStart.x) + 'px';
        roiBox.style.top = (height < 0 ? currentY : roiStart.y) + 'px';
    });

    transformLayer.addEventListener('mouseup', async (e) => {
        if (!isRoiSelecting || currentTool !== 'roi') return;
        isRoiSelecting = false;

        // Bounding Box Validation
        const boxLeft = parseInt(roiBox.style.left);
        const boxTop = parseInt(roiBox.style.top);
        const boxW = parseInt(roiBox.style.width);
        const boxH = parseInt(roiBox.style.height);

        if (boxW < 10 || boxH < 10) {
            roiBox.style.display = 'none'; // Filter micro-selects
            return;
        }

        await handleRoiSelection(boxLeft, boxTop, boxW, boxH);
    });
}

// ROI Processing Handler
async function handleRoiSelection(x, y, w, h) {
    // Client-side Cropping
    const cropCanvas = document.createElement('canvas');
    cropCanvas.width = w;
    cropCanvas.height = h;
    const cCtx = cropCanvas.getContext('2d');

    const dispW = mainImg.width;
    const natW = mainImg.naturalWidth;
    const ratio = natW / dispW;

    cCtx.drawImage(mainImg, x * ratio, y * ratio, w * ratio, h * ratio, 0, 0, w, h);
    const cropDataUrl = cropCanvas.toDataURL('image/png');

    // API Request
    try {
        const res = await fetch(cropDataUrl);
        const blob = await res.blob();

        const formData = new FormData();
        formData.append('file', blob, 'roi_crop.png');
        formData.append('model', modelSel ? modelSel.value : 'default');
        formData.append('viz_mode', 'fill');
        formData.append('color', selectedColor);
        formData.append('glow', isGlow);

        const apiRes = await fetch('../predict', { method: 'POST', body: formData }); // Relative path for Repo
        const data = await apiRes.json();

        updateStack({
            type: 'roi',
            img: data.visualization,
            risk: data.risk_score,
            area: data.area,
            has_stroke: data.has_stroke,
            label: `ROI #${analysisStack.length} (${data.risk_score.toFixed(1)}%)`
        });

    } catch (err) {
        console.error("[ERROR] ROI Analysis Failed:", err);
        alert("ROI Analysis Failed: " + err.message);
    }
}

// Report Generation
async function saveReport() {
    if (analysisStack.length === 0) return;
    const item = analysisStack[currentIndex];

    try {
        const res = await fetch('../save_report', { // Relative path
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image: item.img,
                risk: item.risk,
                area: item.area
            })
        });
        const d = await res.json();
        if (d.status === 'success') {
            alert("Report Saved Successfully!\nPath: " + d.path);
        } else {
            alert("Error: " + d.error);
        }
    } catch (e) {
        alert("Connection Error");
    }
}

// Initialize Save Button
const btnSave = document.querySelector('.result-panel .btn-primary');
if (btnSave) {
    btnSave.removeAttribute('onclick'); // Remove legacy inline handlers
    btnSave.addEventListener('click', saveReport);
    btnSave.innerHTML = '<i class="fa-solid fa-floppy-disk"></i> Confirm & Save';
}


// Viewport Transformation Logic
if (btnZoomIn) btnZoomIn.addEventListener('click', () => updateZoom(0.1));
if (btnZoomOut) btnZoomOut.addEventListener('click', () => updateZoom(-0.1));
if (btnReset) btnReset.addEventListener('click', () => {
    resetView();
    if (ctx && drawCanvas) ctx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
});

function updateZoom(delta) {
    scale = Math.max(0.1, Math.min(5, scale + delta));
    applyTransform();
}

function resetView() {
    scale = 1; translateX = 0; translateY = 0; applyTransform();
}

function applyTransform() {
    if (transformLayer) transformLayer.style.transform = `translate(${translateX}px, ${translateY}px) scale(${scale})`;
}

if (transformLayer) {
    // Pan Handlers
    transformLayer.addEventListener('mousedown', (e) => {
        if (currentTool === 'pan') {
            e.preventDefault();
            isDragging = true;
            startX = e.clientX - translateX;
            startY = e.clientY - translateY;
            transformLayer.style.cursor = 'grabbing';
        }
    });

    window.addEventListener('mousemove', (e) => {
        if (isDragging && currentTool === 'pan') {
            e.preventDefault();
            translateX = e.clientX - startX;
            translateY = e.clientY - startY;
            applyTransform();
        }
    });

    window.addEventListener('mouseup', () => {
        isDragging = false;
        if (currentTool === 'pan' && transformLayer) transformLayer.style.cursor = 'grab';
    });
}

// Annotation (Draw) Handlers
if (drawCanvas) {
    drawCanvas.addEventListener('mousedown', (e) => {
        if (currentTool !== 'draw') return;
        e.preventDefault();
        isDrawing = true;

        if (!ctx) {
            ctx = drawCanvas.getContext('2d');
            ctx.lineWidth = 3;
            ctx.lineCap = 'round';
        }
        ctx.strokeStyle = penColor;

        ctx.beginPath();
        ctx.moveTo(e.offsetX, e.offsetY);
    });

    drawCanvas.addEventListener('mousemove', (e) => {
        if (!isDrawing || currentTool !== 'draw') return;
        e.preventDefault();
        ctx.lineTo(e.offsetX, e.offsetY);
        ctx.stroke();
    });

    drawCanvas.addEventListener('mouseup', () => {
        if (isDrawing) {
            isDrawing = false;
            ctx.closePath();
        }
    });

    drawCanvas.addEventListener('mouseleave', () => {
        isDrawing = false;
    });
}

// =============================================================================
// [IO] FILE HANDLING
// =============================================================================

if (fileInput) fileInput.addEventListener('change', (e) => handleFile(e.target.files[0]));
if (dropZone) {
    dropZone.addEventListener('dragover', (e) => e.preventDefault());
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        handleFile(e.dataTransfer.files[0]);
    });
    dropZone.addEventListener('click', () => {
        if ((!mainImg.src || mainImg.classList.contains('hidden'))) fileInput.click();
    });
}

function handleFile(file) {
    if (!file || !file.type.startsWith('image/')) return;
    currentFile = file;

    // UI State Reset
    uploadVisual.style.display = 'none';
    if (mainImg) mainImg.classList.add('hidden');

    const reader = new FileReader();
    reader.onload = (e) => {
        if (mainImg) {
            mainImg.src = e.target.result;
            mainImg.onload = () => {
                mainImg.classList.remove('hidden');

                // Canvas Synchronization
                if (drawCanvas) {
                    drawCanvas.width = mainImg.naturalWidth;
                    drawCanvas.height = mainImg.naturalHeight;
                    ctx = drawCanvas.getContext('2d');
                    ctx.strokeStyle = '#ef4444';
                    ctx.lineWidth = 3;
                    ctx.lineCap = 'round';
                }
                const maskCanvas = document.getElementById('mask-canvas');
                if (maskCanvas) {
                    maskCanvas.width = mainImg.naturalWidth;
                    maskCanvas.height = mainImg.naturalHeight;
                }

                resetView();
                performAnalysis(file);
            };
        }
    };
    reader.readAsDataURL(file);
}

// Runtime Globals
let currentFile = null;

// =============================================================================
// [EVENTS] REACTIVE UPDATES
// =============================================================================

// Model Switch
const modelSel = document.getElementById('model-select');
if (modelSel) {
    modelSel.addEventListener('change', () => {
        if (currentFile) performAnalysis(currentFile, true);
    });
}

// View Mode Switch
const viewSelect = document.getElementById('mask-mode');
if (viewSelect) {
    viewSelect.addEventListener('change', () => {
        if (currentFile) performAnalysis(currentFile, true);
    });
}

// Color Picker
const colorDots = document.querySelectorAll('.color-dot');
let selectedColor = '#ff0000';

colorDots.forEach(dot => {
    dot.addEventListener('click', (e) => {
        colorDots.forEach(d => d.style.border = 'none');
        e.target.style.border = '2px solid white';
        selectedColor = e.target.getAttribute('data-color') || '#ff0000';
        if (currentFile) performAnalysis(currentFile, true);
    });
});

// Pen Settings
const btnPenOpt = document.getElementById('btn-pen-opt');
const penPalette = document.getElementById('pen-palette');
const penDots = document.querySelectorAll('.pen-dot');
let penColor = '#ef4444';

if (btnPenOpt) {
    btnPenOpt.addEventListener('click', (e) => {
        e.stopPropagation();
        penPalette.style.display = penPalette.style.display === 'flex' ? 'none' : 'flex';
    });
}

penDots.forEach(dot => {
    dot.addEventListener('click', (e) => {
        e.stopPropagation();
        penColor = e.target.getAttribute('data-color');

        penDots.forEach(d => d.style.border = '1px solid var(--border)');
        e.target.style.border = '2px solid white';

        if (ctx) ctx.strokeStyle = penColor;
        if (penPalette) penPalette.style.display = 'none';

        if (btnDraw) btnDraw.click();
    });
});

window.addEventListener('click', (e) => {
    if (penPalette && penPalette.style.display === 'flex') {
        if (!penPalette.contains(e.target) && e.target !== btnPenOpt && !btnPenOpt.contains(e.target)) {
            penPalette.style.display = 'none';
        }
    }
});

// Glow Effect Toggle
const glowBtn = document.getElementById('glow-toggle');
let isGlow = false;
if (glowBtn) {
    glowBtn.addEventListener('click', () => {
        isGlow = !isGlow;
        if (isGlow) {
            glowBtn.style.color = "#facc15";
            glowBtn.classList.remove('fa-regular');
            glowBtn.classList.add('fa-solid');
            glowBtn.style.textShadow = "0 0 10px #facc15";
        } else {
            glowBtn.style.color = "var(--text-muted)";
            glowBtn.classList.remove('fa-solid');
            glowBtn.classList.add('fa-regular');
            glowBtn.style.textShadow = "none";
        }
        if (currentFile) performAnalysis(currentFile, true);
    });
}

// Inference Pipeline
async function performAnalysis(fileObj, preserveHistory = false) {
    currentFile = fileObj;

    const riskEl = document.getElementById('risk-res');
    const pathEl = document.getElementById('pathology-res');

    if (pathEl) pathEl.innerText = "Analyzing...";

    const formData = new FormData();
    formData.append('file', fileObj);

    if (modelSel) formData.append('model', modelSel.value);
    const viewMode = document.getElementById('mask-mode');
    if (viewMode) formData.append('viz_mode', viewMode.value);
    formData.append('color', selectedColor);
    formData.append('glow', isGlow);

    try {
        const response = await fetch('../predict', { // Relative URL
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || ("Status: " + response.status));
        }

        if (data.visualization) {
            updateStack({
                type: 'main',
                img: data.visualization,
                risk: data.risk_score,
                area: data.area,
                has_stroke: data.has_stroke,
                label: 'Main Analysis'
            }, preserveHistory);
        }

    } catch (err) {
        console.error(err);
        if (riskEl) riskEl.innerText = "Err";
        if (pathEl) {
            pathEl.innerText = "Sys Error";
            pathEl.title = err.message;
        }
        // Fallback: Client-side rendering of raw image
        const reader = new FileReader();
        reader.onload = (e) => {
            if (mainImg) {
                mainImg.onload = null;
                mainImg.src = e.target.result;
                mainImg.classList.remove('hidden');
            }
        };
        reader.readAsDataURL(fileObj);
    }
}

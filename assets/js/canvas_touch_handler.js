/**
 * Canvas Responsive Scaling and Touch Handler
 * Multi-Box Professional Mobile Editor with Streamlit-Like UI
 */

(function () {
    const silence = (w) => {
        try {
            if (!w || !w.console) return;
            ['warn', 'error', 'log'].forEach(m => {
                const original = w.console[m];
                if (!original || original.__isMuted) return;
                w.console[m] = function (...args) {
                    const msg = String(args[0] || "");
                    if (/Invalid color|theme\.sidebar|widgetBackground|Unrecognized feature|ambient-light|battery|wake-lock|sandbox|document-domain|oversized-images|vr|layout-animations|legacy-image-formats|allow-scripts|allow-same-origin|payment|microphone|camera|geolocation/i.test(msg)) return;
                    original.apply(this, args);
                };
                w.console[m].__isMuted = true;
            });
        } catch (e) { }
    };
    silence(window);

    // Config Check
    let config = window.CANVAS_CONFIG || (window.parent ? window.parent.CANVAS_CONFIG : null);
    if (!config && window.CANVAS_CONFIG_JSON) {
        try { config = JSON.parse(window.CANVAS_CONFIG_JSON); } catch (e) { }
    }
    if (!config) return;

    const { CANVAS_WIDTH, CANVAS_HEIGHT } = config;

    const getActiveIframe = () => {
        const all = parent.document.querySelectorAll('iframe[title="streamlit_drawable_canvas.st_canvas"], iframe[src*="streamlit_drawable_canvas"]');
        for (let i = all.length - 1; i >= 0; i--) {
            const f = all[i];
            if (!f.closest('[data-stale="true"]')) return f;
        }
        return all[all.length - 1];
    };

    let lastHistoryCall = 0;
    const throttledReplaceState = (url) => {
        const now = Date.now();
        if (now - lastHistoryCall < 50) return false;
        parent.history.replaceState({}, '', url.toString());
        try { parent.dispatchEvent(new Event('popstate')); } catch (e) { }
        lastHistoryCall = now;
        return true;
    };

    const triggerRerun = () => {
        const now = Date.now();
        if (window.lastRerunTrigger && (now - window.lastRerunTrigger < 400)) return;
        window.lastRerunTrigger = now;
        setTimeout(() => {
            // Robust Text-Based Search for the Sync Button
            const buttons = Array.from(parent.document.querySelectorAll('button'));
            const syncBtn = buttons.find(b => b.textContent && b.textContent.includes("GLOBAL SYNC"));

            if (syncBtn) {
                console.log("JS: Found Sync Button, clicking...");
                syncBtn.click();
            } else {
                console.error("JS: Sync Button 'GLOBAL SYNC' NOT FOUND in parent document.");
                // Fallback: Try searching for the marker's sibling if text fails (e.g. if text is hidden/empty)
                const globalMarker = parent.document.querySelector('[data-sync-id="global_sync"]');
                if (globalMarker) {
                    // Try going up to block and finding *last* button
                    const block = globalMarker.closest('[data-testid="stVerticalBlock"]');
                    if (block) {
                        const blockBtns = block.querySelectorAll('button');
                        if (blockBtns.length > 0) blockBtns[blockBtns.length - 1].click();
                    }
                }
            }
        }, 50);
    };

    /**
     * Multi-Box Editor
     */
    class BoxEditor {
        constructor() {
            this.overlay = null;
            this.toolbar = null;
            this.handles = []; // DOM elements
            this.handleContainer = null;
            this.boxContainer = null;

            this.boxes = []; // Array of {x,y,w,h}
            this.selectedIndex = -1;

            this.isDrawing = false;
            this.isMoving = false;
            this.activeHandle = null;
            this.startX = 0;
            this.startY = 0;
            this.startBox = null;

            this.currentWrapper = null; // Track attachment

            this.createDOM();
        }

        createDOM() {
            // Remove existing toolbar
            let existingBar = parent.document.getElementById('box-editor-fixed-toolbar');
            if (existingBar) existingBar.remove();

            // Overlay
            this.overlay = parent.document.createElement('div');
            this.overlay.id = 'box-editor-overlay';
            Object.assign(this.overlay.style, {
                position: 'absolute',
                top: '0', left: '0', width: '100%', height: '100%',
                zIndex: '999',
                touchAction: 'none',
                display: 'none',
                cursor: 'crosshair',
                overflow: 'visible'
            });

            // Container for Boxes
            this.boxContainer = parent.document.createElement('div');
            Object.assign(this.boxContainer.style, {
                position: 'absolute', top: '0', left: '0', width: '100%', height: '100%',
                pointerEvents: 'none'
            });
            this.overlay.appendChild(this.boxContainer);

            // Handles Container
            this.handleContainer = parent.document.createElement('div');
            Object.assign(this.handleContainer.style, {
                position: 'absolute', top: '0', left: '0', width: '100%', height: '100%',
                pointerEvents: 'none',
                zIndex: '1001'
            });
            this.overlay.appendChild(this.handleContainer);

            // Pre-create Handles (24px)
            const positions = ['nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w'];
            positions.forEach(pos => {
                const h = parent.document.createElement('div');
                h.dataset.handle = pos;
                Object.assign(h.style, {
                    position: 'absolute',
                    width: '24px', height: '24px',
                    backgroundColor: '#FFFFFF',
                    border: '2px solid #FF4B4B',
                    borderRadius: '50%',
                    transform: 'translate(-50%, -50%)',
                    zIndex: '1002',
                    display: 'none',
                    touchAction: 'none',
                    boxShadow: '0 2px 4px rgba(0,0,0,0.3)',
                    pointerEvents: 'auto',
                    cursor: 'pointer'
                });
                this.handleContainer.appendChild(h);
                this.handles.push(h);
            });

            // Toolbar (Flow Layout below Image)
            this.toolbar = parent.document.createElement('div');
            this.toolbar.id = 'box-editor-flow-toolbar';
            Object.assign(this.toolbar.style, {
                display: 'none',
                gap: '12px',
                padding: '16px 0', // Vertical spacing
                background: 'transparent',
                alignItems: 'center',
                justifyContent: 'center',
                width: '100%',
                pointerEvents: 'auto',
                flexDirection: 'row',
                position: 'relative',
                zIndex: '2000'
            });

            // Streamlit-like Button Style
            const btnBase = `
                display: flex; align-items: center; justify-content: center; gap: 8px;
                height: 44px; padding: 0 24px; min-width: 120px;
                border-radius: 8px; font-family: "Source Sans Pro", sans-serif; font-weight: 600; font-size: 16px;
                cursor: pointer; transition: all 0.2s;
                border: 1px solid rgba(49, 51, 63, 0.2);
                background: white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            `;

            // Delete
            const btnDelete = parent.document.createElement('button');
            btnDelete.innerHTML = "🗑️ Delete";
            btnDelete.style.cssText = btnBase + "color: #333;"; // Standard text color
            btnDelete.title = "Delete Selected";
            const curDelete = (e) => { e.preventDefault(); e.stopPropagation(); this.deleteSelected(); };
            btnDelete.onclick = curDelete;
            btnDelete.ontouchend = curDelete;

            // Apply
            const btnApply = parent.document.createElement('button');
            btnApply.innerHTML = "✅ Apply";
            btnApply.style.cssText = btnBase + "border-color: #10B981; color: #10B981; background: #F0FDF4;"; // Green accent
            btnApply.title = "Apply Paint";
            const curApply = (e) => { e.preventDefault(); e.stopPropagation(); this.commit(); };
            btnApply.onclick = curApply;
            btnApply.ontouchend = curApply;

            this.toolbar.appendChild(btnDelete);
            this.toolbar.appendChild(btnApply);

            // Initial mount check will handle insertion

            // Bind Events strictly
            this.overlay.onpointerdown = this.onPointerDown.bind(this);
            this.overlay.onpointermove = this.onPointerMove.bind(this);
            this.overlay.onpointerup = this.onPointerUp.bind(this);
            this.overlay.onpointercancel = this.onPointerUp.bind(this);
        }

        mount() {
            const iframe = getActiveIframe();
            if (!iframe) return;
            const wrapper = iframe.parentElement;
            if (!wrapper) return;

            // 1. Overlay (inside wrapper)
            if (this.currentWrapper !== wrapper || !this.overlay.isConnected) {
                const stale = wrapper.querySelector('#box-editor-overlay');
                if (stale && stale !== this.overlay) stale.remove();
                wrapper.appendChild(this.overlay);
                this.currentWrapper = wrapper;
            }

            // 2. Toolbar (AFTER wrapper, in flow)
            // Ensure toolbar is the immediate next sibling
            if (wrapper.nextElementSibling !== this.toolbar) {
                // If it's elsewhere, move it. If not attached, attach it.
                wrapper.after(this.toolbar);
            }
        }

        checkMode() {
            this.mount();

            let mode = 'point';
            if (window.CANVAS_CONFIG && window.CANVAS_CONFIG.DRAWING_MODE) mode = window.CANVAS_CONFIG.DRAWING_MODE;

            if (mode === 'rect') {
                this.overlay.style.display = 'block';
                // Show toolbar if boxes exist -> actually user might want to commit empty? no.
                if (this.boxes.length > 0) this.toolbar.style.display = 'flex';
                else this.toolbar.style.display = 'none';
            } else {
                this.overlay.style.display = 'none';
                this.toolbar.style.display = 'none';
                this.boxes = [];
                this.selectedIndex = -1;
                this.updateDOM();
            }
        }

        deleteSelected() {
            if (this.selectedIndex !== -1) {
                this.boxes.splice(this.selectedIndex, 1);
                this.selectedIndex = -1;
                this.updateDOM();
            }
        }

        onPointerDown(e) {
            e.preventDefault();
            this.overlay.setPointerCapture(e.pointerId);

            const rect = this.overlay.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            // 1. Check Active Handles
            if (e.target.dataset.handle) {
                this.isMoving = false;
                this.isDrawing = false;
                this.activeHandle = e.target.dataset.handle;
                this.startBox = { ...this.boxes[this.selectedIndex] };
                this.startX = e.clientX;
                this.startY = e.clientY;
                return;
            }

            // 2. Check Box Click (Any Box)
            for (let i = this.boxes.length - 1; i >= 0; i--) {
                const b = this.boxes[i];
                if (x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h) {
                    this.selectedIndex = i;
                    this.isMoving = true;
                    this.isDrawing = false;
                    this.activeHandle = null;
                    this.startX = e.clientX;
                    this.startY = e.clientY;
                    this.startBox = { ...b };
                    this.updateDOM();
                    return;
                }
            }

            // 3. New Box
            this.isDrawing = true;
            this.isMoving = false;
            this.activeHandle = null;
            this.startX = x;
            this.startY = y;

            const newBox = { x, y, w: 0, h: 0 };
            this.boxes.push(newBox);
            this.selectedIndex = this.boxes.length - 1;
            this.startBox = newBox;

            this.updateDOM();
        }

        onPointerMove(e) {
            if (!this.isDrawing && !this.isMoving && !this.activeHandle) return;
            e.preventDefault();

            const rect = this.overlay.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            if (this.isDrawing) {
                const b = this.boxes[this.selectedIndex];
                const minX = Math.min(this.startX, x);
                const minY = Math.min(this.startY, y);
                const w = Math.abs(x - this.startX);
                const h = Math.abs(y - this.startY);
                const clampX = Math.max(0, Math.min(minX, rect.width - w));
                const clampY = Math.max(0, Math.min(minY, rect.height - h));
                b.x = minX; b.y = minY; b.w = w; b.h = h;
            }
            else if (this.isMoving) {
                const dx = e.clientX - this.startX;
                const dy = e.clientY - this.startY;
                const b = this.boxes[this.selectedIndex];
                let nx = this.startBox.x + dx;
                let ny = this.startBox.y + dy;
                nx = Math.max(0, Math.min(nx, rect.width - this.startBox.w));
                ny = Math.max(0, Math.min(ny, rect.height - this.startBox.h));
                b.x = nx; b.y = ny;
            }
            else if (this.activeHandle) {
                const dx = e.clientX - this.startX;
                const dy = e.clientY - this.startY;
                const b = this.boxes[this.selectedIndex];
                const sb = this.startBox;
                let nx = sb.x, ny = sb.y, nw = sb.w, nh = sb.h;

                if (this.activeHandle.includes('e')) nw = sb.w + dx;
                if (this.activeHandle.includes('s')) nh = sb.h + dy;
                if (this.activeHandle.includes('w')) { nw = sb.w - dx; nx = sb.x + dx; }
                if (this.activeHandle.includes('n')) { nh = sb.h - dy; ny = sb.y + dy; }

                if (nw < 20) { if (this.activeHandle.includes('w')) nx = sb.x + sb.w - 20; nw = 20; }
                if (nh < 20) { if (this.activeHandle.includes('n')) ny = sb.y + sb.h - 20; nh = 20; }

                b.x = nx; b.y = ny; b.w = nw; b.h = nh;
            }

            requestAnimationFrame(() => this.updateDOM());
        }

        onPointerUp(e) {
            e.preventDefault();
            this.overlay.releasePointerCapture(e.pointerId);
            this.isDrawing = false;
            this.isMoving = false;
            this.activeHandle = null;

            if (this.selectedIndex !== -1) {
                const b = this.boxes[this.selectedIndex];
                if (b.w < 10 || b.h < 10) {
                    this.boxes.splice(this.selectedIndex, 1);
                    this.selectedIndex = -1;
                }
            }
            this.updateDOM();
        }

        updateDOM() {
            this.boxContainer.innerHTML = '';

            this.boxes.forEach((b, idx) => {
                const el = parent.document.createElement('div');
                const isSelected = (idx === this.selectedIndex);

                Object.assign(el.style, {
                    position: 'absolute',
                    left: b.x + 'px', top: b.y + 'px',
                    width: b.w + 'px', height: b.h + 'px',
                    border: isSelected ? '2px solid #FF4B4B' : '2px solid rgba(255, 75, 75, 0.4)',
                    boxShadow: isSelected ? '0 0 0 1px white, 0 4px 8px rgba(0,0,0,0.2)' : 'none',
                    backgroundColor: isSelected ? 'rgba(255, 75, 75, 0.1)' : 'rgba(255, 75, 75, 0.05)',
                    pointerEvents: 'auto',
                    cursor: 'move'
                });
                el.dataset.index = idx;
                this.boxContainer.appendChild(el);
            });

            if (this.selectedIndex !== -1) {
                const b = this.boxes[this.selectedIndex];
                const map = {
                    nw: [b.x, b.y], n: [b.x + b.w / 2, b.y], ne: [b.x + b.w, b.y],
                    e: [b.x + b.w, b.y + b.h / 2], se: [b.x + b.w, b.y + b.h],
                    s: [b.x + b.w / 2, b.y + b.h], sw: [b.x, b.y + b.h],
                    w: [b.x, b.y + b.h / 2]
                };

                this.handles.forEach(h => {
                    const pos = h.dataset.handle;
                    const coords = map[pos];
                    h.style.left = coords[0] + 'px';
                    h.style.top = coords[1] + 'px';
                    h.style.display = 'block';
                });
                this.handleContainer.style.display = 'block';
            } else {
                this.handleContainer.style.display = 'none';
            }

            if (this.boxes.length > 0) {
                this.toolbar.style.display = 'flex';
            } else {
                this.toolbar.style.display = 'none';
            }
        }

        commit() {
            if (this.boxes.length === 0) return;
            console.log("JS: Committing boxes...", this.boxes);

            const rect = this.overlay.getBoundingClientRect();
            const scale = CANVAS_WIDTH / rect.width;

            const parts = this.boxes.map(b => {
                const cx1 = Math.round(b.x * scale);
                const cy1 = Math.round(b.y * scale);
                const cx2 = Math.round((b.x + b.w) * scale);
                const cy2 = Math.round((b.y + b.h) * scale);
                return `${cx1},${cy1},${cx2},${cy2}`;
            });

            const val = parts.join('|') + ',' + Date.now();
            console.log("JS: Box param val:", val);

            const url = new URL(parent.location.href);
            url.searchParams.set('box', val);
            url.searchParams.delete('tap');

            throttledReplaceState(url);

            // Delay rerun significantly to ensure URL update propagates
            setTimeout(() => {
                console.log("JS: Triggering Rerun");
                triggerRerun();
            }, 500);

            this.boxes = [];
            this.selectedIndex = -1;
            this.updateDOM();
        }
    }

    function applyResponsiveScale() {
        try {
            const iframes = parent.document.getElementsByTagName('iframe');
            if (!iframes.length) return;

            const winW = parent.window.innerWidth;
            const targetWidth = winW < 1024 ? winW - 20 : winW;

            if (targetWidth <= 0 || !CANVAS_WIDTH) return;

            const scale = Math.min(1.0, targetWidth / CANVAS_WIDTH);

            for (let iframe of iframes) {
                if (iframe.title === "streamlit_drawable_canvas.st_canvas" || iframe.src.includes('streamlit_drawable_canvas')) {
                    const wrapper = iframe.parentElement;
                    wrapper.style.width = (CANVAS_WIDTH * scale) + "px";
                    wrapper.style.height = (CANVAS_HEIGHT * scale) + "px";
                    wrapper.style.position = "relative";
                    wrapper.style.overflow = "hidden";
                    wrapper.style.margin = "0 auto";
                    wrapper.style.touchAction = "none";

                    iframe.style.width = CANVAS_WIDTH + "px";
                    iframe.style.height = CANVAS_HEIGHT + "px";
                    iframe.style.transform = `scale(${scale})`;
                    iframe.style.transformOrigin = "top left";
                    iframe.style.position = "absolute";
                    iframe.style.touchAction = "none";
                }
            }
        } catch (e) { }
    }

    let boxEditor = null;
    function mainLoop() {
        applyResponsiveScale();
        if (!boxEditor) boxEditor = new BoxEditor();
        boxEditor.checkMode();
    }

    window.addEventListener("resize", mainLoop);
    setInterval(mainLoop, 500);
    mainLoop();
})();

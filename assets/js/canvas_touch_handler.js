/**
 * Canvas Responsive Scaling and Touch Handler
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
                    if (/Invalid color|theme\.sidebar|widgetBackground|Unrecognized feature|ambient-light|battery|wake-lock|sandbox|document-domain|oversized-images|vr/i.test(msg)) return;
                    original.apply(this, args);
                };
                w.console[m].__isMuted = true;
            });
        } catch (e) { }
    };
    silence(window);
    try {
        let curr = window;
        while (curr.parent && curr.parent !== curr) {
            silence(curr.parent);
            curr = curr.parent;
        }
    } catch (e) { }
    setInterval(() => {
        silence(window);
        try {
            let curr = window;
            while (curr.parent && curr.parent !== curr) {
                silence(curr.parent);
                curr = curr.parent;
            }
        } catch (e) { }
    }, 1000);
})();

(function (config) {
    if (!config) return;

    // Configuration from Python
    const {
        CANVAS_WIDTH,
        CANVAS_HEIGHT,
        CUR_PAN_X,
        CUR_PAN_Y,
        VIEW_W,
        IMAGE_W,
        VIEW_H,
        IMAGE_H,
        DRAWING_MODE
    } = config;

    /**
     * Get the active (non-stale) iframe canvas element
     */
    const getActiveIframe = () => {
        const all = parent.document.querySelectorAll('iframe[title="streamlit_drawable_canvas.st_canvas"]');
        for (let i = all.length - 1; i >= 0; i--) {
            const f = all[i];
            if (!f.closest('[data-stale="true"]')) return f;
        }
        return all[all.length - 1];
    };

    /**
     * Apply responsive scaling to canvas iframe
     */
    function applyResponsiveScale() {
        try {
            const iframe = getActiveIframe();
            if (!iframe) return;
            const wrapper = iframe.parentElement;
            if (!wrapper) return;

            // Go up to find the element container which has the true width
            const container = wrapper.closest('.element-container') || wrapper.parentElement;
            if (!container) return;

            let containerWidth = container.getBoundingClientRect().width;

            // On mobile, force better fit within Streamlit's default padding
            if (parent.window.innerWidth < 1024) {
                // Streamlit usually has ~16px padding on mobile
                containerWidth = Math.min(containerWidth, parent.window.innerWidth - 16);
            }

            if (containerWidth <= 0) return;

            // Standardize scaling - remove the -2 offset for better fit
            const scale = Math.min(1.0, containerWidth / CANVAS_WIDTH);

            // Apply wrapper styles (Visual size)
            wrapper.style.width = (CANVAS_WIDTH * scale) + "px";
            wrapper.style.height = (CANVAS_HEIGHT * scale) + "px";
            wrapper.style.overflow = "hidden";
            wrapper.style.touchAction = "none";
            wrapper.style.position = "relative";
            wrapper.style.margin = "0 auto";
            wrapper.style.border = "none"; // Ensure no borders cause offset

            // Apply iframe styles with transform scaling
            iframe.style.width = CANVAS_WIDTH + "px";
            iframe.style.height = CANVAS_HEIGHT + "px";
            iframe.style.transformOrigin = "top left"; // Explicitly top left
            iframe.style.transform = "scale(" + scale + ")";
            iframe.style.border = "none";
            iframe.style.display = "block";
            iframe.style.position = "absolute";
            iframe.style.top = "0";
            iframe.style.left = "0";
            iframe.style.zIndex = "100";
            iframe.style.pointerEvents = "auto";
            iframe.style.touchAction = "none";
        } catch (e) { }
    }

    // Double-click tracking (must persist across setupInteraction() calls)
    let globalLastTapTime = 0;
    const DBL_TAP_WINDOW = 250; // Reduced from 350ms for more precise detection

    // --- THROTTLE HISTORY UPDATES TO PREVENT "Bad message format" ERROR ---
    // Browser limit: 100 pushState calls per 10 seconds
    // We throttle to 50/10s (5/s) for safety
    let historyCallTimes = [];
    const MAX_HISTORY_CALLS = 50; // Per 10 seconds
    const HISTORY_WINDOW = 10000; // 10 seconds
    const MIN_HISTORY_INTERVAL = 200; // Minimum 200ms between calls (5 per second)
    let lastHistoryCall = 0;

    const throttledPushState = (url) => {
        const now = Date.now();

        // Remove calls older than 10 seconds
        historyCallTimes = historyCallTimes.filter(t => (now - t) < HISTORY_WINDOW);

        // Check if we've hit the limit
        if (historyCallTimes.length >= MAX_HISTORY_CALLS) {
            console.warn('[Throttle] History pushState limit reached, skipping update');
            return false;
        }

        // Check minimum interval
        if (now - lastHistoryCall < MIN_HISTORY_INTERVAL) {
            console.warn('[Throttle] Too soon since last pushState, skipping');
            return false;
        }

        // Safe to call
        parent.history.pushState({}, '', url.toString());
        historyCallTimes.push(now);
        lastHistoryCall = now;
        return true;
    };

    /**
     * Setup pointer event proxying for precise interaction
     */
    function setupInteraction() {
        try {
            const iframe = getActiveIframe();
            if (!iframe || !iframe.contentDocument) return;
            const canvasElements = iframe.contentDocument.querySelectorAll('canvas');
            canvasElements.forEach(canvas => {
                canvas.dataset.drawingMode = DRAWING_MODE;

                // SYNC: Always check if points are cleared (runs every 1s)
                // This ensures dots are removed if Python clears the points
                if (!window.parent.STREAMLIT_POLY_POINTS || window.parent.STREAMLIT_POLY_POINTS.length === 0) {
                    const dots = window.parent.POLYGON_DOTS || [];
                    if (dots.length > 0) {
                        dots.forEach(d => d.remove());
                        window.parent.POLYGON_DOTS = [];
                    }
                }

                if (canvas.dataset.hasProxy === "true") return;

                canvas.style.touchAction = "none";
                canvas.style.userSelect = "none";
                canvas.style.webkitUserSelect = "none";
                canvas.style.webkitTapHighlightColor = "transparent";

                let startX = 0, startY = 0;
                let isDragging = false;
                const TAP_THRESHOLD = 15; // Increased for Desktop mouse clicks

                const showFeedbackDot = (clientX, clientY) => {
                    try {
                        const feedback = parent.document.createElement('div');
                        feedback.style.position = 'fixed';
                        feedback.style.left = clientX + 'px';
                        feedback.style.top = clientY + 'px';
                        feedback.style.width = '24px';
                        feedback.style.height = '24px';
                        feedback.style.backgroundColor = 'rgba(255, 75, 75, 0.4)';
                        feedback.style.border = '2px solid white';
                        feedback.style.borderRadius = '50%';
                        feedback.style.pointerEvents = 'none';
                        feedback.style.zIndex = '999999';
                        feedback.style.transform = 'translate(-50%, -50%) scale(0.5)';
                        feedback.style.transition = 'all 0.3s ease-out';
                        parent.document.body.appendChild(feedback);

                        requestAnimationFrame(() => {
                            feedback.style.transform = 'translate(-50%, -50%) scale(1.5)';
                            feedback.style.opacity = '0';
                        });
                        setTimeout(() => feedback.remove(), 400);
                    } catch (e) { }
                };

                const updateTap = (x, y) => {
                    const currentUrl = new URL(parent.location.href);
                    currentUrl.searchParams.set('tap', `${Math.round(x)},${Math.round(y)}`);

                    // Store tap in window for Python to read
                    window.parent.LAST_TAP_COORDS = { x: Math.round(x), y: Math.round(y), timestamp: Date.now() };

                    // Try throttled pushState first
                    const pushed = throttledPushState(currentUrl);

                    if (!pushed) {
                        // Throttled! Use replaceState instead (doesn't count toward limit)
                        // This updates the URL without adding to history
                        console.log('[Tap] Using replaceState due to throttle');
                        parent.history.replaceState({}, '', currentUrl.toString());
                    }

                    // Always trigger rerun for taps
                    triggerRerun();
                };

                const updatePan = (px, py) => {
                    const currentUrl = new URL(parent.location.href);
                    currentUrl.searchParams.set('pan_update', `${px.toFixed(4)},${py.toFixed(4)}`);

                    const pushed = throttledPushState(currentUrl);
                    if (!pushed) {
                        // Fallback to replaceState if pushState is throttled
                        parent.history.replaceState({}, '', currentUrl.toString());
                    }

                    // Always trigger rerun for panning to maintain responsiveness
                    triggerRerun();
                };

                const updatePolyPts = (pts) => {
                    if (canvas.dataset.drawingMode !== "polygon") return;
                    window.parent.STREAMLIT_POLY_POINTS = pts;

                    if (pts && pts.length > 0) {
                        const ptsStr = pts.map(p => `${Math.round(p.x)},${Math.round(p.y)}`).join(";");
                        const currentUrl = new URL(parent.location.href);
                        currentUrl.searchParams.set('poly_pts', ptsStr);

                        // IMPORTANT: Use replaceState for point updates to avoid history limit
                        // This updates the URL so Python can see it, but doesn't count against pushState limit
                        parent.history.replaceState({}, '', currentUrl.toString());
                    }
                };

                const triggerFinishButton = () => {
                    const allBtns = Array.from(parent.document.querySelectorAll('button'));
                    const finishBtn = allBtns.find(b => {
                        const txt = (b.innerText || "").toUpperCase();
                        return txt.includes("FINISH") || txt.includes("DONE") || txt.includes("APPLY");
                    });

                    if (finishBtn) {
                        finishBtn.click();
                        return true;
                    }
                    return false;
                };

                const triggerRerun = () => {
                    const now = Date.now();
                    if (window.lastRerunTrigger && (now - window.lastRerunTrigger < 400)) return;
                    window.lastRerunTrigger = now;

                    setTimeout(() => {
                        const globalMarker = parent.document.querySelector('[data-sync-id="global_sync"]');
                        if (globalMarker) {
                            const globalBtn = globalMarker.closest('div[data-testid="stVerticalBlock"]').querySelector('button');
                            if (globalBtn) { globalBtn.click(); return; }
                        }
                    }, 50);
                };

                const addPolygonalPoint = (offX, offY) => {
                    if (canvas.dataset.drawingMode !== "polygon") return;
                    window.parent.STREAMLIT_POLY_POINTS = window.parent.STREAMLIT_POLY_POINTS || [];

                    // Simply add point - double-click will trigger finish
                    window.parent.STREAMLIT_POLY_POINTS.push({ x: offX, y: offY });
                    updatePolyPts(window.parent.STREAMLIT_POLY_POINTS);
                };

                const clearPersistentDots = () => {
                    try {
                        const dots = window.parent.POLYGON_DOTS || [];
                        dots.forEach(d => d.remove());
                        window.parent.POLYGON_DOTS = [];
                    } catch (e) { }
                };

                const showPersistentDot = (clientX, clientY) => {
                    try {
                        const dot = parent.document.createElement('div');
                        dot.className = 'polygon-dot-marker';
                        dot.style.position = 'fixed';
                        dot.style.left = clientX + 'px';
                        dot.style.top = clientY + 'px';
                        dot.style.width = '12px';
                        dot.style.height = '12px';
                        dot.style.backgroundColor = '#ff4b4b';
                        dot.style.border = '2px solid white';
                        dot.style.borderRadius = '50%';
                        dot.style.zIndex = '999999';
                        dot.style.transform = 'translate(-50%, -50%)';
                        dot.style.pointerEvents = 'none';
                        dot.style.boxShadow = '0 2px 4px rgba(0,0,0,0.3)';
                        parent.document.body.appendChild(dot);

                        window.parent.POLYGON_DOTS = window.parent.POLYGON_DOTS || [];
                        window.parent.POLYGON_DOTS.push(dot);
                    } catch (e) { }
                };

                // Removed ineffective sync logic from here

                const sendEvent = (type, e) => {
                    const rect = canvas.getBoundingClientRect();
                    const scaleX = rect.width > 0 ? (CANVAS_WIDTH / rect.width) : 1;
                    const scaleY = rect.height > 0 ? (CANVAS_HEIGHT / rect.height) : 1;
                    const offX = (e.clientX - rect.left) * scaleX;
                    const offY = (e.clientY - rect.top) * scaleY;

                    canvas.dispatchEvent(new PointerEvent(type, {
                        bubbles: true, cancelable: true, isPrimary: true, pointerId: 1,
                        pointerType: e.pointerType || 'mouse',
                        clientX: e.clientX, clientY: e.clientY,
                        offsetX: offX, offsetY: offY,
                        button: 0, buttons: (type === 'pointerup') ? 0 : 1
                    }));
                };

                // --- 🎯 POINTER LISTENERS (Universal) ---
                canvas.addEventListener('pointerdown', (e) => {
                    startX = e.clientX; startY = e.clientY;
                    isDragging = false;
                    sendEvent('pointerdown', e);
                });

                canvas.addEventListener('pointermove', (e) => {
                    if (Math.abs(e.clientX - startX) > TAP_THRESHOLD || Math.abs(e.clientY - startY) > TAP_THRESHOLD) {
                        isDragging = true;
                    }
                    sendEvent('pointermove', e);
                });

                canvas.addEventListener('pointerup', (e) => {
                    sendEvent('pointerup', e);

                    const rect = canvas.getBoundingClientRect();
                    const scaleX = rect.width > 0 ? (CANVAS_WIDTH / rect.width) : 1;
                    const scaleY = rect.height > 0 ? (CANVAS_HEIGHT / rect.height) : 1;
                    const offX = (e.clientX - rect.left) * scaleX;
                    const offY = (e.clientY - rect.top) * scaleY;

                    if (!isDragging) {
                        showFeedbackDot(e.clientX, e.clientY);
                        if (canvas.dataset.drawingMode === "polygon") {
                            const now = Date.now();
                            const isDoubleClick = (now - globalLastTapTime < DBL_TAP_WINDOW);

                            if (isDoubleClick) {
                                // Double-click detected: Sync points and trigger auto-finish
                                const polyPts = window.parent.STREAMLIT_POLY_POINTS || [];
                                console.log('[Polygon] Double-tap detected, points:', polyPts.length);

                                if (polyPts && polyPts.length > 0) {
                                    const ptsStr = polyPts.map(p => `${Math.round(p.x)},${Math.round(p.y)}`).join(";");
                                    const currentUrl = new URL(parent.location.href);
                                    currentUrl.searchParams.set('poly_pts', ptsStr);
                                    currentUrl.searchParams.set('force_finish', 'true');

                                    const pushed = throttledPushState(currentUrl);
                                    if (!pushed) {
                                        // Use replaceState if throttled
                                        console.log('[Polygon] Using replaceState for finish');
                                        parent.history.replaceState({}, '', currentUrl.toString());
                                    }

                                    // Always trigger rerun for polygon finish
                                    triggerRerun();
                                    clearPersistentDots(); // CLEAR DOTS ON FINISH
                                }
                            } else {
                                // Single click: Add point
                                console.log('[Polygon] Single tap - adding point at', offX, offY);
                                addPolygonalPoint(offX, offY);

                                // FIX: Calculate GLOBAL coordinates for the dot (including iframe offset)
                                const iframe = getActiveIframe(); // Re-fetch to be safe
                                if (iframe) {
                                    const rect = iframe.getBoundingClientRect();
                                    const globalX = rect.left + e.clientX;
                                    const globalY = rect.top + e.clientY;
                                    showPersistentDot(globalX, globalY);
                                } else {
                                    showPersistentDot(e.clientX, e.clientY); // Fallback
                                }
                            }
                            globalLastTapTime = now;
                        } else if (["point", "freedraw", "rect"].includes(canvas.dataset.drawingMode)) {
                            updateTap(offX, offY);
                        }
                    } else if (canvas.dataset.drawingMode === "point") {
                        // Normalized Pan
                        const dragScale = VIEW_W / CANVAS_WIDTH;
                        const dx = -(e.clientX - startX) * dragScale;
                        const dy = -(e.clientY - startY) * dragScale;
                        const max_px = IMAGE_W - VIEW_W;
                        const max_py = IMAGE_H - VIEW_H;
                        if (max_px > 0) {
                            const npx = Math.max(0, Math.min(1, CUR_PAN_X + (dx / max_px)));
                            const npy = max_py > 0 ? Math.max(0, Math.min(1, CUR_PAN_Y + (dy / max_py))) : CUR_PAN_Y;
                            updatePan(npx, npy);
                        }
                    }
                });

                canvas.dataset.hasProxy = "true";
            });
        } catch (e) { }
    }

    applyResponsiveScale();
    setupInteraction();
    window.addEventListener("resize", applyResponsiveScale);
    setInterval(() => { applyResponsiveScale(); setupInteraction(); }, 1000);

})(window.CANVAS_CONFIG);

import streamlit as st
import os
import cv2
import numpy as np
import torch
import streamlit.components.v1 as components
import logging
import time
import io
import requests
import textwrap
from PIL import Image
from streamlit_drawable_canvas import st_canvas as raw_st_canvas
from .encoding import image_to_url_patch
from .state_manager import cb_undo, cb_redo, cb_clear_all, cb_delete_layer, cb_apply_pending, cb_cancel_pending
from .image_processing import get_crop_params, composite_image, process_lasso_path
from .sam_loader import get_sam_engine, CHECKPOINT_PATH, MODEL_TYPE

# --- UI CONSTANTS ---
TOOL_MAPPING = {
    "👆": "👆 AI Click (Point)",
    "✨": "✨ AI Object (Box)",
    "🎨": "🎨 Lasso (Freehand)",
    "🕸️": "🕸️ Polygonal Lasso"
}

# --- HELPERS ---
def safe_rerun(scope="fragment"):
    """Prevents StreamlitAPIException during full-rerun transitions."""
    try:
        st.rerun(scope=scope)
    except:
        st.rerun()

def cb_top_tool_sync_v2():
    """Sync top icons to master selection_tool with canvas reset."""
    new_icon = st.session_state.get("top_tool_switcher_control")
    if new_icon and new_icon in TOOL_MAPPING:
        st.session_state["selection_tool"] = TOOL_MAPPING[new_icon]
        st.session_state["sidebar_tool_radio"] = TOOL_MAPPING[new_icon]
        st.session_state["canvas_id"] = st.session_state.get("canvas_id", 0) + 1
        st.session_state["canvas_raw"] = {}
        st.session_state["pending_selection"] = None
    # No rerun needed here as on_change triggers it automatically

def cb_top_wall_sync_v2():
    """Sync top toggle to master is_wall_only."""
    st.session_state["is_wall_only"] = st.session_state.get("top_wall_control", True)
    st.session_state["sidebar_wall_toggle"] = st.session_state["is_wall_only"]

def cb_sidebar_tool_sync():
    """Sync sidebar radio to master selection_tool and top ICON switcher."""
    new_tool = st.session_state.get("sidebar_tool_radio")
    print(f"DEBUG: CALLBACK sidebar_tool_radio -> {new_tool} (Current: {st.session_state.get('selection_tool')})")
    # --- 🛠️ CLEAN TOOL SWITCH ---
    last_tool = st.session_state.get("selection_tool")
    if last_tool != new_tool:
        # Tool changed! Wipe all volatile interaction signals
        for pk in ["tap", "poly_pts", "pan_update", "zoom_update", "force_finish"]:

            if pk in st.query_params: st.query_params.pop(pk, None)
        # Also clear internal state
        st.session_state["force_finish_poly"] = False
        st.session_state["loop_guarded"] = False
        st.session_state["force_finish_poly"] = False
        st.session_state["loop_guarded"] = False
        
        # FIX: Defer JS clearing to the render loop to avoid callback reruns
        st.session_state["tool_switched_reset"] = True
        st.session_state["fill_selection"] = False
        print(f"DEBUG: Tool Switched -> {last_tool} -> {new_tool}. Signals wiped.")

    st.session_state["selection_tool"] = new_tool
    for icon, label in TOOL_MAPPING.items():
        if label == new_tool:
            st.session_state["top_tool_switcher_control"] = icon
            break
    st.session_state["canvas_id"] = st.session_state.get("canvas_id", 0) + 1
    st.session_state["canvas_raw"] = {}
    st.session_state["pending_selection"] = None

def cb_sidebar_wall_sync():
    """Sync sidebar toggle to master is_wall_only."""
    st.session_state["is_wall_only"] = st.session_state.get("sidebar_wall_toggle")
    st.session_state["top_wall_control"] = st.session_state["is_wall_only"]

def cb_sidebar_op_sync():
    """Sync sidebar radio to master selection_op."""
    st.session_state["selection_op"] = st.session_state.get("sidebar_op_radio", "Add")
    st.session_state["top_op_control"] = "➕" if st.session_state["selection_op"] == "Add" else "➖"

def cb_top_op_sync():
    """Sync top op segmented control to master selection_op."""
    icon = st.session_state.get("top_op_control")
    st.session_state["selection_op"] = "Add" if icon == "➕" else "Subtract"
    st.session_state["sidebar_op_radio"] = st.session_state["selection_op"]

def sort_points_clockwise(pts):
    """Sorts a list of (x, y) points in clockwise order around their centroid."""
    if not pts or len(pts) < 3: return pts
    
    # Calculate centroid
    center_x = sum(p[0] for p in pts) / len(pts)
    center_y = sum(p[1] for p in pts) / len(pts)
    
    # Sort by angle from centroid
    import math
    def get_angle(point):
        return math.atan2(point[1] - center_y, point[0] - center_x)
    
    return sorted(pts, key=get_angle)
def snap_box_to_edges(image, box, margin=15):
    """Refines box coordinates to align with strong architectural edges using Sobel gradients."""
    if image is None: return box
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Use Sobel to find gradients
    grad_x = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    grad_y = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    
    def refine_coord(coord, is_x):
        search_range = range(max(0, coord - margin), min((w if is_x else h), coord + margin))
        best_coord = coord
        max_grad = 0
        target_grad = grad_x if is_x else grad_y
        
        for i in search_range:
            if is_x:
                # Vertical edge: sum gradients along the vertical line at x=i
                g_sum = np.sum(target_grad[max(0, y1-margin):min(h, y2+margin), i])
            else:
                # Horizontal edge: sum gradients along the horizontal line at y=i
                g_sum = np.sum(target_grad[i, max(0, x1-margin):min(w, x2+margin)])
            
            if g_sum >= max_grad:
                max_grad = g_sum
                best_coord = i
        return best_coord

    nx1 = refine_coord(x1, True)
    ny1 = refine_coord(y1, False)
    nx2 = refine_coord(x2, True)
    ny2 = refine_coord(y2, False)
    
    return [nx1, ny1, nx2, ny2]

# --- CANVAS WRAPPER ---
def st_canvas(*args, **kwargs):
    """Wrapper to handle background image conversion to data URLs."""
    kwargs["background_color"] = "rgba(0,0,0,0)"
    bg_img = kwargs.get("background_image")
    
    if bg_img is not None:
        width, height = kwargs.get("width"), kwargs.get("height")
        # PERFORMANCE: Cache key MUST include render_hash and comparison state to detect changes
        comp_flag = str(st.session_state.get("show_comparison", False))
        r_hash = str(st.session_state.get("render_id", 0))
        cache_key = f"bg_url_cache_{r_hash}_{comp_flag}"
        
        if cache_key in st.session_state:
            url = st.session_state[cache_key]
        else:
            url = image_to_url_patch(bg_img, width, True, "RGB", "JPEG", r_hash)
            st.session_state[cache_key] = url
        
        if not kwargs.get("initial_drawing"):
            kwargs["initial_drawing"] = {"version": "4.4.0", "objects": []}
            
        kwargs["initial_drawing"]["background"] = "rgba(0,0,0,0)"
        kwargs["initial_drawing"]["backgroundImage"] = {
            "type": "image", "version": "4.4.0", "originX": "left", "originY": "top",
            "left": 0, "top": 0, "width": width, "height": height, "scaleX": 1, "scaleY": 1,
            "visible": True, "src": url
        }
        if "background_image" in kwargs:
             del kwargs["background_image"]
            
    return raw_st_canvas(*args, **kwargs)

# --- STYLES ---
def setup_styles():
    css_path = os.path.join(os.path.dirname(__file__), "..", "assets", "style.css")
    style_content = ""
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            style_content = f.read()
            
    # Combined static and dynamic CSS
    sidebar_transform = '0' if st.session_state.get("sidebar_p_open") else '-100%'
    
    pass
    
    full_css = textwrap.dedent(f"""
        <style>
        {style_content}
        
        /* Force Light Theme */
        :root {{
            --primary-color: #ff4b4b;
            --background-color: #ffffff;
            --secondary-background-color: #f0f2f6;
            --text-color: #31333F;
            --font: "Segoe UI", sans-serif;
        }}
        

        
        [data-testid="stSidebar"] {{
            background-color: #f8f9fa !important; 
            border-right: 1px solid #e6e6e6;
            width: 350px !important;
        }}

        @media (min-width: 769px) {{
            .m-open-container, .mobile-only {{ display: none !important; }}
        }}

        @media (max-width: 768px) {{
            .desktop-only {{ display: none !important; }}
            .mobile-bottom-actions {{
                position: fixed;
                bottom: 20px;
                left: 10px;
                right: 10px;
                background: rgba(255, 255, 255, 0.9);
                backdrop-filter: blur(10px);
                padding: 10px;
                border-radius: 15px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.15);
                z-index: 1000;
            }}
            /* Add padding to bottom of page so content doesn't get hidden under fixed buttons */
            .main .block-container {{
                padding-bottom: 120px !important;
            }}
        }}


        /* Restored native overlay behavior */

        /* Hide Ghost Sync Button */
        div[data-testid="stButton"]:has(button p:contains("GHOST")) {{ display: none !important; }}
        div[data-testid="stButton"]:has(button:contains("GHOST")) {{ display: none !important; }}
        div[data-testid="stButton"]:has(button p:contains("GLOBAL SYNC")) {{ display: none !important; }}
        div[data-testid="stButton"]:has(button:contains("GLOBAL SYNC")) {{ display: none !important; }}

        /* Fully restored native layout behavior */
        
        div.element-container:has(#global-sync-anchor),
        div.element-container:has(.global-sync-marker),
        div.element-container:has(.sync-ghost-marker) {{
            display: none !important;
            height: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
            position: fixed !important;
            top: -10000px !important;
        }}

        .element-container:has(.sync-ghost-marker)+.element-container,
        .element-container:has(.global-sync-marker)+.element-container,
        div.element-container:has(#global-sync-anchor)+div.element-container,
        div.element-container:has(button[key="global_sync_btn"]) {{
            display: none !important;
            position: fixed !important;
            top: -10000px !important;
            width: 0 !important;
        }}
        
        /* Buttons Styling */
        .stButton>button {{
            border-radius: 8px !important;
            font-weight: 600 !important;
            transition: all 0.2s !important;
        }}

        /* Landing Page Centering */
        .landing-container {{
            text-align: center !important;
            padding: 4rem 1rem !important;
            max-width: 800px !important;
            margin: 0 auto !important;
        }}

        .landing-header h1 {{
            font-size: 2.8rem !important;
            font-weight: 800 !important;
            color: #111827 !important;
            margin-bottom: 1rem !important;
        }}

        .landing-sub p {{
            font-size: 1.2rem !important;
            color: #4b5563 !important;
            margin-bottom: 2rem !important;
        }}
        </style>
    """).strip()
    
    st.markdown(full_css, unsafe_allow_html=True)

    # --- SILENCE CONSOLE WARNINGS (Main Window Injection) ---
    # This script intercepts browser console warnings originating from Streamlit or dependencies
    st.markdown("""
        <script>
        (function() {
            const _backupWarn = console.warn;
            const _backupError = console.error;
            
            const filterPattern = /Unrecognized feature|ambient-light-sensor|battery|document-domain|layout-animations|legacy-image-formats|oversized-images|vr|wake-lock|allow-scripts|allow-same-origin|escape its sandboxing|Invalid color|theme\.sidebar|widgetBackground/i;
            
            console.warn = function(...args) {
                const msg = (args[0] || '').toString();
                if (filterPattern.test(msg)) return;
                _backupWarn.apply(console, args);
            };
            
            console.error = function(...args) {
                const msg = (args[0] || '').toString();
                if (filterPattern.test(msg)) return;
                _backupError.apply(console, args);
            };
        })();
        </script>
    """, unsafe_allow_html=True)

    # # Mobile: Aggressive sidebar protection
    # st.markdown("""
    # <script>
    # (function() {
    #     if (window.innerWidth > 1024) return; // Desktop - skip
        
    #     let allowClose = false;
        
    #     // Detect when user EXPLICITLY clicks the toggle button to close
    #     parent.document.addEventListener('mousedown', (e) => {
    #         const toggleBtn = e.target.closest('[data-testid="stSidebarCollapseButton"]');
    #         if (toggleBtn) {
    #             const sidebar = parent.document.querySelector('[data-testid="stSidebar"]');
    #             if (sidebar) {
    #                 const isCurrentlyOpen = sidebar.getBoundingClientRect().width > 50;
    #                 // If sidebar is open and user clicks toggle, they want to close it
    #                 if (isCurrentlyOpen) {
    #                     allowClose = true;
    #                     setTimeout(() => { allowClose = false; }, 500);
    #                 }
    #             }
    #         }
    
    #     }, true);
        
    #     // AGGRESSIVE: Monitor every 50ms and force reopen unless user explicitly closed
    #     setInterval(() => {
    #         if (allowClose) return;
            
    #         const sidebar = parent.document.querySelector('[data-testid="stSidebar"]');
    #         const toggleBtn = parent.document.querySelector('header [data-testid="stSidebarCollapseButton"]');
            
    #         if (sidebar && toggleBtn) {
    #             const isOpen = sidebar.getBoundingClientRect().width > 50;
    #             if (!isOpen) {
    #                 toggleBtn.click(); // Force reopen
    #             }
    #         }
    #     }, 50); // Check every 50ms for instant response
    # })();
    # </script>
    # """, unsafe_allow_html=True)


# --- FRAGMENTS ---
def sidebar_paint_fragment():
    """Isolates the color picker to prevent full app reruns on color change."""
    st.subheader("🖌️ Paint Mode")
    active_color = st.session_state.get("picked_color", "#8FBC8F")
    
    with st.container(border=True):
        col_p, col_t = st.columns([0.3, 0.7])
        active_color = st.session_state.get("picked_color", "#8FBC8F")
        with col_p:
            new_color = st.color_picker("Color", active_color, label_visibility="collapsed", key="sidebar_paint_cp")
        with col_t:
             st.markdown(f"""
             <div style="display: flex; flex-direction: column; justify-content: center; height: 35px;">
                <span style="font-family: 'Segoe UI', sans-serif; font-weight: 700; color: #31333F; font-size: 14px; line-height: 1.2;">Active Paint</span>
                <span style="color: #666; font-size: 11px; font-family: monospace;">{new_color}</span>
             </div>
             """, unsafe_allow_html=True)
    
    if new_color != active_color:
        st.session_state["picked_color"] = new_color
        # No rerun needed here as fragment handles local update


def render_zoom_controls(key_suffix="", context_class=""):
    """Render zoom and pan controls.
    Args:
        key_suffix: Unique suffix for widget keys to prevent duplicates.
        context_class: CSS class to wrap the controls (e.g. 'mobile-zoom-wrapper') for responsive hiding.
    """
    # Wrap in a container that we can target with CSS if possible, or just apply classes to elements
    if context_class:
        st.markdown(f'<div class="{context_class}">', unsafe_allow_html=True)

    z_col2, z_col3, z_col4 = st.columns([1, 1, 1])
    
    def update_zoom(delta):
        st.session_state["zoom_level"] = max(1.0, min(4.0, st.session_state["zoom_level"] + delta))
        st.session_state["render_id"] += 1
        st.session_state["canvas_id"] = st.session_state.get("canvas_id", 0) + 1
    
    with z_col2:
        if st.button("➖", help="Zoom Out", use_container_width=True, key=f"zoom_out_{key_suffix}"):
            update_zoom(-0.2)
            st.rerun()
            
    with z_col3:
        st.markdown(
            f"""
            <div class="{context_class}" style='text-align: center; font-weight: bold; background-color: #f0f2f6; color: #31333F; padding: 6px 10px; border-radius: 4px; border: 1px solid #dcdcdc;'>
                {int(st.session_state.get('zoom_level', 1.0) * 100)}%
            </div>
            """, 
            unsafe_allow_html=True
        )
            
    with z_col4:
        if st.button("➕", help="Zoom In", use_container_width=True, key=f"zoom_in_{key_suffix}"):
            update_zoom(0.2)
            st.rerun()

    if st.session_state.get("zoom_level", 1.0) > 1.0 or st.session_state.get("pan_x", 0.5) != 0.5 or st.session_state.get("pan_y", 0.5) != 0.5:
        st.markdown("<div style='height: 5px'></div>", unsafe_allow_html=True)
        if st.button("🎯 Reset View", use_container_width=True, key=f"reset_view_{key_suffix}"):
            st.session_state["zoom_level"] = 1.0
            st.session_state["pan_x"] = 0.5
            st.session_state["pan_y"] = 0.5
            st.session_state["render_id"] += 1
            st.session_state["canvas_id"] = st.session_state.get("canvas_id", 0) + 1
            st.rerun()
            
    if context_class:
        st.markdown('</div>', unsafe_allow_html=True)

def overlay_pan_controls(image):
    """Draws semi-transparent pan arrows on the image edges."""
    h, w, c = image.shape
    overlay = image.copy()
    color = (255, 255, 255)
    thickness = 2
    margin = 40
    center_x, center_y = w // 2, h // 2
    cv2.arrowedLine(overlay, (center_x, margin), (center_x, 10), color, thickness, tipLength=0.5)
    cv2.arrowedLine(overlay, (center_x, h - margin), (center_x, h - 10), color, thickness, tipLength=0.5)
    cv2.arrowedLine(overlay, (margin, center_y), (10, center_y), color, thickness, tipLength=0.5)
    cv2.arrowedLine(overlay, (w - margin, center_y), (w - 10, center_y), color, thickness, tipLength=0.5)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    return image


@st.fragment
def render_visualizer_canvas_fragment_v11(display_width, start_x, start_y, view_w, view_h, scale_factor, h, w, drawing_mode):
    """Isolated heavy-lifting fragment for canvas and backend mask generation."""
    # --- 1. CAPTURE & CLEAR INTERACTION PARAMS ---
    query_params = st.query_params
    mobile_tap = query_params.get("tap", "")
    mobile_box = query_params.get("box", "")
    mobile_pan = query_params.get("pan_update", "")
    mobile_zoom = query_params.get("zoom_update", "")
    url_pts_raw = query_params.get("poly_pts", "")
    force_finish_raw = query_params.get("force_finish", "")
    
    # 🧼 TOOL SWITCH CLEANUP (Silent)
    if st.session_state.get("tool_switched_reset", False):
        st.session_state["tool_switched_reset"] = False
        components.html("<script>if(window.parent.STREAMLIT_POLY_POINTS) window.parent.STREAMLIT_POLY_POINTS = [];</script>", height=0)

    # Signal ID Handling: Normalize signals to extract timestamp if present
    def extract_signal(raw_str):
        if not raw_str or (isinstance(raw_str, str) and raw_str.strip() == ""): return None, None
        if isinstance(raw_str, list): raw_str = raw_str[0]
        parts = raw_str.split(",")
        
        # If the last part looks like a timestamp (lengthy digit string)
        if len(parts) >= 2:
            last_part = parts[-1].strip()
            if len(last_part) >= 10 and last_part.isdigit():
                return ",".join(parts[:-1]), last_part
        return raw_str, None

    mobile_tap, tap_sid = extract_signal(mobile_tap)
    mobile_box, box_sid = extract_signal(mobile_box)
    mobile_pan, pan_sid = extract_signal(mobile_pan)
    force_finish, finish_sid = extract_signal(force_finish_raw)
    is_finish = (force_finish == "true") or st.session_state.get("force_finish_poly", False)

    # SILENT LOOP BREAKER: Guard against rapid successive fragment reruns
    is_guarded = st.session_state.get("loop_guarded", False)
    has_interaction = any([mobile_tap, mobile_box, mobile_pan, mobile_zoom, url_pts_raw])
    
    # Skip heavy signal processing if guarded, but DO NOT return (renders canvas)
    # UNLESS it is a finish command which is high priority
    skip_processing = has_interaction and is_guarded and not is_finish
    
    if skip_processing:
        # Clear signals immediately to break potential loop
        for pk in ["tap", "box", "pan_update", "zoom_update", "poly_pts"]:
            if pk in st.query_params: st.query_params.pop(pk, None)
    
    if has_interaction and not skip_processing:
        st.session_state["loop_guarded"] = True
        print(f"DEBUG: Processing Interaction -> Tool:{drawing_mode}, Tap:{bool(mobile_tap)}, Box:{bool(mobile_box)}, Finish:{is_finish}")

    # --- 🔔 TOP ACTION BAR (Immediate Visibility) ---
    # Render Apply/Cancel buttons at the TOP for mobile accessibility
    if st.session_state.get("pending_selection") is not None:
        st.info("✨ Selection Active! Confirm below.", icon="👇")
        with st.container(border=True):
            cols = st.columns([1, 1], gap="small")
            with cols[0]: 
                if st.button("✨ APPLY PAINT", use_container_width=True, key="top_frag_apply", type="primary"):
                    cb_apply_pending(); safe_rerun()
            with cols[1]: 
                if st.button("🗑️ CANCEL", use_container_width=True, key="top_frag_cancel"):
                    cb_cancel_pending(); safe_rerun()

    original_img = st.session_state["image"]
    painted_img = composite_image(original_img, st.session_state["masks"])
    show_comp = st.session_state.get("show_comparison", False)
    
    display_img = original_img if show_comp else painted_img
    cropped_view = display_img[start_y:start_y+view_h, start_x:start_x+view_w]
    new_h = int(view_h * (display_width / view_w))
    final_display_image = cv2.resize(cropped_view, (display_width, new_h), interpolation=cv2.INTER_LINEAR)
    
    if st.session_state.get("zoom_level", 1.0) > 1.0:
        final_display_image = overlay_pan_controls(final_display_image)

    display_height = final_display_image.shape[0]
    sam = get_sam_engine(CHECKPOINT_PATH, MODEL_TYPE)

    # --- 🎯 MOBILE INTERACTION HANDLER ---
    if mobile_tap and mobile_tap.strip() != "" and not skip_processing:
        # SIGNAL GUARD: Ensure we only process this specific tap ONCE
        if tap_sid and tap_sid == st.session_state.get("last_tap_sid"):
            mobile_tap = None # Already processed
        else:
            if tap_sid: st.session_state["last_tap_sid"] = tap_sid

        if mobile_tap:
            try:
                parts = mobile_tap.split(",")
                if len(parts) >= 2:
                    x, y = int(parts[0].strip()), int(parts[1].strip())
                    real_x = int(x / scale_factor) + start_x
                    real_y = int(y / scale_factor) + start_y
                    
                    print(f"DEBUG: Mobile Tap Handler -> x:{x}, y:{y}, Tool:{drawing_mode}, Scale:{scale_factor:.4f}")
                    # Check for tool-specific behavior
                    is_poly = drawing_mode == "polygon"
                
                picked_color = st.session_state.get('picked_color', '#3B82F6')
                click_state_key = f"{real_x}_{real_y}_{picked_color}_{drawing_mode}"
                
                if click_state_key != st.session_state.get("last_mobile_tap"):
                    st.session_state["last_mobile_tap"] = click_state_key
                    
                    # 🎯 TOOL SEPARATION: Skip point-tap logic for Polygon and Transform
                    # For these tools, interactions are processed via objects/poly_pts
                    # ENABLED 'rect' (AI Object) to allow "Tap to Select" on mobile
                    is_direct_tap_tool = drawing_mode not in ["polygon", "transform"]
                    
                    if is_direct_tap_tool: # Standard point selection (AI Click, etc.)
                        if not getattr(sam, "is_image_set", False): 
                            sam.set_image(st.session_state["image"])
                        
                        mask = sam.generate_mask(
                            point_coords=[real_x, real_y], 
                            level=st.session_state.get("mask_level", 0), 
                            is_wall_only=st.session_state.get("is_wall_only", False)
                        )
                        
                        if mask is not None:
                            st.session_state["pending_selection"] = {'mask': mask, 'point': (real_x, real_y)}
                            
                            # CRITICAL: Apply paint immediately on mobile ONLY for Point/Lasso
                            # For AI Object (rect), we want to show the Review/Confirm buttons first
                            if drawing_mode != "rect":
                                cb_apply_pending()
                            
                            st.session_state["render_id"] += 1
                            st.session_state["canvas_id"] = st.session_state.get("canvas_id", 0) + 1
                            
                            # Clear the tap parameter
                            st.query_params.pop("tap", None)
                            
                            # CRITICAL FIX: Explicitly trigger rerun to show the painted result
                            print(f"DEBUG: Paint applied for mobile tap at ({real_x},{real_y}), triggering rerun")
                            safe_rerun()
                    else:
                        st.session_state["render_id"] += 1
                        st.query_params.pop("tap", None)
                        # safe_rerun() # Fragment scope is enough for poly pts
            except Exception as e:
                print(f"ERROR in mobile tap: {e}")
                import traceback
                traceback.print_exc()

    if mobile_box and mobile_box.strip() != "" and not skip_processing:
        # SIGNAL GUARD: Ensure we only process this specific box ONCE
        if box_sid and box_sid == st.session_state.get("last_box_sid"):
            mobile_box = None
        else:
            if box_sid: st.session_state["last_box_sid"] = box_sid

        if mobile_box:
            try:
                parts = mobile_box.split(",")
                if len(parts) >= 4:
                    x1, y1, x2, y2 = int(float(parts[0])), int(float(parts[1])), int(float(parts[2])), int(float(parts[3]))
                    print(f"DEBUG: Mobile Box Handler -> {x1},{y1} to {x2},{y2} (Scale: {scale_factor})")
                    
                    # Convert to real image coordinates
                    bx1 = int(x1 / scale_factor) + start_x
                    by1 = int(y1 / scale_factor) + start_y
                    bx2 = int(x2 / scale_factor) + start_x
                    by2 = int(y2 / scale_factor) + start_y
                    
                    # Ensure properly ordered coordinates (min/max)
                    final_box = [min(bx1, bx2), min(by1, by2), max(bx1, bx2), max(by1, by2)]
                    
                    # Validate box size (ignore tiny accidentals)
                    if abs(final_box[2] - final_box[0]) > 5 and abs(final_box[3] - final_box[1]) > 5:
                        if not getattr(sam, "is_image_set", False): 
                            sam.set_image(st.session_state["image"])
                            
                        mask = sam.generate_mask(
                            box_coords=final_box, 
                            level=st.session_state.get("mask_level", 0), 
                            is_wall_only=st.session_state.get("is_wall_only", False)
                        )
                        
                        if mask is not None:
                            st.session_state["pending_selection"] = {'mask': mask}
                            st.session_state["pending_box_coords"] = final_box 
                            
                            # ⚡ AUTO-APPLY: JS "Apply" button confirmed this action
                            cb_apply_pending()

                            st.session_state["render_id"] += 1
                            st.session_state["canvas_id"] = st.session_state.get("canvas_id", 0) + 1
                            
                            safe_rerun()
                
                st.query_params.pop("box", None)
            except Exception as e:
                print(f"ERROR in mobile box: {e}")
                import traceback
                traceback.print_exc()
                
    if mobile_pan and mobile_pan.strip() != "":
        # SIGNAL GUARD: Ensure we only process this specific pan ONCE
        if pan_sid and pan_sid == st.session_state.get("last_pan_sid"):
            mobile_pan = None # Already processed
        else:
            if pan_sid: st.session_state["last_pan_sid"] = pan_sid

        if mobile_pan:
            try:
                parts = mobile_pan.split(",")
                if len(parts) >= 2:
                    px, py = float(parts[0]), float(parts[1])
                    st.session_state["pan_x"], st.session_state["pan_y"] = max(0.0, min(1.0, px)), max(0.0, min(1.0, py))
                    st.session_state["render_id"] += 1
                
                st.query_params.pop("pan_update", None)
            except: pass

    if mobile_zoom and mobile_zoom.strip() != "":
        try:
            new_zoom = float(mobile_zoom)
            st.session_state["zoom_level"] = max(1.0, min(4.0, new_zoom))
            st.session_state["render_id"] += 1
            
            st.query_params.pop("zoom_update", None)
        except: pass

    # --- 3. PERSIST DRAWING & UNFINISHED POLYGON ---
    initial_drawing = {"version": "4.4.0", "objects": []}
    
    # Check if we just finished a polygon (to prevent ghost persistence)
    was_just_finished = st.session_state.get("just_finished_poly", False)
    # Reset for next run
    st.session_state["just_finished_poly"] = False

    if st.session_state.get("canvas_raw") and not was_just_finished:
        for obj in (st.session_state.get("canvas_raw") or {}).get("objects", []):
            obj_type = obj.get("type")
            # --- 🛠️ STRICT TOOL FILTERING ---
            # Only persist objects that belong to the CURRENT tool
            is_valid = False
            if drawing_mode == "point" and obj_type == "circle": is_valid = True
            elif drawing_mode == "rect" and obj_type == "rect": is_valid = True
            elif drawing_mode == "freedraw" and obj_type == "path": is_valid = True
            elif drawing_mode == "polygon" and obj_type == "polygon": is_valid = True
            elif drawing_mode == "transform": is_valid = True # Show all in move mode
            
            if is_valid:
                initial_drawing["objects"].append(obj)

    url_pts_raw = st.query_params.get("poly_pts", url_pts_raw)
    if url_pts_raw:
        try:
            pts = []
            for p in str(url_pts_raw).split(";"):
                if "," in p:
                    px, py = p.split(",")
                    pts.append({"x": float(px), "y": float(py)})
            if pts:
                initial_drawing["objects"].append({
                    "type": "polygon", "points": pts,
                    "left": 0, "top": 0, "width": display_width, "height": display_height,
                    "fill": "rgba(255, 75, 75, 0.2)", "stroke": "#FF4B4B", "strokeWidth": 3,
                    "selectable": False, "evented": False
                })
        except: pass

    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)", 
        stroke_width=st.session_state.get("lasso_thickness", 6), 
        stroke_color="#FF4B4B",
        background_image=final_display_image, update_streamlit=True, height=display_height, width=display_width,
        drawing_mode=drawing_mode, initial_drawing=initial_drawing, 
        point_display_radius=20 if drawing_mode in ["point", "freedraw", "polygon"] else 0,
        key=f"canvas_main_{st.session_state.get('canvas_id', 0)}", 
        display_toolbar=False
    )

    # 📱 JS Handler (Silent, no elements)
    js_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "js", "canvas_touch_handler.js")
    if os.path.exists(js_file_path):
        with open(js_file_path, 'r', encoding='utf-8') as f: js_template = f.read()
        if not st.session_state.get("pending_selection"): st.session_state["pending_box_coords"] = None

        # Prepare pending box for JS re-hydration (Editable Box UI)
        p_box = st.session_state.get("pending_box_coords", None)
        p_box_json = str(p_box) if p_box else "null"
        
        js_config = f"<script>const cfg={{ CANVAS_WIDTH: {display_width}, CANVAS_HEIGHT: {display_height}, CUR_PAN_X: {st.session_state['pan_x']}, CUR_PAN_Y: {st.session_state['pan_y']}, VIEW_W: {view_w}, IMAGE_W: {w}, VIEW_H: {view_h}, IMAGE_H: {h}, DRAWING_MODE: '{drawing_mode}', PENDING_BOX: {p_box_json} }}; window.CANVAS_CONFIG=cfg; if(window.parent) window.parent.CANVAS_CONFIG=cfg;</script><script>{js_template}</script>"
        components.html(js_config, height=0)

    # 🔄 SYNC & PROCESS (Silent)
    if canvas_result.json_data:
        json_data = canvas_result.json_data
        st.session_state["canvas_raw"] = json_data
        objects = json_data.get("objects", [])
        
        # 🛑 LOOP GUARD: If we just applied, objects should be empty.
        if st.session_state.get("just_applied", False):
            st.session_state["just_applied"] = False
            if objects:
                # Force clear persistence if st_canvas refused to reset
                objects = []
                st.session_state["canvas_raw"] = {}
                st.session_state["pending_boxes"] = []
                # Force another rerun to clear the UI ghost
                safe_rerun()

        tool_mode = st.session_state.get("selection_tool", "✨ AI Object (Box)")
        
        if objects:
            if "AI Click" in tool_mode:
                for obj in reversed(objects):
                    if obj["type"] in ["circle", "path"]:
                        rel_x, rel_y = obj["left"], obj["top"]
                        real_x, real_y = int(rel_x / scale_factor) + start_x, int(rel_y / scale_factor) + start_y
                        click_key = f"{real_x}_{real_y}_{st.session_state['picked_color']}"
                        if click_key != st.session_state.get("last_click_global"):
                            st.session_state["last_click_global"] = click_key
                            if not getattr(sam, "is_image_set", False): sam.set_image(st.session_state["image"])
                            mask = sam.generate_mask(point_coords=[real_x, real_y], level=st.session_state.get("mask_level", 0), is_wall_only=st.session_state.get("is_wall_only", False))
                            if mask is not None:
                                st.session_state["pending_selection"] = {'mask': mask, 'point': (real_x, real_y)}
                                if st.session_state.get("ai_click_instant_apply", True): cb_apply_pending()
                                st.session_state["render_id"] += 1
                                safe_rerun()
                        break 
            
            elif "Lasso" in tool_mode and "Polygonal" not in tool_mode:
                for obj in reversed(objects):
                    if obj["type"] == "path":
                        path_data = obj.get("path", [])
                        if path_data:
                            scaled_path = []
                            for cmd in path_data:
                                if len(cmd) > 1:
                                    scaled_cmd = [cmd[0]]
                                    for i in range(1, len(cmd), 2):
                                        scaled_cmd.append(int(cmd[i] / scale_factor) + start_x)
                                        scaled_cmd.append(int(cmd[i+1] / scale_factor) + start_y)
                                    scaled_path.append(scaled_cmd)
                            # 🎨 FILL SELECTION MODE vs 🧠 AI LASSO MODE
                            is_fill_mode = st.session_state.get("fill_selection", False)
                            
                            if is_fill_mode:
                                mask = process_lasso_path(scaled_path, w, h, thickness=st.session_state.get("lasso_thickness", 6), fill=True)
                            else:
                                # 🧠 AI LASSO MODE (Box Prompt from Lasso)
                                x_coords = [c[i] for c in scaled_path for i in range(1, len(c), 2)]
                                y_coords = [c[i+1] for c in scaled_path for i in range(1, len(c), 2)]
                                
                                if x_coords:
                                    box = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                                    # Ensure minimal box size
                                    if (box[2]-box[0]) > 5 and (box[3]-box[1]) > 5:
                                        if not getattr(sam, "is_image_set", False): 
                                            sam.set_image(st.session_state["image"])
                                        mask = sam.generate_mask(
                                            box_coords=box, 
                                            level=st.session_state.get("mask_level", 0), 
                                            is_wall_only=st.session_state.get("is_wall_only", False)
                                        )
                                    else:
                                        mask = np.zeros((h, w), dtype=bool)
                                else:
                                    mask = np.zeros((h, w), dtype=bool)
                                
                                # Fallback to brush if AI returns nothing (optional, but let's stick to AI-only for now to match behavior)
                                if mask is None: mask = np.zeros((h, w), dtype=bool)
                            
                            if mask.any():
                                # DEBUG: Log operation mode before applying
                                current_op = st.session_state.get("selection_op", "Unknown")
                                print(f"DEBUG: FREEHAND LASSO APPLY -> Operation: {current_op}, Mask pixels: {np.sum(mask)}, Fill: {is_fill_mode}")
                                
                                st.session_state["pending_selection"] = {'mask': mask}
                                cb_apply_pending()
                                st.session_state["render_id"] += 1
                                safe_rerun()
                        break

            elif "AI Object" in tool_mode:
                current_boxes = []
                combined_mask = None
                for obj in objects:
                    if "width" in obj and "height" in obj and obj.get("type") not in ["circle"]:
                        left, top = obj["left"], obj["top"]
                        width, height = obj["width"] * obj.get("scaleX", 1.0), obj["height"] * obj.get("scaleY", 1.0)
                        bx1, by1 = int(left / scale_factor) + start_x, int(top / scale_factor) + start_y
                        bx2, by2 = int((left + width) / scale_factor) + start_x, int((top + height) / scale_factor) + start_y
                        current_boxes.append([bx1, by1, bx2, by2])
                
                if current_boxes != st.session_state.get("pending_boxes") and current_boxes:
                    st.session_state["pending_boxes"] = current_boxes
                    for box in current_boxes:
                        # 🛡️ INVALID BOX GUARD: Ignore empty or tech signals at 0,0
                        box_w, box_h = abs(box[2] - box[0]), abs(box[3] - box[1])
                        if box_w < 5 or box_h < 5: continue 
                        if box[0] == 0 and box[1] == 0: continue

                        final_box = box
                        if st.session_state.get("snap_to_edges", False): final_box = snap_box_to_edges(st.session_state["image"], box)
                        h_img, w_img = st.session_state["image"].shape[:2]
                        final_box = [max(0, min(w_img, final_box[0])), max(0, min(h_img, final_box[1])), max(0, min(w_img, final_box[2])), max(0, min(h_img, final_box[3]))]
                        
                        # 🎨 FILL SELECTION MODE (Non-AI)
                        if st.session_state.get("fill_selection", False):
                            # Create a simple rectangular mask
                            mask = np.zeros((h_img, w_img), dtype=np.bool_)
                            mask[final_box[1]:final_box[3], final_box[0]:final_box[2]] = True
                        else:
                            # Standard AI Mode
                            sam.set_image(st.session_state["image"])
                            mask = sam.generate_mask(box_coords=final_box, level=st.session_state.get("mask_level", 0), is_wall_only=st.session_state.get("is_wall_only", False))
                        
                        if mask is not None: combined_mask = mask.copy() if combined_mask is None else combined_mask | mask
                    
                    if combined_mask is not None:
                         st.session_state["pending_selection"] = {'mask': combined_mask}
                         st.session_state["render_id"] += 1
                         safe_rerun()

            elif "Polygonal Lasso" in tool_mode:
                try:
                    # Capture force finish from global parser (unifies signals)
                    if is_finish:
                        st.session_state["just_finished_poly"] = True
                        print(f"DEBUG: Lasso Logic -> is_finish trigger: {is_finish}")
                    
                    url_pts = []
                    if url_pts_raw:
                        try:
                            # 🧩 Robustly handle list or string from Streamlit query params
                            raw_str = url_pts_raw[0] if isinstance(url_pts_raw, (list, tuple)) else str(url_pts_raw)
                            
                            # Handle mobile box input
                            mobile_box = st.query_params.get("box")
                            if mobile_box:
                                try:
                                    # Support multiple boxes separated by '|'
                                    # Format: "x1,y1,x2,y2|x1,y1,x2,y2,timestamp"
                                    box_strs = mobile_box.split("|")
                                    
                                    accumulated_mask = None
                                    last_box_coords = None # For persisting last view
                                    
                                    for b_str in box_strs:
                                        params = b_str.split(",")
                                        if len(params) < 4: continue
                                        
                                        # Extract coords 
                                        x1, y1, x2, y2 = float(params[0]), float(params[1]), float(params[2]), float(params[3])
                                        
                                        # Convert Canvas Coords -> Image Coords
                                        # Use existing context variables
                                        real_x1 = int(x1 / scale_factor) + start_x
                                        real_y1 = int(y1 / scale_factor) + start_y
                                        real_x2 = int(x2 / scale_factor) + start_x
                                        real_y2 = int(y2 / scale_factor) + start_y
                                        
                                        final_box = [real_x1, real_y1, real_x2, real_y2]
                                        last_box_coords = final_box

                                        # SAM Prediction
                                        # Assuming 'sam' is the SAM predictor object
                                        if not getattr(sam, "is_image_set", False): sam.set_image(st.session_state["image"])
                                        mask = sam.generate_mask(
                                            box_coords=final_box,
                                            level=st.session_state.get("mask_level", 0),
                                            is_wall_only=st.session_state.get("is_wall_only", False)
                                        )
                                        
                                        if mask is not None:
                                             if accumulated_mask is None: accumulated_mask = mask
                                             else: accumulated_mask = np.logical_or(accumulated_mask, mask)

                                    # Process final mask
                                    if accumulated_mask is not None:
                                        st.session_state["pending_selection"] = {'mask': accumulated_mask}
                                        st.session_state["pending_box_coords"] = last_box_coords 
                                        
                                        # ⚡ AUTO-APPLY
                                        cb_apply_pending()

                                        st.session_state["render_id"] += 1
                                        st.session_state["canvas_id"] = st.session_state.get("canvas_id", 0) + 1
                                        
                                        safe_rerun()
                                    
                                    st.query_params.pop("box", None)

                                except Exception as e:
                                    print(f"ERROR in mobile box: {e}")
                            
                            for p in raw_str.split(";"):
                                if "," in p:
                                    try:
                                        parts = p.split(",")
                                        # Only take first 2 parts as coordinates (ignore timestamp if appended)
                                        px, py = parts[0], parts[1]
                                        url_pts.append([int(float(px) / scale_factor) + start_x, int(float(py) / scale_factor) + start_y])
                                    except: pass
                            if is_finish: print(f"DEBUG: Parsed URL Pts: {len(url_pts)}")
                        except Exception as pe: print(f"DEBUG: Poly Parse Err: {pe}")

                    target_obj = None
                    for obj in reversed(objects):
                        if obj.get("type") in ["polygon", "path"]:
                            target_obj = obj
                            break
                    
                    if force_finish:
                        print(f"DEBUG: POLY FINISH -> target_obj: {target_obj.get('type') if target_obj else 'None'}, objects_count: {len(objects)}")
                        if not target_obj and not url_pts:
                             # Print objects for deep introspection if nothing found
                             print(f"DEBUG: Canvas Objects Sample: {str(objects)[:300]}")
                    
                    if force_finish: print(f"DEBUG: Lasso Logic -> force_finish: {force_finish}, target_obj: {target_obj.get('type') if target_obj else 'None'}")
                    
                    # 🏁 FINISH TRIGGER - Only trigger on explicit FINISH button click
                    if force_finish:
                        # SIGNAL GUARD: Ensure we only process this specific finish ONCE
                        if finish_sid and finish_sid == st.session_state.get("last_finish_sid"):
                            force_finish = False # Already processed
                        else:
                            if finish_sid: st.session_state["last_finish_sid"] = finish_sid

                    if force_finish:
                        # ALWAYS clear the session state signal immediately to prevent loops
                        st.session_state["force_finish_poly"] = False 
                        pts = []
                        
                        # DEBUG: Log extraction sources
                        print(f"DEBUG: Point Extraction -> url_pts count: {len(url_pts) if url_pts else 0}, target_obj: {target_obj.get('type') if target_obj else 'None'}")
                        
                        # Priority 1: URL Points (Most stable for mobile)
                        if url_pts and len(url_pts) > 2:
                            pts = url_pts
                            print(f"DEBUG: Using URL points: {len(pts)}")
                        # Priority 2: Native Canvas Object
                        elif target_obj:
                             obj_left, obj_top = target_obj.get("left", 0), target_obj.get("top", 0)
                             if target_obj["type"] == "polygon":
                                 canvas_points = target_obj.get("points", [])
                                 print(f"DEBUG: Canvas polygon points count: {len(canvas_points)}")
                                 for p in canvas_points:
                                     pts.append([int((p["x"] + obj_left) / scale_factor) + start_x, int((p["y"] + obj_top) / scale_factor) + start_y])
                             elif target_obj["type"] == "path":
                                 # 🧩 IMPROVED PATH PARSING (Handle unclosed shapes)
                                 for cmd in target_obj.get("path", []):
                                     # Supports MoveTo (M), LineTo (L), etc.
                                     if len(cmd) >= 3:
                                         pts.append([int(cmd[1] / scale_factor) + start_x, int(cmd[2] / scale_factor) + start_y])
                                     elif len(cmd) == 2 and isinstance(cmd[1], (list, tuple)): # Some versions nested
                                         pts.append([int(cmd[1][0] / scale_factor) + start_x, int(cmd[1][1] / scale_factor) + start_y])
                        
                        if len(pts) > 2:
                            is_fill_mode = st.session_state.get("fill_selection", False)
                            print(f"DEBUG: APPLY POLYGON -> Points: {len(pts)}, Finish: {is_finish}, Fill: {is_fill_mode}")
                            
                            if is_fill_mode or "Polygonal" in tool_mode:
                                # 🎨 EXACT SHAPE (Manual Fill)
                                # Default for Polygonal Lasso to ensure precision
                                mask = np.zeros((h, w), dtype=np.uint8)
                                cv2.fillPoly(mask, [np.array(pts, np.int32)], 255)
                                final_mask = mask > 0
                                print(f"DEBUG: Exact Shape Applied (Points: {len(pts)})")
                            else:
                                # 🧠 AI SAM MASK (Using lasso as box hint)
                                sam.set_image(st.session_state["image"])
                                pts_arr = np.array(pts)
                                box = [np.min(pts_arr[:,0]), np.min(pts_arr[:,1]), np.max(pts_arr[:,0]), np.max(pts_arr[:,1])]
                                final_mask = sam.generate_mask(box_coords=box, level=st.session_state.get("mask_level", 1), is_wall_only=st.session_state.get("is_wall_only", False))

                            if final_mask is not None and final_mask.any():
                                st.session_state["pending_selection"] = {'mask': final_mask}
                                cb_apply_pending() 
                                st.session_state["canvas_id"] = st.session_state.get("canvas_id", 0) + 1
                                st.session_state["render_id"] += 1
                                for pk in ["poly_pts", "force_finish", "tap"]:
                                    if pk in st.query_params: st.query_params.pop(pk, None)
                                components.html("<script>window.parent.STREAMLIT_POLY_POINTS = [];</script>", height=0)
                                safe_rerun()
                            else:
                                if force_finish: print(f"DEBUG: Skipping Poly Apply - Empty Mask")
                                for pk in ["poly_pts", "force_finish", "tap"]:
                                    if pk in st.query_params: st.query_params.pop(pk, None)
                                safe_rerun()
                        else:
                            if force_finish: 
                                print(f"DEBUG: Ignoring FINISH signal - Not enough points ({len(pts)})")
                                for pk in ["poly_pts", "force_finish", "tap"]:
                                    if pk in st.query_params: st.query_params.pop(pk, None)
                                safe_rerun()
                except Exception as e: logging.error(f"Poly Error: {e}")
                
            # --- PENDING SELECTION ACTIONS (Inside Fragment) ---
            if st.session_state.get("pending_selection") is not None:
                st.toast("✨ Selection Ready! Apply or Cancel below 👇", icon="🎨")
                st.markdown("### ✨ Confirm Selection")
                with st.container(border=True):
                    b_col1, b_col2, b_col3 = st.columns([1, 0.4, 1], gap="small", vertical_alignment="center")
                    with b_col1: 
                        if st.button("✨ APPLY", use_container_width=True, key="frag_apply", type="primary"):
                            cb_apply_pending(); safe_rerun() # Fragment scope default
                    with b_col2: st.color_picker("Color", st.session_state.get("picked_color", "#8FBC8F"), label_visibility="collapsed", key="frag_pending_color")
                    with b_col3: 
                        if st.button("🗑️ CANCEL", use_container_width=True, key="frag_cancel"):
                            cb_cancel_pending(); safe_rerun() # Fragment scope default

            # --- FINAL SIGNAL CLEANUP ---
            # Release guard if no interaction is currently being processed
            if not has_interaction:
                st.session_state["loop_guarded"] = False

    # --- MOBILE SYNC MARKER (Silent) ---
    st.markdown('<div class="sync-ghost-marker" style="display:none;" data-sync-id="mobile_ghost"></div>', unsafe_allow_html=True)
    
    # 5. BOTTOM NAV (Inside Fragment)
    st.markdown('<div class="mobile-zoom-wrapper" style="margin-top: 10px;"></div>', unsafe_allow_html=True)
    render_zoom_controls(key_suffix="frag", context_class="mobile-zoom-wrapper")
    
    # 6. UNDO/REDO (Inside Fragment)
    if st.session_state["masks"] or st.session_state.get("masks_redo"):
        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
        with btn_col1:
            if st.button("⏪ Undo", use_container_width=True, key="frag_undo_btn", disabled=not st.session_state["masks"]):
                cb_undo(); safe_rerun()
        with btn_col2:
            if st.button("⏩ Redo", use_container_width=True, key="frag_redo_btn", disabled=not st.session_state.get("masks_redo")):
                cb_redo(); safe_rerun()
        with btn_col3:
             if st.button("🗑️ Clear", use_container_width=True, key="frag_clear_btn"):
                cb_clear_all(); safe_rerun()

def render_visualizer_engine_v11(display_width):
    """De-fragmented wrapper for the canvas and external UI controls."""
    h, w, _ = st.session_state["image"].shape
    start_x, start_y, view_w, view_h = get_crop_params(w, h, st.session_state["zoom_level"], st.session_state["pan_x"], st.session_state["pan_y"])
    scale_factor = display_width / view_w
    tool_mode = st.session_state.get("selection_tool", TOOL_MAPPING["👆"])
    
    # 1. TOP TOOLBAR (Outside fragment) - HIDDEN
    # st.markdown('<div class="top-quick-toolbar" style="margin-bottom: 15px;">', unsafe_allow_html=True)
    # m_col1, m_col2, m_col3, m_col4 = st.columns([0.4, 0.2, 0.2, 0.2], gap="small", vertical_alignment="center")
    # with m_col1:
    #     # 🎯 SYNC FIX: Ensure top control follows session_state master
    #     current_icon = next((icon for icon, label in TOOL_MAPPING.items() if label == tool_mode), "👆")
    #     st.segmented_control("Tool", list(TOOL_MAPPING.keys()), default=current_icon, key="top_tool_switcher_control", label_visibility="collapsed", on_change=cb_top_tool_sync_v2)
    # with m_col2:
    #     st.toggle("Wall 🧱", value=st.session_state.get("is_wall_only", True), key="top_wall_control", label_visibility="collapsed", on_change=cb_top_wall_sync_v2)
    # with m_col3:
    #     st.segmented_control("Op", ["➕", "➖"], default="➕" if st.session_state.get("selection_op") == "Add" else "➖", key="top_op_control", label_visibility="collapsed", on_change=cb_top_op_sync)
    # with m_col4:
    #     if "Polygonal" in tool_mode:
    #         if st.button("🏁 FINISH", use_container_width=True, key="top_poly_finish", type="primary", help="Finish the polygon and apply paint."):
    #             st.session_state["force_finish_poly"] = True
    #             safe_rerun()
    #     elif st.session_state.get("pending_selection") is not None:
    #          if st.button("✨ APPLY", use_container_width=True, key="top_apply", type="primary"):
    #              cb_apply_pending()
    #              safe_rerun(scope="app")
    # st.markdown('</div>', unsafe_allow_html=True)

    # 2. COMPARISON (Outside fragment)
    if st.session_state.get("show_comparison", False):
        st.markdown("### 👁️ Comparison: Before vs After")
        c1, c2 = st.columns(2)
        with c1: st.image(st.session_state["image"], caption="Original", use_container_width=True)
        with c2: st.image(composite_image(st.session_state["image"], st.session_state["masks"]), caption="Painted", use_container_width=True)
        st.divider()

    # 3. THE CANVAS (Isolated fragment)
    if "AI Click" in tool_mode: drawing_mode = "point"
    elif "AI Object" in tool_mode: drawing_mode = "rect" if st.session_state.get("ai_drag_sub_tool") == "🆕 Draw New" else "transform"
    elif "Lasso" in tool_mode and "Polygonal" not in tool_mode: drawing_mode = "freedraw"
    elif "Polygonal" in tool_mode: drawing_mode = "polygon"
    else: drawing_mode = "transform"
    
    render_visualizer_canvas_fragment_v11(display_width, start_x, start_y, view_w, view_h, scale_factor, h, w, drawing_mode)

    # UI Controls (Zoom, Undo, etc.) have been moved INSIDE the fragment
    pass

    # Removed tuning_container from top - moved to bottom



def render_sidebar(sam, device_str):
    # --- 🌐 EXTRACT TRACKED POINTS (FOR DEBUG) ---
    url_pts_count = 0
    raw_param = st.query_params.get("poly_pts")
    if raw_param:
        try:
            if isinstance(raw_param, (list, tuple)): raw_pts_str = raw_param[0]
            else: raw_pts_str = str(raw_param)
            url_pts_count = len(raw_pts_str.split(";")) if raw_pts_str else 0
        except: pass



    with st.sidebar:
        # --- Native Sidebar Logic handles close ---
        pass

        st.markdown("<h3 style='margin:0 0 15px 35px; padding:0; color:#31333F;'>Visualizer Studio</h3>", unsafe_allow_html=True)
        
        if st.session_state.get("image") is not None:
            if st.button("🔄 Reset Project / Clear All", use_container_width=True):
                st.session_state["image"] = None
                st.session_state["image_path"] = None
                st.session_state["masks"] = []
                for k in list(st.session_state.keys()):
                    if any(k.startswith(p) for p in ["bg_url_cache_", "base_l_", "comp_cache_"]):
                        del st.session_state[k]
                st.session_state["render_cache"] = None
                st.session_state["composited_cache"] = None
                st.cache_data.clear()
                st.session_state["uploader_id"] += 1 
                safe_rerun()
            st.divider()

        uploader_key = f"uploader_{st.session_state.get('uploader_id', 0)}"
        uploaded_file = st.file_uploader("Start Project", type=["jpg", "png", "jpeg"], label_visibility="collapsed", key=uploader_key)
        
        if uploaded_file is not None:
            file_key = getattr(uploaded_file, "file_id", f"{uploaded_file.name}_{uploaded_file.size}")
            if st.session_state.get("image_path") != file_key:
                st.toast(f"📸 Loading New Image: {uploaded_file.name}", icon="🔄")
                uploaded_file.seek(0)
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.cvtColor(cv2.imdecode(file_bytes, 1), cv2.COLOR_BGR2RGB)
                st.session_state["image_original"] = image.copy()
                max_dim = 1100
                h, w = image.shape[:2]
                if max(h, w) > max_dim:
                    scale = max_dim / max(h, w)
                    image = cv2.resize(image, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
                st.session_state["image"] = image
                st.session_state["image_path"] = file_key
                st.session_state["masks"] = []
                st.session_state["pending_selection"] = None # 🎯 Reset pending selections
                st.session_state["pending_boxes"] = []      # 🎯 Reset pending boxes
                st.session_state["last_click_global"] = None
                st.session_state["zoom_level"] = 1.0
                st.session_state["pan_x"] = 0.5
                st.session_state["pan_y"] = 0.5
                st.session_state["render_id"] = 0
                st.session_state["canvas_id"] = st.session_state.get("canvas_id", 0) + 1
                
                # Clear all technical caches
                for k in ["global_base_lab", "lab_cache_id", "lab_cache_dim", "render_cache", "composited_cache"]:
                    if k in st.session_state:
                         del st.session_state[k]
                
                for k in list(st.session_state.keys()):
                    if any(k.startswith(p) for p in ["bg_url_cache_", "base_l_", "comp_cache_"]):
                        del st.session_state[k]
                
                st.cache_data.clear()
                # 🧠 Prime the engine immediately
                try:
                    sam.set_image(image)
                except:
                    pass
                st.rerun() 

        if st.session_state.get("image") is None:
             st.markdown("<div style='background:#f3f4f6; padding:15px; border-radius:10px; border:1px dashed #d1d5db; margin:10px 0;'><p style='margin:0; font-size:0.85rem; color:#4b5563; line-height:1.4;'><b>Ready to paint?</b><br>Upload a photo of your wall or room to begin.</p></div>", unsafe_allow_html=True)
             st.caption("Supported formats: JPG, PNG, JPEG")
             return

        if st.session_state.get("image") is not None:
            if st.session_state["masks"]:
                if st.button("💎 Prepare High-Res Download", use_container_width=True):
                    st.toast("Processing 4K Export...", icon="💎")
                    try:
                        original_img = st.session_state["image_original"]
                        oh, ow = original_img.shape[:2]
                        high_res_masks = []
                        for m_data in st.session_state["masks"]:
                            hr_m = m_data.copy()
                            mask_uint8 = (m_data['mask'] * 255).astype(np.uint8)
                            hr_mask_uint8 = cv2.resize(mask_uint8, (ow, oh), interpolation=cv2.INTER_LINEAR)
                            hr_m['mask'] = hr_mask_uint8 > 127
                            hr_m['mask_soft'] = None 
                            high_res_masks.append(hr_m)
                        from core.colorizer import ColorTransferEngine
                        dl_comp = ColorTransferEngine.composite_multiple_layers(original_img, high_res_masks)
                        dl_pil = Image.fromarray(dl_comp)
                        dl_buf = io.BytesIO()
                        dl_pil.save(dl_buf, format="PNG")
                        st.session_state["last_export"] = dl_buf.getvalue()
                        st.success("✅ Download Ready!")
                    except Exception as e:
                        st.error(f"Export failed: {e}")
                        import logging
                        logging.error(f"High-res export failed: {e}", exc_info=True)

                if st.session_state.get("last_export"):
                    st.download_button(label="📥 Save Final Image", data=st.session_state["last_export"], file_name="pro_visualizer_design.png", mime="image/png", use_container_width=True)
            
            
            st.divider()
            st.subheader("🛠️ Selection Tool")
            
            # Use standardized tool options
            tool_options = list(TOOL_MAPPING.values())
            current_tool = st.session_state.get("selection_tool", TOOL_MAPPING["👆"])
            try:
                tool_idx = tool_options.index(current_tool)
            except ValueError:
                tool_idx = 0

            # Use index parameter directly - don't set via session state before widget creation
            # This prevents Streamlit's "default value + Session State API" warning
            
            st.radio("Method", tool_options, 
                                    index=tool_idx, 
                                    horizontal=True, 
                                    key="sidebar_tool_radio",
                                    on_change=cb_sidebar_tool_sync)
            
            # DEBUG TRACE
            if "selection_tool" in st.session_state:
                print(f"DEBUG: SIDEBAR RENDER -> Tool: {st.session_state['selection_tool']}, Radio Key: {st.session_state.get('sidebar_tool_radio')}")
            
            st.radio("Operation", ["Add", "Subtract"], 
                                    horizontal=True, 
                                    index=0 if st.session_state.get("selection_op") == "Add" else 1,
                                    key="sidebar_op_radio",
                                    on_change=cb_sidebar_op_sync)
            
            # --- Tool Specific Controls (Full Width) ---
            if any(t in current_tool for t in ["AI Click", "Lasso", "AI Object"]):
                st.caption("Layer Scope:")
                prec_options = ["Standard Walls", "Small Details", "Whole Object"]
                current_idx = min(st.session_state.get("mask_level", 0), 2)
                prec_mode = st.radio("Segmentation Mode", prec_options, index=current_idx, key="sidebar_prec_mode_radio", label_visibility="collapsed")
                new_level = prec_options.index(prec_mode)
                if new_level != st.session_state.get("mask_level", 0):
                    st.session_state["mask_level"] = new_level
                    safe_rerun()
                # ⚡ Always Instant Apply
                st.session_state["ai_click_instant_apply"] = True

            if "AI Object" in current_tool:
                st.session_state["ai_drag_sub_tool"] = st.radio("Action", ["🆕 Draw New", "🖱️ Move"], index=0 if st.session_state.get("ai_drag_sub_tool") == "🆕 Draw New" else 1, horizontal=True)
                # st.session_state["snap_to_edges"] = st.toggle("Snap to Edges 🔍", value=st.session_state.get("snap_to_edges", False), help="Attempts to automatically align your box to nearby architectural lines.")
            
            if "Polygonal" in current_tool:
                # Use columns for buttons, but now they have full sidebar width to split
                p_colA, p_colB = st.columns(2)
                with p_colA:
                    if st.button("✅ FINISH", use_container_width=True, type="primary", key="side_poly_finish", help="Complete the shape (or triple-click last point)"):
                        st.session_state["force_finish_poly"] = True
                        safe_rerun()
                with p_colB:
                    if st.button("🧹 CLEAR", use_container_width=True, key="side_poly_clear", help="Clear drawing"):
                        st.session_state["canvas_id"] = st.session_state.get("canvas_id", 0) + 1
                        st.session_state["force_finish_poly"] = False
                        st.query_params.pop("poly_pts", None)
                        import streamlit.components.v1 as components
                        components.html("<script>window.parent.STREAMLIT_POLY_POINTS = [];</script>", height=0)
                if "Polygonal Lasso" in current_tool:
                    st.caption("Instructions: Click to add points. **Double-click** or click **FINISH** below to apply paint.")
                elif "Lasso (Freehand)" in current_tool:
                    st.caption("Instructions: Draw your area. Paint applies when you release.")
            
            # 🛠️ LOGIC: If "Fill Selection" (Manual) is ON, "Wall Priority" (AI) is irrelevant
            is_fill_active = st.session_state.get("fill_selection", False)

            if "Lasso (Freehand)" in current_tool:
                st.toggle("Apply Paint Fully Inside 🎨", 
                      value=is_fill_active, 
                      key="fill_selection_toggle",
                      on_change=lambda: st.session_state.update({"fill_selection": st.session_state.fill_selection_toggle}),
                      help="**ON:** Paints the *entire* inside of your Lasso selection (Manual Mode).\n\n**OFF (AI Mode):** Uses AI to intelligently detect objects within your lasso.")

            st.toggle("Wall Priority Mode 🧱", 
                      value=st.session_state.get("is_wall_only", True), 
                      key="wall_priority_toggle",
                      disabled=(is_fill_active and "Lasso (Freehand)" in current_tool), 
                      on_change=lambda: st.session_state.update({"is_wall_only": st.session_state.wall_priority_toggle}),
                      help="**ON (Default):** Stricter borders. Keeps paint on the specific wall face you clicked.\n\n**OFF:** Relaxes borders. Use this if paint is leaving gaps or for furniture/small objects.")

            # HIDDEN: User requested to remove this to avoid confusion
            # st.toggle("Fill Entire Selection (Non-AI) 🎨", 
            #           value=is_fill_active, 
            #           key="fill_selection_toggle",
            #           on_change=lambda: st.session_state.update({"fill_selection": st.session_state.fill_selection_toggle}),
            #           help="**ON:** Paints the *entire* inside of your Box or Lasso selection. Best for simple walls or when AI misses details.\n\n**OFF (AI Mode):** Uses AI to intelligently find objects within your marked area.")

            if st.session_state.get("pending_selection") is not None or st.session_state.get("pending_boxes"):
                if st.button("🚫 Clear Selection Draft", use_container_width=True, help="Reset the current selection without applying it."):
                    st.session_state["pending_selection"] = None
                    st.session_state["pending_boxes"] = []
                    st.session_state["canvas_id"] = st.session_state.get("canvas_id", 0) + 1
                    safe_rerun()

            # --- Selection Layer Tuning ---
            if st.session_state.get("pending_selection") is not None:
                with st.expander("✨ Selection Layer Tuning", expanded=True):
                    # Local selection opacity (visual only for the highlight)
                    st.slider("Selection Visibility", 0.1, 1.0, 0.5, key="selection_highlight_opacity")
                    # Edge Feathering (applied before final apply)
                    st.slider("Edge Softness", 0, 5, 0, key="selection_softness", help="Feather the edges of the selection before applying paint.")
                    # Finish Selection
                    st.radio("Paint Finish", ["Standard", "Matte", "Satin", "Gloss"], horizontal=True, key="selection_finish")
            
            st.divider()
            st.subheader("👁️ View Settings")
            st.toggle("Compare Before/After", key="show_comparison")

            # Render Desktop Zoom Controls (Hidden on Mobile)
            # render_zoom_controls(key_suffix="sidebar", context_class="desktop-zoom-wrapper")
            st.divider()
            sidebar_paint_fragment()
            st.divider()
            if st.session_state["masks"] or st.session_state.get("masks_redo"):
                col_u1, col_u2, col_u3 = st.columns(3)
                with col_u1: st.button("⏪ Undo", use_container_width=True, on_click=cb_undo, key="sidebar_undo", disabled=not st.session_state["masks"])
                with col_u2: st.button("⏩ Redo", use_container_width=True, on_click=cb_redo, key="sidebar_redo", disabled=not st.session_state.get("masks_redo"))
                with col_u3: st.button("🗑️ Clear", use_container_width=True, on_click=cb_clear_all, key="sidebar_clear")
                st.write("---")
                for i in range(len(st.session_state["masks"]) - 1, -1, -1):
                    mask_data = st.session_state["masks"][i]
                    with st.container(border=True):
                        try: r1, r2, r3, r4 = st.columns([0.45, 0.15, 0.25, 0.15], vertical_alignment="center")
                        except: r1, r2, r3, r4 = st.columns([0.45, 0.15, 0.25, 0.15])
                        cur_c = mask_data.get('color', '#FFFFFF')
                        with r1: st.markdown(f"**Layer {i+1}**")
                        with r2: st.markdown(f"<div style='width:24px;height:24px;background:{cur_c};border-radius:4px;border:1px solid #ddd;'></div>", unsafe_allow_html=True)
                        with r3: st.markdown(f"`{cur_c}`")
                        with r4: st.button("🗑️", key=f"sidebar_del_{i}", on_click=cb_delete_layer, args=(i,))
                        
                        # Real-time Refinement Slider
                        # HIDDEN: User requested to hide refinement (Step 2281)
                        # new_ref = st.slider("Expansion / Contraction", -10, 10, int(mask_data.get('refinement', 0)), key=f"refine_{i}")
                        # if new_ref != mask_data.get('refinement'):
                        #     mask_data['refinement'] = new_ref
                        #     st.session_state["render_id"] += 1
            else: st.caption("No active layers.")

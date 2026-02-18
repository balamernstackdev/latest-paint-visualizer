
import streamlit as st
import os
import cv2
import numpy as np
import torch
import streamlit.components.v1 as components
from scipy import sparse

import logging
import time
import io
import requests
import textwrap
from PIL import Image
from streamlit_drawable_canvas import st_canvas as raw_st_canvas
from streamlit_image_comparison import image_comparison
from .encoding import image_to_url_patch
from .state_manager import cb_undo, cb_redo, cb_clear_all, cb_delete_layer, cb_apply_pending, cb_cancel_pending
from .image_processing import get_crop_params, composite_image, process_lasso_path
from .sam_loader import get_sam_engine, CHECKPOINT_PATH, MODEL_TYPE
from .performance import cleanup_session_caches

# --- UI CONSTANTS ---
TOOL_MAPPING = {
    "👆": "👆 AI Click (Point)",
    "✨": "✨ AI Object (Box)",
    "🕸️": "🕸️ Polygonal Lasso",
    "✏️": "✏️ Paint Brush",
    "🧹": "🧹 Eraser Tool",
    "🪄": "🪄 Magic Wand"
}

# --- HELPERS ---
def safe_rerun(scope="fragment"):
    """Prevents StreamlitAPIException during full-rerun transitions."""
    try:
        st.rerun(scope=scope)
    except:
        st.rerun()

def cb_top_tool_sync_v2():
    new_icon = st.session_state.get("top_tool_switcher_control")
    if new_icon and new_icon in TOOL_MAPPING:
        st.session_state["selection_tool"] = TOOL_MAPPING[new_icon]
        st.session_state["sidebar_tool_radio"] = TOOL_MAPPING[new_icon]
        st.session_state["canvas_id"] = st.session_state.get("canvas_id", 0) + 1
        st.session_state["canvas_raw"] = {}
        st.session_state["pending_selection"] = None

def cb_top_wall_sync_v2():
    st.session_state["is_wall_only"] = st.session_state.get("top_wall_control", True)
    st.session_state["sidebar_wall_toggle"] = st.session_state["is_wall_only"]

def cb_sidebar_tool_sync(widget_key=None):
    widget_key = "sidebar_tool_radio"
    new_tool = st.session_state.get(widget_key)
    
    if new_tool is None: return

    last_tool = st.session_state.get("selection_tool")
    if last_tool != new_tool:
        for pk in ["tap", "poly_pts", "pan_update", "zoom_update", "force_finish"]:
            if pk in st.query_params: st.query_params.pop(pk, None)
        st.session_state["force_finish_poly"] = False
        st.session_state["loop_guarded"] = False
        
        if "Eraser" in new_tool:
            st.session_state["selection_op"] = "Subtract"
            st.session_state["sidebar_op_radio"] = "Subtract"
        else:
            st.session_state["selection_op"] = "Add"
            st.session_state["sidebar_op_radio"] = "Add"
        
        st.session_state["tool_switched_reset"] = True
        st.session_state["fill_selection"] = False
        print(f"DEBUG: Tool Switched -> {last_tool} -> {new_tool}. Signals wiped.")

    st.session_state["selection_tool"] = new_tool
    st.session_state["sidebar_tool_radio"] = new_tool
    for icon, label in TOOL_MAPPING.items():
        if label == new_tool:
            st.session_state["top_tool_switcher_control"] = icon
            break
    st.session_state["canvas_id"] = st.session_state.get("canvas_id", 0) + 1
    st.session_state["canvas_raw"] = {}
    st.session_state["pending_selection"] = None

def cb_sidebar_wall_sync():
    st.session_state["is_wall_only"] = st.session_state.get("sidebar_wall_toggle")
    st.session_state["top_wall_control"] = st.session_state["is_wall_only"]

def cb_sidebar_op_sync():
    st.session_state["selection_op"] = st.session_state.get("sidebar_op_radio", "Add")
    st.session_state["top_op_control"] = "➕" if st.session_state["selection_op"] == "Add" else "➖"

def cb_top_op_sync():
    icon = st.session_state.get("top_op_control")
    st.session_state["selection_op"] = "Add" if icon == "➕" else "Subtract"
    st.session_state["sidebar_op_radio"] = st.session_state["selection_op"]

def sort_points_clockwise(pts):
    if not pts or len(pts) < 3: return pts
    center_x = sum(p[0] for p in pts) / len(pts)
    center_y = sum(p[1] for p in pts) / len(pts)
    import math
    def get_angle(point):
        return math.atan2(point[1] - center_y, point[0] - center_x)
    return sorted(pts, key=get_angle)

def snap_box_to_edges(image, box, margin=15):
    if image is None: return box
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    grad_y = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    
    def refine_coord(coord, is_x):
        search_range = range(max(0, coord - margin), min((w if is_x else h), coord + margin))
        best_coord = coord
        max_grad = 0
        target_grad = grad_x if is_x else grad_y
        for i in search_range:
            if is_x:
                g_sum = np.sum(target_grad[max(0, y1-margin):min(h, y2+margin), i])
            else:
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

def st_canvas(*args, **kwargs):
    """Wrapper to handle background image conversion to data URLs."""
    kwargs["background_color"] = "rgba(0,0,0,0)"
    bg_img = kwargs.get("background_image")
    
    if bg_img is not None:
        width, height = kwargs.get("width"), kwargs.get("height")
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

def setup_styles():
    css_path = os.path.join(os.path.dirname(__file__), "..", "assets", "style.css")
    style_content = ""
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            style_content = f.read()
            
    full_css = textwrap.dedent(f"""
        <style>
        {style_content}
        :root {{ --primary-color: #ff4b4b; --background-color: #ffffff; --secondary-background-color: #f0f2f6; --text-color: #31333F; --font: "Segoe UI", sans-serif; }}
        html, body, .stApp {{ touch-action: manipulation !important; overscroll-behavior-y: none; }}
        iframe[title="streamlit_drawable_canvas.st_canvas"], .element-container iframe, [id$="-overlay"] {{ touch-action: none !important; -webkit-touch-callout: none !important; -webkit-user-select: none !important; user-select: none !important; -webkit-user-drag: none !important; user-drag: none !important; pointer-events: auto !important; }}
        html, body, .stApp {{ -webkit-touch-callout: none !important; }}
        [data-testid="stSidebar"] {{ background-color: #f8f9fa !important; border-right: 1px solid #e6e6e6; width: 350px !important; }}
        @media (min-width: 769px) {{ .m-open-container, .desktop-only {{ display: none !important; }} }}
        header[data-testid="stSidebarHeader"] {{ visibility: visible !important; display: flex !important; justify-content: space-between !important; alignItems: center !important; padding: 10px 15px !important; background: transparent !important; }}
        [data-testid="stSidebarCollapseButton"] {{ visibility: visible !important; display: flex !important; opacity: 1 !important; }}
        [data-testid="stSidebarCollapseButton"] button {{ background: transparent !important; border: none !important; box-shadow: none !important; color: #31333F !important; padding: 0 !important; width: auto !important; height: auto !important; }}
        [data-testid="stSidebarCollapseButton"] button:hover {{ background: rgba(0,0,0,0.05) !important; border-radius: 4px !important; }}
        [data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] {{ position: absolute !important; top: 15px !important; right: 15px !important; z-index: 10000 !important; }}
        section[data-testid="stMain"] [data-testid="stSidebarCollapseButton"] {{ position: fixed !important; top: 15px !important; left: 15px !important; z-index: 10000 !important; }}
        [data-testid="stSidebarHeader"] svg {{ fill: #31333F !important; width: 22px !important; height: 22px !important; }}
        @media (max-width: 768px) {{
            .desktop-only {{ display: none !important; }}
            html, body, .stApp {{ touch-action: manipulation !important; overscroll-behavior-y: none; }}
            iframe[title="streamlit_drawable_canvas.st_canvas"], .element-container iframe, [id$="-overlay"] {{ touch-action: none !important; -webkit-touch-callout: none !important; -webkit-user-select: none !important; user-select: none !important; -webkit-user-drag: none !important; user-drag: none !important; pointer-events: auto !important; }}
            html, body, .stApp {{ -webkit-touch-callout: none !important; }}
            [data-testid="stSidebar"] {{ transition: transform 0.3s cubic-bezier(0, 0, 0.2, 1) !important; width: 82vw !important; min-width: 82vw !important; }}
            [data-testid="stSidebar"] > div:first-child {{ width: 100% !important; min-width: 100% !important; }}
            .mobile-bottom-actions {{ position: fixed; bottom: 20px; left: 10px; right: 10px; background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(15px); padding: 12px; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); z-index: 1000; border: 1px solid rgba(255,255,255,0.3); }}
            .main .block-container {{ padding-bottom: 140px !important; padding-top: 1rem !important; }}
        }}
        .sidebar-divider {{ height: 1px; background: rgba(0,0,0,0.05); margin: 5px 0 !important; width: 100%; }}
        .sidebar-card {{ padding: 2px 0 !important; margin-bottom: 2px !important; background: transparent !important; border: none !important; box-shadow: none !important; }}
        .sidebar-header-text {{ font-size: 0.8rem !important; font-weight: 700; color: #666; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 5px !important; display: flex; align-items: center; gap: 6px; }}
        div[data-testid="stButton"]:has(button p:contains("GHOST")) {{ display: none !important; }}
        div[data-testid="stButton"]:has(button:contains("GHOST")) {{ display: none !important; }}
        div[data-testid="stButton"]:has(button p:contains("GLOBAL SYNC")) {{ display: none !important; }}
        div[data-testid="stButton"]:has(button:contains("GLOBAL SYNC")) {{ display: none !important; }}
        div.element-container:has(#global-sync-anchor), div.element-container:has(.global-sync-marker), div.element-container:has(.sync-ghost-marker) {{ display: none !important; height: 0 !important; margin: 0 !important; padding: 0 !important; position: fixed !important; top: -10000px !important; }}
        .element-container:has(.sync-ghost-marker)+.element-container, .element-container:has(.global-sync-marker)+.element-container, div.element-container:has(#global-sync-anchor)+div.element-container, div.element-container:has(button[key="global_sync_btn"]) {{ display: none !important; position: fixed !important; top: -10000px !important; width: 0 !important; }}
        .stButton>button {{ border-radius: 8px !important; font-weight: 600 !important; transition: all 0.2s !important; }}
        .landing-container {{ text-align: center !important; padding: 4rem 1rem !important; max-width: 800px !important; margin: 0 auto !important; }}
        .landing-header h1 {{ font-size: 2.8rem !important; font-weight: 800 !important; color: #111827 !important; margin-bottom: 1rem !important; }}
        .landing-sub p {{ font-size: 1.2rem !important; color: #4b5563 !important; margin-bottom: 2rem !important; }}
        </style>
    """).strip()
    
    st.markdown(full_css, unsafe_allow_html=True)

    st.markdown("""
        <script>
        (function() {
            const meta = window.parent.document.createElement('meta');
            meta.name = 'viewport';
            meta.content = 'width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover';
            const existing = window.parent.document.querySelector('meta[name="viewport"]');
            if (existing) { existing.content = meta.content; } else { window.parent.document.getElementsByTagName('head')[0].appendChild(meta); }
            const _backupWarn = console.warn;
            const _backupError = console.error;
            console.warn = function() { if (arguments[0] && typeof arguments[0] === 'string' && arguments[0].includes('Streamlit')) return; _backupWarn.apply(console, arguments); };
            console.error = function() { if (arguments[0] && typeof arguments[0] === 'string' && arguments[0].includes('Streamlit')) return; _backupError.apply(console, arguments); };
            const ensureUI = () => {
                try {
                    const doc = window.parent.document;
                    const overlayers = doc.querySelectorAll('[data-testid="stSidebar"] + div');
                    overlayers.forEach(ov => { if (!ov.id && !ov.classList.contains('stMain')) { ov.style.pointerEvents = 'none'; ov.style.opacity = '0'; } });
                } catch(e) {}
            };
            setInterval(ensureUI, 500); 
            window.parent.addEventListener('mouseup', ensureUI);
            window.parent.addEventListener('touchend', ensureUI);
        })();
        </script>
    """, unsafe_allow_html=True)

def sidebar_paint_fragment():
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

def render_zoom_controls(key_suffix="", context_class=""):
    if context_class:
        st.markdown(f'<div class="{context_class}">', unsafe_allow_html=True)
    if st.session_state.get("zoom_level", 1.0) > 1.0 or st.session_state.get("pan_x", 0.5) != 0.5 or st.session_state.get("pan_y", 0.5) != 0.5:
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
    query_params = st.query_params
    mobile_tap = query_params.get("tap", "")
    mobile_box = query_params.get("box", "")
    mobile_pan = query_params.get("pan_update", "")
    mobile_zoom = query_params.get("zoom_update", "")
    url_pts_raw = query_params.get("poly_pts", "")
    force_finish_raw = query_params.get("force_finish", "")
    
    if st.session_state.get("tool_switched_reset", False):
        st.session_state["tool_switched_reset"] = False
        components.html("<script>if(window.parent.STREAMLIT_POLY_POINTS) window.parent.STREAMLIT_POLY_POINTS = [];</script>", height=0)

    def extract_signal(raw_str):
        if not raw_str or (isinstance(raw_str, str) and raw_str.strip() == ""): return None, None
        if isinstance(raw_str, list): raw_str = raw_str[0]
        parts = raw_str.split(",")
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

    is_guarded = st.session_state.get("loop_guarded", False)
    has_interaction = any([mobile_tap, mobile_box, mobile_pan, mobile_zoom, url_pts_raw])
    skip_processing = has_interaction and is_guarded and not is_finish
    
    if skip_processing:
        for pk in ["tap", "box", "pan_update", "zoom_update", "poly_pts"]:
            if pk in st.query_params: st.query_params.pop(pk, None)
    
    if has_interaction and not skip_processing:
        st.session_state["loop_guarded"] = True
        print(f"DEBUG: Processing Interaction -> Tool:{drawing_mode}, Tap:{bool(mobile_tap)}, Box:{bool(mobile_box)}, Finish:{is_finish}")

    if st.session_state.get("pending_selection") is not None:
        op_label = "Add"
        if st.session_state.get("selection_op") == "Subtract":
            op_label = "Subtract"
            btn_label = "🧹 ERASE PAINT"
            btn_type = "secondary" 
        else:
            btn_label = "✨ APPLY PAINT"
            btn_type = "primary"
            
        st.info(f"✨ Selection Active ({op_label})! Confirm below.", icon="👇")
        
        with st.container(border=True):
            cols = st.columns([1, 1], gap="small")
            with cols[0]: 
                if st.button(btn_label, use_container_width=True, key="top_frag_apply", type="primary"):
                    print(f"DEBUG: PROCEEDING WITH OP: {op_label}")
                    cb_apply_pending(); safe_rerun()
            with cols[1]: 
                if st.button("🗑️ CANCEL", use_container_width=True, key="top_frag_cancel"):
                    cb_cancel_pending(); safe_rerun()

    original_img = st.session_state["image"]
    
    # 🔍 debug logic
    msg_len = len(st.session_state.get("masks", []))
    print(f"DEBUG: Entering Render. Masks Count: {msg_len}")
    
    painted_img = composite_image(original_img, st.session_state["masks"])
    show_comp = st.session_state.get("show_comparison", False)
    
    display_img = original_img if show_comp else painted_img
    
    # 🔍 debug check
    if not show_comp and msg_len > 0:
        diff = np.sum(np.abs(original_img.astype(float) - painted_img.astype(float)))
        print(f"DEBUG: Painted Diff: {diff}")
        
    cropped_view = display_img[start_y:start_y+view_h, start_x:start_x+view_w]
    new_h = int(view_h * (display_width / view_w))
    final_display_image = cv2.resize(cropped_view, (display_width, new_h), interpolation=cv2.INTER_LINEAR)
    
    if st.session_state.get("zoom_level", 1.0) > 1.0:
        final_display_image = overlay_pan_controls(final_display_image)

    display_height = final_display_image.shape[0]
    sam = get_sam_engine(CHECKPOINT_PATH, MODEL_TYPE)

    if mobile_box and mobile_box.strip() != "" and not skip_processing:
        if box_sid and box_sid == st.session_state.get("last_box_sid"):
            mobile_box = None
        else:
            if box_sid: st.session_state["last_box_sid"] = box_sid

        if mobile_box:
            try:
                print(f"DEBUG: Msg received -> {mobile_box}")
                box_strs = mobile_box.split("|")
                accumulated_mask = None
                last_box_coords = None

                for b_str in box_strs:
                    print(f"DEBUG: Processing box token -> {b_str}")
                    parts = b_str.split(",")
                    if len(parts) < 4: continue
                    
                    x1, y1, x2, y2 = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                    
                    bx1 = int(x1 / scale_factor) + start_x
                    by1 = int(y1 / scale_factor) + start_y
                    bx2 = int(x2 / scale_factor) + start_x
                    by2 = int(y2 / scale_factor) + start_y
                    
                    final_box = [min(bx1, bx2), min(by1, by2), max(bx1, bx2), max(by1, by2)]
                    last_box_coords = final_box

                    if abs(final_box[2] - final_box[0]) > 5 and abs(final_box[3] - final_box[1]) > 5:
                        if not getattr(sam, "is_image_set", False): 
                            with st.spinner("🧠 AI is analyzing image..."):
                                sam.set_image(st.session_state["image"])
                            
                        mask = sam.generate_mask(
                            box_coords=final_box, 
                            level=st.session_state.get("mask_level", 0), 
                            is_wall_only=st.session_state.get("is_wall_only", False)
                        )
                        
                        if mask is not None:
                             if accumulated_mask is None: accumulated_mask = mask
                             else: accumulated_mask = np.logical_or(accumulated_mask, mask)

                if accumulated_mask is not None:
                    st.session_state["pending_selection"] = {'mask': accumulated_mask}
                    st.session_state["pending_box_coords"] = last_box_coords 
                    cb_apply_pending()
                    st.session_state["render_id"] += 1
                    st.session_state["canvas_id"] = st.session_state.get("canvas_id", 0) + 1
                    safe_rerun()
                
                st.query_params.pop("box", None)
            except Exception as e:
                print(f"ERROR in mobile box logic: {e}")
                
    if mobile_pan and mobile_pan.strip() != "":
        if pan_sid and pan_sid == st.session_state.get("last_pan_sid"):
            mobile_pan = None
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

    initial_drawing = {"version": "4.4.0", "objects": []}
    was_just_finished = st.session_state.get("just_finished_poly", False)
    st.session_state["just_finished_poly"] = False

    if st.session_state.get("canvas_raw") and not was_just_finished:
        for obj in (st.session_state.get("canvas_raw") or {}).get("objects", []):
            obj_type = obj.get("type")
            is_valid = False
            if drawing_mode == "point" and obj_type == "circle": is_valid = True
            elif drawing_mode == "rect" and obj_type == "rect": is_valid = True
            elif drawing_mode == "freedraw" and obj_type == "path": is_valid = True
            elif drawing_mode == "polygon" and obj_type == "polygon": is_valid = True
            elif drawing_mode == "transform": is_valid = True
            if is_valid:
                initial_drawing["objects"].append(obj)

    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=2,
        stroke_color="#000",
        background_image=final_display_image,
        update_streamlit=True,
        width=display_width,
        height=display_height,
        drawing_mode=drawing_mode,
        point_display_radius=0 if drawing_mode == "point" else 3,
        key=f"canvas_{st.session_state.get('canvas_id', 0)}",
        initial_drawing=initial_drawing,
        display_toolbar=False,
    )

    if canvas_result.json_data:
        st.session_state["canvas_raw"] = canvas_result.json_data

# --- RESTORED FUNCTIONS ---

def render_visualizer_engine_v11(display_width=800):
    """Main engine that orchestrates the canvas and drawing modes."""
    if "image" not in st.session_state or st.session_state["image"] is None:
        return

    # 1. Dimensions
    img = st.session_state["image"]
    h, w = img.shape[:2]
    
    # 2. View Calculation
    zoom = st.session_state.get("zoom_level", 1.0)
    pan_x = st.session_state.get("pan_x", 0.5)
    pan_y = st.session_state.get("pan_y", 0.5)
    
    start_x, start_y, view_w, view_h = get_crop_params(w, h, zoom, pan_x, pan_y)
    scale_factor = display_width / view_w
    
    # 3. Determine Mode
    tool = st.session_state.get("selection_tool", "👆 AI Click (Point)")
    mode = "point"
    if "Box" in tool: mode = "rect"
    elif "Lasso" in tool: mode = "freedraw"
    elif "Polygon" in tool: mode = "polygon"
    elif "Paint" in tool: mode = "freedraw"
    elif "Wand" in tool: mode = "point"
    elif "Eraser" in tool: mode = "transform" # Use transform for erasing/moving? Or freedraw? Typically Eraser is just an operation mode using click/drag.
    
    # 4. Render Canvas Fragment
    render_visualizer_canvas_fragment_v11(
        display_width, start_x, start_y, view_w, view_h, 
        scale_factor, h, w, mode
    )

def render_sidebar(sam_engine, device_str):
    """Renders the sidebar controls."""
    with st.sidebar:
        st.title("🎨 Visualizer")
        
        # 1. Image Upload
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"], key="uploaded_file")
        if uploaded_file is not None:
             # Basic handling if not handled elsewhere
             try:
                 file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                 image = cv2.imdecode(file_bytes, 1)
                 image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                 
                 # Only update if new
                 if "image_hash" not in st.session_state or st.session_state["image_hash"] != uploaded_file.name:
                     st.session_state["image"] = image
                     st.session_state["image_hash"] = uploaded_file.name
                     st.session_state["masks"] = []
                     st.session_state["canvas_id"] += 1
                     st.rerun()
             except: pass
        
        st.divider()

        # 2. Tool Selection
        st.subheader("🛠️ Tools")
        st.radio("Select Tool", list(TOOL_MAPPING.values()), key="sidebar_tool_radio", on_change=cb_sidebar_tool_sync)
        
        # Paint Color
        sidebar_paint_fragment()
        
        st.divider()
        
        # 3. Operation
        st.subheader("⚙️ Operation")
        st.radio("Mode", ["Add", "Subtract"], key="sidebar_op_radio", on_change=cb_sidebar_op_sync)
        
        st.checkbox("Restict to Walls", key="sidebar_wall_toggle", on_change=cb_sidebar_wall_sync)
        
        st.divider()
        
        # 4. Layers
        st.subheader("📚 Layers")
        if st.button("🗑️ Clear All Layers", use_container_width=True):
            cb_clear_all()
            st.rerun()
            
        for i, layer in enumerate(st.session_state.get("masks", [])):
            with st.expander(f"{layer['name']}", expanded=False):
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("👁️ Toggle", key=f"vis_{i}"):
                        layer['visible'] = not layer['visible']
                        st.session_state["render_id"] += 1
                        st.rerun()
                with col2:
                    if st.button("❌ Delete", key=f"del_{i}"):
                        cb_delete_layer(i)
                        st.rerun()

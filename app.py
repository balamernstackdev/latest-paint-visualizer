import sys
import types
from io import BytesIO
import base64
import streamlit as st
import os
import torch
import warnings
import logging
import numpy as np
import cv2
from scipy import sparse

# 🎯 CRITICAL: Must be the VERY FIRST Streamlit command
st.set_page_config(
    page_title="Paint Visualizer",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- UTILITIES IMPORT ---
from utils.encoding import image_to_url_patch
from utils.sam_loader import get_sam_engine, ensure_model_exists, CHECKPOINT_PATH, MODEL_TYPE
from utils.state_manager import initialize_session_state, cb_apply_pending
from utils.ui_components import setup_styles, render_sidebar, render_visualizer_engine_v11, TOOL_MAPPING
from utils.image_processing import get_crop_params, magic_wand_selection
from config.constants import PerformanceConfig

# --- 1️⃣ SESSION INITIALIZATION (VERY TOP)
initialize_session_state()

# --- WARNING SHIELD: Titanium Silence v4 ---
st.components.v1.html("""
    <script>
        (function() {
            const silence = (w) => {
                try {
                    if (!w || !w.console || w.console.__isMuted) return;
                    ['warn', 'error', 'log'].forEach(m => {
                        const original = w.console[m];
                        if (!original) return;
                        w.console[m] = function(...args) {
                            try {
                                const msg = String(args[0] || "");
                                if (/Invalid color|theme\\.sidebar|widgetBackground|skeletonBackground|Unrecognized feature|ambient-light|battery|wake-lock|sandbox|document-domain|oversized-images|vr|fragment rerun/i.test(msg)) return;
                            } catch(e) {}
                            original.apply(this, args);
                        };
                    });
                    w.console.__isMuted = true;
                } catch(e) {}
            };
            const run = () => {
                silence(window);
                try { if (window.parent && window.parent !== window) silence(window.parent); } catch(e) {}
            };
            run();
            setInterval(run, 500);
        })();
    </script>
""", height=0)

def main():
    setup_styles()
    
    # Reset loop guard for the new app cycle
    st.session_state["loop_guarded"] = False
    ensure_model_exists()
    
    # Identify device for engine optimization
    device_str = "cpu"
    if torch.cuda.is_available(): device_str = "cuda"
    elif torch.backends.mps.is_available(): device_str = "mps"

    # Load SAM early
    sam = get_sam_engine(CHECKPOINT_PATH, MODEL_TYPE)

    # --- 2️⃣ CAPTURE BOX PARAM IMMEDIATELY (BEFORE SIDEBAR) ---
    q_params = st.query_params
    box_param = q_params.get("box", None)
    
    print(f"DEBUG: ALL PARAMS AT START: {dict(q_params)}")
    
    # --- 3️⃣ PROCESS BOX SEGMENTATION IMMEDIATELY ---
    if box_param and st.session_state.get("image") is not None:
        print(f"DEBUG: BOX PARAM DETECTED -> {box_param}")
        
        try:
            # Parse Timestamp (Suffix)
            if "," in box_param:
                timestamp = box_param.split(",")[-1] 
                # boxes_str is everything before the last comma
                # If there are multiple boxes separate by |, the timestamp is at the very end
                # Format: x,y,x,y|x,y,x,y,TIMESTAMP
                parts = box_param.split(",")
                if len(parts[-1]) > 9 and parts[-1].isdigit(): # Simple timestamp check
                     boxes_str = box_param[:-(len(parts[-1])+1)]
                else: 
                     boxes_str = box_param
            else:
                boxes_str = box_param
            
            # Replicate View/Scale Logic to map Canvas -> Image
            img = st.session_state["image"]
            h, w = img.shape[:2]
            display_width = 800
            
            zoom = st.session_state.get("zoom_level", 1.0)
            pan_x = st.session_state.get("pan_x", 0.5)
            pan_y = st.session_state.get("pan_y", 0.5)
            
            # USE CENTRALIZED LOGIC
            start_x, start_y, view_w, view_h = get_crop_params(w, h, zoom, pan_x, pan_y)
            
            scale_factor = display_width / view_w
            
            # Process Boxes
            accumulated_mask = None
            
            for b_token in boxes_str.split("|"):
                if not b_token.strip(): continue
                coords = list(map(float, b_token.split(",")))
                if len(coords) == 4:
                    cx1, cy1, cx2, cy2 = coords
                    x1 = int(cx1 / scale_factor) + start_x
                    y1 = int(cy1 / scale_factor) + start_y
                    x2 = int(cx2 / scale_factor) + start_x
                    y2 = int(cy2 / scale_factor) + start_y
                    
                    final_box = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
                    
                    if not getattr(sam, "is_image_set", False): sam.set_image(img)
                    mask = sam.generate_mask(box_coords=final_box, level=st.session_state.get("mask_level", 0), is_wall_only=st.session_state.get("is_wall_only", False))
                    
                    print(f"DEBUG: MASK GENERATED -> Box: {final_box}, Mask Sum: {np.sum(mask) if mask is not None else 'None'}")
                    
                    if mask is not None:
                        if accumulated_mask is None: accumulated_mask = mask
                        else: accumulated_mask = np.logical_or(accumulated_mask, mask)
            
            if accumulated_mask is not None:
                # Store in pending so cb_apply_pending can pick it up or apply directly?
                # The user said: "Apply paint using alpha blending... Store result... Set session_state.mask"
                # To adhere to the existing architecture where masks are accumulated in st.session_state["masks"], 
                # we will manually construct the mask entry and append it.
                
                operation = st.session_state.get("selection_op", "Add")
                
                if operation == "Subtract":
                     # ERASE Mode for AI Object (Box)
                     cleaned_any = False
                     if st.session_state["masks"]:
                         for layer in st.session_state["masks"]:
                             if layer.get("visible", True):
                                 # Apply subtraction: Keep existing True only if New is False
                                 # Resize to match execution context
                                 # Resize to match execution context
                                 target_mask = layer['mask']
                                 if sparse.issparse(target_mask):
                                     target_mask = target_mask.toarray()

                                 mask_to_subtract = accumulated_mask
                                 if target_mask.shape != mask_to_subtract.shape:
                                     mask_to_subtract = cv2.resize(mask_to_subtract.astype(np.uint8), (target_mask.shape[1], target_mask.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
                                 
                                 before_count = np.sum(target_mask)
                                 # Convert result back to boolean just in case
                                 result_mask = (target_mask > 0) & ~mask_to_subtract
                                 
                                 # Optional: Compress if it was sparse? For now keep it simple/robust
                                 layer['mask'] = result_mask
                                 if np.sum(layer['mask']) < before_count:
                                     cleaned_any = True
                         
                     if cleaned_any:
                         st.session_state["render_id"] += 1
                         st.toast("✅ Paint Erased!", icon="🧹")
                     else:
                         st.toast("⚠️ Nothing to erase!", icon="✨")

                else:
                    # ADD Mode
                    print("DEBUG: PAINT APPLIED (Adding to State)")
                    new_mask_entry = {
                        'mask': accumulated_mask,
                        'color': st.session_state.get("picked_color", "#8FBC8F"),
                        'visible': True,
                        'name': f"Layer {len(st.session_state['masks'])+1}",
                        'refinement': 0,
                        'softness': st.session_state.get("selection_softness", 0),
                        'brightness': 0.0, 'contrast': 1.0, 'saturation': 1.0, 'hue': 0.0, 
                        'opacity': st.session_state.get("selection_highlight_opacity", 1.0), 
                        'finish': st.session_state.get("selection_finish", 'Standard')
                    }
                    
                    # Directly append to avoid cb overhead or dependency on UI state
                    st.session_state["masks"].append(new_mask_entry)
                    st.session_state["render_id"] += 1
                    # st.toast("✅ Paint Applied!", icon="🎨")
            else:
                print("DEBUG: ⚠️ SAM returned None in Early Processor")
                st.toast("⚠️ No object detected.", icon="🤷‍♂️")
            
        except Exception as e:
            print(f"DEBUG: Early Processor Error: {e}")
            import traceback
            traceback.print_exc()
        
        # Clear param to prevent loop (Step 2 requirement)
        if "box" in st.query_params:
            st.query_params.pop("box", None)

    # --- 2c️⃣ CAPTURE & PROCESS TAP PARAM IMMEDIATELY ---
    tap_param = q_params.get("tap", None)
    if tap_param and st.session_state.get("image") is not None:
        print(f"DEBUG: Processing Mobile Tap at Top Level -> {tap_param}")
        try:
            parts = tap_param.split(",")
            if len(parts) >= 2:
                # 1. Parse & Scale Coords
                x, y = int(parts[0].strip()), int(parts[1].strip())
                img = st.session_state["image"]
                h, w = img.shape[:2]
                display_width = 800
                zoom = st.session_state.get("zoom_level", 1.0)
                pan_x = st.session_state.get("pan_x", 0.5)
                pan_y = st.session_state.get("pan_y", 0.5)
                start_x, start_y, view_w, view_h = get_crop_params(w, h, zoom, pan_x, pan_y)
                scale_factor = display_width / view_w
                
                real_x = int(x / scale_factor) + start_x
                real_y = int(y / scale_factor) + start_y

                # 2. SELECTION LOGIC BRANCH
                tool_mode = st.session_state.get("selection_tool", "")
                
                if "Magic Wand" in tool_mode:
                    print(f"DEBUG: Magic Wand Selection at ({real_x},{real_y})")
                    mask = magic_wand_selection(
                        st.session_state["image"], 
                        (real_x, real_y), 
                        tolerance=st.session_state.get("wand_tolerance", 25)
                    )
                else:
                    # Default: SAM Segmentation (AI Click)
                    if not getattr(sam, "is_image_set", False): sam.set_image(img)
                    mask = sam.generate_mask(
                        point_coords=[real_x, real_y], 
                        level=st.session_state.get("mask_level", 0), 
                        is_wall_only=st.session_state.get("is_wall_only", False)
                    )

                if mask is not None:
                    # 3. Apply Instantly
                    st.session_state["pending_selection"] = {'mask': mask, 'point': (real_x, real_y)}
                    
                    # Set Op for Magic Wand (default to sidebar op if not eraser)
                    # Use current radio selection
                    st.session_state["selection_op"] = st.session_state.get("selection_op", "Add")
                    
                    # ⚡ INSTANT APPLY Logic
                    # Magic Wand and AI Click are now 'one-click' tools
                    cb_apply_pending(increment_canvas=False, silent=True)
                    st.session_state["render_id"] += 1
                    print(f"DEBUG: Top-Level Tap Success at ({real_x},{real_y}) using mode: {tool_mode}")
        
        except Exception as tap_err:
            print(f"DEBUG: Top-Level Tap Error: {tap_err}")
        
        # 4. Clean up & Rerun to commit
        if "tap" in st.query_params: st.query_params.pop("tap", None)
        st.rerun()

    # --- 2b️⃣ CAPTURE POLY PARAM IMMEDIATELY ---
    poly_param = q_params.get("poly_pts", None)

    # --- 3b️⃣ PROCESS POLYGON SEGMENTATION IMMEDIATELY ---
    if poly_param and st.session_state.get("image") is not None:
        print(f"DEBUG: POLY PARAM DETECTED -> {poly_param}")
        try:
            # Parse Timestamp
            if "," in poly_param:
                parts = poly_param.split(",")
                # Format: x,y;x,y;...TIMESTAMP
                # The timestamp is the LAST element after the last comma or semicolon?
                # JS sends: `ptsStr + ',' + Date.now()` where ptsStr is `x,y;x,y`
                # So splitting by `,` might mix coords. 
                # Better: look for last comma
                last_comma_idx = poly_param.rfind(",")
                if last_comma_idx != -1:
                    timestamp = poly_param[last_comma_idx+1:]
                    if len(timestamp) > 9 and timestamp.isdigit():
                        poly_str = poly_param[:last_comma_idx]
                    else:
                        poly_str = poly_param
                else:
                    poly_str = poly_param
            else:
                poly_str = poly_param

            # Replicate View/Scale Logic
            img = st.session_state["image"]
            h, w = img.shape[:2]
            display_width = 800
            
            zoom = st.session_state.get("zoom_level", 1.0)
            pan_x = st.session_state.get("pan_x", 0.5)
            pan_y = st.session_state.get("pan_y", 0.5)
            start_x, start_y, view_w, view_h = get_crop_params(w, h, zoom, pan_x, pan_y)
            scale_factor = display_width / view_w
            
            # Parse Points
            pts = []
            for p_pair in poly_str.split(";"):
                if "," in p_pair:
                    px, py = map(float, p_pair.split(","))
                    pts.append([int(px / scale_factor) + start_x, int(py / scale_factor) + start_y])
            
            if len(pts) > 2:
                # Determine Mode: Manual (Freehand) vs AI (Polygon)
                # "Lasso (Freehand)" -> Manual Fill (User expectation: "Draw the place to apply color")
                # "Polygonal Lasso" -> AI Assisted (SAM Box Prompts)
                
                current_tool = st.session_state.get("selection_tool", "")
                is_freehand = "Lasso (Freehand)" in current_tool
                
                # Check for explicit "Fill Selection" toggle (overrides default)
                # Default for Freehand is True (Manual), Default for Polygon is False (AI)
                force_manual = st.session_state.get("fill_selection", is_freehand)
                
                if is_freehand or force_manual:
                     print("DEBUG: processing as MANUAL MASK (Freehand)")
                     # Create blank mask
                     mask_shape = (h, w)
                     manual_mask = np.zeros(mask_shape, dtype=np.uint8)
                     
                     # Fill Polygon
                     pts_arr = np.array(pts, dtype=np.int32)
                     cv2.fillPoly(manual_mask, [pts_arr], 1)
                     
                     mask = manual_mask.astype(bool)
                else:
                    # STRATEGY: Use Bounding Box of Polygon as SAM Prompt
                    print("DEBUG: processing as AI MASK (SAM Box from Poly)")
                    pts_arr = np.array(pts)
                    x_min, y_min = np.min(pts_arr, axis=0)
                    x_max, y_max = np.max(pts_arr, axis=0)
                    box = [x_min, y_min, x_max, y_max]
                    
                    if not getattr(sam, "is_image_set", False): sam.set_image(img)
                    mask = sam.generate_mask(box_coords=box, level=st.session_state.get("mask_level", 0), is_wall_only=st.session_state.get("is_wall_only", False))

                if mask is not None:
                     # Check Operation: Add vs Subtract
                     operation = st.session_state.get("selection_op", "Add")
                     
                     if operation == "Subtract":
                         # ERASE Mode: Remove this mask from ALL existing active layers
                         cleaned_any = False
                         for layer in st.session_state["masks"]:
                             if layer.get("visible", True):
                                 # Apply subtraction: Keep existing True only if New is False
                                 target_mask = layer['mask']
                                 if sparse.issparse(target_mask):
                                     target_mask = target_mask.toarray()
                                     
                                 layer['mask'] = (target_mask > 0) & ~mask
                                 cleaned_any = True
                         
                         if cleaned_any:
                             st.session_state["render_id"] += 1
                             # st.toast("✅ Area Erased!", icon="🧹")
                     else:
                         # ADD Mode: Append new layer
                         new_mask_entry = {
                            'mask': mask,
                            'color': st.session_state.get("picked_color", "#8FBC8F"),
                            'visible': True,
                            'name': f"Layer {len(st.session_state['masks'])+1}",
                            'refinement': 0,
                            'softness': st.session_state.get("selection_softness", 0),
                            'brightness': 0.0, 'contrast': 1.0, 'saturation': 1.0, 'hue': 0.0, 
                            'opacity': st.session_state.get("selection_highlight_opacity", 1.0), 
                            'finish': st.session_state.get("selection_finish", 'Standard')
                        }
                         st.session_state["masks"].append(new_mask_entry)
                         st.session_state["render_id"] += 1
                         # st.toast("✅ Paint Applied!", icon="🎨")

                else:
                    # st.toast("⚠️ No object found.", icon="🤷‍♂️")
                    pass
            
        except Exception as e:
            print(f"DEBUG: Poly Processor Error: {e}")
            import traceback
            traceback.print_exc()

        if "poly_pts" in st.query_params:
             st.query_params.pop("poly_pts", None)
        st.rerun()
            
    # --- 4️⃣ RENDER IMAGE ---
    if st.session_state.get("image") is not None:
        render_visualizer_engine_v11(800)
    else:
        # Landing Page
        empty_top = st.empty()
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.markdown("""
            <div class="landing-container">
                <div class="landing-header">
                    <h1>Welcome to Color Visualizer</h1>
                </div>
                <div class="landing-sub">
                    <p>Upload a photo of your room to start experimenting with colors.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.info("👈 Use the sidebar to upload an image.")

    # --- 5️⃣ RENDER SIDEBAR LAST ---
    print("DEBUG: SIDEBAR RENDER")
    render_sidebar(sam, device_str)

    # --- 🤖 HIDDEN TECHNICAL BRIDGE (Bottom of script) ---
    st.markdown('<div id="global-sync-anchor"></div>', unsafe_allow_html=True)
    st.button("GLOBAL SYNC", key="global_sync_btn", help="Hidden sync for JS", type="secondary")
    st.markdown('<div class="global-sync-marker" style="display:none;" data-sync-id="global_sync"></div>', unsafe_allow_html=True)

    
if __name__ == "__main__":
    main()

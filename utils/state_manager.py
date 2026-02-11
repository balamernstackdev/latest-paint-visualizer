import streamlit as st
import numpy as np
from utils.performance import cleanup_session_caches, should_trigger_cleanup

def initialize_session_state():
    """Initialize all session state variables with multi-layer safety."""
    defaults = {
        "image": None,          # 640px preview image
        "image_original": None, # Full resolution original
        "file_name": None,
        "masks": [],
        "masks_redo": [],
        "selection_op": "Add",
        "is_wall_only": True,
        "selection_softness": 0,
        "selection_highlight_opacity": 0.5,
        "zoom_level": 1.0,
        "pan_x": 0.5,
        "pan_y": 0.5,
        "last_click_global": None,
        "mask_level": 0,    # 0, 1, or 2 for granularity
        "selection_tool": "👆 AI Click (Point)",
        "ai_drag_sub_tool": "🆕 Draw New",
        "picked_color": "#8FBC8F",
        "pending_selection": None,
        "pending_boxes": [],
        "render_id": 0,
        "canvas_id": 0,
        "uploader_id": 0,
        "sidebar_p_open": False,
        "last_export": None,
        "selected_layer_idx": None,
        "loop_guarded": False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def cb_apply_pending():
    if st.session_state.get("pending_selection") is not None:
        new_mask = st.session_state["pending_selection"].copy()
        new_mask.update({
            'color': st.session_state["picked_color"],
            'visible': True,
            'name': f"Layer {len(st.session_state['masks'])+1}",
            'refinement': 0, # Expansion/Contraction (-10 to 10)
            'softness': st.session_state.get("selection_softness", 0),
            'brightness': 0.0, 'contrast': 1.0, 'saturation': 1.0, 'hue': 0.0, 
            'opacity': st.session_state.get("selection_highlight_opacity", 1.0), 
            'finish': st.session_state.get("selection_finish", 'Standard')
        })
        
        # Handle Subtraction Logic
        if st.session_state.get("selection_op") == "Subtract" and st.session_state["masks"]:
            # Target the selected layer if valid, otherwise the last one
            target_idx = st.session_state.get("selected_layer_idx")
            if target_idx is None or not (0 <= target_idx < len(st.session_state["masks"])):
                target_idx = len(st.session_state["masks"]) - 1
            
            # Perform logical subtraction
            # Ensure masks are the same shape before bitwise operation
            target_mask = st.session_state["masks"][target_idx]['mask']
            new_selection_mask = new_mask['mask']
            
            # Simple check/resize if needed (should already be same shape)
            if target_mask.shape != new_selection_mask.shape:
                new_selection_mask = cv2.resize(new_selection_mask.astype(np.uint8), (target_mask.shape[1], target_mask.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            
            st.session_state["masks"][target_idx]['mask'] = target_mask & ~new_selection_mask
        else:
            st.session_state["masks"].append(new_mask)
            
        st.session_state["masks_redo"] = [] # Clear redo stack on new action
        st.session_state["pending_selection"] = None
        st.session_state["pending_boxes"] = []
        st.session_state["render_id"] += 1
        st.session_state["canvas_id"] = st.session_state.get("canvas_id", 0) + 1  # Re-enabled to clear box on apply
        st.session_state["canvas_raw"] = {} # Force clear cached objects

def cb_cancel_pending():
    st.session_state["pending_selection"] = None
    st.session_state["pending_boxes"] = []
    st.session_state["render_id"] += 1
    st.session_state["canvas_id"] = st.session_state.get("canvas_id", 0) + 1
    st.session_state["canvas_raw"] = {} # Force clear cached objects

def cb_undo():
    """Undo last paint layer with automatic memory cleanup."""
    if st.session_state["masks"]:
        last_mask = st.session_state["masks"].pop()
        st.session_state["masks_redo"].append(last_mask)
        st.session_state["render_id"] += 1
        st.session_state["canvas_id"] = st.session_state.get("canvas_id", 0) + 1
        
        # Check if cleanup needed after undo
        if should_trigger_cleanup():
            cleanup_session_caches(aggressive=False)

def cb_redo():
    """Redo the last undone paint layer."""
    if st.session_state.get("masks_redo"):
        mask = st.session_state["masks_redo"].pop()
        st.session_state["masks"].append(mask)
        st.session_state["render_id"] += 1
        st.session_state["canvas_id"] = st.session_state.get("canvas_id", 0) + 1

def cb_clear_all():
    """Clear all paint layers and perform memory cleanup."""
    st.session_state["masks"] = []
    st.session_state["masks_redo"] = []
    
    # Aggressive cleanup when clearing all
    cleanup_session_caches(aggressive=True)
    st.session_state["render_id"] += 1
    st.session_state["canvas_id"] = st.session_state.get("canvas_id", 0) + 1

def cb_delete_layer(idx):
    if st.session_state.get("masks") and 0 <= idx < len(st.session_state["masks"]):
        st.session_state["masks"].pop(idx)
        st.session_state["selected_layer_idx"] = None
        st.session_state["render_id"] += 1
        st.session_state["canvas_id"] = st.session_state.get("canvas_id", 0) + 1

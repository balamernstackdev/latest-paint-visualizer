import cv2
import numpy as np
import streamlit as st
from PIL import Image
from core.colorizer import ColorTransferEngine
from app_config.constants import UIConfig

def get_crop_params(image_width, image_height, zoom_level, pan_x, pan_y):
    """
    Calculate crop coordinates based on zoom and normalized pan (0.0 to 1.0).
    """
    if zoom_level <= 1.0:
        return 0, 0, image_width, image_height

    # visible width/height
    view_w = int(image_width / zoom_level)
    view_h = int(image_height / zoom_level)

    # ensure pan stays within bounds
    # pan_x=0 -> left edge, pan_x=1 -> right edge
    max_x = image_width - view_w
    max_y = image_height - view_h
    
    start_x = int(max_x * pan_x)
    start_y = int(max_y * pan_y)
    
    return start_x, start_y, view_w, view_h

def composite_image(original_rgb, masks_data):
    """
    Apply all color layers to user image using optimized multi-layer blending.
    PERFORMANCE: Uses incremental background caching to speed up tuning.
    """
    visible_masks = [m for m in masks_data if m.get('visible', True)]
    
    # 🎯 ULTRA-STABLE FULL COMPOSITE 🎯
    # Optimization: Pass original mask objects to Core to enable identity-based caching.
    # The Core engine handles sparse decompression and refinement internally.
    result = ColorTransferEngine.composite_multiple_layers(original_rgb, visible_masks)

    # --- Selection Highlight (Fast Blending) ---
    try:
        # ALWAYS apply highlight at the end, regardless of mask count.
        pending = st.session_state.get("pending_selection")
        if pending is not None:
            mask = pending['mask']
            # Decompress Pending if sparse (rare but possible)
            if sparse.issparse(mask): mask = mask.toarray()
            pc = st.session_state.get("picked_color", "#3B82F6")
            try:
                c_rgb = tuple(int(pc.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                highlight = np.array([c_rgb[0], c_rgb[1], c_rgb[2]], dtype=np.uint8) 
            except:
                highlight = np.array(UIConfig.DEFAULT_HIGHLIGHT_COLOR, dtype=np.uint8) 
            
            # --- FASTER BLENDING (No float32 conversion) ---
            m_idx = mask
            if mask.shape[:2] != result.shape[:2]:
                m_idx = cv2.resize(mask.astype(np.uint8), (result.shape[1], result.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            else:
                m_idx = mask > 0

            if m_idx.any():
                # Create a copy only if there's a highlight, to avoid mutating cached backgrounds
                result = result.copy()
                # Blend with 50% opacity using integer math: (A + B) / 2
                # We use uint16 to prevent overflow during addition
                result[m_idx] = ((result[m_idx].astype(np.uint16) + highlight.astype(np.uint16)) // 2).astype(np.uint8)
    except Exception as e:
        # Fallback to prevent crash if highlighting fails
        st.error(f"Highlight Error: {e}")
        pass

    return result

def process_lasso_path(scaled_path, w, h, thickness=6, fill=False):
    """
    Renders the SVG path from st_canvas into a binary mask.
    Handles 'M', 'L', 'Q' commands. 
    If fill=True, ensures the shape is closed and filled (Manual Fill Mode).
    If fill=False, draws a thick polyline (Brush/AI Mode).
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    points = []
    
    try:
        for cmd in scaled_path:
            if not cmd: continue
            c_type = cmd[0]
            if c_type in ['M', 'L']:
                points.append([int(cmd[1]), int(cmd[2])])
            elif c_type == 'Q':
                points.append([int(cmd[3]), int(cmd[4])])
                
        if len(points) > 1:
            if fill and len(points) > 2:
                # 🎨 Manual Fill Mode: Solid enclosed shape
                cv2.fillPoly(mask, [np.array(points, np.int32)], 255)
            else:
                # 🧠 Brush/AI Mode: Thick polyline
                cv2.polylines(mask, [np.array(points, np.int32)], isClosed=False, color=255, thickness=max(1, thickness))
            
    except Exception as e:
        print(f"Lasso Path processing error: {e}")
        
    return mask > 0


def magic_wand_selection(image, seed_point, tolerance=10):
    if image is None: return None
    h, w = image.shape[:2]
    
    # Create mask compatible with floodFill (h+2, w+2)
    mask = np.zeros((h+2, w+2), np.uint8)
    
    flags = 4 | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY
    
    lo_diff = (tolerance, tolerance, tolerance)
    up_diff = (tolerance, tolerance, tolerance)
    
    try:
        cv2.floodFill(image, mask, seed_point, (255, 255, 255), lo_diff, up_diff, flags)
        return mask[1:-1, 1:-1] > 0
    except Exception as e:
        print(f'Magic Wand Error: {e}')
        return None


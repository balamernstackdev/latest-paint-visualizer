import numpy as np
import torch
import cv2
import logging
from mobile_sam import sam_model_registry, SamPredictor
from config.constants import SegmentationConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SegmentationEngine:
    def __init__(self, checkpoint_path=None, model_type="vit_b", device=None, model_instance=None):
        """
        Initialize the SAM model.
        Args:
            checkpoint_path: Path to weights (if loading new).
            model_type: SAM architecture type.
            device: 'cuda' or 'cpu'.
            model_instance: Pre-loaded sam_model_registry instance (optional).
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        if model_instance is not None:
             self.sam = model_instance
        elif checkpoint_path:
             logger.info(f"Loading SAM model ({model_type}) on {self.device}...")
             self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
             self.sam.to(device=self.device)
        else:
             raise ValueError("Either checkpoint_path or model_instance must be provided.")

        self.predictor = SamPredictor(self.sam)
        self.is_image_set = False

    def set_image(self, image_rgb):
        """
        Process the image and compute embeddings.
        Args:
            image_rgb: NumPy array (H, W, 3) in RGB format.
        """
        # OPTIMIZATION: Check if image is already set to avoid expensive re-encoding
        if self.is_image_set and hasattr(self, 'image_rgb') and self.image_rgb is not None:
            if image_rgb.shape == self.image_rgb.shape and np.array_equal(image_rgb, self.image_rgb):
                logger.info("Image already set. Skipping embedding computation.")
                print("DEBUG: SAM Engine - Image already set.")
                return

        logger.info("Computing image embeddings...")
        print(f"DEBUG: SAM Engine {id(self)} - Computing embeddings...")
        self.predictor.set_image(image_rgb)
        self.is_image_set = True
        self.image_rgb = image_rgb
        
        # --- PRE-COMPUTE FEATURES FOR FASTER CLICKS ---
        # 1. Grayscale
        self.image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        self.image_u16 = image_rgb.astype(np.uint16)
        
        # 2. Gaussian Blur (for small objects/edge detection)
        k_size = SegmentationConfig.GAUSSIAN_KERNEL_SIZE
        self.image_blurred = cv2.GaussianBlur(self.image_gray, k_size, 0)
        
        # 3. Lapalcian Edges (base)
        edges = cv2.Laplacian(self.image_blurred, cv2.CV_16S, ksize=3)
        self.image_edges_abs = cv2.convertScaleAbs(edges)
        
        print(f"DEBUG: SAM Engine {id(self)} - is_image_set = True ✅ (Features Pre-computed)")
        logger.info("Embeddings and features computed.")

    def generate_mask(self, point_coords=None, point_labels=None, box_coords=None, level=None, is_wall_only=False, cleanup=True):
        print(f"DEBUG: Entering generate_mask v4.2 (Edge Map Renamed)")
        is_small_object = False # Initialize to prevent UnboundLocalError in Box Mode
        if self.predictor is None:
            return None
        """
        Generate a mask for a given point or box.
        Args:
            point_coords: List of [x, y] or NumPy array.
            point_labels: List of labels (1 for foreground, 0 for background).
            box_coords: [x1, y1, x2, y2]
            level: int (0, 1, 2) or None. 
                   0=Fine Details, 1=Sub-segment, 2=Whole Object. 
                   If None, auto-selects highest score.
            is_wall_only: bool. If True, uses stricter wall-specific thresholds.
            cleanup: bool. If True, removes disconnected components to prevent leaks.
        """
        if not self.is_image_set:
            raise RuntimeError("Image not set. Call set_image() first.")

        # Prepare inputs
        sam_point_coords = None
        sam_point_labels = None
        sam_box = None

        if point_coords is not None:
            # Check input structure
            # Case 1: Single point [x, y] -> wrap to [[x, y]]
            # Case 2: List of points [[x, y], ...] -> use as is
            
            arr = np.array(point_coords)
            if arr.ndim == 1:
                sam_point_coords = np.array([point_coords])
            else:
                 sam_point_coords = arr
            
            if point_labels is None:
                # We have N points, so we need N labels
                sam_point_labels = np.array([1] * len(sam_point_coords))
            else:
                sam_point_labels = np.array(point_labels)
        
        if box_coords is not None:
            sam_box = np.array(box_coords)

        with torch.inference_mode():
            masks, scores, logits = self.predictor.predict(
                point_coords=sam_point_coords,
                point_labels=sam_point_labels,
                box=sam_box,
                multimask_output=True # Generate multiple masks and choose best
            )

        # Handle batch dimension if present (MobileSAM/TinySAM might return (1, 3, H, W))
        if len(masks.shape) == 4:
            masks = masks[0]
        if len(scores.shape) == 2:
            scores = scores[0]

        # Select best mask
        if level is not None and 0 <= level < 3:
            # User forced a specific level
            if level == 1:
                # For Small Objects, we usually want Index 0 (most granular)
                # But if Index 0 is tiny (e.g. noise), fallback to Index 1 (Sub-segment)
                area0 = np.sum(masks[0])
                area1 = np.sum(masks[1])
                if area0 < SegmentationConfig.MIN_MASK_AREA_PIXELS * 10 and area1 > area0 * 2: # Heuristic for "too small"
                    best_mask = masks[1]
                else:
                    best_mask = masks[0]
            elif level == 0:
                # --- INTELLIGENT STANDARD WALLS MODE ---
                if box_coords is not None:
                    # Box Mode: Always want the largest/whole object (Index 2)
                    best_mask = masks[2] if scores[2] > SegmentationConfig.SAM_MIN_SCORE else masks[1]
                else:
                    # Point Click Mode: Analyze ALL 3 masks and pick best one for doors/windows
                    h, w = masks[0].shape
                    image_area = h * w
                    
                    # Analyze all 3 masks
                    best_mask = masks[0]  # Default
                    best_door_score = -1
                    
                    for idx in range(3):
                        mask_area = np.sum(masks[idx])
                        if mask_area == 0:
                            continue
                            
                        mask_coords = np.argwhere(masks[idx] > 0)
                        y_coords, x_coords = mask_coords[:, 0], mask_coords[:, 1]
                        mask_height = np.max(y_coords) - np.min(y_coords) + 1
                        mask_width = np.max(x_coords) - np.min(x_coords) + 1
                        
                        # Calculate characteristics
                        area_ratio = mask_area / image_area
                        aspect_ratio = mask_height / max(mask_width, 1)
                        width_ratio = mask_width / w
                        
                        # Calculate "door/window score" - how well does this match a door/window?
                        door_score = 0
                        
                        # Small area is good (max 5 points)
                        if area_ratio < 0.05:  # Very small
                            door_score += 5
                        elif area_ratio < 0.10:  # Small
                            door_score += 3
                        elif area_ratio < 0.15:  # Medium-small
                            door_score += 1
                        
                        # Tall aspect ratio is good (max 5 points)
                        if aspect_ratio > 2.0:  # Very tall
                            door_score += 5
                        elif aspect_ratio > 1.5:  # Tall
                            door_score += 3
                        elif aspect_ratio > 1.3:  # Medium-tall
                            door_score += 1
                        
                        # Narrow width is good (max 3 points)
                        if width_ratio < 0.15:  # Very narrow
                            door_score += 3
                        elif width_ratio < 0.25:  # Narrow
                            door_score += 2
                        elif width_ratio < 0.30:  # Medium-narrow
                            door_score += 1
                        
                        print(f"\nMask {idx}: area={area_ratio*100:.1f}%, aspect={aspect_ratio:.2f}, width={width_ratio*100:.1f}%, door_score={door_score}")
                        
                        # Track best door-like mask
                        if door_score > best_door_score:
                            best_door_score = door_score
                            best_mask = masks[idx]
                    
                    print(f"Selected mask with door_score={best_door_score}")
                    
                    # SAFETY MARGIN: If we detected a door/window (score >= 5), apply erosion
                    # to create a margin that prevents bleeding onto adjacent walls
                    if best_door_score >= 5:
                        # Calculate erosion strength based on mask size
                        mask_area = np.sum(best_mask)
                        area_ratio = mask_area / image_area
                        
                        # Smaller objects need more aggressive erosion
                        if area_ratio < 0.05:  # Very small (< 5%)
                            kernel_size = 3    # Reduced from 4
                            iterations = 1     # Reduced from 2 to preserve lattice details
                        elif area_ratio < 0.10:  # (Small < 10%)
                            kernel_size = 3
                            iterations = 2
                        elif area_ratio < 0.15:  # Medium small (< 15%)
                            kernel_size = 3
                            iterations = 1
                        else:  # Large objects need less erosion
                            kernel_size = 3
                            iterations = 1
                        
                        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                        best_mask = cv2.erode(best_mask.astype(np.uint8), erode_kernel, iterations=iterations).astype(bool)
                        
                        print(f"Applied erosion: kernel={kernel_size}x{kernel_size}, iterations={iterations}")
                    
                    # If no mask looks like a door (score < 3), use default behavior
                    if best_door_score < 3:
                        # Doesn't look like a door, use mask 0 for walls
                        area0 = np.sum(masks[0])
                        if area0 < SegmentationConfig.MIN_MASK_AREA_PIXELS:
                            best_idx = np.argmax(scores)
                            best_mask = masks[best_idx]
                        else:
                            best_mask = masks[0]
            else:
                best_mask = masks[level]
        else:
            # Heuristic: Favor 'Fine Detail' (Index 0) for Point Clicks
            # Previously we favored Index 1, which caused "wrong object" selection for thin walls.
            if box_coords is not None:
                best_mask = masks[2] if scores[2] > SegmentationConfig.SAM_MIN_SCORE else masks[1]
            else:
                # Point Mode: We want the EXACT part user clicked.
                # Index 0 is usually the most granular (e.g., just the side strip).
                # Index 1 often merges neighbors (e.g., side strip + main wall).
                
                # Only skip Index 0 if it's basically noise
                area0 = np.sum(masks[0])
                if area0 < SegmentationConfig.MIN_MASK_AREA_PIXELS: 
                     best_idx = np.argmax(scores)
                     best_mask = masks[best_idx]
                else:
                     best_mask = masks[0]
        
        if cleanup:
            h, w = best_mask.shape
            mask_uint8 = (best_mask * 255).astype(np.uint8)
            
            # Use a reference point for connectivity filtering
            ref_x, ref_y = None, None
            if point_coords is not None and len(point_coords) > 0:
                pos_indices = np.where(sam_point_labels == 1)[0]
                if len(pos_indices) > 0:
                    idx = pos_indices[-1]
                    ref_x, ref_y = int(sam_point_coords[idx][0]), int(sam_point_coords[idx][1])
            elif box_coords is not None:
                ref_x = int((box_coords[0] + box_coords[2]) / 2)
                ref_y = int((box_coords[1] + box_coords[3]) / 2)

            if ref_x is not None:
                is_small_object = False  # Initialize safely for Box mode
                # --- ADAPTIVE FILTERING ---
                # Calculate reference color (from click point or box center)
                y1, y2 = max(0, ref_y-1), min(h, ref_y+2)
                x1, x2 = max(0, ref_x-1), min(w, ref_x+2)
                seed_patch = self.image_rgb[y1:y2, x1:x2]
                # FIX: Use MEDIAN instead of MEAN for seed color.
                # If user clicks on an edge, MEAN blends the two colors (e.g. Brown + Cream = Light Brown).
                # MEDIAN picks the dominant color, ignoring the outlier neighbor.
                seed_color = np.median(seed_patch, axis=(0, 1))
                
                img_u16 = self.image_u16
                intensity_raw = np.mean(img_u16, axis=2)
                seed_intensity = np.mean(seed_color)
                intensity_diff = intensity_raw - seed_intensity

                if box_coords is not None:
                     # BOX MODE: Enhanced Logic (Mask-Based Seed + Color Diff)
                     # Instead of using the box center (which might be a window), use the median 
                     # color of the PREDICTED object itself.
                     mask_indices = np.where(mask_uint8 > 0)
                     if len(mask_indices[0]) > 0:
                         seed_patch = self.image_rgb[mask_indices]
                         # Take representative color from object (Median is robust to shadows/highlights)
                         seed_color = np.median(seed_patch, axis=0) 
                     
                     # Use COLOR Difference (RGB) instead of Intensity
                     diff_r = np.abs(img_u16[:,:,0] - seed_color[0])
                     diff_g = np.abs(img_u16[:,:,1] - seed_color[1])
                     diff_b = np.abs(img_u16[:,:,2] - seed_color[2])
                     color_diff = np.maximum(np.maximum(diff_r, diff_g), diff_b)
                     
                     # Box Mode Tolerance: Relaxed to include entire object, but stricter than before if color deviates
                     valid_mask = (color_diff < SegmentationConfig.COLOR_DIFF_BOX_MODE).astype(np.uint8)
                     
                     # Enable Edge Detection to snap to lines
                     # Use pre-computed edges
                     edge_barrier = 0
                     _, edge_barrier = cv2.threshold(self.image_edges_abs, SegmentationConfig.EDGE_THRESHOLD_BOX_MODE, 255, cv2.THRESH_BINARY_INV)
                     edge_barrier = (edge_barrier / 255).astype(np.uint8)
                     
                     mask_refined = (mask_uint8 & valid_mask & edge_barrier)
                     
                     # If validation killed the mask (e.g. wrong seed), fallback to original SAM mask
                     if np.sum(mask_refined) < (np.sum(mask_uint8) * 0.1):
                         mask_refined = mask_uint8 
                else:
                    # Point Click Mode Logic
                    # Check if the SAM mask implies a very small/thin object
                    h, w = mask_uint8.shape
                    mask_area_px = np.sum(mask_uint8)
                    
                    # Increased threshold from 1% to 3% to capture vertical wall strips/pillars
                    # These "medium" objects also need the edge-barrier disabled to paint fully.
                    is_small_object = mask_area_px < (h * w * SegmentationConfig.SMALL_OBJECT_THRESHOLD)
                    
                    std_dev = np.std(seed_color)
                    
                    if level == 0:
                        if is_small_object:
                            # --- SMALL WALL/THIN OBJECT LOGIC ---
                            # FIX: Use COLOR Difference (RGB) instead of Intensity (Brightness).
                            # Previous issue: Brown Arch and Green Wall had same brightness, so they merged.
                            # Now: We check color distance. Brown (Reddish) vs Green is a huge difference.
                            
                            # standard intensity was: intensity_diff = intensity_raw - seed_intensity
                            # New Color Diff: MAX(Abs(Image - Seed)) across R,G,B channels
                            # This is stricter than Mean. If ANY channel deviates by > 40, we stop.
                            diff_r = np.abs(img_u16[:,:,0] - seed_color[0])
                            diff_g = np.abs(img_u16[:,:,1] - seed_color[1])
                            diff_b = np.abs(img_u16[:,:,2] - seed_color[2])
                            # Use MAX difference to catch tint shifts (e.g. Red channel spiking)
                            color_diff = np.maximum(np.maximum(diff_r, diff_g), diff_b)
                                                        # Tolerance: Balanced for precision vs coverage.
                            tol = SegmentationConfig.COLOR_DIFF_WALL_MODE if is_wall_only else SegmentationConfig.COLOR_DIFF_SMALL_OBJECT
                            valid_mask = (color_diff < tol).astype(np.uint8) 
                            
                            # 2. Edge Barrier: Catches strong edges while allowing texture detail.
                            # Use pre-computed
                            edge_thresh = SegmentationConfig.EDGE_THRESHOLD_WALL_MODE if is_wall_only else SegmentationConfig.EDGE_THRESHOLD_SMALL_OBJECT
                            _, edge_barrier = cv2.threshold(self.image_edges_abs, edge_thresh, 255, cv2.THRESH_BINARY_INV)
                            edge_barrier = (edge_barrier / 255).astype(np.uint8)
                            
                            mask_refined = (mask_uint8 & valid_mask & edge_barrier)
                        else:
                            # --- DISTANCE-DECAYING TOLERANCE (Standard Walls) ---
                            # FIX: Use COLOR Difference here too!
                            # Objects > 3% size (like large pillars) were still using Intensity, causing leaks.
                            
                            # Calculate Color Diff (Max Absolute Error)
                            diff_r = np.abs(img_u16[:,:,0] - seed_color[0])
                            diff_g = np.abs(img_u16[:,:,1] - seed_color[1])
                            diff_b = np.abs(img_u16[:,:,2] - seed_color[2])
                            color_diff = np.maximum(np.maximum(diff_r, diff_g), diff_b)

                            Y, X = np.ogrid[:h, :w]
                            dist_from_click = np.sqrt((X - ref_x)**2 + (Y - ref_y)**2)
                            decay_factor = np.clip(1.0 - (dist_from_click / SegmentationConfig.DECAY_DISTANCE_MAX), SegmentationConfig.DECAY_FACTOR_MIN, 1.0)
                            
                            # Base tolerance with distance decay
                            base_tol = SegmentationConfig.COLOR_DIFF_WALL_MODE if is_wall_only else SegmentationConfig.COLOR_DIFF_STANDARD_WALL
                            tol = base_tol * decay_factor 
                            valid_mask = (color_diff < tol).astype(np.uint8)
                            
                            # Edge detection: allows paint to flow over soft architectural edges but stops at strong object borders
                            edge_thresh = SegmentationConfig.EDGE_THRESHOLD_WALL_MODE if is_wall_only else SegmentationConfig.EDGE_THRESHOLD_STANDARD_WALL
                            _, edge_barrier = cv2.threshold(self.image_edges_abs, edge_thresh, 255, cv2.THRESH_BINARY_INV)
                            edge_barrier = (edge_barrier / 255).astype(np.uint8)
                            
                            mask_refined = (mask_uint8 & valid_mask & edge_barrier)
                            
                    elif level == 1:
                        valid_mask = (np.abs(intensity_diff) < SegmentationConfig.INTENSITY_DIFF_LEVEL_1).astype(np.uint8) 
                        edge_barrier = np.ones((h, w), dtype=np.uint8) # No edge barrier for Level 1
                        mask_refined = (mask_uint8 & valid_mask)
                    else:
                        valid_mask = (np.abs(intensity_diff) < SegmentationConfig.INTENSITY_DIFF_LEVEL_2).astype(np.uint8)
                        edge_barrier = np.ones((h, w), dtype=np.uint8)
                        mask_refined = (mask_uint8 & valid_mask)

                    # Ensure click point is always preserved
                    if ref_x is not None and ref_y is not None:
                         cv2.circle(mask_refined, (ref_x, ref_y), SegmentationConfig.CLICK_PRESERVE_RADIUS, 1, -1) 

                # --- SELECTIVE HOLE FILLING ---
                if level == 0 or box_coords is not None:
                    # Skip erosion/closing for Small Objects to prevent deleting thin lines
                    # CRITICAL: Also skip for level 0 to preserve eroded safety margins for doors/windows
                    if not (level == 0 and is_small_object):
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, SegmentationConfig.MORPH_KERNEL_SIZE)
                        mask_refined = cv2.morphologyEx(mask_refined, cv2.MORPH_CLOSE, kernel)
                    
                    # Selective internal hole filling using HIERARCHY
                    # RETR_CCOMP returns hierarchy [Next, Previous, First_Child, Parent]
                    # If a contour has a parent (hierarchy[i][3] != -1), it's an internal hole.
                    cnts, hierarchy = cv2.findContours(mask_refined, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                    out_mask = np.copy(mask_refined)

                    if hierarchy is not None:
                        hierarchy = hierarchy[0] # flatten
                        for i, c in enumerate(cnts):
                            parent_idx = hierarchy[i][3]
                            if parent_idx != -1: # It is a hole (has a parent)
                                area = cv2.contourArea(c)
                                # Fill if it's small noise OR if it's a medium gap in a wall (up to 5% of image)
                                # Windows are usually larger than 5%
                                if area < (h * w * 0.05): 
                                    cv2.drawContours(out_mask, [c], -1, 1, thickness=-1)
                    mask_refined = out_mask
                elif level == 1:
                    # Level 1 (Small Objects): No hole filling or closing to preserve lattice/mesh details
                    pass

                if np.sum(mask_refined) > SegmentationConfig.MIN_MASK_AREA_PIXELS:
                    mask_uint8 = mask_refined
            
            # Connectivity filtering and Object Recovery
            if ref_x is not None:
                num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=SegmentationConfig.CONNECTED_COMPONENTS_CONNECTIVITY)
                if num_labels > 1:
                    if box_coords is not None:
                        # --- BOX MODE: MULTI-COMPONENT RECOVERY ---
                        # In Photoshop style, if you box an object with holes (like a perforated wall),
                        # we want to keep ALL pieces of that object that are inside the box.
                        recovered_mask = np.zeros_like(best_mask)
                        bx1, by1, bx2, by2 = box_coords
                        
                        for i in range(1, num_labels):
                            cx, cy = centroids[i]
                            # If centroid is inside box, or it's the largest component
                            if (bx1 < cx < bx2 and by1 < cy < by2) or stats[i, cv2.CC_STAT_AREA] > (h*w*0.05):
                                recovered_mask |= (labels_im == i)
                        
                        if np.any(recovered_mask):
                            best_mask = recovered_mask
                    else:
                        # Point Click Mode: Keep only the target component
                        # Ensure coordinates are integers for array indexing
                        ix = int(max(0, min(ref_x, w - 1)))
                        iy = int(max(0, min(ref_y, h - 1)))
                        
                        target_label = labels_im[iy, ix]
                        if target_label != 0:
                            # Use intelligent filter to remove small/far disconnected objects (pots, artifacts)
                            best_mask = self._filter_small_components(mask_uint8, ref_x, ref_y, target_label, labels_im, stats, centroids)
                        else:
                            max_area = 0
                            max_label = 1
                            for i in range(1, num_labels):
                                if stats[i, cv2.CC_STAT_AREA] > max_area:
                                    max_area = stats[i, cv2.CC_STAT_AREA]
                                    max_label = i
                            best_mask = (labels_im == max_label)
        
        return best_mask

    def _filter_small_components(self, mask, click_x, click_y, target_label, labels_im, stats, centroids):
        """
        Internal helper to remove disconnected components that are too small or too far from click.
        Helps prevent painting unintended pots/decorations when walls are selected.
        """
        num_labels = len(stats)
        h, w = mask.shape
        
        # Get clicked component stats
        main_area = stats[target_label, cv2.CC_STAT_AREA]
        
        # Build cleaned mask
        clean_mask = np.zeros((h, w), dtype=np.uint8)
        
        for i in range(1, num_labels):
            component_area = stats[i, cv2.CC_STAT_AREA]
            cx, cy = centroids[i]
            
            # Distance from click
            dist = np.sqrt((cx - click_x)**2 + (cy - click_y)**2)
            
            # Keep component if:
            # 1. It's exactly the clicked component
            # 2. OR it's a reasonably large piece (>=10% of main) AND close enough
            if i == target_label:
                clean_mask[labels_im == i] = 1
            elif (component_area >= main_area * SegmentationConfig.MIN_COMPONENT_RATIO and 
                  dist < SegmentationConfig.MAX_COMPONENT_DISTANCE):
                clean_mask[labels_im == i] = 1
        
        return clean_mask.astype(bool)


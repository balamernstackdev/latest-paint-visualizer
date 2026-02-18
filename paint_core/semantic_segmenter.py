
import numpy as np
import torch
import cv2
import logging
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SemanticSegmenter:
    """
    A lightweight semantic segmentation engine using Segformer (ADE20K).
    Used to identify and exclude non-wall objects (furniture, lamps, plants, etc.)
    from the painting mask.
    """
    
    # ADE20K indices for common indoor objects to EXCLUDE from wall painting
    # Full list: https://github.com/CSAILVision/sceneparsing/blob/master/data/object150_info.csv
    EXCLUSION_CLASSES = [
        3,  # floor, flooring
        4,  # bed
        5,  # cabinet
        6,  # windowpane
        7,  # person
        8,  # door
        9,  # table
        10, # curtain
        11, # chair
        12, # car
        13, # painting, picture
        14, # sofa
        15, # shelf
        18, # mirror
        19, # rug
        20, # armchair
        21, # seat
        22, # fence
        23, # desk
        24, # wardrobe
        25, # lamp
        26, # bathtub
        27, # rail
        28, # cushion
        29, # base, pedestal
        30, # box
        31, # column
        32, # signboard
        33, # chest of drawers
        34, # counter
        35, # sand
        36, # sink
        37, # skyscraper
        38, # fireplace
        39, # refrigerator
        40, # grandstand
        41, # path
        42, # stairs
        43, # runway
        44, # case
        45, # pool table
        46, # pillow
        47, # screen door
        48, # stairway
        49, # river
        50, # bridge
        51, # bookcase
        52, # blind
        53, # coffee table
        54, # toilet
        55, # flower
        56, # book
        57, # hill
        58, # bench
        59, # countertop
        60, # stove
        61, # palm
        62, # kitchen island
        63, # computer
        64, # swivel chair
        65, # boat
        66, # bar
        67, # arcade machine
        68, # hovel
        69, # bus
        70, # towel
        71, # light
        72, # truck
        73, # tower
        74, # chandelier
        75, # awning
        76, # streetlight
        77, # booth
        78, # television
        79, # airplane
        80, # dirt track
        81, # apparel
        82, # pole
        83, # land
        84, # bannister
        85, # escalator
        86, # ottoman
        87, # bottle
        88, # buffet
        89, # poster
        90, # stage
        91, # van
        92, # ship
        93, # fountain
        94, # conveyer belt
        95, # canopy
        96, # washer
        97, # plaything
        98, # swimming pool
        99, # stool
        100, # barrel
        101, # basket
        102, # waterfall
        103, # tent
        104, # bag
        105, # minibike
        106, # cradle
        107, # oven
        108, # ball
        109, # food
        110, # step
        111, # tank
        112, # trade name
        113, # microwave
        114, # pot
        115, # animal
        116, # bicycle
        117, # lake
        118, # dishwasher
        119, # screen
        120, # blanket
        121, # sculpture
        122, # hood
        123, # sconce
        124, # vase
        125, # traffic light
        126, # tray
        127, # ashcan
        128, # fan
        129, # pier
        130, # crt screen
        131, # plate
        132, # monitor
        133, # bulletin board
        134, # shower
        135, # radiator
        136, # glass
        137, # clock
        138, # flag
    ]
    
    # Explicitly what IS a wall-like structure (for positive reinforcement if needed)
    # 0 = wall, 1 = building, 2 = sky, 16 = house
    WALL_CLASSES = [0, 1, 16] 

    def __init__(self, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = None
        self.model = None
        self.current_mask = None
        self.model_name = "nvidia/segformer-b0-finetuned-ade20k-512-512"
        
    def load_model(self):
        if self.model is not None:
            return

        try:
            logger.info(f"Loading Semantic Segmentation model: {self.model_name}")
            self.processor = SegformerImageProcessor.from_pretrained(self.model_name)
            self.model = SegformerForSemanticSegmentation.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Optimization: Half precision on CUDA
            if self.device == "cuda":
                self.model.half()
                
            logger.info("Semantic model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load semantic model: {e}")
            self.model = None

    def process_image(self, image_rgb):
        """
        Run inference on the image and cache the semantic map.
        Args:
            image_rgb: Numpy array (H, W, 3)
        """
        if self.model is None:
            self.load_model()
            if self.model is None: return

        try:
            # Convert to PIL for the processor
            pil_image = Image.fromarray(image_rgb)
            
            # Prepare inputs
            inputs = self.processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            if self.device == "cuda":
                inputs["pixel_values"] = inputs["pixel_values"].half()
                
            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits  # shape (1, 150, H/4, W/4)
            
            # Upsample logits to original image size
            upsampled_logits = torch.nn.functional.interpolate(
                logits,
                size=image_rgb.shape[:2],
                mode="bilinear",
                align_corners=False,
            )
            
            # Get class map (H, W)
            pred_seg = upsampled_logits.argmax(dim=1)[0]
            
            # Move to CPU numpy
            self.current_mask = pred_seg.cpu().numpy().astype(np.uint8)
            logger.info("Semantic segmentation map computed.")
            
        except Exception as e:
            logger.error(f"Error in semantic segmentation inference: {e}")
            self.current_mask = None

    def get_exclusion_mask(self, shape):
        """
        Returns a boolean mask where True = "Do Not Paint" (Foreground Object).
        """
        if self.current_mask is None:
            return np.zeros(shape, dtype=bool)
            
        h, w = shape
        if self.current_mask.shape != (h, w):
            # Resizing should theoretically not happen if processed correct, 
            # but safety first
            seg_map = cv2.resize(self.current_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            seg_map = self.current_mask
            
        # Create exclusion mask matching any of the forbidden classes
        exclusion = np.isin(seg_map, self.EXCLUSION_CLASSES)
        
        return exclusion

    def get_wall_mask(self, shape):
        """Returns mask of high-confidence wall areas."""
        if self.current_mask is None:
            return np.zeros(shape, dtype=bool)
        
        h, w = shape
        if self.current_mask.shape != (h, w):
            seg_map = cv2.resize(self.current_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            seg_map = self.current_mask
            
        return np.isin(seg_map, self.WALL_CLASSES)

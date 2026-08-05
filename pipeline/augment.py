"""Domain-specific data augmentation engine for semiconductor wafer maps."""

from __future__ import annotations
import random
import numpy as np
import cv2
from typing import Tuple, List

def rotate_wafer_map(wafer_map: np.ndarray, k: int) -> np.ndarray:
    """Rotate wafer map by k * 90 degrees."""
    return np.rot90(wafer_map, k=k).copy()

def flip_wafer_map(wafer_map: np.ndarray, mode: str = "horizontal") -> np.ndarray:
    """Flip wafer map horizontally, vertically, or both."""
    if mode == "horizontal":
        return np.fliplr(wafer_map).copy()
    elif mode == "vertical":
        return np.flipud(wafer_map).copy()
    elif mode == "both":
        return np.flipud(np.fliplr(wafer_map)).copy()
    return wafer_map.copy()

def morphological_perturbation(wafer_map: np.ndarray, operation: str = "dilation") -> np.ndarray:
    """
    Apply subtle morphological dilation or erosion specifically to defect dies (value = 2).
    Preserves wafer background mask (value = 0).
    """
    augmented = wafer_map.copy()
    defect_mask = (augmented == 2).astype(np.uint8)
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    
    if operation == "dilation":
        modified_mask = cv2.dilate(defect_mask, kernel, iterations=1)
    else:
        modified_mask = cv2.erode(defect_mask, kernel, iterations=1)
    
    # Apply change only inside non-background silicon area (wafer_map != 0)
    silicon_mask = (augmented > 0)
    augmented[silicon_mask & (modified_mask == 1)] = 2
    if operation == "erosion":
        augmented[silicon_mask & (modified_mask == 0) & (defect_mask == 1)] = 1
        
    return augmented

def add_die_noise(wafer_map: np.ndarray, noise_prob: float = 0.005) -> np.ndarray:
    """Simulate random sensor read noise across active silicon dies."""
    augmented = wafer_map.copy()
    silicon_indices = np.argwhere(augmented > 0)
    if len(silicon_indices) == 0:
        return augmented
    
    num_noise = max(1, int(len(silicon_indices) * noise_prob))
    chosen_idx = np.random.choice(len(silicon_indices), size=num_noise, replace=False)
    
    for idx in chosen_idx:
        r, c = silicon_indices[idx]
        current_val = augmented[r, c]
        # Flip between normal die (1) and defect die (2)
        augmented[r, c] = 2 if current_val == 1 else 1
        
    return augmented

def apply_elastic_warp(wafer_map: np.ndarray, alpha: float = 8.0, sigma: float = 3.0) -> np.ndarray:
    """Apply spatial elastic deformation to simulate crystal lattice / transport stress."""
    shape = wafer_map.shape
    dx = cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
    
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = np.float32(x + dx)
    map_y = np.float32(y + dy)
    
    warped = cv2.remap(wafer_map.astype(np.float32), map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped.astype(wafer_map.dtype)

def generate_random_augmentation(wafer_map: np.ndarray) -> Tuple[np.ndarray, str]:
    """Apply a randomized combination of physical & spatial transformations."""
    aug_types = ["rotate_90", "rotate_180", "rotate_270", "hflip", "vflip", "hflip_rotate", "morph_dilate", "morph_erode", "elastic", "noise"]
    chosen = random.choice(aug_types)
    
    if chosen == "rotate_90":
        return rotate_wafer_map(wafer_map, 1), "Rotation 90°"
    elif chosen == "rotate_180":
        return rotate_wafer_map(wafer_map, 2), "Rotation 180°"
    elif chosen == "rotate_270":
        return rotate_wafer_map(wafer_map, 3), "Rotation 270°"
    elif chosen == "hflip":
        return flip_wafer_map(wafer_map, "horizontal"), "Horizontal Flip"
    elif chosen == "vflip":
        return flip_wafer_map(wafer_map, "vertical"), "Vertical Flip"
    elif chosen == "hflip_rotate":
        return rotate_wafer_map(flip_wafer_map(wafer_map, "horizontal"), 1), "H-Flip + Rotation 90°"
    elif chosen == "morph_dilate":
        return morphological_perturbation(wafer_map, "dilation"), "Defect Morph Dilation"
    elif chosen == "morph_erode":
        return morphological_perturbation(wafer_map, "erosion"), "Defect Morph Erosion"
    elif chosen == "elastic":
        return apply_elastic_warp(wafer_map), "Elastic Warp Distortion"
    elif chosen == "noise":
        return add_die_noise(wafer_map, noise_prob=0.008), "Silicon Die Sensor Noise"
    
def create_multi_defect_wafer(wafer1: np.ndarray, wafer2: np.ndarray) -> np.ndarray:
    """
    Superimpose two distinct defect pattern wafer maps (e.g. Scratch + Donut, Edge-Ring + Loc).
    Creates realistic multi-defect silicon maps.
    """
    h1, w1 = wafer1.shape
    h2, w2 = wafer2.shape

    if (h1, w1) != (h2, w2):
        wafer2_resized = cv2.resize(wafer2.astype(np.uint8), (w1, h1), interpolation=cv2.INTER_NEAREST)
    else:
        wafer2_resized = wafer2

    merged = wafer1.copy()
    silicon_area = (wafer1 > 0)
    
    # Overlay defect dies (value = 2) from wafer2 onto wafer1 silicon area
    merged[silicon_area & (wafer2_resized == 2)] = 2
    return merged


import cv2
import numpy as np
import os
from skimage import morphology
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_crack_image(img,
                           median_blur_kernel_size=5,
                           morph_open_kernel_size=(3, 3),
                           morph_close_kernel_size=(7, 7),
):
    """
    Preprocesses an input image to detect cracks, returning a binary image.
    """
    if img is None:
        logger.error("Error: Could not load image")
        return None

    logger.info(f"Image shape: {img.shape}")
    
    # Convert to grayscale if needed
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif len(img.shape) == 2:
        img = img.copy()
    else:
        logger.error(f"Error: Unsupported image format. Shape: {img.shape}")
        return None

    # Apply median blur
    img = cv2.medianBlur(img, median_blur_kernel_size)
    
    # Apply Sobel edge detection
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    edge_strength_map = cv2.magnitude(sobel_x, sobel_y)
    edge_strength_map_normalized = cv2.normalize(edge_strength_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    _, img = cv2.threshold(edge_strength_map_normalized, 75, 255, cv2.THRESH_BINARY)

    # Apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    img = cv2.dilate(img, kernel, iterations=1)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_close_kernel_size)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, close_kernel, iterations=2)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_close_kernel_size)
    final_binary_crack_image = cv2.morphologyEx(img, cv2.MORPH_CLOSE, close_kernel, iterations=2)

    return final_binary_crack_image


def analyze_cracks(image):
    """
    Analyzes cracks in the input image and returns annotated image and metrics.
    """
    if image is None:
        logger.error("Error: Input image to analyze_cracks is None.")
        return None, {"error": "Input image was None"}

    final_binary_crack_image = preprocess_crack_image(image) 
    
    if final_binary_crack_image is None:
        logger.error("Error: Preprocessing returned None or failed.")
        annotated_image_on_failure = None
        if image is not None:
            annotated_image_on_failure = image.copy()
            if len(annotated_image_on_failure.shape) == 2:
                annotated_image_on_failure = cv2.cvtColor(annotated_image_on_failure, cv2.COLOR_GRAY2BGR)
            elif len(annotated_image_on_failure.shape) == 3 and annotated_image_on_failure.shape[2] == 1:
                 annotated_image_on_failure = cv2.cvtColor(annotated_image_on_failure, cv2.COLOR_GRAY2BGR)

        return annotated_image_on_failure, {"error": "Preprocessing failed", "Number of Cracks": 0}

    results = {}
    annotated_image = image.copy()

    if len(annotated_image.shape) == 2: 
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_GRAY2BGR)
    elif len(annotated_image.shape) == 3 and annotated_image.shape[2] == 1: 
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_GRAY2BGR)

    if final_binary_crack_image.dtype != np.uint8:
        final_binary_crack_image = final_binary_crack_image.astype(np.uint8)

    contours, _ = cv2.findContours(final_binary_crack_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results["Number of Cracks"] = len(contours)

    if annotated_image is not None and contours:
        cv2.drawContours(annotated_image, contours, -1, (0, 255, 0), 1) 

    dist_transform = cv2.distanceTransform(final_binary_crack_image, cv2.DIST_L2, maskSize=5)

    individual_crack_lengths = []
    individual_crack_areas = []
    individual_crack_avg_widths = []
    individual_crack_tortuosities = []
    max_crack_width_overall = 0.0

    for i, contour in enumerate(contours):
        area_pixels = cv2.contourArea(contour)
        individual_crack_areas.append(area_pixels)

        mask = np.zeros_like(final_binary_crack_image, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

        isolated_crack_segment = cv2.bitwise_and(final_binary_crack_image, final_binary_crack_image, mask=mask)
        
        # Skeletonize using skimage
        isolated_crack_boolean = isolated_crack_segment > 0
        try:
            current_crack_skeleton_boolean = morphology.medial_axis(isolated_crack_boolean)
            current_crack_skeleton = current_crack_skeleton_boolean.astype(np.uint8)
        except Exception as e_skel:
            logger.error(f"Error skeletonizing isolated crack: {e_skel}")
            current_crack_skeleton = np.zeros_like(isolated_crack_segment, dtype=np.uint8)

        length_pixels = np.sum(current_crack_skeleton > 0) 
        individual_crack_lengths.append(length_pixels)

        avg_width_pixels = 0.0
        if length_pixels > 0:
            skeleton_coords = np.argwhere(current_crack_skeleton > 0) 
            
            current_crack_local_widths = []
            for r_coord, c_coord in skeleton_coords: 
                local_width = dist_transform[r_coord, c_coord] * 2.0
                current_crack_local_widths.append(local_width)
            
            if current_crack_local_widths:
                avg_width_pixels = np.mean(current_crack_local_widths)
                max_width_this_crack = np.max(current_crack_local_widths)
                if max_width_this_crack > max_crack_width_overall:
                    max_crack_width_overall = max_width_this_crack
        individual_crack_avg_widths.append(avg_width_pixels)

        tortuosity = np.nan
        if length_pixels > 5: 
            skeleton_points_for_tortuosity = np.argwhere(current_crack_skeleton > 0)
            if len(skeleton_points_for_tortuosity) > 1:
                points_xy = skeleton_points_for_tortuosity[:, ::-1].reshape(-1,1,2) 
                x_skel, y_skel, w_skel, h_skel = cv2.boundingRect(points_xy)
                max_euclidean_dist = np.sqrt(w_skel**2 + h_skel**2)

                if max_euclidean_dist > 0:
                    tortuosity = length_pixels / max_euclidean_dist
                else:
                    tortuosity = 1.0 
        elif length_pixels > 0: 
            tortuosity = 1.0 
        individual_crack_tortuosities.append(tortuosity)

    results["Total Crack Area (pixels)"] = np.sum(individual_crack_areas) if individual_crack_areas else 0.0
    results["Total Crack Length (pixels)"] = np.sum(individual_crack_lengths) if individual_crack_lengths else 0.0
    results["Overall Max Crack Width (pixels)"] = max_crack_width_overall
    
    image_height, image_width = final_binary_crack_image.shape[:2]
    total_image_area_pixels = float(image_height * image_width)
    results["Crack Density (pixel ratio)"] = \
        results["Total Crack Area (pixels)"] / total_image_area_pixels if total_image_area_pixels > 0 else 0.0


    results["Individual Crack Metrics"] = {
        "Lengths_px": [round(l,1) for l in individual_crack_lengths],
        "Areas_px": [round(a,1) for a in individual_crack_areas],
        "Avg_Widths_px": [round(w,2) for w in individual_crack_avg_widths],
        "Tortuosities": [round(t, 2) if not np.isnan(t) else "N/A" for t in individual_crack_tortuosities]
    }

    return annotated_image, results 
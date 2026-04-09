"""
Shared utility functions for mitochondrial analysis tools.
"""

import numpy as np
from matplotlib.colors import ListedColormap


def weighted_average_scan(image, x, y, radius):
    """
    Calculate weighted average of image centered on (x, y) with given radius.
    
    Args:
        image: 2D numpy array (image)
        x: x-coordinate (row index)
        y: y-coordinate (column index)
        radius: radius around the point to sample
        
    Returns:
        Weighted average with weights 1/(1+distance)
    """
    height, width = image.shape
    weighted_sum = 0.0
    weight_sum = 0.0
    
    # Define the bounding box for sampling
    x_min = max(0, int(x - radius))
    x_max = min(height, int(x + radius + 1))
    y_min = max(0, int(y - radius))
    y_max = min(width, int(y + radius + 1))
    
    # Sample all points within the radius
    for i in range(x_min, x_max):
        for j in range(y_min, y_max):
            distance = np.sqrt((i - x)**2 + (j - y)**2)
            if distance <= radius:
                weight = 1.0 / (1.0 + distance)
                weighted_sum += image[i, j] * weight
                weight_sum += weight
    
    if weight_sum > 0:
        return weighted_sum / weight_sum
    else:
        return 0.0


def create_colormaps():
    """Create custom colormaps for mitochondria and scan visualization."""
    # Mitochondria colormap
    N = 256
    vals = np.ones((N, 4))
    vals[:, 0] = np.sqrt(np.linspace(0/256, 1, N))
    vals[:, 1] = np.sqrt(np.linspace(0/256, 64/256, N))
    vals[:, 2] = np.sqrt(np.linspace(0/256, 1, N))
    vals[:, 3] = np.sqrt(np.linspace(0/256, 256/256, N))
    mito_cmap = ListedColormap(vals)

    # Scan colormap
    N = 256
    vals = np.ones((N, 4))
    vals[:, 0] = np.sqrt(np.linspace(0/256, 64/256, N))
    vals[:, 1] = np.sqrt(np.linspace(64/256, 1, N))
    vals[:, 2] = np.sqrt(np.linspace(0/256, 64/256, N))
    vals[:, 3] = np.sqrt(np.linspace(0/256, 256/256, N))
    scan_cmap = ListedColormap(vals)
    
    return mito_cmap, scan_cmap

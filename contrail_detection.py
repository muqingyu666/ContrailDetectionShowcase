import numpy as np
from skimage import io, color, filters
from scipy import ndimage
import cv2
import matplotlib.pyplot as plt


def load_and_preprocess(image_path: str):
    """Load an image and apply basic preprocessing."""
    img = io.imread(image_path)
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[:, :, :3]
        gray = color.rgb2gray(img)
    else:
        gray = img

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_uint8 = (gray * 255).astype(np.uint8)
    enhanced = clahe.apply(gray_uint8)
    blurred = filters.gaussian(enhanced, sigma=1.0, preserve_range=True)
    return img, blurred.astype(np.uint8)


def create_line_kernel(size: int, angle_deg: float) -> np.ndarray:
    """Return a normalized line kernel at a given angle."""
    kernel = np.zeros((size, size))
    center = size // 2
    angle_rad = np.radians(angle_deg)
    for i in range(-center, center + 1):
        x = int(center + i * np.cos(angle_rad))
        y = int(center + i * np.sin(angle_rad))
        if 0 <= x < size and 0 <= y < size:
            kernel[y, x] = 1
    kernel -= np.mean(kernel)
    norm = np.linalg.norm(kernel)
    if norm > 0:
        kernel /= norm
    return kernel


def multi_angle_convolution_detection(img_gray: np.ndarray, angles=None, kernel_size: int = 21):
    """Detect linear features using multiple directional convolution kernels."""
    if angles is None:
        angles = np.arange(0, 180, 15)
    kernels = [create_line_kernel(kernel_size, a) for a in angles]
    responses = [ndimage.convolve(img_gray.astype(float), k) for k in kernels]
    responses = np.array(responses)
    max_response = responses.max(axis=0)
    max_angle_idx = responses.argmax(axis=0)
    angle_map = np.array([angles[idx] for idx in max_angle_idx.flatten()]).reshape(max_angle_idx.shape)
    threshold = np.percentile(max_response, 95)
    binary_detection = max_response > threshold
    return max_response, binary_detection, responses, angle_map, max_angle_idx


def visualize_results(original_img, preprocessed, responses, binary_detection,
                       angle_responses, angle_map, angles, max_angle_idx=None):
    """Display detection results using matplotlib."""
    n_angles = len(angles)
    n_cols = 4
    n_rows = int(np.ceil(n_angles / n_cols)) + 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))

    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(preprocessed, cmap="gray")
    axes[0, 1].set_title("Preprocessed")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(responses, cmap="hot")
    axes[0, 2].set_title("Combined Response")
    axes[0, 2].axis("off")

    axes[0, 3].imshow(binary_detection, cmap="gray")
    axes[0, 3].set_title("Binary Detection")
    axes[0, 3].axis("off")

    for i, angle in enumerate(angles):
        row = (i // n_cols) + 1
        col = i % n_cols
        if row < axes.shape[0] and col < axes.shape[1]:
            axes[row, col].imshow(angle_responses[i], cmap="hot")
            axes[row, col].set_title(f"Angle: {angle}°")
            axes[row, col].axis("off")

    overlay_ax = axes[n_rows - 1, 0]
    overlay_ax.imshow(original_img)
    mask = np.zeros((*original_img.shape[:2], 4))
    mask[binary_detection, 0] = 1.0
    mask[binary_detection, 3] = 0.5
    overlay_ax.imshow(mask)
    overlay_ax.set_title("Detected Lines Overlay")
    overlay_ax.axis("off")

    angle_vis_ax = axes[n_rows - 1, 1]
    norm_angle_map = angle_map / 180.0
    colored_angles = plt.cm.hsv(norm_angle_map)
    colored_angles[..., 3] = binary_detection * 0.7
    angle_vis_ax.imshow(original_img)
    angle_vis_ax.imshow(colored_angles)
    angle_vis_ax.set_title("Angle Direction Map")
    angle_vis_ax.axis("off")

    kernel_ax = axes[n_rows - 1, 2]
    kernels_display = np.hstack([create_line_kernel(15, a) for a in angles[:4]])
    kernel_ax.imshow(kernels_display, cmap="gray")
    kernel_ax.set_title("Sample Kernels")
    kernel_ax.axis("off")

    for i in range(n_rows):
        for j in range(n_cols):
            if i >= 1 and i < n_rows - 1 and (i - 1) * n_cols + j >= n_angles:
                axes[i, j].axis("off")
                axes[i, j].set_visible(False)
            if i == n_rows - 1 and j >= 3:
                axes[i, j].axis("off")
                axes[i, j].set_visible(False)

    plt.tight_layout()
    plt.show()

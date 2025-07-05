                                                                         Preprocessing
Median Filtering
---------------
import cv2
import os
import matplotlib.pyplot as plt

# === Define Main Input and Output Folder Paths ===
main_input_folder = ""  # Main folder containing subfolders
main_output_folder = ""  # Output root folder

# === Set Kernel Size ===
kernel_size = 3  # Choose odd number like 3, 5, or 7

# === Traverse all subfolders and process images ===
for subdir, dirs, files in os.walk(main_input_folder):
    for filename in files:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
            input_path = os.path.join(subdir, filename)

            # Generate relative path and use it for output path
            rel_path = os.path.relpath(input_path, main_input_folder)
            output_path = os.path.join(main_output_folder, rel_path)

            # Ensure output subfolder exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Read image in grayscale
            image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

            # Apply Median Filter
            filtered_image = cv2.medianBlur(image, kernel_size)

            # Save filtered image
            cv2.imwrite(output_path, filtered_image)

            # Optional: Display original and filtered side-by-side
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.title("Original")
            plt.imshow(image, cmap='gray')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.title("Filtered")
            plt.imshow(filtered_image, cmap='gray')
            plt.axis('off')

            plt.suptitle(rel_path)
            plt.tight_layout()
            plt.show()
M-LCM
-----
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# === PARAMETERS ===
sigma = 2
alpha = 0.5
beta = 0.7
block_size = 32
I_lower = 0
I_upper = 255

# === Define Main Input and Output Folder Paths ===
main_input_folder = ""       # Main folder containing input images
main_output_folder = ""       # Output folder for enhanced images

# === Ensure output root exists ===
os.makedirs(main_output_folder, exist_ok=True)

# === Function: M-LCM Enhancement ===
def apply_mlcm(image):
    image = image.astype(np.float32)

    # Step 1: Base & Detail
    I_base = gaussian_filter(image, sigma=sigma)
    I_detail = image - I_base

    # Step 2: Histogram Modification
    h, w = image.shape
    I_hist_mod = np.zeros_like(image)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = I_base[i:i+block_size, j:j+block_size]
            min_val, max_val = block.min(), block.max()
            if max_val - min_val != 0:
                stretched = ((block - min_val) / (max_val - min_val)) * (I_upper - I_lower) + I_lower
            else:
                stretched = block
            I_hist_mod[i:i+block_size, j:j+block_size] = stretched

    # Step 3: Gradient Enhancement
    Gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    Gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    G_mag = np.sqrt(Gx**2 + Gy**2)
    I_enhanced_detail = I_detail * (1 + alpha * G_mag / 255.0)

    # Step 4: Combine Layers
    I_MLCM = I_hist_mod + beta * I_enhanced_detail
    return np.clip(I_MLCM, 0, 255).astype(np.uint8)

# === Walk Through Input Folder and Process ===
for root, dirs, files in os.walk(main_input_folder):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
            input_path = os.path.join(root, file)
            relative_path = os.path.relpath(input_path, main_input_folder)
            output_path = os.path.join(main_output_folder, relative_path)

            # Create output subfolder
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Read and process image
            img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                enhanced_img = apply_mlcm(img)

                # Save enhanced image
                cv2.imwrite(output_path, enhanced_img)

                # Optional: Show images
                plt.figure(figsize=(10, 4))
                plt.subplot(1, 2, 1)
                plt.title("Original")
                plt.imshow(img, cmap='gray')
                plt.axis('off')

                plt.subplot(1, 2, 2)
                plt.title("M-LCM Enhanced")
                plt.imshow(enhanced_img, cmap='gray')
                plt.axis('off')

                plt.suptitle(file)
                plt.tight_layout()
                plt.show()
CLAHE
-----
import cv2
import os
import matplotlib.pyplot as plt

# === Parameters for CLAHE ===
clip_limit = 2.0
tile_grid_size = (8, 8)

# === Input and Output Folder Paths ===
main_input_folder = ""      # Main input folder with subfolders
main_output_folder = ""     # Where to save enhanced images

# === Ensure output root exists ===
os.makedirs(main_output_folder, exist_ok=True)

# === CLAHE Object ===
clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

# === Walk Through Input Folder and Process Images ===
for root, dirs, files in os.walk(main_input_folder):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
            input_path = os.path.join(root, file)
            relative_path = os.path.relpath(input_path, main_input_folder)
            output_path = os.path.join(main_output_folder, relative_path)

            # Create output subfolder if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Load image in grayscale
            image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            # Apply CLAHE
            enhanced_image = clahe.apply(image)

            # Save enhanced image
            cv2.imwrite(output_path, enhanced_image)

            # Optional: Show side-by-side
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.title("Original")
            plt.imshow(image, cmap='gray')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.title("CLAHE Enhanced")
            plt.imshow(enhanced_image, cmap='gray')
            plt.axis('off')

            plt.suptitle(file)
            plt.tight_layout()
            plt.show()

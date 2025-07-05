                                                                    Feature Extraction
import os
import cv2
import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam

# === Define Input & Output Paths ===
main_input_folder = ""
output_csv = ""

# === Build CAE ===
def build_cae(input_shape=(128, 128, 1)):
    input_img = Input(shape=input_shape)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = Conv2D(4, (3, 3), activation='relu', padding='same')(x)
    x = Conv2DTranspose(8, (3, 3), activation='relu', padding='same')(encoded)
    x = Conv2DTranspose(16, (3, 3), activation='relu', padding='same')(x)
    decoded = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(x)
    autoencoder = Model(input_img, decoded)
    encoder = Model(input_img, encoded)
    autoencoder.compile(optimizer=Adam(), loss='mse')
    return autoencoder, encoder

autoencoder, encoder = build_cae()

# === Feature Extraction Functions ===
def compute_glcm(image):
    max_gray = 256
    glcm = np.zeros((max_gray, max_gray), dtype=np.float32)
    for i in range(image.shape[0] - 1):
        for j in range(image.shape[1] - 1):
            row = image[i, j]
            col = image[i, j + 1]
            glcm[row, col] += 1
    glcm /= (np.sum(glcm) + 1e-10)
    return glcm

def extract_glcm_features(image):
    image = cv2.resize(image, (128, 128))
    glcm = compute_glcm(image)
    contrast = np.sum([(i - j)**2 * glcm[i, j] for i in range(256) for j in range(256)])
    energy = np.sum(glcm ** 2)
    entropy = -np.sum(glcm * np.log2(glcm + 1e-10))
    homogeneity = np.sum([glcm[i, j] / (1 + (i - j)**2) for i in range(256) for j in range(256)])
    dissimilarity = np.sum([np.abs(i - j) * glcm[i, j] for i in range(256) for j in range(256)])
    mean_i = np.sum([i * np.sum(glcm[i, :]) for i in range(256)])
    mean_j = np.sum([j * np.sum(glcm[:, j]) for j in range(256)])
    std_i = np.sqrt(np.sum([(i - mean_i)**2 * np.sum(glcm[i, :]) for i in range(256)]))
    std_j = np.sqrt(np.sum([(j - mean_j)**2 * np.sum(glcm[:, j]) for j in range(256)]))
    correlation = np.sum([((i - mean_i)*(j - mean_j)*glcm[i, j]) for i in range(256) for j in range(256)]) / (std_i * std_j + 1e-10)
    return [contrast, correlation, energy, entropy, homogeneity, dissimilarity]

def extract_shape_features(image):
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return [0, 0, 0]
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00']) if M['m00'] else 0
    left = image[:, :cx]
    right = image[:, cx:]
    asymmetry = np.abs(np.sum(left) - np.sum(right)) / (np.sum(image) + 1e-5)
    return [area, perimeter, asymmetry]

def extract_wavelet_features(image):
    coeffs2 = pywt.dwt2(image, 'haar')
    A, (Dh, Dv, Dd) = coeffs2
    return [np.mean(A), np.std(A), np.mean(Dh), np.mean(Dv), np.mean(Dd)]

# === Process All Images ===
feature_list = []

for root, dirs, files in os.walk(main_input_folder):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            file_path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(file_path))
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (128, 128))
            norm_image = image.astype('float32') / 255.0
            reshaped = norm_image.reshape(1, 128, 128, 1)
            base_name = os.path.splitext(file)[0]
            output_folder = root

            # === Train Autoencoder & Get Deep Features ===
            autoencoder.fit(reshaped, reshaped, epochs=5, verbose=0)
            deep_features = encoder.predict(reshaped).flatten()[:20]
            reconstructed = autoencoder.predict(reshaped).reshape(128, 128)

            # === Wavelet ===
            A, _ = pywt.dwt2(image, 'haar')

            # === Save CAE and Wavelet Images ===
            cae_path = os.path.join(output_folder, base_name + "_cae.png")
            wavelet_path = os.path.join(output_folder, base_name + "_wavelet.png")
            cv2.imwrite(cae_path, (reconstructed * 255).astype(np.uint8))
            A_resized = cv2.resize(A, (128, 128))
            A_normalized = cv2.normalize(A_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            cv2.imwrite(wavelet_path, A_normalized)

            # === Display Input, CAE, Wavelet ===
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1); plt.imshow(image, cmap='gray'); plt.title("Original"); plt.axis('off')
            plt.subplot(1, 3, 2); plt.imshow(reconstructed, cmap='gray'); plt.title("CAE Output"); plt.axis('off')
            plt.subplot(1, 3, 3); plt.imshow(A, cmap='gray'); plt.title("Wavelet A"); plt.axis('off')
            plt.suptitle(f"{file} | Label: {label}")
            plt.tight_layout(); plt.show()

            # === Visualize CAE Layers ===
            layer_names = [layer.name for layer in autoencoder.layers if 'conv' in layer.name]
            layer_outputs = [autoencoder.get_layer(name).output for name in layer_names]
            vis_models = [Model(inputs=autoencoder.input, outputs=output) for output in layer_outputs]

            for layer_name, vis_model in zip(layer_names, vis_models):
                feature_maps = vis_model.predict(reshaped)[0]  # shape: (H, W, C)
                num_channels = feature_maps.shape[-1]

                plt.figure(figsize=(15, 3))
                for i in range(min(8, num_channels)):
                    plt.subplot(1, 8, i+1)
                    plt.imshow(feature_maps[:, :, i], cmap='viridis')
                    plt.title(f"{layer_name}\nCh {i}")
                    plt.axis('off')
                plt.suptitle(f"{file} - Layer: {layer_name}")
                plt.tight_layout()
                plt.show()

            # === Extract Other Features ===
            glcm_feats = extract_glcm_features(image)
            shape_feats = extract_shape_features(image)
            wavelet_feats = extract_wavelet_features(image)

            # === Save Combined Features ===
            all_feats = [file, label] + glcm_feats + shape_feats + wavelet_feats + list(deep_features)
            feature_list.append(all_feats)

# === Save CSV ===
columns = ['filename', 'label',
           'GLCM_Contrast', 'GLCM_Correlation', 'GLCM_Energy', 'GLCM_Entropy', 'GLCM_Homogeneity', 'GLCM_Dissimilarity',
           'Shape_Area', 'Shape_Perimeter', 'Shape_Asymmetry',
           'Wavelet_A', 'Wavelet_A_std', 'Wavelet_Dh', 'Wavelet_Dv', 'Wavelet_Dd'] + \
          [f'Deep_{i}' for i in range(20)]

df = pd.DataFrame(feature_list, columns=columns)
df.to_csv(output_csv, index=False)

print(f"✅ Feature extraction and visualization complete.\nSaved to: {output_csv}")
df
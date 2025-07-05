Segmentation
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
# === Residual Block ===
class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(in_channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)

# === Squeeze-and-Excitation Block ===
class SEBlock3D(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock3D, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        b, c, d, h, w = x.size()
        squeeze = self.global_pool(x).view(b, c)
        excitation = self.sigmoid(self.fc2(self.relu(self.fc1(squeeze)))).view(b, c, 1, 1, 1)
        return x * excitation

# === Multi-Scale Attention Block ===
class MultiScaleAttention3D(nn.Module):
    def __init__(self, in_channels):
        super(MultiScaleAttention3D, self).__init__()
        self.scale1 = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.scale2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.scale3 = nn.Conv3d(in_channels, in_channels, kernel_size=5, padding=2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        s1 = self.scale1(x)
        s2 = self.scale2(x)
        s3 = self.scale3(x)
        stacked = torch.stack([s1, s2, s3], dim=1)  # shape: [B, 3, C, D, H, W]
        attention = self.softmax(stacked)
        out = (attention * stacked).sum(dim=1)
        return out

# === Dilated Convolution Block ===
class DilatedConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=2):
        super(DilatedConvBlock3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# === Full 3DRMAU-Net-DCSE Model ===
class ThreeDRMAUNetDCSE(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_filters=32):
        super(ThreeDRMAUNetDCSE, self).__init__()

        # Encoder
        self.enc1 = nn.Conv3d(in_channels, base_filters, kernel_size=3, padding=1)
        self.res1 = ResidualBlock3D(base_filters)
        self.se1 = SEBlock3D(base_filters)

        self.enc2 = nn.Conv3d(base_filters, base_filters * 2, kernel_size=3, stride=2, padding=1)
        self.res2 = ResidualBlock3D(base_filters * 2)

        # Bottleneck with attention
        self.attention = MultiScaleAttention3D(base_filters * 2)

        # Decoder
        self.dilated = DilatedConvBlock3D(base_filters * 2, base_filters)
        self.dec1 = nn.ConvTranspose3d(base_filters, base_filters, kernel_size=2, stride=2)
        self.final_conv = nn.Conv3d(base_filters, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x1 = F.relu(self.enc1(x))
        x1 = self.res1(x1)
        x1 = self.se1(x1)

        x2 = F.relu(self.enc2(x1))
        x2 = self.res2(x2)

        # Bottleneck
        x_att = self.attention(x2)

        # Decoder
        x_dil = self.dilated(x_att)
        x_up = self.dec1(x_dil)
        x_out = self.final_conv(x_up)

        return self.sigmoid(x_out)
# === Define Input and Output Folders ===
input_folder = "/content/drive/MyDrive/Colab Notebooks/archive (75)/clahe_output/ST000001"
output_root = "/content/drive/MyDrive/Colab Notebooks/archive (75)/segmentation_output"

# === Supported image extensions ===
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')

# === Loop Through All Images Recursively ===
for root, dirs, files in os.walk(input_folder):
    for file in files:
        if file.lower().endswith(image_extensions):
            img_path = os.path.join(root, file)
            print(f"Processing: {img_path}")

            # Load Image
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # CLAHE Enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)

            # Otsu Threshold
            _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Morphological Processing
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

            # Segmented region from original image
            segmented_region = cv2.bitwise_and(image, image, mask=closed)

            # === Create Output Folder Path ===
            relative_path = os.path.relpath(root, input_folder)
            output_folder = os.path.join(output_root, relative_path)
            os.makedirs(output_folder, exist_ok=True)

            # === Save Only Segmented Image ===
            base_name = os.path.splitext(file)[0]
            output_path = os.path.join(output_folder, base_name + "_segmented.png")
            cv2.imwrite(output_path, segmented_region)

            # === Display Input, Mask, and Segmented Image ===
            plt.figure(figsize=(18, 5))

            plt.subplot(1, 3, 1)
            plt.title("Original")
            plt.imshow(image, cmap='gray')
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.title("Mask")
            plt.imshow(closed, cmap='gray')
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.title("Segmented Region")
            plt.imshow(segmented_region, cmap='gray')
            plt.axis('off')

            plt.tight_layout()
            plt.show()
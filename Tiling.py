import sys
import rasterio
import numpy as np
import cv2
import os

# =========================
# CONFIG
# =========================
input_path = r"D:/PAN/P5_PAN_CD_N27_250_E071_625_OIM.tif"
output_dir = r"D:/Solar panel/OIM/tiles"

tile_size = 640  # try 512 if memory issue
sys.stdout.reconfigure(encoding='utf-8')
os.makedirs(output_dir, exist_ok=True)

# Initialize CLAHE once outside the loop for efficiency
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

# =========================
# PROCESS
# =========================
with rasterio.open(input_path) as src:
    print("Image size:", src.width, src.height)
    print("Bands:", src.count)

    width = src.width
    height = src.height

    count = 0

    for i in range(0, height, tile_size):
        for j in range(0, width, tile_size):

            window = rasterio.windows.Window(
                j, i,
                min(tile_size, width - j),
                min(tile_size, height - i)
            )

            # Read first band directly as a 2D array (H, W) for OpenCV compatibility
            patch = src.read(1, window=window)
            patch = np.nan_to_num(patch)

            # =========================
            # FILTER EMPTY / BAD TILES
            # =========================
            if patch.shape[0] == 0 or patch.shape[1] == 0:
                continue

            if np.mean(patch) == 0:
                continue

            # =========================
            # ADVANCED CONTRAST ENHANCEMENT
            # =========================
            
            # 1. Scale 16-bit to 8-bit dynamically
            if patch.dtype != np.uint8:
                patch = cv2.normalize(patch, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # 2. Apply CLAHE to dramatically boost the local contrast
            patch_enhanced = clahe.apply(patch)

            # 3. Convert enhanced grayscale back to 3-channel RGB
            patch_rgb = cv2.cvtColor(patch_enhanced, cv2.COLOR_GRAY2RGB)

            # Double check to ensure we don't save an completely empty/black tile
            if np.max(patch_rgb) == 0:
                continue

            # =========================
            # SAVE
            # =========================
            out_path = os.path.join(output_dir, f"tile_{i}_{j}.png")

            # Convert RGB to BGR right before saving via OpenCV
            success = cv2.imwrite(
                out_path,
                cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR)
            )

            if success:
                count += 1
                print(f"Saved: {out_path}")

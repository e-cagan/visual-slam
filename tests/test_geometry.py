import sys
import os

# Main path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import cv2
from src.dataset import KittiDataset
from src.features import extract_features, match_features, get_matched_points
from src.geometry import triangulate, solve_pnp

# 1. Dataset ve Matrisleri yükle
ds = KittiDataset(sequence_id="04", base_path="data/dataset")
print(f"P_left shape: {ds.P_left.shape}")

# Baseline hesabı (Sağlama)
baseline = ds.P_right[0, 3] / -ds.P_left[0, 0] # Focal length'e bölüyoruz
print(f"Calculated Baseline: {baseline:.4f} meters (Expected ~0.54m)")

# 2. İlk kareyi (Frame 0) stereo olarak oku
img_L, img_R = ds.get_stereo_frame(0)

# 3. Özellikleri çıkar ve eşleştir
kp_L, desc_L = extract_features(img_L)
kp_R, desc_R = extract_features(img_R)

matches = match_features(desc_L, desc_R)
pts_L, pts_R = get_matched_points(kp_L, kp_R, matches)

print(f"\nExtracted {len(matches)} stereo matches.")

# 4. Triangulation (İçinde 0 < Z < 100 filtresi zaten var)
# valid_mask, bu derinlik filtresinden geçen noktaların boolean indeksidir.
pts_3d_valid, valid_mask = triangulate(pts_L, pts_R, ds.P_left, ds.P_right, max_depth=100)

# DİKKAT: 2D noktaları orijinal diziden (pts_L) sadece maskeyi kullanarak filtreliyoruz.
pts_L_filt = pts_L[valid_mask]

# 5. Sonuçları İnceleme (Sağlama)
print(f"\n3D Points shape: {pts_3d_valid.shape}")
print("First 5 points [X, Y, Z (Depth)]:")
print(pts_3d_valid[:5])

print(f"\nMedian Depth (Z): {np.median(pts_3d_valid[:, 2]):.2f} meters")
print(f"Max Depth: {np.max(pts_3d_valid[:, 2]):.2f} meters")
print(f"Min Depth: {np.min(pts_3d_valid[:, 2]):.2f} meters")

# 6. Sanity Loop (PnP) Testi
print("\n--- PnP Sanity Loop Test ---")
R, t, inliers = solve_pnp(pts_3d_valid, pts_L_filt, ds.K)

if R is not None:
    print(f"R matrix (Expected ~Identity):\n{np.round(R, 4)}")
    print(f"t vector (Expected ~Zeros):\n{np.round(t, 4).ravel()}")
    print(f"||t|| (Translation Norm): {np.linalg.norm(t):.4e} meters")
    print(f"PnP Inliers: {len(inliers) if inliers is not None else 0} / {len(pts_3d_valid)}")
else:
    print("PnP failed!")
"""
Module for feature management.
"""

import cv2
import numpy as np


# ORB detector and Brute-force matcher
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

def extract_features(gray):
    """gray: grayscale (H,W). Returns: extracted (keypoints, descriptors)"""
    # Extract keypoints and descriptors using ORB
    keypoint, descriptor = orb.detectAndCompute(gray, None)
    
    return keypoint, descriptor


def match_features(desc1, desc2, ratio=0.75):
    """Matches description features using BFMatcher with Lowe's Ratio Test.""" 
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        return []
    
    # Match descriptions and sort them by distance
    matches = bf.knnMatch(desc1, desc2, k=2)
    
    good_matches = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            # Lowe's Ratio Test ile güvenilir eşleşmeleri seç
            if m.distance < ratio * n.distance:
                good_matches.append(m)
        elif len(m_n) == 1:
            good_matches.append(m_n[0])
            
    good_matches = sorted(good_matches, key=lambda x: x.distance)
    return good_matches


def get_matched_points(kp1, kp2, matches):
    """Converts keypoints to (N, 2) shape."""
    # Retrieve the matched (X, Y) coordinates between points
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    return pts1, pts2
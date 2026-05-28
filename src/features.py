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


def match_features(desc1, desc2):
    """Matches description features using BFMatcher. Returns matches."""  
    # Match descriptions and sort them by distance
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


def get_matched_points(kp1, kp2, matches):
    """Converts keypoints to (N, 2) shape."""
    # Retrieve the matched (X, Y) coordinates between points
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    return pts1, pts2
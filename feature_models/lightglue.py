"""
LightGlue wrapper.

Replaces BFMatcher with learned attention-based matching.
Takes two sets of (keypoints, descriptors) and returns matched pairs.

Reference: Lindenberger et al. 2023
License: Apache 2.0 (safe for any use)
"""

import torch
import numpy as np
from lightglue import LightGlue


class LightGlueMatcher:
    def __init__(self, device='cuda', features='superpoint'):
        """
        Initializes the LightGlue matching network.
        
        Args:
            device: 'cuda' or 'cpu'
            features: The type of local features used (e.g., 'superpoint')
        """
        self.device = device
        self.model = self._load_model(features)
        self.model.eval()
    
    def _load_model(self, features_type):
        """Loads the pretrained LightGlue model."""
        return LightGlue(features=features_type).to(self.device)
    
    @torch.no_grad()
    def match(self, kp0, desc0, kp1, desc1, image_size_hw):
        """
        Matches two sets of keypoints and descriptors.
        
        Args:
            kp0, kp1: (N, 2) keypoints in pixels (Numpy arrays)
            desc0, desc1: (N, 256) float32 descriptors (Numpy arrays)
            image_size_hw: (Height, Width) tuple of the original image
            
        Returns:
            matches: (M, 2) int numpy array — matched indices [(idx_0, idx_1), ...]
            confidence: (M,) float numpy array — match confidence per pair
        """
        # Return empty if there's nothing to match
        if len(kp0) == 0 or len(kp1) == 0:
            return np.empty((0, 2), dtype=int), np.empty((0,), dtype=np.float32)
        
        # Convert inputs to torch tensors
        kp0_t = torch.from_numpy(kp0).float().to(self.device)
        desc0_t = torch.from_numpy(desc0).float().to(self.device)
        kp1_t = torch.from_numpy(kp1).float().to(self.device)
        desc1_t = torch.from_numpy(desc1).float().to(self.device)
        
        # LightGlue expects image size as (Width, Height) for positional encoding
        h, w = image_size_hw
        size_t = torch.tensor([w, h], dtype=torch.float32, device=self.device)
        
        # Build LightGlue input dictionaries (Adding batch dimensions)
        data0 = {
            'keypoints': kp0_t.unsqueeze(0),
            'descriptors': desc0_t.unsqueeze(0),
            'image_size': size_t.unsqueeze(0)
        }
        
        data1 = {
            'keypoints': kp1_t.unsqueeze(0),
            'descriptors': desc1_t.unsqueeze(0),
            'image_size': size_t.unsqueeze(0)
        }
        
        # Forward pass
        result = self.model({'image0': data0, 'image1': data1})
        
        # Extract matched indices and confidence scores
        matches = result['matches'][0].cpu().numpy()
        confidence = result['scores'][0].cpu().numpy()
        
        return matches, confidence
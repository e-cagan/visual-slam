"""
SuperPoint wrapper.

Loads pretrained Magic Leap weights, runs inference on grayscale image,
returns keypoints + descriptors compatible with downstream pipeline.

Reference: DeTone et al. 2018
Model weights: https://github.com/magicleap/SuperPointPretrainedNetwork
"""

import torch
import numpy as np
from lightglue import SuperPoint


class SuperPointExtractor:
    def __init__(self, device='cuda', max_keypoints=1024, detection_threshold=0.005, nms_radius=4):
        """
        Initializes the SuperPoint network.
        
        Args:
            device: 'cuda' or 'cpu'
            max_keypoints: Maximum number of keypoints to extract.
            detection_threshold: Minimum confidence score for a keypoint.
            nms_radius: Radius for non-maximum suppression (in pixels).
        """
        self.device = device
        self.max_keypoints = max_keypoints
        self.detection_threshold = detection_threshold
        self.nms_radius = nms_radius
        
        self.model = self._load_model()
        self.model.eval()
    
    def _load_model(self):
        """
        Loads the pretrained SuperPoint model from the lightglue package.
        """
        return SuperPoint(
            max_num_keypoints=self.max_keypoints,
            detection_threshold=self.detection_threshold,
            nms_radius=self.nms_radius
        ).to(self.device)
    
    @torch.no_grad()
    def extract(self, image_np):
        """
        Extracts keypoints and descriptors from a grayscale image.
        
        Args:
            image_np: (H, W) grayscale uint8 numpy array
            
        Returns:
            keypoints: (N, 2) numpy float32, [x, y] pixel coords
            descriptors: (N, 256) numpy float32, L2-normalized
            scores: (N,) numpy float32, keypoint confidence
        """
        # Convert numpy array (uint8) to torch float32 tensor in range [0, 1]
        image_tensor = torch.from_numpy(image_np).float() / 255.0
        
        # Add batch and channel dimensions -> (1, 1, H, W)
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Forward pass through the network
        result = self.model({'image': image_tensor})
        
        # Extract results and move back to CPU as numpy arrays
        keypoints = result['keypoints'][0].cpu().numpy()
        descriptors = result['descriptors'][0].cpu().numpy()
        scores = result['keypoint_scores'][0].cpu().numpy()
        
        return keypoints, descriptors, scores
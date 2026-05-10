"""
CS3IV Kinect Object Recognition - Person B: Feature Extraction

Provides feature extraction functions used by train_classifier.py.
All functions take numpy image arrays (not file paths) so they work
on both saved crops and live video frames.

Three feature types are combined into one vector per image:
  - HOG on the depth image   (shape descriptor, depth-invariant to colour)
  - HOG on the RGB image     (appearance descriptor, extra marks for RGB use)
  - ORB on the depth image   (keypoint descriptor, catches local structure)

HOG parameters are tuned for 128x128 inputs (the fixed crop size from Person A).
ORB is zero-padded to a fixed length so the classifier always sees the same
vector size regardless of how many keypoints were found.

- Karim
"""

import cv2
import numpy as np

# HOG params tuned for 128x128 crops from Person A's pipeline
_HOG_WIN   = (128, 128)
_HOG_BLOCK = (16, 16)
_HOG_STEP  = (8, 8)
_HOG_CELL  = (8, 8)
_HOG_BINS  = 9

# built once at import, reused across all calls
_hog = cv2.HOGDescriptor(_HOG_WIN, _HOG_BLOCK, _HOG_STEP, _HOG_CELL, _HOG_BINS)

# 100 keypoints * 32 bytes = 3200 element vector, zero padded if fewer found
_ORB_NFEATURES = 100
_orb = cv2.ORB_create(nfeatures=_ORB_NFEATURES)


def extract_hog_features(img_gray):
    """
    Returns a flattened HOG descriptor for a 128x128 grayscale image.

    img_gray: grayscale uint8 ndarray, must be 128x128
    returns: 1D float32 array
    """
    return _hog.compute(img_gray).flatten().astype(np.float32)


def extract_orb_features(img_gray, orb=None, nfeatures=_ORB_NFEATURES):
    """
    Returns a zero-padded ORB descriptor vector.

    img_gray: grayscale uint8 ndarray
    orb: ORB detector instance, defaults to module-level singleton
    nfeatures: must match the detector's budget, output length is nfeatures * 32
    returns: 1D float32 array
    """
    detector = orb if orb is not None else _orb
    _, descriptors = detector.detectAndCompute(img_gray, None)
    out = np.zeros(nfeatures * 32, dtype=np.float32)
    if descriptors is not None:
        flat = descriptors.flatten().astype(np.float32)
        out[:len(flat)] = flat
    return out


def extract_features(depth_img, rgb_img=None):
    """
    Concatenates HOG(depth) + ORB(depth) + HOG(rgb) into one feature vector.

    depth_img: grayscale uint8 ndarray, 128x128
    rgb_img: BGR uint8 ndarray, 128x128, or None to skip RGB HOG
    returns: 1D float32 array (always pass both or always pass neither)
    """
    hog_depth = extract_hog_features(depth_img)
    orb_depth = extract_orb_features(depth_img)

    parts = [hog_depth, orb_depth]

    if rgb_img is not None:
        rgb_gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
        parts.append(extract_hog_features(rgb_gray))

    return np.concatenate(parts)
